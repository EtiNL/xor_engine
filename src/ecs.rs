pub mod ecs_gpu_interface;
pub mod math_op;


pub mod ecs {

    use cuda_driver_sys::{CUdeviceptr};
    use crate::cuda_wrapper::SceneBuffer;
    use crate::ecs::math_op::math_op::{Vec3, Quat, Mat3};
    use crate::ecs::ecs_gpu_interface::ecs_gpu_interface::{CameraObject, SdfType, SdfObject, sdf_type_translation, TextureManager, TextureHandle};
    use std::{error::Error, collections::HashMap, path::Path};



    // Entity
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Entity {
        index: u32,
        generation: u32,          // helps catch use-after-delete
    }

    impl Entity {
        const DEAD: Self = Entity { index: u32::MAX, generation: 0 };
    }

    // Components
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Transform {
        pub position: Vec3,
        pub rotation: Quat,
    }

    #[derive(Clone, Debug)]
    pub struct Renderable {
        pub sdf_type:   SdfType,
        pub params:     [f32; 3],
        pub tex:        TextureHandle,
    }

    impl Renderable {
        pub fn new(sdf_type: SdfType,
                params:   [f32; 3],
                path:     &Path,
                tex_mgr:  &mut TextureManager) -> Result<Self, Box<dyn Error>> {
            let tex = tex_mgr.load(path)?;
            Ok(Self { sdf_type, params, tex })
        }
    }
    impl Default for Renderable {
        fn default() -> Self {
            Self {
                sdf_type: SdfType::Sphere,
                params: [0.0; 3],
                tex: TextureHandle::default(),
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct SpaceFolding {
        pub lattice_basis: Mat3,
        pub lattice_basis_inv: Mat3, 
    }
    impl SpaceFolding {
        pub fn new(lattice_basis: Mat3) -> Self {
            let lattice_basis_inv: Mat3 = lattice_basis.inv();
            Self { lattice_basis, lattice_basis_inv }
        }
    }
    impl Default for SpaceFolding {
        fn default() -> Self {
            Self {
                lattice_basis: Mat3::Zero,
                lattice_basis_inv: Mat3::Zero,
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct Camera {
        pub name: String,
        pub field_of_view: f32,    // in degrees
        pub width: u32,
        pub height: u32,
        pub aperture: f32,
        pub focus_distance: f32
    }


    #[derive(Clone, Copy, Debug, Default)]
    pub struct Rotating {
        pub speed_deg_per_sec: f32,
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Velocity;   // pour plus tard

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Collider;   // pour plus tard

    // Component storage
    pub struct SparseSet<T> {
        dense_entities: Vec<u32>, // who owns each component
        dense_data:     Vec<T>,   // the components, tightly packed
        sparse:         Vec<u32>, // maps entity index → dense slot, or u32::MAX when absent
        dirty_flags:    Vec<bool>,// maps entity index to True if the conponent as been mutated
    }

    impl<T> SparseSet<T> {
        pub fn insert(&mut self, e: Entity, data: T) {
            let idx = e.index as usize;
        
            if idx >= self.sparse.len() { self.sparse.resize(idx + 1, u32::MAX); }
            if self.sparse[idx] != u32::MAX { return; }
        
            let dense_slot = self.dense_entities.len() as u32;
            self.dense_entities.push(e.index);
            self.dense_data.push(data);
            self.sparse[idx] = dense_slot;
            self.dirty_flags.push(true); // mark as dirty
        }

        pub fn get(&self, e: Entity) -> Option<&T> {
            let slot = *self.sparse.get(e.index as usize)?;
            if slot == u32::MAX { return None; }
            Some(&self.dense_data[slot as usize])
        }

        pub fn get_mut(&mut self, e: Entity) -> Option<&mut T> {
            let slot = *self.sparse.get(e.index as usize)?;
            if slot == u32::MAX { return None; }
            self.mark_dirty(e);

            Some(&mut self.dense_data[slot as usize])
        }

        pub fn remove(&mut self, e: Entity) {
            let idx = e.index as usize;
            let slot = match self.sparse.get(idx) { Some(&s) if s != u32::MAX => s, _ => return };
            let last = self.dense_entities.len() as u32 - 1;

            // swap-remove in dense arrays
            self.dense_entities.swap(slot as usize, last as usize);
            self.dense_data.swap(slot as usize, last as usize);

            // update sparse back-pointer of the moved entity
            let moved_entity_idx = self.dense_entities[slot as usize] as usize;
            self.sparse[moved_entity_idx] = slot;

            self.dense_entities.pop();
            self.dense_data.pop();
            self.sparse[idx] = u32::MAX;
        }

        pub fn mark_dirty(&mut self, e: Entity) {
            let idx = e.index as usize;
            if let Some(&slot) = self.sparse.get(idx) {
                if slot != u32::MAX {
                    self.dirty_flags[slot as usize] = true;
                }
            }
        }

        pub fn clear_dirty_flags(&mut self) {
            for flag in &mut self.dirty_flags {
                *flag = false;
            }
        }

        pub fn contains(&self, idx: usize) -> bool {
            idx < self.sparse.len() && self.sparse[idx] != u32::MAX
        }

        /// Iterate over *all* components

        pub fn iter(&self) -> impl Iterator<Item=(&T, u32)> {
            self.dense_entities.iter().enumerate().map(move |(i, &e_idx)| {
                (&self.dense_data[i], e_idx)
            })
        }

        pub fn iter_mut(&mut self) -> impl Iterator<Item=(&mut T, u32)> {
            let data = &mut self.dense_data as *mut Vec<T>;  // raw ptr dance to split borrows
            self.dense_entities.iter().enumerate().map(move |(i,&e_idx)| {
                let arr = unsafe { &mut *data };
                (&mut arr[i], e_idx)
            })
        }

        pub fn iter_dirty(&self) -> impl Iterator<Item=(&T, u32)> + '_ {
            self.dense_entities.iter().enumerate()
                .filter(move |(i, _)| self.dirty_flags[*i])
                .map(move |(i, &e_idx)| (&self.dense_data[i], e_idx))
        }
    }

    // World
    pub struct World {
        gens:  Vec<u32>,          // generation per slot
        free:  Vec<u32>,          // holes we can reuse
        entity_to_scene_index: HashMap<u32, usize>,
        spawned_entities: Vec<u32>,
        entities_to_remove_from_gpu: Vec<usize>,
        empty_spaces_indices_queu: Vec<usize>,
        texture_of_entity: HashMap<u32, TextureHandle>,


        // component pools -----------------------------
        transforms: SparseSet<Transform>,
        velocities: SparseSet<Velocity>,
        cameras: SparseSet<Camera>,
        colliders:  SparseSet<Collider>,
        renderables: SparseSet<Renderable>,
        space_foldings: SparseSet<SpaceFolding>,
        rotatings: SparseSet<Rotating>,
    }

    impl World{
        pub fn new() -> Self {
            Self {
                gens: vec![],
                free: vec![],
                entity_to_scene_index: HashMap::new(),
                entities_to_remove_from_gpu: vec![],
                spawned_entities: vec![],
                empty_spaces_indices_queu: vec![],
                texture_of_entity: HashMap::new(),
                transforms: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                cameras: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                velocities: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                colliders:  SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                renderables: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                space_foldings: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                rotatings:  SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
            }
        }

        pub fn insert_transform(&mut self, e: Entity, t: Transform) {
            self.transforms.insert(e, t);
        }
        pub fn insert_camera(&mut self, e: Entity, c: Camera) {
            self.cameras.insert(e, c);
        }
        pub fn insert_renderable(&mut self, e: Entity, r: Renderable) {
            self.texture_of_entity.insert(e.index, r.tex);
            self.renderables.insert(e, r);
        }
        pub fn insert_space_folding(&mut self, e: Entity, s: SpaceFolding) {
            self.space_foldings.insert(e, s);
        }
        pub fn insert_rotating(&mut self, e: Entity, r: Rotating) {
            self.rotatings.insert(e, r);
        }

        pub fn spawn(&mut self) -> Entity {
            let index = if let Some(recycled) = self.free.pop() {
                recycled
            } else {
                self.gens.push(0);
                (self.gens.len() - 1) as u32
            };

            let generation = self.gens[index as usize];

            self.spawned_entities.push(index);
            Entity { index, generation }
        }

        pub fn despawn(&mut self, e: Entity, tex_mgr: &mut TextureManager) {
            let idx = e.index as usize;

            if self.gens.get(idx).copied() != Some(e.generation) {
                return; // stale entity
            }

            self.gens[idx] += 1;
            self.free.push(e.index);

            self.transforms.remove(e);
            self.velocities.remove(e);
            self.colliders.remove(e);
            self.renderables.remove(e);
            self.cameras.remove(e);
            self.space_foldings.remove(e);
            self.rotatings.remove(e);

            if let Some(index) = self.entity_to_scene_index.remove(&(idx as u32)) {
                self.entities_to_remove_from_gpu.push(index)
            }

            if let Some(h) = self.texture_of_entity.remove(&e.index) {
                let _ = tex_mgr.release(&h);
            }
        }

        pub fn choose_camera(&self, camera_name: &str) -> Result<CameraObject, Box<dyn Error>> {

            for (camera, e_idx) in self.cameras.iter() {
                if camera_name.to_string() == camera.name {
                    let gen = self.gens.get(e_idx as usize).copied().unwrap_or(0);
                    let e = Entity { index: e_idx, generation: gen };

                    if let Some(tr) = self.transforms.get(e) {
                        let rot = tr.rotation;
                        let camera_to_render = CameraObject::new(
                            tr.position,            // position
                            rot * Vec3::X,          // u (right)
                            rot * Vec3::Y,          // v (up)
                            rot * (Vec3::Z*-1.0),   // w (forward)
                            camera.field_of_view,
                            camera.width,
                            camera.height,
                            camera.aperture,
                            camera.focus_distance
                        );

                        return Ok(camera_to_render)
                    }
                    else {
                        let msg = "no camera transform: cannot define the position and rotation (ecs)";
                        return Err(Box::from(format!("{}", msg)))
                    }
                }
                else {
                    let msg = "no camera under that name (ecs)";
                    return Err(Box::from(format!("{}", msg)))
                }
            }
            
            let msg = "no camera defined (ecs)";
            return Err(Box::from(format!("{}", msg)))
            
        }

        pub fn render_scene(&mut self, scene_buffer_capacity: usize) -> Vec<SdfObject> {
            let mut output = Vec::new();
            self.entity_to_scene_index.clear();
            self.empty_spaces_indices_queu.clear();

            for (render, e_idx) in self.renderables.iter() {
                let gen = self.gens.get(e_idx as usize).copied().unwrap_or(0);
                let e = Entity { index: e_idx, generation: gen };

                if let Some(tr) = self.transforms.get(e) {

                    let texture = render.tex;
                    let sdf_type_device = sdf_type_translation(render.sdf_type);
                    let rot = tr.rotation;
                    let default_sf = SpaceFolding::default();
                    let sf = self.space_foldings.get(e).unwrap_or(&default_sf);


                    let sdf = SdfObject {
                        sdf_type: sdf_type_device,
                        params: render.params,
                        center: tr.position,
                        u: rot * Vec3::X,
                        v: rot * Vec3::Y,
                        w: rot * Vec3::Z,
                        texture: texture.d_ptr,
                        tex_width: texture.width,
                        tex_height: texture.height,
                        lattice_basis: sf.lattice_basis,
                        lattice_basis_inv: sf.lattice_basis_inv,
                        active: 1,
                    };
                    output.push(sdf);

                    self.entity_to_scene_index.insert(e_idx, output.len()-1);
                }
            }

            for idx in output.len()..scene_buffer_capacity {
                self.empty_spaces_indices_queu.push(idx);
            }

            // Clear dirty flags
            self.spawned_entities = vec![];
            self.transforms.clear_dirty_flags();
            self.renderables.clear_dirty_flags();
            self.entities_to_remove_from_gpu = vec![];

            output
        }

        pub fn update_entity_on_gpu(
            &self,
            scene_buf: &mut SceneBuffer,
            e: Entity,
        ) -> Result<(), Box<dyn Error>> {
            let render = self.renderables.get(e).ok_or("Missing Renderable")?;
            let sdf_type_device = sdf_type_translation(render.sdf_type);
            let texture = render.tex;
            let tr = self.transforms.get(e).ok_or("Missing Transform")?;
            let default_sf = SpaceFolding::default();
            let sf = self.space_foldings.get(e).unwrap_or(&default_sf);

            
            let sdf = SdfObject {
                sdf_type: sdf_type_device,
                params: render.params,
                center: tr.position,
                u: tr.rotation * Vec3::X,
                v: tr.rotation * Vec3::Y,
                w: tr.rotation * Vec3::Z,
                texture: texture.d_ptr,
                tex_width: texture.width,
                tex_height: texture.height,
                lattice_basis: sf.lattice_basis,
                lattice_basis_inv: sf.lattice_basis_inv,
                active: 1,
            };

            let index = self.entity_to_scene_index.get(&e.index).ok_or("Entity not in scene")?;

            unsafe { scene_buf.update_sdfobject_in_gpu_scene(*index, &sdf) }
            
        }


        pub fn update_scene(
            &mut self,
            scene_buf: &mut SceneBuffer,
        ) -> bool {

            let mut scene_updated = false;

            // handles spawned entites
            for idx in &self.spawned_entities {
                let gen = self.gens[*idx as usize];
                let e = Entity { index: *idx, generation: gen };

                if self.renderables.contains(*idx as usize) {
                    let index_gpu_buffer = if let Some(empty) = self.empty_spaces_indices_queu.first().copied() {
                        self.empty_spaces_indices_queu.remove(0)
                    } else {
                        scene_buf.ensure_capacity(self.entity_to_scene_index.len() + 1);
                        let new_scene = self.render_scene(scene_buf.capacity);
                        scene_buf.upload(&new_scene);

                        return true; // the scene is rerendered entirely so no need to go further
                    };
                    
                    scene_buf.max_index_used = scene_buf.max_index_used.max(index_gpu_buffer);

                    self.entity_to_scene_index.insert(*idx, index_gpu_buffer);
                    self.update_entity_on_gpu(scene_buf, e);

                    scene_updated = true;
                }
            }

            self.spawned_entities.clear();

            // handles updates on transform or renderables
            let mut already_updated_entities:Vec<Entity> = vec![];

            for (transform, idx) in self.transforms.iter_dirty() {
                let gen = self.gens[idx as usize];
                let e = Entity { index: idx, generation: gen };
                self.update_entity_on_gpu(scene_buf, e);
                already_updated_entities.push(e);

                scene_updated = true;
            }
            
            for (renderable, idx) in self.renderables.iter_dirty() {
            
                let gen = self.gens[idx as usize];
                let e = Entity { index: idx, generation: gen };

                if !already_updated_entities.contains(&e) {
                    self.update_entity_on_gpu(scene_buf, e);
                    scene_updated = true;
                }
            }
            
            // Clear dirty flags
            self.transforms.clear_dirty_flags();
            self.renderables.clear_dirty_flags();

            // Handles removed entities
            for idx in &self.entities_to_remove_from_gpu {
                scene_buf.deactivate(*idx);
                scene_updated = true;
                self.empty_spaces_indices_queu.push(*idx);
                self.empty_spaces_indices_queu.sort();
            }
            let max_index_gpu: Option<usize> = self.entity_to_scene_index
                .values()   // iterates over &usize
                .copied()   // turn &usize → usize
                .max();     // returns Option<usize>

            match max_index_gpu {
                Some(m) => scene_buf.max_index_used = m,
                None    => scene_buf.max_index_used = 0, //no more objects in the scene
            }

            self.entities_to_remove_from_gpu = vec![];

            return scene_updated
        }
    }


    // System

    pub fn update_rotation(world: &mut World, dt: f32, input_dir: Vec3) {
        for (rot, idx) in world.rotatings.iter_mut() {
            let gen = world.gens[idx as usize];
            let e = Entity { index: idx, generation: gen };

            if let Some(tr) = world.transforms.get_mut(e) {

                let angle_y= rot.speed_deg_per_sec.to_radians() * dt * input_dir.y;
                let q_y = Quat::from_axis_angle(Vec3::Y, angle_y);

                let angle_x = rot.speed_deg_per_sec.to_radians() * dt * input_dir.x;
                let q_x = Quat::from_axis_angle(Vec3::X, angle_x);
                tr.rotation = (q_y * q_x) * tr.rotation;
            }
        }
    }
}