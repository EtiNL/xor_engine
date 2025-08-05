pub mod ecs_gpu_interface;
pub mod math_op;


pub mod ecs {

    use cuda_driver_sys::{CUdeviceptr};
    
    
    use std::{error::Error, path::Path};
    use std::collections::HashSet;
    use memoffset::offset_of;

    use crate::ecs::ecs_gpu_interface::ecs_gpu_interface::{
        CameraObject, SdfType, sdf_type_translation,
        GpuIndexMap, TextureManager, TextureHandle,
        GpuSdfObjectBase, GpuMaterial, GpuLight, GpuSpaceFolding,
        INVALID_MATERIAL, INVALID_LIGHT, INVALID_FOLDING,
    };
    use crate::cuda_wrapper::GpuBuffer;
    use crate::ecs::math_op::math_op::{Vec3, Quat, Mat3};




    // Entity
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Entity {
        pub index: u32,
        generation: u32,          // helps catch use-after-delete
    }

    impl Entity {
        const DEAD: Self = Entity { index: u32::MAX, generation: 0 };
    }

    // Components

    #[derive(Clone, Copy, Debug)]
    pub struct SdfBase {
        pub sdf_type: SdfType,
        pub params: [f32; 3],
    }

    #[derive(Clone, Debug)]
    pub struct MaterialComponent {
        pub color: [f32; 3],
        pub texture: Option<TextureHandle>,
        pub use_texture: bool,
        // future: roughness, emission, metallic, etc.
    }

    #[derive(Clone, Copy, Debug)]
    pub struct LightComponent {
        pub position: Vec3,
        pub color: Vec3,
        pub intensity: f32,
    }
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Transform {
        pub position: Vec3,
        pub rotation: Quat,
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
            let slot = match self.sparse.get(idx) {
                Some(&s) if s != u32::MAX => s,
                _ => return,
            };
            let last = self.dense_entities.len() as u32 - 1;
        
            // swap-remove from the entity list
            self.dense_entities.swap(slot as usize, last as usize);
            self.dense_data.swap(slot as usize, last as usize);
            // also swap-remove the dirty flag for that slot
            self.dirty_flags.swap(slot as usize, last as usize);
        
            // fix up the sparse back-pointer of the entity we just moved
            let moved_entity_idx = self.dense_entities[slot as usize] as usize;
            self.sparse[moved_entity_idx] = slot;
        
            // pop off the now-unused last slot from all three arrays
            self.dense_entities.pop();
            self.dense_data.pop();
            self.dirty_flags.pop();
        
            // mark this index as “absent”
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
        // entity management
        gens: Vec<u32>,
        free: Vec<u32>,
        spawned_entities: Vec<u32>,
        entities_to_remove_from_gpu: Vec<u32>,

        // component pools
        sdf_bases: SparseSet<SdfBase>,
        materials: SparseSet<MaterialComponent>,
        lights: SparseSet<LightComponent>,
        space_foldings: SparseSet<SpaceFolding>,
        transforms: SparseSet<Transform>,
        rotatings: SparseSet<Rotating>,
        cameras: SparseSet<Camera>,
        velocities: SparseSet<Velocity>,
        colliders: SparseSet<Collider>,

        // GPU-side index maps
        sdf_gpu_indices: GpuIndexMap,
        material_gpu_indices: GpuIndexMap,
        light_gpu_indices: GpuIndexMap,
        folding_gpu_indices: GpuIndexMap,

        // GPU buffers
        pub gpu_sdf_objects: GpuBuffer<GpuSdfObjectBase>,
        pub  gpu_materials: GpuBuffer<GpuMaterial>,
        pub gpu_lights: GpuBuffer<GpuLight>,
        pub gpu_foldings: GpuBuffer<GpuSpaceFolding>,
    }

    impl World{
        pub fn new() -> Result<Self, Box<dyn Error>> {
            Ok(Self {
                gens: vec![],
                free: vec![],
    
                spawned_entities: vec![],
                entities_to_remove_from_gpu: vec![],
    
                sdf_bases: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                materials: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                lights: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                space_foldings: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                transforms: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                rotatings: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                cameras: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                velocities: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
                colliders: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![], dirty_flags: vec![] },
    
                sdf_gpu_indices: GpuIndexMap::new(),
                material_gpu_indices: GpuIndexMap::new(),
                light_gpu_indices: GpuIndexMap::new(),
                folding_gpu_indices: GpuIndexMap::new(),
    
                gpu_sdf_objects: GpuBuffer::<GpuSdfObjectBase>::new(8)?,
                gpu_materials:   GpuBuffer::<GpuMaterial>::new(8)?,
                gpu_lights:      GpuBuffer::<GpuLight>::new(8)?,
                gpu_foldings:    GpuBuffer::<GpuSpaceFolding>::new(8)?,
            })
        }

        pub fn insert_sdf_base(&mut self, e: Entity, sdf: SdfBase) {
            self.sdf_bases.insert(e, sdf);
        }
        pub fn insert_material(&mut self, e: Entity, m: MaterialComponent) {
            self.materials.insert(e, m);
        }
        pub fn insert_space_folding(&mut self, e: Entity, s: SpaceFolding) {
            self.space_foldings.insert(e, s);
        }
        pub fn insert_transform(&mut self, e: Entity, t: Transform) {
            self.transforms.insert(e, t);
        }
        pub fn insert_light(&mut self, e: Entity, l: LightComponent) {
            self.lights.insert(e, l);
        }
        pub fn insert_rotating(&mut self, e: Entity, r: Rotating) {
            self.rotatings.insert(e, r);
        }
        pub fn insert_camera(&mut self, e: Entity, c: Camera) {
            self.cameras.insert(e, c);
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

        pub fn despawn(&mut self, e: Entity) {
            let idx = e.index as usize;
            if self.gens.get(idx).copied() != Some(e.generation) {
                return; // stale
            }
            self.gens[idx] += 1;
            self.free.push(e.index);    
    
            // schedule GPU removal
            self.entities_to_remove_from_gpu.push(e.index);
        }

        pub fn choose_camera(&self, camera_name: &str) -> Result<CameraObject, Box<dyn Error>> {

            for (camera, e_idx) in self.cameras.iter() {
                if camera.name == camera_name {
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
            }
            
            let msg = "no camera under that name or no camera defined (ecs)";
            return Err(Box::from(format!("{}", msg)))
            
        }

        // Full initial upload: builds all GPU-side counterparts from existing host components.
        pub fn update_scene(&mut self, texture_manager: &mut TextureManager) -> Result<bool, Box<dyn Error>> {
            let mut scene_updated = false;
        
            // 1. Handle removals first
            while let Some(entity_idx) = self.entities_to_remove_from_gpu.pop() {
                let e = Entity {
                    index: entity_idx,
                    generation: self.gens[entity_idx as usize],
                };
                if let Some(sdf_slot) = self.sdf_gpu_indices.get(e) {
                    let active_off = offset_of!(GpuSdfObjectBase, active);
                    self.gpu_sdf_objects.deactivate_sdf(sdf_slot, active_off)?;
                    self.sdf_gpu_indices.free_for(e);
                    scene_updated = true;
                }
                if let Some(_mat_slot) = self.material_gpu_indices.get(e) {
                    // release TextureHandle via the TextureManager
                    let mat = self.materials.get(e).unwrap();
                    if let Some(tex) = &mat.texture {
                        texture_manager.release(tex)?;
                    }
                    self.material_gpu_indices.free_for(e);
                    scene_updated = true;
                }
                if let Some(_fold_slot) = self.folding_gpu_indices.get(e) {
                    self.folding_gpu_indices.free_for(e);
                    scene_updated = true;
                }
                if let Some(_light_slot) = self.light_gpu_indices.get(e) {
                    self.light_gpu_indices.free_for(e);
                    scene_updated = true;
                }

                // remove host components
                self.transforms.remove(e);
                self.sdf_bases.remove(e);
                self.materials.remove(e);
                self.space_foldings.remove(e);
                self.lights.remove(e);
                self.rotatings.remove(e);
                self.cameras.remove(e);
                self.velocities.remove(e);
                self.colliders.remove(e);
            }
        
            // 2. Sync space foldings (dirty or new)
            for (_fold, idx) in self.space_foldings.iter_dirty() {
                let e = Entity { index: idx, generation: self.gens[idx as usize] };
                let gpu_slot = self.folding_gpu_indices.get_or_allocate_for(e);
                let folding = self.space_foldings.get(e).unwrap(); // safe
                let gpu_struct = GpuSpaceFolding {
                    lattice_basis: folding.lattice_basis,
                    lattice_basis_inv: folding.lattice_basis_inv,
                };
                self.gpu_foldings.push(gpu_slot, &gpu_struct)?;
                scene_updated = true;
            }
        
            // 3. Sync materials (dirty)
            for (_mat, idx) in self.materials.iter_dirty() {
                let e = Entity { index: idx, generation: self.gens[idx as usize] };
                let gpu_slot = self.material_gpu_indices.get_or_allocate_for(e);
                let mat = self.materials.get(e).unwrap();
        
                let (texture_data_pointer, width, height) = if let Some(tex) = &mat.texture {
                    (tex.d_ptr, tex.width, tex.height)
                } else {
                    (0, 0, 0)
                };
                let use_texture_flag = if mat.use_texture { 1 } else { 0 };
        
                let gpu_struct = GpuMaterial {
                    color: mat.color,
                    use_texture: use_texture_flag,
                    texture_data_pointer: texture_data_pointer,
                    width: width,
                    height: height,
                };
                self.gpu_materials.push(gpu_slot, &gpu_struct)?;
                scene_updated = true;
            }
        
            // 4. Sync lights (dirty)
            for (_light, idx) in self.lights.iter_dirty() {
                let e = Entity { index: idx, generation: self.gens[idx as usize] };
                let gpu_slot = self.light_gpu_indices.get_or_allocate_for(e);
                let light = self.lights.get(e).unwrap();
                let gpu_struct = GpuLight {
                    position: light.position,
                    color: light.color,
                    intensity: light.intensity,
                };
                self.gpu_lights.push(gpu_slot, &gpu_struct)?;
                scene_updated = true;
            }
        
            // 5. Sync SDF bases / transforms / dependency-driven rebuilds
            let mut to_update_sdf_set: HashSet<u32> = HashSet::new();
            for (_sdf, idx) in self.sdf_bases.iter_dirty() { to_update_sdf_set.insert(idx); }
            for (_tr, idx) in self.transforms.iter_dirty() { to_update_sdf_set.insert(idx); }
            for (_mat, idx) in self.materials.iter_dirty() { to_update_sdf_set.insert(idx); }
            for (_fold, idx) in self.space_foldings.iter_dirty() { to_update_sdf_set.insert(idx); }
            for (_light, idx) in self.lights.iter_dirty() { to_update_sdf_set.insert(idx); }
        
            for idx in to_update_sdf_set {
                let e = Entity { index: idx, generation: self.gens[idx as usize] };
        
                let sdf_base = match self.sdf_bases.get(e) { Some(s) => s, None => continue };
                let transform = match self.transforms.get(e) { Some(t) => t, None => continue };
        
                let gpu_slot = self.sdf_gpu_indices.get_or_allocate_for(e);
        
                let material_id = self
                    .material_gpu_indices
                    .get(e)
                    .map(|i| i as u32)
                    .unwrap_or(INVALID_MATERIAL);
                let folding_id = self
                    .folding_gpu_indices
                    .get(e)
                    .map(|i| i as u32)
                    .unwrap_or(INVALID_FOLDING);
                let light_id = self
                    .light_gpu_indices
                    .get(e)
                    .map(|i| i as u32)
                    .unwrap_or(INVALID_LIGHT);
        
                let sdf_obj = GpuSdfObjectBase {
                    sdf_type: sdf_type_translation(sdf_base.sdf_type) as i32,
                    params: sdf_base.params,
                    center: transform.position,
                    u: transform.rotation * Vec3::X,
                    v: transform.rotation * Vec3::Y,
                    w: transform.rotation * Vec3::Z,
                    material_id,
                    light_id,
                    lattice_folding_id: folding_id,
                    active: 1,
                };
                self.gpu_sdf_objects.push(gpu_slot, &sdf_obj)?;
                scene_updated = true;
            }
        
            // 6. Clear dirty flags
            self.space_foldings.clear_dirty_flags();
            self.materials.clear_dirty_flags();
            self.lights.clear_dirty_flags();
            self.sdf_bases.clear_dirty_flags();
            self.transforms.clear_dirty_flags();
        
            self.spawned_entities.clear();
            Ok(scene_updated)
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