use crate::scene_composition::{SdfObject, Vec3, Quat};
use cuda_driver_sys::CUdeviceptr;

// Entity
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug)]
pub struct Renderable {
    pub sdf_type: u32,
    pub params: [f32; 3],
    pub tex_width: u32,
    pub tex_height: u32,
    pub texture: CUdeviceptr // GPU pointer
}
impl Default for Renderable {
    fn default() -> Self {
        Self {
            sdf_type: 0,
            params: [0.0; 3],
            tex_width: 0,
            tex_height: 0,
            texture: 0,
        }
    }
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
}

impl<T> SparseSet<T> {
    pub fn insert(&mut self, e: Entity, data: T) {
        let idx = e.index as usize;

        if idx >= self.sparse.len() { self.sparse.resize(idx + 1, u32::MAX); }
        if self.sparse[idx] != u32::MAX { return; }      // already present

        let dense_slot = self.dense_entities.len() as u32;
        self.dense_entities.push(e.index);
        self.dense_data.push(data);
        self.sparse[idx] = dense_slot;
    }

    pub fn get(&self, e: Entity) -> Option<&T> {
        let slot = *self.sparse.get(e.index as usize)?;
        if slot == u32::MAX { return None; }
        Some(&self.dense_data[slot as usize])
    }

    pub fn get_mut(&mut self, e: Entity) -> Option<&mut T> {
        let slot = *self.sparse.get(e.index as usize)?;    // ?.copied() in nightly
        if slot == u32::MAX { return None; }
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
}

// World
pub struct World {
    gens:  Vec<u32>,          // generation per slot
    free:  Vec<u32>,          // holes we can reuse

    // component pools -----------------------------
    transforms: SparseSet<Transform>,
    velocities: SparseSet<Velocity>,
    colliders:  SparseSet<Collider>,
    renderables: SparseSet<Renderable>,
    rotatings: SparseSet<Rotating>,
}

impl World {
    pub fn new() -> Self {
        Self {
            gens: vec![],
            free: vec![],
            transforms: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![] },
            velocities: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![] },
            colliders:  SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![] },
            renderables: SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![] },
            rotatings:  SparseSet { dense_entities: vec![], dense_data: vec![], sparse: vec![] },
        }
    }

    pub fn insert_transform(&mut self, e: Entity, t: Transform) {
        self.transforms.insert(e, t);
    }
    pub fn insert_renderable(&mut self, e: Entity, r: Renderable) {
        self.renderables.insert(e, r);
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
        Entity { index, generation }
    }

    pub fn despawn(&mut self, e: Entity) {
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
        self.rotatings.remove(e);
    }

    pub fn render_scene(&self) -> Vec<SdfObject> {
        let mut output = Vec::new();

        for (render, e_idx) in self.renderables.iter() {
            let gen = self.gens.get(e_idx as usize).copied().unwrap_or(0);
            let e = Entity { index: e_idx, generation: gen };

            if let Some(tr) = self.transforms.get(e) {
                let rot = tr.rotation;
                let sdf = SdfObject {
                    sdf_type: render.sdf_type,
                    params: render.params,
                    center: tr.position,
                    u: rot * Vec3::X,
                    v: rot * Vec3::Y,
                    w: rot * Vec3::Z,
                    texture: render.texture,
                    tex_width: render.tex_width,
                    tex_height: render.tex_height,
                };
                output.push(sdf);
            }
        }

        output
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