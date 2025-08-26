#[derive(Clone, Copy, Debug, Default)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
}

impl World {
    pub fn insert_transform(&mut self, e: Entity, t: Transform) {
        self.transforms.insert(e, t);
    }
    pub fn remove_transform(&mut self, e: Entity) -> bool{
        self.transforms.remove(e);
        return true
    }
}