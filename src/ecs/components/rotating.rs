#[derive(Clone, Copy, Debug, Default)]
pub struct Rotating {
    pub speed_deg_per_sec: f32,
}

impl World {
    pub fn insert_rotating(&mut self, e: Entity, r: Rotating) {
        self.rotatings.insert(e, r);
    }
    pub fn remove_rotating(&mut self, e: Entity) -> bool{
        self.rotatings.remove(e);
        return true
    }
}