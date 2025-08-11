#[derive(Clone, Copy, Debug)]
pub struct LightComponent {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}
impl Default for GpuLight {
    fn default() -> Self {
        GpuLight {
            position: Vec3::default(),
            color: Vec3::default(),
            intensity: 0.0,
        }
    }
}

impl World{
    pub(crate) fn insert_light(&mut self, e: Entity, l: LightComponent) {
        self.lights.insert(e, l);
    }
    pub(crate) fn remove_light(&mut self, e: Entity) -> bool{
        let mut scene_updated:bool = false;
        if let Some(_light_slot) = self.light_gpu_indices.get(e) {
            self.light_gpu_indices.free_for(e);
            scene_updated = true;
        }
        self.lights.remove(e);
        return scene_updated
    }
    pub(crate) fn sync_light(&mut self) -> Result<bool, Box<dyn Error>> {
        let mut scene_updated:bool = false;
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
        return Ok(scene_updated)
    }
}
