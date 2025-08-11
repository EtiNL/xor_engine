#[derive(Clone, Copy, Debug)]
pub struct SdfBase {
    pub sdf_type: SdfType,
    pub params: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub enum SdfType {
    Sphere, 
    Cube,
}

pub fn sdf_type_translation(sdf_type: SdfType) -> u32 {
    match sdf_type {
        SdfType::Sphere => 0,
        SdfType::Cube => 1,
    }
}


pub const INVALID_MATERIAL: u32 = 0xFFFFFFFF;
pub const INVALID_LIGHT: u32 = 0xFFFFFFFF;
pub const INVALID_FOLDING: u32 = 0xFFFFFFFF;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuSdfObjectBase {
    pub sdf_type: i32,
    pub params: [f32; 3],
    pub center: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub material_id: u32,
    pub light_id: u32,
    pub lattice_folding_id: u32,
    pub active: u32,
}
impl Default for GpuSdfObjectBase {
    fn default() -> Self {
        GpuSdfObjectBase {
            sdf_type: 0,
            params: [0.0; 3],
            center: Vec3::default(),
            u: Vec3::default(),
            v: Vec3::default(),
            w: Vec3::default(),
            material_id: INVALID_MATERIAL,
            light_id: INVALID_LIGHT,
            lattice_folding_id: INVALID_FOLDING,
            active: 0,
        }
    }
}

impl World {
    pub(crate) fn insert_sdf_base(&mut self, e: Entity, sdf: SdfBase) {
        self.sdf_bases.insert(e, sdf);
    }
    pub(crate) fn remove_sdf_base(&mut self, e: Entity) -> Result<bool, Box<dyn Error>> {
        let mut updated = false;

        if let Some(sdf_slot) = self.sdf_gpu_indices.get(e) {
            let active_off = offset_of!(GpuSdfObjectBase, active);
            // propagate any device copy error
            self.gpu_sdf_objects.deactivate_sdf(sdf_slot, active_off)?;
            self.sdf_gpu_indices.free_for(e);
            updated = true;
        }

        self.sdf_bases.remove(e);
        Ok(updated)
    }
    pub(crate) fn sync_sdf_base(&mut self) -> Result<bool, Box<dyn Error>> {
        let mut scene_updated:bool = false;
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
        return Ok(scene_updated)
    }
}