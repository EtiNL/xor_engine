#[derive(Clone, Copy, Debug)]
pub enum Axis { U, V, W }

impl Axis {
    #[inline] fn to_mask(self) -> u32 {
        match self {
            Axis::U => 0b001,
            Axis::V => 0b010,
            Axis::W => 0b100,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpaceFolding {
    pub lattice_basis: Mat3,
    pub lattice_basis_inv: Mat3, 
    pub min_half_thickness: f32,
    pub active_mask: u32, // bit0=u, bit1=v, bit2=w
}
impl SpaceFolding {
    pub fn new_3d(basis: Mat3) -> Self {
        Self::new_with_mask(basis, 0b111)
    }

    pub fn new_2d(basis: Mat3, a: Axis, b: Axis) -> Self {
        let mask = a.to_mask() | b.to_mask();
        Self::new_with_mask(basis, mask)
    }

    pub fn new_1d(basis: Mat3, a: Axis) -> Self {
        Self::new_with_mask(basis, a.to_mask())
    }

    pub fn new_with_mask(lattice_basis: Mat3, active_mask: u32) -> Self {
        // sanity: basis must be invertible
        let det = lattice_basis.det();
        debug_assert!(det != 0.0, "SpaceFolding basis must be invertible");
        let lattice_basis_inv = lattice_basis.inv();

        // axis lengths in world space
        let u_len = (lattice_basis.a11*lattice_basis.a11
                + lattice_basis.a21*lattice_basis.a21
                + lattice_basis.a31*lattice_basis.a31).sqrt();
        let v_len = (lattice_basis.a12*lattice_basis.a12
                + lattice_basis.a22*lattice_basis.a22
                + lattice_basis.a32*lattice_basis.a32).sqrt();
        let w_len = (lattice_basis.a13*lattice_basis.a13
                + lattice_basis.a23*lattice_basis.a23
                + lattice_basis.a33*lattice_basis.a33).sqrt();

        // only consider active axes when computing safety thickness
        let mut min_len = f32::INFINITY;
        if (active_mask & 0b001) != 0 { min_len = min_len.min(u_len); }
        if (active_mask & 0b010) != 0 { min_len = min_len.min(v_len); }
        if (active_mask & 0b100) != 0 { min_len = min_len.min(w_len); }
        let min_half_thickness = if min_len.is_finite() { 0.5 * min_len } else { 0.0 };

        Self { lattice_basis, lattice_basis_inv, min_half_thickness, active_mask }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuSpaceFolding {
    pub lattice_basis: Mat3,
    pub lattice_basis_inv: Mat3,
    pub min_half_thickness: f32,
    pub active_mask: u32, // bit0=u, bit1=v, bit2=w
}
impl Default for GpuSpaceFolding {
    fn default() -> Self {
        GpuSpaceFolding {
            lattice_basis: Mat3::Zero,
            lattice_basis_inv: Mat3::Zero,
            min_half_thickness: 0.0,
            active_mask: 0,
        }
    }
}

impl World {
    pub fn insert_space_folding(&mut self, e: Entity, s: SpaceFolding) {
        self.space_foldings.insert(e, s);
    }
    pub fn remove_space_folding(&mut self, e: Entity) -> bool{
        let mut scene_updated:bool = false;
        if let Some(_fold_slot) = self.folding_gpu_indices.get(e) {
            self.folding_gpu_indices.free_for(e);
            scene_updated = true;
        }
        self.space_foldings.remove(e);
        return scene_updated
    }
    pub(crate) fn sync_space_folding(&mut self) -> Result<bool, Box<dyn Error>> {
        let mut scene_updated:bool = false;
        // 2. Sync space foldings (dirty or new)
        for (_fold, idx) in self.space_foldings.iter_dirty() {
            let e = Entity { index: idx, generation: self.gens[idx as usize] };
            let gpu_slot = self.folding_gpu_indices.get_or_allocate_for(e);
            let folding = self.space_foldings.get(e).unwrap(); // safe
            let gpu_struct = GpuSpaceFolding {
                lattice_basis: folding.lattice_basis,
                lattice_basis_inv: folding.lattice_basis_inv,
                min_half_thickness: folding.min_half_thickness,
                active_mask: folding.active_mask,
            };
            self.gpu_foldings.push(gpu_slot, &gpu_struct)?;
            scene_updated = true;
        }

        return Ok(scene_updated)
    }
}