pub mod ecs_gpu_interface {

    use std::{error::Error, collections::HashMap, path::PathBuf, path::Path};
    use image::ImageReader;
    use cuda_driver_sys::CUdeviceptr;
    use crate::ecs::math_op::math_op::{Vec3, Mat3};
    use crate::cuda_wrapper::CudaContext;
    use crate::ecs::ecs::Entity;


    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct ImageRayAccum {
        pub ray_per_pixel: CUdeviceptr,
        pub image: CUdeviceptr,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct GpuCamera {
        /* ── camera intrinsics / pose ───────────────────────────── */
        pub position:        Vec3,
        pub u:               Vec3,
        pub v:               Vec3,
        pub w:               Vec3,
        pub aperture:        f32,
        pub focus_dist:      f32,
        pub viewport_width:  f32,
        pub viewport_height: f32,

        /* ── image & ray buffers ────────────────────────────────── */
        pub rand_states: CUdeviceptr,          // 0 when aperture == 0
        pub origins:     CUdeviceptr,          // float3 * W*H
        pub directions:  CUdeviceptr,          // float3 * W*H
        pub accum:       ImageRayAccum,
        pub image:       CUdeviceptr,

        /* ── misc ───────────────────────────────────────────────── */
        pub spp:    u32,
        pub width:  u32,
        pub height: u32,
        pub rand_seed_init_count: u32,
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
    
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct GpuMaterial {
        pub color: [f32; 3],
        pub use_texture: u32,
        pub texture_data_pointer: CUdeviceptr,
        pub width: u32,
        pub height: u32,
        // pad if necessary for alignment
    }
    impl Default for GpuMaterial {
        fn default() -> Self {
            GpuMaterial {
                color: [0.0; 3],
                use_texture: 0,
                texture_data_pointer: 0,
                width: 0,
                height: 0,
            }
        }
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
    
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct GpuSpaceFolding {
        pub lattice_basis: Mat3,
        pub lattice_basis_inv: Mat3,
    }
    impl Default for GpuSpaceFolding {
        fn default() -> Self {
            GpuSpaceFolding {
                lattice_basis: Mat3::Zero,
                lattice_basis_inv: Mat3::Zero,
            }
        }
    }

    #[derive(Clone, Debug, Copy)]
    pub struct TextureHandle {
        pub d_ptr:  CUdeviceptr,
        pub width:  u32,
        pub height: u32,
        id:         usize,        // identifiant unique
    }
    impl Default for TextureHandle {
        fn default() -> Self {
            TextureHandle {
                d_ptr:  0,
                width:  0,
                height: 0,
                id:     0,  
            }
        }
    }

    struct Entry {
        handle:    TextureHandle,
        ref_count: usize,
    }

    pub struct TextureManager {
        next_id: usize,
        cache:   HashMap<PathBuf, Entry>,
    }

    impl TextureManager {
        pub fn new() -> Self { Self { next_id: 1, cache: HashMap::new() } }
    
        pub fn load(&mut self, path: &Path) -> Result<TextureHandle, Box<dyn Error>> {
            if let Some(e) = self.cache.get_mut(path) {
                e.ref_count += 1;
                return Ok(e.handle.clone());
            }

            let img = ImageReader::open(path)?.decode()?.to_rgb8();
            let (w, h) = img.dimensions();
            let buffer = img.into_raw();
            let bytes   = buffer.len() * std::mem::size_of::<u8>();
            let d_ptr   = CudaContext::allocate_tensor(&buffer, bytes)?;

            let handle = TextureHandle { d_ptr, width: w, height: h, id: self.next_id };
            self.next_id += 1;

            self.cache.insert(path.to_path_buf(), Entry { handle: handle.clone(), ref_count: 1 });
            Ok(handle)
        }

        pub fn release(&mut self, h: &TextureHandle) -> Result<(), Box<dyn Error>> {
        // find the path whose entry owns this handle
            if let Some(path) = self.cache.iter()
            .find_map(|(k, e)| (e.handle.id == h.id).then_some(k.clone())) {

            let entry = self.cache.get_mut(&path).unwrap();
            if entry.ref_count == 0 { return Ok(()); }          // already freed – shouldn’t happen
            entry.ref_count -= 1;

            if entry.ref_count == 0 {
                CudaContext::synchronize();
                CudaContext::free_device_memory(entry.handle.d_ptr)?;
                self.cache.remove(&path);
            }
            }
            Ok(())
        }
    }

    /// Maps Entity → GPU index, with free-list reuse.
    pub struct GpuIndexMap {
        sparse: Vec<usize>,    // entity.index -> gpu index or usize::MAX
        free: Vec<usize>,      // recycled gpu indices
        next: usize,           // next fresh index when free is empty
    }

    impl GpuIndexMap {
        pub fn new() -> Self {
            Self {
                sparse: vec![],
                free: vec![],
                next: 0,
            }
        }

        pub fn allocate_for(&mut self, e: Entity) -> usize {
            let gpu_index = self.free.pop().unwrap_or_else(|| {
                let idx = self.next;
                self.next += 1;
                idx
            });
            if e.index as usize >= self.sparse.len() {
                self.sparse.resize(e.index as usize + 1, usize::MAX);
            }
            self.sparse[e.index as usize] = gpu_index;
            gpu_index
        }

        pub fn get(&self, e: Entity) -> Option<usize> {
            self.sparse
                .get(e.index as usize)
                .copied()
                .filter(|&v| v != usize::MAX)
        }

        pub fn get_or_allocate_for(&mut self, e: Entity) -> usize {
            if let Some(idx) = self.get(e) {
                idx
            } else {
                self.allocate_for(e)
            }
        }

        pub fn free_for(&mut self, e: Entity) {
            if let Some(idx) = self.get(e) {
                self.free.push(idx);
                self.sparse[e.index as usize] = usize::MAX;
            }
        }
    }
}