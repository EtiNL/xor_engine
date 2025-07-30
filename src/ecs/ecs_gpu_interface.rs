pub mod ecs_gpu_interface {

    use std::{error::Error, collections::HashMap, path::PathBuf, path::Path};
    use image::ImageReader;
    use cuda_driver_sys::CUdeviceptr;
    use crate::ecs::math_op::math_op::{Vec3, Mat3};
    use crate::cuda_wrapper::CudaContext;

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct CameraObject {
        pub position: Vec3,
        pub u: Vec3,
        pub v: Vec3,
        pub w: Vec3,
        pub aperture: f32,
        pub focus_dist: f32,
        pub viewport_width: f32,
        pub viewport_height: f32,

    }

    impl CameraObject {
        pub fn new(
            position: Vec3,
            u: Vec3,
            v: Vec3,
            w: Vec3,
            fov: f32,
            width: u32,
            height: u32,
            aperture: f32,
            focus_dist: f32,
        ) -> Self {
            let aspect_ratio = width as f32 / height as f32;

            // Champ de vision vertical → taille plan image
            let theta = fov.to_radians();
            let viewport_height = 2.0 * (theta / 2.0).tan();
            let viewport_width = aspect_ratio * viewport_height;

            Self {
                position,
                u,
                v,
                w,
                aperture,
                focus_dist,
                viewport_width,
                viewport_height,
            }
        }
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


    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct SdfObject {
        pub sdf_type: u32,
        pub params: [f32; 3],
        pub center: Vec3,
        pub u: Vec3,
        pub v: Vec3,
        pub w: Vec3,
        pub texture: CUdeviceptr,           // pointeur vers image device
        pub tex_width: u32,
        pub tex_height: u32,
        pub lattice_basis: Mat3,
        pub lattice_basis_inv: Mat3,
        pub active: u32,
    }

    impl Default for SdfObject {
        fn default() -> Self {
            SdfObject {
                sdf_type: 0,
                params: [0.0; 3],
                center: Vec3::default(),
                u: Vec3::default(),
                v: Vec3::default(),
                w: Vec3::default(),
                texture: 0,
                tex_width: 0,
                tex_height: 0,
                lattice_basis: Mat3::Zero,
                lattice_basis_inv: Mat3::Zero,
                active: 0,
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
                CudaContext::free_device_memory(entry.handle.d_ptr)?;
                self.cache.remove(&path);
            }
            }
            Ok(())
        }
    }
}