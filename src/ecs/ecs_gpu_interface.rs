pub mod ecs_gpu_interface {

    use std::error::Error;
    use image::ImageReader;
    use cuda_driver_sys::CUdeviceptr;
    use crate::ecs::math_op::math_op::Vec3;
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
                active: 0,
            }
        }
    }

    pub fn load_texture(path: &str) -> Result<(CUdeviceptr, u32, u32), Box<dyn Error>> {
        let img = ImageReader::open(path)?.decode()?.to_rgb8();
        let (width, height) = img.dimensions();
        let buffer = img.into_raw(); // Vec<u8>
        let texture_size = buffer.len() * std::mem::size_of::<u8>();
        let d_texture = CudaContext::allocate_tensor(&buffer, texture_size)?;
        Ok((d_texture, width, height))
    }
}