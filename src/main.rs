mod cuda_wrapper;
mod display;
mod ecs;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use std::error::Error;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use std::path::Path;
use cuda_driver_sys::cuGraphAddDependencies;
use cuda_driver_sys::CUdeviceptr;

use display::{Display, FpsCounter};
use cuda_wrapper::{CudaContext, dim3, CameraBuffers};
use ecs::ecs::{World, update_rotation, Transform, Camera, SdfBase, MaterialComponent, Rotating, SpaceFolding};
use ecs::math_op::math_op::{Vec3, Quat, Mat3};
use crate::ecs::ecs_gpu_interface::ecs_gpu_interface::{SdfType, TextureManager};


fn main() -> Result<(), Box<dyn Error>> {
    // Initialization
    let mut cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;

    let mut last_frame_time = Instant::now();

    let mut world = World::new()?;

    // — CAMERA — 
    let cam_ent = world.spawn();

    let width: u32 = 800;
    let height: u32 = 600;
    let sample_per_pixel: u32 = 1;
    let field_of_view: f32 = 45.0;
    let aperture: f32  = 0.0;
    let focus_distance: f32 = 10.0;

    world.insert_camera(
            cam_ent,
            Camera::new(field_of_view, width, height, aperture, focus_distance, sample_per_pixel),
            &cuda_context,
    )?;
    world.insert_transform(cam_ent, Transform::default());
    world.active_camera(cam_ent); 

    // Texture manager stays the same
    let mut tex_mgr = TextureManager::new();

    // — CUBE — 
    let mut cube_ent = world.spawn();
    world.insert_transform(cube_ent, Transform {
        position: Vec3::new(0.0, 0.0, -10.0),
        rotation: Quat::identity(),
    });
    // 1) SDF shape
    world.insert_sdf_base(cube_ent, SdfBase {
        sdf_type: SdfType::Cube,
        params: [1.0, 1.0, 1.0], // half-extents
    });
    // 2) Material
    let cube_tex = tex_mgr.load(Path::new("./src/textures/lines_texture.png"))?;
    world.insert_material(cube_ent, MaterialComponent {
        color: [1.0, 1.0, 1.0],
        texture: Some(cube_tex),
        use_texture: true,
    });
    // 3) Rotation
    world.insert_rotating(cube_ent, Rotating {
        speed_deg_per_sec: 30.0,
    });

    // — SPHERE — 
    let sphere_ent = world.spawn();
    world.insert_transform(sphere_ent, Transform {
        position: Vec3::new(0.0, 0.0, -10.0),
        rotation: Quat::identity(),
    });
    // SDF
    world.insert_sdf_base(sphere_ent, SdfBase {
        sdf_type: SdfType::Sphere,
        params: [1.5, 0.0, 0.0], // radius, unused, unused
    });
    // Material
    let wood_tex = tex_mgr.load(Path::new("./src/textures/Wood_texture.png"))?;
    world.insert_material(sphere_ent, MaterialComponent {
        color: [1.0, 1.0, 1.0],
        texture: Some(wood_tex),
        use_texture: true,
    });
    // Lattice‐folding
    world.insert_space_folding(sphere_ent, SpaceFolding::new(Mat3::Id * 10.0));
    // Rotation
    world.insert_rotating(sphere_ent, Rotating {
        speed_deg_per_sec: 30.0,
    });

    // Initialize the SDL2 context
    let mut display = Display::new(
        "xor Renderer",
        width,
        height,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )?;
    let mut fps_counter = FpsCounter::new();
    let mut host_img: Vec<u8> = vec![0; (width * height * 3) as usize];
    let img_size = host_img.len() * std::mem::size_of::<u8>();

    // load kernels
    cuda_context.load_kernel("generate_rays")?;
    cuda_context.load_kernel("raymarch")?;
    cuda_context.load_kernel("reset_accum")?;

    // Create CUDA streams
    cuda_context.create_stream("stream1")?;

    // Allocate GPU memory
    let mut first_image: bool = true;
    world.update_scene(&mut tex_mgr)?;

    // grid dim and block dim
    let grid_dim = dim3 {
        x: ((width + 15) / 16) as u32,
        y: ((height + 15) / 16) as u32,
        z: 1,
    };
    
    let block_dim = dim3 { x: 16, y: 16, z: 1 };

    // Create a CUDA graph
    let mut graph = cuda_context.create_cuda_graph()?;

    let mut cam_ptr       : CUdeviceptr = world.gpu_cameras.ptr();
    let mut cam_index : u32         = world.active_camera_index() as u32;

    let mut sdf_ptr       : CUdeviceptr = world.gpu_sdf_objects.ptr();
    let mut num_sdfs  : u32         = world.gpu_sdf_objects.len as u32;

    let mut mat_ptr       : CUdeviceptr = world.gpu_materials.ptr();
    let mut light_ptr     : CUdeviceptr = world.gpu_lights.ptr();
    let mut fold_ptr      : CUdeviceptr = world.gpu_foldings.ptr();

    let params_generate_rays: [*const c_void; 4] = [
        &width   as *const _ as *const c_void,
        &height  as *const _ as *const c_void,
        &cam_ptr     as *const _ as *const c_void,
        &cam_index as *const _ as *const c_void,
    ];

    let params_raymarch: [*const c_void; 9] = [
        &width   as *const _ as *const c_void,
        &height  as *const _ as *const c_void,
        &cam_ptr      as *const _ as *const c_void,
        &cam_index as *const _ as *const c_void,
        &sdf_ptr      as *const _ as *const c_void,
        &num_sdfs as *const _ as *const c_void,
        &mat_ptr      as *const _ as *const c_void,
        &light_ptr    as *const _ as *const c_void,
        &fold_ptr     as *const _ as *const c_void,
    ];

    cuda_context.add_graph_kernel_node(
        &mut graph, "generate_rays", &params_generate_rays[..], grid_dim, block_dim)?;
    cuda_context.add_graph_kernel_node(
        &mut graph, "raymarch",      &params_raymarch[..],      grid_dim, block_dim)?;

    // Instantiate the CUDA graph
    let graph_exec = cuda_context.instantiate_graph(graph)?;

    // Parameters to be used in the graph
    let mut x_click = 0f32;
    let mut y_click = 0f32;
    let mouse_down = Arc::new(Mutex::new(false)); // mouse_down wrapped in Arc<Mutex<_>> for thread-safe access
    let mut rotation_dir_x_axis = 0.0f32;
    let mut rotation_dir_y_axis = 0.0f32;

    // Main loop
    'running: loop {
        for event in display.poll_events() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::KeyDown { keycode: Some(Keycode::Left), .. } => {
                    rotation_dir_y_axis = -1.0;
                },
                Event::KeyDown { keycode: Some(Keycode::Right), .. } => {
                    rotation_dir_y_axis = 1.0;
                },
                Event::KeyUp { keycode: Some(Keycode::Left | Keycode::Right), .. } => {
                    rotation_dir_y_axis = 0.0;
                },
                Event::KeyDown { keycode: Some(Keycode::Up), .. } => {
                    rotation_dir_x_axis = -1.0;
                },
                Event::KeyDown { keycode: Some(Keycode::Down), .. } => {
                    rotation_dir_x_axis = 1.0;
                },
                Event::KeyUp { keycode: Some(Keycode::Up | Keycode::Down), .. } => {
                    rotation_dir_x_axis = 0.0;
                },

                Event::KeyDown { keycode: Some(Keycode::S), .. } => {
                    cube_ent = world.spawn();
                    world.insert_transform(cube_ent, Transform {
                        position: Vec3::new(0.0, 0.0, -10.0),
                        rotation: Quat::identity(),
                    });
                    world.insert_sdf_base(cube_ent, SdfBase {
                    sdf_type: SdfType::Cube,
                    params: [1.0,1.0,1.0],
                    });
                    let tex = tex_mgr.load(Path::new("./src/textures/lines_texture.png"))?;
                    world.insert_material(cube_ent, MaterialComponent {
                    color: [1.0,1.0,1.0],
                    texture: Some(tex),
                    use_texture: true,
                    });
                    world.insert_rotating(cube_ent, Rotating { speed_deg_per_sec:30.0 });
                },

                Event::MouseButtonDown { x, y, .. } => {
                    world.despawn(cube_ent);

                    let mut mouse_down_lock = mouse_down.lock().unwrap();
                    *mouse_down_lock = true;
                    x_click = x as f32;
                    y_click = y as f32;
                },
                Event::MouseButtonUp { .. } => {
                    let mut mouse_down_lock = mouse_down.lock().unwrap();
                    *mouse_down_lock = false;
                },
                Event::MouseMotion { x, y, .. } => {
                    if *mouse_down.lock().unwrap() {
                        x_click = x as f32;
                        y_click = y as f32;
                    }
                },
                _ => {}
            }
        }

        let now = Instant::now();
        let dt = now.duration_since(last_frame_time).as_secs_f32();
        last_frame_time = now;

        if (rotation_dir_y_axis != 0.0) || (rotation_dir_x_axis != 0.0){
            let rotation_dir: Vec3 = Vec3{x:rotation_dir_x_axis, y:rotation_dir_y_axis, z: 0.0};
            update_rotation(&mut world, dt, rotation_dir);
        }

        if first_image || world.update_scene(&mut tex_mgr)? {

            if !first_image {
                cam_ptr       = world.gpu_cameras.ptr();
                cam_index = world.active_camera_index() as u32;

                sdf_ptr       = world.gpu_sdf_objects.ptr();
                num_sdfs  = world.gpu_sdf_objects.len as u32;

                mat_ptr       = world.gpu_materials.ptr();
                light_ptr     = world.gpu_lights.ptr();
                fold_ptr      = world.gpu_foldings.ptr();


                /* ---------- re-apply the same pointer arrays ---------- */
                cuda_context.exec_kernel_node_set_params(
                        graph_exec, "generate_rays", &params_generate_rays[..])?;
                cuda_context.exec_kernel_node_set_params(
                        graph_exec, "raymarch",      &params_raymarch[..])?;
            }

            for _ in 0..sample_per_pixel {
                cuda_context.launch_graph(graph_exec)?;
            }

            CudaContext::synchronize()?;
            // Copy the result from GPU to CPU memory
            
            {
                let cb = world.cam_bufs.as_ref().expect("camera buffers not set");
                cuda_context.retrieve_tensor(cb.image, &mut host_img, img_size)?;
            
                if sample_per_pixel > 1 {

                    let total_pixels   = (width * height) as usize;
                    let d_ray_per_pixel = cb.ray_per_pixel; // CUdeviceptr

                    let reset_params = vec![
                        &d_ray_per_pixel as *const _ as *const c_void,
                        &total_pixels as *const _ as *const c_void,
                    ];

                    cuda_context.launch_kernel(
                        "reset_accum",
                        dim3 { x: ((total_pixels as u32 + 255) / 256), y: 1, z: 1 },
                        dim3 { x: 256, y: 1, z: 1 },
                        &reset_params,
                        "stream1",
                    )?;
                }
            }
        }

        // display Image and fps
        let fps = fps_counter.update();
        display.render_texture(&host_img, (width * 3) as usize)?;
        display.render_fps(fps)?;
        display.present();

        if first_image { first_image = false;}
    }

    CudaContext::free_device_memory( world.cam_bufs.as_ref().unwrap().image)?;
    
    
    // After graph execution is complete
    cuda_context.free_graph(graph)?;
    cuda_context.free_graph_exec(graph_exec)?;

    Ok(())
}
