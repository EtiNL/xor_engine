mod cuda_wrapper;
mod display;
mod ecs;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use std::error::Error;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use display::{Display, FpsCounter};
use cuda_wrapper::{CudaContext, SceneBuffer, ImageRayAccum, dim3};
use ecs::ecs::{World, update_rotation, Transform, Camera, Renderable, Rotating};
use ecs::math_op::math_op::{Vec3, Quat};
use crate::ecs::ecs_gpu_interface::ecs_gpu_interface::SdfType;


fn main() -> Result<(), Box<dyn Error>> {
    // Initialize CUDA context
    let mut cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;

    let width: u32 = 800;
    let height: u32 = 600;
    let sample_per_pixel: u32 = 2;

    let mut last_frame_time = Instant::now();

    let mut world = World::new();

    let camera = world.spawn();
    world.insert_camera(camera, Camera {
        name: "first".to_string(),
        field_of_view: 45.0,    // in degrees
        width: width,
        height: height,
        aperture: 0.0,
        focus_distance: 1.0
    });
    world.insert_transform(camera, Transform {
        position: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        rotation: Quat::identity(), // need to be changed so z = -1
    });

    let mut cube = world.spawn();
    world.insert_transform(cube, Transform {
        position: Vec3::new(0.0, 0.0, -5.0),
        rotation: Quat::identity(),
    });
    world.insert_renderable(cube, Renderable::new(
        SdfType::Cube,
        [1.0, 1.0, 1.0], // params
        "./src/textures/lines_texture.png",

    ));
    world.insert_rotating(cube, Rotating {
        speed_deg_per_sec: 30.0,
    });

    let sphere = world.spawn();
    world.insert_transform(sphere, Transform {
        position: Vec3::new(0.0, 0.0, -5.0),
        rotation: Quat::identity(),
    });
    world.insert_renderable(sphere, Renderable::new(
        SdfType::Sphere,
        [1.5, 0.0, 0.0], // params
        "./src/textures/lines_texture.png",
    ));
    world.insert_rotating(sphere, Rotating {
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

    // load kernels
    cuda_context.load_kernel("init_random_states")?;
    cuda_context.load_kernel("generate_rays")?;
    cuda_context.load_kernel("raymarch")?;
    cuda_context.load_kernel("reset_accum")?;

    // Create CUDA streams
    cuda_context.create_stream("stream1")?;

    // Allocate GPU memory
    let mut scene_buf = SceneBuffer::new(2)?;   // 8 = nombre de sdf object initiale arbitraire
    let scene_cpu = world.render_scene(scene_buf.capacity);
    let mut first_image: bool = true;
    scene_buf.upload(&scene_cpu)?;

    let cam = world.choose_camera("first")?;
    let d_camera = CudaContext::allocate_struct(&cam)?;
    
    let d_rand_states = CudaContext::allocate_curand_states(width, height)?;

    let mut directions: Vec<f32> = vec![0f32; (width * height * 3) as usize];
    let directions_size = directions.len() * std::mem::size_of::<f32>();
    let d_directions = CudaContext::allocate_tensor(&directions, directions_size)?;

    let mut origins: Vec<f32> = vec![0f32; (width * height * 3) as usize];
    let origins_size = origins.len() * std::mem::size_of::<f32>();
    let d_origins = CudaContext::allocate_tensor(&origins, origins_size)?;

    let total_pixels = (width * height) as usize;
    let rays_size = total_pixels * std::mem::size_of::<i32>();
    let d_ray_per_pixel = CudaContext::allocate_tensor(&vec![0i32; total_pixels], rays_size)?;


    let mut image: Vec<u8> = vec![0u8; (width * height * 3) as usize];
    let image_size = image.len() * std::mem::size_of::<u8>();
    let d_image = CudaContext::allocate_tensor(&image, image_size)?;

    let ray_accum = ImageRayAccum {
        ray_per_pixel: d_ray_per_pixel,
        image: d_image,
    };
    let d_ray_accum = CudaContext::allocate_struct(&ray_accum)?;

    // grid dim and block dim
    let grid_dim = dim3 {
        x: ((width + 31) / 32) as u32,
        y: ((height + 31) / 32) as u32,
        z: 1,
    };
    
    let block_dim = dim3 { x: 32, y: 32, z: 1 };

    // Initialize Cuda random state
    let seed = 1530u32;
    let init_rng_params: Vec<*const c_void> = vec![
        &d_rand_states as *const _ as *const c_void,
        &width as *const _ as *const c_void,
        &height as *const _ as *const c_void,
        &seed as *const _ as *const c_void,
    ];

    cuda_context.launch_kernel("init_random_states", 
        grid_dim, 
        block_dim, 
        &init_rng_params, 
        "stream1")?;

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
                    cube = world.spawn();
                    world.insert_transform(cube, Transform {
                        position: Vec3::new(0.0, 0.0, -5.0),
                        rotation: Quat::identity(),
                    });
                    world.insert_renderable(cube, Renderable::new(
                        SdfType::Cube,
                        [1.0, 1.0, 1.0], // params
                        "./src/textures/lines_texture.png",

                    ));
                    world.insert_rotating(cube, Rotating {
                        speed_deg_per_sec: 30.0,
                    });
                },

                Event::MouseButtonDown { x, y, .. } => {
                    world.despawn(cube);

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

        if first_image || world.update_scene(&mut scene_buf) {

            // Create a CUDA graph
            let mut graph = cuda_context.create_cuda_graph()?;

            let params_generate_rays: Vec<*const c_void> = vec![
                &width as *const _ as *const c_void,
                &height as *const _ as *const c_void,
                &d_camera as *const _ as *const c_void,
                &d_rand_states as *const _ as *const c_void,
                &d_origins as *const _ as *const c_void,
                &d_directions as *const _ as *const c_void,
            ];
            
            cuda_context.add_graph_kernel_node(
                &mut graph,
                "generate_rays",
                &params_generate_rays,
                grid_dim,
                block_dim,
            )?;

            let params_raymarch = vec![
                &width as *const _ as *const c_void,
                &height as *const _ as *const c_void,
                &d_origins as *const _ as *const c_void,
                &d_directions as *const _ as *const c_void,
                &scene_buf.ptr() as *const _ as *const c_void,
                &(scene_buf.capacity as i32) as *const _ as *const c_void,
                &d_ray_accum as *const _ as *const c_void,
            ];

            cuda_context.add_graph_kernel_node(
                &mut graph,
                "raymarch",
                &params_raymarch,
                grid_dim,
                block_dim,
            )?;

            // Instantiate the CUDA graph
            let graph_exec = cuda_context.instantiate_graph(graph)?;
            // Launch the graph
            cuda_context.launch_graph(graph_exec)?;
            cuda_context.synchronize("stream1");

            
            if cam.aperture > 0.0 {
                for _i in 1..sample_per_pixel {
                    // Launch the graph
                    cuda_context.launch_graph(graph_exec)?;
                    cuda_context.synchronize("stream1");
                }
            }
            // Copy the result from GPU to CPU memory
            cuda_context.retrieve_tensor(d_image, &mut image, image_size)?;

            // After graph execution is complete
            cuda_context.free_graph(graph)?;
            cuda_context.free_graph_exec(graph_exec)?;

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

        // display Image and fps
        let fps = fps_counter.update();
        display.render_texture(&image, (width * 3) as usize)?;
        display.render_fps(fps)?;
        display.present();

        if first_image { first_image = false;}
    }

    CudaContext::free_device_memory(d_image)?;

    Ok(())
}
