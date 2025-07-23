mod cuda_wrapper;
mod scene_composition;
mod texture_utils;
mod display;
mod ecs;

use cuda_wrapper::{CudaContext, DeviceBuffer, KernelArg};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use std::error::Error;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use display::{Display, FpsCounter};
use cuda_wrapper::{CudaContext,SceneBuffer, dim3};
use scene_composition::{Camera, ImageRayAccum, Vec3, Quat};
use texture_utils::load_texture;
use ecs::{World, update_rotation, Transform, Renderable, Rotating};

<<<<<<< HEAD
fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the SDL2 context
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string())?;

    // Customize screen size
    let width = 960u32;
    let height = 540u32;

    let window = video_subsystem
        .window("XOR", width, height)
        .position_centered()
        .build()
        .expect("Failed to create window");

    let mut canvas = window.into_canvas().build().expect("Failed to create canvas");
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator.create_texture_streaming(PixelFormatEnum::RGB24, width, height)?;

    let mut sdl_event_pump = sdl_context.event_pump()?;

=======

fn main() -> Result<(), Box<dyn Error>> {
>>>>>>> debbug_branch
    // Initialize CUDA context
    let mut cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;

<<<<<<< HEAD
    // Load a font
    let font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; // Replace with a path to a valid TTF file
    let font = ttf_context.load_font(font_path, 16)
        .map_err(|e| {
            eprintln!("Failed to load font: {}", e);
            e.to_string()
        })?;

    // Main loop
    let mut last_frame = Instant::now();
    let mut frame_count = 0;
    let mut fps = 0;
    let mut x_screen = 0.0f32;
    let mut y_screen = 0.0f32;
    let mut theta_0 = std::f32::consts::PI / 2.0;
    let mut phi_0 = 0.0f32;
    let mut theta_1 = 0.0f32;
    let mut phi_1 = 0.0f32;
    let mut mouse_down = false; // Track if mouse button is pressed

=======
    let sample_per_pixel: u32 = 10;
    let width: u32 = 800;
    let height: u32 = 600;

    // Camera and Scene initialization

    let cam = Camera::new(
        Vec3 { x: 0.0, y: 0.0, z: 0.0 },           // position
        Vec3 { x: 1.0, y: 0.0, z: 0.0 },           // u (right)
        Vec3 { x: 0.0, y: 1.0, z: 0.0 },           // v (up)
        Vec3 { x: 0.0, y: 0.0, z: -1.0 },          // w (forward)
        45.0,                                      // fov in degrees
        width,                                     // width
        height,                                    // height
        0.0,                                       // aperture
        1.0                                        // focus_dist
    );

    let mut last_frame_time = Instant::now();
    

    let (wood_texture, tex_w, tex_h) = load_texture("./src/textures/lines_texture.png")?;
    let wood_texture_size = wood_texture.len() * std::mem::size_of::<u8>();
    let d_wood_texture = CudaContext::allocate_tensor(&wood_texture, wood_texture_size)?;

    let mut world = World::new();

    let  cube = world.spawn();
    world.insert_transform(cube, Transform {
        position: Vec3::new(0.0, 0.0, -5.0),
        rotation: Quat::identity(),
    });
    world.insert_renderable(cube, Renderable {
        sdf_type: 1,
        params: [1.0, 1.0, 1.0],
        texture: d_wood_texture,
        tex_width: tex_w,
        tex_height: tex_h,
    });
    world.insert_rotating(cube, Rotating {
        speed_deg_per_sec: 30.0,
    });

    let sphere = world.spawn();
    world.insert_transform(sphere, Transform {
        position: Vec3::new(0.0, 0.0, -5.0),
        rotation: Quat::identity(),
    });
    world.insert_renderable(sphere, Renderable {
        sdf_type: 0,
        params: [1.5, 0.0, 0.0],
        texture: d_wood_texture,
        tex_width: tex_w,
        tex_height: tex_h,
    });
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
    let mut scene_buf = SceneBuffer::new(8)?;   // 8 = nombre de sdf object initiale arbitraire

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
>>>>>>> debbug_branch
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

                Event::MouseButtonDown { x, y, .. } => {
<<<<<<< HEAD
                    mouse_down = true;
                    // Adjust angle based on mouse position
                    x_screen = x as f32;
                    y_screen = y as f32;
                },
                Event::MouseButtonUp { .. } => {
                    mouse_down = false;
                    let theta_0 = theta_1;
                    let phi_0 = phi_1;
                    theta_1 = 0.0f32;
                    phi_1 = 0.0f32;
                },
                Event::MouseMotion { x, y, .. } => {
                    if mouse_down {
                        let delta_x = x as f32 - x_screen;
                        let delta_y = y as f32 - y_screen;
                        theta_1 = (delta_y - height as f32 / 2.0).atan();
                        phi_1 = (delta_x - width as f32 / 2.0).atan();
=======
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
>>>>>>> debbug_branch
                    }
                },
                _ => {}
            }
        }

<<<<<<< HEAD
        // Update the kernel arguments
        let width_cuda = DeviceBuffer::new(vec![width as i32]);
        let height_cuda = DeviceBuffer::new(vec![height as i32]);
        let sphere_x = DeviceBuffer::new(vec![0.0f32]);
        let sphere_y = DeviceBuffer::new(vec![10.0f32]);
        let sphere_z = DeviceBuffer::new(vec![0.0f32]);
        let radius = DeviceBuffer::new(vec![5.0f32]);
        let image = DeviceBuffer::new(vec![0u8; (width * height * 3) as usize]);

        let theta = DeviceBuffer::new(vec![theta_0 + theta_1]);
        let phi = DeviceBuffer::new(vec![phi_0 + phi_1]);

        let mut args: Vec<Box<dyn KernelArg>> = vec![
            Box::new(width_cuda),
            Box::new(height_cuda),
            Box::new(sphere_x),
            Box::new(sphere_y),
            Box::new(sphere_z),
            Box::new(radius),
            Box::new(theta),
            Box::new(phi),
            Box::new(image),
        ];

        // println!("theta: {}, phi: {}", theta_0 + theta_1, phi_0 + phi_1);

        // Launch the CUDA kernel
        match cuda_context.launch_kernel(&mut args, width, height) {
            Ok(_) => (),
            Err(e) => eprintln!("Failed to launch CUDA kernel: {}", e),
        }

        // Update the texture with the new image data
        let image_arg = args.last().unwrap();
        if let Some(image_buffer) = image_arg.as_any().downcast_ref::<DeviceBuffer<u8>>() {
            for i in 0..10 {
                println!("{}: {}", i, image_buffer.get_host_data()[i]);
            }
            texture.update(None, image_buffer.get_host_data(), (width * 3) as usize)?;
        }

        // Drop all the args here
        drop(args);

        canvas.copy(&texture, None, None)?;
=======
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
>>>>>>> debbug_branch

        let now = Instant::now();
        let dt = now.duration_since(last_frame_time).as_secs_f32();
        last_frame_time = now;
        let rotation_dir: Vec3 = Vec3{x:rotation_dir_x_axis, y:rotation_dir_y_axis, z: 0.0};
        update_rotation(&mut world, dt, rotation_dir);

        // créer la scène à envoyer vers CUDA
        let scene_cpu = world.render_scene();
        scene_buf.upload(&cuda_context, &scene_cpu)?;

        let params_raymarch = vec![
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &d_origins as *const _ as *const c_void,
            &d_directions as *const _ as *const c_void,
            &scene_buf.ptr() as *const _ as *const c_void,
            &(scene_cpu.len() as i32) as *const _ as *const c_void,
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

<<<<<<< HEAD
        // Render FPS text
        let fps_text = format!("FPS: {}", fps);
        let surface = font.render(&fps_text)
                          .blended(sdl2::pixels::Color::RGBA(255, 0, 255, 255))
                          .map_err(|e| {
                              eprintln!("Failed to render text surface: {}", e);
                              e.to_string()
                          })?;
        let fps_texture = texture_creator.create_texture_from_surface(&surface)
                                         .map_err(|e| {
                                             eprintln!("Failed to create texture from surface: {}", e);
                                             e.to_string()
                                         })?;
        
        let TextureQuery { width, height, .. } = fps_texture.query();
        let target = Rect::new(128 - width as i32 - 10, 10, width, height);
        
        canvas.set_blend_mode(BlendMode::Blend);
        canvas.copy(&fps_texture, None, Some(target))?;
        
        canvas.present();
    }

    // Drop CUDA context explicitly to unload the module and free GPU memory
    drop(cuda_context);
=======
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

        // display Image and fps
        let fps = fps_counter.update();
        display.render_texture(&image, (width * 3) as usize)?;
        display.render_fps(fps)?;
        display.present();
    }

    CudaContext::free_device_memory(d_image)?;
>>>>>>> debbug_branch

    Ok(())
}
