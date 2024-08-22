mod cuda_wrapper;
mod dynamic_computation_graph;

use sdl2::event::Event;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, TextureQuery};
use std::error::Error;
use std::ffi::c_void;
use std::time::{Instant, Duration};
use rand::Rng;

use cuda_wrapper::{CudaContext, dim3};
use dynamic_computation_graph::{ComputationGraph, OperationType};

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the SDL2 context
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string())?;

    let width = 800u32;
    let height = 600u32;

    let window = video_subsystem
        .window("xor Renderer", width, height)
        .position_centered()
        .build()
        .expect("Failed to create window");

    let mut canvas = window.into_canvas().build().expect("Failed to create canvas");
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator.create_texture_streaming(PixelFormatEnum::RGB24, width, height)?;

    let mut sdl_event_pump = sdl_context.event_pump()?;

    let font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    let font = ttf_context.load_font(font_path, 16)
        .map_err(|e| {
            eprintln!("Failed to load font: {}", e);
            e.to_string()
        })?;

    // Initialize CUDA context and load kernels
    let mut cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;
    cuda_context.load_kernel("produit_scalaire")?;
    cuda_context.load_kernel("init_light_source")?;
    cuda_context.load_kernel("test_norm_distrib_light_source")?;
    cuda_context.load_kernel("sdf_sphere")?;
    cuda_context.load_kernel("grad_sdf_sphere")?;
    cuda_context.load_kernel("ray_march")?;
    cuda_context.load_kernel("reflexion")?;
    cuda_context.load_kernel("camera_diffusion")?;
    cuda_context.load_kernel("ray_collection")?;
    cuda_context.load_kernel("render")?;
    cuda_context.load_kernel("findMaxValue")?;
    cuda_context.load_kernel("normalizeImage")?;

    // Create CUDA streams
    cuda_context.create_stream("stream1")?;
    cuda_context.create_stream("stream2")?;


    let num_rays: u32 = 100000;
    let mu = 300f32;
    let sigma = 100f32;
    let mut rng = rand::thread_rng();

    let light_basis: Vec<f32> = vec![
        0.0, 0.0, 1.0, 
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0
    ];
    let light_basis_size = light_basis.len() * std::mem::size_of::<f32>();
    let d_light_basis = CudaContext::allocate_tensor(&light_basis, light_basis_size)?;

    let mut test: Vec<f32> = vec![0.0f32; (num_rays * 2) as usize];
    let test_size = test.len() * std::mem::size_of::<f32>();
    let d_test = CudaContext::allocate_tensor(&test, test_size)?;

    let mut o: Vec<f32> = vec![0.0f32; (num_rays * 3) as usize];
    let o_size = o.len() * std::mem::size_of::<f32>();
    let d_o = CudaContext::allocate_tensor(&o, o_size)?;

    let mut d: Vec<f32> = vec![0.0f32; (num_rays * 3) as usize];
    let d_size = d.len() * std::mem::size_of::<f32>();
    let d_d = CudaContext::allocate_tensor(&d, d_size)?;

    let mut t: Vec<f32> = vec![0.0f32; num_rays as usize];
    let t_size = t.len() * std::mem::size_of::<f32>();
    let d_t = CudaContext::allocate_tensor(&t, t_size)?;

    let mut ti: Vec<f32> = vec![0.0f32; num_rays as usize];
    let ti_size = ti.len() * std::mem::size_of::<f32>();
    let d_ti = CudaContext::allocate_tensor(&ti, ti_size)?;

    let mut tf: Vec<f32> = vec![0.0f32; num_rays as usize];
    let tf_size = tf.len() * std::mem::size_of::<f32>();
    let d_tf = CudaContext::allocate_tensor(&tf, tf_size)?;

    let mut sdf: Vec<f32> = vec![0.0f32; num_rays as usize];
    let sdf_size = sdf.len() * std::mem::size_of::<f32>();
    let d_sdf = CudaContext::allocate_tensor(&sdf, sdf_size)?;

    let mut grad_sdf: Vec<f32> = vec![0.0f32; (num_rays * 3) as usize];
    let grad_sdf_size = sdf.len() * std::mem::size_of::<f32>();
    let d_grad_sdf = CudaContext::allocate_tensor(&grad_sdf, grad_sdf_size)?;
    // Create and configure computation graph
    let mut graph = ComputationGraph::new(&cuda_context);

    let sphere_x = 3.0f32;
    let sphere_y = 0.0f32;
    let sphere_z = 0.0f32;
    let radius = 250.0f32;

    let mut x_click = 0f32;
    let mut y_click = 0f32;
    let mut theta_0 = std::f32::consts::PI / 2.0;
    let mut phi_0 = 0f32;
    let mut theta_1 = 0f32;
    let mut phi_1 = 0f32;
    let mut mouse_down = false;

    // 256 threads / blocks (max is 1024) which allows for up to 4 streams computing in parallel
    let image_grid_dim = dim3 {
        x: ((width + 15) / 16) as u32,
        y: ((height + 15) / 16) as u32,
        z: 1,
    };

    let image_block_dim = dim3 { x: 16, y: 16, z: 1 };

    let ray_grid_dim = dim3 {
        x: ((num_rays + 255) / 256) as u32,
        y: 1,
        z: 1,
    };
    let ray_block_dim = dim3 { x: 256, y: 1, z: 1 };

    // Calculate FPS
    let mut last_frame = Instant::now();
    let mut frame_count = 0;
    let mut fps = 0;

    'running: loop {
        for event in sdl_event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseButtonDown { x, y, .. } => {
                    if !mouse_down {
                        mouse_down = true;
                        x_click = x as f32;
                        y_click = y as f32;
                    }
                },
                Event::MouseButtonUp { .. } => {
                    if mouse_down {
                        mouse_down = false;
                        theta_0 = theta_1;
                        phi_0 = phi_1;
                        theta_1 = 0f32;
                        phi_1 = 0f32;
                    }
                },
                Event::MouseMotion { x, y, .. } => {
                    if mouse_down {
                        phi_1 = phi_0 + (x_click - x as f32).atan();
                        theta_1 = theta_0 + (y_click - y as f32).atan();
                        x_click = x as f32;
                        y_click = y as f32;
                    }
                },
                _ => {}
            }
        }

        // Reset the graph for each frame
        graph.clear_operations();

        // Allocate GPU memory
        let mut image: Vec<u8> = vec![0u8; (width * height * 3) as usize];
        let image_size = image.len() * std::mem::size_of::<u8>();
        let d_image = CudaContext::allocate_tensor(&image, image_size)?;

        // Seed for random number generation
        let seed: u64 = rng.gen();  
        let params_light_source_init = vec![
            &d_o as *const _ as *const c_void,
            &d_d as *const _ as *const c_void,
            &d_light_basis as *const _ as *const c_void,
            &num_rays as *const _ as *const c_void,
            &mu as *const _ as *const c_void,
            &sigma as *const _ as *const c_void,
            &seed as *const _ as *const c_void,
        ];

        // Add operations to the graph
        graph.add_operation(
            OperationType::Kernel("init_light_source".to_string()),
            params_light_source_init.clone(),
            ray_grid_dim,
            ray_block_dim,
        );

        // sdf_sphere(int num_rays, float sphereX, float sphereY, float sphereZ, float radius, unsigned char *O, unsigned char *D, unsigned char *T, unsigned char *result)
        let params_sdf_sphere = vec![
            &num_rays as *const _ as *const c_void, 
            &sphere_x as *const _ as *const c_void, 
            &sphere_y as *const _ as *const c_void, 
            &sphere_z as *const _ as *const c_void, 
            &radius as *const _ as *const c_void, 
            &d_o as *const _ as *const c_void, 
            &d_d as *const _ as *const c_void, 
            &d_t as *const _ as *const c_void, 
            &d_sdf as *const _ as *const c_void,
        ];

        graph.add_operation(
            OperationType::Kernel("sdf_sphere".to_string()),
            params_sdf_sphere.clone(),
            ray_grid_dim,
            ray_block_dim,
        );

        // grad_sdf_sphere(int num_rays, float sphereX, float sphereY, float sphereZ, float radius, unsigned char *O, unsigned char *D, unsigned char *T, unsigned char *result)
        let params_grad_sdf_sphere = vec![
            &num_rays as *const _ as *const c_void, 
            &sphere_x as *const _ as *const c_void, 
            &sphere_y as *const _ as *const c_void, 
            &sphere_z as *const _ as *const c_void, 
            &radius as *const _ as *const c_void, 
            &d_o as *const _ as *const c_void, 
            &d_d as *const _ as *const c_void, 
            &d_t as *const _ as *const c_void, 
            &d_grad_sdf as *const _ as *const c_void,
        ];

        graph.add_operation(
            OperationType::Kernel("grad_sdf_sphere".to_string()),
            params_grad_sdf_sphere.clone(),
            ray_grid_dim,
            ray_block_dim,
        );

        // Execute the computation graph
        graph.execute()?;

        // Copy the result from GPU to CPU memory
        cuda_context.retrieve_tensor(d_image, &mut image, image_size)?;

        CudaContext::free_tensor(d_image)?;

        // Update texture with the new image
        texture.update(None, &image, (width * 3) as usize)?;

        canvas.copy(&texture, None, None)?;

        frame_count += 1;
        if last_frame.elapsed() >= Duration::from_secs(1) {
            fps = frame_count;
            frame_count = 0;
            last_frame = Instant::now();
        }

        // Render FPS text
        let fps_text = format!("FPS: {}", fps);
        let surface = font.render(&fps_text)
            .blended(sdl2::pixels::Color::RGBA(255, 255, 255, 200))
            .map_err(|e| e.to_string())?;
        let texture = texture_creator.create_texture_from_surface(&surface)
            .map_err(|e| e.to_string())?;

        let TextureQuery { width, height, .. } = texture.query();
        let target = Rect::new(128 - width as i32 - 10, 10, width, height);

        canvas.set_blend_mode(BlendMode::Blend);
        canvas.copy(&texture, None, Some(target))?;
        canvas.present();
    }

    Ok(())
}
