mod cuda_wrapper;
mod dynamic_computation_graph;

use sdl2::event::Event;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, TextureQuery};
use std::error::Error;
use std::ffi::c_void;
use std::time::{Instant, Duration};

use cuda_wrapper::{CudaContext, dim3};
use dynamic_computation_graph::{ComputationGraph, OperationType}; // Remove Operation if not used

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

    // Initialize CUDA context and load kernels
    let mut cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;
    cuda_context.load_kernel("generate_image")?;

    // Create CUDA streams
    cuda_context.create_stream("stream1")?;
    cuda_context.create_stream("stream2")?;

    // Allocate GPU memory
    let mut image: Vec<u8> = vec![0u8; (width * height * 3) as usize];
    let d_image = CudaContext::allocate_tensor(&image, (width * height * 3) as usize)?;

    let font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
    let font = ttf_context.load_font(font_path, 16)
        .map_err(|e| {
            eprintln!("Failed to load font: {}", e);
            e.to_string()
        })?;

    // Create and configure computation graph
    let mut graph = ComputationGraph::new(cuda_context);

    let sphere_x = 0.0f32;
    let sphere_y = 10.0f32;
    let sphere_z = 0.0f32;
    let radius = 5.0f32;

    let mut x_click = 0f32;
    let mut y_click = 0f32;
    let mut theta_0 = std::f32::consts::PI / 2.0;
    let mut phi_0 = std::f32::consts::PI / 2.0;
    let mut theta_1 = 0f32;
    let mut phi_1 = 0f32;
    let mut mouse_down = false;

    let grid_dim = dim3 {
        x: ((width + 15) / 16) as u32,
        y: ((height + 15) / 16) as u32,
        z: 1,
    };
    let block_dim = dim3 { x: 16, y: 16, z: 1 };

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
                        println!("phi: {}, theta: {}", phi_1, theta_1)
                    }
                },
                _ => {}
            }

            // Parameters for the CUDA kernels
            // let params_sdf = vec![
            //     &width as *const _ as *const c_void,
            //     &height as *const _ as *const c_void,
            //     &sphere_x as *const _ as *const c_void,
            //     &sphere_y as *const _ as *const c_void,
            //     &sphere_z as *const _ as *const c_void,
            //     &radius as *const _ as *const c_void,
            //     &theta_1 as *const _ as *const c_void,
            //     &phi_1 as *const _ as *const c_void,
            //     &d_image as *const _ as *const c_void,
            // ];

            let params2 = vec![
                &width as *const _ as *const f32, // Change c_void to f32
                &height as *const _ as *const f32, // Change c_void to f32
                &x_click as *const _ as *const f32, // mouse_x
                &y_click as *const _ as *const f32, // mouse_y
                &d_image as *const _ as *const f32, // Change c_void to f32
            ];

            // Reset the graph for each frame
            graph.clear_operations();

            // Add operations to the graph
            // graph.add_operation(OperationType::Kernel("computeDepthMap".to_string()), params_sdf.clone(), None);
            graph.add_operation(OperationType::Kernel("generate_image".to_string()), params2.clone(), None);

            // Execute the computation graph
            graph.execute(grid_dim, block_dim)?;

            // Copy the result from GPU to CPU memory
            cuda_context.retrieve_tensor(d_image, &mut image, (width * height * 3) as usize)?;

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
                .blended(sdl2::pixels::Color::RGBA(255, 0, 255, 255))
                .map_err(|e| {
                    eprintln!("Failed to render text surface: {}", e);
                    e.to_string()
                })?;
            let texture = texture_creator.create_texture_from_surface(&surface)
                .map_err(|e| {
                    eprintln!("Failed to create texture from surface: {}", e);
                    e.to_string()
                })?;

            let TextureQuery { width, height, .. } = texture.query();
            let target = Rect::new(128 - width as i32 - 10, 10, width, height);

            canvas.set_blend_mode(BlendMode::Blend);
            canvas.copy(&texture, None, Some(target))?;
            canvas.present();

            //thread::sleep(Duration::from_millis(16)); // ~60 FPS
        }
    }

    CudaContext::free_tensor(d_image)?;

    Ok(())
}
