mod cuda_wrapper;

use sdl2::event::Event;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, TextureQuery};
use std::error::Error;
use std::ffi::c_void;
use std::time::{Instant, Duration};

use cuda_driver_sys::{cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, CUdeviceptr};
use cuda_wrapper::{CudaContext, dim3, check_cuda_result};


fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the SDL2 context
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string())?;

    // Customize screen size
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

    // Initialize CUDA context
    let cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx", "computeDepthMap")?;

    let sphere_x = 0.0f32;
    let sphere_y = 10.0f32;
    let sphere_z = 0.0f32;
    let radius = 5.0f32;
    let mut image = vec![0u8; (width * height * 3) as usize];

    // Allocate device memory for the image
    let mut d_image: CUdeviceptr = 0;
    unsafe {
        check_cuda_result(cuMemAlloc_v2(&mut d_image, (width * height * 3) as usize), "cuMemAlloc_v2")?;
        check_cuda_result(cuMemcpyHtoD_v2(d_image, image.as_ptr() as *const _, (width * height * 3) as usize), "cuMemcpyHtoD_v2")?;
    }

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
    let mut x_click = 0f32;
    let mut y_click = 0f32;
    let mut theta_0 = 0f32;
    let mut phi_0 = 0f32;
    let mut theta_1 = 0f32;
    let mut phi_1 = 0f32;
    let mut mouse_down = false; // Track if mouse button is pressed

    'running: loop {
        for event in sdl_event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseButtonDown { x, y, .. } => {
                    mouse_down = true;
                    x_click = x as f32;
                    y_click = y as f32;
                },
                Event::MouseButtonUp { .. } => {
                    mouse_down = false;
                    theta_0 = theta_1;
                    phi_0 = phi_1;
                    theta_1 = 0f32;
                    phi_1 = 0f32;
                },
                Event::MouseMotion { x, y, .. } => {
                    if mouse_down {
                        phi_1 = phi_0 + (x_click - x as f32).atan();
                        theta_1 = theta_0 + (y_click - y as f32).atan();
                    }
                },
                _ => {}
            }
        }

        let params = vec![
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &sphere_x as *const _ as *const c_void,
            &sphere_y as *const _ as *const c_void,
            &sphere_z as *const _ as *const c_void,
            &radius as *const _ as *const c_void,
            &theta_1 as *const _ as *const c_void,
            &phi_1 as *const _ as *const c_void,
            &d_image as *const _ as *const c_void,
        ];

        let grid_dim = dim3 {
            x: ((width + 15) / 16) as u32,
            y: ((height + 15) / 16) as u32,
            z: 1,
        };
        let block_dim = dim3 { x: 16, y: 16, z: 1 };

        cuda_context.launch_kernel(grid_dim, block_dim, params)?;

        unsafe {
            check_cuda_result(cuMemcpyDtoH_v2(image.as_mut_ptr() as *mut _, d_image, (width * height * 3) as usize), "cuMemcpyDtoH_v2")?;
        }

        texture.update(None, &image, (width * 3) as usize)?;
        canvas.copy(&texture, None, None)?;

        // Calculate FPS
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

    // Free device memory
    unsafe {
        check_cuda_result(cuMemFree_v2(d_image), "cuMemFree_v2")?;
    }

    Ok(())
}
