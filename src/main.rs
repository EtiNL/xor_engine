mod cuda_wrapper;

use cuda_wrapper::{CudaContext, DeviceBuffer, KernelArg};
use sdl2::event::Event;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, TextureQuery};
use std::error::Error;
use std::thread;
use std::time::{Duration, Instant};

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

    // Initialize CUDA context
    let cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;

    // Initialize kernel arguments
    let sphere_x = DeviceBuffer::new(vec![0.0f32]);
    let sphere_y = DeviceBuffer::new(vec![10.0f32]);
    let sphere_z = DeviceBuffer::new(vec![0.0f32]);
    let radius = DeviceBuffer::new(vec![5.0f32]);
    let theta = DeviceBuffer::new(vec![0.0f32]);
    let phi = DeviceBuffer::new(vec![0.0f32]);
    let image = DeviceBuffer::new(vec![0u8; (width * height * 3) as usize]);

    let mut args: Vec<Box<dyn KernelArg>> = vec![
        Box::new(DeviceBuffer::new(vec![width as i32])),
        Box::new(DeviceBuffer::new(vec![height as i32])),
        Box::new(sphere_x),
        Box::new(sphere_y),
        Box::new(sphere_z),
        Box::new(radius),
        Box::new(theta),
        Box::new(phi),
        Box::new(image),
    ];

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

    'running: loop {
        for event in sdl_event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseButtonDown { x, y, .. } => {
                    mouse_down = true;
                    // Adjust angle based on mouse position
                    x_screen = x as f32;
                    y_screen = y as f32;
                },
                Event::MouseButtonUp { .. } => {
                    mouse_down = false;
                    theta_0 = theta_1;
                    phi_0 = phi_1;
                    theta_1 = 0.0f32;
                    phi_1 = 0.0f32;
                },
                Event::MouseMotion { x, y, .. } => {
                    if mouse_down {
                        let delta_x = x as f32 - x_screen;
                        let delta_y = y as f32 - y_screen;
                        theta_1 = (delta_y - height as f32 / 2.0).atan();
                        phi_1 = (delta_x - width as f32 / 2.0).atan();
                    }
                },
                _ => {}
            }
        }

        // Update the kernel arguments
        let theta_combined = theta_0 + theta_1;
        let phi_combined = phi_0 + phi_1;

        println!("theta: {}, phi: {}", theta_combined, phi_combined);

        args[5] = Box::new(DeviceBuffer::new(vec![theta_combined]));
        args[6] = Box::new(DeviceBuffer::new(vec![phi_combined]));

        // Launch the CUDA kernel
        match cuda_context.launch_kernel(&mut args, width, height) {
            Ok(_) => println!("CUDA kernel launched successfully."),
            Err(e) => eprintln!("Failed to launch CUDA kernel: {}", e),
        }

        // Update the texture with the new image data
        let image_arg = args.last().unwrap();
        if let Some(image_buffer) = image_arg.as_any().downcast_ref::<DeviceBuffer<u8>>() {
            texture.update(None, image_buffer.get_host_data(), (width * 3) as usize)?;
        }

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

    Ok(())
}
