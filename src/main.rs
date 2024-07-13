mod cuda_wrapper;

use sdl2::event::Event;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, TextureQuery};
use std::error::Error;
use std::thread;
use std::time::{Duration, Instant};

use cuda_wrapper::CudaContext;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the SDL2 context
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string())?;
    
    let window = video_subsystem
        .window("Depth Map Renderer", 128, 128)
        .position_centered()
        .build()
        .expect("Failed to create window");

    let mut canvas = window.into_canvas().present_vsync().build().expect("Failed to create canvas");
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator.create_texture_streaming(PixelFormatEnum::RGB24, 128, 128)?;

    let mut sdl_event_pump = sdl_context.event_pump()?;

    // Initialize CUDA context
    let cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;

    let width = 128;
    let height = 128;
    let sphere_x = 0.0f32;
    let sphere_y = 10.0f32;
    let sphere_z = 0.0f32;
    let radius = 5.0f32;
    let mut image = vec![0u8; (width * height * 3) as usize];

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

    'running: loop {
        for event in sdl_event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                _ => {}
            }
        }

        cuda_context.launch_kernel(width, height, sphere_x, sphere_y, sphere_z, radius, &mut image);

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

    Ok(())
}
