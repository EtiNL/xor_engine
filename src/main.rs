// src/main.rs

mod cuda_wrapper;

use sdl2::event::Event;
use sdl2::pixels::PixelFormatEnum;
use std::error::Error;
use std::thread;
use std::time::Duration;

use cuda_wrapper::CudaContext;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the SDL2 context
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window("XOR Renderer", 256, 256)
        .position_centered()
        .build()
        .expect("Failed to create window");

    let mut canvas = window.into_canvas().present_vsync().build().expect("Failed to create canvas");
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator.create_texture_streaming(PixelFormatEnum::RGB24, 256, 256)?;

    let mut sdl_event_pump = sdl_context.event_pump()?;

    // Initialize CUDA context
    let cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;

    let width = 256;
    let height = 256;
    let mut mouse_x = width / 2;
    let mut mouse_y = height / 2;
    let mut image = vec![0u8; (width * height * 3) as usize];

    // Main loop
    'running: loop {
        for event in sdl_event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseMotion { x, y, .. } => {
                    mouse_x = x;
                    mouse_y = y;
                },
                _ => {}
            }
        }

        cuda_context.launch_kernel(width, height, mouse_x, mouse_y, &mut image);

        texture.update(None, &image, (width * 3) as usize)?;
        canvas.copy(&texture, None, None)?;
        canvas.present();

        thread::sleep(Duration::from_millis(16)); // ~60 FPS
    }

    Ok(())
}
