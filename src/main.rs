use cust::prelude::*;
use image::{ImageBuffer, Rgb};
use sdl2::event::Event;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::TextureCreator;
use sdl2::video::WindowContext;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the SDL2 context
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window("CUDA Image", 256, 256)
        .position_centered()
        .opengl()
        .build()
        .expect("Failed to create window");

    let mut canvas = window
        .into_canvas()
        .present_vsync()
        .build()
        .expect("Failed to create canvas");

    let texture_creator: TextureCreator<WindowContext> = canvas.texture_creator();

    // Initialize the CUDA driver
    cust::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;
    println!("Using device: {}", device.name()?);

    // Create a CUDA context
    let _context = Context::new(device)?;

    // Image dimensions
    let width = 256;
    let height = 256;
    let image_size = (width * height * 3) as usize;

    // Allocate memory for the image on the host
    let mut image = vec![0u8; image_size];

    // Allocate memory on the device
    let d_image = DeviceBuffer::from_slice(&image)?;

    // Load the PTX from file
    let ptx_path = "src/gpu_utils/kernel.ptx";
    if !std::path::Path::new(ptx_path).exists() {
        return Err(Box::from(format!("PTX file not found: {}", ptx_path)));
    }

    let ptx_content = std::fs::read_to_string(ptx_path)?;
    let ptx_cstr = std::ffi::CString::new(ptx_content)?;

    // Load the PTX and create the module
    let module = Module::from_ptx_cstr(&ptx_cstr, &[])?;
    let function = module.get_function("generate_image")?;

    // Create a CUDA stream
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Define grid and block sizes
    let block_size = (16, 16, 1);
    let grid_size = ((width + block_size.0 - 1) / block_size.0, (height + block_size.1 - 1) / block_size.1, 1);

    // Shared variables for mouse coordinates
    let mouse_coords = Arc::new(Mutex::new((0, 0)));

    let mut sdl_event_pump = sdl_context.event_pump()?;
    // Main loop to update the image
    'running: loop {
        for event in sdl_event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseMotion { x, y, .. } => {
                    let mut coords = mouse_coords.lock().unwrap();
                    *coords = (x, y);
                }
                _ => {}
            }
        }

        // Get the current mouse coordinates
        let (mouse_x, mouse_y) = {
            let coords = mouse_coords.lock().unwrap();
            *coords
        };

        // println!("Mouse coordinates: ({}, {})", mouse_x, mouse_y);

        // Launch the kernel
        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                width as i32,
                height as i32,
                mouse_x,
                mouse_y,
                d_image.as_device_ptr()
            ))?;
        }

        // Wait for the kernel to finish
        stream.synchronize()?;

        // Copy the result back to host
        d_image.copy_to(&mut image)?;

        // Create an image buffer
        let img_buffer = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
            let idx = ((y * width as u32 + x) * 3) as usize;
            Rgb([image[idx], image[idx + 1], image[idx + 2]])
        });

        // Convert the ImageBuffer to a format SDL2 can use
        let raw_image = img_buffer.into_raw();
        let mut texture = texture_creator
            .create_texture_streaming(PixelFormatEnum::RGB24, width as u32, height as u32)
            .map_err(|e| e.to_string())?;

        texture.update(None, &raw_image, (width * 3) as usize)?;

        canvas.clear();
        canvas.copy(&texture, None, Some(Rect::new(0, 0, width as u32, height as u32)))?;
        canvas.present();

        thread::sleep(Duration::from_millis(100)); // Adjust as needed
    }

    Ok(())
}
