use cust::prelude::*;
use image::{ImageBuffer, Rgb};
use std::error::Error;
use std::ffi::CString;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
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
    if !Path::new(ptx_path).exists() {
        return Err(Box::from(format!("PTX file not found: {}", ptx_path)));
    }

    let ptx_content = fs::read_to_string(ptx_path)?;
    let ptx_cstr = CString::new(ptx_content)?;

    // Load the PTX and create the module
    let module = Module::from_ptx_cstr(&ptx_cstr, &[])?;
    let function = module.get_function("generate_image")?;

    // Create a CUDA stream
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Define grid and block sizes
    let block_size = (16, 16, 1);
    let grid_size = ((width + block_size.0 - 1) / block_size.0, (height + block_size.1 - 1) / block_size.1, 1);

    // Launch the kernel
    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            width as i32,
            height as i32,
            d_image.as_device_ptr()
        ))?;
    }

    // Wait for the kernel to finish
    stream.synchronize()?;

    // Copy the result back to host
    d_image.copy_to(&mut image)?;

    // Create an image buffer and save the image
    let img_buffer = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        let idx = ((y * width as u32 + x) * 3) as usize;
        Rgb([image[idx], image[idx + 1], image[idx + 2]])
    });

    let output_path = Path::new("output.png");
    img_buffer.save(output_path)?;
    println!("Image saved as {}", output_path.display());

    // Open the image using an external viewer
    if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(&["/C", "start", output_path.to_str().unwrap()])
            .spawn()?;
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open")
            .arg(output_path)
            .spawn()?;
    } else if cfg!(target_os = "linux") {
        println!("target_os = linux");
        std::process::Command::new("xdg-open")
            .arg(output_path)
            .spawn()?;
    }

    Ok(())
}
