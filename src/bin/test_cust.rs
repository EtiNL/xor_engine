use cust::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Begin test cust !");

    cust::init(cust::CudaFlags::empty())?;
    println!("cust::init passed");

    let device = Device::get_device(0)?;
    let _ctx = Context::new(device)?;
    println!("CUDA context initialized on device: {}", device.name()?);

    let width = 800i32;
    let height = 600i32;
    let image_size = (width * height * 3) as usize;

    let ptx = std::fs::read_to_string("src/gpu_utils/kernel.ptx")?;
    let module = Module::from_ptx(ptx, &[])?;
    let function = module.get_function("generate_image")?;
    println!("Kernel 'generate_image' loaded successfully.");

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    // Alloue le buffer image sur le GPU
    let mut d_image = DeviceBuffer::<u8>::zeroed(image_size)?;

    // Lancement du kernel avec les bons paramètres
    unsafe {
        let result = launch!(
            function<<<(width * height / 256) as u32, 256, 0, stream>>>(
                width,
                height,
                d_image.as_device_ptr()
            )
        );
        match result {
            Ok(_) => println!("Kernel launched."),
            Err(e) => {
                println!("CUDA error during kernel launch: {:?}", e);
                return Err(Box::new(e));
            }
        }
    }

    stream.synchronize()?;
    println!("Kernel launch completed.");

    // Récupération des données
    let mut image = vec![0u8; image_size];
    d_image.copy_to(&mut image)?;
    println!("Image data retrieved from GPU.");

    Ok(())
}

