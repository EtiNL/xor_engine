use image::ImageReader;
use std::error::Error;

pub fn load_texture(path: &str) -> Result<(Vec<u8>, u32, u32), Box<dyn Error>> {
    let img = ImageReader::open(path)?.decode()?.to_rgb8();
    let (width, height) = img.dimensions();
    let buffer = img.into_raw(); // Vec<u8>
    Ok((buffer, width, height))
}