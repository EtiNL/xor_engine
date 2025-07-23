mod cuda_wrapper;

use sdl2::event::Event;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{BlendMode, TextureQuery};
use std::error::Error;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};

use cuda_wrapper::{CudaContext, dim3};


struct FpsCounter {
    last_frame_time: Instant,
    frame_count: u32,
    last_fps: u32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            last_frame_time: Instant::now(),
            frame_count: 0,
            last_fps: 0,
        }
    }

    fn update(&mut self) -> u32 {
        self.frame_count += 1;
        let now = Instant::now();

        if now.duration_since(self.last_frame_time) >= Duration::from_secs(1) {
            self.last_fps = self.frame_count;
            self.frame_count = 0;
            self.last_frame_time = now;
        }

        self.last_fps
    }
}

#[test]
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

    // Load the font and start the fps counter
    let font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";  // Ensure the font path is correct
    let font = ttf_context.load_font(font_path, 16)?;
    let mut fps_counter = FpsCounter::new();

    let mut sdl_event_pump = sdl_context.event_pump()?;

    // Initialize CUDA context and load kernels
    let mut cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;
    cuda_context.load_kernel("generate_image")?;
    cuda_context.load_kernel("draw_circle")?;

    // Create CUDA streams
    cuda_context.create_stream("stream1")?;
    cuda_context.create_stream("stream2")?;

    // Allocate GPU memory
    let mut image: Vec<u8> = vec![0u8; (width * height * 3) as usize];
    let image_size = image.len() * std::mem::size_of::<u8>();
    let d_image = CudaContext::allocate_tensor(&image, image_size)?;

    // Parameters to be used in the graph
    let mut x_click = 0f32;
    let mut y_click = 0f32;
    let mouse_down = Arc::new(Mutex::new(false)); // mouse_down wrapped in Arc<Mutex<_>> for thread-safe access

    let grid_dim = dim3 {
        x: ((width + 31) / 32) as u32,
        y: ((height + 31) / 32) as u32,
        z: 1,
    };

    let block_dim = dim3 { x: 32, y: 32, z: 1 };

    let params_generate_image = vec![
        &width as *const _ as *const c_void,
        &height as *const _ as *const c_void,
        &d_image as *const _ as *const c_void,
    ];

    let mut params_draw_circle = vec![
        &width as *const _ as *const c_void,
        &height as *const _ as *const c_void,
        &x_click as *const _ as *const c_void,
        &y_click as *const _ as *const c_void,
        &d_image as *const _ as *const c_void,
    ];

    // Main loop
    'running: loop {
        for event in sdl_event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseButtonDown { x, y, .. } => {
                    let mut mouse_down_lock = mouse_down.lock().unwrap();
                    *mouse_down_lock = true;
                    x_click = x as f32;
                    y_click = y as f32;
                },
                Event::MouseButtonUp { .. } => {
                    let mut mouse_down_lock = mouse_down.lock().unwrap();
                    *mouse_down_lock = false;
                },
                Event::MouseMotion { x, y, .. } => {
                    if *mouse_down.lock().unwrap() {
                        x_click = x as f32;
                        y_click = y as f32;
                    }
                },
                _ => {}
            }
        }

        // Update parameters for draw_circle kernel
        params_draw_circle[2] = &x_click as *const _ as *const c_void;
        params_draw_circle[3] = &y_click as *const _ as *const c_void;

        // Create a CUDA graph
        let mut graph = cuda_context.create_cuda_graph()?;
        
        cuda_context.add_graph_kernel_node(
            &mut graph,
            "generate_image",
            &params_generate_image,
            grid_dim,
            block_dim,
            "stream1",
        )?;

        cuda_context.add_graph_kernel_node(
            &mut graph,
            "draw_circle",
            &params_draw_circle,
            grid_dim,
            block_dim,
            "stream2",
        )?;

        // Instantiate the CUDA graph
        let graph_exec = cuda_context.instantiate_graph(graph)?;

        // Launch the graph
        cuda_context.launch_graph(graph_exec)?;

        // Copy the result from GPU to CPU memory
        cuda_context.retrieve_tensor(d_image, &mut image, image_size)?;

        // After graph execution is complete
        cuda_context.free_graph(graph)?;
        cuda_context.free_graph_exec(graph_exec)?;

        // Update texture with the new image
        texture.update(None, &image, (width * 3) as usize)?;
        canvas.copy(&texture, None, None)?;

        // Calculate and render FPS
        let fps = fps_counter.update();
        let fps_text = format!("FPS: {}", fps);
        render_fps(&fps_text, &mut canvas, &texture_creator, &font)?;

        canvas.present();
    }

    CudaContext::free_device_memory(d_image)?;

    Ok(())
}

fn render_fps(
    fps_text: &str,
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    texture_creator: &sdl2::render::TextureCreator<sdl2::video::WindowContext>,
    font: &sdl2::ttf::Font,
) -> Result<(), String> {
    let surface = font
        .render(fps_text)
        .blended(sdl2::pixels::Color::RGBA(255, 255, 255, 200))
        .map_err(|e| e.to_string())?;
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())?;

    let TextureQuery { width, height, .. } = texture.query();
    let target = Rect::new(128 - width as i32 - 10, 10, width, height);

    canvas.set_blend_mode(BlendMode::Blend);
    canvas.copy(&texture, None, Some(target))?;

    Ok(())
}
