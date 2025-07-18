mod cuda_wrapper;
mod scene_composition;
mod texture_utils;
mod display;

use sdl2::event::Event;
use std::error::Error;
use std::ffi::{c_void, CString};
use std::sync::{Arc, Mutex};

use display::{Display, FpsCounter};
use cuda_wrapper::{CudaContext, dim3, check_cuda_result};
use scene_composition::{Camera, SdfObject, Vec3};
use cuda_driver_sys::*;
use texture_utils::load_texture;


fn main() -> Result<(), Box<dyn Error>> {
    // Initialize CUDA context
    let mut cuda_context = CudaContext::new("./src/gpu_utils/kernel.ptx")?;

    let sample_per_pixel: u32 = 16;
    let width: u32 = 800;
    let height: u32 = 600;

    // Camera and Scene initialization

    let cam = Camera::new(
        Vec3 { x: 0.0, y: 0.0, z: 0.0 },           // position
        Vec3 { x: 1.0, y: 0.0, z: 0.0 },           // u (right)
        Vec3 { x: 0.0, y: 1.0, z: 0.0 },           // v (up)
        Vec3 { x: 0.0, y: 0.0, z: -1.0 },          // w (forward)
        45.0,                                      // fov in degrees
        width,                                     // width
        height,                                    // height
        0.0,                                       // aperture
        1.0                                        // focus_dist
    );
    

    let (wood_texture, tex_w, tex_h) = load_texture("./src/textures/Wood_texture.png")?;
    let wood_texture_size = wood_texture.len() * std::mem::size_of::<u8>();
    let d_wood_texture = CudaContext::allocate_tensor(&wood_texture, wood_texture_size)?;

    let scene = vec![
        SdfObject {
            sdf_type: 0, // sphere
            params: [1.0, 0.0, 0.0], // radius 1.0
            center: Vec3 { x: 0.0, y: 0.0, z: -5.0 },     // center
            u: Vec3 { x: 1.0, y: 0.0, z: 0.0 },           // u (right)
            v: Vec3 { x: 0.0, y: 1.0, z: 0.0 },           // v (up)
            w: Vec3 { x: 0.0, y: 0.0, z: -1.0 },          // w (forward)
            texture: d_wood_texture as *mut u8, 
            tex_width: tex_w as i32,
            tex_height: tex_h as i32,
        }
    ];


    // Initialize the SDL2 context
    let mut display = Display::new(
        "xor Renderer",
        width,
        height,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )?;
    let mut fps_counter = FpsCounter::new();

    // load kernels
    cuda_context.load_kernel("init_random_states")?;
    cuda_context.load_kernel("generate_rays")?;
    cuda_context.load_kernel("raymarch")?;

    // Create CUDA streams
    cuda_context.create_stream("stream1")?;
    cuda_context.create_stream("stream2")?;

    // Allocate GPU memory
    let d_camera = CudaContext::allocate_struct(&cam)?;
    let d_rand_states = CudaContext::allocate_curand_states(width, height)?;
    let d_scene = CudaContext::allocate_scene(&scene)?;

    let mut directions: Vec<f32> = vec![0f32; (width * height * 3) as usize];
    let directions_size = directions.len() * std::mem::size_of::<f32>();
    let d_directions = CudaContext::allocate_tensor(&directions, directions_size)?;

    let mut origins: Vec<f32> = vec![0f32; (width * height * 3) as usize];
    let origins_size = origins.len() * std::mem::size_of::<f32>();
    let d_origins = CudaContext::allocate_tensor(&origins, origins_size)?;

    let mut image: Vec<u8> = vec![0u8; (width * height * 3) as usize];
    let image_size = image.len() * std::mem::size_of::<u8>();
    let d_image = CudaContext::allocate_tensor(&image, image_size)?;

    // grid dim and block dim
    let grid_dim = dim3 {
        x: ((width + 31) / 32) as u32,
        y: ((height + 31) / 32) as u32,
        z: 1,
    };
    
    let block_dim = dim3 { x: 32, y: 32, z: 1 };

    // Initialize Cuda random state
    let seed = 1530u32;
    let init_rng_params: Vec<*const c_void> = vec![
        &d_rand_states as *const _ as *const c_void,
        &width as *const _ as *const c_void,
        &height as *const _ as *const c_void,
        &seed as *const _ as *const c_void,
    ];

    cuda_context.launch_kernel("init_random_states", 
        grid_dim, 
        block_dim, 
        &init_rng_params, 
        "stream1")?;

    // Parameters to be used in the graph
    let mut x_click = 0f32;
    let mut y_click = 0f32;
    let mouse_down = Arc::new(Mutex::new(false)); // mouse_down wrapped in Arc<Mutex<_>> for thread-safe access

    // Main loop
    'running: loop {
        for event in display.poll_events() {
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

        // Create a CUDA graph
        let mut graph = cuda_context.create_cuda_graph()?;

        let params_generate_rays: Vec<*const c_void> = vec![
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &d_camera as *const _ as *const c_void,
            &d_rand_states as *const _ as *const c_void,
            &d_origins as *const _ as *const c_void,
            &d_directions as *const _ as *const c_void,
        ];
        
        cuda_context.add_graph_kernel_node(
            &mut graph,
            "generate_rays",
            &params_generate_rays,
            grid_dim,
            block_dim,
            "stream1",
        )?;

        let scene_len = scene.len();
        let params_raymarch = vec![
            &width as *const _ as *const c_void,
            &height as *const _ as *const c_void,
            &d_origins as *const _ as *const c_void,
            &d_directions as *const _ as *const c_void,
            &d_scene as *const _ as *const c_void,
            &scene_len as *const _ as *const c_void,
            &d_image as *const _ as *const c_void,
        ];

        cuda_context.add_graph_kernel_node(
            &mut graph,
            "raymarch",
            &params_raymarch,
            grid_dim,
            block_dim,
            "stream1",
        )?;

        for i in 0..2 {
            
        }

        // Instantiate the CUDA graph
        let graph_exec = cuda_context.instantiate_graph(graph)?;

        // Launch the graph
        cuda_context.launch_graph(graph_exec)?;

        // Copy the result from GPU to CPU memory
        cuda_context.retrieve_tensor(d_image, &mut image, image_size)?;

        // After graph execution is complete
        cuda_context.free_graph(graph)?;
        cuda_context.free_graph_exec(graph_exec)?;

        // display Image and fps
        let fps = fps_counter.update();
        display.render_texture(&image, (width * 3) as usize)?;
        display.render_fps(fps)?;
        display.present();
    }

    CudaContext::free_device_memory(d_image)?;

    Ok(())
}
