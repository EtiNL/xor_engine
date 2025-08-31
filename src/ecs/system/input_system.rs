use sdl2::event::Event;
use sdl2::keyboard::Keycode;

#[derive(Debug, Clone, Copy, Default)]
pub struct InputState {
    pub cam_yaw_dir: f32,
    pub cam_pitch_dir: f32,
    pub cam_forward: bool,
    pub mouse_down: bool,
    pub mouse_x: f32,
    pub mouse_y: f32,
}

/// Update input state from a single SDL2 event
pub fn handle_input_event(world: &mut World, event: &Event) {
    match *event {
        // yaw
        Event::KeyDown { keycode: Some(Keycode::Left),  repeat: false, .. } => { world.input.cam_yaw_dir = -1.0; }
        Event::KeyDown { keycode: Some(Keycode::Right), repeat: false, .. } => { world.input.cam_yaw_dir =  1.0; }
        Event::KeyUp   { keycode: Some(Keycode::Left),  .. } => { if world.input.cam_yaw_dir < 0.0 { world.input.cam_yaw_dir = 0.0; } }
        Event::KeyUp   { keycode: Some(Keycode::Right), .. } => { if world.input.cam_yaw_dir > 0.0 { world.input.cam_yaw_dir = 0.0; } }

        // pitch
        Event::KeyDown { keycode: Some(Keycode::Up),   repeat: false, .. } => { world.input.cam_pitch_dir = -1.0; }
        Event::KeyDown { keycode: Some(Keycode::Down), repeat: false, .. } => { world.input.cam_pitch_dir =  1.0; }
        Event::KeyUp   { keycode: Some(Keycode::Up),   .. } => { if world.input.cam_pitch_dir < 0.0 { world.input.cam_pitch_dir = 0.0; } }
        Event::KeyUp   { keycode: Some(Keycode::Down), .. } => { if world.input.cam_pitch_dir > 0.0 { world.input.cam_pitch_dir = 0.0; } }

        // forward
        Event::KeyDown { keycode: Some(Keycode::Space), repeat: false, .. } => { world.input.cam_forward = true; }
        Event::KeyUp   { keycode: Some(Keycode::Space), .. } => { world.input.cam_forward = false; }

        // mouse
        Event::MouseButtonDown { x, y, .. } => { world.input.mouse_down = true; world.input.mouse_x = x as f32; world.input.mouse_y = y as f32; }
        Event::MouseButtonUp { .. } => { world.input.mouse_down = false; }
        Event::MouseMotion { x, y, .. } => {
            if world.input.mouse_down {
                world.input.mouse_x = x as f32;
                world.input.mouse_y = y as f32;
            }
        }

        _ => {}
    }
}

/// Apply movement once per frame using the current input state.
pub fn apply_input(world: &mut World, dt: f32, cam_ent: Entity) {
    let i = world.input;
    if i.cam_yaw_dir != 0.0 || i.cam_pitch_dir != 0.0 || i.cam_forward {
        move_entity(world, cam_ent, dt, i.cam_yaw_dir, i.cam_pitch_dir, i.cam_forward);
    }
}
