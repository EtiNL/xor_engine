
pub fn update_rotation(world: &mut World, dt: f32, input_dir: Vec3) {
    for (rot, idx) in world.rotatings.iter_mut() {
        let gen = world.gens[idx as usize];
        let e = Entity { index: idx, generation: gen };

        if let Some(tr) = world.transforms.get_mut(e) {

            let angle_y= rot.speed_deg_per_sec.to_radians() * dt * input_dir.y;
            let q_y = Quat::from_axis_angle(Vec3::Y, angle_y);

            let angle_x = rot.speed_deg_per_sec.to_radians() * dt * input_dir.x;
            let q_x = Quat::from_axis_angle(Vec3::X, angle_x);
            tr.rotation = (q_y * q_x) * tr.rotation;
        }
    }
}