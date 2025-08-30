const ROT_DEG_PER_SEC: f32 = 60.0; // yaw/pitch speed
const MOVE_UNITS_PER_SEC: f32 = 15.0; // forward speed

pub fn move_entity(world: &mut World, e: Entity, dt: f32, yaw_dir: f32, pitch_dir: f32, cam_forward:bool) {
    if let Some(tr) = world.transforms.get_mut(e) {
        let yaw   = -yaw_dir   * ROT_DEG_PER_SEC.to_radians() * dt;
        let pitch = -pitch_dir * ROT_DEG_PER_SEC.to_radians() * dt;

        // local/body-frame: post-multiply by local deltas
        let q_local = Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, pitch);
        tr.rotation = tr.rotation * q_local;

        if cam_forward {
            let forward = tr.rotation * Vec3::Z * -1.0; // same convention as sync_camera()
            tr.position = tr.position + forward * (MOVE_UNITS_PER_SEC * dt);
        }
    }
}
