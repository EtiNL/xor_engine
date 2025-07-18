#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub aperture: f32,
    pub focus_dist: f32,
    pub viewport_width: f32,
    pub viewport_height: f32,

}

impl Camera {
    pub fn new(
        position: Vec3,
        u: Vec3,
        v: Vec3,
        w: Vec3,
        fov: f32,
        width: u32,
        height: u32,
        aperture: f32,
        focus_dist: f32,
    ) -> Self {
        let aspect_ratio = width as f32 / height as f32;

        // Champ de vision vertical → taille plan image
        let theta = fov.to_radians();
        let viewport_height = 2.0 * (theta / 2.0).tan();
        let viewport_width = aspect_ratio * viewport_height;

        Self {
            position,
            u,
            v,
            w,
            aperture,
            focus_dist,
            viewport_width,
            viewport_height,
        }
    }
}
