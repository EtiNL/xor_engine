
pub mod math_op {
    use std::ops::Mul;

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct Vec3 {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }
    impl Vec3 {
        pub const X: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
        pub const Y: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
        pub const Z: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

        pub const fn new(x: f32, y: f32, z: f32) -> Self {
            Self { x, y, z }
        }
    }
    impl Default for Vec3 {
        fn default() -> Self {
            Vec3::new(0.0, 0.0, 0.0)
        }
    }

    // Produit Vec3 * float
    impl Mul<f32> for Vec3 {
        type Output = Vec3;
        fn mul(self, k: f32) -> Vec3 {
            Vec3::new(
                k*self.x,
                k*self.y,
                k*self.z,
            )
        }
    }

    // scène_composition.rs
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct Quat {
        pub w: f32,
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    impl Quat {
        pub const fn identity() -> Self {
            Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
        }

        pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
            let half = angle * 0.5;
            let s = half.sin();
            Self { w: half.cos(), x: axis.x * s, y: axis.y * s, z: axis.z * s }
        }
    }

    impl Default for Quat {
        fn default() -> Self {
            Quat::identity()
        }
    }

    impl Mul<Quat> for Quat {
        type Output = Quat;
        fn mul(self, rhs: Quat) -> Quat {
            Quat {
                w: self.w*rhs.w - self.x*rhs.x - self.y*rhs.y - self.z*rhs.z,
                x: self.w*rhs.x + self.x*rhs.w + self.y*rhs.z - self.z*rhs.y,
                y: self.w*rhs.y - self.x*rhs.z + self.y*rhs.w + self.z*rhs.x,
                z: self.w*rhs.z + self.x*rhs.y - self.y*rhs.x + self.z*rhs.w,
            }
        }
    }

    // Produit quat * vec3  (rotation)
    impl Mul<Vec3> for Quat {
        type Output = Vec3;
        fn mul(self, v: Vec3) -> Vec3 {
            // formule q * v * q^-1 (optimisée)
            let u = Vec3::new(self.x, self.y, self.z);
            let s = self.w;
            let uv   = cross(u, v);
            let uuv  = cross(u, uv);
            Vec3::new(
                v.x + 2.0 * (s*uv.x  + uuv.x),
                v.y + 2.0 * (s*uv.y  + uuv.y),
                v.z + 2.0 * (s*uv.z  + uuv.z),
            )
        }
    }
    fn cross(a: Vec3, b: Vec3) -> Vec3 {
        Vec3::new(
            a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x,
        )
    }
}