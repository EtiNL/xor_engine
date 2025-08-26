
pub mod math_op {
    
    use std::ops::{Add, Sub, AddAssign, SubAssign, Mul};

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

    impl Vec3 {
        pub const X: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
        pub const Y: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
        pub const Z: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

        pub const fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

        #[inline] pub fn dot(self, b: Vec3) -> f32 { self.x*b.x + self.y*b.y + self.z*b.z }
        #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
        #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }
    }
    impl Default for Vec3 { fn default() -> Self { Vec3::new(0.0, 0.0, 0.0) } }

    impl Add for Vec3 {
        type Output = Vec3;
        fn add(self, rhs: Vec3) -> Vec3 { Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) }
    }
    impl AddAssign for Vec3 {
        fn add_assign(&mut self, rhs: Vec3) { self.x += rhs.x; self.y += rhs.y; self.z += rhs.z; }
    }
    impl Sub for Vec3 {
        type Output = Vec3;
        fn sub(self, rhs: Vec3) -> Vec3 { Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z) }
    }
    impl SubAssign for Vec3 {
        fn sub_assign(&mut self, rhs: Vec3) { self.x -= rhs.x; self.y -= rhs.y; self.z -= rhs.z; }
    }

    impl Mul<f32> for Vec3 {
        type Output = Vec3;
        fn mul(self, k: f32) -> Vec3 { Vec3::new(k*self.x, k*self.y, k*self.z) }
    }

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

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct Mat3 {
        pub a11: f32,
        pub a12: f32,
        pub a13: f32,
        pub a21: f32,
        pub a22: f32,
        pub a23: f32,
        pub a31: f32,
        pub a32: f32,
        pub a33: f32,
    }
    impl Mat3 {
        pub const Id: Self = Self { 
            a11: 1.0,
            a12: 0.0,
            a13: 0.0,
            a21: 0.0,
            a22: 1.0,
            a23: 0.0,
            a31: 0.0,
            a32: 0.0,
            a33: 1.0, };
        pub const Zero: Self = Self { 
            a11: 0.0,
            a12: 0.0,
            a13: 0.0,
            a21: 0.0,
            a22: 0.0,
            a23: 0.0,
            a31: 0.0,
            a32: 0.0,
            a33: 0.0, };

        pub const fn det(&self) -> f32 {
            self.a11 * (self.a22*self.a33 - self.a23*self.a32) 
            - self.a12 * (self.a21*self.a33 - self.a31*self.a23)
            + self.a13 * (self.a21*self.a32 - self.a31*self.a22)
        }

        pub const fn inv(&self) -> Mat3 {
            let det: f32 = self.det();
            Self { a11: (self.a22*self.a33 - self.a23*self.a32)/det, 
                a12: (self.a13*self.a32 - self.a12*self.a33)/det,
                a13: (self.a12*self.a23 - self.a22*self.a13)/det, 
                a21: (self.a31*self.a23 - self.a21*self.a33)/det,
                a22: (self.a11*self.a33 - self.a31*self.a13)/det,
                a23: (self.a21*self.a13 - self.a12*self.a23)/det,
                a31: (self.a21*self.a32 - self.a31*self.a22)/det,
                a32: (self.a12*self.a31 - self.a11*self.a32)/det,
                a33: (self.a11*self.a22 - self.a12*self.a21)/det }
        } 


    }

    impl Mul<f32> for Mat3 {
        type Output = Mat3;
        fn mul(self, k: f32) -> Mat3 {
            Mat3 { a11: self.a11 * k, 
                a12: self.a12 * k,
                a13: self.a13 * k, 
                a21: self.a21 * k,
                a22: self.a22 * k,
                a23: self.a23 * k,
                a31: self.a31 * k,
                a32: self.a32 * k,
                a33: self.a33 * k }
        }
    }

    impl Default for Mat3 {
        fn default() -> Self {
            Mat3::Zero
        }
    }
}