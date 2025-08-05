#include <math.h>
#include <curand_kernel.h>
#include <stdio.h>

struct Vec2 {
    float x, y;

    __device__ Vec2() : x(0), y(0) {}
    __device__ Vec2(float x, float y) : x(x), y(y) {}

    __device__ Vec2 operator+(const Vec2& b) const { return Vec2(x + b.x, y + b.y); }
    __device__ Vec2 operator-(const Vec2& b) const { return Vec2(x - b.x, y - b.y); }
    __device__ Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
    __device__ Vec2 operator/(float s) const { return Vec2(x / s, y / s); }

};
struct Mat3 {
    float a11, a12, a13, a21, a22, a23, a31, a32, a33;

    __device__ Mat3() : a11(0), a12(0), a13(0), a21(0), a22(0), a23(0), a31(0), a32(0), a33(0) {}
    __device__ Mat3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) : a11(a11), a12(a12), a13(a13), a21(a21), a22(a22), a23(a23), a31(a31), a32(a32), a33(a33) {}

    __device__ bool is_null() const { return a11 == 0 && a12 == 0 && a13 == 0 && a21 == 0 && a22 == 0 && a23 == 0 && a31 == 0 && a32 == 0 && a33 == 0; }

};

// Vec3 struct compatible with CUDA
struct Vec3 {
    float x, y, z;

    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(float s) : x(s), y(s), z(s) {}
    __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __device__ Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    __device__ Vec3 operator*(const Vec3& b) const { return Vec3(x * b.x, y * b.y, z * b.z); }
    __device__ Vec3 operator*(const Mat3& a) const { return Vec3(x * a.a11 + y * a.a12+ z * a.a13, x * a.a21 + y * a.a22+ z * a.a23, x * a.a31 + y * a.a32+ z * a.a33); }

    __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
    __device__ Vec3 normalize() const {
        float len = length();
        return (len > 0) ? (*this) / len : Vec3(0, 0, 0);
    }
    __device__ Vec3 floor() const {
        return Vec3(floorf(x), floorf(y), floorf(z));
    }


    __device__ static Vec3 cross(const Vec3& a, const Vec3& b) {
        return Vec3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    __device__ static float dot(const Vec3& a, const Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __device__ static float distance(const Vec3& a, const Vec3& b) {
        return (a-b).length();
    }

    __device__ Vec3 abs() const {
    return Vec3(fabsf(x), fabsf(y), fabsf(z));
    }

    __device__ static Vec3 max(const Vec3& a, const Vec3& b) {
        return Vec3(
            fmaxf(a.x, b.x),
            fmaxf(a.y, b.y),
            fmaxf(a.z, b.z)
        );
    }
};

struct Camera {
    Vec3 position;
    Vec3 u, v, w;       // camera basis
    float aperture;
    float focus_dist;
    float viewport_width;
    float viewport_height;
};

enum SdfType {
    SDF_SPHERE = 0,
    SDF_BOX    = 1,
    SDF_PLANE  = 2,
    // Extend as needed
};

#define INVALID_TEXTURE 0xFFFFFFFFu
#define INVALID_LIGHT   0xFFFFFFFFu
#define INVALID_FOLDING 0xFFFFFFFFu

// GPU mirror of Rust GpuSdfObjectBase
struct GpuSdfObjectBase {
    int    sdf_type;
    float  params[3];
    Vec3   center;
    Vec3   u, v, w;
    unsigned int material_id;
    unsigned int light_id;
    unsigned int lattice_folding_id;
    unsigned int active;
};

// GPU mirror of Rust GpuMaterial
struct GpuMaterial {
    float        color[3];
    unsigned int use_texture;
    const unsigned char* texture_data;
    unsigned int width;
    unsigned int height;
};

// GPU mirror of Rust GpuLight
struct GpuLight {
    Vec3   position;
    Vec3   color;
    float  intensity;
};

// GPU mirror of Rust GpuSpaceFolding
struct GpuSpaceFolding {
    Mat3   lattice_basis;
    Mat3   lattice_basis_inv;
};


struct Image_ray_accum {
    int* ray_per_pixel;
    unsigned char* image;
};

__device__ void accumulate(Image_ray_accum* acc,
                           const Vec3& color,
                           int pixel_idx)
{
    int rpp      = acc->ray_per_pixel[pixel_idx]; 
    int new_rpp  = rpp + 1;
    int base     = 3 * pixel_idx;

    acc->image[base + 0] =
        (acc->image[base + 0] * rpp + color.x * 255.0f) / new_rpp;
    acc->image[base + 1] =
        (acc->image[base + 1] * rpp + color.y * 255.0f) / new_rpp;
    acc->image[base + 2] =
        (acc->image[base + 2] * rpp + color.z * 255.0f) / new_rpp;

    acc->ray_per_pixel[pixel_idx] = new_rpp;
}

extern "C"
__global__ void reset_accum(int* ray_per_pixel, int total_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_pixels) {
        ray_per_pixel[i] = 0;
    }
}

extern "C"
__global__ void init_random_states(curandState *rand_states, int width, int height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    if (x < width && y < height) {
        curand_init(seed, i, 0, &rand_states[i]);
    }
}

__device__ Vec3 random_in_unit_disk(curandState* rand_state) {
    while (true) {
        float x = curand_uniform(rand_state) * 2.0f - 1.0f;
        float y = curand_uniform(rand_state) * 2.0f - 1.0f;
        if (x*x + y*y >= 1.0f) continue;
        return Vec3(x, y, 0);
    }
}

extern "C" __global__
void generate_rays(int width, int height, Camera* cam_ptr, curandState *rand_states, float* origins, float* directions) {
    Camera cam = *cam_ptr;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    if (x >= width || y >= height) return;

    // For floating-point calculations
    float u = (x + 0.5f) / width * 2.0f - 1.0f;
    float v = 1.0f - ((y + 0.5f) / height) * 2.0f;

    Vec3 dir_camera = Vec3(
        u * cam.viewport_width * 0.5f,
        v * cam.viewport_height * 0.5f,
        1.0f
    ).normalize();

    Vec3 dir_world = cam.u * dir_camera.x + cam.v * dir_camera.y + cam.w * dir_camera.z;
    Vec3 origin, direction;

    curandState local_rand = rand_states[i];

    if (cam.aperture > 0.0f) {
        float lens_radius = cam.aperture * 0.5f;
        Vec3 rd = random_in_unit_disk(&local_rand) * lens_radius;
        Vec3 offset = cam.u * rd.x + cam.v * rd.y;
        origin = cam.position + offset;
        Vec3 focal_point = cam.position + dir_world * cam.focus_dist;
        direction = (focal_point - origin).normalize();

        rand_states[i] = local_rand;

    } else {
        origin = cam.position;
        direction = dir_world.normalize();
    }

    origins[i * 3 + 0] = origin.x;
    origins[i * 3 + 1] = origin.y;
    origins[i * 3 + 2] = origin.z;

    directions[i * 3 + 0] = direction.x;
    directions[i * 3 + 1] = direction.y;
    directions[i * 3 + 2] = direction.z;
}

__device__ float sdf_sphere(Vec3 sphere_center, float radius, Vec3 ray_point) {
    return Vec3::distance(sphere_center, ray_point) - radius;
}

__device__ Vec3 grad_sdf_sphere(Vec3 sphere_center, float radius, Vec3 ray_point) {
    return (ray_point - sphere_center)/radius;
}

__device__ float sdf_box(Vec3 p, Vec3 center, Vec3 u, Vec3 v, Vec3 w, Vec3 half_extents) {
    Vec3 d = p - center;
    Vec3 local = Vec3(
        Vec3::dot(d, u),
        Vec3::dot(d, v),
        Vec3::dot(d, w)
    );

    Vec3 q = local.abs() - half_extents;
    return fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f) + Vec3::max(q, Vec3(0,0,0)).length();
}

__device__ Vec3 grad_sdf_box(Vec3 p, Vec3 center, Vec3 u, Vec3 v, Vec3 w, Vec3 half_extents) {
    Vec3 d = p - center;
    Vec3 local = Vec3(
        Vec3::dot(d, u),
        Vec3::dot(d, v),
        Vec3::dot(d, w)
    );
    Vec3 q = local.abs() - half_extents;

    Vec3 g_local;
    if (q.x > q.y && q.x > q.z)
        g_local = Vec3((local.x > 0.0f) ? 1.0f : -1.0f, 0, 0);
    else if (q.y > q.z)
        g_local = Vec3(0, (local.y > 0.0f) ? 1.0f : -1.0f, 0);
    else
        g_local = Vec3(0, 0, (local.z > 0.0f) ? 1.0f : -1.0f);

    // Remonter en coord. monde
    return u * g_local.x + v * g_local.y + w * g_local.z;
}

__device__ float evaluate_sdf(const GpuSdfObjectBase& obj, Vec3 p) {
    if (obj.sdf_type == SDF_SPHERE) {
        float radius = obj.params[0];
        return sdf_sphere(obj.center, radius, p);
    } else if (obj.sdf_type == SDF_BOX) {
        Vec3 half_extents = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        return sdf_box(p, obj.center, obj.u, obj.v, obj.w, half_extents);
    }
    return 1e9;
}

__device__ Vec3 evaluate_grad_sdf(const GpuSdfObjectBase& obj, Vec3 p) {
    if (obj.sdf_type == SDF_SPHERE) {
        float radius = obj.params[0];
        return grad_sdf_sphere(obj.center, radius, p);
    } else if (obj.sdf_type == SDF_BOX) {
        Vec3 half_extents = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        return grad_sdf_box(p, obj.center, obj.u, obj.v, obj.w, half_extents);
    }
}

// sample from the raw 3-channel byte buffer in the material
__device__ Vec3 sample_texture(
    const GpuMaterial& mat,
    Vec2 uv
) {
    // clamp uv into [0,1]
    float u = fminf(fmaxf(uv.x, 0.0f), 1.0f);
    float v = fminf(fmaxf(uv.y, 0.0f), 1.0f);

    // convert to texel coords
    int tx = min((int)(u * (mat.width  - 1)), (int)mat.width  - 1);
    int ty = min((int)(v * (mat.height - 1)), (int)mat.height - 1);

    // each pixel is 3 bytes
    int idx = (ty * mat.width + tx) * 3;
    unsigned char r = mat.texture_data[idx + 0];
    unsigned char g = mat.texture_data[idx + 1];
    unsigned char b = mat.texture_data[idx + 2];

    // convert [0,255] → [0,1]
    const float inv255 = 1.0f / 255.0f;
    return Vec3(r * inv255, g * inv255, b * inv255);
}

// triplanar blending based on face normals
__device__ Vec3 triplanar_sample(
    const GpuMaterial&   mat,
    const GpuSdfObjectBase& obj,
    Vec3                 p,
    Vec3                 normal
) {
    // local space coords
    Vec3 d = p - obj.center;
    Vec3 local = Vec3(
        Vec3::dot(d, obj.u),
        Vec3::dot(d, obj.v),
        Vec3::dot(d, obj.w)
    );
    Vec3 he = Vec3(obj.params[0], obj.params[1], obj.params[2]);

    // blend weights = abs(normal) normalized
    Vec3 w = Vec3(fabsf(normal.x),
                  fabsf(normal.y),
                  fabsf(normal.z));
    float sum = w.x + w.y + w.z + 1e-5f;
    w = w / sum;

    // build UVs for each projection
    Vec2 uvx = Vec2(local.z/(2*he.z)+0.5f,
                    local.y/(2*he.y)+0.5f);
    Vec2 uvy = Vec2(local.x/(2*he.x)+0.5f,
                    local.z/(2*he.z)+0.5f);
    Vec2 uvz = Vec2(local.x/(2*he.x)+0.5f,
                    local.y/(2*he.y)+0.5f);

    Vec3 cx = sample_texture(mat, uvx);
    Vec3 cy = sample_texture(mat, uvy);
    Vec3 cz = sample_texture(mat, uvz);

    return cx * w.x + cy * w.y + cz * w.z;
}

// dispatch to sphere or box UVs, then sample
__device__ Vec3 obj_mapping(
    const GpuMaterial&    mat,
    const GpuSdfObjectBase& obj,
    Vec3                  p
) {
    if (obj.sdf_type == SDF_SPHERE) {
        // spherical mapping
        Vec3 dir = (p - obj.center).normalize();
        Vec3 local_dir = Vec3(
            Vec3::dot(dir, obj.u),
            Vec3::dot(dir, obj.v),
            Vec3::dot(dir, obj.w)
        );
        float u = 0.5f + atan2f(local_dir.z, local_dir.x)/(2.0f*M_PI);
        float v = 0.5f - asinf(local_dir.y)/M_PI;
        return sample_texture(mat, Vec2(u,v));
    } else {
        // triplanar box
        Vec3 normal = (evaluate_grad_sdf(obj, p) * -1.0f).normalize();
        return triplanar_sample(mat, obj, p, normal);
    }
}

extern "C" __global__
void raymarch(
    int width, int height,
    const float* origins,
    const float* directions,
    const GpuSdfObjectBase* sdf_objs,
    int num_objs,
    const GpuMaterial* materials,
    const GpuLight* lights,
    const GpuSpaceFolding* foldings,
    Image_ray_accum* accum
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;
    if (x >= width || y >= height) return;

    Vec3 origin(
        origins[i*3 + 0],
        origins[i*3 + 1],
        origins[i*3 + 2]
    );
    Vec3 dir(
        directions[i*3 + 0],
        directions[i*3 + 1],
        directions[i*3 + 2]
    );

    Vec3 p = origin;
    float total_dist = 0.0f;
    const float eps      = 0.001f;
    const float max_dist = 200.0f;
    const int   max_steps = 100;

    int steps    = 0;
    float min_dist;
    int   best_j = -1;

    // temp copy for per‐object folding
    GpuSdfObjectBase eval;

    bool hit = false;
    while (steps < max_steps) {
        min_dist = 1e20f;
        best_j   = -1;

        for (int j = 0; j < num_objs; ++j) {
            const GpuSdfObjectBase& src = sdf_objs[j];
            if (src.active == 0) continue;

            // copy so we can modify center if folded
            eval = src;

            // folding?
            unsigned fid = src.lattice_folding_id;
            float lattice_x=0, lattice_y=0, lattice_z=0;
            if (fid != INVALID_FOLDING) {
                const GpuSpaceFolding& fold = foldings[fid];
                Vec3 diff = p - eval.center;
                Vec3 lc   = diff * fold.lattice_basis_inv;
                Vec3 kf( roundf(lc.x), roundf(lc.y), roundf(lc.z) );
                eval.center = eval.center + kf * fold.lattice_basis;

                lattice_x = Vec3(fold.lattice_basis.a11, fold.lattice_basis.a21, fold.lattice_basis.a31).length();
                lattice_y = Vec3(fold.lattice_basis.a12, fold.lattice_basis.a22, fold.lattice_basis.a32).length();
                lattice_z = Vec3(fold.lattice_basis.a13, fold.lattice_basis.a23, fold.lattice_basis.a33).length();
            }

            float d = evaluate_sdf(eval, p);
            // if inside, push out
            if (d < 0) {
                float center_dist = (eval.center - p).length();
                d = fminf(fminf(lattice_x, lattice_y), lattice_z)/2 - center_dist;
            }

            if (d < min_dist) {
                min_dist = d;
                best_j   = j;
            }
        }

        if (best_j < 0 || min_dist < eps || total_dist > max_dist) {
            if (min_dist < eps) {
                hit = true;
            }
            break;
        }

        p = p + dir * min_dist;
        total_dist = total_dist + min_dist;
        ++steps;
    }

    // shading
    Vec3 color(0,0,0);
    if (best_j >= 0 && hit) {
        // reconstruct final eval for shading (including fold)
        eval = sdf_objs[best_j];
        unsigned fid = eval.lattice_folding_id;
        if (fid != INVALID_FOLDING) {
            const GpuSpaceFolding& fold = foldings[fid];
            Vec3 diff = p - eval.center;
            Vec3 lc   = diff * fold.lattice_basis_inv;
            Vec3 kf( roundf(lc.x), roundf(lc.y), roundf(lc.z) );
            eval.center = eval.center + kf * fold.lattice_basis;
        }

        const GpuMaterial& mat = materials[ eval.material_id ];
        // normal
        Vec3 normal = evaluate_grad_sdf(eval, p) * -1.0f;

        if (mat.use_texture) {
            color = obj_mapping(mat, eval, p);
        } else {
            color = Vec3(mat.color[0], mat.color[1], mat.color[2]);
        }

        Vec3 L = Vec3(0.5f,1.0f,-0.6f).normalize();
        float lam = fmaxf(0.0f, Vec3::dot(normal, L));
        color = color*lam;
    }

    accumulate(accum, color, i);
}



// Test kernels
extern "C" __global__ void print_str(const char* message) {
    printf("Message from host: %s\n", message);
}

extern "C" __global__ void generate_image(int width, int height, unsigned char *image) {
    // Use float for pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        image[idx] = x % 256;     // Red channel
        image[idx + 1] = y % 256; // Green channel
        image[idx + 2] = (x + y) % 256; // Blue channel
    }
}


extern "C" __global__ void draw_circle(int width, int height, float mouse_x, float mouse_y, unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = 20; // radius of the circle

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float dx = x - mouse_x;
        float dy = y - mouse_y;
        if (dx * dx + dy * dy <= radius * radius) {
            image[idx] = 255;     // Red channel
            image[idx + 1] = 0;   // Green channel
            image[idx + 2] = 0;   // Blue channel
        }
    }
}


extern "C" __global__
void generate_rays_test(int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int i = y * width + x;
    if (i >= width * height) return;

    // Aucun accès mémoire ici !
    if (i == 0) {
        printf("generate_rays_test: launched on (%d x %d)\n", width, height);
    }
}


extern "C" __global__ void print_curand_state_size() {
    printf("[CUDA] sizeof(curandStateXORWOW_t) = %lu\n", sizeof(curandStateXORWOW_t));
}
