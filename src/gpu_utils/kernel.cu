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

struct __align__(8) SdfObject {
    int sdf_type;           // 0 = sphere, ...
    float params[3];        // general-purpose parameters
    Vec3 center, u, v, w;   // center and local basis
    unsigned char* texture; // pointer to device image data
    int tex_width;          // texture width
    int tex_height;         // texture height
    Mat3 lattice_basis;     // basis of the lattice
    Mat3 lattice_basis_inv;
    int active;             // if this sdf object is still active in the world or if it has been despawned
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

__device__ float evaluate_sdf(const SdfObject& obj, Vec3 p) {
    if (obj.sdf_type == SDF_SPHERE) {
        float radius = obj.params[0];
        return sdf_sphere(obj.center, radius, p);
    } else if (obj.sdf_type == SDF_BOX) {
        Vec3 half_extents = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        return sdf_box(p, obj.center, obj.u, obj.v, obj.w, half_extents);
    }
    return 1e9;
}

__device__ Vec3 evaluate_grad_sdf(const SdfObject& obj, Vec3 p) {
    if (obj.sdf_type == SDF_SPHERE) {
        float radius = obj.params[0];
        return grad_sdf_sphere(obj.center, radius, p);
    } else if (obj.sdf_type == SDF_BOX) {
        Vec3 half_extents = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        return grad_sdf_box(p, obj.center, obj.u, obj.v, obj.w, half_extents);
    }
}

__device__ Vec2 spherical_mapping(Vec3 point, Vec3 center, Vec3 u, Vec3 v, Vec3 w) {
    Vec3 dir = (point - center).normalize(); // Point sur la sphère normalisée

    Vec3 local_dir = Vec3(
        Vec3::dot(dir, u),
        Vec3::dot(dir, v),
        Vec3::dot(dir, w)
    );

    float u_map = 0.5f + atan2f(local_dir.z, local_dir.x) / (2.0f * M_PI);
    float v_map = 0.5f - asinf(local_dir.y) / M_PI;

    return Vec2(u_map, v_map);
}

__device__ Vec3 sample_texture(const SdfObject& obj, Vec2 uv) {
    // Clamp u,v to [0,1]
    float u = fminf(fmaxf(uv.x, 0.0f), 1.0f);
    float v = fminf(fmaxf(uv.y, 0.0f), 1.0f);

    int tex_x = static_cast<int>(uv.x * (obj.tex_width - 1));
    int tex_y = static_cast<int>(uv.y * (obj.tex_height - 1));

    int tex_idx = (tex_y * obj.tex_width + tex_x) * 3;

    unsigned char r = obj.texture[tex_idx + 0];
    unsigned char g = obj.texture[tex_idx + 1];
    unsigned char b = obj.texture[tex_idx + 2];

    return Vec3(r / 255.0f, g / 255.0f, b / 255.0f); // normalized RGB
}

__device__ Vec3 triplanar_sample(const SdfObject& obj, Vec3 p, Vec3 normal) {
    Vec3 d = p - obj.center;
    Vec3 local = Vec3(
        Vec3::dot(d, obj.u),
        Vec3::dot(d, obj.v),
        Vec3::dot(d, obj.w)
    );
    Vec3 he = Vec3(obj.params[0], obj.params[1], obj.params[2]);

    float blend_sharpness = 4.0f; // Higher = sharper transitions

    Vec3 weights = Vec3(
        powf(fabsf(normal.x), blend_sharpness),
        powf(fabsf(normal.y), blend_sharpness),
        powf(fabsf(normal.z), blend_sharpness)
    );

    float wsum = weights.x + weights.y + weights.z + 1e-5f;

    weights = weights / wsum;

    Vec2 uv_x = Vec2(local.z / (2.0f * he.z) + 0.5f, local.y / (2.0f * he.y) + 0.5f);
    Vec2 uv_y = Vec2(local.x / (2.0f * he.x) + 0.5f, local.z / (2.0f * he.z) + 0.5f);
    Vec2 uv_z = Vec2(local.x / (2.0f * he.x) + 0.5f, local.y / (2.0f * he.y) + 0.5f);

    Vec3 color_x = sample_texture(obj, uv_x);
    Vec3 color_y = sample_texture(obj, uv_y);
    Vec3 color_z = sample_texture(obj, uv_z);

    return color_x * weights.x + color_y * weights.y + color_z * weights.z;
}

__device__ Vec3 obj_mapping(const SdfObject& obj, Vec3 p) {
    if (obj.sdf_type == SDF_SPHERE) {
        Vec2 uv = spherical_mapping(p, obj.center, obj.u, obj.v, obj.w);
        Vec3 color = sample_texture(obj, uv);
        return color;
    }
    else if (obj.sdf_type == SDF_BOX) {
        Vec3 normal = evaluate_grad_sdf(obj, p) * -1.0f;
        Vec3 color = triplanar_sample(obj, p, normal);
        return color;
    }
}


extern "C" __global__
void raymarch(int width, int height, float* origins, float* directions, SdfObject* scene, int num_objects, Image_ray_accum* image_acc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    if (x >= width || y >= height) return;

    // Load ray origin and direction
    Vec3 origin = Vec3(
        origins[i * 3 + 0],
        origins[i * 3 + 1],
        origins[i * 3 + 2]
    );
    Vec3 dir = Vec3(
        directions[i * 3 + 0],
        directions[i * 3 + 1],
        directions[i * 3 + 2]
    );

    Vec3 p = origin;
    float total_dist = 0.0f;
    const float eps = 0.001f;
    const float max_dist = 6.0f;
    const int max_steps = 100;
    int steps = 0;
    int j_min = -1;
    float   min_dist = 1e20f;
    float d = min_dist;

    Vec3 p_loc_min = Vec3(0); // used for periodic lattice folding
    Vec3 p_loc = Vec3(0);
    bool periodic_lattice_folding = false;

    while (steps < max_steps) {
        min_dist = 1e20f;
        j_min    = -1;
        

        for (int j = 0; j < num_objects; ++j) {

            SdfObject sdf_obj = scene[j];
            
            if (sdf_obj.active == 0) {continue;}

            periodic_lattice_folding = sdf_obj.lattice_basis.is_null();
            if (periodic_lattice_folding) {
                d = evaluate_sdf(sdf_obj, p);
            } else {
                if (i==1) {printf(" no folding \n");}
                Mat3 A = sdf_obj.lattice_basis;
                Mat3 A_inv = sdf_obj.lattice_basis_inv;

                p_loc = p*A;
                p_loc = p_loc - p_loc.floor() - Vec3(0.5);
                p_loc = p_loc * A_inv;

                d = evaluate_sdf(sdf_obj, p_loc);
            }

            if (d < min_dist) {
                p_loc_min = p_loc;
                min_dist = d;
                j_min = j;
            }
        }

        if (min_dist < eps || total_dist > max_dist)
            break;

        p = p + dir * min_dist;
        total_dist += min_dist;
        steps++;
    }

    Vec3 color;
    SdfObject hit_object;
    
    bool hit = (min_dist < eps && j_min >= 0);

    if (!hit){
        color = Vec3(0.0, 0.0, 0.0);
    } else {
        hit_object = scene[j_min];

        color = obj_mapping(hit_object, p);

        Vec3 normal = Vec3(0);
        if (periodic_lattice_folding) {
                normal = evaluate_grad_sdf(hit_object, p)*-1; // ou gradient numérique si type générique
            } else {
                normal = evaluate_grad_sdf(hit_object, p_loc_min)*-1; // ou gradient numérique si type générique
            }
        Vec3 light_dir = Vec3(0.5, 1.0, -0.6).normalize();
        float shade = fmaxf(0.0f, Vec3::dot(normal, light_dir));
        color = color * shade;
    }
    accumulate(image_acc, color, i);
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
