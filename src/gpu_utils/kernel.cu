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

    __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
    __device__ Vec3 normalize() const {
        float len = length();
        return (len > 0) ? (*this) / len : Vec3(0, 0, 0);
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

struct SdfObject {
    int sdf_type;           // 0 = sphere, ...
    float params[8];        // general-purpose parameters
    unsigned char* texture; // pointer to device image data
    int tex_width;          // texture width
    int tex_height;         // texture height
    float mapping_params[8];
};

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

    // if (i == 0){
    //         printf("Camera aperture = %d", cam.aperture);
    //     }


    if (cam.aperture > 0.0f) {
        float lens_radius = cam.aperture * 0.5f;
        Vec3 rd = random_in_unit_disk(&local_rand) * lens_radius;
        Vec3 offset = cam.u * rd.x + cam.v * rd.y;
        origin = cam.position + offset;
        Vec3 focal_point = cam.position + dir_world * cam.focus_dist;
        direction = (focal_point - origin).normalize();

        rand_states[i] = local_rand;

    } else {
        // if (i == 0){
        //     printf("Aperture = 0, pinhole camera model");
        // }
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

__device__ float sdf_box(Vec3 p, Vec3 center, Vec3 half_extents) {
    Vec3 d = (p - center).abs() - half_extents;
    return fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0f) + Vec3::max(d, Vec3(0,0,0)).length();
}

__device__ Vec3 grad_sdf_box(Vec3 p, Vec3 center, Vec3 half_extents) {
    Vec3 d = (p - center);
    Vec3 q = d.abs() - half_extents;

    // gradient approximatif (analytique hors discontinuités)
    Vec3 g = Vec3(
        d.x >= 0.0f ? 1.0f : -1.0f,
        d.y >= 0.0f ? 1.0f : -1.0f,
        d.z >= 0.0f ? 1.0f : -1.0f
    );

    // Si point à l'extérieur de la box
    if (q.x > 0.0f || q.y > 0.0f || q.z > 0.0f) {
        Vec3 clamped = Vec3(
            fmaxf(q.x, 0.0f),
            fmaxf(q.y, 0.0f),
            fmaxf(q.z, 0.0f)
        );
        return clamped.normalize() * g;
    }

    // Sinon, on est à l'intérieur : normal vers la face la plus proche
    if (q.x > q.y && q.x > q.z)
        return Vec3(g.x, 0, 0);
    else if (q.y > q.z)
        return Vec3(0, g.y, 0);
    else
        return Vec3(0, 0, g.z);
}

__device__ float evaluate_sdf(const SdfObject& obj, Vec3 p) {
    if (obj.sdf_type == SDF_SPHERE) {
        Vec3 center = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        float radius = obj.params[3];
        return sdf_sphere(center, radius, p);
    } else if (obj.sdf_type == SDF_BOX) {
        Vec3 center = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        Vec3 half_extents = Vec3(obj.params[3], obj.params[4], obj.params[5]);
        return sdf_box(p, center, half_extents);
    }
    return 1e9;
}

__device__ Vec3 evaluate_grad_sdf(const SdfObject& obj, Vec3 p) {
    if (obj.sdf_type == SDF_SPHERE) {
        Vec3 center = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        float radius = obj.params[3];
        return grad_sdf_sphere(center, radius, p);
    } else if (obj.sdf_type == SDF_BOX) {
        Vec3 center = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        Vec3 half_extents = Vec3(obj.params[3], obj.params[4], obj.params[5]);
        return grad_sdf_box(p, center, half_extents);
    }
}

__device__ Vec2 spherical_mapping(Vec3 point, Vec3 center) {
    Vec3 dir = (point - center).normalize(); // Point sur la sphère normalisée

    float u = 0.5f + atan2f(dir.z, dir.x) / (2.0f * M_PI);
    float v = 0.5f - asinf(dir.y) / M_PI;

    return Vec2(u, v);
}
__device__ Vec2 blend_uv(Vec2 a, Vec2 b, float wa, float wb) {
    return (a * wa + b * wb) / (wa + wb + 1e-5f);
}

__device__ Vec3 sample_texture(const SdfObject& obj, float u, float v) {
    // Clamp u,v to [0,1]
    u = fminf(fmaxf(u, 0.0f), 1.0f);
    v = fminf(fmaxf(v, 0.0f), 1.0f);

    int tex_x = static_cast<int>(u * (obj.tex_width - 1));
    int tex_y = static_cast<int>(v * (obj.tex_height - 1));

    int tex_idx = (tex_y * obj.tex_width + tex_x) * 3;

    unsigned char r = obj.texture[tex_idx + 0];
    unsigned char g = obj.texture[tex_idx + 1];
    unsigned char b = obj.texture[tex_idx + 2];

    return Vec3(r / 255.0f, g / 255.0f, b / 255.0f); // normalized RGB
}

__device__ Vec3 triplanar_sample(const SdfObject& obj, Vec3 p, Vec3 normal) {
    Vec3 local = p - Vec3(obj.params[0], obj.params[1], obj.params[2]);
    Vec3 he = Vec3(obj.params[3], obj.params[4], obj.params[5]);

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

    Vec3 color_x = sample_texture(obj, uv_x.x, uv_x.y);
    Vec3 color_y = sample_texture(obj, uv_y.x, uv_y.y);
    Vec3 color_z = sample_texture(obj, uv_z.x, uv_z.y);

    return color_x * weights.x + color_y * weights.y + color_z * weights.z;
}

__device__ Vec3 obj_mapping(const SdfObject& obj, Vec3 p) {
    if (obj.sdf_type == SDF_SPHERE) {
        Vec3 center = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        Vec2 uv = spherical_mapping(p, center);
        Vec3 color = sample_texture(obj, uv.x, uv.y);
        return color;
    }
    else if (obj.sdf_type == SDF_BOX) {
        Vec3 normal = evaluate_grad_sdf(obj, p) * -1.0f;
        Vec3 color = triplanar_sample(obj, p, normal);
        return color;
    }
}


extern "C" __global__
void raymarch(int width, int height, float* origins, float* directions, SdfObject* scene, int num_objects, unsigned char *image) {
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
    const float eps = 0.0001f;
    const float max_dist = 6.0f;
    const int max_steps = 100;
    int steps = 0;
    int j_min = -1;
    float min_dist = 1e9;

    while (steps < max_steps) {
        //    if ((x == width/2)&&(y==height/2)) {
        //         float d = evaluate_sdf(scene[0], p);
        //         printf("Step = %d: SDF = %f at p = (%f, %f, %f)\n", steps, d, p.x, p.y, p.z);
        //     }

        for (int j = 0; j < num_objects; ++j) {
            float d = evaluate_sdf(scene[j], p);
            if (d < min_dist) {
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
    if (eps < min_dist ){
        color = Vec3(0.0, 0.0, 0.0);
    } else {
        hit_object = scene[j_min];

        color = obj_mapping(hit_object, p);

        Vec3 normal = evaluate_grad_sdf(hit_object, p)*-1; // ou gradient numérique si type générique
        Vec3 light_dir = Vec3(0.5, 1.0, -0.6).normalize();
        float shade = fmaxf(0.0f, Vec3::dot(normal, light_dir));
        color = color * shade;
    }

    int idx = (y * width + x) * 3;
    image[idx + 0] = static_cast<unsigned char>(color.x * 255.0f);
    image[idx + 1] = static_cast<unsigned char>(color.y * 255.0f);
    image[idx + 2] = static_cast<unsigned char>(color.z * 255.0f);
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
