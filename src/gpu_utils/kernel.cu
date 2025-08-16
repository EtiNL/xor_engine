#include <math.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h> 

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

struct Image_ray_accum {
    int* ray_per_pixel;
    unsigned char* image;
};

__device__ void accumulate(Image_ray_accum& acc,
                           const Vec3& color,
                           int pixel_idx)
{
    int rpp      = acc.ray_per_pixel[pixel_idx]; 
    int new_rpp  = rpp + 1;
    int base     = 3 * pixel_idx;

    acc.image[base + 0] =
        (acc.image[base + 0] * rpp + color.x * 255.0f) / new_rpp;
    acc.image[base + 1] =
        (acc.image[base + 1] * rpp + color.y * 255.0f) / new_rpp;
    acc.image[base + 2] =
        (acc.image[base + 2] * rpp + color.z * 255.0f) / new_rpp;

    acc.ray_per_pixel[pixel_idx] = new_rpp;
}

struct GpuCamera {
    /* intrinsics / pose */
    Vec3  position, u, v, w;
    float aperture;
    float focus_dist;
    float viewport_width;
    float viewport_height;

    /* buffers */
    curandState*     rand_states;   // nullptr when aperture == 0
    float*           origins;
    float*           directions;
    Image_ray_accum  accum;         // .ray_per_pixel may be nullptr
    unsigned char*   image;

    /* misc */
    unsigned int spp;
    unsigned int width;
    unsigned int height;
    
    unsigned int rand_seed_init_count;
};

#define MAX_LEAFS   65
#define MAX_NODES   (2*MAX_LEAFS - 1)

enum CsgOp : uint32_t { OP_LEAF=0, OP_UNION=1, OP_INTER=2, OP_DIFF=3 };

struct GpuCsgTree {
    // counts and root
    uint32_t node_count;          // total nodes used (<= MAX_NODES)
    uint32_t leaf_count;          // leaves used (<= MAX_LEAFS)
    uint32_t root;                // node id of the root

    // structure-of-arrays for binary tree
    uint32_t op      [MAX_NODES]; // OP_LEAF/OP_UNION/OP_INTER/OP_DIFF
    uint32_t left    [MAX_NODES]; // child node id (undefined for leaves)
    uint32_t right   [MAX_NODES]; // child node id (undefined for leaves)
    uint32_t payload [MAX_NODES]; // for leaves: sdf_objs index; internal: unused

    // postorder evaluation schedule (node ids in postorder)
    uint32_t eval_postorder[MAX_NODES]; // length = node_count

    // optional: per-leaf cheap bounds for interval tests (fill as needed)
    // float leaf_bound_sphere[MAX_LEAFS];

    // your existing extras
    uint32_t material_id;
    uint32_t tree_folding_id;
    uint32_t active;
};

enum SdfType {
    SDF_SPHERE = 0,
    SDF_BOX    = 1,
    SDF_PLANE  = 2,
    // Extend as needed
};

#define INVALID_MATERIAL 0xFFFFFFFFu
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
    unsigned int in_csg_tree;
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
    float min_half_thickness;
    unsigned int active_mask;    // bit0=u/X, bit1=v/Y, bit2=w/Z
};

extern "C"
__global__ void reset_accum(int* ray_per_pixel, int total_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_pixels) {
        ray_per_pixel[i] = 0;
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
void generate_rays(int width, int height, GpuCamera* cameras, int camera_index)
{   
    GpuCamera& cam = cameras[camera_index];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = y * width + x;
    if (x >= width || y >= height) return;

    /* ----- camera-space ray dir ------------------------------------------------ */
    const float u = (x + 0.5f) / width  * 2.0f - 1.0f;
    const float v = 1.0f - (y + 0.5f) / height * 2.0f;

    Vec3 dir_cam(
        u * cam.viewport_width  * 0.5f,
        v * cam.viewport_height * 0.5f,
        1.0f);
    dir_cam = dir_cam.normalize();

    Vec3 dir_world = cam.u * dir_cam.x + cam.v * dir_cam.y + cam.w * dir_cam.z;
    Vec3 origin, direction;

    /* ----- depth of field ------------------------------------------------------ */
    if (cam.aperture > 0.0f)
    {
        // one-time RNG setup per pixel
        if (cam.rand_seed_init_count < width*height)
        {   
            curand_init(42u,            /* seed  */
                        i,              /* sequence */
                        0u,             /* offset  */
                        &cam.rand_states[i]);

            /* SAFELY bump the counter in global memory */
            atomicAdd(&cam.rand_seed_init_count, 1u);
        }

        curandState local = cam.rand_states[i];
        const float lens_r = cam.aperture * 0.5f;
        Vec3  rd = random_in_unit_disk(&local) * lens_r;
        Vec3  offset = cam.u * rd.x + cam.v * rd.y;

        origin    = cam.position + offset;
        Vec3 fp   = cam.position + dir_world * cam.focus_dist;
        direction = (fp - origin).normalize();

        cam.rand_states[i] = local;            // write back
    }
    else
    {
        origin    = cam.position;
        direction = dir_world;                 // already unit length
    }

    /* ----- store ---------------------------------------------------------------- */
    cam.origins   [3*i + 0] = origin.x;
    cam.origins   [3*i + 1] = origin.y;
    cam.origins   [3*i + 2] = origin.z;

    cam.directions[3*i + 0] = direction.x;
    cam.directions[3*i + 1] = direction.y;
    cam.directions[3*i + 2] = direction.z;
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

__device__ float evaluate_sdf(const GpuSdfObjectBase& obj, Vec3 p, Vec3 center) {
    if (obj.sdf_type == SDF_SPHERE) {
        float radius = obj.params[0];
        return sdf_sphere(center, radius, p);
    } else if (obj.sdf_type == SDF_BOX) {
        Vec3 half_extents = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        return sdf_box(p, center, obj.u, obj.v, obj.w, half_extents);
    }
    return 1e9;
}

__device__ Vec3 evaluate_grad_sdf(const GpuSdfObjectBase& obj, Vec3 p, Vec3 center) {
    if (obj.sdf_type == SDF_SPHERE) {
        float radius = obj.params[0];
        return grad_sdf_sphere(center, radius, p);
    } else if (obj.sdf_type == SDF_BOX) {
        Vec3 half_extents = Vec3(obj.params[0], obj.params[1], obj.params[2]);
        return grad_sdf_box(p, center, obj.u, obj.v, obj.w, half_extents);
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
    Vec3 p,
    Vec3 normal,
    Vec3 center
) {
    // local space coords
    Vec3 d = p - center;
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
    Vec3 p, 
    Vec3 center
) {
    if (obj.sdf_type == SDF_SPHERE) {
        // spherical mapping
        Vec3 dir = (p - center).normalize();
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
        Vec3 normal = (evaluate_grad_sdf(obj, p, center) * -1.0f).normalize();
        return triplanar_sample(mat, obj, p, normal, center);
    }
}
// --- folding helper (tree-level override takes precedence) -------------------
__device__ __forceinline__
Vec3 folded_center_for(const GpuSdfObjectBase& obj,
                       Vec3 p,
                       const GpuSpaceFolding* __restrict__ foldings,
                       unsigned int tree_fid)
{
    unsigned int fid = (tree_fid != INVALID_FOLDING) ? tree_fid : obj.lattice_folding_id;
    Vec3 c = obj.center;
    if (fid != INVALID_FOLDING) {
        const GpuSpaceFolding& fold = foldings[fid];
        Vec3 diff = p - c;
        Vec3 lc   = diff * fold.lattice_basis_inv;
        Vec3 kf( (fold.active_mask & 1u) ? roundf(lc.x) : 0.0f,
                 (fold.active_mask & 2u) ? roundf(lc.y) : 0.0f,
                 (fold.active_mask & 4u) ? roundf(lc.z) : 0.0f );
        c = c + kf * fold.lattice_basis;
    }
    return c;
}

// --- cheap leaf lower bound using a bounding sphere --------------------------
__device__ __forceinline__
float leaf_bound_radius(const GpuSdfObjectBase& obj)
{
    if (obj.sdf_type == SDF_SPHERE) {
        return obj.params[0];
    } else if (obj.sdf_type == SDF_BOX) {
        // circumscribed sphere of OBB
        float x = obj.params[0], y = obj.params[1], z = obj.params[2];
        return sqrtf(x*x + y*y + z*z);
    }
    // fallback (very loose)
    return 1e9f;
}

__device__ __forceinline__
float leaf_lower_bound(const GpuSdfObjectBase& obj,
                       Vec3 p,
                       Vec3 c_fold)
{
    return (p - c_fold).length() - leaf_bound_radius(obj);
}

// --- bottom-up lower bounds (SoA tree, postorder) ----------------------------
__device__ __forceinline__
void compute_node_lower_bounds(const GpuCsgTree& tree,
                               Vec3 p,
                               const GpuSdfObjectBase* __restrict__ sdf,
                               const GpuSpaceFolding* __restrict__ folds,
                               unsigned int tree_fid,
                               float* __restrict__ lb /*size >= MAX_NODES*/)
{
    #pragma unroll 1
    for (int i=0; i<tree.node_count; ++i) {
        uint32_t n  = tree.eval_postorder[i];
        uint32_t op = tree.op[n];

        if (op == OP_LEAF) {
            const GpuSdfObjectBase& o = sdf[ tree.payload[n] ];
            Vec3 c = folded_center_for(o, p, folds, tree_fid);
            lb[n]  = leaf_lower_bound(o, p, c);
        } else {
            uint32_t L = tree.left [n];
            uint32_t R = tree.right[n];
            float bL = lb[L];
            float bR = lb[R];
            if (op == OP_UNION)      lb[n] = fminf(bL, bR);
            else if (op == OP_INTER) lb[n] = fmaxf(bL, bR);
            else /*OP_DIFF*/         lb[n] = bL;   // conservative (needs upper bound for tighter)
        }
    }
}

// --- branch & bound evaluation (iterative, prunes UNION & DIFFERENCE) --------
struct EvalTmp {
    float   d;
    uint32_t leaf_id;   // sdf index
    float   sign;
    uint8_t done;
};

struct CsgCombineResult {
    float     d;
    uint32_t  leaf_id;
    float     grad_sign;   // +1 or -1 (for DIFFERENCE)
};


__device__ __forceinline__
CsgCombineResult eval_csg_tree_bnb(const GpuCsgTree& tree,
                                   Vec3 p,
                                   const GpuSdfObjectBase* __restrict__ sdf,
                                   const GpuSpaceFolding* __restrict__ folds)
{
    float   lb[MAX_NODES];
    EvalTmp tmp[MAX_NODES];

    // 1) lower bounds for ordering/pruning
    compute_node_lower_bounds(tree, p, sdf, folds, tree.tree_folding_id, lb);

    // 2) best-first + pruning using a tiny explicit stack
    uint32_t firstChild[MAX_NODES];
    uint32_t secondChild[MAX_NODES];

    uint32_t stackNode[MAX_NODES];
    uint8_t  stackState[MAX_NODES]; // 0=setup, 1=after first, 2=after second
    int sp = -1;

    // init temps
    #pragma unroll 1
    for (int n=0; n<tree.node_count; ++n) { tmp[n].done=0; }

    stackNode[++sp]  = tree.root;
    stackState[sp]   = 0;

    while (sp >= 0) {
        uint32_t n  = stackNode[sp];
        uint8_t  st = stackState[sp];

        uint32_t op = tree.op[n];
        if (op == OP_LEAF) {
            const GpuSdfObjectBase& o = sdf[ tree.payload[n] ];
            Vec3 c = folded_center_for(o, p, folds, tree.tree_folding_id);
            float d = evaluate_sdf(o, p, c);
            tmp[n] = { d, tree.payload[n], 1.0f, 1u };
            --sp;
            continue;
        }

        uint32_t L = tree.left [n];
        uint32_t R = tree.right[n];

        if (st == 0) {
            // decide order once using bounds
            if (op == OP_UNION) {
                if (lb[L] <= lb[R]) { firstChild[n]=L; secondChild[n]=R; }
                else                { firstChild[n]=R; secondChild[n]=L; }
            } else if (op == OP_INTER) {
                // evaluate the one with larger lower bound first (helps a bit)
                if (lb[L] >= lb[R]) { firstChild[n]=L; secondChild[n]=R; }
                else                { firstChild[n]=R; secondChild[n]=L; }
            } else { // OP_DIFF: evaluate left first
                firstChild[n]  = L;
                secondChild[n] = R;
            }
            // descend into first if not done
            if (!tmp[firstChild[n]].done) {
                stackState[sp] = 1;
                stackNode[++sp] = firstChild[n];
                stackState[sp]  = 0;
                continue;
            }
            st = 1; // already computed by someone else
        }

        if (st == 1) {
            const uint32_t F = firstChild[n];
            const uint32_t S = secondChild[n];

            // pruning tests
            if (op == OP_UNION) {
                // if d_first <= lb(second) then min == d_first
                if (tmp[F].d <= lb[S]) {
                    tmp[n] = { tmp[F].d, tmp[F].leaf_id, tmp[F].sign, 1u };
                    --sp;
                    continue;
                }
            } else if (op == OP_DIFF) {
                // if d_left >= -lb(right) then max(dL, -dR) == d_left
                if (tmp[F].d >= -lb[S]) {
                    tmp[n] = { tmp[F].d, tmp[F].leaf_id, tmp[F].sign, 1u };
                    --sp;
                    continue;
                }
            }
            // need the second child
            if (!tmp[S].done) {
                stackState[sp] = 2;
                stackNode[++sp] = S;
                stackState[sp]  = 0;
                continue;
            }
            st = 2;
        }

        // st == 2 → both children computed → combine exactly
        {
            const uint32_t Lc = tree.left [n];
            const uint32_t Rc = tree.right[n];
            float dl = tmp[Lc].d, dr = tmp[Rc].d;
            float gl = tmp[Lc].sign, gr = tmp[Rc].sign;

            if (op == OP_UNION) {
                if (dr < dl) tmp[n] = { dr, tmp[Rc].leaf_id, gr, 1u };
                else         tmp[n] = { dl, tmp[Lc].leaf_id, gl, 1u };
            } else if (op == OP_INTER) {
                if (dr > dl) tmp[n] = { dr, tmp[Rc].leaf_id, gr, 1u };
                else         tmp[n] = { dl, tmp[Lc].leaf_id, gl, 1u };
            } else { // OP_DIFF
                if (dl > -dr) tmp[n] = { dl,   tmp[Lc].leaf_id,  gl,   1u };
                else          tmp[n] = { -dr,  tmp[Rc].leaf_id, -gr,   1u };
            }
            --sp;
        }
    }

    CsgCombineResult out;
    out.d         = tmp[tree.root].d;
    out.leaf_id   = tmp[tree.root].leaf_id;
    out.grad_sign = tmp[tree.root].sign;
    return out;
}



extern "C" __global__
void raymarch(
    int width, int height,
    GpuCamera* cameras,
    int camera_index,
    const GpuCsgTree* __restrict__ csg_trees,
    int num_trees,
    const GpuSdfObjectBase* __restrict__ sdf_objs,
    int num_objs,
    const GpuMaterial* __restrict__ materials,
    const GpuLight* __restrict__ lights,
    const GpuSpaceFolding* __restrict__ foldings
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;
    if (x >= width || y >= height) return;

    GpuCamera& cam = cameras[camera_index];

    Vec3 origin(
        cam.origins[i*3 + 0],
        cam.origins[i*3 + 1],
        cam.origins[i*3 + 2]
    );
    Vec3 dir(
        cam.directions[i*3 + 0],
        cam.directions[i*3 + 1],
        cam.directions[i*3 + 2]
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
    Vec3 center;
    Vec3 center_min;
    unsigned int best_tree_material = INVALID_MATERIAL;
    unsigned int best_tree_folding = INVALID_FOLDING;
    float best_grad_sign = 1.0f;

    bool hit = false;
    while (steps < max_steps) {
        min_dist = 1e20f;
        best_j   = -1;

        for (int j = 0; j < num_objs; ++j) {
            const GpuSdfObjectBase& src = sdf_objs[j];
            center = src.center;
            if (src.active == 0 || src.in_csg_tree != 0) continue;

            // folding?
            unsigned fid = src.lattice_folding_id;
            float min_half_thickness = 0.0;
            if (fid != INVALID_FOLDING) {
                const GpuSpaceFolding& fold = foldings[fid];
                Vec3 diff = p - center;
                Vec3 lc   = diff * fold.lattice_basis_inv;
                float kx = (fold.active_mask & 1u) ? roundf(lc.x) : 0.0f;
                float ky = (fold.active_mask & 2u) ? roundf(lc.y) : 0.0f;
                float kz = (fold.active_mask & 4u) ? roundf(lc.z) : 0.0f;
                Vec3 kf(kx, ky, kz);
                center = center + kf * fold.lattice_basis;
                min_half_thickness = fold.min_half_thickness;
            }

            float d = evaluate_sdf(src, p, center);
            // if inside, push out
            if (d < 0) {
                float center_dist = (center - p).length();
                d = min_half_thickness - center_dist;
            }

            if (d < min_dist) {
                center_min = center;
                min_dist = d;
                best_j   = j;
            }
        }

        for (int j = 0; j < num_trees; ++j) {
            const GpuCsgTree& T = csg_trees[j];
            if (!T.active) continue;

            // quick root lower bound to possibly skip the whole tree
            float lb_root[MAX_NODES];
            compute_node_lower_bounds(T, p, sdf_objs, foldings, T.tree_folding_id, lb_root);
            if (lb_root[T.root] >= min_dist) continue;

            CsgCombineResult comb = eval_csg_tree_bnb(T, p, sdf_objs, foldings);

            float d = comb.d;
            if (d < min_dist) {
                const GpuSdfObjectBase& wsrc = sdf_objs[ comb.leaf_id ];
                Vec3 wcenter = folded_center_for(wsrc, p, foldings, T.tree_folding_id);
                center_min         = wcenter;
                min_dist           = d;
                best_j             = (int)comb.leaf_id;
                best_tree_folding  = T.tree_folding_id;
                best_tree_material = T.material_id;
                best_grad_sign     = comb.grad_sign;
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
        const GpuSdfObjectBase& eval = sdf_objs[best_j];

        unsigned int fid;
        if (best_tree_folding != INVALID_FOLDING) {fid = best_tree_folding;} else {fid = eval.lattice_folding_id;}

        unsigned int mid;
        if (best_tree_material != INVALID_MATERIAL) {mid = best_tree_material;} else {mid = eval.material_id;}
        
        if (mid != INVALID_MATERIAL) {
            
            const GpuMaterial& mat = materials[ mid ];

            if (mat.use_texture) {
                color = obj_mapping(mat, eval, p, center_min);
            } else {
                color = Vec3(mat.color[0], mat.color[1], mat.color[2]);
            }

            Vec3 normal = (evaluate_grad_sdf(eval, p, center_min) * (-1.0f * best_grad_sign)).normalize();
            Vec3 L = Vec3(0.5f,-1.0f,-0.6f).normalize();

            float lam = fmaxf(0.0f, Vec3::dot(normal, L));
            color = color*fminf(lam+0.3f, 1.0f);
            if (cam.spp > 1) {
                accumulate(cam.accum, color, i);
            }
            else {
                cam.image[i*3 + 0] = color.x * 255.0f;
                cam.image[i*3 + 1] = color.y * 255.0f;
                cam.image[i*3 + 2] = color.z * 255.0f;
            }
        }
    }
    else {
                cam.image[i*3 + 0] = color.x * 255.0f;
                cam.image[i*3 + 1] = color.y * 255.0f;
                cam.image[i*3 + 2] = color.z * 255.0f;
    }
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
