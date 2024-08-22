#include <curand_kernel.h>
#include <stdio.h>

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
    // Use float for pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = 20; // radius of the circle

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float dx = x - mouse_x;
        float dy = y - mouse_y;
        if (dx * dx + dy * dy <= radius * radius) {
            // Inside the circle: set to red
            image[idx] = 255;     // Red channel
            image[idx + 1] = 0;   // Green channel
            image[idx + 2] = 0;   // Blue channel
        }
    }
}


extern "C" __global__ void produit_scalaire(int num_rays, unsigned char *A, unsigned char *B, unsigned char *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {
        result[idx] = A[3*idx] * B[3*idx] + A[3*idx + 1] * B[3*idx + 1] + A[3*idx + 2] * B[3*idx + 2];
    }
}


extern "C" __global__ void init_light_source(float *O, float *D, float *light_basis, int num_rays, float mu, float sigma, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {
        // Initialisation du générateur de nombres aléatoires pour chaque thread
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Générer deux nombres aléatoires uniformes entre 0 et 1
        float u1 = curand_uniform(&state);
        float u2 = curand_uniform(&state);

        // Méthode de Box-Muller pour générer deux échantillons indépendants
        float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * M_PI * u2);

        // Appliquer la moyenne et l'écart-type pour obtenir les points (x, y) de distribution normale
        float d1 = mu + sigma * z1;
        float d2 = mu + sigma * z2;
        
        // printf("Thread %d: d1 = %f, d2 = %f\n", idx, d1, d2);


        O[3*idx] = d1 * light_basis[0] + d2 * light_basis[3];
        O[3*idx + 1] = d1 * light_basis[1] + d2 * light_basis[4];
        O[3*idx + 2] = d1 * light_basis[2] + d2 * light_basis[5];

        D[3*idx] = light_basis[6];
        D[3*idx + 1] = light_basis[7];
        D[3*idx + 2] = light_basis[8];
    }
}

extern "C" __global__ void test_norm_distrib_light_source(float *test, unsigned char *image, int num_rays, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        // Add simple debug print here
        // printf("Thread (%d, %d), idx: %d running.\n", x, y, idx);
        // printf("width, height (%d, %d)\n", width, height);

        for (int k = 0; k < num_rays; k++) {
            float dx = test[2*k] - x;
            float dy = test[2*k + 1] - y;
            float d = dx*dx + dy*dy;
            
            // Add debug prints for values
            // printf("Thread (%d, %d), k=%d: dx=%f, dy=%f, d=%f\n", x, y, k, dx, dy, d);

            if (d < 5.0f) {
                // printf("width, height (%d, %d)\n", width, height);
                image[idx] = 255;
                image[idx + 1] = 255;
                image[idx + 2] = 255;
            }

            // else {
            //     image[idx] = 0;
            //     image[idx + 1] = 0;
            //     image[idx + 2] = 0;
            // }
        }
    }
}




extern "C" __global__ void sdf_sphere(int num_rays, float sphereX, float sphereY, float sphereZ, float radius, unsigned char *O, unsigned char *D, unsigned char *T, unsigned char *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {
        // Compute distance from ray to sphere center
        float dx = O[3*idx] + T[idx] * D[3*idx] - sphereX;
        float dy = O[3*idx + 1] + T[idx] * D[3*idx + 1] - sphereY;
        float dz = O[3*idx + 2] + T[idx] * D[3*idx + 2] - sphereZ;
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Compute the SDF value and convert to depth map value
        float sdf = distance - radius;
        result[idx] = sdf;
    }
}

extern "C" __global__ void grad_sdf_sphere(int num_rays, float sphereX, float sphereY, float sphereZ, float radius, unsigned char *O, unsigned char *D, unsigned char *T, unsigned char *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {
        result[3*idx] = (O[3*idx] + T[idx] * D[3*idx] - sphereX) / radius;
        result[3*idx + 1] = (O[3*idx + 1] + T[idx] * D[3*idx + 1] - sphereY) / radius;
        result[3*idx + 2] = (O[3*idx + 2] + T[idx] * D[3*idx + 2] - sphereZ) / radius;
    }
}


extern "C" __global__ void ray_march(int num_rays, float epsilon_grad, unsigned char *Grad_sdf_dot_d, unsigned char *Sdf, unsigned char *Sdf_i, unsigned char *T, unsigned char *T_i, unsigned char *T_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {

        float grad = Grad_sdf_dot_d[idx];
        float sdf = Sdf[idx];
        float sdf_i = Sdf_i[idx];
        float t = T[idx];

        if (grad > epsilon_grad) {
            //Newton_Raphson
            T[idx] = t - sdf / grad;
        } else {
            // Bisection
            if ((sdf > 0 && sdf_i < 0) || (sdf < 0 && sdf_i > 0)) {
                T_f[idx] = t;
            } else {
                T_i[idx] = t;
            }
            T[idx] = (T_f[idx] + T_i[idx]) / 2;
        }
    }
}

extern "C" __global__ void reflexion(int num_rays, unsigned char *D, unsigned char *O, unsigned char *T, unsigned char *surf_grad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {
        float norm_surf_grad = surf_grad[3*idx] * surf_grad[3*idx] + surf_grad[3*idx + 1] * surf_grad[3*idx + 1] + surf_grad[3*idx + 2] * surf_grad[3*idx + 2];
        float surf_grad_dot_D = (surf_grad[3*idx] * D[3*idx] + surf_grad[3*idx + 1] * D[3*idx + 1] + surf_grad[3*idx + 2] * D[3*idx + 2]) / norm_surf_grad ;

        O[3*idx] = O[3*idx] + T[idx] * D[3*idx];
        O[3*idx + 1] = O[3*idx + 1] + T[idx] * D[3*idx + 1];
        O[3*idx + 2] = O[3*idx + 2] + T[idx] * D[3*idx + 2];

        D[3*idx] = D[3*idx] - 2 * surf_grad_dot_D * surf_grad[3*idx];
        D[3*idx + 1] = D[3*idx + 1] - 2 * surf_grad_dot_D * surf_grad[3*idx + 1];
        D[3*idx + 2] = D[3*idx + 2] - 2 * surf_grad_dot_D * surf_grad[3*idx + 2];

        T[idx] = 0;
    }
}
extern "C" __global__ void camera_diffusion(int num_rays, float width, float height, unsigned char *screen_center, unsigned char *repere_camera, unsigned char *O, unsigned char *D, unsigned char *ray_screen_collision) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {

        float screen_x = screen_center[0];
        float screen_y = screen_center[1];
        float screen_z = screen_center[2];

        float u_rho_x = repere_camera[0]; // sin(theta) * cos(phi)
        float u_rho_y = repere_camera[1]; // sin(theta) * sin(phi)
        float u_rho_z = repere_camera[2]; // cos(theta)

        float u_theta_x = repere_camera[3]; // cos(theta) * cos(phi)
        float u_theta_y = repere_camera[4]; // cos(theta) * sin(phi)
        float u_theta_z = repere_camera[5]; // -sin(theta)

        float u_phi_x = repere_camera[6]; // -sin(phi)
        float u_phi_y = repere_camera[7]; // cos(phi)
        float u_phi_z = repere_camera[8]; // 0

        float ray_x = O[3*idx];
        float ray_y = O[3*idx + 1];
        float ray_z = O[3*idx + 2];

        float diffusion_coef = - (u_rho_x * D[3*idx] + u_rho_y * D[3*idx + 1] + u_rho_z * D[3*idx + 2]); // cos of the angle between - normale camera (u_rho) et ray reflected direction (modèle de Lambert)

        if (diffusion_coef > 0.1f) {

            float sdf_screen = u_rho_x * (ray_x - screen_x) + u_rho_y * (ray_y - screen_y) + u_rho_z * (ray_z - screen_z);
            float x = 0.0;
            float y = 0.0;

            if (u_theta_z != 0.0f) {

                y = -((1 + sdf_screen) * u_rho_z - ray_z) / u_theta_z;

                if (u_phi_y != 0.0f) { x = -(ray_y - y * u_theta_y - (1 + sdf_screen) * u_rho_y) / u_phi_y ;
                }

                else { x = ray_x / u_phi_x ;
                }
            }

            else {

                x = u_phi_x * ray_x + u_phi_y * ray_y;
                y = u_phi_y * ray_x - u_phi_x * ray_y;

                if (u_rho_z != 1.0f) {
                    float y_flip = -y;
                    y = y_flip;
                }
            }

            if (fabs(x + width / 2 ) < width / 2 && fabs(y + height / 2 ) < height) {
                
                ray_screen_collision[4*idx] = x;
                ray_screen_collision[4*idx + 1] = y;
                ray_screen_collision[4*idx + 2] = diffusion_coef;
                ray_screen_collision[4*idx + 3] = sdf_screen;
            }
        }
    }
}




extern "C" __global__ void ray_collection(float blockDim_x, float blockDim_y, float RayDim, float GridDim_x, int max_ray_per_block, unsigned char *ray_screen_collision, unsigned char *Ray_collector, int *Ray_collector_sizes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float radius_ray = 1.0f;

    if (x < blockDim_x && y < blockDim_x && z < RayDim) {

        float x_ray = ray_screen_collision[4*z];
        float y_ray = ray_screen_collision[4*z+1];

        float P_x = fmaxf(x - blockDim_x / 2, fminf(x_ray, x + blockDim_x / 2));
        float P_y = fmaxf(y - blockDim_y / 2, fminf(y_ray, y + blockDim_y / 2));

        float d_x = P_x - x_ray;
        float d_y = P_y - y_ray;

        float dist_block_to_ray = d_x * d_x + d_y * d_y;

        int id = max_ray_per_block*(GridDim_x * y + x);
        int blockId = GridDim_x * y + x;

        if (dist_block_to_ray < radius_ray && Ray_collector_sizes[blockId] + 1 < max_ray_per_block){
            int position = atomicAdd(&Ray_collector_sizes[blockId], 1);
            Ray_collector[id + position] = z;
        }
    }
}

extern "C" __global__ void render(int width, int height, int max_ray_per_block, unsigned char *ray_screen_collision, unsigned char *Ray_collector, int *Ray_collector_sizes, int *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        int id_Block = blockIdx.y * gridDim.x + blockIdx.x;

        for (int i = 0; i < Ray_collector_sizes[id_Block]; i++) {
            
            int id_ray = Ray_collector[id_Block + i];

            float x_ray = ray_screen_collision[4*id_ray];
            float y_ray = ray_screen_collision[4*id_ray + 1];
            float diffusion_coef = ray_screen_collision[4*id_ray + 2];
            // float sdf_screen = ray_screen_collision[4*id_ray + 3];

            float dx = x - x_ray;
            float dy = y - y_ray;

            float D = dx * dx + dy * dy;

            if (D < 2.0) {
                
                int idx = (y * width + x) * 3;

                image[idx] += diffusion_coef;     // Red channel
                image[idx + 1] += diffusion_coef;   // Green channel
                image[idx + 2] += diffusion_coef;   // Blue channel

            }
        }
    }
}

extern "C" __global__ void findMaxValue(const int *image, int width, int height, int *maxVal) {
    extern __shared__ int shared_max[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int local_max = 0;

    // Load elements into shared memory
    for (int i = idx; i < width * height * 3; i += stride) {
        local_max = max(local_max, image[i]);
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = max(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Store the result from this block
    if (threadIdx.x == 0) {
        atomicMax(maxVal, shared_max[0]);
    }
}

extern "C" __global__ void normalizeImage(int *image, int width, int height, int maxVal) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int size = width * height * 3;
    if (idx < size) {
        image[idx] = (image[idx] * 255) / maxVal;
        image[idx] = min(image[idx], 255); // Ensure the value is capped at 255
    }
}


// extern "C" __global__ void camera_reflexion(int num_rays, float width, float height, unsigned char *screen, unsigned char *u_theta, unsigned char *u_phi, unsigned char *u_rho, unsigned char *O, unsigned char *D, unsigned char *T, unsigned char *coordinates) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < num_rays) {

//         float screen_x = screen[0];
//         float screen_y = screen[1];
//         float screen_z = screen[2];

//         float u_rho_x = u_rho[0];
//         float u_rho_y = u_rho[1];
//         float u_rho_z = u_rho[2];

//         float ray_x = O[3*idx] + T[idx] * D[3*idx];
//         float ray_y = O[3*idx + 1] + T[idx] * D[3*idx + 1];
//         float ray_z = O[3*idx + 2] + T[idx] * D[3*idx + 2];

//         float sdf_screen = u_rho_x * (ray_x - screen_x) + u_rho_y * (ray_y - screen_y) + u_rho_z * (ray_z - screen_z);
//         float u_rho_dot_OR = u_rho_x * T[idx] * D[3*idx] + u_rho_y * T[idx] * D[3*idx + 1] + u_rho_z * T[idx] * D[3*idx + 2];

//         if (sdf_screen < 0 && u_rho_dot_OR < - 0.8) {
            
//             float alpha = (1 + (1 - u_rho_x - u_rho_y - u_rho_z) / u_rho_dot_OR) * T[idx];

//             if (alpha > 0) {
//                 float ray_intersect_x = alpha * D[3*idx] + O[3*idx];
//                 float ray_intersect_y = alpha * D[3*idx + 1] + O[3*idx + 1];
//                 float ray_intersect_z = alpha * D[3*idx + 2] + O[3*idx + 2];

//                 float u_theta_x = u_theta[0];
//                 float u_theta_y = u_theta[1];
//                 float u_theta_z = u_theta[2];

//                 float u_phi_x = u_phi[0];
//                 float u_phi_y = u_phi[1];
//                 float u_phi_z = u_phi[2];

//                 float ray_screen_x =  ray_intersect_x * u_phi_x + ray_intersect_y * u_phi_y + ray_intersect_z * u_phi_z + width / 2;
//                 float ray_screen_y =  ray_intersect_x * u_theta_x + ray_intersect_y * u_theta_y + ray_intersect_z * u_theta_z + height / 2;

//                 if ((fabs(ray_screen_x) < width / 2) && (fabs(ray_screen_x) < height / 2)) {
//                     coordinates[2*idx] = ray_screen_x;
//                     coordinates[2*idx + 1] = ray_screen_y;
//                 }
//             }
//         }
//     }
// }