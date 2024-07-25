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


extern "C" __global__ void newton_march(int num_rays, float epsilon_grad, unsigned char *Grad_sdf_dot_d, unsigned char *Sdf, unsigned char *Sdf_i, unsigned char *T, unsigned char *T_i, unsigned char *T_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {

        float grad = Grad_sdf_dot_d[idx];
        float sdf = Sdf[idx];
        float sdf_i = Sdf_i[idx];
        float t = T[idx];

        float grad_step = - sdf / grad;

        if (grad_step > epsilon_grad) {
            T[idx] = t + grad_step;
        } else {
            if ((sdf > 0 && sdf_i < 0) || (sdf < 0 && sdf_i > 0)) {
                T_f[idx] = t;
            } else {
                T_i[idx] = t;
            }
            T[idx] = (T_f[idx] + T_i[idx]) / 2;
        }
    }
}



extern "C" __global__ void camera(int num_rays, float width, float height, unsigned char *screen, unsigned char *u_theta, unsigned char *u_phi, unsigned char *u_rho, unsigned char *O, unsigned char *D, unsigned char *T, unsigned char *coordinates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rays) {

        float screen_x = screen[0];
        float screen_y = screen[1];
        float screen_z = screen[2];

        float u_rho_x = u_rho[0];
        float u_rho_y = u_rho[1];
        float u_rho_z = u_rho[2];

        float ray_x = O[3*idx] + T[idx] * D[3*idx];
        float ray_y = O[3*idx + 1] + T[idx] * D[3*idx + 1];
        float ray_z = O[3*idx + 2] + T[idx] * D[3*idx + 2];

        float sdf_screen = u_rho_x * (ray_x - screen_x) + u_rho_y * (ray_y - screen_y) + u_rho_z * (ray_z - screen_z);
        float u_rho_dot_OR = u_rho_x * T[idx] * D[3*idx] + u_rho_y * T[idx] * D[3*idx + 1] + u_rho_z * T[idx] * D[3*idx + 2];

        if (sdf_screen < 0 && u_rho_dot_OR < - 0.8) {
            
            float alpha = (1 + (1 - u_rho_x - u_rho_y - u_rho_z) / u_rho_dot_OR) * T[idx];

            if (alpha > 0) {
                float ray_intersect_x = alpha * D[3*idx] + O[3*idx];
                float ray_intersect_y = alpha * D[3*idx + 1] + O[3*idx + 1];
                float ray_intersect_z = alpha * D[3*idx + 2] + O[3*idx + 2];

                float u_theta_x = u_theta[0];
                float u_theta_y = u_theta[1];
                float u_theta_z = u_theta[2];

                float u_phi_x = u_phi[0];
                float u_phi_y = u_phi[1];
                float u_phi_z = u_phi[2];

                float ray_screen_x =  ray_intersect_x * u_phi_x + ray_intersect_y * u_phi_y + ray_intersect_z * u_phi_z + width / 2;
                float ray_screen_y =  ray_intersect_x * u_theta_x + ray_intersect_y * u_theta_y + ray_intersect_z * u_theta_z + height / 2;

                if ((fabs(ray_screen_x) < width / 2) && (fabs(ray_screen_x) < height / 2)) {
                    coordinates[2*idx] = ray_screen_x;
                    coordinates[2*idx + 1] = ray_screen_y;
                }
            }
        }
    }
}

extern "C" __global__ void render(int num_rays, float width, float height, unsigned char *coordinates, unsigned char *color_rays, unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float radius_ray = 1.0f;

    if (x < width && y < height) {
        int pixel_index = 3 * (y * (int)width + x);
        float red = 0.0f, green = 0.0f, blue = 0.0f;
        int counter = 0;

        for (int idx = 0; idx < num_rays; idx++) {
            float x_ray = coordinates[2 * idx];
            float y_ray = coordinates[2 * idx + 1];

            float dx = x_ray - (float)x;
            float dy = y_ray - (float)y;

            if (dx * dx + dy * dy <= radius_ray * radius_ray) {
                red += color_rays[3 * idx];
                green += color_rays[3 * idx + 1];
                blue += color_rays[3 * idx + 2];
                counter++;
            }
        }

        if (counter > 0) {
            red /= counter;
            green /= counter;
            blue /= counter;
        }

        image[pixel_index] = (unsigned char)red;
        image[pixel_index + 1] = (unsigned char)green;
        image[pixel_index + 2] = (unsigned char)blue;
    }
}