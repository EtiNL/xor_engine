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


extern "C" __global__ void produit_scalaire(int width, int height, unsigned char *A, unsigned char *B, unsigned char *result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        result[idx] = A[3*idx] * B[3*idx] + A[3*idx + 1] * B[3*idx + 1] + A[3*idx + 2] * B[3*idx + 2];
    }
}



extern "C" __global__ void sdf_sphere(int width, int height, float sphereX, float sphereY, float sphereZ, float radius, unsigned char *O, unsigned char *D, unsigned char *T, unsigned char *result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
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

extern "C" __global__ void grad_sdf_sphere(int width, int height, float sphereX, float sphereY, float sphereZ, float radius, unsigned char *O, unsigned char *D, unsigned char *T, unsigned char *result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        result[idx] = (O[idx] + T[idx] * D[3*idx] - sphereX) / radius;
        result[idx + 1] = (O[idx + 1] + T[idx] * D[idx + 1] - sphereY) / radius;
        result[idx + 2] = (O[idx + 2] + T[idx] * D[idx + 2] - sphereZ) / radius;
    }
}


extern "C" __global__ void newton_march(int width, int height, float step, float epsilon_grad, float epsilon_dist, unsigned char *Grad_sdf_dot_d, unsigned char *Sdf, unsigned char *Sdf_step, unsigned char *T) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float grad = Grad_sdf_dot_d[idx];
        float sdf = Sdf[idx];
        float sdf_step = Sdf_step[idx];

        float grad_step = -sdf / grad;

        if (grad_step < epsilon_grad) {
            T[idx] = grad_step;
        } else {
            if (sdf > 0 && sdf_step < 0) {
                T[idx] = step / 2;
            } else if (sdf < 0) {
                T[idx] = step;
            }
        }
    }
}



extern "C" __global__ void screen_sdf(int width, int height, float x_screen, float y_screen, float z_screen, float theta, float phi, unsigned char *O, unsigned char *D, unsigned char *T) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        float screen_x = sinf(theta) * cosf(phi) + (x - width / 2.0f);
        float screen_y = sinf(theta) * sinf(phi) + (y - height / 2.0f);
        float screen_z = cosf(theta);

    }
}