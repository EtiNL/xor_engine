#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// CUDA kernel to compute depth map of a sphere
extern "C" __global__ void computeDepthMap(int width, int height, float sphereX, float sphereY, float sphereZ, float radius, float theta, float phi, unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        // Normalized device coordinates
        float ndc_x = (2.0f * x / width) - 1.0f;
        float ndc_y = (2.0f * y / height) - 1.0f;

        // Apply rotation based on angle
        float rad = theta * M_PI / 180.0f;
        float cos_a = cosf(rad);
        float sin_a = sinf(rad);
        float rotated_x = ndc_x * cos_a - ndc_y * sin_a;
        float rotated_y = ndc_x * sin_a + ndc_y * cos_a;

        // Assuming a simple orthographic projection
        float screen_x = rotated_x * 10.0f; // Scale to screen size
        float screen_y = rotated_y * 10.0f; // Scale to screen size
        float screen_z = 0.0f;

        // Compute distance from point on screen to sphere center
        float dx = screen_x - sphereX;
        float dy = screen_y - sphereY;
        float dz = screen_z - sphereZ;
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Compute the SDF value and convert to depth map value
        float sdf = distance - radius;
        float depth = fmaxf(0.0f, radius - fabs(sdf));

        // Map depth to grayscale value (0-255)
        unsigned char grayscale = static_cast<unsigned char>(depth / radius * 255.0f);

        // Set the color based on depth
        image[idx] = grayscale;        // Red channel
        image[idx + 1] = grayscale;    // Green channel
        image[idx + 2] = grayscale;    // Blue channel
    }
}
