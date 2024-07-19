#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// CUDA kernel to compute depth map of a sphere
extern "C" __global__ void computeDepthMap(int width, int height, float sphereX, float sphereY, float sphereZ, float radius, float theta, float phi, unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        float screen_x = sinf(theta) * cosf(phi) + (x - width / 2.0f);
        float screen_y = sinf(theta) * sinf(phi) + (y - height / 2.0f);
        float screen_z = cosf(theta);

        // Compute distance from point on screen to sphere center
        float dx = screen_x - sphereX;
        float dy = screen_y - sphereY;
        float dz = screen_z - sphereZ;
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Compute the SDF value and convert to depth map value
        float sdf = distance - radius;

        // Map depth to grayscale value (0-255)
        unsigned char grayscale = static_cast<unsigned char>(sdf/400.0 * 255.0f);

        // Set the color based on depth
        image[idx] = grayscale;        // Red channel
        image[idx + 1] = grayscale;    // Green channel
        image[idx + 2] = grayscale;    // Blue channel
    }
}
