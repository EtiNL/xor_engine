#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// CUDA kernel to compute depth map of a sphere
extern "C" __global__ void computeSDF(int width, int height, float sphereX, float sphereY, float sphereZ, float radius, float theta, float phi, unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        // Adjust x and y to be centered around (0,0) for the screen coordinates
        float screen_x = x - width / 2.0f;
        float screen_y = y - height / 2.0f;

        // Rotate the screen coordinates by theta and phi
        float rotated_x = screen_x * cos(phi) - screen_y * sin(phi);
        float rotated_y = screen_x * sin(phi) + screen_y * cos(phi);
        float screen_z = cos(theta);

        // Compute distance from point on screen to sphere center
        float dx = rotated_x - sphereX;
        float dy = rotated_y - sphereY;
        float dz = screen_z - sphereZ;
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Compute the SDF value and convert to depth map value
        float sdf = distance - radius;

        // Set the color based on depth
        unsigned char depth = static_cast<unsigned char>(255.0f * (sdf + radius) / (2 * radius)); // Normalizing depth value
        image[idx] = depth;        // Red channel
        image[idx + 1] = depth;    // Green channel
        image[idx + 2] = depth;    // Blue channel
    }
}
