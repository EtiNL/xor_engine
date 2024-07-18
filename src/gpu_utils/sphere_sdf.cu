#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(1);                                                            \
    } }

extern "C" __global__ void computeSDF(int width, int height, float sphereX, float sphereY, float sphereZ, float radius, float theta, float phi, unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        if (idx >= 0 && idx < width * height * 3) {
            // Debug output to verify kernel execution
            printf("Pixel (%d, %d): idx %d\n", x, y, idx);

            image[idx] = 255;        // Red channel
            image[idx + 1] = 255;    // Green channel
            image[idx + 2] = 255;    // Blue channel
        }
    }
}