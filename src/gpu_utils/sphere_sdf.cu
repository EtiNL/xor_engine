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
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        // Simplified kernel: Just set a fixed value
        image[idx] = 128;        // Red channel
        image[idx + 1] = 128;    // Green channel
        image[idx + 2] = 128;    // Blue channel
    }
}
