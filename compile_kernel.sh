#!/bin/bash

# Script to compile CUDA kernel

# Check if nvcc is available
if ! command -v nvcc &> /dev/null
then
    echo "nvcc could not be found. Make sure CUDA toolkit is installed."
    exit 1
fi

# Compile the CUDA kernel
nvcc -ptx -o src/gpu_utils/kernel.ptx src/gpu_utils/kernel.cu

if [ $? -eq 0 ]; then
    echo "Kernel compilation successful."
else
    echo "Kernel compilation failed."
    exit 1
fi
