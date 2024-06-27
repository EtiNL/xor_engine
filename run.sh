#!/bin/bash

# Compile the CUDA kernel
nvcc -ptx -o src/gpu_utils/kernel.ptx src/gpu_utils/kernel.cu

# Run the Rust application
cargo run
