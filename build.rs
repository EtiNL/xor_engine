use std::process::Command;

fn main() {
    // Compile the CUDA kernel
    let status = Command::new("nvcc")
        .args(&[
            "--ptx",
            "src/gpu_utils/kernel.cu",
            "-o",
            "src/gpu_utils/kernel.ptx",
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("Failed to compile CUDA kernel");
    }
}
