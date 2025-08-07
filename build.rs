use std::{env, fs, path::Path, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/gpu_utils/kernel.cu");

    let out_dir = env::var("OUT_DIR").unwrap();
    let ptx_out = format!("{out_dir}/kernel.ptx");
    let ptx_dst = "src/gpu_utils/kernel.ptx";

    let mem_check_enabled = env::var("CARGO_FEATURE_GPU_MEM_CHECK").is_ok();
    let fast_math_enabled = env::var("CARGO_FEATURE_FAST_MATH").is_ok();

    // Build a Vec<String> so we OWN all strings (no dangling &str).
    let mut nvcc_args: Vec<String> = vec![
        "--ptx".into(),
        "-lineinfo".into(),
        "-Xptxas".into(), "-v".into(),
        "-Xcompiler".into(), "-rdynamic".into(),
        "src/gpu_utils/kernel.cu".into(),
        "-o".into(), ptx_out.clone(),
    ];

    // Optional: PTX ISA (e.g., export CUDA_COMPUTE=compute_86). Safe to omit.
    if let Ok(compute) = env::var("CUDA_COMPUTE") {
        nvcc_args.push("-arch".into());
        nvcc_args.push(compute); // own the string
    }

    if mem_check_enabled {
        // Debug device code for memcheck
        nvcc_args.extend(["-G", "-g", "--device-debug"].into_iter().map(String::from));
    } else if fast_math_enabled {
        nvcc_args.push("--use_fast_math".into());
    }

    let status = Command::new("nvcc")
        .args(&nvcc_args)
        .status()
        .expect("failed to run nvcc");
    assert!(status.success(), "CUDA kernel compilation failed");

    // Copy to where your code expects it
    let dst_dir = Path::new("src/gpu_utils");
    if !dst_dir.exists() {
        fs::create_dir_all(dst_dir).expect("failed to create src/gpu_utils");
    }
    fs::copy(&ptx_out, ptx_dst).expect("failed to copy PTX into src/gpu_utils/kernel.ptx");

    // Still expose the OUT_DIR path if you want it elsewhere
    println!("cargo:rustc-env=KERNEL_PTX={}", ptx_out);
}
