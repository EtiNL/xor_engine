use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/gpu_utils/kernel.cu");

    let out_dir  = env::var("OUT_DIR").unwrap();
    let ptx_path = format!("{out_dir}/kernel.ptx");

    // Enable this with:  cargo build --features gpu-mem-check
    let mem_check_enabled = env::var("CARGO_FEATURE_GPU_MEM_CHECK").is_ok();

    // ── Base flags ──────────────────────────────────────────────────────────────
    //   • --ptx             → output PTX so Rust can JIT it at runtime
    //   • -lineinfo         → emit .loc directives (PC ↔︎ line mapping)
    //   • -Xcompiler -rdynamic
    //     keeps host symbols so the host part of the back-trace is readable.
    let mut nvcc_args = vec![
        "--ptx",
        "-lineinfo",
        "-Xcompiler", "-rdynamic",
        "src/gpu_utils/kernel.cu",
        "-o", &ptx_path,
    ];

    // ── Extra flags when the ‘gpu-mem-check’ feature is ON ─────────────────────
    if mem_check_enabled {
        // -G      : compile device code in debug mode (no opt, full DWARF)
        // -g      : emit host-side debug info as well (helps cuda-gdb)
        // --device-debug : keeps debug sections in the fatbin that JIT generates
        nvcc_args.splice(1..1, ["-G", "-g", "--device-debug"]);
    }

    // ── Kick off NVCC ──────────────────────────────────────────────────────────
    let status = Command::new("nvcc")
        .args(&nvcc_args)
        .status()
        .expect("failed to run nvcc");
    assert!(status.success(), "CUDA kernel compilation failed");

    // Expose PTX path to the Rust crate so it can be loaded at runtime.
    println!("cargo:rustc-env=KERNEL_PTX={ptx_path}");
}
