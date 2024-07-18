use cuda_driver_sys::*;
use std::ffi::CString;
use std::ptr::null_mut;
use std::error::Error;

#[repr(C)]
pub struct dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub struct CudaContext {
    context: CUcontext,
    module: CUmodule,
    function: CUfunction,
}

pub fn check_cuda_result(result: CUresult, msg: &str) -> Result<(), Box<dyn Error>> {
    if result != cudaError_enum::CUDA_SUCCESS {
        return Err(Box::from(format!("{} failed: {:?}", msg, result)));
    }
    Ok(())
}

impl CudaContext {
    pub fn new(ptx_path: &str, kernel_name: &str) -> Result<Self, Box<dyn Error>> {
        unsafe {
            check_cuda_result(cuInit(0), "cuInit")?;

            let mut context = null_mut();
            check_cuda_result(cuCtxCreate_v2(&mut context, 0, 0), "cuCtxCreate_v2")?;

            let mut module = null_mut();
            let ptx_cstr = CString::new(ptx_path)?;
            check_cuda_result(cuModuleLoad(&mut module, ptx_cstr.as_ptr()), "cuModuleLoad")?;

            let mut function = null_mut();
            let kernel_cstr = CString::new(kernel_name)?;
            check_cuda_result(cuModuleGetFunction(&mut function, module, kernel_cstr.as_ptr()), "cuModuleGetFunction")?;

            Ok(Self { context, module, function })
        }
    }

    pub fn launch_kernel(
        &self,
        grid_dim: dim3,
        block_dim: dim3,
        params: Vec<*const std::ffi::c_void>
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            check_cuda_result(cuLaunchKernel(
                self.function,
                grid_dim.x, grid_dim.y, grid_dim.z,
                block_dim.x, block_dim.y, block_dim.z,
                0, null_mut(), params.as_ptr() as *mut _, null_mut(),
            ), "cuLaunchKernel")?;

            check_cuda_result(cuCtxSynchronize(), "cuCtxSynchronize")?;
        }

        Ok(())
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.module);
            cuCtxDestroy_v2(self.context);
        }
    }
}
