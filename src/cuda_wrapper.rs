use cuda_driver_sys::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr::null_mut;
use std::error::Error;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub struct CudaContext {
    context: CUcontext,
    module: CUmodule,
    functions: HashMap<String, CUfunction>,
    streams: HashMap<String, CUstream>,
}

pub fn check_cuda_result(result: CUresult, msg: &str) -> Result<(), Box<dyn Error>> {
    if result != cudaError_enum::CUDA_SUCCESS {
        return Err(Box::from(format!("{} failed: {:?}", msg, result)));
    }
    Ok(())
}

impl CudaContext {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn Error>> {
        unsafe {
            check_cuda_result(cuInit(0), "cuInit")?;

            let mut context = null_mut();
            check_cuda_result(cuCtxCreate_v2(&mut context, 0, 0), "cuCtxCreate_v2")?;

            let mut module = null_mut();
            let ptx_cstr = CString::new(ptx_path)?;
            check_cuda_result(cuModuleLoad(&mut module, ptx_cstr.as_ptr()), "cuModuleLoad")?;

            Ok(Self {
                context,
                module,
                functions: HashMap::new(),
                streams: HashMap::new(),
            })
        }
    }

    pub fn load_kernel(&mut self, kernel_name: &str) -> Result<(), Box<dyn Error>> {
        unsafe {
            let mut function = null_mut();
            let kernel_cstr = CString::new(kernel_name)?;
            check_cuda_result(cuModuleGetFunction(&mut function, self.module, kernel_cstr.as_ptr()), "cuModuleGetFunction")?;
            self.functions.insert(kernel_name.to_string(), function);
        }
        Ok(())
    }

    pub fn create_stream(&mut self, stream_name: &str) -> Result<(), Box<dyn Error>> {
        let mut stream = null_mut();
        unsafe {
            check_cuda_result(cuStreamCreate(&mut stream, 0), "cuStreamCreate")?;
        }
        self.streams.insert(stream_name.to_string(), stream);
        Ok(())
    }

    pub fn get_stream(&self, stream_name: &str) -> Result<CUstream, Box<dyn Error>> {
        if let Some(&stream) = self.streams.get(stream_name) {
            Ok(stream)
        } else {
            Err(Box::from(format!("Stream {} not found", stream_name)))
        }
    }

    pub fn launch_kernel(&self,
                        kernel_name: &str,
                        grid_dim: dim3,
                        block_dim: dim3,
                        params: &Vec<*const std::ffi::c_void>,
                        stream_name: &str) -> Result<(), Box<dyn Error>> {
                            
        let stream = self.get_stream(stream_name)?;
        if let Some(&function) = self.functions.get(kernel_name) {
            unsafe {
                check_cuda_result(cuLaunchKernel(
                    function,
                    grid_dim.x, grid_dim.y, grid_dim.z,
                    block_dim.x, block_dim.y, block_dim.z,
                    0, stream, params.as_ptr() as *mut _, null_mut(),
                ), "cuLaunchKernel")?;
            }
        } else {
            return Err(Box::from(format!("Kernel {} not found", kernel_name)));
        }
        Ok(())
    }

    pub fn synchronize(&self, stream_name: &str) -> Result<(), Box<dyn Error>> {
        let stream = self.get_stream(stream_name)?;
        unsafe {
            check_cuda_result(cuStreamSynchronize(stream), "cuStreamSynchronize")?;
        }
        Ok(())
    }

    pub fn allocate_tensor<T>(image: &[T], size: usize) -> Result<CUdeviceptr, Box<dyn Error>> {
        let mut d_ptr: CUdeviceptr = 0;
        unsafe {
            check_cuda_result(cuMemAlloc_v2(&mut d_ptr, size), "cuMemAlloc_v2")?;
            check_cuda_result(cuMemcpyHtoD_v2(d_ptr, image.as_ptr() as *const _, size), "cuMemcpyHtoD_v2")?;
        }
        Ok(d_ptr)
    }

    pub fn free_tensor(d_ptr: CUdeviceptr) -> Result<(), Box<dyn Error>> {
        unsafe {
            check_cuda_result(cuMemFree_v2(d_ptr), "cuMemFree_v2")?;
        }
        Ok(())
    }

    pub fn retrieve_tensor<T>(&self, d_ptr: CUdeviceptr, image: &mut [T], size: usize) -> Result<(), Box<dyn Error>> {
        unsafe {
            check_cuda_result(cuMemcpyDtoH_v2(image.as_mut_ptr() as *mut _, d_ptr, size), "cuMemcpyDtoH_v2")?;
        }
        Ok(())
    }
}


impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            for &stream in self.streams.values() {
                cuStreamDestroy_v2(stream);
            }
            cuModuleUnload(self.module);
            cuCtxDestroy_v2(self.context);
        }
    }
}
