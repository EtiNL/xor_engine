use cuda_driver_sys::{
    cuCtxCreate_v2, cuInit, cuLaunchKernel, cuMemAlloc_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
    cuModuleGetFunction, cuModuleLoad, cuCtxDestroy_v2, cuModuleUnload, cuMemFree_v2,
    CUdeviceptr, CUcontext, CUfunction, CUmodule, CUresult, cudaError_enum::CUDA_SUCCESS,
};
use std::any::Any;
use std::ffi::CString;
use std::ptr::null_mut;
use std::mem;

#[derive(Debug)]
#[repr(C)]
struct dim3 {
    x: u32,
    y: u32,
    z: u32,
}

fn check_cuda_result(result: CUresult, msg: &str) -> Result<(), Box<dyn std::error::Error>> {
    if result != CUDA_SUCCESS {
        return Err(Box::from(format!("{} failed: {:?}", msg, result)));
    }
    Ok(())
}

pub struct CudaContext {
    context: CUcontext,
    module: CUmodule,
    function: CUfunction,
}

impl CudaContext {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            check_cuda_result(cuInit(0), "cuInit")?;

            let mut context = null_mut();
            check_cuda_result(cuCtxCreate_v2(&mut context, 0, 0), "cuCtxCreate_v2")?;

            let mut module = null_mut();
            let ptx_cstr = CString::new(ptx_path)?;
            check_cuda_result(cuModuleLoad(&mut module, ptx_cstr.as_ptr()), "cuModuleLoad")?;

            let mut function = null_mut();
            let kernel_name = CString::new("computeSDF")?;
            check_cuda_result(cuModuleGetFunction(&mut function, module, kernel_name.as_ptr()), "cuModuleGetFunction")?;

            Ok(Self { context, module, function })
        }
    }

    pub fn launch_kernel(&self, args: &mut [Box<dyn KernelArg>], width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        let grid_dim = dim3 { x: (width + 15) / 16, y: (height + 15) / 16, z: 1 };
        let block_dim = dim3 { x: 16, y: 16, z: 1 };

        unsafe {
            let mut device_ptrs: Vec<CUdeviceptr> = Vec::new();
            for arg in args.iter_mut() {
                arg.allocate_on_device();
                device_ptrs.push(arg.device_ptr());
            }

            let mut params: Vec<*mut std::ffi::c_void> = Vec::new();
            for ptr in &device_ptrs {
                params.push(ptr as *const CUdeviceptr as *mut std::ffi::c_void);
            }

            // println!("Launching kernel with grid_dim: {:?}, block_dim: {:?}", grid_dim, block_dim);
            // for (i, param) in params.iter().enumerate() {
                // println!("Param {}: {:?}", i, param);
            // }

            // println!("finished loading params");

            let result = cuLaunchKernel(
                self.function,
                grid_dim.x, grid_dim.y, grid_dim.z,
                block_dim.x, block_dim.y, block_dim.z,
                0, null_mut(), params.as_mut_ptr(), null_mut(),
            );

            if result != CUDA_SUCCESS {
                eprintln!("cuLaunchKernel failed: {:?}", result);
                return Err(Box::from(format!("cuLaunchKernel failed: {:?}", result)));
            }

            for arg in args.iter_mut() {
                arg.copy_to_host();
            }
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

pub trait KernelArg: Any {
    fn allocate_on_device(&mut self);
    fn device_ptr(&self) -> CUdeviceptr;
    fn copy_to_host(&mut self);
    fn as_any(&self) -> &dyn Any;
}

pub struct DeviceBuffer<T> {
    host_data: Vec<T>,
    device_ptr: CUdeviceptr,
}

impl<T> DeviceBuffer<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self {
            host_data: data,
            device_ptr: 0,
        }
    }

    pub fn get_host_data(&self) -> &Vec<T> {
        &self.host_data
    }

    pub fn get_host_data_mut(&mut self) -> &mut Vec<T> {
        &mut self.host_data
    }
}

impl<T: 'static> KernelArg for DeviceBuffer<T> where T: Copy + 'static {
    fn allocate_on_device(&mut self) {
        let size = self.host_data.len() * mem::size_of::<T>();
        unsafe {
            let result = cuMemAlloc_v2(&mut self.device_ptr, size);
            check_cuda_result(result, "cuMemAlloc_v2").expect("Failed to allocate device memory");

            let result = cuMemcpyHtoD_v2(self.device_ptr, self.host_data.as_ptr() as *const _, size);
            check_cuda_result(result, "cuMemcpyHtoD_v2").expect("Failed to copy memory to device");
        }
    }

    fn device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    fn copy_to_host(&mut self) {
        let size = self.host_data.len() * mem::size_of::<T>();
        unsafe {
            let result = cuMemcpyDtoH_v2(self.host_data.as_mut_ptr() as *mut _, self.device_ptr, size);
            check_cuda_result(result, "cuMemcpyDtoH_v2").expect("Failed to copy memory from device");
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            cuMemFree_v2(self.device_ptr);
        }
    }
}
