use cuda_driver_sys::*;
use std::any::Any;
use std::ffi::CString;
use std::ptr::null_mut;
use std::mem;

#[repr(C)]
struct dim3 {
    x: u32,
    y: u32,
    z: u32,
}

pub struct CudaContext {
    context: CUcontext,
    module: CUmodule,
    function: CUfunction,
}

impl CudaContext {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let result = cuInit(0);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuInit failed: {:?}", result);
                return Err(Box::from("cuInit failed"));
            }

            let mut context = null_mut();
            let result = cuCtxCreate_v2(&mut context, 0, 0);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuCtxCreate_v2 failed: {:?}", result);
                return Err(Box::from("cuCtxCreate_v2 failed"));
            }

            let mut module = null_mut();
            let ptx_cstr = CString::new(ptx_path)?;
            let result = cuModuleLoad(&mut module, ptx_cstr.as_ptr());
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuModuleLoad failed: {:?}", result);
                return Err(Box::from("cuModuleLoad failed"));
            }

            let mut function = null_mut();
            let kernel_name = CString::new("computeSDF")?;
            let result = cuModuleGetFunction(&mut function, module, kernel_name.as_ptr());
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuModuleGetFunction failed: {:?}", result);
                return Err(Box::from("cuModuleGetFunction failed"));
            }

            Ok(Self { context, module, function })
        }
    }

    pub fn launch_kernel(&self, args: &mut [Box<dyn KernelArg>]) {
        let grid_dim = dim3 { x: 16, y: 16, z: 1 }; // Example values, adapt as needed
        let block_dim = dim3 { x: 16, y: 16, z: 1 }; // Example values, adapt as needed

        unsafe {
            let mut device_ptrs: Vec<CUdeviceptr> = Vec::new();
            for arg in args.iter_mut() {
                arg.allocate_on_device();
                device_ptrs.push(arg.device_ptr());
            }

            let params: Vec<*const std::ffi::c_void> = device_ptrs.iter()
                .map(|&ptr| &ptr as *const CUdeviceptr as *const std::ffi::c_void)
                .collect();

            let result = cuLaunchKernel(
                self.function,
                grid_dim.x, grid_dim.y, grid_dim.z,
                block_dim.x, block_dim.y, block_dim.z,
                0, null_mut(), params.as_ptr() as *mut _, null_mut(),
            );

            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuLaunchKernel failed: {:?}", result);
            }

            for arg in args.iter_mut() {
                arg.copy_to_host();
            }
        }
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
}

impl<T> KernelArg for DeviceBuffer<T> where T: Copy + 'static {
    fn allocate_on_device(&mut self) {
        let size = self.host_data.len() * mem::size_of::<T>();
        unsafe {
            cuMemAlloc_v2(&mut self.device_ptr, size);
            cuMemcpyHtoD_v2(self.device_ptr, self.host_data.as_ptr() as *const _, size);
        }
    }

    fn device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    fn copy_to_host(&mut self) {
        let size = self.host_data.len() * mem::size_of::<T>();
        unsafe {
            cuMemcpyDtoH_v2(self.host_data.as_mut_ptr() as *mut _, self.device_ptr, size);
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
