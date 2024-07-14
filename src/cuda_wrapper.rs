use cuda_driver_sys::*;
use std::ffi::CString;
use std::ptr::null_mut;

// Define the dim3 structure
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
            // Initialize the CUDA driver
            let result = cuInit(0);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuInit failed: {:?}", result);
                return Err(Box::from("cuInit failed"));
            }

            // Create a context
            let mut context = null_mut();
            let result = cuCtxCreate_v2(&mut context, 0, 0);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuCtxCreate_v2 failed: {:?}", result);
                return Err(Box::from("cuCtxCreate_v2 failed"));
            }

            // Load the module
            let mut module = null_mut();
            let ptx_cstr = CString::new(ptx_path)?;
            let result = cuModuleLoad(&mut module, ptx_cstr.as_ptr());
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuModuleLoad failed: {:?}", result);
                return Err(Box::from("cuModuleLoad failed"));
            }

            // Get the function
            let mut function = null_mut();
            let kernel_name = CString::new("computeDepthMap")?;
            let result = cuModuleGetFunction(&mut function, module, kernel_name.as_ptr());
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuModuleGetFunction failed: {:?}", result);
                return Err(Box::from("cuModuleGetFunction failed"));
            }

            Ok(Self { context, module, function })
        }
    }

    pub fn launch_kernel(&self, width: i32, height: i32, sphere_x: f32, sphere_y: f32, sphere_z: f32, radius: f32, angle: f32, image: &mut [u8]) {
        let grid_dim = dim3 {
            x: ((width + 15) / 16) as u32,
            y: ((height + 15) / 16) as u32,
            z: 1,
        };
        let block_dim = dim3 { x: 16, y: 16, z: 1 };
    
        unsafe {
            let mut d_image: CUdeviceptr = 0;
            let result = cuMemAlloc_v2(&mut d_image, (width * height * 3) as usize);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuMemAlloc_v2 failed: {:?}", result);
                return;
            }
    
            let result = cuMemcpyHtoD_v2(d_image, image.as_ptr() as *const _, (width * height * 3) as usize);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuMemcpyHtoD_v2 failed: {:?}", result);
                cuMemFree_v2(d_image);
                return;
            }
    
            let params = [
                &width as *const _ as *const std::ffi::c_void,
                &height as *const _ as *const std::ffi::c_void,
                &sphere_x as *const _ as *const std::ffi::c_void,
                &sphere_y as *const _ as *const std::ffi::c_void,
                &sphere_z as *const _ as *const std::ffi::c_void,
                &radius as *const _ as *const std::ffi::c_void,
                &angle as *const _ as *const std::ffi::c_void,
                &d_image as *const _ as *const std::ffi::c_void,
            ];
    
            let result = cuLaunchKernel(
                self.function,
                grid_dim.x, grid_dim.y, grid_dim.z,
                block_dim.x, block_dim.y, block_dim.z,
                0, null_mut(), params.as_ptr() as *mut _, null_mut(),
            );
    
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuLaunchKernel failed: {:?}", result);
            }
    
            let result = cuMemcpyDtoH_v2(image.as_mut_ptr() as *mut _, d_image, (width * height * 3) as usize);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("cuMemcpyDtoH_v2 failed: {:?}", result);
            }
    
            cuMemFree_v2(d_image);
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
