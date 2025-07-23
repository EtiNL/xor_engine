<<<<<<< HEAD
use cuda_driver_sys::{
    cuCtxCreate_v2, cuCtxSynchronize, cuInit, cuLaunchKernel, cuMemAlloc_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
    cuModuleGetFunction, cuModuleLoad, cuCtxDestroy_v2, cuModuleUnload, cuMemFree_v2,
    CUdeviceptr, CUcontext, CUfunction, CUmodule, CUresult, cudaError_enum::CUDA_SUCCESS,
};
use std::any::Any;
use std::ffi::CString;
use std::ptr::null_mut;
use std::mem;

#[derive(Debug)]
=======
use cuda_driver_sys::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr::null_mut;
use std::error::Error;
use std::ffi::c_void;
use crate::scene_composition::SdfObject;

>>>>>>> debbug_branch
#[repr(C)]
#[derive(Clone, Copy)]
pub struct dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
pub struct cudaGraphKernelNodeParams {
    pub func: CUfunction,
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
    pub block_dim_z: u32,
    pub shared_mem_bytes: usize,
    pub kernel_params: *mut *mut c_void,
    pub extra: *mut *mut c_void,
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

<<<<<<< HEAD
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

            println!("Launching kernel with grid_dim: {:?}, block_dim: {:?}", grid_dim, block_dim);
            for (i, param) in params.iter().enumerate() {
                println!("Param {}: {:?}", i, param);
            }

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

            // Ensure kernel execution is complete
            let sync_result = cuCtxSynchronize();
            if sync_result != CUDA_SUCCESS {
                eprintln!("cuCtxSynchronize failed: {:?}", sync_result);
                return Err(Box::from(format!("cuCtxSynchronize failed: {:?}", sync_result)));
            }

            for arg in args.iter_mut() {
                arg.copy_to_host();
            }
        }
        Ok(())
    }
=======
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
            check_cuda_result(cuStreamCreate(&mut stream, 0x01), "cuStreamCreate")?; // 0x01 = cudaStreamNonBlocking
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

    pub fn allocate_struct<T>(object: &T) -> Result<CUdeviceptr, Box<dyn Error>> {
        let mut d_ptr: CUdeviceptr = 0;
        let size = std::mem::size_of::<T>();
        unsafe {
            check_cuda_result(cuMemAlloc_v2(&mut d_ptr, size), "cuMemAlloc struct")?;
            check_cuda_result(
                cuMemcpyHtoD_v2(d_ptr, object as *const _ as *const c_void, size),
                "cuMemcpyHtoD struct"
            )?;
        }
        Ok(d_ptr)
    }

    pub fn allocate_curand_states(width: u32, height: u32) -> Result<CUdeviceptr, Box<dyn Error>> {
        let count = (width * height) as usize;
        let state_size = 48; // Mesuré depuis device
        let total_bytes = count * state_size;
    
        let mut device_ptr: CUdeviceptr = 0;
        unsafe {
            check_cuda_result(cuMemAlloc_v2(&mut device_ptr, total_bytes), "alloc curand states")?;
        }
    
        Ok(device_ptr)
    }

    pub fn allocate_scene(scene: &[SdfObject]) -> Result<CUdeviceptr, Box<dyn Error>> {
        let size = scene.len() * std::mem::size_of::<SdfObject>();
        let mut d_ptr: CUdeviceptr = 0;
        unsafe {
            check_cuda_result(cuMemAlloc_v2(&mut d_ptr, size), "cuMemAlloc scene")?;
            check_cuda_result(
                cuMemcpyHtoD_v2(d_ptr, scene.as_ptr() as *const c_void, size),
                "cuMemcpyHtoD scene"
            )?;
        }
        Ok(d_ptr)
    }

    pub fn copy_host_to_device<T>(&self, dst: CUdeviceptr, src: &[T]) -> Result<(), Box<dyn Error>> {
        let size = src.len() * std::mem::size_of::<T>();
        unsafe { check_cuda_result(cuMemcpyHtoD_v2(dst, src.as_ptr() as *const c_void, size), "copy (cuMemcpyHtoD_v2)")?; }
        Ok(())
    }

    pub fn free_device_memory(d_ptr: CUdeviceptr) -> Result<(), Box<dyn Error>> {
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

    pub fn create_cuda_graph(&self) -> Result<CUgraph, Box<dyn Error>> {
        let mut graph = null_mut();
        unsafe {
            check_cuda_result(cuGraphCreate(&mut graph, 0), "cuGraphCreate")?;
        }
        Ok(graph)
    }

    pub fn add_graph_kernel_node(
        &self,
        graph: &mut CUgraph,
        kernel_name: &str,
        params: &[*const c_void],
        grid_dim: dim3,
        block_dim: dim3,
    ) -> Result<(), Box<dyn Error>> {
        let function = self.functions.get(kernel_name)
            .ok_or(format!("Kernel {} not found", kernel_name))?;
        
        let mut kernel_node_params = cudaGraphKernelNodeParams {
            func: *function,
            grid_dim_x: grid_dim.x,
            grid_dim_y: grid_dim.y,
            grid_dim_z: grid_dim.z,
            block_dim_x: block_dim.x,
            block_dim_y: block_dim.y,
            block_dim_z: block_dim.z,
            shared_mem_bytes: 0,
            kernel_params: params.as_ptr() as *mut _,
            extra: null_mut(),
        };

        let mut kernel_node = null_mut();
        unsafe {
            check_cuda_result(cuGraphAddKernelNode(
                &mut kernel_node,
                *graph,
                null_mut(),
                0,
                &mut kernel_node_params as *mut cudaGraphKernelNodeParams as *const CUDA_KERNEL_NODE_PARAMS_st
            ), "cuGraphAddKernelNode")?;
        }
        Ok(())
    }

    pub fn instantiate_graph(&self, graph: CUgraph) -> Result<CUgraphExec, Box<dyn Error>> {
        let mut graph_exec = null_mut();
        unsafe {
            check_cuda_result(cuGraphInstantiate(&mut graph_exec, graph, null_mut(), null_mut(), 0), "cuGraphInstantiate")?;
        }
        Ok(graph_exec)
    }

    pub fn launch_graph(&self, graph_exec: CUgraphExec) -> Result<(), Box<dyn Error>> {
        let stream = self.get_stream("stream1")?;  // Use any valid stream to launch the graph
        unsafe {
            check_cuda_result(cuGraphLaunch(graph_exec, stream), "cuGraphLaunch")?;
        }
        Ok(())
    }

    pub fn free_graph(&self, graph: CUgraph) -> Result<(), Box<dyn Error>> {
        unsafe {
            check_cuda_result(cuGraphDestroy(graph), "cuGraphDestroy")?;
        }
        Ok(())
    }
    
    pub fn free_graph_exec(&self, graph_exec: CUgraphExec) -> Result<(), Box<dyn Error>> {
        unsafe {
            check_cuda_result(cuGraphExecDestroy(graph_exec), "cuGraphExecDestroy")?;
        }
        Ok(())
    }
>>>>>>> debbug_branch
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

<<<<<<< HEAD
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

            println!("Allocated {} bytes on device at {:?}", size, self.device_ptr);
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

            println!("Copied {} bytes from device at {:?}", size, self.device_ptr);
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
=======
// Gère le buffer GPU qui contient la scène.
pub struct SceneBuffer {
    d_ptr:    CUdeviceptr,
    capacity: usize,            // Nombre maximum de SdfObject que le buffer peut contenir
}

impl SceneBuffer {
    pub fn new(capacity: usize) -> Result<Self, Box<dyn Error>> {
        // alloue capacity objets, zéro au départ
        let dummy: Vec<SdfObject> = vec![SdfObject::default(); capacity];
        let d_ptr = CudaContext::allocate_scene(&dummy)?;
        Ok(Self { d_ptr, capacity })
    }

    /// s’assure que `capacity >= needed`; réalloue si nécessaire
    fn ensure_capacity(
        &mut self,
        needed: usize,
    ) -> Result<(), Box<dyn Error>> {
        if needed <= self.capacity { return Ok(()); }

        // libère l’ancien
        CudaContext::free_device_memory(self.d_ptr)?;

        // nouvelle taille = puissance de 2 supérieure → évite de réallouer à chaque frame
        self.capacity = needed.next_power_of_two();
        let dummy: Vec<SdfObject> = vec![SdfObject::default(); self.capacity];
        self.d_ptr = CudaContext::allocate_scene(&dummy)?;
        Ok(())
    }

    pub fn upload(
        &mut self,
        ctx: &CudaContext,
        objects: &[SdfObject],
    ) -> Result<(), Box<dyn Error>> {
        self.ensure_capacity(objects.len());
        ctx.copy_host_to_device(self.d_ptr, objects)
    }

    pub fn ptr(&self) -> CUdeviceptr { self.d_ptr }

}

impl Drop for SceneBuffer {
    fn drop(&mut self) {                                      // libère proprement
        let _ = CudaContext::free_device_memory(self.d_ptr);
    }
}
>>>>>>> debbug_branch
