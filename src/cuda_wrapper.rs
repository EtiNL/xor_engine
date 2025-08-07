use cuda_driver_sys::*;
use std::{collections::HashMap, ffi::{c_void, CString}, ptr::null_mut, error::Error};
use memoffset::offset_of;
use std::mem::size_of;
use std::marker::PhantomData;

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

pub struct CudaContext {
    context: CUcontext,
    module: CUmodule,
    functions: HashMap<String, CUfunction>,
    streams: HashMap<String, CUstream>,
    kernel_nodes: HashMap<String, CUgraphNode>,
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
                kernel_nodes: HashMap::new(),
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

    pub fn synchronize_stream(&self, stream_name: &str) -> Result<(), Box<dyn Error>> {
        let stream = self.get_stream(stream_name)?;
        unsafe {
            check_cuda_result(cuStreamSynchronize(stream), "cuStreamSynchronize")?;
        }
        Ok(())
    }

    pub fn synchronize() -> Result<(), Box<dyn Error>> {
        unsafe {
            check_cuda_result(cuCtxSynchronize(), "cuCtxSynchronize")?;
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

    pub fn copy_host_to_device<T>(dst: CUdeviceptr, src: &[T]) -> Result<(), Box<dyn Error>> {
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
        &mut self,
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
        
        self.kernel_nodes.insert(kernel_name.to_owned(), kernel_node);
        
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

    pub fn exec_kernel_node_set_params(
        &self,
        graph_exec: CUgraphExec,
        node_name:  &str,
        new_params: &[*const c_void],
    ) -> Result<(), Box<dyn Error>> {
        // look up the node we saved earlier
        let node = *self.kernel_nodes
                        .get(node_name)
                        .ok_or(format!("node {} not found", node_name))?;

        // fetch the current params so we keep gridDim, blockDim, etc. unchanged
        let mut p: CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        unsafe { check_cuda_result(cuGraphKernelNodeGetParams(node, &mut p),
                                   "cuGraphKernelNodeGetParams")?; }

        // patch only the kernelParams pointer array
        p.kernelParams = new_params.as_ptr() as *mut *mut c_void;

        unsafe {
            check_cuda_result(
                cuGraphExecKernelNodeSetParams(graph_exec, node, &p),
                "cuGraphExecKernelNodeSetParams"
            )?;
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

pub struct GpuBuffer<T> {
    d_ptr: CUdeviceptr,
    pub capacity: usize, // how many T elements allocated
    pub len: usize,
    _marker: PhantomData<T>,
}

impl<T> GpuBuffer<T> {
    pub fn new(initial_capacity: usize) -> Result<Self, Box<dyn Error>> {
        let mut d_ptr: CUdeviceptr = 0;
        let size_bytes = (initial_capacity * size_of::<T>()) as usize;

        unsafe {
        check_cuda_result(cuMemAlloc_v2(&mut d_ptr, size_bytes), "cuMemAlloc_v2 GpuBuffer.new")?;
        }

        Ok(Self {
            d_ptr,
            capacity: initial_capacity,
            len: 0,
            _marker: PhantomData,
        })
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.d_ptr
    }

    pub fn ensure_capacity(&mut self, needed: usize) -> Result<(), Box<dyn Error>> {
        if needed <= self.capacity {
            return Ok(());
        }
        let new_capacity = needed.next_power_of_two();
        let new_size = (new_capacity * size_of::<T>()) as usize;
        let mut new_ptr: CUdeviceptr = 0;
        unsafe {
            check_cuda_result(cuMemAlloc_v2(&mut new_ptr, new_size), "cuMemAlloc_v2 GpuBuffer::ensure_capacity")?;
            // copy existing data device->device
            let old_size = (self.capacity * size_of::<T>()) as usize;
            check_cuda_result(cuMemcpyDtoD_v2(new_ptr, self.d_ptr, old_size), "cuMemcpyDtoD_v2 GpuBuffer::ensure_capacity")?;
            // free old
            check_cuda_result(cuMemFree_v2(self.d_ptr), "cuMemFree_v2 GpuBuffer::ensure_capacity")?;
        }
        self.d_ptr = new_ptr;
        self.capacity = new_capacity;
        Ok(())
    }

    pub fn upload_all(&mut self, slice: &[T]) -> Result<(), Box<dyn Error>> {
        let required = slice.len();
        self.ensure_capacity(required)?;
        let bytes = required * size_of::<T>();
        unsafe {
            check_cuda_result( cuMemcpyHtoD_v2(self.d_ptr, slice.as_ptr() as *const _, bytes), "cuMemcpyHtoD_v2 GpuBuffer::upload_all")?;
        }
        self.len = required;
        Ok(())
    }

    fn update_element(&self, index: usize, elem: &T) -> Result<(), Box<dyn Error>> {
        if index >= self.capacity {
            return Err(Box::from("Index out of bounds in GpuBuffer::update_element"));
        }
        let offset = (index * size_of::<T>()) as usize;
        unsafe {
            check_cuda_result(
                cuMemcpyHtoD_v2(self.d_ptr + offset as CUdeviceptr, elem as *const _ as *const _, size_of::<T>()),
                "cuMemcpyHtoD_v2 GpuBuffer::update_element",
            )?;
        }
        Ok(())
    }

    pub fn push(&mut self, index: usize, elem: &T) -> Result<usize, Box<dyn Error>> {
        // 1) grow if needed
        if index >= self.capacity {
            self.ensure_capacity(index + 1)?;
        }

        // 2) copy into the device at that slot
        self.update_element(index, elem)?;

        // 3) adjust len
        if index + 1 > self.len {
            self.len = index + 1;
        }

        Ok(index)
    }

    pub fn deactivate_sdf(&self, index: usize, active_offset: usize) -> Result<(), Box<dyn Error>> {
        if index >= self.capacity {
            return Err(Box::from("Index out of bounds in GpuBuffer::deactivate_element"));
        }
        let elem_size = std::mem::size_of::<T>();
        let active_size = std::mem::size_of::<u32>(); // adjust if `active` is a different type
        if active_offset + active_size > elem_size {
            return Err(Box::from("active_offset out of bounds in GpuBuffer::deactivate_element"));
        }

        // Compute device pointer to the active field of the given element
        let dest_ptr: CUdeviceptr = self.d_ptr + (index * elem_size + active_offset) as CUdeviceptr;
        let zero: u32 = 0;
        unsafe {
            check_cuda_result(
                cuMemcpyHtoD_v2(dest_ptr, &zero as *const _ as *const c_void, active_size),
                "GpuBuffer::deactivate_element",
            )?;
        }
        Ok(())
    }
}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFree_v2(self.d_ptr);
        }
    }
}


pub struct CameraBuffers {
    pub rand_states: CUdeviceptr,
    pub origins:     CUdeviceptr,
    pub directions:  CUdeviceptr,
    pub ray_per_pixel: CUdeviceptr, // int*
    pub image:       CUdeviceptr,
    pub w: u32,
    pub h: u32,
}

impl CameraBuffers {
    /// allocate once – call *before* you build any `GpuCamera`
    pub fn new(cuda:&CudaContext, w:u32, h:u32) -> Result<CameraBuffers, Box<dyn Error>> {
        let total  = (w*h) as usize;

        let rand_states = CudaContext::allocate_curand_states(w,h)?;

        let origins     = CudaContext::allocate_tensor::<f32>(
                              &vec![0.0; total*3], total*3*std::mem::size_of::<f32>())?;
        let directions  = CudaContext::allocate_tensor::<f32>(
                              &vec![0.0; total*3], total*3*std::mem::size_of::<f32>())?;

        // ray counter + rgb image
        let d_ray_per_pixel = CudaContext::allocate_tensor::<i32>(&vec![0; total], total * std::mem::size_of::<i32>())?;
        let d_image         = CudaContext::allocate_tensor::<u8>(
                                   &vec![0u8; total*3], total*3)?;

        Ok(Self { rand_states, origins, directions, ray_per_pixel: d_ray_per_pixel,
                image: d_image, w, h })
    }
}
impl Drop for CameraBuffers {
    fn drop(&mut self) {
        // free in reverse order – ignore errors because we’re in Drop
        let _ = unsafe { cuMemFree_v2(self.image) };
        let _ = unsafe { cuMemFree_v2(self.directions) };
        let _ = unsafe { cuMemFree_v2(self.origins) };
        let _ = unsafe { cuMemFree_v2(self.rand_states) };
    }
}