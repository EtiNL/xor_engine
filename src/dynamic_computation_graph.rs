use crate::cuda_wrapper::{CudaContext, dim3};
use std::ffi::c_void;
use std::error::Error;

pub enum OperationType {
    Kernel(String),
    Conditional(Box<dyn Fn() -> bool>, Vec<Operation>),
}

pub struct Operation {
    op_type: OperationType,
    inputs: Vec<*mut f32>,
    output: Option<*mut f32>,
}

pub struct ComputationGraph {
    operations: Vec<Operation>,
    cuda_context: CudaContext,
}

impl ComputationGraph {
    pub fn new(cuda_context: CudaContext) -> Self {
        ComputationGraph {
            operations: Vec::new(),
            cuda_context,
        }
    }

    pub fn add_operation(&mut self, op_type: OperationType, inputs: Vec<*mut f32>, output: Option<*mut f32>) {
        self.operations.push(Operation { op_type, inputs, output });
    }

    pub fn add_conditional_operation<F>(&mut self, condition: F, true_branch: Vec<Operation>)
    where
        F: Fn() -> bool + 'static,
    {
        self.operations.push(Operation {
            op_type: OperationType::Conditional(Box::new(condition), true_branch),
            inputs: vec![],
            output: None,
        });
    }

    pub fn execute(&self, grid_dim: dim3, block_dim: dim3) -> Result<(), Box<dyn Error>> {
        for op in &self.operations {
            match &op.op_type {
                OperationType::Kernel(kernel_name) => {
                    let args: Vec<*mut c_void> = op.inputs.iter().map(|&input| input as *mut c_void).collect();
                    self.cuda_context.launch_kernel(kernel_name, grid_dim, block_dim, args, "stream1")?;
                }
                OperationType::Conditional(cond, true_branch) => {
                    if cond() {
                        for sub_op in true_branch {
                            let args: Vec<*mut c_void> = sub_op.inputs.iter().map(|&input| input as *mut c_void).collect();
                            if let OperationType::Kernel(kernel_name) = &sub_op.op_type {
                                self.cuda_context.launch_kernel(kernel_name, grid_dim, block_dim, args, "stream1")?;
                            }
                        }
                    }
                }
            }
            self.cuda_context.synchronize("stream1")?;
        }
        Ok(())
    }
    
    pub fn clear_operations(&mut self) {
        self.operations.clear();
    }
}
