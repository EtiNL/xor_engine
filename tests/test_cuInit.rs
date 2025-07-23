use cuda_driver_sys::*;

#[test]
fn test_cuinit_direct() {
    let res = unsafe { cuInit(0) };
    assert_eq!(res, cudaError_enum::CUDA_SUCCESS);
}