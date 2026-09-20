//! GPU host-side orchestration using `cudarc` and embedded `kernel.ptx`.

use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

const KERNEL_PTX: &str = include_str!("../../kernel.ptx");
const SMOKE_INPUT: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

/// Runs the PTX smoke-test through the CUDA driver.
///
/// This verifies the versioned PTX artifact can be loaded by `cudarc` and
/// launched on the configured CUDA device before 2-opt orchestration is added.
pub fn smoke_add_one() -> Result<Vec<f32>, DriverError> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();

    let module = context.load_module(Ptx::from_src(KERNEL_PTX))?;
    let function = module.load_function("add_one")?;

    let input_device = stream.clone_htod(&SMOKE_INPUT)?;
    let mut output_device = stream.alloc_zeros::<f32>(SMOKE_INPUT.len())?;

    let input_len = SMOKE_INPUT.len() as u64;
    let output_len = input_len;
    let config = LaunchConfig::for_num_elems(SMOKE_INPUT.len() as u32);

    let mut launch_args = stream.launch_builder(&function);
    launch_args
        .arg(&input_device)
        .arg(&input_len)
        .arg(&mut output_device)
        .arg(&output_len);

    // SAFETY: `add_one` expects input pointer/length followed by output
    // pointer/length. Both device buffers contain `SMOKE_INPUT.len()` f32s,
    // and the one-dimensional launch is bounds-checked by `get_mut`.
    unsafe {
        launch_args.launch(config)?;
    }

    stream.synchronize()?;
    stream.clone_dtoh(&output_device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA driver"]
    fn test_smoke_add_one_executes() {
        let expected: Vec<f32> = SMOKE_INPUT.iter().map(|value| value + 1.0).collect();

        assert_eq!(smoke_add_one().unwrap(), expected);
    }
}
