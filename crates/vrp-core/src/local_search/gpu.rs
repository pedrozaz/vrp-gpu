//! GPU host-side orchestration using `cudarc` and embedded `kernel.ptx`.

use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

use crate::{instance::SolomonInstance, solution::Route};

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

/// Evaluates all 2-opt deltas for one route on the GPU.
///
/// The returned vector is a row-major `route_len x route_len` matrix. Valid
/// entries `i < j` match `local_search::cpu::two_opt_delta`; all other entries
/// are `f32::INFINITY`.
pub fn evaluate_two_opt_deltas(
    route: &Route,
    instance: &SolomonInstance,
) -> Result<Vec<f32>, DriverError> {
    let route_len = route.len();
    let output_len = route_len * route_len;

    if route_len < 2 {
        return Ok(vec![f32::INFINITY; output_len]);
    }

    let route_nodes: Vec<u32> = route.nodes.iter().map(|&node| node as u32).collect();
    let matrix_width = instance.num_nodes as u32;

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();

    let module = context.load_module(Ptx::from_src(KERNEL_PTX))?;
    let function = module.load_function("two_opt_deltas")?;

    let distance_matrix_device = stream.clone_htod(&instance.distance_matrix)?;
    let route_nodes_device = stream.clone_htod(&route_nodes)?;
    let mut deltas_device = stream.alloc_zeros::<f32>(output_len)?;
    let distance_matrix_len = instance.distance_matrix.len() as u64;
    let route_nodes_len = route_nodes.len() as u64;
    let deltas_len = output_len as u64;

    let config = LaunchConfig::for_num_elems(output_len as u32);
    let mut launch_args = stream.launch_builder(&function);
    launch_args
        .arg(&distance_matrix_device)
        .arg(&distance_matrix_len)
        .arg(&route_nodes_device)
        .arg(&route_nodes_len)
        .arg(&matrix_width)
        .arg(&mut deltas_device)
        .arg(&deltas_len);

    // SAFETY: the argument order matches the PTX ABI:
    // distance matrix pointer/length, route-node pointer/length, matrix width,
    // and output pointer/length. The one-dimensional launch owns one output
    // cell per thread.
    unsafe {
        launch_args.launch(config)?;
    }

    stream.synchronize()?;
    stream.clone_dtoh(&deltas_device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instance::{VehicleConfig, compute_distance_matrix},
        local_search::cpu::two_opt_delta,
    };

    const EPSILON: f32 = 1e-5;

    fn create_test_instance() -> SolomonInstance {
        let xs = vec![0.0, 0.0, 1.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0, 0.0];

        SolomonInstance {
            name: "TestGpuTwoOpt".into(),
            vehicle: VehicleConfig {
                num_vehicles: 1,
                capacity: 4.0,
            },
            num_nodes: xs.len(),
            demands: vec![0.0; xs.len()],
            ready_times: vec![0.0; xs.len()],
            due_times: vec![1000.0; xs.len()],
            service_times: vec![0.0; xs.len()],
            distance_matrix: compute_distance_matrix(&xs, &ys),
            xs,
            ys,
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA driver"]
    fn test_smoke_add_one_executes() {
        let expected: Vec<f32> = SMOKE_INPUT.iter().map(|value| value + 1.0).collect();

        assert_eq!(smoke_add_one().unwrap(), expected);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA driver"]
    fn test_two_opt_deltas_match_cpu_reference() {
        let instance = create_test_instance();
        let route = Route::from_nodes(vec![1, 2, 3, 4]);
        let deltas = evaluate_two_opt_deltas(&route, &instance).unwrap();
        let route_len = route.len();

        assert_eq!(deltas.len(), route_len * route_len);

        for i in 0..route_len {
            for j in 0..route_len {
                let actual = deltas[i * route_len + j];

                if i < j {
                    let expected = two_opt_delta(&route, &instance, i, j);
                    assert!(
                        (actual - expected).abs() < EPSILON,
                        "delta mismatch at ({i}, {j}): expected {expected}, got {actual}",
                    );
                } else {
                    assert_eq!(actual, f32::INFINITY, "invalid pair ({i}, {j})");
                }
            }
        }
    }
}
