//! GPU host-side orchestration using `cudarc` and embedded `kernel.ptx`.

use std::sync::Arc;

use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

use super::{TwoOptMove, cpu};
use crate::{
    instance::SolomonInstance,
    solution::{Route, Solution},
};

const KERNEL_PTX: &str = include_str!("../../kernel.ptx");
const SMOKE_INPUT: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
// Must match the shared-array size and reduction tree in the versioned PTX.
const REDUCTION_THREADS: u32 = 256;

#[cfg(test)]
#[path = "gpu_reduction_tests.rs"]
mod reduction_tests;

#[cfg(test)]
#[path = "gpu_search_tests.rs"]
mod search_tests;

/// Input validation, CUDA execution, or selected-move consistency failure.
#[derive(Debug)]
pub enum GpuEvaluationError {
    /// The instance or route violates the evaluator's input invariants.
    InvalidInput(&'static str),
    /// A matrix or launch dimension cannot be represented safely.
    SizeOverflow,
    /// A selected move was invalid or did not reduce recomputed route cost.
    InconsistentMove,
    /// The CUDA driver rejected an operation.
    Driver(DriverError),
}

impl std::fmt::Display for GpuEvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid GPU input: {message}"),
            Self::SizeOverflow => write!(f, "input exceeds the kernel indexing or launch limits"),
            Self::InconsistentMove => {
                write!(
                    f,
                    "selected 2-opt move is invalid or does not reduce route cost"
                )
            }
            Self::Driver(error) => write!(f, "CUDA evaluation failed: {error}"),
        }
    }
}

impl std::error::Error for GpuEvaluationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DriverError> for GpuEvaluationError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

/// Result of a completed GPU-selected 2-opt search.
///
/// Improvement is computed from complete route costs accumulated in `f64`
/// from the instance's `f32` distance matrix, not from a sum of move deltas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSearchReport {
    /// Number of reversals accepted by the host.
    pub accepted_moves: usize,
    /// Total decrease in recomputed route distance.
    pub distance_improvement: f64,
}

fn output_size(route_len: usize) -> Result<u32, GpuEvaluationError> {
    route_len
        .checked_mul(route_len)
        .and_then(|size| u32::try_from(size).ok())
        .ok_or(GpuEvaluationError::SizeOverflow)
}

/// Runs the PTX smoke-test through the CUDA driver.
///
/// This verifies the versioned PTX artifact can be loaded by `cudarc` and
/// launched on the configured CUDA device.
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
///
/// Validates the instance, customer IDs and launch dimensions before accessing
/// CUDA. Empty and singleton routes return without creating a CUDA context.
/// This initial API evaluates one route per call, with no persistent GPU cache.
pub fn evaluate_two_opt_deltas(
    route: &Route,
    instance: &SolomonInstance,
) -> Result<Vec<f32>, GpuEvaluationError> {
    let (route_nodes, matrix_width, launch_len) = prepare_route(route, instance)?;
    if route.len() < 2 {
        return Ok(vec![f32::INFINITY; launch_len as usize]);
    }
    Ok(launch_two_opt_deltas(
        &route_nodes,
        &instance.distance_matrix,
        matrix_width,
        launch_len,
    )?)
}

fn prepare_route(
    route: &Route,
    instance: &SolomonInstance,
) -> Result<(Vec<u32>, u32, u32), GpuEvaluationError> {
    let route_len = route.len();
    let launch_len = output_size(route_len)?;
    let matrix_width =
        u32::try_from(instance.num_nodes).map_err(|_| GpuEvaluationError::SizeOverflow)?;
    instance
        .validate()
        .map_err(GpuEvaluationError::InvalidInput)?;
    let mut visited = vec![false; instance.num_nodes];
    let mut route_nodes = Vec::with_capacity(route_len);
    for &node in &route.nodes {
        if node == 0 || node >= instance.num_nodes || visited[node] {
            return Err(GpuEvaluationError::InvalidInput(
                "route must contain distinct customer IDs within the instance",
            ));
        }
        visited[node] = true;
        route_nodes.push(u32::try_from(node).map_err(|_| GpuEvaluationError::SizeOverflow)?);
    }

    Ok((route_nodes, matrix_width, launch_len))
}

/// Finds the best finite negative delta on the GPU without changing the route.
///
/// Exact ties select the smallest row-major `(i, j)`, matching the CPU oracle.
/// Returns `None` if no improving candidate exists. Empty and singleton routes
/// do not access CUDA. Inputs undergo the same validation as delta evaluation.
/// The n² deltas remain on the device: repeated 256-thread block reductions
/// transfer only one f32 delta and one u32 original index to the host.
pub fn best_two_opt_move(
    route: &Route,
    instance: &SolomonInstance,
) -> Result<Option<TwoOptMove>, GpuEvaluationError> {
    let (route_nodes, matrix_width, launch_len) = prepare_route(route, instance)?;
    if route.len() < 2 {
        return Ok(None);
    }
    let (stream, deltas) = launch_two_opt_deltas_device(
        &route_nodes,
        &instance.distance_matrix,
        matrix_width,
        launch_len,
    )?;
    Ok(reduce_device_deltas(&stream, deltas, route.len() as u32)?)
}

/// Runs GPU-selected best-improvement 2-opt until no improving move remains.
///
/// Each iteration uses [`best_two_opt_move`], which currently creates a CUDA
/// context and transfers the matrix and route for every nontrivial call. The
/// host applies each reversal and accepts it only if a complete `f64` cost
/// recomputation strictly decreases. This is a route-local search, not a
/// time-window-aware solver or a GPU speedup claim.
///
/// The input is validated before CUDA access. If validation, CUDA execution,
/// or a selected-move consistency check fails at any iteration, `route` is
/// unchanged. Empty and singleton routes return a zero report without CUDA.
pub fn two_opt_route(
    route: &mut Route,
    instance: &SolomonInstance,
) -> Result<GpuSearchReport, GpuEvaluationError> {
    search_route_with(route, instance, &mut best_two_opt_move)
}

/// Runs GPU-selected 2-opt independently on every route in a solution.
///
/// Routes are validated individually; this does not check that the solution
/// visits every customer or satisfies fleet and capacity constraints. Reversal
/// preserves route membership and demand. The entire `solution` is unchanged
/// if any route fails validation, CUDA execution, or move verification.
pub fn two_opt(
    solution: &mut Solution,
    instance: &SolomonInstance,
) -> Result<GpuSearchReport, GpuEvaluationError> {
    search_solution_with(solution, instance, &mut best_two_opt_move)
}

fn route_cost_f64(route: &Route, instance: &SolomonInstance) -> f64 {
    let mut previous = 0;
    let mut cost = 0.0;
    for &node in route.nodes.iter().chain(std::iter::once(&0)) {
        cost += f64::from(instance.distance(previous, node));
        previous = node;
    }
    cost
}

fn search_route_with<F>(
    route: &mut Route,
    instance: &SolomonInstance,
    selector: &mut F,
) -> Result<GpuSearchReport, GpuEvaluationError>
where
    F: FnMut(&Route, &SolomonInstance) -> Result<Option<TwoOptMove>, GpuEvaluationError>,
{
    prepare_route(route, instance)?;
    let mut candidate = route.clone();
    let initial_cost = route_cost_f64(&candidate, instance);
    let mut current_cost = initial_cost;
    let mut accepted_moves = 0usize;

    while let Some(selected) = selector(&candidate, instance)? {
        if selected.i >= selected.j
            || selected.j >= candidate.len()
            || !selected.delta.is_finite()
            || selected.delta >= 0.0
        {
            return Err(GpuEvaluationError::InconsistentMove);
        }
        cpu::apply_two_opt(&mut candidate, selected.i, selected.j);
        let next_cost = route_cost_f64(&candidate, instance);
        if next_cost >= current_cost {
            return Err(GpuEvaluationError::InconsistentMove);
        }
        current_cost = next_cost;
        accepted_moves += 1;
    }

    *route = candidate;
    Ok(GpuSearchReport {
        accepted_moves,
        distance_improvement: initial_cost - current_cost,
    })
}

fn search_solution_with<F>(
    solution: &mut Solution,
    instance: &SolomonInstance,
    selector: &mut F,
) -> Result<GpuSearchReport, GpuEvaluationError>
where
    F: FnMut(&Route, &SolomonInstance) -> Result<Option<TwoOptMove>, GpuEvaluationError>,
{
    instance
        .validate()
        .map_err(GpuEvaluationError::InvalidInput)?;
    for route in &solution.routes {
        prepare_route(route, instance)?;
    }

    let mut candidate = solution.clone();
    let mut report = GpuSearchReport {
        accepted_moves: 0,
        distance_improvement: 0.0,
    };
    for route in &mut candidate.routes {
        let route_report = search_route_with(route, instance, selector)?;
        report.accepted_moves += route_report.accepted_moves;
        report.distance_improvement += route_report.distance_improvement;
    }
    *solution = candidate;
    Ok(report)
}

// Inputs are a nonempty n² delta buffer, n > 0, and n² fits u32. Intermediate
// indices always refer to the original buffer, never to the preceding pass.
fn reduce_device_deltas(
    stream: &Arc<CudaStream>,
    mut values: CudaSlice<f32>,
    route_len: u32,
) -> Result<Option<TwoOptMove>, DriverError> {
    let module = stream.context().load_module(Ptx::from_src(KERNEL_PTX))?;
    let function = module.load_function("reduce_two_opt_candidates")?;
    // The first pass derives indices; a one-element allocation supplies a valid
    // unused pointer without allocating an n² host or device index vector.
    let mut indices = stream.alloc_zeros::<u32>(1)?;
    let mut first_pass_width = route_len;
    loop {
        let count = values.len() as u32;
        let blocks = count.div_ceil(REDUCTION_THREADS);
        let mut next_values = stream.alloc_zeros::<f32>(blocks as usize)?;
        let mut next_indices = stream.alloc_zeros::<u32>(blocks as usize)?;
        let values_len = values.len() as u64;
        let indices_len = indices.len() as u64;
        let output_len = u64::from(blocks);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (REDUCTION_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = stream.launch_builder(&function);
        args.arg(&values)
            .arg(&values_len)
            .arg(&indices)
            .arg(&indices_len)
            .arg(&first_pass_width)
            .arg(&mut next_values)
            .arg(&output_len)
            .arg(&mut next_indices)
            .arg(&output_len);
        // SAFETY: PTX expects four pointer/u64-length pairs and a u32 width in
        // this order. Exactly 256 lanes participate in every shared-memory
        // barrier. Each block writes one distinct output slot. Later passes
        // have one index per value; the first pass never reads indices. Buffers
        // are disjoint and cudarc tracks their lifetimes on this stream.
        unsafe {
            args.launch(config)?;
        }
        values = next_values;
        indices = next_indices;
        if blocks == 1 {
            break;
        }
        first_pass_width = 0;
    }
    stream.synchronize()?;
    let delta = stream.clone_dtoh(&values)?[0];
    let index = stream.clone_dtoh(&indices)?[0];
    if index == u32::MAX {
        return Ok(None);
    }
    Ok(Some(TwoOptMove {
        i: (index / route_len) as usize,
        j: (index % route_len) as usize,
        delta,
    }))
}

// Caller validates matrix dimensions, node IDs, and the u32 launch size.
fn launch_two_opt_deltas(
    route_nodes: &[u32],
    distance_matrix: &[f32],
    matrix_width: u32,
    launch_len: u32,
) -> Result<Vec<f32>, DriverError> {
    let (stream, deltas) =
        launch_two_opt_deltas_device(route_nodes, distance_matrix, matrix_width, launch_len)?;
    stream.clone_dtoh(&deltas)
}

fn launch_two_opt_deltas_device(
    route_nodes: &[u32],
    distance_matrix: &[f32],
    matrix_width: u32,
    launch_len: u32,
) -> Result<(Arc<CudaStream>, CudaSlice<f32>), DriverError> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();

    let module = context.load_module(Ptx::from_src(KERNEL_PTX))?;
    let function = module.load_function("two_opt_deltas")?;

    let distance_matrix_device = stream.clone_htod(distance_matrix)?;
    let route_nodes_device = stream.clone_htod(route_nodes)?;
    let mut deltas_device = stream.alloc_zeros::<f32>(launch_len as usize)?;
    let distance_matrix_len = distance_matrix.len() as u64;
    let route_nodes_len = route_nodes.len() as u64;
    let deltas_len = u64::from(launch_len);

    let config = LaunchConfig::for_num_elems(launch_len);
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
    // cell per thread. Validated customer IDs index a complete square matrix;
    // the output length fits the launch and every buffer outlives execution.
    unsafe {
        launch_args.launch(config)?;
    }

    stream.synchronize()?;
    Ok((stream, deltas_device))
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
    fn test_small_routes_do_not_require_cuda() {
        let instance = create_test_instance();
        assert!(
            evaluate_two_opt_deltas(&Route::new(), &instance)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            evaluate_two_opt_deltas(&Route::from_nodes(vec![1]), &instance).unwrap(),
            vec![f32::INFINITY]
        );
    }

    #[test]
    fn test_invalid_inputs_are_rejected_before_cuda() {
        let mut instance = create_test_instance();
        for nodes in [
            vec![0],
            vec![usize::MAX],
            vec![instance.num_nodes],
            vec![1, 1],
        ] {
            assert!(matches!(
                evaluate_two_opt_deltas(&Route::from_nodes(nodes), &instance),
                Err(GpuEvaluationError::InvalidInput(_))
            ));
        }
        let route = Route::from_nodes(vec![1, 2]);
        instance.distance_matrix.pop();
        assert!(matches!(
            evaluate_two_opt_deltas(&route, &instance),
            Err(GpuEvaluationError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_launch_size_rejects_overflow_without_allocating() {
        assert_eq!(output_size(65_535).unwrap(), 4_294_836_225);
        assert!(matches!(
            output_size(65_536),
            Err(GpuEvaluationError::SizeOverflow)
        ));
        assert!(matches!(
            output_size(usize::MAX),
            Err(GpuEvaluationError::SizeOverflow)
        ));
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
        for nodes in [
            vec![1, 2],
            vec![1, 2, 3],
            vec![1, 2, 3, 4],
            vec![4, 2, 1, 3],
        ] {
            assert_gpu_parity(&Route::from_nodes(nodes), &instance);
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA driver"]
    fn test_singleton_kernel_initializes_invalid_cell() {
        let instance = create_test_instance();
        // Exercise the kernel directly; the public API has a CPU fast path.
        let deltas = launch_two_opt_deltas(&[1], &instance.distance_matrix, 5, 1).unwrap();
        assert_eq!(deltas, vec![f32::INFINITY]);
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA driver"]
    fn test_two_opt_deltas_across_blocks_and_scales() {
        // 33² and 65² exercise multiple blocks and a partial final block.
        for route_len in [31, 32, 33, 65] {
            for scale in [0.001f32, 1.0, 10_000.0] {
                let rows: String = (0..=route_len)
                    .map(|node| {
                        let x = ((node * 17) % 71) as f32 * scale;
                        let y = ((node * 29) % 73) as f32 * scale;
                        format!("{node} {x} {y} 0 0 100 0\n")
                    })
                    .collect();
                let instance: SolomonInstance = format!("Parity\nVEHICLE\n1 100\nCUSTOMER\n{rows}")
                    .parse()
                    .unwrap();
                let mut nodes: Vec<usize> = (1..=route_len).rev().collect();
                nodes.rotate_left(route_len / 3);
                assert_gpu_parity(&Route::from_nodes(nodes), &instance);
            }
        }
    }

    fn assert_gpu_parity(route: &Route, instance: &SolomonInstance) {
        let deltas = evaluate_two_opt_deltas(route, instance).unwrap();
        let route_len = route.len();

        assert_eq!(deltas.len(), route_len * route_len);

        for i in 0..route_len {
            for j in 0..route_len {
                let actual = deltas[i * route_len + j];

                if i < j {
                    let expected = two_opt_delta(route, instance, i, j);
                    let tolerance = EPSILON * expected.abs().max(1.0);
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= tolerance,
                        "delta mismatch at ({i}, {j}): expected {expected}, got {actual}",
                    );
                } else {
                    assert_eq!(actual, f32::INFINITY, "invalid pair ({i}, {j})");
                }
            }
        }
    }
}
