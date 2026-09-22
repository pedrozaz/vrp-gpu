#![no_std]

//! PTX kernels for batched 2-opt candidate evaluation.

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

/// Reduces 256 candidates per block to a (delta, original row-major index) pair.
///
/// Launch exactly (256, 1, 1) threads per block in a 1D grid. On the first pass,
/// route_len > 0 selects valid upper-triangle cells and derives their indices;
/// on later passes route_len == 0 preserves indices from the preceding pass.
/// Each output slice must contain at least gridDim.x entries. All input slices
/// must cover the indicated pass. Input/output allocations must not alias.
/// Non-finite and nonnegative values reduce to (0.0, u32::MAX).
#[kernel]
pub fn reduce_two_opt_candidates(
    values: &[f32],
    indices: &[u32],
    route_len: u32,
    mut best_values: DisjointSlice<f32>,
    mut best_indices: DisjointSlice<u32>,
) {
    static mut VALUES: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut INDICES: SharedArray<u32, 256> = SharedArray::UNINIT;
    let lane = thread::threadIdx_x() as usize;
    let block = thread::blockIdx_x() as usize;
    let offset = block * 256 + lane;
    let mut value = 0.0f32;
    let mut original_index = u32::MAX;
    if offset < values.len() {
        let candidate = values[offset];
        let valid_cell = route_len == 0
            || (offset / (route_len as usize) < offset % (route_len as usize)
                && offset / (route_len as usize) < route_len as usize);
        // Comparisons reject NaN, infinities and signed zero without tolerances.
        if valid_cell && candidate < 0.0 && candidate.is_finite() {
            value = candidate;
            original_index = if route_len == 0 {
                indices[offset]
            } else {
                offset as u32
            };
        }
    }

    // SAFETY: each lane initializes its own slot, all 256 lanes participate in
    // every barrier, and active lanes only read the inactive half at each step.
    // Raw pointers avoid creating overlapping references to the shared arrays.
    unsafe {
        let shared_values = SharedArray::as_raw_mut_ptr(&raw mut VALUES);
        let shared_indices = SharedArray::as_raw_mut_ptr(&raw mut INDICES);
        *shared_values.add(lane) = value;
        *shared_indices.add(lane) = original_index;
        thread::sync_threads();
        let mut stride = 128;
        while stride > 0 {
            if lane < stride {
                let other_value = *shared_values.add(lane + stride);
                let other_index = *shared_indices.add(lane + stride);
                let current_value = *shared_values.add(lane);
                let current_index = *shared_indices.add(lane);
                if other_value < current_value
                    || (other_value == current_value && other_index < current_index)
                {
                    *shared_values.add(lane) = other_value;
                    *shared_indices.add(lane) = other_index;
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if lane == 0 {
            // Exactly one writer per block and one output slot per block.
            *best_values.get_unchecked_mut(block) = *shared_values;
            *best_indices.get_unchecked_mut(block) = *shared_indices;
        }
    }
}

/// Adds one to each element.
///
/// This is a toolchain smoke-test kernel. It verifies Rust-To-PTX compilation
/// before the actual 2-opt evaluation kernel is introduced.
#[kernel]
pub fn add_one(input: &[f32], mut output: DisjointSlice<f32>) {
    let index = thread::index_1d();
    let raw_index = index.get();

    if let Some(output_value) = output.get_mut(index) {
        *output_value = input[raw_index] + 1.0;
    }
}

/// Evaluates every 2-opt candidate for one route.
///
/// `deltas[i * route_len + j]` equals the CPU 2-opt delta for valid `i < j`.
/// All other cells are `f32::INFINITY`, so they cannot be selected as an
/// improving move by a later reduction step.
#[kernel]
pub fn two_opt_deltas(
    distance_matrix: &[f32],
    route_nodes: &[u32],
    matrix_width: u32,
    mut deltas: DisjointSlice<f32>,
) {
    let index = thread::index_1d();
    let candidate = index.get();
    let route_len = route_nodes.len();

    if let Some(delta) = deltas.get_mut(index) {
        *delta = f32::INFINITY;
        if route_len < 2 {
            return;
        }

        let i = candidate / route_len;
        let j = candidate % route_len;

        if i >= route_len || j <= i {
            return;
        }

        let matrix_width = matrix_width as usize;
        let previous_i = if i == 0 {
            0
        } else {
            route_nodes[i - 1] as usize
        };
        let node_i = route_nodes[i] as usize;
        let node_j = route_nodes[j] as usize;
        let next_j = if j == route_len - 1 {
            0
        } else {
            route_nodes[j + 1] as usize
        };

        let removed = distance_matrix[previous_i * matrix_width + node_i]
            + distance_matrix[node_j * matrix_width + next_j];
        let added = distance_matrix[previous_i * matrix_width + node_j]
            + distance_matrix[node_i * matrix_width + next_j];

        *delta = added - removed;
    }
}
