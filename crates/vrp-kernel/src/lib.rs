#![no_std]

//! PTX kernels for batched 2-opt candidate evaluation.

use cuda_device::{DisjointSlice, kernel, thread};

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

    if route_len < 2 {
        return;
    }

    if let Some(delta) = deltas.get_mut(index) {
        let i = candidate / route_len;
        let j = candidate % route_len;

        if i >= route_len || j <= i {
            *delta = f32::INFINITY;
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
