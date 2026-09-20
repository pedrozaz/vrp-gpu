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
