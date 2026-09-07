//! CPU reference implementation of 2-opt local serach (validation gold standard).
//!
//! This module is the correctness baseline against which all GPU kernel outputs
//! must be validated (see §0.5 of the project foundation document).

use crate::instance::SolomonInstance;
use crate::solution::{Route, Solution};

/// Calculates the cost delta of a 2-opt swap on a single route.
///
/// A 2-opt move reverses the segment between positions `i` and `j` (inclusive)
/// in the route's node sequence. The route is implicitly closed: depot (0) at
/// both ends.
///
/// Given route nodes `[n0, n1, ..., nk]` (depot excluded), the edges affected
/// by reversing segment `[i..=j]` are:
/// - Removed: `(prev_i -> nodes[i])` and `(nodes[j] -> next_j)`
/// - Added:   `(prev_i -> nodes[j])` and `(nodes[i] -> next_j)`
///
/// Returns `delta < 0.0` if the swap improves the route cost, `>= 0.0` otherwise.
///
/// # Panics
///
/// Panics in debug mode if `i >= j` or if indices are out of bounds.
pub fn two_opt_delta(route: &Route, instance: &SolomonInstance, i: usize, j: usize) -> f32 {
    debug_assert!(i < j, "i must be strictly less than j");
    debug_assert!(j < route.nodes.len(), "j must be within route bounds");

    let nodes = &route.nodes;
    let n = nodes.len();

    // Node before position i: depot (0) if i == 0, else nodes[i-1]
    let prev_i = if i == 0 { 0 } else { nodes[i - 1] };
    // Node after position j: depot (0) if j == last, else nodes[j+1]
    let next_j = if j == n - 1 { 0 } else { nodes[j + 1] };

    let removed = instance.distance(prev_i, nodes[i]) + instance.distance(nodes[j], next_j);
    let added = instance.distance(prev_i, nodes[j]) + instance.distance(nodes[i], next_j);

    added - removed
}

/// Applies a 2-opt swap in-place on `route`, reversing segment `[i..=j]`.
pub fn apply_two_opt(route: &mut Route, i: usize, j: usize) {
    debug_assert!(i < j, "i must be strictly less than j");
    debug_assert!(j < route.nodes.len(), "j must be within route bounds");
    route.nodes[i..=j].reverse();
}

/// Runs best-improvement 2-opt local search on a single route until no
/// improving swaps exists.
///
/// Returns the total distance improvement achieved (>= 0.0).
pub fn two_opt_route(route: &mut Route, instance: &SolomonInstance) -> f32 {
    let mut total_improvement = 0.0f32;

    loop {
        let n = route.nodes.len();
        if n < 2 {
            break;
        }

        let mut best_delta = 0.0f32;
        let mut best_i = 0;
        let mut best_j = 0;
        for i in 0..n - 1 {
            for j in i + 1..n {
                let delta = two_opt_delta(route, instance, i, j);
                if delta < best_delta {
                    best_delta = delta;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_delta < 0.0 {
            apply_two_opt(route, best_i, best_j);
            total_improvement -= best_delta;
        } else {
            break;
        }
    }

    total_improvement
}

/// Runs 2-opt local search on every route in the solution independently.
///
/// Returns the total distance improvement across all routes.
pub fn two_opt(solution: &mut Solution, instance: &SolomonInstance) -> f32 {
    solution
        .routes
        .iter_mut()
        .map(|route| two_opt_route(route, instance))
        .sum()
}
