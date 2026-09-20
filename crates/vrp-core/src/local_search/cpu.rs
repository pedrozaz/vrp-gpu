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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{VehicleConfig, compute_distance_matrix};

    const EPSILON: f32 = 1e-5;

    fn create_test_instance() -> SolomonInstance {
        // Depot: (0, 0)
        // Customers 1, 2, and 3 form a unit square with the depot.
        // Customer 4 extends the instance for multi-route tests.
        let xs = vec![0.0, 0.0, 1.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let demands = vec![0.0, 1.0, 1.0, 1.0, 1.0];
        let distance_matrix = compute_distance_matrix(&xs, &ys);

        SolomonInstance {
            name: "TestTwoOpt".into(),
            vehicle: VehicleConfig {
                num_vehicles: 2,
                capacity: 4.0,
            },
            num_nodes: xs.len(),
            xs,
            ys,
            demands,
            ready_times: vec![0.0; 5],
            due_times: vec![1000.0; 5],
            service_times: vec![0.0; 5],
            distance_matrix,
        }
    }

    fn assert_approx_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_two_opt_delta_matches_recomputed_route_distance() {
        let instance = create_test_instance();
        let route = Route::from_nodes(vec![1, 2, 3, 4]);
        let distance_before = route.distance(&instance);

        for i in 0..route.len() - 1 {
            for j in i + 1..route.len() {
                let expected_delta = two_opt_delta(&route, &instance, i, j);
                let mut swapped = route.clone();
                apply_two_opt(&mut swapped, i, j);
                let actual_delta = swapped.distance(&instance) - distance_before;

                assert_approx_eq(actual_delta, expected_delta);
            }
        }
    }

    #[test]
    fn test_two_opt_delta_for_known_improving_move() {
        let instance = create_test_instance();
        let route = Route::from_nodes(vec![1, 2, 3]);

        let delta = two_opt_delta(&route, &instance, 1, 2);
        let expected = 2.0 - 2.0 * 2.0f32.sqrt();

        assert!(delta < 0.0);
        assert_approx_eq(delta, expected);
    }

    #[test]
    fn test_apply_two_opt_reverses_only_selected_segment() {
        let mut route = Route::from_nodes(vec![1, 2, 3, 4]);

        apply_two_opt(&mut route, 1, 3);

        assert_eq!(route.nodes, vec![1, 4, 3, 2]);
    }

    #[test]
    fn test_two_opt_route_reaches_local_optimum() {
        let instance = create_test_instance();
        let mut route = Route::from_nodes(vec![1, 2, 3]);
        let distance_before = route.distance(&instance);

        let improvement = two_opt_route(&mut route, &instance);
        let distance_after = route.distance(&instance);

        assert_eq!(route.nodes, vec![1, 3, 2]);
        assert!(distance_after < distance_before);
        assert_approx_eq(improvement, distance_before - distance_after);

        for i in 0..route.len() - 1 {
            for j in i + 1..route.len() {
                assert!(two_opt_delta(&route, &instance, i, j) >= 0.0);
            }
        }

        assert_eq!(two_opt_route(&mut route, &instance), 0.0);
    }

    #[test]
    fn test_two_opt_route_leaves_small_routes_unchanged() {
        let instance = create_test_instance();
        let routes = [
            Route::new(),
            Route::from_nodes(vec![1]),
            Route::from_nodes(vec![1, 2]),
        ];

        for original in routes {
            let mut route = original.clone();
            let improvement = two_opt_route(&mut route, &instance);

            assert_eq!(route, original);
            assert_eq!(improvement, 0.0);
        }
    }

    #[test]
    fn test_two_opt_improves_solution_without_changing_feasibility() {
        let instance = create_test_instance();
        let mut solution = Solution::new(vec![
            Route::from_nodes(vec![1, 2, 3]),
            Route::from_nodes(vec![4]),
        ]);
        let distance_before = solution.total_distance(&instance);

        let improvement = two_opt(&mut solution, &instance);
        let distance_after = solution.total_distance(&instance);

        assert!(solution.is_feasible(&instance));
        assert_eq!(solution.routes[0].nodes, vec![1, 3, 2]);
        assert_eq!(solution.routes[1].nodes, vec![4]);
        assert!(distance_after < distance_before);
        assert_approx_eq(improvement, distance_before - distance_after);
    }
}
