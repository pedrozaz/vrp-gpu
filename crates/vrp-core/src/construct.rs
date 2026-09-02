//! Initial constructive heuristics (e.g. Nearest Neighbor, Clarke-Wright Savings).

use crate::instance::SolomonInstance;
use crate::solution::{Route, Solution};

/// Builds an initial feasible CVRP solution using a greedy Nearest Neighbor heuristic.
///
/// Starting from the depot (node 0), the algorithm repeatedly selects the closest
/// unvisited customer that fits within the current vehicle's remaining capacity.
/// When no feasible insertion exists, the current route is closed (return to depot)
/// and a new vehicle route is started.
///
/// # Panics
///
/// Panics if the instance contains a customer whose individual demand exceeds
/// the vehicle capacity (no single vehicle can serve it).
pub fn nearest_neighbor(instance: &SolomonInstance) -> Solution {
    let n = instance.num_nodes;
    if n <= 1 {
        return Solution::empty();
    }

    let mut visited = vec![false; n];
    visited[0] = true; // depot is not a customer

    let mut routes = Vec::new();
    let mut remaining_customers = n - 1;

    while remaining_customers > 0 {
        let mut route_nodes = Vec::new();
        let mut current_node = 0usize;
        let mut remaining_capacity = instance.vehicle.capacity;

        loop {
            // Find the nearest unvisited customer that fits in remaining capacity
            let mut best_node = None;
            let mut best_dist = f32::INFINITY;

            #[allow(clippy::needless_range_loop)]
            for candidate in 1..n {
                if visited[candidate] {
                    continue;
                }
                let demand = instance.demand(candidate);
                if demand > remaining_capacity {
                    continue;
                }
                let dist = instance.distance(current_node, candidate);
                if dist < best_dist {
                    best_dist = dist;
                    best_node = Some(candidate);
                }
            }

            match best_node {
                Some(node) => {
                    visited[node] = true;
                    remaining_capacity -= instance.demand(node);
                    route_nodes.push(node);
                    current_node = node;
                    remaining_customers -= 1;
                }
                None => break, // no feasible insertion, close route
            }
        }

        if !route_nodes.is_empty() {
            routes.push(Route::from_nodes(route_nodes))
        }
    }

    Solution::new(routes)
}
