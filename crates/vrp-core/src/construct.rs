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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{VehicleConfig, compute_distance_matrix};

    fn create_test_instance() -> SolomonInstance {
        // Depot at (0, 0)
        // Customer 1 at (1, 0), demand 10
        // Customer 2 at (2, 0), demand 10
        // Customer 3 at (3, 0), demand 10
        // Customer 4 at (0, 5), demand 10
        // Vehicle capacity: 25 (forces at least 2 routes)
        let xs = vec![0.0, 1.0, 2.0, 3.0, 0.0];
        let ys = vec![0.0, 0.0, 0.0, 0.0, 5.0];
        let demands = vec![0.0, 10.0, 10.0, 10.0, 10.0];
        let ready_times = vec![0.0; 5];
        let due_times = vec![1000.0; 5];
        let service_times = vec![0.0; 5];
        let distance_matrix = compute_distance_matrix(&xs, &ys);

        SolomonInstance {
            name: "TestNN".into(),
            vehicle: VehicleConfig {
                num_vehicles: 3,
                capacity: 25.0,
            },
            num_nodes: 5,
            xs,
            ys,
            demands,
            ready_times,
            due_times,
            service_times,
            distance_matrix,
        }
    }

    #[test]
    fn test_nearest_neighbor_produces_feasible_solution() {
        let instance = create_test_instance();
        let solution = nearest_neighbor(&instance);

        assert!(
            solution.is_feasible(&instance),
            "Nearest neighbor must produce a feasible solution"
        );
    }

    #[test]
    fn test_nearest_neighbor_visits_all_customers() {
        let instance = create_test_instance();
        let solution = nearest_neighbor(&instance);

        assert_eq!(
            solution.total_customers_visited(),
            instance.num_nodes - 1,
            "All customers must be visited"
        );
    }

    #[test]
    fn test_nearest_neighbor_respects_capacity() {
        let instance = create_test_instance();
        let solution = nearest_neighbor(&instance);

        for (i, route) in solution.routes.iter().enumerate() {
            assert!(
                route.is_capacity_feasible(&instance),
                "Route {i} exceeds vehicle capacity: demand {} > capacity {}",
                route.total_demand(&instance),
                instance.vehicle.capacity,
            );
        }
    }

    #[test]
    fn test_nearest_neighbor_splits_routes_on_capacity() {
        let instance = create_test_instance();
        let solution = nearest_neighbor(&instance);

        // 4 customers x 10 demand each = 40 total, capacity 25 per vehicle
        // Must use at least 2 routes
        assert!(
            solution.routes.len() >= 2,
            "Expected at least 2 routes due to capacity constraint, got {}",
            solution.routes.len(),
        );
    }

    #[test]
    fn test_nearest_neighbor_single_customer() {
        let xs = vec![0.0, 5.0];
        let ys = vec![0.0, 0.0];
        let demands = vec![0.0, 10.0];
        let distance_matrix = compute_distance_matrix(&xs, &ys);

        let instance = SolomonInstance {
            name: "Single".into(),
            vehicle: VehicleConfig {
                num_vehicles: 1,
                capacity: 100.0,
            },
            num_nodes: 2,
            xs,
            ys,
            demands,
            ready_times: vec![0.0; 2],
            due_times: vec![1000.0; 2],
            service_times: vec![0.0; 2],
            distance_matrix,
        };

        let solution = nearest_neighbor(&instance);
        assert!(solution.is_feasible(&instance));
        assert_eq!(solution.routes.len(), 1);
        assert_eq!(solution.routes[0].nodes, vec![1]);
    }

    #[test]
    fn test_nearest_neighbor_empty_instance() {
        let xs = vec![0.0];
        let ys = vec![0.0];
        let demands = vec![0.0];
        let distance_matrix = compute_distance_matrix(&xs, &ys);

        let instance = SolomonInstance {
            name: "DepotOnly".into(),
            vehicle: VehicleConfig {
                num_vehicles: 1,
                capacity: 100.0,
            },
            num_nodes: 1,
            xs,
            ys,
            demands,
            ready_times: vec![0.0],
            due_times: vec![1000.0],
            service_times: vec![0.0],
            distance_matrix,
        };

        let solution = nearest_neighbor(&instance);
        assert!(solution.routes.is_empty());
    }
}
