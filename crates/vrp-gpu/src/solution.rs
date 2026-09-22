//! Route representations and solution cost calculations for CVRP.

use crate::instance::SolomonInstance;

/// Represents a single vehicle route servicing a sequence of customers.
///
/// The route implicitly starts at the depot (node 0) and ends at the depot (node 0).
/// The `nodes` vector stores only the customer sequence (nodes >= 1).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Route {
    /// Ordered sequence of customer indices visited by this vehicle.
    pub nodes: Vec<usize>,
}

impl Route {
    /// Creates a new empty route.
    #[inline]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Creates a route with a predefined sequence of customer nodes.
    #[inline]
    pub fn from_nodes(nodes: Vec<usize>) -> Self {
        Self { nodes }
    }

    /// Returns the number of customers in this route.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the route contains no customers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Calculates the total euclidean distance traveled by the route:
    /// `depot (0) -> nodes[0] -> ... -> nodes[n-1] -> depot (0)`.
    /// An empty route has a distance of `0.0`.
    pub fn distance(&self, instance: &SolomonInstance) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let mut total_dist = instance.distance(0, self.nodes[0]);
        for window in self.nodes.windows(2) {
            total_dist += instance.distance(window[0], window[1]);
        }
        total_dist += instance.distance(*self.nodes.last().unwrap(), 0);

        total_dist
    }

    /// Calculates the total demand delivered to all customers in this route.
    pub fn total_demand(&self, instance: &SolomonInstance) -> f32 {
        self.nodes.iter().map(|&node| instance.demand(node)).sum()
    }

    /// Returns `true` if the total route demand does not exceed vehicle capacity.
    #[inline]
    pub fn is_capacity_feasible(&self, instance: &SolomonInstance) -> bool {
        self.total_demand(instance) <= instance.vehicle.capacity
    }
}

/// Represents a complete CVRP solution comprising multiple vehicle routes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Solution {
    /// Collection of active vehicle routes.
    pub routes: Vec<Route>,
}

impl Solution {
    /// Creates a new solution from a vector of routes.
    #[inline]
    pub fn new(routes: Vec<Route>) -> Self {
        Self { routes }
    }

    /// Creates an empty solution with no routes.
    #[inline]
    pub fn empty() -> Self {
        Self { routes: Vec::new() }
    }

    /// Calculates the total distance across all routes in the solution.
    pub fn total_distance(&self, instance: &SolomonInstance) -> f32 {
        self.routes
            .iter()
            .map(|route| route.distance(instance))
            .sum()
    }

    /// Returns the total number of customers visited across all routes.
    pub fn total_customers_visited(&self) -> usize {
        self.routes.iter().map(Route::len).sum()
    }

    /// Validates full solution feasibility:
    /// 1. Number of active non-empty routes <= `instance.vehicle.num_vehicles`.
    /// 2. Each route respects vehicle capacity constraints (`total_demand <= capacity`).
    /// 3. Every customer node `1..instance.num_nodes` is visited exactly once.
    pub fn is_feasible(&self, instance: &SolomonInstance) -> bool {
        if instance.validate().is_err() {
            return false;
        }
        let active_routes = self.routes.iter().filter(|r| !r.is_empty());

        // 1. Fleet size constraint
        if active_routes.clone().count() > instance.vehicle.num_vehicles {
            return false;
        }

        // 3. Customer visitation constraint (each customer 1..num_nodes visited exactly once)
        let mut visited = vec![false; instance.num_nodes];

        for route in active_routes {
            for &node in &route.nodes {
                // Depot cannot be in route.nodes, and node must be within valid range
                if node == 0 || node >= instance.num_nodes {
                    return false;
                }
                // Duplicate visit check
                if visited[node] {
                    return false;
                }
                visited[node] = true;
            }
            // Validate IDs before indexing demands in the capacity calculation.
            if !route.is_capacity_feasible(instance) {
                return false;
            }
        }

        // Verify that all customers 1..num_nodes were visited
        visited[1..].iter().all(|&v| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::VehicleConfig;

    #[test]
    fn test_solution_rejects_invalid_ids_without_panicking() {
        let instance = create_mock_instance();
        for node in [instance.num_nodes, usize::MAX] {
            let solution = Solution::new(vec![Route::from_nodes(vec![node])]);
            assert!(!solution.is_feasible(&instance));
        }
    }

    #[test]
    fn test_solution_rejects_malformed_instance() {
        let mut instance = create_mock_instance();
        instance.demands.clear();
        let solution = Solution::new(vec![Route::from_nodes(vec![1, 2, 3])]);
        assert!(!solution.is_feasible(&instance));
    }

    #[test]
    fn test_solution_ignores_empty_routes_in_fleet_count() {
        let instance = create_mock_instance();
        let solution = Solution::new(vec![
            Route::new(),
            Route::from_nodes(vec![1, 2]),
            Route::from_nodes(vec![3]),
        ]);
        assert!(solution.is_feasible(&instance));
    }

    fn create_mock_instance() -> SolomonInstance {
        // Depot (0, 0)
        // Node 1: (3, 0), demand 10
        // Node 2: (3, 4), demand 20
        // Node 3: (0, 4), demand 15
        // Vehicle: 2 vehicles, capacity 40.0
        let xs = vec![0.0, 3.0, 3.0, 0.0];
        let ys = vec![0.0, 0.0, 4.0, 4.0];
        let demands = vec![0.0, 10.0, 20.0, 15.0];
        let ready_times = vec![0.0, 0.0, 0.0, 0.0];
        let due_times = vec![1000.0, 1000.0, 1000.0, 1000.0];
        let service_times = vec![0.0, 10.0, 10.0, 10.0];
        let distance_matrix = crate::instance::compute_distance_matrix(&xs, &ys);

        SolomonInstance {
            name: "MockInstance".into(),
            vehicle: VehicleConfig {
                num_vehicles: 2,
                capacity: 40.0,
            },
            num_nodes: 4,
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
    fn test_empty_route_and_solution() {
        let instance = create_mock_instance();
        let route = Route::new();
        assert!(route.is_empty());
        assert_eq!(route.len(), 0);
        assert_eq!(route.distance(&instance), 0.0);
        assert_eq!(route.total_demand(&instance), 0.0);
        assert!(route.is_capacity_feasible(&instance));

        let solution = Solution::empty();
        assert_eq!(solution.total_distance(&instance), 0.0);
        assert_eq!(solution.total_customers_visited(), 0);
    }

    #[test]
    fn test_route_distance_and_demand() {
        let instance = create_mock_instance();
        // Route: 0 -> 1 -> 2 -> 0
        // dist(0, 1) = 3.0
        // dist(1, 2) = 4.0
        // dist(2, 0) = 5.0 (hypotenuse sqrt(3^2 + 4^2))
        // total dist = 12.0
        // total demand = 10 + 20 = 30.0 <= 40.0
        let route = Route::from_nodes(vec![1, 2]);

        assert_eq!(route.len(), 2);
        assert!(!route.is_empty());
        assert_eq!(route.total_demand(&instance), 30.0);
        assert!(route.is_capacity_feasible(&instance));

        let dist = route.distance(&instance);
        assert!((dist - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_solution_feasible_valid() {
        let instance = create_mock_instance();
        // 2 routes covering all customers 1, 2, 3
        // Route 1: 0 -> 1 -> 2 -> 0 (demand 30 <= 40)
        // Route 2: 0 -> 3 -> 0 (demand 15 <= 40)
        let r1 = Route::from_nodes(vec![1, 2]);
        let r2 = Route::from_nodes(vec![3]);
        let solution = Solution::new(vec![r1, r2]);

        assert_eq!(solution.total_customers_visited(), 3);
        assert!(solution.is_feasible(&instance));
    }

    #[test]
    fn test_solution_infeasible_exceeds_capacity() {
        let instance = create_mock_instance();
        // Route covering 1, 2, 3: demand = 10 + 20 + 15 = 45 > 40.0 capacity
        let r = Route::from_nodes(vec![1, 2, 3]);
        let solution = Solution::new(vec![r]);

        assert!(!solution.is_feasible(&instance));
    }

    #[test]
    fn test_solution_infeasible_duplicate_customer() {
        let instance = create_mock_instance();
        // Customer 1 visited twice
        let r1 = Route::from_nodes(vec![1, 2]);
        let r2 = Route::from_nodes(vec![1, 3]);
        let solution = Solution::new(vec![r1, r2]);

        assert!(!solution.is_feasible(&instance));
    }

    #[test]
    fn test_solution_infeasible_missing_customer() {
        let instance = create_mock_instance();
        // Customer 3 missing
        let r1 = Route::from_nodes(vec![1, 2]);
        let solution = Solution::new(vec![r1]);

        assert!(!solution.is_feasible(&instance));
    }

    #[test]
    fn test_solution_infeasible_too_many_vehicles() {
        let instance = create_mock_instance();
        // 3 active routes when max vehicles is 2
        let r1 = Route::from_nodes(vec![1]);
        let r2 = Route::from_nodes(vec![2]);
        let r3 = Route::from_nodes(vec![3]);
        let solution = Solution::new(vec![r1, r2, r3]);

        assert!(!solution.is_feasible(&instance));
    }

    #[test]
    fn test_solution_infeasible_depot_in_route() {
        let instance = create_mock_instance();
        // Depot node 0 explicitly included in nodes list
        let r1 = Route::from_nodes(vec![0, 1, 2]);
        let r2 = Route::from_nodes(vec![3]);
        let solution = Solution::new(vec![r1, r2]);

        assert!(!solution.is_feasible(&instance));
    }
}
