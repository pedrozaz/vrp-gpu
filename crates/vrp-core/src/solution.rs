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
        let active_routes: Vec<&Route> = self.routes.iter().filter(|r| !r.is_empty()).collect();

        // 1. Fleet size constraint
        if active_routes.len() > instance.vehicle.num_vehicles {
            return false;
        }

        // 2. Capacity constraint for each route
        for route in &active_routes {
            if !route.is_capacity_feasible(instance) {
                return false;
            }
        }

        // 3. Customer visitation constraint (each customer 1..num_nodes visited exactly once)
        if instance.num_nodes <= 1 {
            return active_routes.is_empty();
        }

        let mut visited = vec![false; instance.num_nodes];

        for route in &active_routes {
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
        }

        // Verify that all customers 1..num_nodes were visited
        visited[1..].iter().all(|&v| v)
    }
}
