//! Solomon instance parser and Stucture of Arrays (SoA) layout.

/// Configuration and fleet constraints for a Solomon VRP instance.
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleConfig {
    /// Total number of available vehicles.
    pub num_vehicles: usize,
    /// Maximum capacity per vehicle.
    pub capacity: f32,
}

/// A parsed Solomon CVRP instance with contiguous memory layouts (SoA)
/// optimized for CPU caching and direct GPU device transfers.
#[derive(Debug, Clone, PartialEq)]
pub struct SolomonInstance {
    /// Instance identifier / name (e.g., "C101").
    pub name: String,
    /// Fleet and vehicle capacity settings.
    pub vehicle: VehicleConfig,
    /// Total number of nodes (1 depot at index 0 + N customers).
    pub num_nodes: usize,
    /// X-coordinates of all nodes (depot at index 0).
    pub xs: Vec<f32>,
    /// X-coordinates of all nodes (depot at index 0).
    pub ys: Vec<f32>,
    /// Demand of each node (depot demand at index 0 is always 0.0).
    pub demands: Vec<f32>,
    /// Ready times for time-window constraints (available for rich VRP extensions).
    pub ready_times: Vec<f32>,
    /// Due times for time-window constraints.
    pub due_times: Vec<f32>,
    /// Service times at each customer.
    pub service_times: Vec<f32>,
    /// Flattered 1D row-major euclidean distance matrix of size `num_nodes * num_nodes`.
    /// Distance from node `i` to node `j` is accessed via `distance_matrix[i * num_nodes + j]`.
    pub distance_matrix: Vec<f32>,
}

impl SolomonInstance {
    /// Returns the euclidean distance between node `i` and node `j`.
    #[inline(always)]
    pub fn demand(&self, node: usize) -> f32 {
        debug_assert!(node < self.num_nodes);
        self.demands[node]
    }
}
