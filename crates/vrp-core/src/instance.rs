//! Solomon instance data structures and flat Structure-of-Arrays (SoA) layout.

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
    /// Y-coordinates of all nodes (depot at index 0).
    pub ys: Vec<f32>,
    /// Demand of each node (depot demand at index 0 is always 0.0).
    pub demands: Vec<f32>,
    /// Ready times for time-window constraints (available for rich VRP extensions).
    pub ready_times: Vec<f32>,
    /// Due times for time-window constraints.
    pub due_times: Vec<f32>,
    /// Service times at each customer.
    pub service_times: Vec<f32>,
    /// Flattened 1D row-major euclidean distance matrix of size `num_nodes * num_nodes`.
    /// Distance from node `i` to node `j` is accessed via `distance_matrix[i* num_nodes + j]`.
    pub distance_matrix: Vec<f32>,
}

impl SolomonInstance {
    /// Returns the euclidean distance between node `from` and node `to`.
    #[inline(always)]
    pub fn distance(&self, from: usize, to: usize) -> f32 {
        debug_assert!(from < self.num_nodes && to < self.num_nodes);
        self.distance_matrix[from * self.num_nodes + to]
    }

    /// Returns the demand of `node`.
    #[inline(always)]
    pub fn demand(&self, node: usize) -> f32 {
        debug_assert!(node < self.num_nodes);
        self.demands[node]
    }
}

/// Computes a flattened 1D row-major euclidean distance matrix from coordinate slices.
///
/// Returns a vector of size `n * n`, where `n = xs.len()`.
/// Element at index `i * n + j` is `sqrt((xs[i] - xs[j])^2 + (ys[i] - ys[j])^2)`.
pub fn compute_distance_matrix(xs: &[f32], ys: &[f32]) -> Vec<f32> {
    assert_eq!(
        xs.len(),
        ys.len(),
        "X and Y coordinate slices must have identical length"
    );
    let n = xs.len();
    let mut matrix = Vec::with_capacity(n * n);

    for i in 0..n {
        let xi = xs[i];
        let yi = ys[i];
        for j in 0..n {
            if i == j {
                matrix.push(0.0);
            } else {
                let dx = xi - xs[j];
                let dy = yi - ys[j];
                matrix.push((dx * dx + dy * dy).sqrt());
            }
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_distance_matrix_triangle() {
        // Triângulo retângulo: (0, 0), (3, 0), (0, 4)
        let xs = vec![0.0, 3.0, 0.0];
        let ys = vec![0.0, 0.0, 4.0];
        let matrix = compute_distance_matrix(&xs, &ys);

        assert_eq!(matrix.len(), 9);

        let dist = |i: usize, j: usize| matrix[i * 3 + j];

        // Distância para si mesmo é 0.0
        assert_eq!(dist(0, 0), 0.0);
        assert_eq!(dist(1, 1), 0.0);
        assert_eq!(dist(2, 2), 0.0);

        // Distância 0 <-> 1 é 3.0
        assert_eq!(dist(0, 1), 3.0);
        assert_eq!(dist(1, 0), 3.0);

        // Distância 0 <-> 2 é 4.0
        assert_eq!(dist(0, 2), 4.0);
        assert_eq!(dist(2, 0), 4.0);

        // Distância 1 <-> 2 é 5.0 (Hipotenusa 3-4-5)
        assert_eq!(dist(1, 2), 5.0);
        assert_eq!(dist(2, 1), 5.0);
    }
}
