//! Solomon instance data structures and flat Structure-of-Arrays (SoA) layout.

use std::fmt;
use std::str::FromStr;

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
    /// Distance from node `i` to node `j` is accessed via `distance_matrix[i * num_nodes + j]`.
    pub distance_matrix: Vec<f32>,
}

impl SolomonInstance {
    /// Validates the public arrays and the symmetric CVRP cost model.
    ///
    /// Call this after constructing or mutating an instance manually. Validation
    /// is O(n²), including finite, nonnegative, symmetric distances.
    pub fn validate(&self) -> Result<(), &'static str> {
        let n = self.num_nodes;
        if n == 0 {
            return Err("instance must contain a depot");
        }
        if self.vehicle.num_vehicles == 0
            || !self.vehicle.capacity.is_finite()
            || self.vehicle.capacity <= 0.0
        {
            return Err("fleet size and finite vehicle capacity must be positive");
        }
        let arrays = [
            &self.xs,
            &self.ys,
            &self.demands,
            &self.ready_times,
            &self.due_times,
            &self.service_times,
        ];
        if arrays.iter().any(|values| values.len() != n) {
            return Err("node arrays must match num_nodes");
        }
        if arrays
            .iter()
            .any(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err("node values must be finite");
        }
        if self.demands[0] != 0.0 || self.demands.iter().any(|&demand| demand < 0.0) {
            return Err("demands must be nonnegative and depot demand must be zero");
        }
        if self
            .ready_times
            .iter()
            .zip(&self.due_times)
            .any(|(&ready, &due)| ready < 0.0 || due < ready)
            || self.service_times.iter().any(|&service| service < 0.0)
        {
            return Err("time windows and service times must be nonnegative and ordered");
        }
        if n.checked_mul(n) != Some(self.distance_matrix.len()) {
            return Err("distance matrix must contain num_nodes squared entries");
        }
        for i in 0..n {
            for j in 0..n {
                let distance = self.distance_matrix[i * n + j];
                if !distance.is_finite()
                    || distance < 0.0
                    || (i == j && distance != 0.0)
                    || distance != self.distance_matrix[j * n + i]
                {
                    return Err(
                        "distance matrix must be finite, nonnegative, symmetric, with zero diagonal",
                    );
                }
            }
        }
        Ok(())
    }

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

/// Errors that can occur during Solomon instance parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolomonParseError {
    /// Input file or string was empty.
    EmptyInput,
    /// Vehicle section header or data was missing or malformed.
    InvalidVehicleSection(String),
    /// Customer section header or data row was malformed.
    InvalidCustomerRow(String),
    /// No customer/depot nodes found in the instance.
    NoNodesFound,
    /// Parsed values do not form a valid symmetric CVRP instance.
    InvalidInstance(String),
}

impl fmt::Display for SolomonParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "Solomon instance input is empty"),
            Self::InvalidVehicleSection(msg) => write!(f, "Invalid vehicle section: {msg}"),
            Self::InvalidCustomerRow(msg) => write!(f, "Invalid customer row: {msg}"),
            Self::NoNodesFound => write!(f, "Instance does not contain any customer/depot nodes"),
            Self::InvalidInstance(msg) => write!(f, "Invalid instance: {msg}"),
        }
    }
}

impl std::error::Error for SolomonParseError {}

impl FromStr for SolomonInstance {
    type Err = SolomonParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut name = None;
        let mut num_vehicles = None;
        let mut capacity = None;

        let mut in_vehicle_section = false;
        let mut in_customer_section = false;

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut demands = Vec::new();
        let mut ready_times = Vec::new();
        let mut due_times = Vec::new();
        let mut service_times = Vec::new();

        for raw_line in s.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            // The first non-empty line before sections is the instance name
            if name.is_none()
                && !line.eq_ignore_ascii_case("VEHICLE")
                && !line.eq_ignore_ascii_case("CUSTOMER")
            {
                name = Some(line.to_string());
                continue;
            }

            if line.eq_ignore_ascii_case("VEHICLE") {
                in_vehicle_section = true;
                in_customer_section = false;
                continue;
            }

            if line.eq_ignore_ascii_case("CUSTOMER") {
                in_vehicle_section = false;
                in_customer_section = true;
                continue;
            }

            if in_vehicle_section {
                if line.contains("NUMBER") || line.contains("CAPACITY") {
                    continue;
                }
                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() >= 2 && num_vehicles.is_none() {
                    num_vehicles = Some(tokens[0].parse::<usize>().map_err(|e| {
                        SolomonParseError::InvalidVehicleSection(format!(
                            "Invalid vehicle number: {e}"
                        ))
                    })?);
                    capacity = Some(tokens[1].parse::<f32>().map_err(|e| {
                        SolomonParseError::InvalidVehicleSection(format!("Invalid capacity: {e}"))
                    })?);
                }
                continue;
            }

            if in_customer_section {
                if line.contains("CUST") || line.contains("XCOORD") || line.contains("DEMAND") {
                    continue;
                }

                let tokens: Vec<&str> = line.split_whitespace().collect();
                if tokens.len() < 7 {
                    return Err(SolomonParseError::InvalidCustomerRow(format!(
                        "Expected 7 columns, found {}: '{}'",
                        tokens.len(),
                        line
                    )));
                }

                let id = tokens[0].parse::<usize>().map_err(|e| {
                    SolomonParseError::InvalidCustomerRow(format!("Invalid node ID: {e}"))
                })?;
                if id != xs.len() {
                    return Err(SolomonParseError::InvalidCustomerRow(
                        "Node IDs must be consecutive from depot 0, in row order".into(),
                    ));
                }

                let x = tokens[1].parse::<f32>().map_err(|e| {
                    SolomonParseError::InvalidCustomerRow(format!("Invalid X coordinate: {e}"))
                })?;
                let y = tokens[2].parse::<f32>().map_err(|e| {
                    SolomonParseError::InvalidCustomerRow(format!("Invalid Y coordinate: {e}"))
                })?;
                let demand = tokens[3].parse::<f32>().map_err(|e| {
                    SolomonParseError::InvalidCustomerRow(format!("Invalid Demand: {e}"))
                })?;
                let ready = tokens[4].parse::<f32>().map_err(|e| {
                    SolomonParseError::InvalidCustomerRow(format!("Invalid Ready time: {e}"))
                })?;
                let due = tokens[5].parse::<f32>().map_err(|e| {
                    SolomonParseError::InvalidCustomerRow(format!("Invalid Due time: {e}"))
                })?;
                let service = tokens[6].parse::<f32>().map_err(|e| {
                    SolomonParseError::InvalidCustomerRow(format!("Invalid Service time: {e}"))
                })?;

                xs.push(x);
                ys.push(y);
                demands.push(demand);
                ready_times.push(ready);
                due_times.push(due);
                service_times.push(service);
            }
        }

        let name = name.ok_or(SolomonParseError::EmptyInput)?;
        let num_vehicles = num_vehicles.ok_or_else(|| {
            SolomonParseError::InvalidVehicleSection("Vehicle count not found".into())
        })?;
        let capacity = capacity.ok_or_else(|| {
            SolomonParseError::InvalidVehicleSection("Vehicle capacity not found".into())
        })?;

        let num_nodes = xs.len();
        if num_nodes == 0 {
            return Err(SolomonParseError::NoNodesFound);
        }

        let distance_matrix = compute_distance_matrix(&xs, &ys);

        let instance = Self {
            name,
            vehicle: VehicleConfig {
                num_vehicles,
                capacity,
            },
            num_nodes,
            xs,
            ys,
            demands,
            ready_times,
            due_times,
            service_times,
            distance_matrix,
        };
        instance
            .validate()
            .map_err(|message| SolomonParseError::InvalidInstance(message.into()))?;
        Ok(instance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_rejects_invalid_ids_and_numeric_values() {
        let prefix = "Test\nVEHICLE\n2 10\nCUSTOMER\n";
        for row in [
            "1 0 0 0 0 100 0",
            "bad 0 0 0 0 100 0",
            "0 NaN 0 0 0 100 0",
            "0 inf 0 0 0 100 0",
            "0 0 0 -1 0 100 0",
            "0 0 0 1 0 100 0",
            "0 0 0 0 100 0 0",
            "0 0 0 0 0 100 -1",
        ] {
            assert!(
                format!("{prefix}{row}").parse::<SolomonInstance>().is_err(),
                "{row}"
            );
        }
        for id in [0, 2] {
            let input = format!("{prefix}0 0 0 0 0 100 0\n{id} 1 1 1 0 100 0");
            assert!(input.parse::<SolomonInstance>().is_err());
        }
        for fleet in ["0 10", "1 0", "1 -1", "1 NaN", "1 inf"] {
            let input = format!("Test\nVEHICLE\n{fleet}\nCUSTOMER\n0 0 0 0 0 100 0");
            assert!(input.parse::<SolomonInstance>().is_err());
        }
    }

    #[test]
    fn test_validation_rejects_malformed_public_arrays() {
        let input = "Test\nVEHICLE\n2 10\nCUSTOMER\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0";
        let original: SolomonInstance = input.parse().unwrap();
        let mut instance = original.clone();
        instance.demands.pop();
        assert!(instance.validate().is_err());
        instance = original.clone();
        instance.distance_matrix.pop();
        assert!(instance.validate().is_err());
        for value in [f32::NAN, f32::INFINITY, -1.0, 2.0] {
            instance = original.clone();
            instance.distance_matrix[1] = value;
            assert!(instance.validate().is_err());
        }
        instance = original;
        instance.num_nodes = usize::MAX;
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_compute_distance_matrix_triangle() {
        let xs = vec![0.0, 3.0, 0.0];
        let ys = vec![0.0, 0.0, 4.0];
        let matrix = compute_distance_matrix(&xs, &ys);

        assert_eq!(matrix.len(), 9);

        let dist = |i: usize, j: usize| matrix[i * 3 + j];

        assert_eq!(dist(0, 0), 0.0);
        assert_eq!(dist(1, 1), 0.0);
        assert_eq!(dist(2, 2), 0.0);

        assert_eq!(dist(0, 1), 3.0);
        assert_eq!(dist(1, 0), 3.0);

        assert_eq!(dist(0, 2), 4.0);
        assert_eq!(dist(2, 0), 4.0);

        assert_eq!(dist(1, 2), 5.0);
        assert_eq!(dist(2, 1), 5.0);
    }

    #[test]
    fn test_parse_solomon_c101_sample() {
        let sample = r#"
C101

VEHICLE
NUMBER     CAPACITY
  25          200

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
    0       40.0      50.0       0.0        0.0     1236.0        0.0
    1       45.0      68.0      10.0      912.0      967.0       90.0
    2       45.0      70.0      30.0      825.0      870.0       90.0
"#;

        let instance: SolomonInstance = sample
            .parse()
            .expect("failed to parse valid Solomon string");

        assert_eq!(instance.name, "C101");
        assert_eq!(instance.vehicle.num_vehicles, 25);
        assert_eq!(instance.vehicle.capacity, 200.0);
        assert_eq!(instance.num_nodes, 3);

        // Coordenadas SoA
        assert_eq!(instance.xs, vec![40.0, 45.0, 45.0]);
        assert_eq!(instance.ys, vec![50.0, 68.0, 70.0]);

        // Demandas
        assert_eq!(instance.demand(0), 0.0);
        assert_eq!(instance.demand(1), 10.0);
        assert_eq!(instance.demand(2), 30.0);

        // Janelas e tempos
        assert_eq!(instance.ready_times, vec![0.0, 912.0, 825.0]);
        assert_eq!(instance.due_times, vec![1236.0, 967.0, 870.0]);
        assert_eq!(instance.service_times, vec![0.0, 90.0, 90.0]);

        // Distância 1 <-> 2: dx = 0, dy = 2 -> dist = 2.0
        assert_eq!(instance.distance(1, 2), 2.0);
        assert_eq!(instance.distance(2, 1), 2.0);

        // Distância 0 <-> 1: dx = 5, dy = 18 -> sqrt(25 + 324) = sqrt(349)
        let expected_dist_0_1 = (5.0f32 * 5.0 + 18.0 * 18.0).sqrt();
        assert!((instance.distance(0, 1) - expected_dist_0_1).abs() < 1e-5);
    }

    #[test]
    fn test_parse_solomon_empty_fails() {
        let result: Result<SolomonInstance, _> = "".parse();
        assert_eq!(result.unwrap_err(), SolomonParseError::EmptyInput);
    }

    #[test]
    fn test_parse_solomon_missing_vehicle_fails() {
        let sample = "C101\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 0 0";
        let result: Result<SolomonInstance, _> = sample.parse();
        assert!(matches!(
            result.unwrap_err(),
            SolomonParseError::InvalidVehicleSection(_)
        ));
    }

    #[test]
    fn test_parse_solomon_invalid_customer_row_fails() {
        let sample = r#"
C101
VEHICLE
NUMBER CAPACITY
25 200
CUSTOMER
CUST NO. X Y DEMAND READY DUE SERVICE
0 0 0
"#;
        let result: Result<SolomonInstance, _> = sample.parse();
        assert!(matches!(
            result.unwrap_err(),
            SolomonParseError::InvalidCustomerRow(_)
        ));
    }
}
