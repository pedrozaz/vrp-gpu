use vrp_gpu::{
    instance::{SolomonInstance, VehicleConfig, compute_distance_matrix},
    local_search::{TwoOptMove, cpu},
    solution::Route,
};

pub(crate) const SEED: u32 = 0x12345678;

pub(crate) fn synthetic(n: usize) -> (SolomonInstance, Route) {
    let xs: Vec<_> = (0..=n).map(|i| ((i * 17) % 997) as f32).collect();
    let ys: Vec<_> = (0..=n).map(|i| ((i * 29) % 991) as f32).collect();
    let instance = SolomonInstance {
        name: format!("synthetic-{n}"),
        vehicle: VehicleConfig {
            num_vehicles: 1,
            capacity: n.max(1) as f32,
        },
        num_nodes: n + 1,
        distance_matrix: compute_distance_matrix(&xs, &ys),
        xs,
        ys,
        demands: (0..=n).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect(),
        ready_times: vec![0.0; n + 1],
        due_times: vec![1e9; n + 1],
        service_times: vec![0.0; n + 1],
    };
    let mut nodes: Vec<_> = (1..=n).collect();
    let mut seed = SEED;
    for i in (1..n).rev() {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        nodes.swap(i, seed as usize % (i + 1));
    }
    (instance, Route::from_nodes(nodes))
}

pub(crate) fn close(actual: f32, expected: f32) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0)
}

pub(crate) fn check_move(
    actual: Option<TwoOptMove>,
    expected: Option<TwoOptMove>,
) -> Result<(), String> {
    match (actual, expected) {
        (None, None) => Ok(()),
        (Some(a), Some(e)) if (a.i, a.j) == (e.i, e.j) && close(a.delta, e.delta) => Ok(()),
        _ => Err(format!(
            "selection mismatch: expected {expected:?}, got {actual:?}"
        )),
    }
}

pub(crate) fn check_matrix(
    route: &Route,
    instance: &SolomonInstance,
    values: &[f32],
) -> Result<(), String> {
    let n = route.len();
    if values.len() != n * n {
        return Err("wrong delta matrix length".into());
    }
    for i in 0..n {
        for j in 0..n {
            let actual = values[i * n + j];
            if i < j {
                let expected = cpu::two_opt_delta(route, instance, i, j);
                if !close(actual, expected) {
                    return Err(format!(
                        "delta mismatch at ({i}, {j}): {actual} vs {expected}"
                    ));
                }
            } else if actual != f32::INFINITY {
                return Err(format!("invalid cell ({i}, {j}) was not positive infinity"));
            }
        }
    }
    Ok(())
}

pub(crate) fn cost(route: &Route, instance: &SolomonInstance) -> f64 {
    let mut previous = 0;
    let mut total = 0.0;
    for &node in &route.nodes {
        total += f64::from(instance.distance(previous, node));
        previous = node;
    }
    total + f64::from(instance.distance(previous, 0))
}

pub(crate) fn check_reversal(
    route: &Route,
    instance: &SolomonInstance,
    selected: Option<TwoOptMove>,
) -> Result<f64, String> {
    let before = cost(route, instance);
    let mut after = route.clone();
    if let Some(m) = selected {
        if m.i >= m.j || m.j >= route.len() || !m.delta.is_finite() || m.delta >= 0.0 {
            return Err("invalid selected move".into());
        }
        after.nodes[m.i..=m.j].reverse();
        // Error scale is the four affected f32 edge lengths, not the whole tour.
        let prev = if m.i == 0 { 0 } else { route.nodes[m.i - 1] };
        let next = if m.j + 1 == route.len() {
            0
        } else {
            route.nodes[m.j + 1]
        };
        let a = route.nodes[m.i];
        let b = route.nodes[m.j];
        let scale = [(prev, a), (b, next), (prev, b), (a, next)]
            .iter()
            .map(|&(a, b)| f64::from(instance.distance(a, b)))
            .sum::<f64>()
            .max(1.0);
        let difference = cost(&after, instance) - before;
        if difference >= 0.0 || (difference - f64::from(m.delta)).abs() > 1e-5 * scale {
            return Err(format!(
                "selected reversal does not match recomputed cost: {difference} vs {}",
                m.delta
            ));
        }
    }
    Ok(cost(&after, instance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_is_reproducible_and_valid() {
        for n in [0, 1, 2, 17, 257] {
            let (instance, route) = synthetic(n);
            assert_eq!((instance.clone(), route.clone()), synthetic(n));
            instance.validate().unwrap();
            let mut nodes = route.nodes.clone();
            nodes.sort_unstable();
            assert_eq!(nodes, (1..=n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn validator_rejects_corrupted_matrix() {
        let (instance, route) = synthetic(4);
        let mut values = vec![f32::INFINITY; 16];
        for i in 0..4 {
            for j in i + 1..4 {
                values[i * 4 + j] = cpu::two_opt_delta(&route, &instance, i, j);
            }
        }
        check_matrix(&route, &instance, &values).unwrap();
        assert!(check_matrix(&route, &instance, &values[..15]).is_err());
        for bad in [f32::NAN, f32::NEG_INFINITY, 123456.0] {
            let mut corrupted = values.clone();
            corrupted[1] = bad;
            assert!(check_matrix(&route, &instance, &corrupted).is_err());
        }
        values[0] = f32::NEG_INFINITY;
        assert!(check_matrix(&route, &instance, &values).is_err());
    }

    #[test]
    fn selection_requires_indices_even_for_equal_deltas() {
        let expected = Some(TwoOptMove {
            i: 0,
            j: 1,
            delta: -2.0,
        });
        check_move(expected, expected).unwrap();
        assert!(check_move(None, expected).is_err());
        assert!(
            check_move(
                Some(TwoOptMove {
                    i: 1,
                    j: 2,
                    delta: -2.0
                }),
                expected
            )
            .is_err()
        );
        assert!(!close(f32::NAN, 0.0));
        assert!(!close(f32::INFINITY, f32::INFINITY));
        assert!(close(0.000001, 0.0));
        assert!(!close(0.001, 0.0));
    }

    #[test]
    fn reversal_uses_independent_full_route_cost() {
        let (instance, route) = synthetic(17);
        let selected = cpu::best_two_opt_move(&route, &instance).unwrap();
        assert!(
            check_reversal(&route, &instance, Some(selected)).unwrap() < cost(&route, &instance)
        );
        assert!(
            check_reversal(
                &route,
                &instance,
                Some(TwoOptMove {
                    delta: -1e9,
                    ..selected
                })
            )
            .is_err()
        );
        assert!(check_reversal(&route, &instance, Some(TwoOptMove { j: 17, ..selected })).is_err());
    }
}
