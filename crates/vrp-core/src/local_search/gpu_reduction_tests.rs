use super::*;
use crate::local_search::cpu;

fn instance(n: usize, scale: f32) -> SolomonInstance {
    let rows: String = (0..=n)
        .map(|node| {
            let x = ((node * 17) % 71) as f32 * scale;
            let y = ((node * 29) % 73) as f32 * scale;
            format!("{node} {x} {y} 0 0 100 0\n")
        })
        .collect();
    format!("Reduction\nVEHICLE\n1 100\nCUSTOMER\n{rows}")
        .parse()
        .unwrap()
}

#[test]
fn small_routes_return_none_without_cuda() {
    let instance = instance(1, 1.0);
    for nodes in [vec![], vec![1]] {
        assert_eq!(
            best_two_opt_move(&Route::from_nodes(nodes), &instance).unwrap(),
            None
        );
    }
}

#[test]
fn rejects_bad_routes_and_instances_before_cuda() {
    let instance = instance(3, 1.0);
    for nodes in [vec![0], vec![usize::MAX], vec![4], vec![1, 1]] {
        assert!(matches!(
            best_two_opt_move(&Route::from_nodes(nodes), &instance),
            Err(GpuEvaluationError::InvalidInput(_))
        ));
    }
    let route = Route::from_nodes(vec![1, 2]);
    let mut bad = instance.clone();
    bad.distance_matrix.pop();
    assert!(matches!(
        best_two_opt_move(&route, &bad),
        Err(GpuEvaluationError::InvalidInput(_))
    ));
    bad = instance.clone();
    bad.distance_matrix[1] = f32::NAN;
    assert!(matches!(
        best_two_opt_move(&route, &bad),
        Err(GpuEvaluationError::InvalidInput(_))
    ));
    bad = instance.clone();
    bad.num_nodes = usize::MAX;
    assert!(matches!(
        best_two_opt_move(&route, &bad),
        Err(GpuEvaluationError::SizeOverflow)
    ));
    let too_large = Route::from_nodes(vec![1; 65_536]);
    assert!(matches!(
        best_two_opt_move(&too_large, &instance),
        Err(GpuEvaluationError::SizeOverflow)
    ));
}

// Independent reference for synthetic delta matrices, including poisoned invalid
// cells. Sorting makes the tie contract independent of the parallel tree shape.
fn matrix_oracle(values: &[f32], n: usize) -> Option<TwoOptMove> {
    let mut candidates: Vec<_> = (0..n)
        .flat_map(|i| {
            (i + 1..n).map(move |j| TwoOptMove {
                i,
                j,
                delta: values[i * n + j],
            })
        })
        .filter(|item| item.delta.is_finite() && item.delta < 0.0)
        .collect();
    candidates.sort_by(|a, b| {
        a.delta
            .partial_cmp(&b.delta)
            .unwrap()
            .then(a.i.cmp(&b.i))
            .then(a.j.cmp(&b.j))
    });
    candidates.first().copied()
}

fn check_matrix(stream: &Arc<CudaStream>, values: &[f32], n: usize) {
    let device = stream.clone_htod(values).unwrap();
    let actual = reduce_device_deltas(stream, device, n as u32).unwrap();
    assert_eq!(actual, matrix_oracle(values, n), "matrix width {n}");
}

#[test]
#[ignore = "requires an NVIDIA GPU and CUDA driver"]
fn reduction_boundaries_ties_and_nonfinite_values() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    // 16² fills one block; 17² has a partial block. 257² requires three passes.
    for n in [1, 2, 15, 16, 17, 255, 256, 257] {
        let mut values = vec![f32::INFINITY; n * n];
        for i in 0..n {
            for j in 0..n {
                values[i * n + j] = if i >= j {
                    -f32::MAX
                } else {
                    [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, 3.0][(i + j) % 6]
                };
            }
        }
        check_matrix(&stream, &values, n);
        if n < 2 {
            continue;
        }
        // Winner at the last valid cell, which is in a partial block for n=17.
        let last = (n - 2) * n + n - 1;
        values[last] = -7.0;
        check_matrix(&stream, &values, n);
        // Cross-block and cross-pass tie, then an adjacent f32 that is smaller.
        values[1] = -7.0;
        check_matrix(&stream, &values, n);
        values[last] = f32::from_bits((-7.0f32).to_bits() + 1);
        check_matrix(&stream, &values, n);
    }
    let n = 257;
    let mut values = vec![f32::INFINITY; n * n];
    // Both sides of the 256-thread boundary are valid in row zero.
    values[255] = -1.0;
    values[256] = -2.0;
    check_matrix(&stream, &values, n);
    values[255] = -2.0;
    for _ in 0..5 {
        check_matrix(&stream, &values, n);
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU and CUDA driver"]
fn best_move_matches_cpu_across_blocks_and_scales() {
    for n in [2, 3, 15, 16, 17, 31, 32, 33, 65, 257] {
        for scale in [0.001, 1.0, 10_000.0] {
            let instance = instance(n, scale);
            let mut nodes: Vec<_> = (1..=n).collect();
            // Reproducible Fisher-Yates permutation without an extra dependency.
            let mut seed = 0x12345678u32;
            for i in (1..n).rev() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                nodes.swap(i, seed as usize % (i + 1));
            }
            let route = Route::from_nodes(nodes);
            let before = route.clone();
            let expected = cpu::best_two_opt_move(&route, &instance);
            let actual = best_two_opt_move(&route, &instance).unwrap();
            assert_eq!(
                actual.map(|m| (m.i, m.j)),
                expected.map(|m| (m.i, m.j)),
                "n={n}, scale={scale}"
            );
            if let (Some(actual), Some(expected)) = (actual, expected) {
                assert!(
                    (actual.delta - expected.delta).abs() <= 1e-5 * expected.delta.abs().max(1.0)
                );
                let mut improved = route.clone();
                cpu::apply_two_opt(&mut improved, actual.i, actual.j);
                // Check against a complete route-cost recomputation in f64.
                let cost = |route: &Route| {
                    let nodes: Vec<_> = std::iter::once(0)
                        .chain(route.nodes.iter().copied())
                        .chain(std::iter::once(0))
                        .collect();
                    nodes
                        .windows(2)
                        .map(|edge| f64::from(instance.distance(edge[0], edge[1])))
                        .sum::<f64>()
                };
                let difference = cost(&improved) - cost(&route);
                assert!(difference < 0.0);
                assert!(
                    (difference - f64::from(actual.delta)).abs() <= 1e-4 * cost(&route).max(1.0)
                );
            }
            assert_eq!(route, before);
        }
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU and CUDA driver"]
fn best_move_returns_none_for_optimal_and_equal_cost_routes() {
    let mut instance = instance(4, 1.0);
    let n = instance.num_nodes;
    for i in 0..n {
        for j in 0..n {
            instance.distance_matrix[i * n + j] = i.abs_diff(j) as f32;
        }
    }
    assert_eq!(
        best_two_opt_move(&Route::from_nodes(vec![1, 2, 3, 4]), &instance).unwrap(),
        None
    );
    for i in 0..n {
        for j in 0..n {
            instance.distance_matrix[i * n + j] = if i == j { 0.0 } else { 1.0 };
        }
    }
    assert_eq!(
        best_two_opt_move(&Route::from_nodes(vec![1, 2, 3, 4]), &instance).unwrap(),
        None
    );
    // Two equal improving moves: exact tie must choose (0, 1).
    instance.distance_matrix.fill(2.0);
    for i in 0..n {
        instance.distance_matrix[i * n + i] = 0.0;
    }
    for (i, j) in [(0, 2), (2, 0), (1, 3), (3, 1)] {
        instance.distance_matrix[i * n + j] = 1.0;
    }
    let route = Route::from_nodes(vec![1, 2, 3]);
    assert_eq!(
        best_two_opt_move(&route, &instance).unwrap(),
        Some(TwoOptMove {
            i: 0,
            j: 1,
            delta: -2.0
        })
    );
}
