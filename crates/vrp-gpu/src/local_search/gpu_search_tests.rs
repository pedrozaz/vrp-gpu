use super::*;
use crate::instance::VehicleConfig;

fn search_instance() -> SolomonInstance {
    let points: [(f32, f32); 8] = [
        (0.0, 0.0),
        (0.0, 4.0),
        (3.0, 1.0),
        (6.0, 5.0),
        (5.0, 0.0),
        (2.0, 6.0),
        (9.0, 0.0),
        (9.0, 3.0),
    ];
    let distance_matrix = points
        .iter()
        .flat_map(|&(x, y)| {
            points
                .iter()
                .map(move |&(other_x, other_y)| (x - other_x).abs() + (y - other_y).abs())
        })
        .collect();
    SolomonInstance {
        name: "GpuSearch".into(),
        vehicle: VehicleConfig {
            num_vehicles: 2,
            capacity: 5.0,
        },
        num_nodes: points.len(),
        xs: points.iter().map(|&(x, _)| x).collect(),
        ys: points.iter().map(|&(_, y)| y).collect(),
        demands: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].to_vec(),
        ready_times: vec![0.0; points.len()],
        due_times: vec![100.0; points.len()],
        service_times: vec![0.0; points.len()],
        distance_matrix,
    }
}

fn cpu_selector(
    route: &Route,
    instance: &SolomonInstance,
) -> Result<Option<TwoOptMove>, GpuEvaluationError> {
    Ok(cpu::best_two_opt_move(route, instance))
}

#[test]
fn route_search_matches_cpu_for_multiple_moves() {
    let instance = search_instance();
    let mut route = Route::from_nodes(vec![1, 2, 3, 4, 5]);
    let before = route_cost_f64(&route, &instance);
    let mut expected = route.clone();
    cpu::two_opt_route(&mut expected, &instance);

    let report = search_route_with(&mut route, &instance, &mut cpu_selector).unwrap();

    assert_eq!(route, expected);
    assert_eq!(route.nodes, vec![1, 5, 3, 4, 2]);
    assert_eq!(report.accepted_moves, 2);
    assert_eq!(
        report.distance_improvement,
        before - route_cost_f64(&route, &instance)
    );
    assert_eq!(report.distance_improvement, 14.0);
    assert_eq!(
        search_route_with(&mut route, &instance, &mut cpu_selector).unwrap(),
        GpuSearchReport {
            accepted_moves: 0,
            distance_improvement: 0.0,
        }
    );
}

#[test]
fn small_routes_and_empty_solution_do_not_require_cuda() {
    let instance = search_instance();
    for nodes in [vec![], vec![1]] {
        let mut route = Route::from_nodes(nodes.clone());
        assert_eq!(
            two_opt_route(&mut route, &instance).unwrap(),
            GpuSearchReport {
                accepted_moves: 0,
                distance_improvement: 0.0,
            }
        );
        assert_eq!(route.nodes, nodes);
    }
    let mut solution = Solution::empty();
    assert_eq!(
        two_opt(&mut solution, &instance).unwrap(),
        GpuSearchReport {
            accepted_moves: 0,
            distance_improvement: 0.0,
        }
    );
}

#[test]
fn invalid_input_is_rejected_before_selector() {
    let instance = search_instance();
    for nodes in [vec![0], vec![8], vec![1, 1]] {
        let mut route = Route::from_nodes(nodes.clone());
        let mut called = false;
        let result = search_route_with(&mut route, &instance, &mut |_, _| {
            called = true;
            Ok(None)
        });
        assert!(matches!(result, Err(GpuEvaluationError::InvalidInput(_))));
        assert!(!called);
        assert_eq!(route.nodes, nodes);
    }

    let mut bad = instance.clone();
    bad.distance_matrix.pop();
    let mut route = Route::from_nodes(vec![1, 2]);
    assert!(matches!(
        search_route_with(&mut route, &bad, &mut cpu_selector),
        Err(GpuEvaluationError::InvalidInput(_))
    ));
    assert_eq!(route.nodes, vec![1, 2]);
}

#[test]
fn inconsistent_selection_is_rejected_without_mutation() {
    let instance = search_instance();
    let original = Route::from_nodes(vec![1, 2, 3, 4, 5]);
    for selected in [
        TwoOptMove {
            i: 1,
            j: 1,
            delta: -1.0,
        },
        TwoOptMove {
            i: 1,
            j: 5,
            delta: -1.0,
        },
        TwoOptMove {
            i: 0,
            j: 1,
            delta: f32::NAN,
        },
        TwoOptMove {
            i: 0,
            j: 1,
            delta: 0.0,
        },
    ] {
        let mut route = original.clone();
        assert!(matches!(
            search_route_with(&mut route, &instance, &mut |_, _| Ok(Some(selected))),
            Err(GpuEvaluationError::InconsistentMove)
        ));
        assert_eq!(route, original);
    }

    let mut route = Route::from_nodes(vec![1, 2]);
    let original = route.clone();
    assert!(matches!(
        search_route_with(&mut route, &instance, &mut |_, _| {
            Ok(Some(TwoOptMove {
                i: 0,
                j: 1,
                delta: -1.0,
            }))
        }),
        Err(GpuEvaluationError::InconsistentMove)
    ));
    assert_eq!(route, original);
}

#[test]
fn selector_failure_after_a_move_keeps_route_unchanged() {
    let instance = search_instance();
    let mut route = Route::from_nodes(vec![1, 2, 3, 4, 5]);
    let original = route.clone();
    let mut calls = 0;
    let result = search_route_with(&mut route, &instance, &mut |route, instance| {
        calls += 1;
        if calls == 1 {
            cpu_selector(route, instance)
        } else {
            Err(GpuEvaluationError::InvalidInput(
                "injected selector failure",
            ))
        }
    });
    assert!(matches!(result, Err(GpuEvaluationError::InvalidInput(_))));
    assert_eq!(calls, 2);
    assert_eq!(route, original);
}

#[test]
fn solution_search_matches_cpu_and_is_atomic() {
    let instance = search_instance();
    let original = Solution::new(vec![
        Route::from_nodes(vec![1, 2, 3, 4, 5]),
        Route::from_nodes(vec![6, 7]),
    ]);
    assert!(original.is_feasible(&instance));
    let mut expected = original.clone();
    cpu::two_opt(&mut expected, &instance);

    let mut solution = original.clone();
    let report = search_solution_with(&mut solution, &instance, &mut cpu_selector).unwrap();
    assert_eq!(solution, expected);
    assert!(solution.is_feasible(&instance));
    assert_eq!(report.accepted_moves, 2);
    assert_eq!(report.distance_improvement, 14.0);

    let mut solution = original.clone();
    let result = search_solution_with(&mut solution, &instance, &mut |route, instance| {
        if route.nodes.first() == Some(&6) {
            Err(GpuEvaluationError::InvalidInput(
                "injected second-route failure",
            ))
        } else {
            cpu_selector(route, instance)
        }
    });
    assert!(matches!(result, Err(GpuEvaluationError::InvalidInput(_))));
    assert_eq!(solution, original);

    let mut solution = original.clone();
    solution.routes[1] = Route::from_nodes(vec![0]);
    let invalid = solution.clone();
    let mut called = false;
    assert!(matches!(
        search_solution_with(&mut solution, &instance, &mut |_, _| {
            called = true;
            Ok(None)
        }),
        Err(GpuEvaluationError::InvalidInput(_))
    ));
    assert!(!called);
    assert_eq!(solution, invalid);
}

#[test]
#[ignore = "requires an NVIDIA GPU and CUDA driver"]
fn gpu_route_search_matches_cpu_and_recomputed_cost() {
    let instance = search_instance();
    let mut route = Route::from_nodes(vec![1, 2, 3, 4, 5]);
    let before = route_cost_f64(&route, &instance);
    let mut expected = route.clone();
    cpu::two_opt_route(&mut expected, &instance);

    let report = two_opt_route(&mut route, &instance).unwrap();

    assert_eq!(route, expected);
    assert_eq!(report.accepted_moves, 2);
    assert_eq!(
        report.distance_improvement,
        before - route_cost_f64(&route, &instance)
    );
    assert_eq!(report.distance_improvement, 14.0);
    assert_eq!(
        two_opt_route(&mut route, &instance).unwrap().accepted_moves,
        0
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU and CUDA driver"]
fn gpu_solution_search_matches_cpu_and_preserves_feasibility() {
    let instance = search_instance();
    let mut solution = Solution::new(vec![
        Route::from_nodes(vec![1, 2, 3, 4, 5]),
        Route::from_nodes(vec![6, 7]),
    ]);
    let before: f64 = solution
        .routes
        .iter()
        .map(|route| route_cost_f64(route, &instance))
        .sum();
    let mut expected = solution.clone();
    cpu::two_opt(&mut expected, &instance);

    let report = two_opt(&mut solution, &instance).unwrap();

    let after: f64 = solution
        .routes
        .iter()
        .map(|route| route_cost_f64(route, &instance))
        .sum();
    assert_eq!(solution, expected);
    assert!(solution.is_feasible(&instance));
    assert_eq!(report.accepted_moves, 2);
    assert_eq!(report.distance_improvement, before - after);
    assert_eq!(report.distance_improvement, 14.0);
}
