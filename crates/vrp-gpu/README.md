# vrp-gpu

`vrp-gpu` provides CVRP data structures, CPU reference heuristics and optional
CUDA evaluation of 2-opt moves. It parses Solomon-style text, constructs routes
with a greedy nearest-neighbor heuristic, and improves routes on the CPU.
The published GPU API computes candidate deltas or selects a single move. The
development branch also provides host-orchestrated GPU-selected 2-opt search
for routes and solutions; this new API is not in the published `0.1.1-alpha`.

The current alpha is
[`0.1.1-alpha`](https://crates.io/crates/vrp-gpu/0.1.1-alpha).
The API may change before a stable release. See the
[documentation for that version](https://docs.rs/vrp-gpu/0.1.1-alpha/vrp_gpu/).

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `gpu` | No | Exposes `local_search::gpu`, adds `cudarc`, and embeds the checked-in PTX targeting `sm_120` |

Without `gpu`, the library has no third-party runtime dependencies. Enabling
`gpu` requires a working NVIDIA driver when a GPU operation is executed. The
PTX is precompiled; consumers do not need cuda-oxide or a nightly Rust compiler.

## Installation

To use the published alpha, specify its prerelease version:

```toml
[dependencies]
vrp-gpu = "0.1.1-alpha"
```

For changes not yet published, pin a tested Git revision:

```toml
[dependencies]
vrp-gpu = { git = "https://github.com/pedrozaz/vrp-gpu", rev = "<tested-commit>" }
```

## Example

```rust
use std::error::Error;

use vrp_gpu::{
    construct::nearest_neighbor,
    instance::SolomonInstance,
    local_search::cpu::two_opt,
};

fn main() -> Result<(), Box<dyn Error>> {
    let input = "\
DEMO
VEHICLE
2 10
CUSTOMER
0 0 0 0 0 100 0
1 1 0 5 0 100 0
2 0 1 5 0 100 0
";
    let instance: SolomonInstance = input.parse()?;
    let mut solution = nearest_neighbor(&instance);

    assert!(solution.is_feasible(&instance));
    let before = solution.total_distance(&instance);
    two_opt(&mut solution, &instance);
    assert!(solution.is_feasible(&instance));
    assert!(solution.total_distance(&instance) <= before);
    Ok(())
}
```

`nearest_neighbor` panics on malformed instances, a customer whose demand
exceeds vehicle capacity, or exhaustion of the configured fleet. Fleet
exhaustion only means this greedy construction failed. If instance data are
constructed or changed through public fields, call `SolomonInstance::validate`
before passing them to CPU routines.

To use the optional GPU API in the published alpha, enable the feature:

```toml
[dependencies]
vrp-gpu = { version = "0.1.1-alpha", features = ["gpu"] }
```

`gpu::evaluate_two_opt_deltas` returns a row-major `n × n` matrix with computed
`f32` deltas for valid `i < j` and positive infinity elsewhere. Extreme finite
input distances can still yield non-finite arithmetic results.
`gpu::best_two_opt_move` returns the best finite negative delta as
`Option<TwoOptMove>`. Exact ties choose
the lowest row-major `(i, j)`. Both functions return `GpuEvaluationError` for
invalid input, size overflow or CUDA driver failure. Empty and singleton routes
return without creating a CUDA context. A returned move is only a proposal:
apply it with `cpu::apply_two_opt` if your application accepts it.

## Development-only GPU search API

On the development branch, `gpu::two_opt_route` repeatedly selects and applies
best-improvement moves to one route. `gpu::two_opt` does the same independently
for each route in a solution. Both return `GpuSearchReport` with accepted move
count and distance improvement computed by `f64` accumulation of the `f32`
matrix. Each accepted reversal must strictly decrease recomputed cost. A
validation, CUDA, or selected-move consistency error leaves the input
unchanged. A solution search does not itself establish fleet, capacity, or
customer-coverage feasibility; check `Solution::is_feasible` when needed.

The following example uses the development-branch search API. It compiles
without running in rustdoc because execution requires a compatible NVIDIA GPU
and driver:

```rust,no_run
#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use vrp_gpu::{
        construct::nearest_neighbor,
        instance::SolomonInstance,
        local_search::gpu,
    };

    let input = r#"DEMO
VEHICLE
1 10
CUSTOMER
0 0 0 0 0 100 0
1 1 0 0 0 100 0
2 0 1 0 0 100 0
3 2 1 0 0 100 0
"#;
    let instance: SolomonInstance = input.parse()?;
    let mut solution = nearest_neighbor(&instance);
    let report = gpu::two_opt(&mut solution, &instance)?;
    assert!(solution.is_feasible(&instance));
    println!("accepted {} moves", report.accepted_moves);
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {}
```

After enabling `gpu`, one selection and application can be written as follows.
The documentation build compiles this example without running it because a
GPU and driver are required:

```rust,no_run
#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use vrp_gpu::{
        instance::SolomonInstance,
        local_search::{cpu, gpu},
        solution::Route,
    };

    let input = r#"DEMO
VEHICLE
1 10
CUSTOMER
0 0 0 0 0 100 0
1 1 0 0 0 100 0
2 0 1 0 0 100 0
3 2 1 0 0 100 0
"#;
    let instance: SolomonInstance = input.parse()?;
    let mut route = Route::from_nodes(vec![1, 2, 3]);
    if let Some(best) = gpu::best_two_opt_move(&route, &instance)? {
        cpu::apply_two_opt(&mut route, best.i, best.j);
    }
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {}
```

Each nontrivial GPU selection currently creates a CUDA context, loads the
embedded PTX and transfers the matrix and route. A search repeats that work on
each iteration; there is no reusable device session. The checked-in artifact
targets `sm_120` and has been
hardware-validated on an NVIDIA GeForce RTX 5060 Ti. Other devices have no
compatibility claim. The [recorded validation](https://github.com/pedrozaz/vrp-gpu/blob/develop/docs/benchmarks/2026-09-22-cpu-gpu.md)
found this end-to-end GPU API slower than the CPU reference for all measured
nontrivial routes; no speedup is claimed. See the [input and algorithm guide](https://github.com/pedrozaz/vrp-gpu/blob/develop/docs/user-guide.md)
for exact model and numerical contracts.

## Minimum supported Rust version

The supported feature set requires Rust 1.88 or newer (edition 2024). Raising
the MSRV is a release-level change and must be recorded in the changelog.

See the [repository documentation](https://github.com/pedrozaz/vrp-gpu/tree/develop/docs)
for architecture, kernel contracts and release procedures.

Licensed under Apache-2.0.
