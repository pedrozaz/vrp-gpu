# vrp-gpu

`vrp-gpu` provides CVRP data structures, CPU reference heuristics and optional
CUDA evaluation of 2-opt moves. It parses Solomon-style text, constructs routes
with a greedy nearest-neighbor heuristic, and improves routes on the CPU.
The GPU API computes candidate deltas or selects a single move; it does not run
a complete GPU search loop or mutate a route.

This is an experimental alpha. Check crates.io for available versions; a
version in the repository does not mean the package has been published.

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `gpu` | No | Exposes `local_search::gpu`, adds `cudarc`, and embeds the checked-in PTX targeting `sm_120` |

Without `gpu`, the library has no third-party runtime dependencies. Enabling
`gpu` requires a working NVIDIA driver when a GPU operation is executed. The
PTX is precompiled; consumers do not need cuda-oxide or a nightly Rust compiler.

## Installation

Before publication, use the Git repository and pin a tested revision for a
reproducible application build:

```toml
[dependencies]
vrp-gpu = { git = "https://github.com/pedrozaz/vrp-gpu", rev = "<tested-commit>" }
```

Once `0.1.0-alpha.1` is available on crates.io, opt in to that prerelease
explicitly:

```toml
[dependencies]
vrp-gpu = "0.1.0-alpha.1"
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

To use the optional GPU API, enable the feature in the dependency:

```toml
[dependencies]
vrp-gpu = { git = "https://github.com/pedrozaz/vrp-gpu", rev = "<tested-commit>", features = ["gpu"] }
```

Replace `<tested-commit>` with the exact Git commit you validated.
`gpu::evaluate_two_opt_deltas` returns a row-major `n × n` matrix with computed
`f32` deltas for valid `i < j` and positive infinity elsewhere. Extreme finite
input distances can still yield non-finite arithmetic results.
`gpu::best_two_opt_move` returns the best finite negative delta as
`Option<TwoOptMove>`. Exact ties choose
the lowest row-major `(i, j)`. Both functions return `GpuEvaluationError` for
invalid input, size overflow or CUDA driver failure. Empty and singleton routes
return without creating a CUDA context. A returned move is only a proposal:
apply it with `cpu::apply_two_opt` if your application accepts it.

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

Each nontrivial GPU call currently creates a CUDA context, loads the embedded
PTX and transfers the matrix and route. There is no reusable device session or
GPU convergence loop. The checked-in artifact targets `sm_120` and has been
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
