# VRP-GPU

Massively parallel GPU acceleration for Vehicle Routing Problem (VRP) local search in Rust.
This project evaluates candidate local search moves (such as 2-opt) in batch using custom PTX kernels
compiled with [cuda-oxide](https://github.com/NVLabs/cuda-oxide).

## Project Status

> **Status**: **Experimental / Active Alpha**.
> This project is in its early experimental phase. Internal APIs, kernel signatures, and abstractions are subject to breaking changes.

## System Requirements

- **GPU**: NVIDIA GPU with Compute Capability `sm_120` or higher (Blackwell consumer architecture).
- **CPU Fallback**: Full CPU reference implementation available (no NVIDIA GPU required for CPU-only execution).
- **Operating System**: Linux (developed and verified on Arch Linux with rustup nightly toolchain).
- **Rust Toolchain**:
    - Stable edition for published crates (`vrp-core`, `vrp-bench`, `vrp-cli`).
    - Nightly toolchain pinned via `rust-toolchain.toml` specifically for `vrp-kernel` (dev-only).

## Minimal Installation & Usage

Add `vrp-core` to your `Cargo.toml`:

```toml
[dependencies]
vrp-core = { version = "0.1.0", features = ["gpu"] }
```

Basic usage example:

```rust
use vrp_core::{
    instance::SolomonInstance,
    local_search::gpu::evaluate_two_opt_deltas,
    solution::Route,
};

fn evaluate_route(
    input: &str,
    route_nodes: Vec<usize>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let instance: SolomonInstance = input.parse()?;
    let route = Route::from_nodes(route_nodes);
    Ok(evaluate_two_opt_deltas(&route, &instance)?)
}
```

The GPU API returns a row-major `route_len × route_len` matrix. Entries with
`i < j` contain the 2-opt delta; all other entries are `f32::INFINITY`.

To select just the best improving move, use
`local_search::gpu::best_two_opt_move(&route, &instance)`. It returns
`Result<Option<TwoOptMove>, GpuEvaluationError>` without changing the route.
The GPU reduces finite negative deltas in 256-thread blocks, repeating until
one candidate remains; only its `f32` delta and `u32` index are downloaded.
Exact ties select the smallest `(i, j)` in row-major order; `None` means no
finite negative delta exists. `local_search::cpu::best_two_opt_move` is the
CPU reference and is available without the `gpu` feature.

This API still evaluates one route at a time and retains the full delta matrix
on the device. Applying moves, iterating to convergence, persistent GPU state,
and performance benchmarks are separate steps.
See [the reduction contract and QA commands](docs/2opt-gpu-reduction.md).

## Benchmarks

Historical benchmark results comparing CPU vs. GPU throughput and solution quality against `vrp-cli` are versioned in [docs/benchmarks](./docs/benchmarks/).

## Contributing

Please review [CONTRIBUTING.md](./CONTRIBUTING.md) for development environment setup, Git branching strategies, and commit conventions.

## License

Distributed under the **Apache-2.0** License. See [LICENSE](./LICENSE) for details.
 
