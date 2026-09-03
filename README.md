# VRP-GPU

Massively parallel GPU acceleration for Vehicle Routing Problem (VRP) local search in Rust.
This project evaluates candidate local search moves (such as 2-opt) in batch using custom PTX kernels
compiled with [cuda-oxide](https://github.com/NVLabs/cuda-oxide).

## Project Status

> **Status**: /! **Experimental / Active Alpha**.
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

```Rust
use vrp_core::instance::SolomonInstance;
use vrp_core::local_search::gpu::GpuLocalSearch;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load Solomon instance and run 2-opt local search
    Ok(())
}
```

## Benchmarks

Historical benchmark results comparing CPU vs. GPU throughput and solution quality against `vrp-cli` are versioned in [docs/benchmarks](./docs/benchmarks/).

## Contributing

Please review [CONTRIBUTING.md](./CONTRIBUTING.md) for development environment setup, Git branching strategies, and commit conventions.

## License

Distribuited under the **Apache-2.0** License. See [LICENSE](./LICENSE) for details
 
