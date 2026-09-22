# VRP-GPU

[![CI](https://github.com/pedrozaz/vrp-gpu/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/pedrozaz/vrp-gpu/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

VRP-GPU is an experimental Rust library for capacitated vehicle routing (CVRP).
It parses Solomon-style instances, builds routes with a greedy nearest-neighbor
heuristic, and provides CPU 2-opt local search. An optional CUDA path evaluates
2-opt candidates and selects one improving move. The CPU implementation is the
correctness reference for GPU validation.

> **Status:** active alpha. The crate has not been published to crates.io yet,
> and public APIs may change before the first stable release.

## Workspace

| Package | Role | Published |
| --- | --- | --- |
| `vrp-gpu` | Public library: instances, solutions, CPU heuristics and optional CUDA orchestration | Planned |
| `vrp-gpu-cli` | Internal command-line frontend | No |
| `vrp-gpu-bench` | Internal benchmark harness | No |
| `vrp-gpu-kernel` | Standalone nightly workspace that generates the versioned PTX artifact | No |

The kernel package is deliberately excluded from the stable root workspace.
Users of `vrp-gpu` receive the generated PTX and do not need cuda-oxide, LLVM or
the pinned nightly compiler.

## Using the library before publication

Until the first crates.io release, depend on the repository explicitly:

```toml
[dependencies]
vrp-gpu = { git = "https://github.com/pedrozaz/vrp-gpu", default-features = false }
```

Enable CUDA host orchestration with the opt-in feature:

```toml
[dependencies]
vrp-gpu = { git = "https://github.com/pedrozaz/vrp-gpu", features = ["gpu"] }
```

The default feature set is CPU-only. See the
[crate-specific README](crates/vrp-gpu/README.md) for a minimal example and the
public feature contract. The current internal CLI prints a placeholder message;
use the library API for actual routing work.

## Compatibility

- **Rust:** MSRV 1.88 for the supported feature set; edition 2024.
- **CPU path:** does not require CUDA or an NVIDIA GPU.
- **GPU path:** currently developed and hardware-validated on Linux with an
  NVIDIA GeForce RTX 5060 Ti (`sm_120`). Broader hardware support is not yet
  claimed. A CUDA driver and a device able to load the checked-in PTX are
  required at runtime; enabling the Cargo feature alone does not supply them.
- **Kernel development:** uses the toolchain pinned in
  `crates/vrp-gpu-kernel/rust-toolchain.toml` and the pinned cuda-oxide revision.

## Documentation

- [Library usage and API contract](crates/vrp-gpu/README.md)
- [Input model and algorithm semantics](docs/user-guide.md)
- [Architecture and publication boundaries](docs/architecture.md)
- [Deterministic 2-opt reduction](docs/2opt-gpu-reduction.md)
- [Validation matrix and evidence rules](docs/testing.md)
- [Kernel development and PTX regeneration](docs/kernel-development.md)
- [Release and crates.io checklist](docs/releasing.md)
- [Benchmark record format](docs/benchmarks/README.md)
- [Changelog](CHANGELOG.md)

## Development

Run the complete stable-workspace quality gate from the repository root:

```sh
just ci
```

Kernel validation remains explicit and separate:

```sh
just kernel-ci
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch, commit and pull-request
workflow.

## Security

Please report vulnerabilities according to [SECURITY.md](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
