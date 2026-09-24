# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and published versions will follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Host-orchestrated GPU-selected 2-opt search for one route or every route in a
  solution. Accepted moves are checked against independently recomputed route
  cost, and search errors leave the caller's data unchanged.

## [0.1.1-alpha] - 2026-09-22

Documentation-only prerelease. No Rust API, algorithm, dependency, or PTX
behavior changed from `0.1.0-alpha.1`.

### Changed

- Updated the packaged README and installation examples to describe an alpha
  that is available on crates.io rather than a future publication.
- Clarified the package description to distinguish optional CUDA move
  evaluation from a claimed end-to-end GPU speedup.

## [0.1.0-alpha.1] - 2026-09-22

First public alpha release of the `vrp-gpu` library. The CLI, benchmark harness,
and kernel compiler workspace are not published.

### Added

- Solomon-style instance parsing, CVRP route and solution types, greedy
  nearest-neighbor construction, and CPU 2-opt local search.
- An opt-in `gpu` feature for CUDA 2-opt delta evaluation and deterministic
  best-move selection, with the versioned `sm_120` PTX embedded in the crate.
- CPU/GPU correctness and latency validation records, plus public API,
  contributor, kernel, security, and release documentation.

### Known limitations

- The GPU path is validated only on the NVIDIA GeForce RTX 5060 Ti (`sm_120`);
  no compatibility claim is made for other devices.
- The GPU API evaluates one batch and does not run a solver to convergence.
  Its end-to-end latency was higher than the CPU reference in the recorded
  validation cases with at least two customers.
- The supported input model is CVRP; Solomon time windows are parsed but not
  enforced by the current construction or local-search routines.

[Unreleased]: https://github.com/pedrozaz/vrp-gpu/compare/v0.1.1-alpha...develop
[0.1.1-alpha]: https://github.com/pedrozaz/vrp-gpu/compare/v0.1.0-alpha.1...v0.1.1-alpha
[0.1.0-alpha.1]: https://github.com/pedrozaz/vrp-gpu/releases/tag/v0.1.0-alpha.1
