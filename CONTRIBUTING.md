# Contributing to VRP-GPU

Thank you for contributing. This document describes the repository boundaries
and the checks expected before a pull request.

## Development environments

The repository intentionally has two Cargo workspaces:

- The root workspace contains `vrp-gpu`, `vrp-gpu-cli` and `vrp-gpu-bench`.
  It uses stable Rust and contains everything needed by library users.
- `crates/vrp-gpu-kernel` is a standalone, unpublished workspace. It uses the
  pinned nightly and cuda-oxide toolchain to regenerate `kernel.ptx`.

Do not introduce a root-workspace dependency on the kernel package. Generated
PTX is copied into `crates/vrp-gpu/kernel.ptx` and committed with its source
change.

## Git workflow

- `main`: tagged releases only.
- `develop`: integration branch.
- `feature/<slug>`: changes branched from `develop`.
- `release/<vX.Y.Z>`: release stabilization.
- `hotfix/<slug>`: critical fixes branched from `main`.

Use atomic Conventional Commits such as `feat(gpu): ...` or
`docs(release): ...`. Every commit must contain a Developer Certificate of
Origin trailer and a cryptographic signature:

```sh
git commit -s -S
```

Feature branches merge into `develop` through a pull request using a non-fast-
forward merge.

## Quality gates

Run the stable workspace checks before opening a pull request:

```sh
just ci
```

This checks formatting, all targets and features, Clippy, tests, rustdoc,
package contents and dependency policy. To run parts independently, use
`just --list`.

Changes to kernel source or the PTX artifact also require:

```sh
just kernel-ci
cd crates/vrp-gpu-kernel
cargo oxide inspect --arch sm_120
cmp vrp_gpu_kernel.ptx ../vrp-gpu/kernel.ptx
```

GPU behavior must be checked against the CPU reference on supported hardware.
Hardware-only tests stay marked `#[ignore]` so CPU-only CI remains portable.

## Pull requests

A pull request should:

1. Explain the user-visible behavior and scope.
2. Identify tests performed locally, including hardware when relevant.
3. Update public rustdoc, README material and `CHANGELOG.md` when behavior changes.
4. Keep unrelated refactoring in separate commits or pull requests.
5. Avoid performance claims without a reproducible benchmark record.

Publication and release steps are documented in [docs/releasing.md](docs/releasing.md).
