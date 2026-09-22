# Contributing to VRP-GPU

Thank you for contributing. This document defines the repository boundaries
and evidence required for a pull request. Start at the repository root unless
a command explicitly changes directory.

## Development environments

The repository intentionally has two Cargo workspaces:

- The root workspace contains `vrp-gpu`, `vrp-gpu-cli` and `vrp-gpu-bench`.
  It uses stable Rust and contains everything needed by library users.
- `crates/vrp-gpu-kernel` is a standalone, unpublished workspace. It uses the
  pinned nightly and cuda-oxide toolchain to regenerate `kernel.ptx`.

For CPU algorithms, parsing, documentation and the CUDA **host** code, install
stable Rust with `rustfmt` and `clippy`, plus `just` and `cargo-deny` for the
full gate. The public crate declares MSRV 1.88; test compatibility against
that version when changing public code or dependencies. No CUDA compiler is
needed for CPU changes.

For kernel source changes, use the nightly pinned in
`crates/vrp-gpu-kernel/rust-toolchain.toml`, the pinned cuda-oxide Git revision,
and the toolchain requirements of that revision. The kernel workspace is
excluded from the stable root workspace. See
[kernel development](docs/kernel-development.md) for setup, regeneration and
hardware checks. Do not add a dependency from the root workspace to the
kernel package.

## Git workflow

- `main`: tagged releases only.
- `develop`: integration branch.
- `feature/<slug>`: changes branched from `develop`.
- `release/<vX.Y.Z>`: release stabilization.
- `hotfix/<slug>`: critical fixes branched from `main`.

Use atomic Conventional Commits such as `feat(gpu): ...` or
`docs(release): ...`. Write code, documentation, commit messages and PRs in
English. Every commit must contain a Developer Certificate of Origin trailer
and a cryptographic signature:

```sh
git commit -s -S
```

Feature branches merge into `develop` through a pull request using a
non-fast-forward merge. Keep data-only benchmark records distinct from the
implementation that produced them. Never rewrite another contributor's
published commits to add a signature.

## Quality gates

Run the stable workspace checks before opening a pull request:

```sh
just ci
```

This checks formatting, all targets and features, Clippy, tests, rustdoc,
package contents and dependency policy. `cargo test` skips the existing GPU
hardware tests because they are marked `#[ignore]`; passing `just ci`
alone is not evidence of GPU execution. To run parts independently, use
`just --list`. The package and dependency-policy stages may need network
access for the crates.io index or advisory database.

Use the [validation matrix](docs/testing.md) to select additional checks for
CPU, GPU, package or documentation changes and to report exactly what ran.

Changes to kernel source or the PTX artifact also require:

```sh
just kernel-ci
cd crates/vrp-gpu-kernel
cargo oxide inspect --arch sm_120
cmp vrp_gpu_kernel.ptx ../vrp-gpu/kernel.ptx
```

GPU behavior must be checked against the CPU reference on supported hardware.
Hardware-only tests stay marked `#[ignore]` so CPU-only CI remains portable.
Regenerate PTX from the pinned source; never edit the checked-in PTX by hand.

## Pull requests

A pull request should:

1. Explain the user-visible behavior and scope.
2. Identify tests performed locally, including hardware when relevant.
3. Update public rustdoc, README material and `CHANGELOG.md` when behavior changes.
4. Keep unrelated refactoring in separate commits or pull requests.
5. Avoid performance claims without a reproducible benchmark record.

For a fork that changes public behavior, also state the supported input model,
error and panic behavior, numerical tolerance, and any new hardware assumption.
Tests should compare GPU output with the CPU oracle and, for 2-opt changes,
with a full route-cost recomputation. Do not infer support for another NVIDIA
architecture from the existence of an `sm_120` artifact.

Publication and release steps are documented in [docs/releasing.md](docs/releasing.md).
