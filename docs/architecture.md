# Architecture and publication boundaries

VRP-GPU separates the code shipped to library users from the toolchain used to
author CUDA kernels. This keeps the published package compatible with stable
Rust and prevents a Git-only compiler dependency from leaking into crates.io.

## Package map

```text
vrp-gpu/
├── Cargo.toml                    # Stable virtual workspace
├── crates/
│   ├── vrp-gpu/                  # Public library package
│   │   ├── src/                  # CPU algorithms and CUDA host orchestration
│   │   └── kernel.ptx            # Versioned runtime artifact
│   ├── vrp-gpu-cli/              # Internal, publish = false
│   ├── vrp-gpu-bench/            # Internal, publish = false
│   └── vrp-gpu-kernel/           # Standalone nightly workspace, publish = false
└── docs/
```

Only `vrp-gpu` is intended for crates.io. The other packages use project-
qualified names to avoid collisions with unrelated crates and are explicitly
marked `publish = false`.

## Runtime boundary

The default feature set is CPU-only. Enabling `gpu` adds `cudarc` and exposes
CUDA host orchestration. The library loads the committed PTX artifact at runtime;
it does not compile kernels in a consumer build.

```text
vrp-gpu-kernel (nightly + cuda-oxide, development only)
                         |
                         | generates and reviews
                         v
              crates/vrp-gpu/kernel.ptx
                         |
                         | embedded by include_str!
                         v
          vrp-gpu --features gpu (stable + cudarc)
```

Kernel source and generated PTX must change together. The checked-in artifact
is part of the public package and therefore part of its compatibility surface.

## Correctness boundary

CPU local-search code is the reference implementation. GPU work is accepted
only after parity tests compare candidate deltas and deterministic selection
against that reference. Floating-point comparisons use documented tolerances;
tie-breaking uses exact row-major order.

## Public API boundary

Public modules live in `crates/vrp-gpu/src`. Workspace binaries may depend on
the library, but the library must not depend on them or on the kernel workspace.
New public APIs require rustdoc, tests and a changelog entry. Feature-gated APIs
must continue to allow the default CPU-only package to compile without CUDA.

## Package contents

The public manifest uses an explicit allowlist containing source, the crate
README and `kernel.ptx`. Cargo always adds normalized manifest and lock metadata
to the archive. Validate the exact archive before a release with:

```sh
cargo package -p vrp-gpu --list
cargo package -p vrp-gpu --locked
```
