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

`vrp-gpu-cli` is currently a placeholder, not a supported solver command.
`vrp-gpu-bench` is an internal tool whose protocols and results must be
documented under [benchmark records](benchmarks/README.md). Read the
[library guide](user-guide.md) for the public APIs.

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
The PTX currently declares `.target sm_120`. The package includes this one
artifact; it does not compile PTX for the consumer's GPU at build time.

## Correctness boundary

CPU local-search code is the reference implementation. GPU work is accepted
only after parity tests compare candidate deltas and deterministic selection
against that reference. Floating-point comparisons use documented tolerances;
tie-breaking uses exact row-major order.

The shared input representation is a row-major `f32` distance matrix of N²
entries plus customer IDs in route order. The implemented 2-opt delta formula
changes two boundary edges and assumes symmetric distances. CPU best-move
selection evaluates O(n²) candidate pairs for a route of n customers, using
O(1) additional selection state. The GPU evaluator materializes n² `f32`
deltas on device; the reduction repeatedly shrinks them to one candidate.
The full-matrix GPU API also downloads n² deltas. These are algorithmic and
allocation bounds, not measured speedup claims.

The GPU host functions validate the instance and route, create a context,
load the embedded module, transfer inputs, launch kernels and synchronize on
each nontrivial call. There is no persistent context cache or GPU route
convergence API. `best_two_opt_move` returns a proposal; applying it and
repeating search remain the caller's responsibility.

## Public API boundary

Public modules live in `crates/vrp-gpu/src`. Workspace binaries may depend on
the library, but the library must not depend on them or on the kernel workspace.
New public APIs require rustdoc, tests and a changelog entry. Feature-gated APIs
must continue to allow the default CPU-only package to compile without CUDA.
The public model currently covers capacitated routing only. Solomon time-window
and service fields are retained but not enforced by route construction,
feasibility checks or 2-opt. See [model semantics](user-guide.md).

## Package contents

The public manifest uses an explicit allowlist containing source, the crate
README and `kernel.ptx`. Cargo always adds normalized manifest and lock metadata
to the archive. Validate the exact archive before a release with:

```sh
cargo package -p vrp-gpu --list
cargo package -p vrp-gpu --locked
```

See [kernel development](kernel-development.md) for PTX regeneration and
[testing](testing.md) for the evidence expected of a change.
