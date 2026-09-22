# Kernel development and PTX regeneration

The public crate embeds `crates/vrp-gpu/kernel.ptx` with `include_str!`.
Consumers compile the stable `vrp-gpu` crate and do not build the separate
kernel workspace. The source of the PTX is
`crates/vrp-gpu-kernel/src/lib.rs`; its `Cargo.toml` pins cuda-oxide to an
exact Git revision and its `rust-toolchain.toml` pins a nightly compiler.

## Prerequisites

Work on a machine configured for the pinned cuda-oxide revision. Install the
pinned nightly and its listed components, the `cargo-oxide` subcommand, and
the LLVM/CUDA tools required by that revision. Check the actual revision's
setup instructions and `cargo oxide doctor` before changing a kernel: this
toolchain changes independently of the stable host crate. The current
generator command targets `sm_120`. A successful PTX build is not by itself
proof that a CUDA driver will load or execute the artifact on a given GPU.

## Regenerate and review

From the repository root:

```sh
just kernel-ci
cd crates/vrp-gpu-kernel
cargo oxide doctor
cargo oxide inspect --arch sm_120
cmp vrp_gpu_kernel.ptx ../vrp-gpu/kernel.ptx
```

`cargo oxide inspect` writes `vrp_gpu_kernel.ptx` for this package. The final
`cmp` checks whether the generated artifact matches the committed one. When
the difference is intentional, inspect the generated PTX, copy it to
`../vrp-gpu/kernel.ptx`, rerun `cmp`, and commit kernel source and PTX in the
same PR. Do not copy the similarly named `vrp_kernel.ptx`; that file can be
left over from an earlier package identity. Generated PTX and LLVM
intermediates in the kernel workspace are ignored by Git; the public crate's
PTX is versioned and included in its archive.

Inspect the resulting `.version`, `.target`, exported entry names, parameter
order/types, launch dimensions, and shared-memory requirements. The host
launch builder in `src/local_search/gpu.rs` must match the actual PTX ABI.
For the reduction kernel, all 256 lanes must reach every barrier, even when
their input offset is out of range; see the detailed
[reduction contract](2opt-gpu-reduction.md).

## Validate behavior

Return to the repository root and run `just ci`. On the target hardware, run:

```sh
cargo test -p vrp-gpu --features gpu -- --ignored --test-threads=1
```

The tests compare GPU deltas and selected moves with CPU references. Record
the GPU model, driver, PTX target, compiler revision and test output. If
memory access or synchronization changed, run the relevant Compute Sanitizer
checks documented in [the reduction contract](2opt-gpu-reduction.md).
Do not claim support for another architecture without regeneration and
hardware validation for that architecture.
