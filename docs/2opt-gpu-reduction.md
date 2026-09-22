# Deterministic 2-opt reduction

`cpu::best_two_opt_move` and `gpu::best_two_opt_move` select the minimum finite
negative delta. Exact ties use the smallest `(i, j)` lexicographically (equivalent
to the smallest index `i * n + j`). Zero, signed zero, NaN and infinities are not
improving candidates. No floating-point tolerance is used in ordering: approximate
equality is not transitive and is unsuitable for a parallel reduction tree.

The GPU API validates inputs using the same path as `evaluate_two_opt_deltas`.
Empty and singleton routes return `None` without CUDA. A successful selection
does not mutate the route or apply the move. Non-finite arithmetic results are
ignored under the candidate contract, not reported as a proof of local optimality.

## Kernel and host contract

The existing evaluation kernel writes an n² row-major device buffer. The new
`reduce_two_opt_candidates` kernel launches exactly 256 threads per block:

1. Each lane loads one input or the neutral pair `(0.0, u32::MAX)` for padding.
2. The first pass masks `i >= j` and derives original row-major indices. Later
   passes retain those indices, never replacing them with temporary positions.
3. A shared-memory tree compares delta, then original index. Every lane reaches
   every barrier, including lanes outside the input length.
4. Lane zero writes one pair per block to disjoint output buffers.
5. The host repeats reductions on the same stream until one pair remains and
   downloads only two scalars (eight payload bytes).

The PTX ABI is, in order: values pointer/u64 length, indices pointer/u64 length,
u32 route width, output values pointer/u64 length, output indices pointer/u64
length. Width zero denotes a later pass. The first pass does not read the index
buffer. Each output buffer has `ceil(input_count / 256)` entries. The host must
use a 1D grid, a `(256, 1, 1)` block, and distinct input/output allocations.

All launches use the versioned `kernel.ptx` built for `sm_120` from the pinned
cuda-oxide dependency. The root workspace does not depend on the kernel crate.
The full n² delta buffer is still allocated on device. Resource caching, fusing
evaluation with reduction, multi-route batching, and convergence are outside
this feature; no speedup claim is made.

## Validation

From the repository root:

```sh
just ci
cargo test --workspace --no-default-features
cargo test -p vrp-gpu --features gpu -- --ignored
cargo test -p vrp-gpu --features gpu --release -- --include-ignored
```

The synthetic reduction tests compare with an independent sorted CPU oracle.
They cover invalid triangle cells with negative values, no improvement, signed
zero, non-finite values, exact ties across blocks/passes, adjacent f32 values,
repeatability, full/partial blocks and widths 1, 2, 15, 16, 17, 255, 256, 257.
Width 257 requires three reduction passes. End-to-end tests cover ten route
lengths and three coordinate scales, comparing indices and deltas with the CPU
oracle and checking the selected reversal by recomputing total route cost in f64.
GPU delta comparisons use `1e-5 * max(1, abs(expected))`; synthetic values and
tie indices are checked exactly. Input rejection and small routes run in CPU CI.

Build the test executable with `cargo test -p vrp-gpu --features gpu --no-run`.
Use the executable path reported by Cargo for each sanitizer command:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 <test-executable> reduction_tests --ignored --test-threads=1
compute-sanitizer --tool racecheck --error-exitcode 1 <test-executable> reduction_tests --ignored --test-threads=1
compute-sanitizer --tool synccheck --error-exitcode 1 <test-executable> reduction_tests --ignored --test-threads=1
compute-sanitizer --tool initcheck --error-exitcode 1 <test-executable> reduction_tests --ignored --test-threads=1
```

The kernel workspace has its own checks and regeneration command:

```sh
cd crates/vrp-gpu-kernel
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
cargo oxide inspect --arch sm_120
cmp vrp_gpu_kernel.ptx ../vrp-gpu/kernel.ptx
```
