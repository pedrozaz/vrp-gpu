# CPU/GPU 2-opt validation protocol

## Scope

This branch compares the existing CPU and GPU `best_two_opt_move` APIs on
identical routes. It implements the correctness-first and reproducible timing
requirements of the project foundation. It does not implement a convergence
loop or compare complete solvers with the external `vrp-cli` tool. Those require
a separate solver integration and equal search budgets before solution-quality
claims are meaningful.

The unpublished `vrp-gpu-bench` package owns this runner. The public library,
kernel, PTX and runtime behavior are unchanged.

## Reproduce

```sh
cargo run --release --locked -p vrp-gpu-bench --features gpu -- \
  --output /tmp/comparison.csv --samples 20 --warmup 3 \
  --solomon /path/to/C101.100.txt
python3 docs/benchmarks/summarize.py /tmp/comparison.csv
```

`--solomon` can be repeated for local Solomon files; omit it for synthetic cases
only. No dataset or external solver is downloaded by the runner. Preserve the
source URL, exact revision and SHA-256 for every input in the accompanying report.
The output path must be new: existing reports are never overwritten. A failure
in validation or a measured call returns an error before creating the CSV.

Synthetic routes have 0, 1, 2, 16, 17, 65, 257, 1024 and 2048 customers. Coordinates
are `(17*i mod 997, 29*i mod 991)`; route order is a Fisher-Yates shuffle with
32-bit wrapping LCG seed `0x12345678`, multiplier 1664525 and increment 1013904223.
These are reproducible stress fixtures, not Solomon benchmark instances.
Solomon cases use the existing nearest-neighbor constructor, outside timing;
the initial complete solution must pass capacity/fleet/customer feasibility.
Time windows are parsed but not enforced: the project currently models CVRP.

## Correctness gates

Before any measurement, for every route:

1. Compare every valid GPU delta (`i < j`) with `cpu::two_opt_delta`, requiring
   finite results and `abs(actual - expected) <= 1e-5 * max(1, abs(expected))`.
2. Require all other cells to equal positive infinity and the matrix size to
   equal `n*n`.
3. Compare CPU/GPU selected indices exactly, including exact-tie ordering, and
   selected deltas using the same tolerance. `None` must match `None`.
4. Reverse the selected segment in a copy and independently sum all route edges
   in `f64`. Require strict cost decrease and agreement with the selected delta
   to `1e-5 * max(1, sum of the four affected edge lengths)`. The matrix remains
   the original `f32` matrix; the accumulation is independent of the delta formula.

Every warm GPU call and every timed CPU/GPU result is also checked against the
reference selection after stopping the clock. Reversal preserves customer IDs,
route demand and fleet size. A no-move result is not a proof of global optimality.
Portable tests deliberately corrupt matrix lengths, invalid cells, finite values,
selection indices and reported deltas to verify the validators reject them.

## Timing contract

Use release builds. Both paths receive the same immutable route and instance.
Construction, parsing, all-delta validation, reversal checking, CSV serialization
and console output are outside measured intervals. Each route receives three
unmeasured calls per backend, then 20 samples (configurable). CPU-first and
GPU-first order alternate. Inputs and outputs use `std::hint::black_box`.

Each sample measures one complete synchronous API call using `Instant`:

- CPU: the single-threaded scan and selection on an already validated instance.
- GPU: the public API, including its input validation, context/module management,
  allocations, transfers, delta evaluation, reductions, synchronization and
  download. The runner does not introduce GPU resource caching.

These are **public-API latency measurements**, not kernel-only times or an
equal-validation-overhead microbenchmark. Small CPU times are especially sensitive
to timer overhead. Empty/singleton GPU calls use CPU fast paths. Initial CUDA/JIT
startup is exercised by validation and excluded; warm calls still include the
resource management performed internally by the existing API.

For each case/route report medians and nearest-rank p95, retain raw nanoseconds,
and compute:

- valid candidates: `n*(n-1)/2` (not the GPU's `n*n` output cells);
- effective candidates/second: valid candidates divided by median API seconds;
- ratio: median CPU time divided by median GPU time (below 1 means GPU slower).

For fewer than two customers, report throughput and ratios as not applicable.
Report before/after single-move cost, not final solver quality. Measurements on a
shared desktop are descriptive, not a statistically established crossover or
speedup claim; record whether clocks, affinity and load were controlled.

## Results and coverage limits

Append dated CSV and Markdown reports here. Record the exact source commit,
command, samples/warmups, dataset provenance, PTX checksum, CPU, GPU, OS,
compiler and driver. Never infer all six Solomon families from a single C101
run. External `vrp-cli`, solver-level budgets and convergence remain future work.
