# Validation matrix and evidence rules

Run commands from the repository root unless stated otherwise. Record the
exact commands and results in a PR. CI is portable and cannot establish GPU
behavior without hardware. A command that did not run must be reported as
such, even when an adjacent command passed.

| Change | Required local evidence | Additional evidence |
| --- | --- | --- |
| Documentation only | `just ci` when commands, examples, package metadata or rustdoc change; check local links and executable examples | Note commands that depend on unavailable hardware or network. |
| CPU/parser/public API | `just ci`; `cargo test --workspace --no-default-features --locked`; `cargo +1.88.0 check -p vrp-gpu --all-features --locked` for API or dependency changes | Add tests for malformed public input, bounds, ties and full-cost agreement where applicable. |
| CUDA host or PTX | All applicable stable-workspace checks; `just kernel-ci` for kernel source; hardware tests with `--ignored` | Record GPU, driver, PTX target and toolchain; use Compute Sanitizer when changing synchronization or memory access. |
| Release/package | `just ci`; inspect `cargo package -p vrp-gpu --list` and `cargo package -p vrp-gpu --locked` from a clean tree | Follow [releasing](releasing.md) before publishing. |

The current root `just ci` runs `fmt`, all-feature check, strict Clippy,
all-feature tests, rustdoc with warnings denied, crate packaging and
`cargo deny`. `just kernel-ci` changes into the standalone kernel workspace
and runs formatting, check, strict Clippy and tests there. Neither recipe runs
hardware-only ignored tests automatically.

## GPU correctness protocol

```sh
cargo test -p vrp-gpu --features gpu -- --ignored --test-threads=1
```

Run this only where an NVIDIA driver and compatible GPU are available. It
runs all ignored tests of that package; review new ignored tests before using
the broad filter. The existing tests compare each valid GPU delta with the CPU
reference using `1e-5 * max(1, abs(cpu_delta))`, verify invalid matrix cells
exactly, and compare selected move indices exactly. Reduction tests also cover
boundary widths, non-finite values, exact ties and repeated runs. These tests
validate the checked-in PTX on the hardware used, not all possible devices.

For a new 2-opt implementation, test every `i < j` on representative routes
and compare the delta with the change obtained after actually reversing the
segment and recomputing complete route cost. Keep tie ordering exact. Use a
documented floating-point tolerance for cost comparisons; never use a
tolerance to decide which candidate wins the deterministic reduction.

`docs/2opt-gpu-reduction.md` records the current reduction ABI and optional
Compute Sanitizer commands. A sanitizer pass is evidence for the tested
configuration only. Record tool availability and exact command in the PR.

## Performance evidence

Correctness is a prerequisite for timing. A benchmark report belongs under
`docs/benchmarks/` and must include source revision, input hashes, hardware,
driver, compiler, feature flags, timing boundary, sample counts and raw
results. Distinguish full public API latency from kernel-only latency and
separate CPU/GPU move selection from a complete solver comparison. Never
describe a single selected improvement as convergence or a solution-quality
benchmark. See [benchmark records](benchmarks/README.md).
