# Benchmark Records

This directory stores versioned CSV and Markdown benchmark results.
Do not overwrite previous benchmark logs; append new reports including execution
timestamp, commit hash, and hardware specifications.

The `vrp-gpu-bench` package is an internal harness and is not published to
crates.io. Benchmark reports must include the exact feature set, dataset,
compiler, driver and GPU model required to reproduce a result.
