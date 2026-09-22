"""Summarize raw API latencies; standard-library only, never modifies inputs."""

import csv
import math
import statistics
import sys
from collections import defaultdict


def summarize(path):
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            n = int(row["customers"])
            assert int(row["candidates"]) == n * (n - 1) // 2
            assert int(row["cpu_ns"]) > 0 and int(row["gpu_ns"]) > 0
            groups[(row["case"], int(row["route"]), n)].append(row)
    assert groups, "empty report"
    print("| Case/route | n | Samples | CPU median us | GPU median us | CPU p95 us | GPU p95 us | CPU/GPU | CPU candidates/s | GPU candidates/s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (case, route, n), rows in groups.items():
        assert [int(r["sample"]) for r in rows] == list(range(len(rows)))
        for index, row in enumerate(rows):
            assert row["order"] == ("cpu-first" if index % 2 == 0 else "gpu-first")
            for field in ("selected_i", "selected_j", "delta", "cost_before_f64", "cost_after_f64"):
                assert row[field] == rows[0][field], f"non-deterministic {field}"
        cpu = sorted(int(r["cpu_ns"]) for r in rows)
        gpu = sorted(int(r["gpu_ns"]) for r in rows)
        c, g = statistics.median(cpu), statistics.median(gpu)
        p95 = math.ceil(len(rows) * 0.95) - 1
        candidates = n * (n - 1) // 2
        metrics = f"{c / g:.6f} | {candidates * 1e9 / c:.0f} | {candidates * 1e9 / g:.0f}" if n >= 2 else "N/A | N/A | N/A"
        print(f"| {case}/{route} | {n} | {len(rows)} | {c / 1e3:.3f} | {g / 1e3:.3f} | {cpu[p95] / 1e3:.3f} | {gpu[p95] / 1e3:.3f} | {metrics} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 docs/benchmarks/summarize.py report.csv")
    summarize(sys.argv[1])
