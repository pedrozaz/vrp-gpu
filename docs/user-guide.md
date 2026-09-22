# Input model and algorithm semantics

This guide describes the behavior of the public `vrp-gpu` library at the
current `develop` revision. Read the [crate README](../crates/vrp-gpu/README.md)
first for installation and a compilable CPU example. This is an experimental
API; pin a commit when consuming it from Git.

## Supported problem

The implemented objective is total route distance for a **CVRP**
with one depot (node `0`), a finite vehicle fleet, identical vehicle capacity
and nonnegative customer demands. `SolomonInstance` stores ready times, due
times and service times from Solomon input, but the constructive and 2-opt
algorithms do **not** enforce time windows or service schedules. They also do
not model multiple depots, heterogeneous fleets, or inter-route moves. The
parser builds Euclidean edge costs from coordinates; callers can construct a
different finite symmetric matrix manually if it satisfies `validate()`.

`Route::nodes` contains customer IDs only; travel from and back to depot `0`
is implicit. `Solution::is_feasible` checks the instance, fleet count, capacity,
and exactly one visit to each customer. It does not check time-window
feasibility. An empty route consumes no vehicle.

## Solomon input and invariants

Parse text with `input.parse::<SolomonInstance>()`. The parser expects a name,
`VEHICLE` section with count and capacity, and a `CUSTOMER` section with rows
containing at least seven whitespace-separated columns:

```text
ID X Y DEMAND READY DUE SERVICE
```

IDs must start at depot `0` and increase consecutively in row order. The
parser accepts seven or more columns and ignores extras. It is intended for
the Solomon-style layout exercised by this repository, not an arbitrary VRP
file format. Parsing reports `SolomonParseError`; malformed input is not a
normal reason to panic.

`SolomonInstance` has public fields, so manually built or mutated values must
be checked with `validate()` before use. The check requires:

- at least one node; positive vehicle count and finite positive capacity;
- all node arrays to have `num_nodes` finite entries;
- nonnegative demands, zero depot demand, and nonnegative service/ready times
  with `due >= ready`;
- a `num_nodes × num_nodes` row-major distance matrix with finite, nonnegative,
  exactly symmetric entries and a zero diagonal.

`compute_distance_matrix(xs, ys)` creates that matrix from equal-length
coordinate slices of `f32` values. It does not itself validate coordinates;
call `validate()` on the completed instance. Matrix construction and instance
validation each take O(N²) time and the matrix takes O(N²) memory for N nodes.

## Constructing and improving a solution

`nearest_neighbor(&instance)` checks the instance, then repeatedly chooses
the closest unvisited customer that fits the current route's remaining
capacity. Ties retain the first customer ID encountered. It closes a route
when no further customer fits. It panics for invalid instances, oversized
individual demands, or when its greedy choices need more than the available
vehicles. The last condition does not prove mathematical infeasibility.

The CPU reference offers two levels of 2-opt:

| API | Effect |
| --- | --- |
| `two_opt_delta(route, instance, i, j)` | Computes the edge-cost change for reversing `i..=j`, with `i < j`; does not mutate. |
| `best_two_opt_move(route, instance)` | Finds the smallest finite negative delta, breaking exact ties by first row-major `(i, j)`; does not mutate. |
| `apply_two_opt(route, i, j)` | Reverses that inclusive segment in place. |
| `two_opt_route(route, instance)` | Repeats best-improvement on one route until no improving move remains. |
| `two_opt(solution, instance)` | Runs that loop independently on each route. |

The index-based CPU functions expect valid indices and customer IDs. Their
index assertions are debug assertions; callers should not use invalid indices
in release builds. CPU 2-opt assumes the symmetric distance model validated
above; reversing a segment does not change the cost of its internal edges
under this model. Reported improvement sums selected `f32` deltas, which may
differ slightly from a freshly summed full route distance due to rounding.

## GPU evaluation

The `gpu` Cargo feature exposes two useful entry points:

| API | Result | Host transfer |
| --- | --- | --- |
| `gpu::evaluate_two_opt_deltas` | All `n²` route-position deltas, with invalid cells set to `+∞` | Downloads the full delta matrix. |
| `gpu::best_two_opt_move` | One finite negative move or `None` | Reduces on the device; downloads one delta and one index. |

Both APIs validate the full instance and route IDs before CUDA access. Empty
and singleton routes take a CPU-only fast path. The GPU selector does not
apply the move; it follows the same exact tie rule as the CPU selector.
Approximate floating-point equality is used only in parity tests, not when
ordering candidates. A `None` result means no *finite negative delta was
selected* under the supplied data; it is not a general optimality proof.

The current GPU host implementation creates a context, loads the embedded PTX,
allocates and transfers data for each nontrivial call. Plan capacity around an
`n²` delta buffer plus temporary reduction buffers. GPU evaluation does not
provide route convergence, batching across routes, or an integrated solver.
The PTX targets `sm_120`; hardware verification has been performed on an RTX
5060 Ti. See [architecture](architecture.md) and the
[reduction contract](2opt-gpu-reduction.md) before changing the kernel.
