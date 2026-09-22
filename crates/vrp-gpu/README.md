# vrp-gpu

`vrp-gpu` provides vehicle-routing data structures, deterministic CPU reference
heuristics and opt-in CUDA acceleration. The current API parses Solomon-style
CVRP instances, constructs feasible routes and evaluates or selects 2-opt moves.

The crate is experimental and has not been published to crates.io yet.

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `gpu` | No | Enables CUDA driver orchestration through `cudarc` and the embedded `sm_120` PTX artifact |

Without `gpu`, the library has no third-party runtime dependencies.

## Installation

During development, use the Git repository:

```toml
[dependencies]
vrp-gpu = { git = "https://github.com/pedrozaz/vrp-gpu" }
```

After the first crates.io release, the dependency will use the normal registry
form:

```toml
[dependencies]
vrp-gpu = "0.1"
```

## Example

```rust
use std::error::Error;

use vrp_gpu::{construct::nearest_neighbor, instance::SolomonInstance};

fn main() -> Result<(), Box<dyn Error>> {
    let input = "\
DEMO
VEHICLE
2 10
CUSTOMER
0 0 0 0 0 100 0
1 1 0 5 0 100 0
2 0 1 5 0 100 0
";
    let instance: SolomonInstance = input.parse()?;
    let solution = nearest_neighbor(&instance);

    assert!(solution.is_feasible(&instance));
    Ok(())
}
```

Enable `gpu` to access `local_search::gpu`. GPU entry points validate their
inputs before using CUDA, while empty and singleton routes use CPU-only fast
paths. The current PTX artifact is hardware-validated on an NVIDIA GeForce RTX
5060 Ti (`sm_120`); compatibility with other architectures is not yet claimed.

## Minimum supported Rust version

The supported feature set requires Rust 1.88 or newer. Raising the MSRV is a
release-level change and must be recorded in the changelog.

See the [repository documentation](https://github.com/pedrozaz/vrp-gpu/tree/develop/docs)
for architecture, kernel contracts and release procedures.

Licensed under Apache-2.0.
