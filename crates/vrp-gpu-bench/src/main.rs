//! Reproducible correctness and latency comparison of CPU/GPU 2-opt selection.

#[cfg(feature = "gpu")]
mod runner;
#[cfg(any(feature = "gpu", test))]
mod validation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "gpu")]
    return runner::run();
    #[cfg(not(feature = "gpu"))]
    Err(
        "enable the gpu feature: cargo run --release -p vrp-gpu-bench --features gpu -- --help"
            .into(),
    )
}
