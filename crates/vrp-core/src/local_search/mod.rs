//! Local search operators (CPU reference and GPU orchestration).

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;
