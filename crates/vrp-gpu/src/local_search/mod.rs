//! Local search operators (CPU reference and GPU orchestration).

pub mod cpu;

/// An improving reversal of the inclusive segment `i..=j`.
///
/// Selection uses the smallest finite negative delta, then the lexicographically
/// smallest `(i, j)` on exact ties. Zero, infinities, and NaNs are not candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoOptMove {
    /// Start of the reversed route segment, inclusive.
    pub i: usize,
    /// End of the reversed route segment, inclusive.
    pub j: usize,
    /// Change in route distance; improving moves have a negative value.
    pub delta: f32,
}

#[cfg(feature = "gpu")]
pub mod gpu;
