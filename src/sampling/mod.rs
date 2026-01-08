// Sampling Phase: Probabilistic Transform Support
//
// This module implements the two-phase execution model:
// 1. Sampling Phase: Resolve randomness, produce deterministic plan
// 2. Execution Phase: Run deterministic plan with no RNG calls
//
// This is the key differentiator vs Albumentations - randomness is resolved
// at planning time, not per-pixel.

// Sub-modules
pub mod traits;
pub mod distributions;
pub mod sampling_nodes;
pub mod macros;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use traits::Rng;
#[cfg(feature = "python")]
pub use traits::ThreadRng;
pub use distributions::{Bernoulli, Uniform, UniformInt, Dist};

// New API: Pure enum, no trait objects, zero RTTI
pub use sampling_nodes::{RandomImageNode, SamplingContext, RandomImageProgram};
