// Sampled IR (Deterministic Intermediate Representation)
//
// This module defines the pure data enums with all randomness resolved.
// These types are serializable via serde and support replay via sample_with_seed().
//
// Also contains Plan - the input to the optimizer.

pub mod ops;
pub mod program;
pub mod batch;
pub mod plan;
pub mod bridge;
pub mod traits;

// Re-export for convenience
pub use ops::SampledImageOp;
pub use program::{SampledImageProgram, SampledImageProgramBuilder, IR_VERSION};
pub use batch::{MosaicLayout, Rect, SampledBatchOp, SampledBatchProgram};
pub use plan::Plan;
