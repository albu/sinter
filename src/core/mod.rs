// Core execution model for Albumentations2
//
// This module defines the foundational types and traits that enable
// the compiler to reason about transform fusion and safe in-place execution.

pub mod image;
pub mod traits;

// Re-export for convenience
pub use image::{FusableImage, BarrierImage};
pub use traits::{AccessPattern, ShapeEffect, ReorderRule, Transform, Executable, LabelTransform, is_fuseable};
