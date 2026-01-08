// Runtime: Fusion infrastructure for transform execution
//
// This module contains:
// - LUT (Look-Up Table) fusion infrastructure
// - Matrix fusion infrastructure
// - Pixel operation fusion
// - Shared utilities (clamp, range tracking)

pub mod lut;
pub mod matrix;
pub mod pixel_op;
pub mod utils;

// Re-export LUT types
pub use lut::{FusedLut, FusedLutExecutor, LutExecutor, LutOp};

// Re-export matrix types
pub use matrix::fused::FusedMatrix;
pub use matrix::{apply_matrix, compose_matrices, MatrixExecutor, MatrixOp};

// Re-export pixel op types
pub use pixel_op::{execute_fused, FusedExecutor, PixelOp};

// Re-export utils
pub use utils::clamp;
pub use utils::range::{RangedOp, ValueRange};
