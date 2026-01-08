// Fusion Strategy Module
//
// This module implements the extractive fusion algorithm for the optimizer.
//
// # Architecture (4-Phase Pipeline)
//
// After canonicalization (Phase 2), we use extractive fusion:
//
// 1. **Geometric groups**: Extract contiguous geometric transforms → compose via D4 group
// 2. **LUT groups**: Extract contiguous LUT transforms → compose into FusedLut
// 3. **Matrix groups**: Extract contiguous Matrix transforms → compose into FusedMatrix
// 4. **Individual**: Single transforms become individual nodes
//
// This replaces the old all-or-nothing strategy decision tree.

use crate::exec_ir::nodes::ExecNode;
use crate::sampled_ir::SampledImageOp;

mod data_dependent_lut;
mod extractive;
mod geometric;

/// Result of a fusion strategy attempt
#[derive(Debug)]
pub enum FusionResult {
    /// Successfully fused into execution nodes (may be empty for identity/no-op)
    Success(Vec<ExecNode>),
    /// This strategy cannot be applied to these transforms
    NotApplicable,
}

/// Try to fuse a block of transforms using extractive fusion
///
/// # Extractive Fusion Algorithm
///
/// After canonicalization (Phase 2) separates geometric from photometric transforms,
/// extractive fusion extracts contiguous homogeneous groups:
///
/// 1. **Geometric groups** (2+): Compose via D4 group operations
/// 2. **LUT groups** (2+): Compose into single FusedLut
/// 3. **Matrix groups** (2+): Compose into single FusedMatrix
/// 4. **Individual**: Single transforms become individual nodes
///
/// # Example
/// Input: [Solarize, Contrast, Brightness, Gamma, Posterize, Saturation]
///
/// After canonicalization: [Solarize, Contrast, Brightness, Posterize, Gamma, Saturation]
/// (assuming geometric hoisting already happened)
///
/// Extractive fusion:
///   - Extract LUT group [Solarize, Contrast, Brightness, Posterize] → FusedLut
///   - Gamma is neither LUT nor Matrix → individual
///   - Saturation is single Matrix → individual
///
/// Output: [FusedLut(...), Gamma, Saturation]
///
/// # Arguments
/// * `fused` - Mutable reference to vector of fuseable transforms to optimize
///
/// # Returns
/// * `FusionResult::Success(nodes)` - Fused nodes (empty if transforms cancel to identity)
/// * `FusionResult::NotApplicable` - Strategy cannot be applied
pub fn fuse_transform_block(fused: &mut Vec<SampledImageOp>) -> FusionResult {
    if fused.is_empty() {
        return FusionResult::NotApplicable;
    }

    // Extractive Fusion - fuse contiguous homogeneous groups
    extractive::try_extractive_fusion(fused)
}
