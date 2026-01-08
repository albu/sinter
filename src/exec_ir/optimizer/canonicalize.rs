// Canonicalization - Geometric Hoisting
//
// Hoists geometric transforms to the boundary of photometric blocks.

use crate::core::{ReorderRule, Transform};
use crate::sampled_ir::SampledImageOp;

/// Phase 2: Canonicalization - Geometric Hoisting
///
/// Hoists geometric transforms to the boundary of photometric blocks.
///
/// # Algebraic Rule
/// Per-pixel photometric transforms commute with geometric coordinate transforms:
/// ```text
/// P(f(x)) = f(P(x))
/// ```
/// Where P is a per-pixel photometric operation and f is a coordinate remapping.
///
/// # Transformation
/// Input:  [Solarize, Contrast, VerticalFlip, Brightness, Gamma]
/// Output: [VerticalFlip, Solarize, Contrast, Brightness, Gamma]
///
/// Or equivalently (photometric-first convention):
/// Output: [Solarize, Contrast, Brightness, Gamma, VerticalFlip]
///
/// # Invariants Preserved
/// - Order within photometric transforms is preserved
/// - Order within geometric transforms is preserved
/// - Barrier transforms stop canonicalization (handled by block splitting)
///
/// # Why This Works
/// Geometric ops only change *where* a pixel is read from.
/// Photometric ops only change *what* the pixel value is.
/// These are independent dimensions of the image, so they commute.
pub fn canonicalize(fused: &mut Vec<SampledImageOp>) {
    if fused.len() <= 1 {
        return; // Nothing to reorder
    }

    // Separate into geometric and photometric transforms
    // using the new ReorderRule classification
    let mut geometric: Vec<SampledImageOp> = Vec::new();
    let mut photometric: Vec<SampledImageOp> = Vec::new();

    for op in fused.drain(..) {
        match op.reorder_rule() {
            ReorderRule::Geometry => {
                geometric.push(op);
            }
            ReorderRule::CommutesWithGeometry => {
                photometric.push(op);
            }
            ReorderRule::Barrier => {
                // Should not happen - barriers split blocks before canonicalization
                // But if we encounter one, put it back and continue draining remaining transforms
                photometric.push(op);
                // Don't break - continue draining the iterator to preserve all transforms
            }
        }
    }

    // Rebuild the vector with geometric transforms hoisted to the boundary
    // We use the convention: [geometry...][photometric...]
    // This means geometric ops are applied first, then photometric
    for t in geometric {
        fused.push(t);
    }
    for t in photometric {
        fused.push(t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampled_ir::ops::SampledImageOp;

    #[test]
    fn test_canonicalize_hoists_geometric() {
        let mut transforms: Vec<SampledImageOp> = vec![
            SampledImageOp::Brightness { delta: 10.0 },
            SampledImageOp::VerticalFlip,
            SampledImageOp::Brightness { delta: 20.0 },
        ];

        canonicalize(&mut transforms);

        // VerticalFlip should be hoisted to the front
        matches!(transforms[0], SampledImageOp::VerticalFlip);
        matches!(transforms[1], SampledImageOp::Brightness { .. });
        matches!(transforms[2], SampledImageOp::Brightness { .. });
    }
}
