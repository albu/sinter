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
/// Check if an op is a pointwise photometric transform that commutes with Crop.
///
/// Algebraic law:
/// For any pointwise photometric transform P (LUT, Matrix, pointwise color):
///     Crop(P(image)) == P(Crop(image))
pub fn is_pointwise_photometric(op: &SampledImageOp) -> bool {
    matches!(
        op,
        SampledImageOp::Brightness { .. }
            | SampledImageOp::Contrast { .. }
            | SampledImageOp::Gamma { .. }
            | SampledImageOp::Invert
            | SampledImageOp::Posterize { .. }
            | SampledImageOp::Solarize { .. }
            | SampledImageOp::RGBShift { .. }
            | SampledImageOp::ToSepia
            | SampledImageOp::ColorTemperature { .. }
            | SampledImageOp::ChannelMix { .. }
            | SampledImageOp::ColorBalance { .. }
            | SampledImageOp::HueSaturationValue { .. }
            | SampledImageOp::ToGray
            | SampledImageOp::Normalize { .. }
    )
}

/// Hoist Crop and RandomCrop before preceding pointwise photometric transforms.
///
/// When a Crop appears after pointwise photometric transforms (e.g. Brightness -> Crop),
/// computing the photometric transform on the full image before discarding 90-99% of the
/// pixels is pure wasted work.
///
/// Because pointwise transforms commute with Crop:
///     Crop(P(image)) == P(Crop(image))
///
/// This pass bubbles Crop leftwards across any preceding pointwise photometric transforms,
/// ensuring that photometric processing is only evaluated on the surviving cropped pixels.
pub fn hoist_crops(ops: &mut Vec<SampledImageOp>) {
    let mut i = 0;
    while i < ops.len() {
        if matches!(ops[i], SampledImageOp::Crop { .. } | SampledImageOp::RandomCrop { .. }) {
            let mut j = i;
            while j > 0 && is_pointwise_photometric(&ops[j - 1]) {
                ops.swap(j, j - 1);
                j -= 1;
            }
        }
        i += 1;
    }
}

pub fn canonicalize(fused: &mut Vec<SampledImageOp>) {
    if fused.len() <= 1 {
        return; // Nothing to reorder
    }

    // Hoist geometric transforms to the left across operations that commute with geometry.
    // If an operation is a barrier (like GaussianBlur or Equalize), geometry stops and
    // will NOT be hoisted across it.
    let mut i = 0;
    while i < fused.len() {
        if matches!(fused[i].reorder_rule(), ReorderRule::Geometry) {
            let mut j = i;
            while j > 0 && matches!(fused[j - 1].reorder_rule(), ReorderRule::CommutesWithGeometry) {
                fused.swap(j, j - 1);
                j -= 1;
            }
        }
        i += 1;
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
