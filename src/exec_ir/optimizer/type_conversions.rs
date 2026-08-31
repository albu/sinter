// Helper functions for the optimizer
//
// Downcast helpers and ColorJitter special-case handling

use crate::core::Transform;
use crate::sampled_ir::SampledImageOp;
use crate::transforms::{
    Brightness, ColorJitter, Contrast, FusedLut, Gamma, Invert, LutOp, Normalize, Posterize,
    Solarize,
};
use crate::transforms::{
    ChannelMix, ChannelOrder, ChannelShuffle, ColorBalance, ColorTemperature, ColorTint,
    FusedMatrix, ToSepia,
};

/// Try to convert a Transform to a LutOp
///
/// This is used by the optimizer to detect transforms that support LUT optimization.
pub fn try_as_lut_op(transform: &dyn Transform) -> Option<Box<dyn LutOp>> {
    // First, try to downcast SampledImageOp to concrete types
    if let Some(sampled) = transform.as_any().downcast_ref::<SampledImageOp>() {
        return try_sampled_as_lut_op(sampled);
    }

    // Try Brightness
    if let Some(b) = transform.as_any().downcast_ref::<Brightness>() {
        return Some(Box::new(b.clone()));
    }

    // Try Contrast
    if let Some(c) = transform.as_any().downcast_ref::<Contrast>() {
        return Some(Box::new(c.clone()));
    }

    // Normalize is NOT a LUT op: it produces float32 output (terminal barrier)
    // and must never fuse into a u8 LUT chain.

    // Try Invert
    if let Some(i) = transform.as_any().downcast_ref::<Invert>() {
        return Some(Box::new(*i));
    }

    // Try Posterize
    if let Some(p) = transform.as_any().downcast_ref::<Posterize>() {
        return Some(Box::new(p.clone()));
    }

    // Try Solarize
    if let Some(s) = transform.as_any().downcast_ref::<Solarize>() {
        return Some(Box::new(s.clone()));
    }

    // Try Gamma
    if let Some(g) = transform.as_any().downcast_ref::<Gamma>() {
        return Some(Box::new(g.clone()));
    }

    // Try FusedLut (already fused)
    if let Some(f) = transform.as_any().downcast_ref::<FusedLut>() {
        return Some(Box::new(f.clone()));
    }

    None
}

/// Convenience wrapper that takes SampledImageOp directly (NO-RTTI version)
pub fn try_as_lut_op_sampled(sampled: &SampledImageOp) -> Option<Box<dyn LutOp>> {
    try_sampled_as_lut_op(sampled)
}

/// Convert a SampledImageOp to a concrete LutOp
///
/// This bridges the sampled IR to the optimizer infrastructure.
fn try_sampled_as_lut_op(sampled: &SampledImageOp) -> Option<Box<dyn LutOp>> {
    match sampled {
        SampledImageOp::Brightness { delta } => Some(Box::new(Brightness::new(*delta))),
        SampledImageOp::Contrast { factor } => Some(Box::new(Contrast::new(*factor))),
        SampledImageOp::Gamma { gamma } => Some(Box::new(Gamma::new(*gamma))),
        SampledImageOp::Invert => Some(Box::new(Invert)),
        // Normalize: NOT LUT-fusable (float32 terminal barrier)
        SampledImageOp::Posterize { bits } => Some(Box::new(Posterize::new(*bits))),
        SampledImageOp::Solarize { threshold } => Some(Box::new(Solarize::new(*threshold))),
        _ => None,
    }
}

/// Internal helper to check if a transform is a MatrixOp
pub fn try_as_matrix_op_internal(
    transform: &dyn Transform,
) -> Option<Box<dyn crate::transforms::runtime::matrix::MatrixOp>> {
    // First, try to downcast SampledImageOp to concrete types
    if let Some(sampled) = transform.as_any().downcast_ref::<SampledImageOp>() {
        return try_sampled_as_matrix_op(sampled);
    }

    // Try ToSepia
    if let Some(t) = transform.as_any().downcast_ref::<ToSepia>() {
        return Some(Box::new(*t));
    }

    // Try ColorTemperature
    if let Some(c) = transform.as_any().downcast_ref::<ColorTemperature>() {
        return Some(Box::new(*c));
    }

    // Try ChannelMix
    if let Some(c) = transform.as_any().downcast_ref::<ChannelMix>() {
        return Some(Box::new(*c));
    }

    // Try ColorBalance
    if let Some(c) = transform.as_any().downcast_ref::<ColorBalance>() {
        return Some(Box::new(*c));
    }

    // Try ChannelShuffle
    if let Some(c) = transform.as_any().downcast_ref::<ChannelShuffle>() {
        return Some(Box::new(*c));
    }

    // Try ColorTint
    if let Some(t) = transform.as_any().downcast_ref::<ColorTint>() {
        return Some(Box::new(*t));
    }

    // Try FusedMatrix (already fused)
    if let Some(f) = transform.as_any().downcast_ref::<FusedMatrix>() {
        return Some(Box::new(*f));
    }

    None
}

/// Convenience wrapper that takes SampledImageOp directly (NO-RTTI version)
pub fn try_as_matrix_op_sampled(
    sampled: &SampledImageOp,
) -> Option<Box<dyn crate::transforms::runtime::matrix::MatrixOp>> {
    try_sampled_as_matrix_op(sampled)
}

/// Convert a SampledImageOp to a concrete MatrixOp
///
/// This bridges the sampled IR to the optimizer infrastructure.
fn try_sampled_as_matrix_op(
    sampled: &SampledImageOp,
) -> Option<Box<dyn crate::transforms::runtime::matrix::MatrixOp>> {
    match sampled {
        SampledImageOp::ToSepia => Some(Box::new(ToSepia)),
        SampledImageOp::ColorTemperature { temperature } => {
            Some(Box::new(ColorTemperature::new(*temperature)))
        }
        SampledImageOp::ChannelMix {
            r_from,
            g_from,
            b_from,
        } => {
            // Convert from separate arrays to 3x3 matrix
            let matrix = [*r_from, *g_from, *b_from];
            Some(Box::new(ChannelMix::new(matrix)))
        }
        SampledImageOp::ColorBalance {
            shadows,
            midtones,
            highlights,
        } => {
            // SampledImageOp uses shadows/midtones/highlights, but ColorBalance::new uses r_scale/g_scale/b_scale
            // The execution code uses highlights[0], highlights[1], highlights[2]
            Some(Box::new(ColorBalance::new(
                highlights[0],
                highlights[1],
                highlights[2],
            )))
        }
        SampledImageOp::ChannelShuffle { order } => {
            // Convert [usize; 3] to ChannelOrder enum
            let channel_order = match *order {
                [0, 1, 2] => ChannelOrder::RGB,
                [0, 2, 1] => ChannelOrder::RBG,
                [1, 0, 2] => ChannelOrder::GRB,
                [1, 2, 0] => ChannelOrder::GBR,
                [2, 0, 1] => ChannelOrder::BRG,
                [2, 1, 0] => ChannelOrder::BGR,
                _ => return None, // Invalid order
            };
            Some(Box::new(ChannelShuffle::new(channel_order)))
        }
        SampledImageOp::ColorTint { tint } => {
            // tint is [f32; 4] = [target_r, target_g, target_b, intensity]
            Some(Box::new(ColorTint::new(tint[0], tint[1], tint[2], tint[3])))
        }
        // HueSaturationValue with hue_shift == 0 is intentionally NOT converted
        // to a matrix here: the exact sat/val transform is
        //   RGB' = vs*ss*RGB + vs*(1-ss)*V*[1,1,1]  (V = max)
        // which has a per-pixel max term (plus an S/V clip for ss>1 or vs>1),
        // so it is not a linear map and cannot be matrix-fused without
        // changing results. The old luma-weighted approximation (0.299/0.587/
        // 0.114) disagreed with the hue-shift path by up to ~50 and was removed.
        _ => None,
    }
}

/// Check if a transform is a geometric transform (Rotate, FlipH, FlipV)
///
/// Geometric transforms can be fused even if they're OutOfPlace because
/// they can be composed via D4 group operations.
pub fn is_geometric_transform(transform: &dyn Transform) -> bool {
    use crate::transforms::geometric::{HorizontalFlip, Rotate, VerticalFlip};

    // Check SampledImageOp variants
    if let Some(sampled) = transform.as_any().downcast_ref::<SampledImageOp>() {
        return is_geometric_transform_sampled(sampled);
    }

    transform
        .as_any()
        .downcast_ref::<HorizontalFlip>()
        .is_some()
        || transform.as_any().downcast_ref::<VerticalFlip>().is_some()
        || transform.as_any().downcast_ref::<Rotate>().is_some()
}

/// Check if a SampledImageOp is a geometric transform
///
/// This is the NO-RTTI version that uses match dispatch.
pub fn is_geometric_transform_sampled(sampled: &SampledImageOp) -> bool {
    matches!(
        sampled,
        SampledImageOp::HorizontalFlip
            | SampledImageOp::VerticalFlip
            | SampledImageOp::Transpose
            | SampledImageOp::Rotate { .. }
    )
}
