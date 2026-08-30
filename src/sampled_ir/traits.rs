// Transform trait implementation for SampledImageOp
//
// This bridges the deterministic IR to the existing optimizer infrastructure.

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, ReorderRule, ShapeEffect, Transform,
};
use crate::sampled_ir::{ops::Interpolation, SampledImageOp};
use std::any::Any;

impl Transform for SampledImageOp {
    fn access(&self) -> AccessPattern {
        match self {
            // Photometric: InPlace + Preserve
            SampledImageOp::Brightness { .. } => AccessPattern::InPlace,
            SampledImageOp::Contrast { .. } => AccessPattern::InPlace,
            SampledImageOp::Gamma { .. } => AccessPattern::InPlace,
            SampledImageOp::Invert => AccessPattern::InPlace,
            SampledImageOp::Normalize { .. } => AccessPattern::InPlace,
            SampledImageOp::ToRGB => AccessPattern::OutOfPlace, // Changes channel count 1->3
            SampledImageOp::ToGray => AccessPattern::InPlace,
            SampledImageOp::ToSepia => AccessPattern::InPlace,
            SampledImageOp::ChannelMix { .. } => AccessPattern::InPlace,
            SampledImageOp::ColorBalance { .. } => AccessPattern::InPlace,
            SampledImageOp::ChannelShuffle { .. } => AccessPattern::InPlace,
            SampledImageOp::ColorTint { .. } => AccessPattern::InPlace,
            SampledImageOp::HueSaturationValue { .. } => AccessPattern::InPlace,
            SampledImageOp::RGBShift { .. } => AccessPattern::InPlace,
            SampledImageOp::ColorTemperature { .. } => AccessPattern::InPlace,
            SampledImageOp::Posterize { .. } => AccessPattern::InPlace,
            SampledImageOp::Solarize { .. } => AccessPattern::InPlace,
            SampledImageOp::AutoContrast { .. } => AccessPattern::InPlace,
            SampledImageOp::Equalize => AccessPattern::InPlace,

            // Noise: InPlace but non-deterministic
            SampledImageOp::GaussNoise { .. } => AccessPattern::InPlace,
            SampledImageOp::MultiplicativeNoise { .. } => AccessPattern::InPlace,
            SampledImageOp::SaltAndPepper { .. } => AccessPattern::InPlace,
            SampledImageOp::NoiseGranularity { .. } => AccessPattern::InPlace,

            // Dropout: InPlace + Preserve
            SampledImageOp::CoarseDropout { .. } => AccessPattern::InPlace,
            SampledImageOp::GridDropout { .. } => AccessPattern::InPlace,

            // Sharpen/Emboss: OutOfPlace (neighborhood)
            SampledImageOp::Sharpen { .. } => AccessPattern::OutOfPlace,
            SampledImageOp::Emboss { .. } => AccessPattern::OutOfPlace,

            // Geometric: Shape-preserving are InPlace
            SampledImageOp::HorizontalFlip => AccessPattern::InPlace,
            SampledImageOp::VerticalFlip => AccessPattern::InPlace,

            // Geometric: Shape-changing are OutOfPlace
            SampledImageOp::Transpose => AccessPattern::OutOfPlace, // Transpose swaps dimensions
            SampledImageOp::Rotate { .. } => AccessPattern::OutOfPlace, // Rotate swaps dimensions for 90°/270°
            SampledImageOp::Affine { .. } => AccessPattern::OutOfPlace,
            SampledImageOp::Resize { .. } => AccessPattern::OutOfPlace,
            SampledImageOp::Crop { .. } => AccessPattern::OutOfPlace,
            SampledImageOp::Pad { .. } => AccessPattern::OutOfPlace,

            // Kernel: OutOfPlace
            SampledImageOp::GaussianBlur { .. } => AccessPattern::OutOfPlace,
            SampledImageOp::MedianBlur { .. } => AccessPattern::OutOfPlace,
            SampledImageOp::EdgeDetection { .. } => AccessPattern::OutOfPlace,
        }
    }

    fn shape_effect(&self) -> ShapeEffect {
        match self {
            // Photometric: Preserve
            SampledImageOp::Brightness { .. } => ShapeEffect::Preserve,
            SampledImageOp::Contrast { .. } => ShapeEffect::Preserve,
            SampledImageOp::Gamma { .. } => ShapeEffect::Preserve,
            SampledImageOp::Invert => ShapeEffect::Preserve,
            SampledImageOp::Normalize { .. } => ShapeEffect::Preserve,
            SampledImageOp::ToRGB => ShapeEffect::Preserve, // Preserves width/height (only changes channels)
            SampledImageOp::ToGray => ShapeEffect::Preserve,
            SampledImageOp::ToSepia => ShapeEffect::Preserve,
            SampledImageOp::ChannelMix { .. } => ShapeEffect::Preserve,
            SampledImageOp::ColorBalance { .. } => ShapeEffect::Preserve,
            SampledImageOp::ChannelShuffle { .. } => ShapeEffect::Preserve,
            SampledImageOp::ColorTint { .. } => ShapeEffect::Preserve,
            SampledImageOp::HueSaturationValue { .. } => ShapeEffect::Preserve,
            SampledImageOp::RGBShift { .. } => ShapeEffect::Preserve,
            SampledImageOp::ColorTemperature { .. } => ShapeEffect::Preserve,
            SampledImageOp::Posterize { .. } => ShapeEffect::Preserve,
            SampledImageOp::Solarize { .. } => ShapeEffect::Preserve,
            SampledImageOp::AutoContrast { .. } => ShapeEffect::Preserve,
            SampledImageOp::Equalize => ShapeEffect::Preserve,
            SampledImageOp::GaussNoise { .. } => ShapeEffect::Preserve,
            SampledImageOp::MultiplicativeNoise { .. } => ShapeEffect::Preserve,
            SampledImageOp::SaltAndPepper { .. } => ShapeEffect::Preserve,
            SampledImageOp::NoiseGranularity { .. } => ShapeEffect::Preserve,
            SampledImageOp::CoarseDropout { .. } => ShapeEffect::Preserve,
            SampledImageOp::GridDropout { .. } => ShapeEffect::Preserve,
            SampledImageOp::Sharpen { .. } => ShapeEffect::Preserve,
            SampledImageOp::Emboss { .. } => ShapeEffect::Preserve,

            // Geometric: Shape-preserving
            SampledImageOp::HorizontalFlip => ShapeEffect::Preserve,
            SampledImageOp::VerticalFlip => ShapeEffect::Preserve,

            // Geometric: Shape-changing
            SampledImageOp::Transpose => ShapeEffect::Resize, // Transpose swaps dimensions
            SampledImageOp::Rotate { .. } => ShapeEffect::Resize, // Rotate swaps dimensions for 90°/270°
            SampledImageOp::Affine { .. } => ShapeEffect::Resize,
            SampledImageOp::Resize { .. } => ShapeEffect::Resize,
            SampledImageOp::Crop { .. } => ShapeEffect::Crop,
            SampledImageOp::Pad { .. } => ShapeEffect::Pad,

            // Kernel: Preserve
            SampledImageOp::GaussianBlur { .. } => ShapeEffect::Preserve,
            SampledImageOp::MedianBlur { .. } => ShapeEffect::Preserve,
            SampledImageOp::EdgeDetection { .. } => ShapeEffect::Preserve,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn reorder_rule(&self) -> ReorderRule {
        match self {
            // Per-pixel photometric: commute with geometry
            SampledImageOp::Brightness { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::Contrast { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::Gamma { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::Invert => ReorderRule::CommutesWithGeometry,
            SampledImageOp::Normalize { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::ToRGB => ReorderRule::CommutesWithGeometry,
            SampledImageOp::ToGray => ReorderRule::CommutesWithGeometry,
            SampledImageOp::ToSepia => ReorderRule::CommutesWithGeometry,
            SampledImageOp::ChannelMix { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::ColorBalance { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::ChannelShuffle { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::ColorTint { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::RGBShift { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::ColorTemperature { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::Posterize { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::Solarize { .. } => ReorderRule::CommutesWithGeometry,
            SampledImageOp::HueSaturationValue { .. } => ReorderRule::CommutesWithGeometry,

            // Geometric: coordinate transforms
            SampledImageOp::HorizontalFlip => ReorderRule::Geometry,
            SampledImageOp::VerticalFlip => ReorderRule::Geometry,
            SampledImageOp::Transpose => ReorderRule::Geometry,
            SampledImageOp::Rotate { .. } => ReorderRule::Geometry,

            // Everything else: barrier
            _ => ReorderRule::Barrier,
        }
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for SampledImageOp {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Direct match dispatch - NO RTTI!
        // This is the zero-cost abstraction path that avoids trait object conversion.
        use crate::sampled_ir::ops::{
            BorderMode, EdgeMethod as SampledEdgeMethod, EmbossDirection as SampledEmbossDirection,
            Interpolation, PadMode as SampledPadMode, RotateAngle as SampledRotateAngle,
        };
        use crate::transforms::geometric::{
            Affine, Crop, HorizontalFlip, Pad, Resize, Rotate, Transpose, VerticalFlip,
        };
        use crate::transforms::*;

        match self {
            // Photometric transforms
            SampledImageOp::Brightness { delta } => Brightness::new(*delta).execute(image),
            SampledImageOp::Contrast { factor } => Contrast::new(*factor).execute(image),
            SampledImageOp::Gamma { gamma } => Gamma::new(*gamma).execute(image),
            SampledImageOp::Invert => Invert.execute(image),
            SampledImageOp::Normalize { mean, std } => {
                Normalize::new(mean[0], std[0]).execute(image)
            }
            SampledImageOp::ToRGB => ToRGB::new().execute(image),
            SampledImageOp::ToGray => ToGray.execute(image),
            SampledImageOp::ToSepia => ToSepia.execute(image),
            SampledImageOp::Posterize { bits } => Posterize::new(*bits).execute(image),
            SampledImageOp::Solarize { threshold } => Solarize::new(*threshold).execute(image),
            SampledImageOp::Equalize => Equalize.execute(image),
            SampledImageOp::AutoContrast {
                cutoff_low,
                cutoff_high,
            } => {
                let cutoff = (*cutoff_low + *cutoff_high) / 2.0;
                AutoContrast::new(cutoff).execute(image)
            }

            // Color transforms
            SampledImageOp::ChannelMix {
                r_from,
                g_from,
                b_from,
            } => {
                let matrix = [*r_from, *g_from, *b_from];
                ChannelMix::new(matrix).execute(image)
            }
            SampledImageOp::ColorBalance {
                shadows: _,
                midtones: _,
                highlights,
            } => ColorBalance::new(highlights[0], highlights[1], highlights[2]).execute(image),
            SampledImageOp::ChannelShuffle { order } => {
                use crate::transforms::photometric::channel_shuffle::ChannelOrder;
                let channel_order = match *order {
                    [0, 1, 2] => ChannelOrder::RGB,
                    [0, 2, 1] => ChannelOrder::RBG,
                    [1, 0, 2] => ChannelOrder::GRB,
                    [1, 2, 0] => ChannelOrder::GBR,
                    [2, 0, 1] => ChannelOrder::BRG,
                    [2, 1, 0] => ChannelOrder::BGR,
                    _ => ChannelOrder::RGB,
                };
                ChannelShuffle::new(channel_order).execute(image)
            }
            SampledImageOp::ColorTint { tint } => {
                ColorTint::new(tint[0], tint[1], tint[2], tint[3]).execute(image)
            }
            SampledImageOp::HueSaturationValue {
                hue_shift,
                saturation_scale,
                value_scale,
            } => HueSaturationValue::new(*hue_shift as f32, *saturation_scale, *value_scale)
                .execute(image),
            SampledImageOp::RGBShift {
                r_shift,
                g_shift,
                b_shift,
            } => RGBShift::new(*r_shift as f32, *g_shift as f32, *b_shift as f32).execute(image),
            SampledImageOp::ColorTemperature { temperature } => {
                ColorTemperature::new(*temperature).execute(image)
            }

            // Noise transforms
            SampledImageOp::GaussNoise { mean, std, seed } => {
                GaussNoise::with_seed(*mean, *std, *seed).execute(image)
            }
            SampledImageOp::MultiplicativeNoise { multiplier, seed } => {
                MultiplicativeNoise::with_seed(*multiplier, 0.1, *seed).execute(image)
            }
            SampledImageOp::SaltAndPepper {
                amount,
                salt_vs_pepper,
                seed,
            } => SaltAndPepper::with_seed(*amount, *salt_vs_pepper, *seed).execute(image),
            SampledImageOp::NoiseGranularity {
                mean,
                std,
                granularity,
            } => {
                use crate::transforms::NoiseGranularity as TransNoiseGranularity;
                let granularity = match *granularity {
                    1 => TransNoiseGranularity::PerPixel,
                    n @ 4..=16 => TransNoiseGranularity::Block(n as usize),
                    _ => TransNoiseGranularity::PerVector,
                };
                MultiplicativeNoise::with_granularity(*mean, *std, granularity).execute(image)
            }

            // Dropout transforms
            SampledImageOp::CoarseDropout {
                holes,
                hole_size,
                seed,
            } => CoarseDropout::with_seed(
                *holes as u32,
                (hole_size.0 as f32 / 255.0, hole_size.1 as f32 / 255.0),
                0,
                *seed,
            )
            .execute(image),
            SampledImageOp::GridDropout {
                ratio,
                unit_size,
                holes: _,
                seed,
            } => GridDropout::with_seed((*unit_size, *unit_size), *ratio, 0, *seed)
                .execute(image),

            // Geometric transforms
            SampledImageOp::HorizontalFlip => HorizontalFlip.execute(image),
            SampledImageOp::VerticalFlip => VerticalFlip.execute(image),
            SampledImageOp::Transpose => Transpose.execute(image),
            SampledImageOp::Rotate { angle } => {
                let geo_angle = match angle {
                    SampledRotateAngle::Rotate90 => {
                        crate::transforms::geometric::rotate::RotateAngle::Rotate90
                    }
                    SampledRotateAngle::Rotate180 => {
                        crate::transforms::geometric::rotate::RotateAngle::Rotate180
                    }
                    SampledRotateAngle::Rotate270 => {
                        crate::transforms::geometric::rotate::RotateAngle::Rotate270
                    }
                };
                Rotate::new(geo_angle).execute(image)
            }
            SampledImageOp::Resize {
                width,
                height,
                interpolation,
            } => {
                use crate::transforms::geometric::resize::ResizeInterpolation;
                let geo_interp = match interpolation {
                    Interpolation::Nearest => ResizeInterpolation::Nearest,
                    Interpolation::Bilinear => ResizeInterpolation::Bilinear,
                    Interpolation::Bicubic => ResizeInterpolation::Bicubic,
                    Interpolation::Lanczos4 => ResizeInterpolation::Lanczos4,
                };
                Resize::with_interpolation(*width as usize, *height as usize, geo_interp)
                    .execute(image)
            }
            SampledImageOp::Crop {
                x,
                y,
                width,
                height,
            } => Crop::new(*x, *y, *width, *height).execute(image),
            SampledImageOp::Pad {
                top,
                bottom,
                left,
                right,
                mode,
                value,
            } => {
                // If value is explicitly set, it overrides mode to Constant(value)
                // Otherwise use the mode as-is
                let pad_mode = if let Some(v) = value {
                    crate::transforms::geometric::pad::PadMode::Constant(*v)
                } else {
                    match mode {
                        SampledPadMode::Constant { value } => {
                            crate::transforms::geometric::pad::PadMode::Constant(*value)
                        }
                        SampledPadMode::Reflect => {
                            crate::transforms::geometric::pad::PadMode::Reflect
                        }
                        SampledPadMode::Replicate => {
                            crate::transforms::geometric::pad::PadMode::Replicate
                        }
                        SampledPadMode::Wrap => crate::transforms::geometric::pad::PadMode::Wrap,
                    }
                };
                Pad::new(*top, *bottom, *left, *right, pad_mode).execute(image)
            }
            SampledImageOp::Affine {
                scale,
                rotate,
                translate,
                shear,
                interpolation,
                border_mode,
            } => {
                use crate::transforms::geometric::affine::{
                    AffineBorderMode, AffineInterpolation, AffineParams,
                };
                let affine_interp = match interpolation {
                    Interpolation::Nearest => AffineInterpolation::Nearest,
                    // Bicubic/Lanczos4 fall back to Bilinear (not supported by Affine)
                    Interpolation::Bilinear | Interpolation::Bicubic | Interpolation::Lanczos4 => {
                        AffineInterpolation::Bilinear
                    }
                };
                let affine_border = match border_mode {
                    crate::sampled_ir::ops::BorderMode::Constant { value } => {
                        AffineBorderMode::Constant { value: *value }
                    }
                    crate::sampled_ir::ops::BorderMode::Reflect => AffineBorderMode::Reflect,
                    crate::sampled_ir::ops::BorderMode::Replicate => AffineBorderMode::Replicate,
                    crate::sampled_ir::ops::BorderMode::Wrap => AffineBorderMode::Wrap,
                };
                let params = AffineParams {
                    scale: *scale,
                    rotate: *rotate,
                    translate: *translate,
                    shear: *shear,
                };
                Affine::with_all(
                    params,
                    image.width,
                    image.height,
                    affine_interp,
                    affine_border,
                )
                .execute(image)
            }

            // Kernel transforms
            SampledImageOp::GaussianBlur { kernel_size, sigma } => {
                if *sigma > 0.0 {
                    crate::transforms::kernel::gaussian::GaussianBlurSigma::with_quality(
                        *sigma,
                        crate::transforms::kernel::gaussian::BlurQuality::Exact,
                    )
                    .execute(image)
                } else {
                    use crate::transforms::kernel::KernelSize;
                    let ks = match *kernel_size {
                        3 => KernelSize::Size3,
                        5 => KernelSize::Size5,
                        7 => KernelSize::Size7,
                        _ => KernelSize::Size3,
                    };
                    GaussianBlur::with_kernel_size(ks).execute(image)
                }
            }
            SampledImageOp::MedianBlur { kernel_size } => {
                use crate::transforms::kernel::median_blur::MedianKernelSize;
                let ks = match *kernel_size {
                    3 => MedianKernelSize::Kernel3,
                    5 => MedianKernelSize::Kernel5,
                    _ => MedianKernelSize::Kernel3,
                };
                MedianBlur::new(ks).execute(image)
            }
            SampledImageOp::Sharpen { strength } => {
                Sharpen::with_strength(*strength).execute(image)
            }
            SampledImageOp::Emboss {
                direction,
                alpha,
                strength,
            } => {
                use crate::transforms::kernel::EmbossDirection as CoreEmbossDirection;
                let emboss_direction = match direction {
                    SampledEmbossDirection::TopLeft => CoreEmbossDirection::SouthWest,
                    SampledEmbossDirection::Top => CoreEmbossDirection::SouthWest,
                    SampledEmbossDirection::TopRight => CoreEmbossDirection::SouthEast,
                    SampledEmbossDirection::Right => CoreEmbossDirection::SouthEast,
                    SampledEmbossDirection::BottomRight => CoreEmbossDirection::NorthEast,
                    SampledEmbossDirection::Bottom => CoreEmbossDirection::NorthEast,
                    SampledEmbossDirection::BottomLeft => CoreEmbossDirection::NorthWest,
                    SampledEmbossDirection::Left => CoreEmbossDirection::NorthWest,
                };
                Emboss::new()
                    .with_direction(emboss_direction)
                    .with_alpha(*alpha)
                    .with_strength(*strength)
                    .execute(image)
            }
            SampledImageOp::EdgeDetection { method } => {
                use crate::transforms::kernel::EdgeMethod as CoreEdgeMethod;
                let edge_method = match method {
                    SampledEdgeMethod::Sobel => CoreEdgeMethod::Sobel,
                    SampledEdgeMethod::Laplacian => CoreEdgeMethod::Laplacian,
                    SampledEdgeMethod::Prewitt => CoreEdgeMethod::Laplacian, // Fall back to Laplacian
                    SampledEdgeMethod::Canny => CoreEdgeMethod::Sobel,       // Fall back to Sobel
                };
                EdgeDetection::new(edge_method).execute(image)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampled_ir::ops::RotateAngle;
    use crate::core::FusableImage;

    #[test]
    fn test_brightness_access_and_shape() {
        let op = SampledImageOp::Brightness { delta: 10.0 };
        assert_eq!(op.access(), AccessPattern::InPlace);
        assert_eq!(op.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_resize_access_and_shape() {
        let op = SampledImageOp::Resize {
            width: 256,
            height: 256,
            interpolation: Interpolation::Bilinear,
        };
        assert_eq!(op.access(), AccessPattern::OutOfPlace);
        assert_eq!(op.shape_effect(), ShapeEffect::Resize);
    }

    #[test]
    fn test_photometric_commutes_with_geometry() {
        let ops = vec![
            SampledImageOp::Brightness { delta: 10.0 },
            SampledImageOp::Contrast { factor: 1.5 },
            SampledImageOp::Invert,
        ];

        for op in ops {
            assert_eq!(op.reorder_rule(), ReorderRule::CommutesWithGeometry);
        }
    }

    #[test]
    fn test_affine_identity_op_exact_path() {
        // Exact path used by Python Compose.apply: SampledImageOp::Affine -> execute
        use crate::sampled_ir::ops::{BorderMode, Interpolation};
        let w = 8;
        let h = 8;
        let mut data = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                data.push((x as u8).wrapping_mul(16).wrapping_add(y as u8));
            }
        }
        let mut img = FusableImage::new(&mut data, w, h, 1);

        let op = SampledImageOp::Affine {
            scale: (1.0, 1.0),
            rotate: 0.0,
            translate: (0.0, 0.0),
            shear: (0.0, 0.0),
            interpolation: Interpolation::Bilinear,
            border_mode: BorderMode::Constant { value: 0 },
        };
        let barrier = op.execute(&mut img).unwrap();

        let mut mismatches = 0usize;
        let mut max_diff = 0i32;
        for (i, (&got, &expected)) in barrier.data.iter().zip(data.iter()).enumerate() {
            let diff = (got as i32 - expected as i32).abs();
            if diff > 0 {
                mismatches += 1;
                max_diff = max_diff.max(diff);
                if mismatches <= 8 {
                    eprintln!("  idx={} (x={}, y={}): got={} expected={}", i, i % w, i / w, got, expected);
                }
            }
        }
        assert_eq!(mismatches, 0, "SampledImageOp identity: {} mismatches, max_diff={}", mismatches, max_diff);
    }

    #[test]
    fn test_geometric_is_geometry() {
        let ops = vec![
            SampledImageOp::HorizontalFlip,
            SampledImageOp::VerticalFlip,
            SampledImageOp::Transpose,
            SampledImageOp::Rotate {
                angle: RotateAngle::Rotate90,
            },
        ];

        for op in ops {
            assert_eq!(op.reorder_rule(), ReorderRule::Geometry);
        }
    }

    #[test]
    fn test_noise_is_barrier() {
        let ops = vec![
            SampledImageOp::GaussNoise {
                mean: 0.0,
                std: 1.0,
                seed: 0,
            },
            SampledImageOp::MultiplicativeNoise { multiplier: 1.0, seed: 0 },
        ];

        for op in ops {
            assert_eq!(op.reorder_rule(), ReorderRule::Barrier);
        }
    }

    #[test]
    fn test_as_executable_returns_some() {
        let op = SampledImageOp::Brightness { delta: 10.0 };
        assert!(op.as_executable().is_some());
    }
}
