// SampledImageOp enum definition
//
// Deterministic image transform with all parameters already sampled.
// This is the core type of the sampled IR.

use crate::core::{AccessPattern, ShapeEffect};
use serde::{Deserialize, Serialize};

/// Reconstruct core `AffineParams` from the sampled inverse-mapping matrix.
///
/// The sampled matrix is `[a, b, c, d, e, f]` describing the INVERSE mapping
/// used for backward resampling:
///
/// ```text
/// x_in = a*x_out + b*y_out + c
/// y_in = d*x_out + e*y_out + f
/// ```
///
/// The core `Affine` stores FORWARD parameters (scale, rotate, translate) and
/// rebuilds the inverse matrix itself. To round-trip exactly we invert the
/// linear part `A = [[a, b], [d, e]]` and extract scale / rotation / translation
/// using the same conventions as `build_inverse_matrix` (forward
/// `F = [[sx*cos, -sy*sin], [sx*sin, sy*cos]]`, translation applied as
/// `c = -(a*tx + d*ty)`, `f = -(b*tx + e*ty)`).
pub(crate) fn affine_params_from_matrix(
    matrix: [f32; 6],
) -> crate::transforms::geometric::affine::AffineParams {
    let a = matrix[0];
    let b = matrix[1];
    let c = matrix[2];
    let d = matrix[3];
    let e = matrix[4];
    let f = matrix[5];

    let det = a * e - b * d;
    if det.abs() < 1e-9 {
        // Degenerate (singular) matrix: fall back to identity-like parameters.
        return crate::transforms::geometric::affine::AffineParams::default();
    }

    // Forward linear map F = A^-1.
    let fwd_a = e / det;
    let fwd_b = -b / det;
    let fwd_d = -d / det;
    let fwd_e = a / det;

    // Core forward convention: [[sx*cos, -sy*sin], [sx*sin, sy*cos]].
    let sx = (fwd_a * fwd_a + fwd_d * fwd_d).sqrt();
    let sy = (fwd_b * fwd_b + fwd_e * fwd_e).sqrt();
    let rotate = fwd_d.atan2(fwd_a).to_degrees();

    // Recover forward translation by inverting c = -(a*tx + d*ty), f = -(b*tx + e*ty).
    let tx = (-c * e + d * f) / det;
    let ty = (-a * f + c * b) / det;

    crate::transforms::geometric::affine::AffineParams {
        scale: (sx, sy),
        rotate,
        translate: (tx, ty),
        shear: (0.0, 0.0),
    }
}

/// Deterministic image transform (sampled, no randomness)
///
/// This enum represents a SINGLE transform with ALL parameters
/// already sampled. It is:
/// - Serializable (via serde)
/// - Replayable (save to disk, load later)
/// - Inspectable (print, debug, analyze)
/// - Zero-cost (no dynamic dispatch)
///
/// # Organization
///
/// Variants are grouped by semantic category:
/// - Photometric: Per-pixel color operations (InPlace + Preserve)
/// - Geometric: Spatial transformations (various access/shape)
/// - Kernel: Convolution operations (OutOfPlace or Barrier)
///
/// # IR Invariant
///
/// `SampledImageProgram.ops` is a flat `Vec<SampledImageOp>`.
/// No structural nesting (OneOf/SomeOf are flattened during sampling).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampledImageOp {
    // ========================================================================
    // Photometric Transforms (InPlace + Preserve = Fusable)
    // ========================================================================
    /// Add a constant to all pixels: `pixel = pixel + delta`
    Brightness { delta: f32 },

    /// Scale pixels: `pixel = pixel * factor`
    Contrast { factor: f32 },

    /// Apply gamma correction: `pixel = 255 * (pixel/255)^gamma`
    Gamma { gamma: f32 },

    /// Hue, Saturation, Value adjustment
    HueSaturationValue {
        hue_shift: i32,        // -180 to 180
        saturation_scale: f32, // 0.0 to inf
        value_scale: f32,      // 0.0 to inf
    },

    /// Shift RGB channels independently
    RGBShift {
        r_shift: i32, // -255 to 255
        g_shift: i32,
        b_shift: i32,
    },

    /// Convert RGB to grayscale
    ToGray,

    /// Convert RGB to sepia tone
    ToSepia,

    /// Convert grayscale to RGB (replicate channel)
    ToRGB,

    /// Invert pixel values: `pixel = 255 - pixel`
    Invert,

    /// Normalize: `pixel = (pixel - mean) / std`
    Normalize {
        mean: [f32; 3], // R, G, B
        std: [f32; 3],  // R, G, B
    },

    /// Adjust color temperature
    ColorTemperature { temperature: f32 },

    /// Mix color channels
    ChannelMix {
        // Mix matrix (3x3)
        r_from: [f32; 3], // [r_src, g_src, b_src]
        g_from: [f32; 3],
        b_from: [f32; 3],
    },

    /// Adjust color balance (shadows/midtones/highlights)
    ColorBalance {
        shadows: [f32; 3],    // R, G, B shifts for shadows
        midtones: [f32; 3],   // R, G, B shifts for midtones
        highlights: [f32; 3], // R, G, B shifts for highlights
    },

    /// Randomly shuffle channels
    ChannelShuffle {
        order: [usize; 3], // Permutation of [0, 1, 2]
    },

    /// Apply color tint
    ColorTint {
        tint: [f32; 4], // [r, g, b, intensity]
    },

    /// Posterize: reduce number of color levels
    Posterize {
        bits: u8, // Number of bits to keep (1-8)
    },

    /// Solarize: invert pixels above threshold
    Solarize {
        threshold: u8, // 0-255
    },

    /// Histogram equalization
    Equalize,

    /// Auto contrast enhancement
    AutoContrast {
        cutoff_low: f32,  // 0.0-1.0
        cutoff_high: f32, // 0.0-1.0
    },

    /// Add Gaussian noise to pixels
    GaussNoise { mean: f32, std: f32, seed: u64 },

    /// Add multiplicative noise: `pixel = pixel * (1 + noise)`
    MultiplicativeNoise { multiplier: f32, seed: u64 },

    /// Salt-and-pepper noise
    SaltAndPepper {
        amount: f32,         // Fraction of pixels to affect
        salt_vs_pepper: f32, // 0.0 = all pepper, 1.0 = all salt
        seed: u64,
    },

    /// Noise with spatial granularity (per-region noise)
    NoiseGranularity {
        mean: f32,
        std: f32,
        granularity: u32, // Size of noise regions
    },

    /// Dropout rectangular regions
    CoarseDropout {
        holes: usize,          // Number of holes
        hole_size: (u32, u32), // (height, width) range
        seed: u64,
    },

    /// Grid-based dropout
    GridDropout {
        ratio: f32,     // Fraction of grid to drop
        unit_size: u32, // Size of grid cells
        holes: usize,   // Number of holes
        seed: u64,
    },

    // ========================================================================
    // Geometric Transforms (Shape-changing or coordinate remapping)
    // ========================================================================
    /// Horizontal flip
    HorizontalFlip,

    /// Vertical flip
    VerticalFlip,

    /// Rotate by 90, 180, or 270 degrees
    Rotate { angle: RotateAngle },

    /// Transpose (swap x and y axes)
    Transpose,

    /// Affine transformation
    Affine {
        scale: (f32, f32),
        rotate: f32,
        translate: (f32, f32),
        shear: (f32, f32),
        interpolation: Interpolation,
        border_mode: BorderMode,
    },

    /// Resize to new dimensions
    Resize {
        width: u32,
        height: u32,
        interpolation: Interpolation,
    },

    /// Crop to rectangular region
    Crop {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },

    /// Crop a fixed-size window at a sampled fractional position.
    /// Anchors (fx, fy in [0, 1)) resolve against the actual image size at
    /// execution time, so one sampled program fits any image size while
    /// image and label transforms agree exactly.
    RandomCrop {
        width: u32,
        height: u32,
        fx: f32,
        fy: f32,
    },

    /// Pad image
    Pad {
        top: u32,
        bottom: u32,
        left: u32,
        right: u32,
        mode: PadMode,
        value: Option<u8>,
    },

    // ========================================================================
    // Kernel Transforms (Convolution - typically barriers)
    // ========================================================================
    /// Gaussian blur
    GaussianBlur {
        kernel_size: u32, // Odd number
        sigma: f32,
    },

    /// Median blur
    MedianBlur {
        kernel_size: u32, // Odd number
    },

    /// Sharpen
    Sharpen {
        strength: f32, // 0.0 to 1.0
    },

    /// Emboss
    Emboss {
        direction: EmbossDirection,
        alpha: f32,
        strength: f32,
    },

    /// Edge detection
    EdgeDetection { method: EdgeMethod },
}

// =============================================================================
// Supporting Types
// =============================================================================

/// Rotation angles (90° increments)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RotateAngle {
    Rotate90,
    Rotate180,
    Rotate270,
}

/// Interpolation method for resize/affine
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Interpolation {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos4,
}

/// Border mode for affine/pad
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BorderMode {
    Constant { value: u8 },
    Reflect,
    Replicate,
    Wrap,
}

/// Pad mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PadMode {
    Constant { value: u8 },
    Reflect,
    Replicate,
    Wrap,
}

/// Emboss direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EmbossDirection {
    TopLeft,
    Top,
    TopRight,
    Right,
    BottomRight,
    Bottom,
    BottomLeft,
    Left,
}

/// Edge detection method
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EdgeMethod {
    Sobel,
    Prewitt,
    Laplacian,
    Canny,
}

// =============================================================================
// Conversion Methods (single source of truth for all conversions)
// =============================================================================

impl SampledImageOp {
    /// Convert this SampledImageOp to a LabelTransform
    ///
    /// Returns Some(Box<dyn LabelTransform>) if this is a geometric transform.
    /// Returns None if this is a photometric transform (does not affect coordinates).
    pub fn to_label_transform(&self) -> Option<Box<dyn crate::core::LabelTransform>> {
        use crate::transforms::*;
        use crate::transforms::{PadMode as TransPadMode, RotateAngle as TransRotateAngle};

        match self {
            SampledImageOp::HorizontalFlip => Some(Box::new(HorizontalFlip)),
            SampledImageOp::VerticalFlip => Some(Box::new(VerticalFlip)),
            SampledImageOp::Transpose => Some(Box::new(Transpose)),
            SampledImageOp::Rotate { angle } => {
                let geo_angle = match angle {
                    crate::sampled_ir::ops::RotateAngle::Rotate90 => TransRotateAngle::Rotate90,
                    crate::sampled_ir::ops::RotateAngle::Rotate180 => TransRotateAngle::Rotate180,
                    crate::sampled_ir::ops::RotateAngle::Rotate270 => TransRotateAngle::Rotate270,
                };
                Some(Box::new(Rotate::new(geo_angle)))
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
                Some(Box::new(Resize::with_interpolation(
                    *width as usize,
                    *height as usize,
                    geo_interp,
                )))
            }
            SampledImageOp::Crop {
                x,
                y,
                width,
                height,
            } => Some(Box::new(Crop::new(*x, *y, *width, *height))),
            SampledImageOp::RandomCrop {
                width,
                height,
                fx,
                fy,
            } => Some(Box::new(crate::transforms::geometric::RandomCrop::new(
                *width, *height, *fx, *fy,
            ))),
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
                    TransPadMode::Constant(*v)
                } else {
                    match mode {
                        crate::sampled_ir::ops::PadMode::Constant { value } => {
                            TransPadMode::Constant(*value)
                        }
                        crate::sampled_ir::ops::PadMode::Reflect => TransPadMode::Reflect,
                        crate::sampled_ir::ops::PadMode::Replicate => TransPadMode::Replicate,
                        crate::sampled_ir::ops::PadMode::Wrap => TransPadMode::Replicate,
                    }
                };
                Some(Box::new(Pad::new(*top, *bottom, *left, *right, pad_mode)))
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
                    Interpolation::Bilinear | Interpolation::Bicubic | Interpolation::Lanczos4 => {
                        AffineInterpolation::Bilinear
                    }
                };
                let affine_border = match border_mode {
                    BorderMode::Constant { value } => AffineBorderMode::Constant { value: *value },
                    BorderMode::Reflect => AffineBorderMode::Reflect,
                    BorderMode::Replicate => AffineBorderMode::Replicate,
                    BorderMode::Wrap => AffineBorderMode::Wrap,
                };
                let params = AffineParams {
                    scale: *scale,
                    rotate: *rotate,
                    translate: *translate,
                    shear: *shear,
                };
                Some(Box::new(Affine::with_all(
                    params,
                    0,
                    0,
                    affine_interp,
                    affine_border,
                )))
            }
            _ => None,
        }
    }
}

// =============================================================================
// Accessor Methods (for optimizer)
// =============================================================================

impl SampledImageOp {
    /// Get the name of this operation for debugging
    pub fn name(&self) -> &'static str {
        match self {
            SampledImageOp::Brightness { .. } => "Brightness",
            SampledImageOp::Contrast { .. } => "Contrast",
            SampledImageOp::Gamma { .. } => "Gamma",
            SampledImageOp::HueSaturationValue { .. } => "HueSaturationValue",
            SampledImageOp::RGBShift { .. } => "RGBShift",
            SampledImageOp::ToGray => "ToGray",
            SampledImageOp::ToSepia => "ToSepia",
            SampledImageOp::ToRGB => "ToRGB",
            SampledImageOp::Invert => "Invert",
            SampledImageOp::Normalize { .. } => "Normalize",
            SampledImageOp::ColorTemperature { .. } => "ColorTemperature",
            SampledImageOp::ChannelMix { .. } => "ChannelMix",
            SampledImageOp::ColorBalance { .. } => "ColorBalance",
            SampledImageOp::ChannelShuffle { .. } => "ChannelShuffle",
            SampledImageOp::ColorTint { .. } => "ColorTint",
            SampledImageOp::Posterize { .. } => "Posterize",
            SampledImageOp::Solarize { .. } => "Solarize",
            SampledImageOp::Equalize => "Equalize",
            SampledImageOp::AutoContrast { .. } => "AutoContrast",
            SampledImageOp::GaussNoise { .. } => "GaussNoise",
            SampledImageOp::MultiplicativeNoise { .. } => "MultiplicativeNoise",
            SampledImageOp::SaltAndPepper { .. } => "SaltAndPepper",
            SampledImageOp::NoiseGranularity { .. } => "NoiseGranularity",
            SampledImageOp::CoarseDropout { .. } => "CoarseDropout",
            SampledImageOp::GridDropout { .. } => "GridDropout",
            SampledImageOp::HorizontalFlip => "HorizontalFlip",
            SampledImageOp::VerticalFlip => "VerticalFlip",
            SampledImageOp::Rotate { .. } => "Rotate",
            SampledImageOp::Transpose => "Transpose",
            SampledImageOp::Affine { .. } => "Affine",
            SampledImageOp::Resize { .. } => "Resize",
            SampledImageOp::Crop { .. } => "Crop",
            SampledImageOp::RandomCrop { .. } => "RandomCrop",
            SampledImageOp::Pad { .. } => "Pad",
            SampledImageOp::GaussianBlur { .. } => "GaussianBlur",
            SampledImageOp::MedianBlur { .. } => "MedianBlur",
            SampledImageOp::Sharpen { .. } => "Sharpen",
            SampledImageOp::Emboss { .. } => "Emboss",
            SampledImageOp::EdgeDetection { .. } => "EdgeDetection",
        }
    }

    /// Check if this operation preserves image shape (for fusion)
    pub fn preserves_shape(&self) -> bool {
        match self {
            // All photometric ops preserve shape
            SampledImageOp::Brightness { .. } => true,
            SampledImageOp::Contrast { .. } => true,
            SampledImageOp::Gamma { .. } => true,
            SampledImageOp::HueSaturationValue { .. } => true,
            SampledImageOp::RGBShift { .. } => true,
            SampledImageOp::ToGray => true,
            SampledImageOp::ToSepia => true,
            SampledImageOp::ToRGB => true,
            SampledImageOp::Invert => true,
            // Normalize changes dtype (u8 -> f32): terminal barrier even
            // though width/height are preserved.
            SampledImageOp::Normalize { .. } => false,
            SampledImageOp::ColorTemperature { .. } => true,
            SampledImageOp::ChannelMix { .. } => true,
            SampledImageOp::ColorBalance { .. } => true,
            SampledImageOp::ChannelShuffle { .. } => true,
            SampledImageOp::ColorTint { .. } => true,
            SampledImageOp::Posterize { .. } => true,
            SampledImageOp::Solarize { .. } => true,
            SampledImageOp::Equalize => true,
            SampledImageOp::AutoContrast { .. } => true,
            SampledImageOp::GaussNoise { .. } => true,
            SampledImageOp::MultiplicativeNoise { .. } => true,
            SampledImageOp::SaltAndPepper { .. } => true,
            SampledImageOp::NoiseGranularity { .. } => true,
            SampledImageOp::CoarseDropout { .. } => true,
            SampledImageOp::GridDropout { .. } => true,

            // Geometric: flips and transpose preserve shape
            SampledImageOp::HorizontalFlip => true,
            SampledImageOp::VerticalFlip => true,
            SampledImageOp::Transpose => true,
            SampledImageOp::Rotate { .. } => true,

            // Geometric: affine may change shape
            SampledImageOp::Affine { .. } => false,
            SampledImageOp::Resize { .. } => false,
            SampledImageOp::Crop { .. } => false,
            SampledImageOp::RandomCrop { .. } => false,
            SampledImageOp::Pad { .. } => false,

            // Kernel ops preserve shape
            SampledImageOp::GaussianBlur { .. } => true,
            SampledImageOp::MedianBlur { .. } => true,
            SampledImageOp::Sharpen { .. } => true,
            SampledImageOp::Emboss { .. } => true,
            SampledImageOp::EdgeDetection { .. } => true,
        }
    }

    /// Check if this operation is a barrier (cannot be fused)
    pub fn is_barrier(&self) -> bool {
        !self.preserves_shape()
    }

    /// Get the access pattern for this operation (via enum dispatch, no RTTI)
    pub fn access_pattern(&self) -> AccessPattern {
        match self {
            // All operations are InPlace (even Resize/Crop work in-place on the buffer)
            // The distinction is in shape_effect, not access_pattern
            _ => AccessPattern::InPlace,
        }
    }

    /// Get the shape effect for this operation (via enum dispatch, no RTTI)
    pub fn shape_effect(&self) -> ShapeEffect {
        if self.preserves_shape() {
            ShapeEffect::Preserve
        } else {
            ShapeEffect::Resize
        }
    }

    /// Check if this operation can be fused with others
    ///
    /// Fuseable operations are InPlace + Preserve.
    /// This is the opposite of is_barrier().
    pub fn is_fuseable(&self) -> bool {
        matches!(self.access_pattern(), AccessPattern::InPlace)
            && matches!(self.shape_effect(), ShapeEffect::Preserve)
    }

    /// Check if this is a LUT-based transform
    ///
    /// LUT transforms use a 256-entry lookup table for per-pixel operations.
    /// They can be fused together for better performance.
    ///
    /// Returns true for parameter-independent LUT ops (Brightness, Contrast, Gamma, etc.)
    /// Returns false for data-dependent LUT ops (Equalize, AutoContrast) - use is_data_dependent_lut_op()
    pub fn is_lut_op(&self) -> bool {
        matches!(
            self,
            SampledImageOp::Brightness { .. }
                | SampledImageOp::Contrast { .. }
                | SampledImageOp::Gamma { .. }
                | SampledImageOp::Invert
                | SampledImageOp::Posterize { .. }
                | SampledImageOp::Solarize { .. }
                | SampledImageOp::RGBShift { .. }
        )
    }


    /// Check if this is a data-dependent LUT transform
    ///
    /// Data-dependent LUT ops need to see the image data first to build their LUT.
    /// They can ONLY fuse with LUT ops that come AFTER them, not before.
    pub fn is_data_dependent_lut_op(&self) -> bool {
        matches!(
            self,
            SampledImageOp::Equalize | SampledImageOp::AutoContrast { .. }
        )
    }

    /// Build a 1-channel lookup table for this op if it is a pointwise LUT op
    pub fn build_lut(&self) -> Option<[u8; 256]> {
        match self {
            SampledImageOp::Brightness { delta } => {
                let mut lut = [0u8; 256];
                for i in 0..256 {
                    let x = i as f32;
                    let y = x + delta;
                    lut[i] = y.clamp(0.0, 255.0) as u8;
                }
                Some(lut)
            }
            SampledImageOp::Contrast { factor } => {
                let mut lut = [0u8; 256];
                let midpoint = 128.0;
                for i in 0..256 {
                    let x = i as f32;
                    let y = (x - midpoint) * factor + midpoint;
                    lut[i] = y.clamp(0.0, 255.0) as u8;
                }
                Some(lut)
            }
            SampledImageOp::Gamma { gamma } => {
                let mut lut = [0u8; 256];
                let g = *gamma;
                for i in 0u8..=255 {
                    let normalized = i as f32 / 255.0;
                    let corrected = normalized.powf(g);
                    lut[i as usize] = (corrected * 255.0).clamp(0.0, 255.0) as u8;
                }
                Some(lut)
            }
            SampledImageOp::Invert => {
                let mut lut = [0u8; 256];
                for i in 0..256 {
                    lut[i] = 255 - i as u8;
                }
                Some(lut)
            }
            SampledImageOp::Posterize { bits } => {
                let mut lut = [0u8; 256];
                let bits_to_discard = 8 - bits;
                for i in 0u8..=255 {
                    lut[i as usize] = (i >> bits_to_discard) << bits_to_discard;
                }
                Some(lut)
            }
            SampledImageOp::Solarize { threshold } => {
                let mut lut = [0u8; 256];
                for i in 0u8..=255 {
                    lut[i as usize] = if i >= *threshold { 255 - i } else { i };
                }
                Some(lut)
            }
            SampledImageOp::RGBShift { r_shift, g_shift, b_shift } => {
                let avg = (*r_shift as f32 + *g_shift as f32 + *b_shift as f32) / 3.0;
                let s = avg.round() as i16;
                let mut lut = [0u8; 256];
                for i in 0..256 {
                    lut[i] = (i as i16 + s).clamp(0, 255) as u8;
                }
                Some(lut)
            }
            _ => None,
        }
    }

    /// Build a 3-channel lookup table for 3-channel-aware LUT ops (like RGBShift)
    pub fn build_lut_3c(&self) -> Option<[[u8; 256]; 3]> {
        match self {
            SampledImageOp::RGBShift { r_shift, g_shift, b_shift } => {
                let r_s = *r_shift as i16;
                let g_s = *g_shift as i16;
                let b_s = *b_shift as i16;
                let mut r = [0u8; 256];
                let mut g = [0u8; 256];
                let mut b = [0u8; 256];
                for i in 0..256 {
                    r[i] = (i as i16 + r_s).clamp(0, 255) as u8;
                    g[i] = (i as i16 + g_s).clamp(0, 255) as u8;
                    b[i] = (i as i16 + b_s).clamp(0, 255) as u8;
                }
                Some([r, g, b])
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampled_image_op_names() {
        let op = SampledImageOp::Brightness { delta: 10.0 };
        assert_eq!(op.name(), "Brightness");

        let op = SampledImageOp::HorizontalFlip;
        assert_eq!(op.name(), "HorizontalFlip");
    }

    #[test]
    fn test_preserves_shape() {
        assert!(SampledImageOp::Brightness { delta: 10.0 }.preserves_shape());
        assert!(SampledImageOp::HorizontalFlip.preserves_shape());
        assert!(!SampledImageOp::Resize {
            width: 256,
            height: 256,
            interpolation: Interpolation::Bilinear
        }
        .preserves_shape());
    }

    #[test]
    fn test_is_barrier() {
        assert!(!SampledImageOp::Brightness { delta: 10.0 }.is_barrier());
        assert!(SampledImageOp::Resize {
            width: 256,
            height: 256,
            interpolation: Interpolation::Bilinear
        }
        .is_barrier());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let op = SampledImageOp::Brightness { delta: 42.0 };
        let bytes = bincode::serialize(&op).unwrap();
        let decoded: SampledImageOp = bincode::deserialize(&bytes).unwrap();

        match decoded {
            SampledImageOp::Brightness { delta } => assert_eq!(delta, 42.0),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_is_lut_op() {
        // Regular LUT ops
        assert!(SampledImageOp::Brightness { delta: 10.0 }.is_lut_op());
        assert!(SampledImageOp::Contrast { factor: 1.2 }.is_lut_op());
        assert!(SampledImageOp::Gamma { gamma: 0.8 }.is_lut_op());
        assert!(SampledImageOp::Invert.is_lut_op());
        // Normalize produces float32 output: it is a terminal barrier, not a LUT op
        assert!(!SampledImageOp::Normalize {
            mean: [0.5; 3],
            std: [0.5; 3]
        }
        .is_lut_op());
        assert!(SampledImageOp::Posterize { bits: 4 }.is_lut_op());
        assert!(SampledImageOp::Solarize { threshold: 128 }.is_lut_op());

        // Data-dependent LUT ops are NOT regular LUT ops
        assert!(!SampledImageOp::Equalize.is_lut_op());
        assert!(!SampledImageOp::AutoContrast {
            cutoff_low: 0.0,
            cutoff_high: 1.0
        }
        .is_lut_op());

        // Other ops are not LUT ops
        assert!(!SampledImageOp::HorizontalFlip.is_lut_op());
        assert!(!SampledImageOp::ToGray.is_lut_op());
        assert!(!SampledImageOp::GaussNoise {
            mean: 0.0,
            std: 1.0,
            seed: 0,
        }
        .is_lut_op());
    }

    #[test]
    fn test_is_data_dependent_lut_op() {
        // Data-dependent LUT ops
        assert!(SampledImageOp::Equalize.is_data_dependent_lut_op());
        assert!(SampledImageOp::AutoContrast {
            cutoff_low: 0.0,
            cutoff_high: 1.0
        }
        .is_data_dependent_lut_op());

        // Regular LUT ops are NOT data-dependent
        assert!(!SampledImageOp::Brightness { delta: 10.0 }.is_data_dependent_lut_op());
        assert!(!SampledImageOp::Contrast { factor: 1.2 }.is_data_dependent_lut_op());

        // Other ops are not data-dependent
        assert!(!SampledImageOp::HorizontalFlip.is_data_dependent_lut_op());
        assert!(!SampledImageOp::ToGray.is_data_dependent_lut_op());
    }
}
