// RandomImageNode: Pure enum for zero-allocation sampling
//
// This module defines the sampling layer as a pure data enum.
// No trait objects, no vtables, no RTTI - just data and pattern matching.
//
// The sampling phase walks the RandomImageNode tree and emits a flat
// Vec<SampledImageOp> to the SampledImageProgram.

use crate::core::{AccessPattern, ShapeEffect};
use crate::sampled_ir::ops::{BorderMode, EdgeMethod, EmbossDirection, Interpolation, PadMode, RotateAngle};
use crate::sampled_ir::SampledImageOp;

// Re-export distributions for use in sampling
pub use super::distributions::{Bernoulli, Dist, Uniform, UniformInt};

/// Draw a per-op 64-bit seed from the pipeline RNG. Keeps noise/dropout
/// transforms stochastic per pipeline execution (same pipeline seed ->
/// reproducible, different seeds -> different noise).
fn draw_seed(rng: &mut dyn Rng) -> u64 {
    let hi = rng.random_i32(i32::MAX) as u64;
    let lo = rng.random_i32(i32::MAX) as u64;
    (hi << 32) | lo
}

// Re-export the Rng trait
pub use super::traits::Rng;

/// Sampling context (passed during sampling phase)
pub struct SamplingContext<'a> {
    pub rng: &'a mut dyn Rng,
    pub seed: u64,
    pub index: usize,
}

impl<'a> SamplingContext<'a> {
    pub fn new(rng: &'a mut dyn Rng, seed: u64, index: usize) -> Self {
        Self { rng, seed, index }
    }
}

// =============================================================================
// Pure enum: No trait objects, no vtables
// =============================================================================

/// Random image transform node (pure enum, no trait objects)
///
/// This is a pure data enum - every variant holds its parameters directly.
/// Sampling is done via pattern matching with zero vtable overhead.
///
/// # Three Orthogonal Axes
///
/// This IR separates three concerns:
///
/// | Axis | Variants | Purpose |
/// |------|----------|---------|
/// | **Parameters** | Leaf transforms | WHAT values to sample |
/// | **Structure** | `All`, `OneOf`, `SomeOf` | HOW transforms compose |
/// | **Activation** | `Maybe { child, p }` | WHETHER something runs |
///
/// # Design Principles
///
/// - **Leaf transforms have NO `p`** - they hold only parameters
/// - **Activation is ALWAYS explicit** via `Maybe` wrapper
/// - **Structure is pure** - groups compose without activation logic
///
/// # Performance
///
/// - No allocation during sampling (just reads enum variants)
/// - No vtable lookups (pattern matching on enum)
/// - Cache-friendly (sequential memory access)
#[derive(Clone)]
pub enum RandomImageNode {
    // =========================================================================
    // Leaf Transforms (Parameters only, NO activation)
    // =========================================================================

    // Geometric transforms
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    Rotate {
        angle: RotateAngle,
    },
    Resize {
        width: u32,
        height: u32,
        interpolation: Interpolation,
    },
    Crop {
        x: Dist,
        y: Dist,
        width: Dist,
        height: Dist,
    },
    RandomCrop {
        width: u32,
        height: u32,
    },
    Pad {
        top: Dist,
        bottom: Dist,
        left: Dist,
        right: Dist,
        mode: PadMode,
        value: u8,
    },
    Affine {
        scale: (Dist, Dist),     // (scale_x, scale_y)
        rotate: Dist,            // rotation in degrees
        translate: (Dist, Dist), // (translate_x, translate_y) in pixels
        shear: (Dist, Dist),     // (shear_x, shear_y)
        interpolation: Interpolation,
        border_mode: BorderMode,
    },

    // Photometric transforms
    Brightness {
        delta: Dist,
    },
    Contrast {
        factor: Dist,
    },
    Posterize {
        bits: Dist,
    },
    Solarize {
        threshold: Dist,
    },
    Invert,
    Gamma {
        gamma: Dist,
    },
    Normalize {
        mean: Dist,
        std: Dist,
    },
    Equalize,
    AutoContrast {
        cutoff: Dist,
    },
    ToGray,
    ToSepia,
    ToRGB,

    // Noise transforms
    GaussNoise {
        mean: Dist,
        std: Dist,
    },
    MultiplicativeNoise {
        multiplier: Dist,
    },
    SaltAndPepper {
        amount: Dist,
        salt_vs_pepper: Dist,
    },

    // Color transforms
    RGBShift {
        r_shift: Dist,
        g_shift: Dist,
        b_shift: Dist,
    },
    HueSaturationValue {
        hue_shift: Dist,
        saturation_scale: Dist,
        value_scale: Dist,
    },
    ColorTemperature {
        temperature: Dist,
    },
    ColorTint {
        tint: [Dist; 4],
    },
    ColorBalance {
        r_scale: Dist,
        g_scale: Dist,
        b_scale: Dist,
    },
    ChannelShuffle {
        order: [usize; 3],
    }, // Permutation of [0, 1, 2]

    // Dropout transforms
    CoarseDropout {
        holes: Dist,
        hole_size: (Dist, Dist),
    },
    GridDropout {
        ratio: Dist,
        unit_size: Dist,
        holes: Dist,
    },

    // Kernel transforms
    GaussianBlur {
        kernel_size: u32,
    },
    GaussianBlurSigma {
        sigma: f32,
    }, // Removed quality parameter - not used in kernel
    MedianBlur {
        kernel_size: u32,
    },
    Sharpen {
        strength: Dist,
    },
    Emboss {
        direction: EmbossDirection,
        alpha: Dist,
        strength: Dist,
    },
    EdgeDetection {
        method: EdgeMethod,
    },

    // =========================================================================
    // Structural Nodes (Pure composition, NO activation)
    // =========================================================================
    /// Apply all children in sequence
    All {
        children: Vec<RandomImageNode>,
    },

    /// Apply exactly one child (chosen uniformly at random)
    OneOf {
        children: Vec<RandomImageNode>,
    },

    /// Apply k children where k is sampled from distribution n
    SomeOf {
        children: Vec<RandomImageNode>,
        n: Dist,
    },

    // =========================================================================
    // Activation (Bernoulli over any node)
    // =========================================================================
    /// Apply child with probability p (Bernoulli distribution)
    Maybe {
        child: Box<RandomImageNode>,
        p: Dist,
    },
}

impl RandomImageNode {
    /// Sample this node and emit ops to the output buffer
    ///
    /// This performs a recursive tree walk, flattening all structure
    /// into a single Vec<SampledImageOp>.
    ///
    /// # Sampling Semantics
    ///
    /// - **Leaves**: Always emit (sample parameters, emit op)
    /// - **Maybe**: If `p.sample_bool()` is true, sample child
    /// - **All**: Sample all children in sequence
    /// - **OneOf**: Sample exactly one child uniformly at random
    /// - **SomeOf**: Sample k children where k is sampled from distribution `n`
    ///
    /// # Parameters
    /// - `ctx`: Sampling context with RNG
    /// - `out`: Output buffer (appends sampled ops)
    ///
    /// # Performance
    ///
    /// - Zero allocations (just reads enum variants)
    /// - Zero vtable lookups (pattern matching)
    /// - Direct sampling logic inlined
    pub fn sample(&self, ctx: &mut SamplingContext, out: &mut Vec<SampledImageOp>) {
        match self {
            // =========================================================================
            // Leaf Transforms (always emit - no activation check)
            // =========================================================================
            RandomImageNode::HorizontalFlip => {
                out.push(SampledImageOp::HorizontalFlip);
            }
            RandomImageNode::VerticalFlip => {
                out.push(SampledImageOp::VerticalFlip);
            }
            RandomImageNode::Transpose => {
                out.push(SampledImageOp::Transpose);
            }
            RandomImageNode::Rotate { angle } => {
                out.push(SampledImageOp::Rotate { angle: *angle });
            }
            RandomImageNode::Resize {
                width,
                height,
                interpolation,
            } => {
                out.push(SampledImageOp::Resize {
                    width: *width,
                    height: *height,
                    interpolation: *interpolation,
                });
            }
            RandomImageNode::Crop {
                x,
                y,
                width,
                height,
            } => {
                let sampled_x = x.sample_i32(ctx.rng).max(0) as u32;
                let sampled_y = y.sample_i32(ctx.rng).max(0) as u32;
                let sampled_w = width.sample_i32(ctx.rng).max(1) as u32;
                let sampled_h = height.sample_i32(ctx.rng).max(1) as u32;
                out.push(SampledImageOp::Crop {
                    x: sampled_x,
                    y: sampled_y,
                    width: sampled_w,
                    height: sampled_h,
                });
            }
            RandomImageNode::RandomCrop { width, height } => {
                // Position resolves against the actual image size at execution
                // time, so only the fractional anchors are sampled here.
                let fx = ctx.rng.random_f32();
                let fy = ctx.rng.random_f32();
                out.push(SampledImageOp::RandomCrop {
                    width: *width,
                    height: *height,
                    fx,
                    fy,
                });
            }
            RandomImageNode::Pad {
                top,
                bottom,
                left,
                right,
                mode,
                value,
            } => {
                let sampled_top = top.sample_i32(ctx.rng).max(0) as u32;
                let sampled_bottom = bottom.sample_i32(ctx.rng).max(0) as u32;
                let sampled_left = left.sample_i32(ctx.rng).max(0) as u32;
                let sampled_right = right.sample_i32(ctx.rng).max(0) as u32;

                // If value is explicitly set (non-zero), override mode to Constant(value)
                // Otherwise use the mode as-is
                let pad_mode = if *value != 0 {
                    PadMode::Constant { value: *value }
                } else {
                    *mode
                };

                out.push(SampledImageOp::Pad {
                    top: sampled_top,
                    bottom: sampled_bottom,
                    left: sampled_left,
                    right: sampled_right,
                    mode: pad_mode,
                    value: if *value != 0 { Some(*value) } else { None },
                });
            }
            RandomImageNode::Affine {
                scale,
                rotate,
                translate,
                shear,
                interpolation,
                border_mode,
            } => {
                let sampled_scale_x = scale.0.sample_f32(ctx.rng);
                let sampled_scale_y = scale.1.sample_f32(ctx.rng);
                let sampled_rotate = rotate.sample_f32(ctx.rng);
                let sampled_translate_x = translate.0.sample_f32(ctx.rng);
                let sampled_translate_y = translate.1.sample_f32(ctx.rng);
                let sampled_shear_x = shear.0.sample_f32(ctx.rng);
                let sampled_shear_y = shear.1.sample_f32(ctx.rng);

                out.push(SampledImageOp::Affine {
                    scale: (sampled_scale_x, sampled_scale_y),
                    rotate: sampled_rotate,
                    translate: (sampled_translate_x, sampled_translate_y),
                    shear: (sampled_shear_x, sampled_shear_y),
                    interpolation: *interpolation,
                    border_mode: *border_mode,
                });
            }

            RandomImageNode::Invert => {
                out.push(SampledImageOp::Invert);
            }
            RandomImageNode::Brightness { delta } => {
                let sampled_delta = delta.sample_f32(ctx.rng);
                out.push(SampledImageOp::Brightness {
                    delta: sampled_delta,
                });
            }
            RandomImageNode::Contrast { factor } => {
                let sampled_factor = factor.sample_f32(ctx.rng);
                out.push(SampledImageOp::Contrast {
                    factor: sampled_factor,
                });
            }
            RandomImageNode::Posterize { bits } => {
                let sampled_bits = bits.sample_i32(ctx.rng) as u8;
                out.push(SampledImageOp::Posterize { bits: sampled_bits });
            }
            RandomImageNode::Solarize { threshold } => {
                let sampled_threshold = threshold.sample_i32(ctx.rng) as u8;
                out.push(SampledImageOp::Solarize {
                    threshold: sampled_threshold,
                });
            }
            RandomImageNode::Gamma { gamma } => {
                let sampled_gamma = gamma.sample_f32(ctx.rng);
                out.push(SampledImageOp::Gamma {
                    gamma: sampled_gamma,
                });
            }
            RandomImageNode::Normalize { mean, std } => {
                let sampled_mean = mean.sample_f32(ctx.rng);
                let sampled_std = std.sample_f32(ctx.rng);
                out.push(SampledImageOp::Normalize {
                    mean: [sampled_mean; 3],
                    std: [sampled_std; 3],
                });
            }
            RandomImageNode::Equalize => {
                out.push(SampledImageOp::Equalize);
            }
            RandomImageNode::AutoContrast { cutoff } => {
                let cutoff = cutoff.sample_f32(ctx.rng);
                out.push(SampledImageOp::AutoContrast {
                    cutoff_low: cutoff,
                    cutoff_high: cutoff,
                });
            }
            RandomImageNode::ToGray => {
                out.push(SampledImageOp::ToGray);
            }
            RandomImageNode::ToSepia => {
                out.push(SampledImageOp::ToSepia);
            }
            RandomImageNode::ToRGB => {
                out.push(SampledImageOp::ToRGB);
            }

            RandomImageNode::GaussNoise { mean, std } => {
                let sampled_mean = mean.sample_f32(ctx.rng);
                let sampled_std = std.sample_f32(ctx.rng);
                out.push(SampledImageOp::GaussNoise {
                    mean: sampled_mean,
                    std: sampled_std,
                    seed: draw_seed(ctx.rng),
                });
            }
            RandomImageNode::MultiplicativeNoise { multiplier } => {
                let sampled_multiplier = multiplier.sample_f32(ctx.rng);
                out.push(SampledImageOp::MultiplicativeNoise {
                    multiplier: sampled_multiplier,
                    seed: draw_seed(ctx.rng),
                });
            }
            RandomImageNode::SaltAndPepper {
                amount,
                salt_vs_pepper,
            } => {
                let sampled_amount = amount.sample_f32(ctx.rng);
                let sampled_salt_vs_pepper = salt_vs_pepper.sample_f32(ctx.rng);
                out.push(SampledImageOp::SaltAndPepper {
                    amount: sampled_amount,
                    salt_vs_pepper: sampled_salt_vs_pepper,
                    seed: draw_seed(ctx.rng),
                });
            }

            RandomImageNode::RGBShift {
                r_shift,
                g_shift,
                b_shift,
            } => {
                let sampled_r = r_shift.sample_i32(ctx.rng);
                let sampled_g = g_shift.sample_i32(ctx.rng);
                let sampled_b = b_shift.sample_i32(ctx.rng);
                out.push(SampledImageOp::RGBShift {
                    r_shift: sampled_r,
                    g_shift: sampled_g,
                    b_shift: sampled_b,
                });
            }
            RandomImageNode::HueSaturationValue {
                hue_shift,
                saturation_scale,
                value_scale,
            } => {
                let sampled_hue = hue_shift.sample_i32(ctx.rng);
                let sampled_sat = saturation_scale.sample_f32(ctx.rng);
                let sampled_val = value_scale.sample_f32(ctx.rng);
                out.push(SampledImageOp::HueSaturationValue {
                    hue_shift: sampled_hue,
                    saturation_scale: sampled_sat,
                    value_scale: sampled_val,
                });
            }
            RandomImageNode::ColorTemperature { temperature } => {
                let sampled_temp = temperature.sample_f32(ctx.rng);
                out.push(SampledImageOp::ColorTemperature {
                    temperature: sampled_temp,
                });
            }
            RandomImageNode::ColorTint { tint } => {
                let sampled_tint: [f32; 4] = [
                    tint[0].sample_f32(ctx.rng),
                    tint[1].sample_f32(ctx.rng),
                    tint[2].sample_f32(ctx.rng),
                    tint[3].sample_f32(ctx.rng),
                ];
                out.push(SampledImageOp::ColorTint { tint: sampled_tint });
            }
            RandomImageNode::ColorBalance {
                r_scale,
                g_scale,
                b_scale,
            } => {
                let sampled_r = r_scale.sample_f32(ctx.rng);
                let sampled_g = g_scale.sample_f32(ctx.rng);
                let sampled_b = b_scale.sample_f32(ctx.rng);
                out.push(SampledImageOp::ColorBalance {
                    shadows: [0.0, 0.0, 0.0],
                    midtones: [0.0, 0.0, 0.0],
                    highlights: [sampled_r, sampled_g, sampled_b],
                });
            }
            RandomImageNode::ChannelShuffle { order } => {
                out.push(SampledImageOp::ChannelShuffle { order: *order });
            }

            RandomImageNode::CoarseDropout { holes, hole_size } => {
                let sampled_holes = holes.sample_i32(ctx.rng);
                let sampled_h = hole_size.0.sample_i32(ctx.rng).max(1) as u32;
                let sampled_w = hole_size.1.sample_i32(ctx.rng).max(1) as u32;
                out.push(SampledImageOp::CoarseDropout {
                    holes: sampled_holes.max(1) as usize,
                    hole_size: (sampled_h, sampled_w),
                    seed: draw_seed(ctx.rng),
                });
            }
            RandomImageNode::GridDropout {
                ratio,
                unit_size,
                holes,
            } => {
                let sampled_ratio = ratio.sample_f32(ctx.rng);
                let sampled_unit = unit_size.sample_i32(ctx.rng).max(1) as u32;
                let sampled_holes = holes.sample_i32(ctx.rng).max(1) as usize;
                out.push(SampledImageOp::GridDropout {
                    ratio: sampled_ratio,
                    unit_size: sampled_unit,
                    holes: sampled_holes,
                    seed: draw_seed(ctx.rng),
                });
            }

            RandomImageNode::GaussianBlur { kernel_size } => {
                out.push(SampledImageOp::GaussianBlur {
                    kernel_size: *kernel_size,
                    sigma: 0.0,
                });
            }
            RandomImageNode::GaussianBlurSigma { sigma } => {
                // Emit sigma-based blur (sigma > 0 triggers sigma-agnostic path)
                out.push(SampledImageOp::GaussianBlur {
                    kernel_size: 0,
                    sigma: *sigma,
                });
            }
            RandomImageNode::MedianBlur { kernel_size } => {
                out.push(SampledImageOp::MedianBlur {
                    kernel_size: *kernel_size,
                });
            }
            RandomImageNode::Sharpen { strength } => {
                let sampled_strength = strength.sample_f32(ctx.rng);
                out.push(SampledImageOp::Sharpen {
                    strength: sampled_strength,
                });
            }
            RandomImageNode::Emboss {
                direction,
                alpha,
                strength,
            } => {
                let sampled_alpha = alpha.sample_f32(ctx.rng);
                let sampled_strength = strength.sample_f32(ctx.rng);
                out.push(SampledImageOp::Emboss {
                    direction: *direction,
                    alpha: sampled_alpha,
                    strength: sampled_strength,
                });
            }
            RandomImageNode::EdgeDetection { method } => {
                out.push(SampledImageOp::EdgeDetection { method: *method });
            }

            // =========================================================================
            // Structural Nodes
            // =========================================================================
            RandomImageNode::All { children } => {
                for child in children {
                    child.sample(ctx, out);
                }
            }

            RandomImageNode::OneOf { children } => {
                assert!(!children.is_empty(), "OneOf requires at least one child");
                let idx = sample_index(ctx.rng, children.len());
                children[idx].sample(ctx, out);
            }

            RandomImageNode::SomeOf { children, n } => {
                assert!(!children.is_empty(), "SomeOf requires at least one child");

                // Sample k from distribution n
                let k = n.sample_i32(ctx.rng).clamp(1, children.len() as i32) as usize;

                // Sample k distinct children
                let indices = sample_indices(ctx.rng, children.len(), k);
                for idx in indices {
                    children[idx].sample(ctx, out);
                }
            }

            // =========================================================================
            // Activation
            // =========================================================================
            RandomImageNode::Maybe { child, p } => {
                if p.sample_bool(ctx.rng) {
                    child.sample(ctx, out);
                }
            }
        }
    }

    /// Get the access pattern for this node
    pub fn access(&self) -> AccessPattern {
        match self {
            // Geometric: InPlace (shape-preserving)
            RandomImageNode::HorizontalFlip
            | RandomImageNode::VerticalFlip
            | RandomImageNode::Transpose
            | RandomImageNode::Rotate { .. } => AccessPattern::InPlace,

            // Geometric: OutOfPlace (shape-changing)
            RandomImageNode::Resize { .. }
            | RandomImageNode::Crop { .. }
            | RandomImageNode::RandomCrop { .. }
            | RandomImageNode::Pad { .. }
            | RandomImageNode::Affine { .. } => AccessPattern::OutOfPlace,

            // Photometric: InPlace
            RandomImageNode::Invert
            | RandomImageNode::Brightness { .. }
            | RandomImageNode::Contrast { .. }
            | RandomImageNode::Posterize { .. }
            | RandomImageNode::Solarize { .. }
            | RandomImageNode::Gamma { .. }
            | RandomImageNode::Normalize { .. }
            | RandomImageNode::Equalize
            | RandomImageNode::AutoContrast { .. }
            | RandomImageNode::ToGray
            | RandomImageNode::ToSepia => AccessPattern::InPlace,

            RandomImageNode::ToRGB => AccessPattern::OutOfPlace, // channel count change

            // Noise: InPlace
            RandomImageNode::GaussNoise { .. }
            | RandomImageNode::MultiplicativeNoise { .. }
            | RandomImageNode::SaltAndPepper { .. } => AccessPattern::InPlace,

            // Color: InPlace
            RandomImageNode::RGBShift { .. }
            | RandomImageNode::HueSaturationValue { .. }
            | RandomImageNode::ColorTemperature { .. }
            | RandomImageNode::ColorTint { .. }
            | RandomImageNode::ColorBalance { .. }
            | RandomImageNode::ChannelShuffle { .. } => AccessPattern::InPlace,

            // Dropout: InPlace
            RandomImageNode::CoarseDropout { .. } | RandomImageNode::GridDropout { .. } => {
                AccessPattern::InPlace
            }

            // Kernel: OutOfPlace (allocates new buffer)
            RandomImageNode::GaussianBlur { .. }
            | RandomImageNode::GaussianBlurSigma { .. }
            | RandomImageNode::MedianBlur { .. }
            | RandomImageNode::Sharpen { .. }
            | RandomImageNode::Emboss { .. }
            | RandomImageNode::EdgeDetection { .. } => AccessPattern::OutOfPlace,

            // Structural: delegate to first child
            RandomImageNode::All { children }
            | RandomImageNode::OneOf { children }
            | RandomImageNode::SomeOf { children, .. } => children
                .first()
                .map(|c| c.access())
                .unwrap_or(AccessPattern::InPlace),

            // Activation: delegate to child
            RandomImageNode::Maybe { child, .. } => child.access(),
        }
    }

    /// Get the shape effect for this node
    pub fn shape_effect(&self) -> ShapeEffect {
        match self {
            // Geometric: Preserve
            RandomImageNode::HorizontalFlip
            | RandomImageNode::VerticalFlip
            | RandomImageNode::Transpose
            | RandomImageNode::Rotate { .. } => ShapeEffect::Preserve,

            // Geometric: shape-changing
            RandomImageNode::Resize { .. } => ShapeEffect::Resize,
            RandomImageNode::Crop { .. } => ShapeEffect::Crop,
            RandomImageNode::RandomCrop { .. } => ShapeEffect::Crop,
            RandomImageNode::Pad { .. } => ShapeEffect::Pad,
            RandomImageNode::Affine { .. } => ShapeEffect::Resize,

            // All photometric/noise/color/kernel/dropout: Preserve
            RandomImageNode::Invert
            | RandomImageNode::Brightness { .. }
            | RandomImageNode::Contrast { .. }
            | RandomImageNode::Posterize { .. }
            | RandomImageNode::Solarize { .. }
            | RandomImageNode::Gamma { .. }
            | RandomImageNode::Normalize { .. }
            | RandomImageNode::Equalize
            | RandomImageNode::AutoContrast { .. }
            | RandomImageNode::ToGray
            | RandomImageNode::ToSepia
            | RandomImageNode::ToRGB
            | RandomImageNode::GaussNoise { .. }
            | RandomImageNode::MultiplicativeNoise { .. }
            | RandomImageNode::SaltAndPepper { .. }
            | RandomImageNode::RGBShift { .. }
            | RandomImageNode::HueSaturationValue { .. }
            | RandomImageNode::ColorTemperature { .. }
            | RandomImageNode::ColorTint { .. }
            | RandomImageNode::ColorBalance { .. }
            | RandomImageNode::ChannelShuffle { .. }
            | RandomImageNode::CoarseDropout { .. }
            | RandomImageNode::GridDropout { .. }
            | RandomImageNode::GaussianBlur { .. }
            | RandomImageNode::GaussianBlurSigma { .. }
            | RandomImageNode::MedianBlur { .. }
            | RandomImageNode::Sharpen { .. }
            | RandomImageNode::Emboss { .. }
            | RandomImageNode::EdgeDetection { .. } => ShapeEffect::Preserve,

            // Structural: delegate to first child
            RandomImageNode::All { children }
            | RandomImageNode::OneOf { children }
            | RandomImageNode::SomeOf { children, .. } => children
                .first()
                .map(|c| c.shape_effect())
                .unwrap_or(ShapeEffect::Preserve),

            // Activation: delegate to child
            RandomImageNode::Maybe { child, .. } => child.shape_effect(),
        }
    }
}

// =============================================================================
// Helper functions for sampling
// =============================================================================

/// Sample a single index in [0, upper)
fn sample_index(rng: &mut dyn Rng, upper: usize) -> usize {
    UniformInt::new(0, upper as i32 - 1).sample(rng) as usize
}

/// Sample k distinct indices from [0, n) using Fisher-Yates
fn sample_indices(rng: &mut dyn Rng, n: usize, k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();

    // Partial Fisher-Yates: shuffle only the elements we need
    for i in (n - k..n).rev() {
        let j = UniformInt::new(0, i as i32).sample(rng) as usize;
        indices.swap(i, j);
    }

    indices.into_iter().skip(n - k).collect()
}

// =============================================================================
// RandomImageProgram - User-facing random transform container
// =============================================================================

/// A random image transform program
///
/// This is the user-facing type that holds random transforms before sampling.
/// It contains a `RandomImageNode` tree which can be sampled to produce
/// a deterministic `SampledImageProgram`.
#[derive(Clone)]
pub struct RandomImageProgram {
    /// The root node (typically All)
    root: RandomImageNode,
}

impl std::fmt::Debug for RandomImageProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomImageProgram")
            .field("num_ops", &self.len())
            .finish()
    }
}

// =============================================================================
// SeededRng wrapper for compatibility
// =============================================================================

/// Simple seeded RNG wrapper that implements our Rng trait
struct SeededRng {
    inner: rand_chacha::ChaCha8Rng,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            inner: rand_chacha::ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl super::Rng for SeededRng {
    fn random_f32(&mut self) -> f32 {
        use rand::Rng;
        self.inner.gen()
    }

    fn random_i32(&mut self, upper: i32) -> i32 {
        use rand::Rng;
        self.inner.gen_range(0..upper)
    }
}

impl RandomImageProgram {
    /// Create a new empty program
    pub fn new() -> Self {
        Self {
            root: RandomImageNode::All {
                children: Vec::new(),
            },
        }
    }

    /// Add a random node to the program
    pub fn add(&mut self, node: RandomImageNode) {
        if let RandomImageNode::All { ref mut children } = self.root {
            children.push(node);
        }
    }

    /// Sample this program with a seed
    ///
    /// Uses a simple ChaCha8 RNG seeded with the provided seed.
    pub fn sample_with_seed(&self, seed: u64) -> crate::sampled_ir::SampledImageProgram {
        let mut rng = SeededRng::new(seed);
        self.sample(&mut rng)
    }

    /// Sample this program with a custom RNG
    pub fn sample(&self, rng: &mut dyn Rng) -> crate::sampled_ir::SampledImageProgram {
        let mut ctx = SamplingContext::new(rng, 0, 0);
        let mut ops = Vec::new();
        self.root.sample(&mut ctx, &mut ops);

        crate::sampled_ir::SampledImageProgram {
            version: crate::sampled_ir::IR_VERSION,
            ops,
        }
    }

    /// Is this program empty?
    pub fn is_empty(&self) -> bool {
        match &self.root {
            RandomImageNode::All { children } => children.is_empty(),
            _ => false,
        }
    }

    /// Number of top-level nodes in the program
    pub fn len(&self) -> usize {
        match &self.root {
            RandomImageNode::All { children } => children.len(),
            _ => 1,
        }
    }
}

impl Default for RandomImageProgram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
