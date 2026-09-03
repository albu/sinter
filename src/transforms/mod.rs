// Photometric and geometric transforms
//
// This module contains:
// - Photometric transforms (per-pixel operations that can be fused)
// - Geometric transforms (spatial rearrangements)
// - Kernel transforms (convolution operations)
// - Runtime: Fusion infrastructure (LUT, matrix, pixel ops, utilities)

pub mod geometric;
pub mod kernel;
pub mod photometric;
pub mod runtime;

#[cfg(test)]
mod tests;

// Re-export for convenience
pub use geometric::{
    Affine, AffineParams, AnyRes, Crop, HorizontalFlip, Orientation, Pad, PadMode, Resize, Rotate,
    RotateAngle, StructuralKernel, Transpose, VerticalFlip,
};
pub use kernel::{
    BlurQuality, EdgeDetection, EdgeMethod, Emboss, EmbossDirection, GaussianBlur,
    GaussianBlurSigma, KernelSize, MedianBlur, MedianKernelSize, MedianMode, Sharpen,
};
pub use photometric::{
    AutoContrast, Brightness, ChannelMix, ChannelOrder, ChannelShuffle, CoarseDropout,
    ColorBalance, ColorJitter, ColorTemperature, ColorTint, Contrast, Equalize, Gamma, GaussNoise,
    GridDropout, HueSaturationValue, Invert, MultiplicativeNoise, NoiseGranularity, Normalize,
    Posterize, RGBShift, SaltAndPepper, Solarize, ToGray, ToRGB, ToSepia,
};

// Re-export runtime types
pub use runtime::{
    clamp, execute_fused, apply_matrix, compose_matrices,
    FusedExecutor, FusedLut, FusedLutExecutor, FusedMatrix, LutExecutor, LutOp, MatrixExecutor,
    MatrixOp, PixelOp, RangedOp, ValueRange,
};
