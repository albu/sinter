// Kernel-based transforms (convolution operations)
//
// These transforms require neighborhood access (convolution with a kernel),
// so they cannot be fused into the single-pass PixelOp executor.
// They are InPlace + Preserve but implement their own execution logic.

pub mod convolve;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub mod convolve_simd;
pub mod box_blur;
pub mod recursive_gaussian;
pub mod sharpen;
pub mod emboss;
pub mod edge_detection;
pub mod gaussian_blur;
pub mod gaussian;
pub mod median_blur;

// Re-export for convenience
pub use sharpen::Sharpen;
pub use emboss::{Emboss, EmbossDirection};
pub use edge_detection::{EdgeDetection, EdgeMethod};
pub use gaussian_blur::{GaussianBlur, KernelSize};
pub use gaussian::{GaussianBlurSigma, BlurQuality};
pub use median_blur::{MedianBlur, MedianKernelSize, MedianMode};
