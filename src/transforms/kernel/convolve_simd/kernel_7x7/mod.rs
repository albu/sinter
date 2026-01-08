// 7x7 kernel convolution implementations (Pascal's triangle [1 6 15 20 15 6 1] / 64)
//
// Provides both 1D horizontal/vertical passes and separable implementation.

use crate::core::FusableImage;

#[cfg(target_arch = "aarch64")]
mod neon;

// ============================================================================
// Public API re-exports
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::convolve_1d_horizontal_neon_7;

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::convolve_1d_vertical_neon_7;

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::convolve_separable_neon_7;
