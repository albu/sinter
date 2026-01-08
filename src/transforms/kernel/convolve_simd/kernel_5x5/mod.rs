// 5x5 kernel convolution implementations (Gaussian [1 4 6 4 1] / 16)
//
// Provides both 1D horizontal/vertical passes and separable implementation.

use crate::core::FusableImage;

mod neon;

// Re-export the NEON functions
pub(crate) use neon::{
    convolve_1d_horizontal_neon_5,
    convolve_1d_vertical_neon_5,
    convolve_separable_neon_5,
};
