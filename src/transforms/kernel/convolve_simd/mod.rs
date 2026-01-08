// SIMD-optimized separable convolution module
//
// Specialized NEON implementations for 3x3, 5x5, and 7x7 kernels.
// For larger kernels or sigma-based blur, use GaussianBlurSigma.

mod kernel_3x3;
mod kernel_5x5;
mod kernel_7x7;

use crate::core::FusableImage;

// Re-export kernel implementations (for internal use within convolve_simd module)
#[cfg(target_arch = "aarch64")]
use kernel_3x3::{
    convolve_1d_horizontal_neon_3, convolve_1d_vertical_neon_3, convolve_separable_neon_3,
};
#[cfg(target_arch = "aarch64")]
use kernel_5x5::{
    convolve_1d_horizontal_neon_5, convolve_1d_vertical_neon_5, convolve_separable_neon_5,
};
#[cfg(target_arch = "aarch64")]
use kernel_7x7::{
    convolve_1d_horizontal_neon_7, convolve_1d_vertical_neon_7, convolve_separable_neon_7,
};

/// Detect and apply full separable convolution (horizontal + vertical)
///
/// Only supports 3, 5, and 7-tap kernels with NEON optimization.
/// For larger kernels, use GaussianBlurSigma which forwards to OpenCV.
pub fn convolve_separable_detect(image: &mut FusableImage, kernel: &[i32], scale: i32) {
    #[cfg(target_arch = "aarch64")]
    {
        match kernel.len() {
            3 => unsafe {
                convolve_separable_neon_3(image, kernel, scale);
                return;
            },
            5 => unsafe {
                convolve_separable_neon_5(image, kernel, scale);
                return;
            },
            7 => unsafe {
                convolve_separable_neon_7(image, kernel, scale);
                return;
            },
            _ => {}
        }
    }

    // Fallback to scalar for unsupported kernel sizes
    super::convolve::convolve_1d_horizontal(image, kernel, scale);
    super::convolve::convolve_1d_vertical(image, kernel, scale);
}

/// Detect CPU features and apply optimized 1D horizontal convolution
///
/// Supports 3, 5, and 7-tap kernels with NEON optimization.
pub fn convolve_1d_horizontal_detect(image: &mut FusableImage, kernel: &[i32], scale: i32) {
    #[cfg(target_arch = "aarch64")]
    {
        match kernel.len() {
            3 => unsafe {
                convolve_1d_horizontal_neon_3(image, kernel, scale);
                return;
            },
            5 => unsafe {
                convolve_1d_horizontal_neon_5(image, kernel, scale);
                return;
            },
            7 => unsafe {
                convolve_1d_horizontal_neon_7(image, kernel, scale);
                return;
            },
            _ => {
                // Use scalar fallback for larger kernels
                super::convolve::convolve_1d_horizontal(image, kernel, scale);
                return;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        super::convolve::convolve_1d_horizontal(image, kernel, scale);
    }
}

/// Detect CPU features and apply optimized 1D vertical convolution
///
/// Supports 3, 5, and 7-tap kernels with NEON optimization.
pub fn convolve_1d_vertical_detect(image: &mut FusableImage, kernel: &[i32], scale: i32) {
    #[cfg(target_arch = "aarch64")]
    {
        match kernel.len() {
            3 => unsafe {
                convolve_1d_vertical_neon_3(image, kernel, scale);
                return;
            },
            5 => unsafe {
                convolve_1d_vertical_neon_5(image, kernel, scale);
                return;
            },
            7 => unsafe {
                convolve_1d_vertical_neon_7(image, kernel, scale);
                return;
            },
            _ => {
                super::convolve::convolve_1d_vertical(image, kernel, scale);
                return;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        super::convolve::convolve_1d_vertical(image, kernel, scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_1d_horizontal_7_constant() {
        let mut data = vec![128u8; 21];
        let mut img = FusableImage::new(&mut data, 7, 3, 1);
        let kernel = [1i32, 6, 15, 20, 15, 6, 1];
        convolve_1d_horizontal_detect(&mut img, &kernel[..], 64);
        assert!(img.data.iter().all(|&p| p == 128));
    }

    #[test]
    fn test_convolve_1d_vertical_7_constant() {
        let mut data = vec![128u8; 21];
        let mut img = FusableImage::new(&mut data, 7, 3, 1);
        let kernel = [1i32, 6, 15, 20, 15, 6, 1];
        convolve_1d_vertical_detect(&mut img, &kernel[..], 64);
        assert!(img.data.iter().all(|&p| p == 128));
    }
}
