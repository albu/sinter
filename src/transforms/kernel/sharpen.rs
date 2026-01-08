// Sharpen transform
//
// Applies a sharpening convolution kernel to enhance edges.
// When the `opencv` feature is enabled, uses OpenCV's optimized filter2D.

use super::convolve::convolve_3x3;
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

#[cfg(feature = "opencv")]
use opencv::{
    core::{Mat, MatTraitConst, CV_8U, CV_MAKETYPE},
    imgproc,
};

/// Sharpen transform
///
/// Enhances edges by applying a sharpening convolution kernel.
/// The standard sharpen kernel is:
///
///   0  -1   0
///  -1   5  -1
///   0  -1   0
///
/// # Parameters
/// - `strength`: Sharpening strength multiplier (default 1.0)
///   - Higher values increase sharpening effect
///   - Negative values produce blur effect
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sharpen {
    pub strength: f32,
}

impl Sharpen {
    /// Create a new Sharpen transform with default strength (1.0)
    pub fn new() -> Self {
        Self { strength: 1.0 }
    }

    /// Create a new Sharpen transform with custom strength
    ///
    /// # Panics
    /// Panics if strength is outside [-5.0, 5.0]
    pub fn with_strength(strength: f32) -> Self {
        assert!(
            (-5.0..=5.0).contains(&strength),
            "strength must be in [-5.0, 5.0], got {}",
            strength
        );
        Self { strength }
    }
}

impl Default for Sharpen {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Sharpen {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for Sharpen {
    fn execute(&self, image: &mut FusableImage) -> Option<crate::core::BarrierImage> {
        #[cfg(feature = "opencv")]
        {
            match self.execute_opencv(image) {
                Ok(_) => return None,
                Err(e) => {
                    eprintln!("OpenCV Sharpen failed: {}, using pure Rust fallback", e);
                    self.execute_rust(image);
                }
            }
        }
        #[cfg(not(feature = "opencv"))]
        {
            self.execute_rust(image);
        }
        None
    }
}

impl Sharpen {
    /// Pure Rust implementation (used as fallback or when opencv feature is disabled)
    fn execute_rust(&self, image: &mut FusableImage) {
        // Standard sharpen kernel
        //  0  -1   0
        // -1   5  -1
        //  0  -1   0

        // Adjust kernel based on strength
        // For strength s: center = 1 + 4s, neighbors = -s
        let s = self.strength;
        let center = (1.0 + 4.0 * s) as i32;
        let neighbor = (-s) as i32;

        let kernel = [0, neighbor, 0, neighbor, center, neighbor, 0, neighbor, 0];

        convolve_3x3(image, &kernel, 1, 0);
    }

    /// OpenCV implementation with zero-copy data wrapping
    #[cfg(feature = "opencv")]
    fn execute_opencv(&self, image: &mut FusableImage) -> opencv::Result<()> {
        let rows = image.height as i32;
        let cols = image.width as i32;
        let channels = image.channels as i32;
        let cv_type = CV_MAKETYPE(CV_8U, channels);

        // Standard sharpen kernel
        //  0  -1   0
        // -1   5  -1
        //  0  -1   0
        let s = self.strength;
        let center = 1.0 + 4.0 * s;
        let neighbor = -s;

        let kernel_data = vec![
            0.0f32, neighbor, 0.0, neighbor, center, neighbor, 0.0, neighbor, 0.0,
        ];

        unsafe {
            let src_mat = Mat::new_rows_cols_with_data_unsafe_def(
                rows,
                cols,
                cv_type,
                image.data.as_mut_ptr() as *mut std::ffi::c_void,
            )?;
            let mut dst_mat = Mat::new_rows_cols_with_data_unsafe_def(
                rows,
                cols,
                cv_type,
                image.data.as_mut_ptr() as *mut std::ffi::c_void,
            )?;

            let kernel_mat = Mat::new_rows_cols_with_data_unsafe_def(
                3,
                3,
                opencv::core::CV_32F,
                kernel_data.as_ptr() as *mut std::ffi::c_void,
            )?;

            imgproc::filter_2d(
                &src_mat,
                &mut dst_mat,
                -1, // Same depth as source
                &kernel_mat,
                opencv::core::Point::default(),
                0.0,
                opencv::core::BORDER_CONSTANT,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpen_new() {
        let s = Sharpen::new();
        assert_eq!(s.strength, 1.0);
    }

    #[test]
    fn test_sharpen_default() {
        let s = Sharpen::default();
        assert_eq!(s.strength, 1.0);
    }

    #[test]
    fn test_sharpen_with_strength() {
        let s = Sharpen::with_strength(2.0);
        assert_eq!(s.strength, 2.0);
    }

    #[test]
    fn test_sharpen_invalid_strength() {
        let result = std::panic::catch_unwind(|| {
            Sharpen::with_strength(10.0);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_sharpen_apply_constant() {
        // Constant image should remain mostly constant
        let mut data = vec![128u8; 9]; // 3x3 constant
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Sharpen::new().execute(&mut img);

        // All pixels should still be 128 (no edges to sharpen)
        assert!(img.data.iter().all(|&p| p == 128));
    }

    #[test]
    fn test_sharpen_apply_edge() {
        // Image with a sharp edge
        // 0 0 0
        // 0 255 255
        // 0 255 255
        let mut data = vec![0u8, 0u8, 0u8, 0u8, 255u8, 255u8, 0u8, 255u8, 255u8];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Sharpen::new().execute(&mut img);

        // Edge pixels should be enhanced (higher contrast)
        // Center pixel (1,1) is on the edge
        // Before: 0, 0, 0, 0, 255, 255, 0, 255, 255
        // After sharpen: center = -0 -0 -0 -0 + 5*255 -255 = 1020 - 255 = 765 -> 255
        assert_eq!(img.data[4], 255);
    }

    #[test]
    fn test_sharpen_rgb() {
        // Test RGB image
        let mut data = vec![
            100u8, 100u8, 100u8, 128u8, 128u8, 128u8, 150u8, 150u8, 150u8,
        ];
        let mut img = FusableImage::new(&mut data, 3, 1, 3);

        Sharpen::new().execute(&mut img);

        // Each channel should be processed independently
        // With a gradient like this, sharpening should enhance differences
        assert!(img.data[3] > 100); // R of middle pixel
        assert!(img.data[4] > 100); // G of middle pixel
        assert!(img.data[5] > 100); // B of middle pixel
    }
}
