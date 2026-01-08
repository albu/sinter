// Emboss transform
//
// Applies an embossing convolution kernel to create a 3D relief effect.
// When the `opencv` feature is enabled, uses OpenCV's optimized filter2D.

use super::convolve::convolve_3x3;
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

#[cfg(feature = "opencv")]
use opencv::{
    core::{Mat, MatTraitConst, BORDER_CONSTANT, CV_8U, CV_MAKETYPE},
    imgproc,
};

/// Direction for emboss effect
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbossDirection {
    /// Emboss from bottom-left to top-right
    SouthEast,
    /// Emboss from top-left to bottom-right
    SouthWest,
    /// Emboss from bottom-right to top-left
    NorthEast,
    /// Emboss from top-right to bottom-left
    NorthWest,
}

/// Emboss transform
///
/// Creates a 3D relief effect by applying an embossing convolution kernel.
/// The effect highlights edges in a specific direction, making the image appear
/// carved or stamped into the surface.
///
/// Uses a blend-based approach compatible with albumentations: the result is
/// a blend between the original image and the emboss effect, controlled by `alpha`.
///
/// # Parameters
/// - `direction`: Direction of the light source for the emboss effect
/// - `alpha`: Blend factor (0.0 = original image, 1.0 = full emboss, default 0.5)
/// - `strength`: Strength of the emboss effect (default 0.5)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Emboss {
    pub direction: EmbossDirection,
    pub alpha: f32,
    pub strength: f32,
}

impl Emboss {
    /// Create a new Emboss transform with default settings
    ///
    /// Default direction is SouthEast with alpha=0.5 and strength=0.5
    pub fn new() -> Self {
        Self {
            direction: EmbossDirection::SouthEast,
            alpha: 0.5,
            strength: 0.5,
        }
    }

    /// Set the direction of the emboss effect
    pub fn with_direction(mut self, direction: EmbossDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set the alpha blend factor (0.0 = original, 1.0 = full emboss)
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0.0, 1.0], got {}",
            alpha
        );
        self.alpha = alpha;
        self
    }

    /// Set the strength of the emboss effect
    ///
    /// # Panics
    /// Panics if strength is negative
    pub fn with_strength(mut self, strength: f32) -> Self {
        assert!(
            strength >= 0.0,
            "strength must be non-negative, got {}",
            strength
        );
        self.strength = strength;
        self
    }
}

impl Default for Emboss {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Emboss {
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

impl Executable for Emboss {
    fn execute(&self, image: &mut FusableImage) -> Option<crate::core::BarrierImage> {
        #[cfg(feature = "opencv")]
        {
            match self.execute_opencv(image) {
                Ok(_) => return None,
                Err(e) => {
                    eprintln!("OpenCV Emboss failed: {}, using pure Rust fallback", e);
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

impl Emboss {
    /// Pure Rust implementation (used as fallback or when opencv feature is disabled)
    fn execute_rust(&self, image: &mut FusableImage) {
        // Blend-based emboss kernel (compatible with albumentations)
        // kernel = (1 - alpha) * identity + alpha * emboss_effect
        // The center is always 1.0, so no offset needed

        let s = self.strength;
        let a = self.alpha;

        // Emboss effect kernel for each direction (diagonal gradient)
        let effect = match self.direction {
            EmbossDirection::SouthEast => {
                // Bottom-left to top-right
                [-1.0 - s, -s, 0.0, -s, 1.0, s, 0.0, s, 1.0 + s]
            }
            EmbossDirection::SouthWest => {
                // Top-left to bottom-right
                [0.0, -s, -1.0 - s, s, 1.0, -s, 1.0 + s, s, 0.0]
            }
            EmbossDirection::NorthEast => {
                // Bottom-right to top-left
                [1.0 + s, s, 0.0, s, 1.0, -s, 0.0, -s, -1.0 - s]
            }
            EmbossDirection::NorthWest => {
                // Top-right to bottom-left
                [0.0, s, 1.0 + s, -s, 1.0, s, -1.0 - s, -s, 0.0]
            }
        };

        // Identity kernel (no change) - center is 1.0
        let identity = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        // Blend: (1 - alpha) * identity + alpha * effect
        let kernel_f32: [f32; 9] = std::array::from_fn(|i| (1.0 - a) * identity[i] + a * effect[i]);

        // Scale by 256 to convert to fixed-point (avoids float operations in inner loop)
        // This gives us 8 bits of fractional precision
        let kernel_i32: [i32; 9] = std::array::from_fn(|i| (kernel_f32[i] * 256.0) as i32);

        // Apply convolution with scale=256 (to undo the fixed-point scaling)
        // No offset needed since center weight is ~1.0
        convolve_3x3(image, &kernel_i32, 256, 0);
    }

    /// OpenCV implementation with zero-copy data wrapping
    #[cfg(feature = "opencv")]
    fn execute_opencv(&self, image: &mut FusableImage) -> opencv::Result<()> {
        // Removed OnceLock - set_num_threads(1) is called once per process startup
        // The overhead from checking OnceLock was measurable

        let rows = image.height as i32;
        let cols = image.width as i32;
        let channels = image.channels as i32;
        let cv_type = CV_MAKETYPE(CV_8U, channels);

        // Blend-based emboss kernel (compatible with albumentations)
        let s = self.strength;
        let a = self.alpha;

        // Emboss effect kernel for each direction
        let effect = match self.direction {
            EmbossDirection::SouthEast => [-1.0f32 - s, -s, 0.0, -s, 1.0, s, 0.0, s, 1.0 + s],
            EmbossDirection::SouthWest => [0.0f32, -s, -1.0 - s, s, 1.0, -s, 1.0 + s, s, 0.0],
            EmbossDirection::NorthEast => [1.0f32 + s, s, 0.0, s, 1.0, -s, 0.0, -s, -1.0 - s],
            EmbossDirection::NorthWest => [0.0f32, s, 1.0 + s, -s, 1.0, s, -1.0 - s, -s, 0.0],
        };

        // Identity kernel
        let identity = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        // Blend: (1 - alpha) * identity + alpha * effect
        let kernel_data: Vec<f32> = (0..9)
            .map(|i| (1.0 - a) * identity[i] + a * effect[i])
            .collect();

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

            // No delta needed - blend-based kernel keeps values in valid range
            imgproc::filter_2d(
                &src_mat,
                &mut dst_mat,
                -1, // Same depth as source
                &kernel_mat,
                opencv::core::Point::default(),
                0.0, // No offset (blend-based approach)
                BORDER_CONSTANT,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emboss_new() {
        let e = Emboss::new();
        assert_eq!(e.direction, EmbossDirection::SouthEast);
        assert_eq!(e.alpha, 0.5);
        assert_eq!(e.strength, 0.5);
    }

    #[test]
    fn test_emboss_default() {
        let e = Emboss::default();
        assert_eq!(e.direction, EmbossDirection::SouthEast);
        assert_eq!(e.alpha, 0.5);
        assert_eq!(e.strength, 0.5);
    }

    #[test]
    fn test_emboss_with_direction() {
        let e = Emboss::new().with_direction(EmbossDirection::NorthWest);
        assert_eq!(e.direction, EmbossDirection::NorthWest);
    }

    #[test]
    fn test_emboss_with_alpha() {
        let e = Emboss::new().with_alpha(0.8);
        assert_eq!(e.alpha, 0.8);
    }

    #[test]
    fn test_emboss_with_strength() {
        let e = Emboss::new().with_strength(1.5);
        assert_eq!(e.strength, 1.5);
    }

    #[test]
    fn test_emboss_invalid_strength() {
        let result = std::panic::catch_unwind(|| {
            Emboss::new().with_strength(-1.0);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_emboss_invalid_alpha() {
        let result = std::panic::catch_unwind(|| {
            Emboss::new().with_alpha(1.5);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_emboss_apply_constant() {
        // Constant image should produce uniform result
        let mut data = vec![128u8; 9]; // 3x3 constant
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Emboss::new().execute(&mut img);

        // With constant input and blend-based emboss, all pixels should have the same value
        // Center weight is ~1.0, so result ~ 128 (no offset with blend approach)
        let first_val = img.data[0];
        assert!(img.data.iter().all(|&p| p == first_val));
    }

    #[test]
    fn test_emboss_apply_gradient() {
        // Image with a diagonal gradient (creates emboss effect)
        // 0   0   0
        // 0   128 128
        // 0   128 128
        let mut data = vec![0u8, 0u8, 0u8, 0u8, 128u8, 128u8, 0u8, 128u8, 128u8];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Emboss::new().execute(&mut img);

        // The center pixel (index 4) should be enhanced by the emboss effect.
        // With SouthEast direction, the diagonal gradient creates a relief effect.
        // The exact value depends on the blend factor (alpha=0.5) and strength (0.5),
        // but it should definitely differ from the original 128.
        assert_ne!(img.data[4], 128, "center pixel should be modified by emboss effect");
    }

    #[test]
    fn test_emboss_all_directions() {
        // Test that all directions compile
        let mut data = vec![128u8; 9];
        let mut _img = FusableImage::new(&mut data, 3, 3, 1);

        for direction in &[
            EmbossDirection::SouthEast,
            EmbossDirection::SouthWest,
            EmbossDirection::NorthEast,
            EmbossDirection::NorthWest,
        ] {
            let mut data = vec![128u8; 9];
            let mut _img = FusableImage::new(&mut data, 3, 3, 1);
            Emboss::new().with_direction(*direction).execute(&mut _img);
            // Should not panic
        }
    }

    #[test]
    fn test_emboss_rgb() {
        // Test RGB image
        let mut data = vec![
            100u8, 100u8, 100u8, 128u8, 128u8, 128u8, 150u8, 150u8, 150u8,
        ];
        let mut img = FusableImage::new(&mut data, 3, 1, 3);

        Emboss::new().execute(&mut img);

        // Each channel should be processed independently
        // We just verify it doesn't panic
        assert_eq!(img.data.len(), 9);
    }
}
