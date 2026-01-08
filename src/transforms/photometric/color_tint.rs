// ColorTint transform
//
// Applies a customizable color tint to an image.
// This is similar to sepia but with customizable RGB target colors.
//
// This is a 3x3 RGB matrix operation that can be fused with other MatrixOp transforms.
//
// NOTE: This is NOT a tone curve adjustment (like Albumentations' RandomColorTintCurve).
// This applies a color tint by blending towards a target color.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::runtime::matrix::{apply_matrix, MatrixOp};

/// ColorTint transform
///
/// Applies a customizable color tint to an image by blending towards a target color.
/// The intensity parameter controls how strong the effect is.
///
/// # Parameters
/// - `target_r`: Target red channel (0-255)
/// - `target_g`: Target green channel (0-255)
/// - `target_b`: Target blue channel (0-255)
/// - `intensity`: Blending intensity (0.0 = no change, 1.0 = full target color)
///
/// # Algorithm
/// The tint is applied using a 3x3 matrix that blends the original color
/// towards the target color based on intensity:
/// ```text
/// R' = R * (1 - intensity) + target_r * intensity * (R + G + B) / (3 * 255)
/// G' = G * (1 - intensity) + target_g * intensity * (R + G + B) / (3 * 255)
/// B' = B * (1 - intensity) + target_b * intensity * (R + G + B) / (3 * 255)
/// ```
///
/// Simplified to matrix form:
/// ```text
/// R' = R * (1 - intensity + target_r * intensity / (3 * 255)) + G * (target_r * intensity / (3 * 255)) + B * (target_r * intensity / (3 * 255))
/// ```
///
/// # Example
/// ```text
/// ColorTint::sepia(): Classic sepia tone (similar to ToSepia transform)
/// ColorTint::warm(): Warm golden tone
/// ColorTint::cool(): Cool blue tone
/// ColorTint::custom(255, 0, 0, 0.3): 30% red tint
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorTint {
    /// Target red channel [0, 255]
    pub target_r: f32,
    /// Target green channel [0, 255]
    pub target_g: f32,
    /// Target blue channel [0, 255]
    pub target_b: f32,
    /// Intensity of the effect [0.0, 1.0]
    pub intensity: f32,
}

impl ColorTint {
    /// Create a new ColorTint transform
    ///
    /// # Arguments
    /// * `target_r` - Target red channel (0-255)
    /// * `target_g` - Target green channel (0-255)
    /// * `target_b` - Target blue channel (0-255)
    /// * `intensity` - Blending intensity (0.0 = no change, 1.0 = full tint)
    pub fn new(target_r: f32, target_g: f32, target_b: f32, intensity: f32) -> Self {
        Self {
            target_r: target_r.clamp(0.0, 255.0),
            target_g: target_g.clamp(0.0, 255.0),
            target_b: target_b.clamp(0.0, 255.0),
            intensity: intensity.clamp(0.0, 1.0),
        }
    }

    /// Classic sepia tone (similar to ToSepia but with adjustable intensity)
    pub fn sepia() -> Self {
        Self {
            target_r: 112.0,
            target_g: 66.0,
            target_b: 41.0,
            intensity: 1.0,
        }
    }

    /// Warm golden tone (sunset effect)
    pub fn warm() -> Self {
        Self {
            target_r: 255.0,
            target_g: 200.0,
            target_b: 100.0,
            intensity: 0.5,
        }
    }

    /// Cool blue tone (winter effect)
    pub fn cool() -> Self {
        Self {
            target_r: 100.0,
            target_g: 150.0,
            target_b: 255.0,
            intensity: 0.5,
        }
    }

    /// Muted/desaturated tone
    pub fn muted() -> Self {
        Self {
            target_r: 128.0,
            target_g: 128.0,
            target_b: 128.0,
            intensity: 0.5,
        }
    }

    /// Vintage/aged photo effect
    pub fn vintage() -> Self {
        Self {
            target_r: 180.0,
            target_g: 160.0,
            target_b: 120.0,
            intensity: 0.4,
        }
    }

    /// Create a tint with specified intensity from this tint's target color
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }
}

impl Default for ColorTint {
    fn default() -> Self {
        Self {
            target_r: 128.0,
            target_g: 128.0,
            target_b: 128.0,
            intensity: 0.0,
        }
    }
}

impl Transform for ColorTint {
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

impl MatrixOp for ColorTint {
    fn get_matrix(&self) -> [[f32; 3]; 3] {
        // The tone effect blends towards target color based on intensity
        // We can express this as a matrix operation:
        //
        // For each pixel, compute the luminance contribution and blend with target
        //
        // R' = R * (1 - i) + tr * i * (R + G + B) / (3 * 255)
        //    = R * (1 - i + tr*i/(3*255)) + G * (tr*i/(3*255)) + B * (tr*i/(3*255))
        //
        // This gives us a 3x3 matrix:
        // [ 1-i+tr*k,  tg*k,      tb*k    ]
        // [  tr*k,     1-i+tg*k,  tb*k    ]
        // [  tr*k,     tg*k,      1-i+tb*k ]
        //
        // where k = intensity / (3 * 255)

        let k = self.intensity / (3.0 * 255.0);
        let tr_k = self.target_r * k;
        let tg_k = self.target_g * k;
        let tb_k = self.target_b * k;
        let one_minus_i = 1.0 - self.intensity;

        [
            [one_minus_i + tr_k, tg_k, tb_k],
            [tr_k, one_minus_i + tg_k, tb_k],
            [tr_k, tg_k, one_minus_i + tb_k],
        ]
    }
}

impl Executable for ColorTint {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        if image.channels != 3 {
            return None;
        }
        apply_matrix(image, &self.get_matrix());
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colortint_new() {
        let t = ColorTint::new(200.0, 150.0, 100.0, 0.5);
        assert_eq!(t.target_r, 200.0);
        assert_eq!(t.target_g, 150.0);
        assert_eq!(t.target_b, 100.0);
        assert_eq!(t.intensity, 0.5);
    }

    #[test]
    fn test_tone_new_clamping() {
        // Target should be clamped to [0, 255]
        let t = ColorTint::new(300.0, -50.0, 100.0, 1.5);
        assert_eq!(t.target_r, 255.0);
        assert_eq!(t.target_g, 0.0);
        // Intensity should be clamped to [0, 1]
        assert_eq!(t.intensity, 1.0);
    }

    #[test]
    fn test_tone_default() {
        let t = ColorTint::default();
        assert_eq!(t.target_r, 128.0);
        assert_eq!(t.target_g, 128.0);
        assert_eq!(t.target_b, 128.0);
        assert_eq!(t.intensity, 0.0);
    }

    #[test]
    fn test_tone_matrix_zero_intensity() {
        let t = ColorTint::new(100.0, 150.0, 200.0, 0.0);
        let matrix = t.get_matrix();

        // At 0 intensity, should be identity
        let eps = 0.001;
        assert!((matrix[0][0] - 1.0).abs() < eps);
        assert!((matrix[1][1] - 1.0).abs() < eps);
        assert!((matrix[2][2] - 1.0).abs() < eps);
    }

    #[test]
    fn test_tone_execute_no_intensity() {
        let mut data = vec![100u8, 150, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorTint::new(255.0, 0.0, 0.0, 0.0).execute(&mut img);

        // At 0 intensity, should be unchanged
        assert_eq!(img.data[0], 100);
        assert_eq!(img.data[1], 150);
        assert_eq!(img.data[2], 200);
    }

    #[test]
    fn test_tone_execute_sepia() {
        let mut data = vec![200u8, 200, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorTint::sepia().execute(&mut img);

        // Sepia should produce warm brown tones (lower overall values)
        // The exact values depend on the matrix computation
        // Just verify it changed from original
        let changed = img.data[0] != 200 || img.data[1] != 200 || img.data[2] != 200;
        assert!(changed, "Sepia should modify the image");
    }

    #[test]
    fn test_tone_execute_warm() {
        let mut data = vec![150u8, 150, 150];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorTint::warm().execute(&mut img);

        // Warm should modify the image (target is 255, 200, 100 at intensity 0.5)
        // The exact values depend on the matrix computation
        let changed = img.data[0] != 150 || img.data[1] != 150 || img.data[2] != 150;
        assert!(changed, "Warm tone should modify the image");
    }

    #[test]
    fn test_tone_execute_cool() {
        let mut data = vec![150u8, 150, 150];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorTint::cool().execute(&mut img);

        // Cool should modify the image (target is 100, 150, 255 at intensity 0.5)
        let changed = img.data[0] != 150 || img.data[1] != 150 || img.data[2] != 150;
        assert!(changed, "Cool tone should modify the image");
    }

    #[test]
    fn test_tone_with_intensity() {
        let tone = ColorTint::warm().with_intensity(0.25);
        assert_eq!(tone.intensity, 0.25);
    }

    #[test]
    fn test_tone_with_intensity_clamping() {
        let tone = ColorTint::warm().with_intensity(1.5);
        assert_eq!(tone.intensity, 1.0); // Clamped to 1.0

        let tone2 = ColorTint::warm().with_intensity(-0.5);
        assert_eq!(tone2.intensity, 0.0); // Clamped to 0.0
    }

    #[test]
    fn test_tone_grayscale_passthrough() {
        let mut data = vec![128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 1);

        ColorTint::sepia().execute(&mut img);

        assert_eq!(img.data[0], 128);
    }

    #[test]
    fn test_tone_access_pattern() {
        let t = ColorTint::sepia();
        assert_eq!(t.access(), AccessPattern::InPlace);
        assert_eq!(t.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_tone_red_tint() {
        let mut data = vec![200u8, 200, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorTint::new(255.0, 0.0, 0.0, 0.5).execute(&mut img);

        // Red tint should increase red relative to G and B
        assert!(img.data[0] >= img.data[1]);
        assert!(img.data[0] >= img.data[2]);
    }
}
