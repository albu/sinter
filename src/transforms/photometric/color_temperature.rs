// Color Temperature transform
//
// Adjusts the color temperature of an image, shifting between warm (yellow/orange)
// and cool (blue) tones.
//
// This is a 3x3 RGB matrix operation that can be fused with other MatrixOp transforms.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::runtime::matrix::{apply_matrix, MatrixOp};

/// ColorTemperature transform
///
/// Adjusts color temperature by shifting between warm (positive values) and cool (negative values).
///
/// # Parameters
/// - `temperature`: Color temperature adjustment in range [-100, 100]
///   - Positive values: warmer (more yellow/orange)
///   - Negative values: cooler (more blue)
///   - 0: no change
///
/// # Algorithm
/// Uses a 3x3 RGB matrix that adjusts color channels based on temperature:
/// - Warm (positive): Boosts red/green, reduces blue
/// - Cool (negative): Boosts blue, reduces red/green
///
/// # Example
/// ```text
/// Temperature(50):  Makes image warmer (yellowish)
/// Temperature(-50): Makes image cooler (bluish)
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorTemperature {
    /// Temperature adjustment [-100, 100]
    pub temperature: f32,
}

impl ColorTemperature {
    /// Create a new ColorTemperature transform
    ///
    /// # Arguments
    /// * `temperature` - Temperature adjustment in range [-100, 100]
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl Default for ColorTemperature {
    fn default() -> Self {
        Self { temperature: 0.0 }
    }
}

impl Transform for ColorTemperature {
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

impl MatrixOp for ColorTemperature {
    fn get_matrix(&self) -> [[f32; 3]; 3] {
        // Temperature adjustment matrix
        // Positive = warm (boost red/green, reduce blue)
        // Negative = cool (boost blue, reduce red/green)

        let t = self.temperature / 100.0; // Normalize to [-1, 1]

        if t >= 0.0 {
            // Warm: boost R and G, reduce B
            [
                [1.0 + 0.1 * t, 0.0, 0.0], // R' = R * (1 + 0.1t)
                [0.0, 1.0 + 0.1 * t, 0.0], // G' = G * (1 + 0.1t)
                [0.0, 0.0, 1.0 - 0.2 * t], // B' = B * (1 - 0.2t)
            ]
        } else {
            // Cool: boost B, reduce R and G
            let t = t.abs(); // Make positive for calculations
            [
                [1.0 - 0.15 * t, 0.0, 0.0], // R' = R * (1 - 0.15t)
                [0.0, 1.0 - 0.15 * t, 0.0], // G' = G * (1 - 0.15t)
                [0.0, 0.0, 1.0 + 0.2 * t],  // B' = B * (1 + 0.2t)
            ]
        }
    }
}

impl Executable for ColorTemperature {
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
    fn test_color_temperature_new() {
        let t = ColorTemperature::new(50.0);
        assert_eq!(t.temperature, 50.0);
    }

    #[test]
    fn test_color_temperature_default() {
        let t = ColorTemperature::default();
        assert_eq!(t.temperature, 0.0);
    }

    #[test]
    fn test_color_temperature_matrix_identity() {
        let t = ColorTemperature::new(0.0);
        let matrix = t.get_matrix();

        // At temperature 0, should be close to identity
        let eps = 0.01;
        assert!((matrix[0][0] - 1.0).abs() < eps);
        assert!((matrix[1][1] - 1.0).abs() < eps);
        assert!((matrix[2][2] - 1.0).abs() < eps);
        assert!(matrix[0][1].abs() < eps);
        assert!(matrix[0][2].abs() < eps);
        assert!(matrix[1][0].abs() < eps);
        assert!(matrix[1][2].abs() < eps);
        assert!(matrix[2][0].abs() < eps);
        assert!(matrix[2][1].abs() < eps);
    }

    #[test]
    fn test_color_temperature_matrix_warm() {
        let t = ColorTemperature::new(50.0);
        let matrix = t.get_matrix();

        // Warm: R and G should be boosted, B reduced
        assert!(matrix[0][0] > 1.0); // R boosted
        assert!(matrix[1][1] > 1.0); // G boosted
        assert!(matrix[2][2] < 1.0); // B reduced
    }

    #[test]
    fn test_color_temperature_matrix_cool() {
        let t = ColorTemperature::new(-50.0);
        let matrix = t.get_matrix();

        // Cool: B should be boosted, R and G reduced
        assert!(matrix[0][0] < 1.0); // R reduced
        assert!(matrix[1][1] < 1.0); // G reduced
        assert!(matrix[2][2] > 1.0); // B boosted
    }

    #[test]
    fn test_color_temperature_execute_warm() {
        // White image with warm temperature should become yellowish
        let mut data = vec![255u8, 255, 255];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorTemperature::new(50.0).execute(&mut img);

        // Red and green should be higher than blue
        assert!(img.data[0] > img.data[2]); // R > B
        assert!(img.data[1] > img.data[2]); // G > B
    }

    #[test]
    fn test_color_temperature_execute_cool() {
        // White image with cool temperature should become bluish
        let mut data = vec![255u8, 255, 255];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ColorTemperature::new(-50.0).execute(&mut img);

        // Blue should be highest
        assert!(img.data[2] >= img.data[0]); // B >= R
        assert!(img.data[2] >= img.data[1]); // B >= G
    }

    #[test]
    fn test_color_temperature_grayscale_passthrough() {
        // Grayscale should be unchanged
        let mut data = vec![128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 1);

        ColorTemperature::new(50.0).execute(&mut img);

        assert_eq!(img.data[0], 128);
    }

    #[test]
    fn test_color_temperature_access_pattern() {
        let t = ColorTemperature::new(50.0);
        assert_eq!(t.access(), AccessPattern::InPlace);
        assert_eq!(t.shape_effect(), ShapeEffect::Preserve);
    }
}
