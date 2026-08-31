// Emboss transform
//
// Applies an embossing convolution kernel to create a 3D relief effect.

use super::convolve::convolve_3x3;
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

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
        self.execute_rust(image);
        None
    }
}

impl Emboss {
    /// Pure Rust implementation (all platforms; no OpenCV dependency)
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

        super::convolve_2d::apply_emboss(image, &kernel_i32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::FusableImage;

    fn reference_emboss(data: &[u8], w: usize, h: usize, k: &[i32; 9]) -> Vec<u8> {
        let mut out = vec![0u8; data.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut sum = 0i32;
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let xx = (x as i32 + kx as i32 - 1).clamp(0, w as i32 - 1) as usize;
                            let yy = (y as i32 + ky as i32 - 1).clamp(0, h as i32 - 1) as usize;
                            sum += data[(yy * w + xx) * 3 + c] as i32 * k[ky * 3 + kx];
                        }
                    }
                    out[(y * w + x) * 3 + c] = (sum / 256).clamp(0, 255) as u8;
                }
            }
        }
        out
    }

    #[test]
    fn test_emboss_neon_matches_reference() {
        let w = 32;
        let h = 32;
        let mut data: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        let input = data.clone();

        // NE kernel with alpha=0.5, strength=0.5 (Q8 fixed point)
        let s = 0.5f32;
        let a = 0.5f32;
        let effect = [
            1.0 + s, s, 0.0, s, 1.0, -s, 0.0, -s, -1.0 - s,
        ];
        let identity = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let kernel_f32: [f32; 9] = std::array::from_fn(|i| (1.0 - a) * identity[i] + a * effect[i]);
        let kernel_i32: [i32; 9] = std::array::from_fn(|i| (kernel_f32[i] * 256.0) as i32);

        let expected = reference_emboss(&input, w, h, &kernel_i32);

        let mut img = FusableImage::new(&mut data, w, h, 3);
        super::super::convolve_2d::apply_emboss(&mut img, &kernel_i32);

        let mut mismatches = 0usize;
        let mut max_diff = 0i32;
        for i in 0..data.len() {
            let diff = (data[i] as i32 - expected[i] as i32).abs();
            if diff > 0 {
                mismatches += 1;
                max_diff = max_diff.max(diff);
                if mismatches <= 10 {
                    let px = i / 3;
                    eprintln!(
                        "  byte {} (x={}, y={}, c={}): got={} expected={}",
                        i,
                        px % w,
                        px / w,
                        i % 3,
                        data[i],
                        expected[i]
                    );
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "emboss NEON vs reference: {} mismatches, max_diff={}",
            mismatches,
            max_diff
        );
    }

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
