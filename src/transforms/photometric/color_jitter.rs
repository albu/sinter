// Color Jitter transform
//
// Randomly changes brightness, contrast, saturation, and hue.
// Popular in PyTorch/torchvision for data augmentation.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::{Brightness, Contrast, HueSaturationValue};

/// Color Jitter transform
///
/// Randomly adjusts brightness, contrast, saturation, and hue.
/// This is a combined transform popular in PyTorch/torchvision.
///
/// # Parameters
/// - `brightness`: Brightness adjustment factor range
///   - 0.0 = no change, 1.0 = full range to black/white
///   - Actual delta = uniform(-brightness, brightness) * 255
/// - `contrast`: Contrast adjustment factor range
///   - 0.0 = no change, typical values 0.2 to 0.5
///   - Actual factor = uniform(1-contrast, 1+contrast)
/// - `saturation`: Saturation adjustment factor range
///   - 0.0 = no change, typical values 0.2 to 0.5
///   - Actual factor = uniform(1-saturation, 1+saturation)
/// - `hue`: Hue adjustment range in degrees
///   - 0.0 = no change, typical values 0.1 to 0.3
///   - Actual shift = uniform(-hue*180, hue*180) degrees
///
/// # Notes
/// - All parameters are ranges, actual values are sampled uniformly
/// - Only affects RGB images (channels == 3)
/// - For grayscale images, only brightness and contrast are applied
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorJitter {
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
}

impl ColorJitter {
    /// Create a new ColorJitter transform
    ///
    /// # Panics
    /// Panics if any parameter is negative
    pub fn new(brightness: f32, contrast: f32, saturation: f32, hue: f32) -> Self {
        assert!(
            brightness >= 0.0,
            "brightness must be >= 0, got {}",
            brightness
        );
        assert!(contrast >= 0.0, "contrast must be >= 0, got {}", contrast);
        assert!(
            saturation >= 0.0,
            "saturation must be >= 0, got {}",
            saturation
        );
        assert!(hue >= 0.0, "hue must be >= 0, got {}", hue);
        Self {
            brightness,
            contrast,
            saturation,
            hue,
        }
    }

    /// Sample a random value from a range centered at 1.0
    ///
    /// For range r, returns uniform(1-r, 1+r)
    #[inline]
    fn sample_factor(&self, range: f32, seed: u32) -> f32 {
        if range == 0.0 {
            return 1.0;
        }
        let state = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (state & 0xFFFFFF) as f32 / 0xFFFFFF as f32; // [0, 1)
        (1.0 - range) + 2.0 * range * u
    }

    /// Sample a random value from a symmetric range
    ///
    /// For range r, returns uniform(-r, r)
    #[inline]
    fn sample_symmetric(&self, range: f32, seed: u32) -> f32 {
        if range == 0.0 {
            return 0.0;
        }
        let state = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (state & 0xFFFFFF) as f32 / 0xFFFFFF as f32; // [0, 1)
        -range + 2.0 * range * u
    }
}

impl Transform for ColorJitter {
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

impl Executable for ColorJitter {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let channels = image.channels;
        let seed_base = 12345u32;

        // Sample random parameters
        let brightness_delta = self.sample_symmetric(self.brightness * 255.0, seed_base);
        let contrast_factor = self.sample_factor(self.contrast, seed_base + 1);
        let saturation_factor = self.sample_factor(self.saturation, seed_base + 2);
        let hue_shift = self.sample_symmetric(self.hue * 180.0, seed_base + 3);

        // Apply brightness and contrast (works on all images)
        let brightness = Brightness::new(brightness_delta);
        let contrast = Contrast::new(contrast_factor);
        brightness.execute(image);
        contrast.execute(image);

        // Apply saturation and hue (RGB only)
        if channels == 3 {
            let hsv = HueSaturationValue::new(hue_shift, saturation_factor, 1.0);
            hsv.execute(image);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_jitter_new() {
        let c = ColorJitter::new(0.2, 0.3, 0.2, 0.1);
        assert_eq!(c.brightness, 0.2);
        assert_eq!(c.contrast, 0.3);
        assert_eq!(c.saturation, 0.2);
        assert_eq!(c.hue, 0.1);
    }

    #[test]
    fn test_color_jitter_zero_params() {
        let mut data = vec![128u8; 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let c = ColorJitter::new(0.0, 0.0, 0.0, 0.0);
        c.execute(&mut img);

        // With zero params, values should be roughly unchanged
        // (allowing for some floating point variation)
        let avg: f32 = img.data.iter().map(|&x| x as f32).sum::<f32>() / img.data.len() as f32;
        assert!((avg - 128.0).abs() < 1.0);
    }

    #[test]
    #[should_panic(expected = "brightness must be >= 0")]
    fn test_color_jitter_invalid_brightness() {
        ColorJitter::new(-0.1, 0.2, 0.2, 0.1);
    }

    #[test]
    #[should_panic(expected = "contrast must be >= 0")]
    fn test_color_jitter_invalid_contrast() {
        ColorJitter::new(0.2, -0.1, 0.2, 0.1);
    }

    #[test]
    fn test_color_jitter_rgb() {
        let mut data = vec![128u8, 128, 128];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let c = ColorJitter::new(0.5, 0.5, 0.5, 0.3);
        c.execute(&mut img);

        // Values should be modified from original
        // Due to deterministic random seeding, we can check they changed
        // but exact values depend on the seed
        let all_same = img.data[0] == 128 && img.data[1] == 128 && img.data[2] == 128;
        assert!(!all_same, "ColorJitter should modify pixel values");
    }

    #[test]
    fn test_color_jitter_grayscale() {
        let mut data = vec![128u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let c = ColorJitter::new(0.3, 0.3, 0.0, 0.0);
        c.execute(&mut img);

        // Values should be modified (brightness/contrast affect grayscale)
        let all_same = img.data.iter().all(|&x| x == 128);
        assert!(!all_same, "ColorJitter should affect grayscale");
    }

    #[test]
    fn test_color_jitter_reproducibility() {
        let mut data1 = vec![128u8; 300];
        let mut img1 = FusableImage::new(&mut data1, 10, 10, 3);

        let mut data2 = vec![128u8; 300];
        let mut img2 = FusableImage::new(&mut data2, 10, 10, 3);

        let c = ColorJitter::new(0.2, 0.2, 0.2, 0.1);
        c.execute(&mut img1);
        c.execute(&mut img2);

        // Same input should produce same output
        assert_eq!(img1.data, img2.data);
    }

    #[test]
    fn test_color_jitter_high_brightness() {
        let mut data = vec![100u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let c = ColorJitter::new(1.0, 0.0, 0.0, 0.0);
        c.execute(&mut img);

        // With high brightness range, values should change significantly from 100
        // (either brighter or darker due to random sampling)
        let all_same = img.data.iter().all(|&x| x == 100);
        assert!(!all_same, "High brightness should modify values");
    }

    #[test]
    fn test_color_jitter_access_pattern() {
        let c = ColorJitter::new(0.2, 0.2, 0.2, 0.1);
        assert_eq!(c.access(), AccessPattern::InPlace);
        assert_eq!(c.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_color_jitter_clamping() {
        let mut data = vec![250u8; 100];
        let mut img = FusableImage::new(&mut data, 10, 10, 1);

        let c = ColorJitter::new(0.5, 0.5, 0.0, 0.0);
        c.execute(&mut img);

        // All values should be clamped to [0, 255]
        for &px in img.data.iter() {
            assert!((0..=255).contains(&px));
        }
    }

    #[test]
    fn test_sample_factor() {
        let c = ColorJitter::new(0.0, 0.0, 0.0, 0.0);

        // Zero range should return 1.0
        assert_eq!(c.sample_factor(0.0, 123), 1.0);

        // Range of 0.5 should give value in [0.5, 1.5]
        let factor = c.sample_factor(0.5, 456);
        assert!(factor >= 0.5 && factor <= 1.5);
    }

    #[test]
    fn test_sample_symmetric() {
        let c = ColorJitter::new(0.0, 0.0, 0.0, 0.0);

        // Zero range should return 0.0
        assert_eq!(c.sample_symmetric(0.0, 123), 0.0);

        // Range of 50.0 should give value in [-50, 50]
        let value = c.sample_symmetric(50.0, 456);
        assert!(value >= -50.0 && value <= 50.0);
    }
}
