// RGB Shift transform
//
// Shifts each color channel by a random amount.
//
// OPTIMIZATION: Uses NEON SIMD on ARM64 for 8-16x speedup.

#[cfg(target_arch = "aarch64")]
mod neon;

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

/// RGB Shift transform
///
/// Shifts each color channel (R, G, B) by a specified amount.
/// This transform is channel-aware and only applies to RGB images.
///
/// # Parameters
/// - `r_shift`: Shift for red channel in range [-255, 255]
/// - `g_shift`: Shift for green channel in range [-255, 255]
/// - `b_shift`: Shift for blue channel in range [-255, 255]
///
/// # Notes
/// - This transform is channel-aware and modifies each channel separately
/// - Only applies to RGB images (channels == 3)
/// - For grayscale images, the average shift is applied
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RGBShift {
    pub r_shift: f32,
    pub g_shift: f32,
    pub b_shift: f32,
}

impl RGBShift {
    /// Create a new RGBShift transform
    ///
    /// # Panics
    /// Panics if any shift value is outside [-255, 255]
    pub fn new(r_shift: f32, g_shift: f32, b_shift: f32) -> Self {
        assert!(
            (-255.0..=255.0).contains(&r_shift),
            "r_shift must be in [-255, 255], got {}",
            r_shift
        );
        assert!(
            (-255.0..=255.0).contains(&g_shift),
            "g_shift must be in [-255, 255], got {}",
            g_shift
        );
        assert!(
            (-255.0..=255.0).contains(&b_shift),
            "b_shift must be in [-255, 255], got {}",
            b_shift
        );
        Self {
            r_shift,
            g_shift,
            b_shift,
        }
    }

    /// Get the integer shift values for NEON SIMD
    #[cfg(target_arch = "aarch64")]
    fn get_i8_shifts(&self) -> (i8, i8, i8) {
        // Clamp to i8 range and round to nearest integer
        fn to_i8_clamped(v: f32) -> i8 {
            v.round().clamp(-128.0, 127.0) as i8
        }
        (
            to_i8_clamped(self.r_shift),
            to_i8_clamped(self.g_shift),
            to_i8_clamped(self.b_shift),
        )
    }
}

impl Transform for RGBShift {
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

// Note: RGBShift is NOT a PixelOp because it's channel-aware
// It needs to know which channel it's operating on
impl Executable for RGBShift {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let channels = image.channels;

        if channels == 3 {
            // RGB image - use NEON SIMD for speed
            #[cfg(target_arch = "aarch64")]
            {
                let r_shift = self.r_shift.round() as i16;
                let g_shift = self.g_shift.round() as i16;
                let b_shift = self.b_shift.round() as i16;
                unsafe {
                    neon::rgb_shift_neon(&mut image.data, r_shift, g_shift, b_shift);
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                // Fallback for non-ARM64
                let len = image.data.len();
                let mut i = 0;

                while i + 9 <= len {
                    image.data[i] =
                        (image.data[i] as i16 + self.r_shift.round() as i16).clamp(0, 255) as u8;
                    image.data[i + 1] = (image.data[i + 1] as i16 + self.g_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    image.data[i + 2] = (image.data[i + 2] as i16 + self.b_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    image.data[i + 3] = (image.data[i + 3] as i16 + self.r_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    image.data[i + 4] = (image.data[i + 4] as i16 + self.g_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    image.data[i + 5] = (image.data[i + 5] as i16 + self.b_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    image.data[i + 6] = (image.data[i + 6] as i16 + self.r_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    image.data[i + 7] = (image.data[i + 7] as i16 + self.g_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    image.data[i + 8] = (image.data[i + 8] as i16 + self.b_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    i += 9;
                }

                while i + 3 <= len {
                    image.data[i] =
                        (image.data[i] as i16 + self.r_shift.round() as i16).clamp(0, 255) as u8;
                    image.data[i + 1] = (image.data[i + 1] as i16 + self.g_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    image.data[i + 2] = (image.data[i + 2] as i16 + self.b_shift.round() as i16)
                        .clamp(0, 255) as u8;
                    i += 3;
                }
            }
        } else {
            // Grayscale image - apply average shift
            let avg_shift = (self.r_shift + self.g_shift + self.b_shift) / 3.0;
            for px in image.data.iter_mut() {
                let v = *px as f32 + avg_shift;
                *px = v.clamp(0.0, 255.0) as u8;
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_shift_new() {
        let r = RGBShift::new(10.0, 20.0, 30.0);
        assert_eq!(r.r_shift, 10.0);
        assert_eq!(r.g_shift, 20.0);
        assert_eq!(r.b_shift, 30.0);
    }

    #[test]
    #[should_panic(expected = "r_shift must be in")]
    fn test_rgb_shift_invalid_r() {
        RGBShift::new(300.0, 20.0, 30.0);
    }

    #[test]
    #[should_panic(expected = "g_shift must be in")]
    fn test_rgb_shift_invalid_g() {
        RGBShift::new(10.0, 300.0, 30.0);
    }

    #[test]
    #[should_panic(expected = "b_shift must be in")]
    fn test_rgb_shift_invalid_b() {
        RGBShift::new(10.0, 20.0, 300.0);
    }

    #[test]
    fn test_rgb_shift_execute_rgb() {
        // RGB image: pixel0=(100, 100, 100)
        let mut data = vec![100u8, 100, 100];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let shift = RGBShift::new(10.0, 20.0, 30.0);
        shift.execute(&mut img);

        assert_eq!(img.data, &[110, 120, 130]);
    }

    #[test]
    fn test_rgb_shift_execute_grayscale() {
        let mut data = vec![100u8; 4];
        let mut img = FusableImage::new(&mut data, 2, 2, 1);

        let shift = RGBShift::new(10.0, 20.0, 30.0);
        // Average shift = 20
        shift.execute(&mut img);

        assert_eq!(img.data, &[120u8; 4]);
    }

    #[test]
    fn test_rgb_shift_clamping() {
        let mut data = vec![200u8, 200, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let shift = RGBShift::new(100.0, 100.0, 100.0);
        shift.execute(&mut img);

        // All channels should clamp to 255
        assert_eq!(img.data, &[255, 255, 255]);
    }

    #[test]
    fn test_rgb_shift_negative() {
        let mut data = vec![50u8, 50, 50];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let shift = RGBShift::new(-10.0, -20.0, -30.0);
        shift.execute(&mut img);

        assert_eq!(img.data, &[40, 30, 20]);
    }

    #[test]
    fn test_rgb_shift_alignment() {
        // Test that NEON works with non-multiple-of-8 pixel counts
        let mut data = vec![
            100u8, 100, 100, // Pixel 0
            100, 100, 100, // Pixel 1
            100, 100, 100, // Pixel 2
            100, 100, 100, // Pixel 3
            100, 100, 100, // Pixel 4
            100, 100, 100, // Pixel 5
            100, 100, 100, // Pixel 6
            100, 100, 100, // Pixel 7
            100, 100, 100, // Pixel 8 (extra pixel)
            100, 100, 100, // Pixel 9 (extra pixel)
        ];
        let mut img = FusableImage::new(&mut data, 10, 1, 3);

        let shift = RGBShift::new(10.0, 20.0, 30.0);
        shift.execute(&mut img);

        // All pixels should have the same shift applied
        for i in 0..10 {
            assert_eq!(img.data[i * 3], 110);
            assert_eq!(img.data[i * 3 + 1], 120);
            assert_eq!(img.data[i * 3 + 2], 130);
        }
    }

    #[test]
    fn test_rgb_shift_8pixels_negative() {
        let mut data = vec![100u8; 24];
        let mut img = FusableImage::new(&mut data, 8, 1, 3);
        let shift = RGBShift::new(-10.0, -20.0, -30.0);
        shift.execute(&mut img);
        assert_eq!(img.data[0], 90);
        assert_eq!(img.data[1], 80);
        assert_eq!(img.data[2], 70);
    }

    #[test]
    fn test_rgb_shift_8pixels_underflow() {
        let mut data = vec![6u8, 1, 19].repeat(8);
        let mut img = FusableImage::new(&mut data, 8, 1, 3);
        let shift = RGBShift::new(10.0, -20.0, 30.0);
        shift.execute(&mut img);
        assert_eq!(img.data[0], 16);
        assert_eq!(img.data[1], 0);
        assert_eq!(img.data[2], 49);
    }
}
