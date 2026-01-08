// Fast HSV implementation using proper HSV color space
//
// This implementation:
// - Uses fast RGB→HSV→RGB conversion
// - Applies hue shift, saturation scaling, and value scaling in HSV space
// - Matches Accurate mode results but with optimized scalar code
// - Not as fast as the old broken YIQ matrix, but correct
// - Can be further optimized with SIMD in the future

use crate::core::FusableImage;
use crate::transforms::photometric::hue_saturation_value::HueSaturationValue;

/// Execute using fast HSV implementation
pub fn execute_fast(hsv: &HueSaturationValue, image: &mut FusableImage) {
    let channels = image.channels;

    if channels == 3 {
        execute_fast_rgb(hsv, image);
    } else {
        execute_fast_grayscale(hsv, image);
    }
}

/// Fast RGB implementation using HSV color space
fn execute_fast_rgb(hsv: &HueSaturationValue, image: &mut FusableImage) {
    let pixel_count = image.data.len() / 3;

    // Precompute value scaling factor (Q8.8 fixed-point)
    let val_scale_q8 = (hsv.val_scale * 256.0) as i32;
    let sat_scale_q8 = (hsv.sat_scale * 256.0) as i32;

    for i in 0..pixel_count {
        let idx = i * 3;
        let r = image.data[idx] as i32;
        let g = image.data[idx + 1] as i32;
        let b = image.data[idx + 2] as i32;

        // RGB to HSV conversion (optimized integer arithmetic)
        let (h, mut s, v) = rgb_to_hsv_int(r, g, b);

        // Apply hue shift
        let h = (h as f32 + hsv.hue_shift) % 360.0;
        let h = if h < 0.0 { h + 360.0 } else { h };

        // Apply saturation scaling (Q8.8 fixed-point)
        let s = ((s as i32) * sat_scale_q8 + 128) >> 8;
        let s = s.clamp(0, 255) as u8;

        // Apply value scaling (Q8.8 fixed-point)
        let v = ((v as i32) * val_scale_q8 + 128) >> 8;
        let v = v.clamp(0, 255) as u8;

        // HSV back to RGB conversion (optimized integer arithmetic)
        let (r_out, g_out, b_out) = hsv_to_rgb_int(h, s, v);

        image.data[idx] = r_out;
        image.data[idx + 1] = g_out;
        image.data[idx + 2] = b_out;
    }
}

/// Convert RGB to HSV using integer arithmetic
///
/// Returns:
/// - h: Hue in degrees [0, 360) as f32
/// - s: Saturation [0, 255] (255 = 100% saturation)
/// - v: Value [0, 255] (255 = 100% brightness)
#[inline]
fn rgb_to_hsv_int(r: i32, g: i32, b: i32) -> (f32, u8, u8) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Value (0-255)
    let v = max as u8;

    // Saturation (0-255)
    let s = if max > 0 {
        ((delta << 8) / max).min(255) as u8
    } else {
        0
    };

    // Hue (0-360 degrees)
    let h = if delta == 0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) as f32 / delta as f32)
    } else if max == g {
        60.0 * ((b - r) as f32 / delta as f32 + 2.0)
    } else {
        60.0 * ((r - g) as f32 / delta as f32 + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    (h, s, v)
}

/// Convert HSV to RGB using integer arithmetic
///
/// Input:
/// - h: Hue in degrees [0, 360) as f32
/// - s: Saturation [0, 255] (255 = 100% saturation)
/// - v: Value [0, 255] (255 = 100% brightness)
///
/// Returns RGB values in [0, 255]
#[inline]
fn hsv_to_rgb_int(h: f32, s: u8, v: u8) -> (u8, u8, u8) {
    if s == 0 {
        // Achromatic (gray)
        return (v, v, v);
    }

    // Convert to Q8.8 fixed-point for calculations
    let s_q8 = (s as i64) << 8;
    let v_q8 = (v as i64) << 8;

    // c = v * s (in Q8.8 format, result is Q16.16)
    let c = ((v_q8 * s_q8) >> 16) as i32;

    // x = c * (1 - |(h / 60) % 2 - 1|)
    let h_div_60 = h / 60.0;
    let x_mod = (h_div_60 % 2.0 - 1.0).abs();
    let x = ((c * (255 - (x_mod * 255.0) as i32)) + 128) >> 8;

    // m = v - c
    let m = v_q8 as i32 - c;

    // Determine RGB based on hue sector
    let (r_temp, g_temp, b_temp) = if h < 60.0 {
        (c, x, 0)
    } else if h < 120.0 {
        (x, c, 0)
    } else if h < 180.0 {
        (0, c, x)
    } else if h < 240.0 {
        (0, x, c)
    } else if h < 300.0 {
        (x, 0, c)
    } else {
        (c, 0, x)
    };

    // Add m and convert back to u8
    let r = ((r_temp + m) >> 8).clamp(0, 255) as u8;
    let g = ((g_temp + m) >> 8).clamp(0, 255) as u8;
    let b = ((b_temp + m) >> 8).clamp(0, 255) as u8;

    (r, g, b)
}

/// Fast grayscale implementation (just value scaling)
pub fn execute_fast_grayscale(hsv: &HueSaturationValue, image: &mut FusableImage) {
    let val_factor_q8 = (hsv.val_scale * 256.0) as i32;

    for px in image.data.iter_mut() {
        let v = ((*px as i32) * val_factor_q8 + 128) >> 8;
        *px = v.clamp(0, 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hue_rotation_red_orange() {
        // Red (255, 0, 0) rotated by 45° should become orange/yellow
        let mut data = vec![255u8, 0, 0];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);
        let hsv = HueSaturationValue {
            hue_shift: 45.0,
            sat_scale: 1.0,
            val_scale: 1.0,
        };
        execute_fast(&hsv, &mut img);

        // Result should be orange-ish
        assert!(img.data[0] > 200); // R still high
        assert!(img.data[1] > 100); // G increased significantly
        assert_eq!(img.data[2], 0);  // B still 0
    }

    #[test]
    fn test_saturation_identity() {
        // sat_scale = 1.0 should not change colors
        let mut data = vec![100u8, 150, 200, 50, 100, 150];
        let mut img = FusableImage::new(&mut data, 2, 1, 3);
        let hsv = HueSaturationValue {
            hue_shift: 0.0,
            sat_scale: 1.0,
            val_scale: 1.0,
        };
        execute_fast(&hsv, &mut img);

        assert_eq!(img.data, &[100, 150, 200, 50, 100, 150]);
    }

    #[test]
    fn test_saturation_desaturate() {
        // sat_scale = 0.0 should grayscale
        let mut data = vec![0u8, 100, 255, 50, 150, 200];
        let mut img = FusableImage::new(&mut data, 2, 1, 3);
        let hsv = HueSaturationValue {
            hue_shift: 0.0,
            sat_scale: 0.0,
            val_scale: 1.0,
        };
        execute_fast(&hsv, &mut img);

        // All channels should equal the value (brightness)
        assert_eq!(img.data[0], img.data[1]);
        assert_eq!(img.data[1], img.data[2]);
        assert_eq!(img.data[3], img.data[4]);
        assert_eq!(img.data[4], img.data[5]);
    }

    #[test]
    fn test_value_identity() {
        // val_scale = 1.0 should not change brightness
        let mut data = vec![100u8, 150, 200, 50, 100, 150];
        let mut img = FusableImage::new(&mut data, 2, 1, 3);
        let hsv = HueSaturationValue {
            hue_shift: 0.0,
            sat_scale: 1.0,
            val_scale: 1.0,
        };
        execute_fast(&hsv, &mut img);

        assert_eq!(img.data, &[100, 150, 200, 50, 100, 150]);
    }

    #[test]
    fn test_value_brighten() {
        // val_scale = 2.0 should double brightness (with clamping)
        let mut data = vec![100u8, 150, 50];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);
        let hsv = HueSaturationValue {
            hue_shift: 0.0,
            sat_scale: 1.0,
            val_scale: 2.0,
        };
        execute_fast(&hsv, &mut img);

        // Values should be higher, clamped at 255
        assert!(img.data[0] > 100);
        assert!(img.data[1] > 150);
        assert!(img.data[2] > 50);
    }

    #[test]
    fn test_grayscale_identity() {
        let mut data = vec![100u8, 150, 200];
        let mut img = FusableImage::new(&mut data, 3, 1, 1);
        let hsv = HueSaturationValue {
            hue_shift: 0.0,
            sat_scale: 1.0,
            val_scale: 1.0,
        };
        execute_fast(&hsv, &mut img);

        assert_eq!(img.data, &[100, 150, 200]);
    }

    #[test]
    fn test_grayscale_value() {
        let mut data = vec![100u8, 150, 200];
        let mut img = FusableImage::new(&mut data, 3, 1, 1);
        let hsv = HueSaturationValue {
            hue_shift: 0.0,
            sat_scale: 1.0,
            val_scale: 0.5,
        };
        execute_fast(&hsv, &mut img);

        assert_eq!(img.data[0], 50);
        assert_eq!(img.data[1], 75);
        assert_eq!(img.data[2], 100);
    }
}
