// Rust implementation of HueSaturationValue transform
//
// Pure Rust scalar implementation using integer arithmetic.
//
// Note: Reference implementation used for test verification and fallback.
#![allow(dead_code)]

use crate::core::FusableImage;
use crate::transforms::photometric::hue_saturation_value::HueSaturationValue;

/// Execute using pure Rust implementation
pub(super) fn execute_rust(hsv: &HueSaturationValue, image: &mut FusableImage) {
    let channels = image.channels;

    if channels == 3 {
        // RGB image - apply HSV transform
        let pixel_count = image.data.len() / 3;

        // Pre-compute saturation and value scaling in Q8.8 fixed-point
        let sat_scale_q8 = (hsv.sat_scale * 256.0) as i32;
        let val_scale_q8 = (hsv.val_scale * 256.0) as i32;

        for i in 0..pixel_count {
            let idx = i * 3;
            let r = image.data[idx] as i32;
            let g = image.data[idx + 1] as i32;
            let b = image.data[idx + 2] as i32;

            // RGB to HSV conversion using integer arithmetic
            let (h, s, v) = rgb_to_hsv_int(r, g, b);

            // Apply hue shift
            let h = (h as f32 + hsv.hue_shift) % 360.0;
            let h = if h < 0.0 { h + 360.0 } else { h };

            // Apply saturation scaling (Q8.8 fixed-point)
            let s = ((s as i32) * sat_scale_q8 + 128) >> 8;
            let s = s.clamp(0, 255) as u8;

            // Apply value scaling (Q8.8 fixed-point)
            let v = ((v as i32) * val_scale_q8 + 128) >> 8;
            let v = v.clamp(0, 255) as u8;

            // HSV back to RGB using integer arithmetic
            let (r_out, g_out, b_out) = hsv_to_rgb_int(h, s, v);

            image.data[idx] = r_out;
            image.data[idx + 1] = g_out;
            image.data[idx + 2] = b_out;
        }
    } else {
        // Grayscale image - apply value scaling as brightness adjustment
        for px in image.data.iter_mut() {
            let v = ((*px as i32) * (hsv.val_scale * 256.0) as i32 + 128) >> 8;
            *px = v.clamp(0, 255) as u8;
        }
    }
}

/// Convert RGB to HSV using integer arithmetic
///
/// Returns:
/// - h: Hue in degrees [0, 360) as f32 (for easier modulo operation)
/// - s: Saturation [0, 255] (255 = 100% saturation)
/// - v: Value [0, 255] (255 = 100% brightness)
pub(super) fn rgb_to_hsv_int(r: i32, g: i32, b: i32) -> (f32, u8, u8) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Value (0-255)
    let v = max as u8;

    // Saturation (0-255)
    let s = if max > 0 {
        // delta * 255 / max for Q8.8 format
        ((delta << 8) / max).min(255) as u8
    } else {
        0
    };

    // Hue (0-360 degrees)
    let h = if delta == 0 {
        0.0
    } else if max == r {
        // 60 * ((g - b) / delta) % 6
        60.0 * ((g - b) as f32 / delta as f32)
    } else if max == g {
        // 60 * ((b - r) / delta + 2)
        60.0 * ((b - r) as f32 / delta as f32 + 2.0)
    } else {
        // 60 * ((r - g) / delta + 4)
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
pub(super) fn hsv_to_rgb_int(h: f32, s: u8, v: u8) -> (u8, u8, u8) {
    if s == 0 {
        // Achromatic (gray)
        return (v, v, v);
    }

    // Convert to Q8.8 fixed-point for calculations
    let s_q8 = (s as i64) << 8; // s * 256
    let v_q8 = (v as i64) << 8; // v * 256

    // c = v * s (in Q8.8 format, result is Q16.16)
    // Use i64 to avoid overflow when s=255 and v=255 (65280 * 65280 exceeds i32)
    let c = ((v_q8 * s_q8) >> 16) as i32; // Back to Q8.8

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
    // (m >> 8) converts Q8.8 to integer
    let r = ((r_temp + m) >> 8).clamp(0, 255) as u8;
    let g = ((g_temp + m) >> 8).clamp(0, 255) as u8;
    let b = ((b_temp + m) >> 8).clamp(0, 255) as u8;

    (r, g, b)
}
