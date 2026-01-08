// Tests for HueSaturationValue transform

use crate::core::{Executable, FusableImage};
use crate::transforms::photometric::hue_saturation_value::{
    rust_impl, HueSaturationValue,
};

#[test]
fn test_hsv_new() {
    let h = HueSaturationValue::new(30.0, 1.2, 1.1);
    assert_eq!(h.hue_shift, 30.0);
    assert_eq!(h.sat_scale, 1.2);
    assert_eq!(h.val_scale, 1.1);
}

#[test]
#[should_panic(expected = "hue_shift must be in")]
fn test_hsv_invalid_hue() {
    HueSaturationValue::new(200.0, 1.0, 1.0);
}

#[test]
#[should_panic(expected = "sat_scale must be >= 0")]
fn test_hsv_invalid_sat() {
    HueSaturationValue::new(0.0, -1.0, 1.0);
}

#[test]
#[should_panic(expected = "val_scale must be >= 0")]
fn test_hsv_invalid_val() {
    HueSaturationValue::new(0.0, 1.0, -1.0);
}

#[test]
fn test_rgb_to_hsv_int_red() {
    // Pure red (255, 0, 0) should be H=0, S=255, V=255
    let (h, s, v) = rust_impl::rgb_to_hsv_int(255, 0, 0);
    assert!((h - 0.0).abs() < 1.0);
    assert_eq!(s, 255);
    assert_eq!(v, 255);
}

#[test]
fn test_rgb_to_hsv_int_green() {
    // Pure green (0, 255, 0) should be H=120, S=255, V=255
    let (h, s, v) = rust_impl::rgb_to_hsv_int(0, 255, 0);
    assert!((h - 120.0).abs() < 1.0);
    assert_eq!(s, 255);
    assert_eq!(v, 255);
}

#[test]
fn test_rgb_to_hsv_int_blue() {
    // Pure blue (0, 0, 255) should be H=240, S=255, V=255
    let (h, s, v) = rust_impl::rgb_to_hsv_int(0, 0, 255);
    assert!((h - 240.0).abs() < 1.0);
    assert_eq!(s, 255);
    assert_eq!(v, 255);
}

#[test]
fn test_hsv_to_rgb_int_roundtrip() {
    // Test roundtrip conversion with some tolerance
    let r_orig = 128;
    let g_orig = 180;
    let b_orig = 90;

    let (h, s, v) = rust_impl::rgb_to_hsv_int(r_orig, g_orig, b_orig);
    let (r, g, b) = rust_impl::hsv_to_rgb_int(h, s, v);

    // Allow small difference due to integer arithmetic rounding
    assert!((r as i32 - r_orig).abs() <= 2);
    assert!((g as i32 - g_orig).abs() <= 2);
    assert!((b as i32 - b_orig).abs() <= 2);
}

#[test]
fn test_hsv_execute_hue_shift() {
    // Pure red becomes yellow with 60 degree hue shift
    let mut data = vec![255u8, 0, 0];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    let hsv = HueSaturationValue::new(60.0, 1.0, 1.0);
    hsv.execute(&mut img);

    // Red -> Yellow should be approximately (255, 255, 0)
    // Allow some tolerance due to integer arithmetic
    assert!(img.data[0] > 250); // R
    assert!(img.data[1] > 250); // G
    assert!(img.data[2] < 10); // B
}

#[test]
fn test_hsv_execute_grayscale() {
    let mut data = vec![128u8; 4];
    let mut img = FusableImage::new(&mut data, 2, 2, 1);

    // Value scale 2x on grayscale should double brightness (clamped)
    let hsv = HueSaturationValue::new(0.0, 1.0, 2.0);
    hsv.execute(&mut img);

    // 128 * 2 = 256 -> clamped to 255
    for &px in img.data.iter() {
        assert_eq!(px, 255);
    }
}

#[test]
fn test_hsv_execute_saturation() {
    // Mid-gray with some color
    let mut data = vec![150u8, 100, 100];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    // Increase saturation
    let hsv = HueSaturationValue::new(0.0, 2.0, 1.0);
    hsv.execute(&mut img);

    // Red channel should be higher, green/blue lower (more saturated)
    assert!(img.data[0] >= 150);
    assert!(img.data[1] <= 100);
    assert!(img.data[2] <= 100);
}

#[test]
fn test_hsv_hue_wraparound() {
    // Blue with hue shift of 120 should wrap correctly
    let mut data = vec![0u8, 0, 255];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    let hsv = HueSaturationValue::new(120.0, 1.0, 1.0);
    hsv.execute(&mut img);

    // Blue (H=240) + 120 = 360 -> wraps to 0 (Red)
    assert!(img.data[0] > 200); // R should be high
    assert!(img.data[1] < 50); // G should be low
    assert!(img.data[2] < 50); // B should be low
}

#[test]
fn test_hsv_zero_saturation() {
    // Color with zero saturation should become gray
    let mut data = vec![255u8, 0, 0];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    let hsv = HueSaturationValue::new(0.0, 0.0, 1.0);
    hsv.execute(&mut img);

    // All channels should be equal (gray)
    assert_eq!(img.data[0], img.data[1]);
    assert_eq!(img.data[1], img.data[2]);
}

#[test]
fn test_hsv_to_rgb_int_grayscale() {
    // Zero saturation should produce gray
    let (r, g, b) = rust_impl::hsv_to_rgb_int(0.0, 0, 128);
    assert_eq!(r, 128);
    assert_eq!(g, 128);
    assert_eq!(b, 128);
}

#[test]
fn test_rgb_to_hsv_int_gray() {
    // Gray should have S=0
    let (_h, s, v) = rust_impl::rgb_to_hsv_int(128, 128, 128);
    assert_eq!(s, 0);
    assert_eq!(v, 128);
}

#[test]
fn test_hsv_fast_simd_batch() {
    // 32 pixels (2 blocks of 16) + 1 pixel remainder = 33 pixels.
    // Use Red color.
    let mut data = vec![0u8; 33 * 3];
    for i in 0..33 {
        data[i * 3] = 255;
        data[i * 3 + 1] = 0;
        data[i * 3 + 2] = 0;
    }
    let mut img = FusableImage::new(&mut data, 33, 1, 3);

    // Shift Hue by 60 -> Yellow (255, 255, 0)
    let hsv = HueSaturationValue::new(60.0, 1.0, 1.0);
    hsv.execute(&mut img);

    for i in 0..33 {
        let r = img.data[i * 3];
        let g = img.data[i * 3 + 1];
        let b = img.data[i * 3 + 2];

        // Should be approximately yellow
        assert!(r > 250, "Pixel {} R too low: {}", i, r);
        assert!(g > 250, "Pixel {} G too low: {}", i, g);
        assert!(img.data[i * 3 + 2] < 10, "Pixel {} B too high: {}", i, b);
    }
}
