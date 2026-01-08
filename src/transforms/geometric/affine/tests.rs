// Tests for Affine transform

use super::interpolation::bilinear_interpolate;
use super::{Affine, AffineParams};
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

#[test]
fn test_affine_new() {
    let params = AffineParams::default();
    let a = Affine::new(params);
    assert_eq!(a.params.scale, (1.0, 1.0));
    assert_eq!(a.params.rotate, 0.0);
}

#[test]
fn test_affine_params_default() {
    let p = AffineParams::default();
    assert_eq!(p.scale, (1.0, 1.0));
    assert_eq!(p.rotate, 0.0);
    assert_eq!(p.translate, (0.0, 0.0));
    assert_eq!(p.shear, (0.0, 0.0));
}

#[test]
fn test_affine_identity() {
    let mut data = vec![128u8; 9];
    let mut img = FusableImage::new(&mut data, 3, 3, 1);

    let params = AffineParams::default();
    let a = Affine::new(params);
    let result = a.execute(&mut img);

    assert!(result.is_some());
    let out_img = result.unwrap();
    // Identity should preserve image (with some interpolation differences)
    assert_eq!(out_img.width, 3);
    assert_eq!(out_img.height, 3);
    // Center pixel should be preserved
    assert_eq!(out_img.data[4], 128);
}

#[test]
fn test_affine_scale_up() {
    let mut data = vec![255u8; 9];
    let mut img = FusableImage::new(&mut data, 3, 3, 1);

    let mut params = AffineParams::default();
    params.scale = (2.0, 2.0);
    let a = Affine::with_output_size(params, 6, 6);
    let result = a.execute(&mut img);

    assert!(result.is_some());
    let out_img = result.unwrap();
    assert_eq!(out_img.width, 6);
    assert_eq!(out_img.height, 6);
    // Center region should have white pixels from source
    assert_eq!(out_img.data[0], 255); // Top-left corner
    assert_eq!(out_img.data[1], 255);
    assert_eq!(out_img.data[6], 255); // Second row
}

#[test]
fn test_affine_access_pattern() {
    let params = AffineParams::default();
    let a = Affine::new(params);
    assert_eq!(a.access(), AccessPattern::OutOfPlace);
    assert_eq!(a.shape_effect(), ShapeEffect::Resize);
}

#[test]
fn test_affine_rgb() {
    let mut data = vec![
        255u8, 0, 0, // Red
        0, 255u8, 0, // Green
        0, 0, 255u8, // Blue
    ];
    let mut img = FusableImage::new(&mut data, 3, 1, 3);

    let params = AffineParams::default();
    let a = Affine::new(params);
    let result = a.execute(&mut img);

    assert!(result.is_some());
    let out_img = result.unwrap();
    assert_eq!(out_img.channels, 3);
    // Check colors are preserved (approximately)
    assert!(out_img.data[0] > 200); // R channel high
    assert!(out_img.data[4] > 200); // G channel high
    assert!(out_img.data[8] > 200); // B channel high
}

#[test]
fn test_affine_output_size() {
    let mut data = vec![128u8; 9];
    let mut img = FusableImage::new(&mut data, 3, 3, 1);

    let params = AffineParams::default();
    let a = Affine::with_output_size(params, 5, 5);
    let result = a.execute(&mut img);

    assert!(result.is_some());
    let out_img = result.unwrap();
    assert_eq!(out_img.width, 5);
    assert_eq!(out_img.height, 5);
}

#[test]
fn test_bilinear_interpolate() {
    let data = vec![0u8, 255];
    let val = bilinear_interpolate(&data, 0.5, 0.0, 2, 1, 1, 0);
    // Midpoint between 0 and 255 should be ~127
    assert!((val as f32 - 127.0).abs() < 1.0);
}

#[test]
fn test_bilinear_interpolate_out_of_bounds() {
    let data = vec![128u8];
    let val = bilinear_interpolate(&data, -1.0, 0.0, 1, 1, 1, 0);
    // Out of bounds should return 0
    assert_eq!(val, 0);
}

#[test]
fn test_affine_with_translation() {
    let mut data = vec![255u8; 9];
    let mut img = FusableImage::new(&mut data, 3, 3, 1);

    let mut params = AffineParams::default();
    params.translate = (1.0, 1.0);
    let a = Affine::new(params);
    let result = a.execute(&mut img);

    assert!(result.is_some());
    let _out_img = result.unwrap();
    // Translation should shift the image
}
