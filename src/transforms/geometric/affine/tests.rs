// Tests for Affine transform

use super::interpolation::bilinear_interpolate;
use super::{Affine, AffineBorderMode, AffineInterpolation, AffineParams};
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
fn test_affine_scale_5x5() {
    let data: Vec<u8> = (0..25).map(|v| v as u8).collect();
    let mut data_3ch: Vec<u8> = Vec::new();
    for &v in &data {
        data_3ch.push(v);
        data_3ch.push(v);
        data_3ch.push(v);
    }
    let mut img = FusableImage::new(&mut data_3ch, 5, 5, 3);
    let params = AffineParams {
        scale: (1.2, 0.8),
        rotate: 0.0,
        translate: (0.0, 0.0),
        shear: (0.0, 0.0),
    };
    let a = Affine::with_all(
        params,
        5,
        5,
        AffineInterpolation::Bilinear,
        AffineBorderMode::Constant { value: 0 },
    );
    let result = a.execute(&mut img).unwrap();
    println!("Matrix: {:?}", a.build_inverse_matrix(5, 5));
    for y in 0..5 {
        let row: Vec<u8> = (0..5).map(|x| result.data[(y * 5 + x) * 3]).collect();
        println!("Rust row {}: {:?}", y, row);
    }
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
fn test_affine_identity_gradient() {
    // Gradient image so any coordinate error shows up immediately
    let w = 16;
    let h = 12;
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            data.push((x as u8).wrapping_mul(16).wrapping_add(y as u8));
        }
    }
    let mut img = FusableImage::new(&mut data, w, h, 1);

    let params = AffineParams::default();
    let a = Affine::with_all(params, w, h, AffineInterpolation::Nearest, AffineBorderMode::Replicate);
    let result = a.execute(&mut img).unwrap();

    let mut mismatches = 0usize;
    let mut max_diff = 0i32;
    for (i, (&got, &expected)) in result.data.iter().zip(data.iter()).enumerate() {
        let diff = (got as i32 - expected as i32).abs();
        if diff > 0 {
            mismatches += 1;
            max_diff = max_diff.max(diff);
            if mismatches <= 5 {
                eprintln!("  idx={} (x={}, y={}): got={} expected={}", i, i % w, i / w, got, expected);
            }
        }
    }
    assert_eq!(mismatches, 0, "identity nearest: {} mismatches, max_diff={}", mismatches, max_diff);
}

#[test]
fn test_affine_identity_bilinear_constant() {
    // Same as Python sampled path: Bilinear + Constant{0} border
    let w = 8;
    let h = 8;
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            data.push((x as u8).wrapping_mul(16).wrapping_add(y as u8));
        }
    }
    let mut img = FusableImage::new(&mut data, w, h, 1);

    let params = AffineParams::default();
    let a = Affine::with_all(
        params,
        w,
        h,
        AffineInterpolation::Bilinear,
        AffineBorderMode::Constant { value: 0 },
    );
    let result = a.execute(&mut img).unwrap();

    let mut mismatches = 0usize;
    let mut max_diff = 0i32;
    for (i, (&got, &expected)) in result.data.iter().zip(data.iter()).enumerate() {
        let diff = (got as i32 - expected as i32).abs();
        if diff > 0 {
            mismatches += 1;
            max_diff = max_diff.max(diff);
            if mismatches <= 8 {
                eprintln!("  idx={} (x={}, y={}): got={} expected={}", i, i % w, i / w, got, expected);
            }
        }
    }
    assert_eq!(mismatches, 0, "identity bilinear+const: {} mismatches, max_diff={}", mismatches, max_diff);
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
