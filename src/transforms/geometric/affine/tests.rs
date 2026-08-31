// Tests for Affine transform

use super::interpolation::bilinear_interpolate;
use super::rust_impl::execute_rust;
use super::{Affine, AffineBorderMode, AffineInterpolation, AffineParams};
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

fn gray_data(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height];
    // Deterministic pseudo-random pattern (including exact 0..255 extremes).
    let mut x = 0x9E3779B97F4A7C15u64;
    for v in data.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = ((x >> 32) % 256) as u8;
    }
    data
}

fn rgb_data(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    let mut x = 0x2545F4914F6CDD1Du64;
    for v in data.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = ((x >> 33) % 256) as u8;
    }
    data
}

fn assert_execute_matches_scalar(
    affine: &Affine,
    data: &mut Vec<u8>,
    width: usize,
    height: usize,
    channels: usize,
) {
    let src = data.clone();
    let neon_out = {
        let mut img = FusableImage::new(data.as_mut_slice(), width, height, channels);
        affine.execute(&mut img).unwrap()
    };
    let mut scalar_data = src;
    let scalar_out = {
        let mut img = FusableImage::new(scalar_data.as_mut_slice(), width, height, channels);
        execute_rust(affine, &mut img)
    };
    assert_eq!(neon_out.width, scalar_out.width, "width mismatch");
    assert_eq!(neon_out.height, scalar_out.height, "height mismatch");
    assert_eq!(neon_out.channels, scalar_out.channels, "channels mismatch");
    assert_eq!(
        neon_out.data, scalar_out.data,
        "NEON (or scalar) path diverged from scalar reference for params {:?}",
        affine.params
    );
    *data = scalar_data;
}

#[test]
fn test_affine_gray_bilinear_neon_matches_scalar_reference() {
    let sizes = [(16usize, 16usize), (37, 53), (64, 64), (65, 67), (9, 9), (8, 8), (71, 40)];
    let params: &[AffineParams] = &[
        // Fast path: pure scale/translate (dy_fp == 0, dx_fp >= 0)
        AffineParams { scale: (1.5, 1.5), rotate: 0.0, translate: (0.0, 0.0), shear: (0.0, 0.0) },
        AffineParams { scale: (1.0, 1.0), rotate: 0.0, translate: (3.5, -2.25), shear: (0.0, 0.0) },
        AffineParams { scale: (0.75, 0.75), rotate: 0.0, translate: (1.0, 1.0), shear: (0.0, 0.0) },
        AffineParams { scale: (0.5, 0.5), rotate: 0.0, translate: (0.0, 0.0), shear: (0.0, 0.0) },
        AffineParams { scale: (2.0, 0.9), rotate: 0.0, translate: (-4.0, 6.0), shear: (0.0, 0.0) },
        AffineParams { scale: (1.0, 1.0), rotate: 0.0, translate: (0.0, 0.0), shear: (0.0, 0.0) },
        // General path: rotation / shear (dy_fp != 0)
        AffineParams { scale: (1.5, 1.5), rotate: 30.0, translate: (0.0, 0.0), shear: (0.0, 0.0) },
        AffineParams { scale: (1.0, 1.0), rotate: 15.0, translate: (2.0, -3.0), shear: (0.2, 0.0) },
        // General path: shallow rotation (long constant-y0 runs, span-1 blocks)
        AffineParams { scale: (1.0, 1.0), rotate: 3.5, translate: (0.0, 0.0), shear: (0.0, 0.0) },
        // General path: steep rotation (y span > 2 per 8 px -> scalar fallback)
        AffineParams { scale: (2.0, 2.0), rotate: 75.0, translate: (0.0, 0.0), shear: (0.0, 0.0) },
        // General path: rotation + shear mix
        AffineParams { scale: (1.3, 1.3), rotate: 45.0, translate: (0.0, 0.0), shear: (5.0, -5.0) },
        // General path: x-mirror (dx_fp < 0)
        AffineParams { scale: (-1.0, 1.0), rotate: 0.0, translate: (0.0, 0.0), shear: (0.0, 0.0) },
        // Zoom-out that drifts > 14 source pixels per 8 output pixels (scalar
        // fallback inside the fast-row structure).
        AffineParams { scale: (0.2, 0.2), rotate: 0.0, translate: (0.0, 0.0), shear: (0.0, 0.0) },
    ];
    let borders = [
        AffineBorderMode::Replicate,
        AffineBorderMode::Reflect,
        AffineBorderMode::Wrap,
        AffineBorderMode::Constant { value: 7 },
    ];

    for &(w, h) in &sizes {
        for p in params {
            for &border in &borders {
                let a = Affine::with_all(*p, w, h, AffineInterpolation::Bilinear, border);
                let mut data = gray_data(w, h);
                assert_execute_matches_scalar(&a, &mut data, w, h, 1);
            }
        }
    }

    // Output-size variants (shape-changing) on a couple of fast-path params.
    for p in &params[..3] {
        let a = Affine::with_output_size_and_interpolation(*p, 77, 63, AffineInterpolation::Bilinear);
        let mut data = gray_data(40, 50);
        assert_execute_matches_scalar(&a, &mut data, 40, 50, 1);
    }
}

#[test]
fn test_affine_rgb_bilinear_unchanged_by_gray_fast_path() {
    // Regression guard: the gray fast path must not affect the RGB path.
    let params = AffineParams { scale: (1.5, 1.5), rotate: 0.0, translate: (0.0, 0.0), shear: (0.0, 0.0) };
    let a = Affine::with_all(params, 64, 64, AffineInterpolation::Bilinear, AffineBorderMode::Replicate);
    let mut data = rgb_data(64, 64);
    assert_execute_matches_scalar(&a, &mut data, 64, 64, 3);
}

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
