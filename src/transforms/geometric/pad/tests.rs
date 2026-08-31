// Tests for Pad transform

use super::*;

#[test]
fn test_pad_new() {
    let p = Pad::new(1, 2, 3, 4, PadMode::Constant(0));
    assert_eq!(p.top, 1);
    assert_eq!(p.bottom, 2);
    assert_eq!(p.left, 3);
    assert_eq!(p.right, 4);
}

#[test]
fn test_pad_symmetric() {
    let p = Pad::symmetric(5, PadMode::Constant(128));
    assert_eq!(p.top, 5);
    assert_eq!(p.bottom, 5);
    assert_eq!(p.left, 5);
    assert_eq!(p.right, 5);
}

#[test]
fn test_pad_with_fill() {
    let p = Pad::with_fill(10, 20, 5, 15, 255);
    assert_eq!(p.top, 10);
    assert_eq!(p.bottom, 20);
    assert_eq!(p.left, 5);
    assert_eq!(p.right, 15);
    if let PadMode::Constant(v) = p.mode {
        assert_eq!(v, 255);
    } else {
        panic!("Expected Constant mode");
    }
}

#[test]
fn test_pad_constant_zero() {
    // Single pixel image
    let mut data = vec![128u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 1);

    let result = Pad::with_fill(1, 1, 1, 1, 0).execute(&mut img);

    assert!(result.is_some());
    let padded = result.unwrap();
    assert_eq!(padded.width, 3);
    assert_eq!(padded.height, 3);
    // Center pixel should be 128
    assert_eq!(padded.data[4], 128); // Center of 3x3
    // All other pixels should be 0
    assert_eq!(padded.data[0], 0);
    assert_eq!(padded.data[1], 0);
    assert_eq!(padded.data[2], 0);
}

#[test]
fn test_pad_constant_rgb() {
    // 1x1 RGB
    let mut data = vec![255u8, 0, 128];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    let result = Pad::with_fill(0, 0, 1, 0, 42).execute(&mut img);

    assert!(result.is_some());
    let padded = result.unwrap();
    assert_eq!(padded.width, 2);
    assert_eq!(padded.height, 1);
    assert_eq!(padded.channels, 3);
    // First pixel should be fill value
    assert_eq!(padded.data[0], 42);
    assert_eq!(padded.data[1], 42);
    assert_eq!(padded.data[2], 42);
    // Second pixel should be original
    assert_eq!(padded.data[3], 255);
    assert_eq!(padded.data[4], 0);
    assert_eq!(padded.data[5], 128);
}

#[test]
fn test_pad_replicate() {
    // 2x2 image
    let mut data = vec![1u8, 2, 3, 4];
    let mut img = FusableImage::new(&mut data, 2, 2, 1);

    let result = Pad::symmetric(1, PadMode::Replicate).execute(&mut img);

    assert!(result.is_some());
    let padded = result.unwrap();
    assert_eq!(padded.width, 4);
    assert_eq!(padded.height, 4);
    // Top-left corner should be 1 (replicated from (0,0))
    assert_eq!(padded.data[0], 1);
    // Center 2x2 should be original
    assert_eq!(padded.data[5], 1); // Row 1, Col 1
    assert_eq!(padded.data[6], 2); // Row 1, Col 2
    assert_eq!(padded.data[9], 3); // Row 2, Col 1
    assert_eq!(padded.data[10], 4); // Row 2, Col 2
}

#[test]
fn test_pad_no_padding() {
    let mut data = vec![42u8; 4];
    let mut img = FusableImage::new(&mut data, 2, 2, 1);

    let result = Pad::with_fill(0, 0, 0, 0, 0).execute(&mut img);

    assert!(result.is_some());
    let padded = result.unwrap();
    assert_eq!(padded.width, 2);
    assert_eq!(padded.height, 2);
    assert_eq!(padded.data, vec![42u8; 4]);
}

#[test]
fn test_pad_access_pattern() {
    let p = Pad::symmetric(5, PadMode::Constant(0));
    assert_eq!(p.access(), AccessPattern::OutOfPlace);
    assert_eq!(p.shape_effect(), ShapeEffect::Resize);
}

#[test]
fn test_pad_asymmetric() {
    let mut data = vec![100u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 1);

    let result = Pad::with_fill(1, 2, 3, 4, 99).execute(&mut img);

    assert!(result.is_some());
    let padded = result.unwrap();
    // 1 + 3 + 4 = 8 width
    // 1 + 1 + 2 = 4 height
    assert_eq!(padded.width, 8);
    assert_eq!(padded.height, 4);
}

/// Bit-exactness of the routed pad-reflect path (NEON gray on aarch64, scalar
/// fallback elsewhere) vs the scalar reference `pad_reflect_scalar`.
/// Covers borders smaller than, equal to, and larger than the source axis.
fn assert_reflect_matches_scalar(
    src: &[u8],
    w: usize,
    h: usize,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
) {
    let new_w = w + left + right;
    let new_h = h + top + bottom;

    // Scalar reference (pad_fast_slice with Reflect)
    let mut expected = vec![0u8; new_w * new_h];
    super::pad_reflect_scalar(&mut expected, src, new_w, w, h, top, left, 1);

    // Routed path
    let mut data = src.to_vec();
    let mut img = FusableImage::new(&mut data, w, h, 1);
    let result =
        Pad::new(top as u32, bottom as u32, left as u32, right as u32, PadMode::Reflect)
            .execute(&mut img);
    let padded = result.expect("pad should return a barrier image");
    assert_eq!(padded.width as usize, new_w);
    assert_eq!(padded.height as usize, new_h);
    assert_eq!(padded.data, expected, "bit-exact vs scalar reflect");
}

#[test]
fn test_pad_reflect_bit_exact_small() {
    let src: Vec<u8> = (0..(5 * 6)).map(|i| (i * 7 % 251) as u8).collect();
    assert_reflect_matches_scalar(&src, 5, 6, 2, 3, 2, 3);
}

#[test]
fn test_pad_reflect_bit_exact_border_equals_dimension() {
    let src: Vec<u8> = (0..(5 * 5)).map(|i| (i * 13 % 255) as u8).collect();
    assert_reflect_matches_scalar(&src, 5, 5, 5, 5, 5, 5);
}

#[test]
fn test_pad_reflect_bit_exact_border_exceeds_dimension() {
    // Borders larger than the axis on both width and height.
    let src: Vec<u8> = (0..(3 * 4)).map(|i| (i * 31 % 250) as u8).collect();
    assert_reflect_matches_scalar(&src, 3, 4, 7, 9, 5, 8);
}

#[test]
fn test_pad_reflect_bit_exact_1x1_huge_border() {
    let src = vec![200u8];
    assert_reflect_matches_scalar(&src, 1, 1, 20, 20, 20, 20);
}

#[test]
fn test_pad_reflect_bit_exact_no_horizontal_padding() {
    let src: Vec<u8> = (0..(8 * 3)).map(|i| (i * 3 % 254) as u8).collect();
    assert_reflect_matches_scalar(&src, 8, 3, 4, 2, 0, 0);
}

#[test]
fn test_pad_reflect_bit_exact_only_horizontal() {
    let src: Vec<u8> = (0..(4 * 4)).map(|i| (i * 17 % 253) as u8).collect();
    assert_reflect_matches_scalar(&src, 4, 4, 0, 0, 6, 6);
}
