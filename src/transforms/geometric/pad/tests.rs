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
