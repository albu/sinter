// Tests for Transpose transform

use super::*;

#[test]
fn test_transpose_new() {
    let _t = Transpose::new();
}

#[test]
fn test_transpose_default() {
    let _t = Transpose::default();
}

#[test]
fn test_transpose_dimensions() {
    let mut data = vec![1u8; 6];
    let mut img = FusableImage::new(&mut data, 3, 2, 1);

    let result = Transpose::new().execute(&mut img);

    assert!(result.is_some());
    let transposed = result.unwrap();
    assert_eq!(transposed.width, 2);
    assert_eq!(transposed.height, 3);
}

#[test]
fn test_transpose_square() {
    let mut data = vec![1u8, 0, 0, 1];
    let mut img = FusableImage::new(&mut data, 2, 2, 1);

    let result = Transpose::new().execute(&mut img);

    assert!(result.is_some());
    let transposed = result.unwrap();
    assert_eq!(transposed.data, &[1, 0, 0, 1]);
}

#[test]
fn test_transpose_values() {
    // [1, 2, 3]
    // [4, 5, 6]
    let mut data = vec![1u8, 2, 3, 4, 5, 6];
    let mut img = FusableImage::new(&mut data, 3, 2, 1);

    let result = Transpose::new().execute(&mut img);

    assert!(result.is_some());
    let transposed = result.unwrap();
    // [1, 4]
    // [2, 5]
    // [3, 6]
    assert_eq!(transposed.width, 2);
    assert_eq!(transposed.height, 3);
    assert_eq!(transposed.data, &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_transpose_rgb() {
    let mut data = vec![
        10, 20, 30, // (0,0)
        40, 50, 60, // (1,0)
        70, 80, 90, // (0,1)
        100, 110, 120, // (1,1)
    ];
    let mut img = FusableImage::new(&mut data, 2, 2, 3);

    let result = Transpose::new().execute(&mut img);

    assert!(result.is_some());
    let transposed = result.unwrap();
    // [10,20,30] [70,80,90]
    // [40,50,60] [100,110,120]
    assert_eq!(
        transposed.data,
        &[10, 20, 30, 70, 80, 90, 40, 50, 60, 100, 110, 120]
    );
}

#[test]
fn test_transpose_access_pattern() {
    let t = Transpose::new();
    assert_eq!(t.access(), AccessPattern::OutOfPlace);
    assert_eq!(t.shape_effect(), ShapeEffect::Resize);
}
