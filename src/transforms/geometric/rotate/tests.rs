// Tests for Rotate transform

use crate::core::{FusableImage, Executable, AccessPattern, ShapeEffect, Transform};
use super::{Rotate, RotateAngle};

#[test]
fn test_rotate_new() {
    let r = Rotate::new(RotateAngle::Rotate90);
    assert_eq!(r.angle, RotateAngle::Rotate90);
}

#[test]
fn test_rotate_convenience_constructors() {
    let r90 = Rotate::rotate_90();
    let r180 = Rotate::rotate_180();
    let r270 = Rotate::rotate_270();

    assert_eq!(r90.angle, RotateAngle::Rotate90);
    assert_eq!(r180.angle, RotateAngle::Rotate180);
    assert_eq!(r270.angle, RotateAngle::Rotate270);
}

#[test]
fn test_rotate_90_dimensions() {
    let mut data = vec![1u8; 36]; // 4x3 RGB
    let mut img = FusableImage::new(&mut data, 4, 3, 3);

    let result = Rotate::rotate_90().execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    assert_eq!(rotated.width, 3); // swapped
    assert_eq!(rotated.height, 4);
    assert_eq!(rotated.channels, 3);
}

#[test]
fn test_rotate_180_dimensions() {
    let mut data = vec![1u8; 12]; // 4x3
    let mut img = FusableImage::new(&mut data, 4, 3, 1);

    let result = Rotate::rotate_180().execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    assert_eq!(rotated.width, 4); // unchanged
    assert_eq!(rotated.height, 3);
}

#[test]
fn test_rotate_270_dimensions() {
    let mut data = vec![1u8; 12]; // 4x3
    let mut img = FusableImage::new(&mut data, 4, 3, 1);

    let result = Rotate::rotate_270().execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    assert_eq!(rotated.width, 3); // swapped
    assert_eq!(rotated.height, 4);
}

#[test]
fn test_rotate_90_values() {
    // 2x3 image:
    // [1, 2]
    // [3, 4]
    // [5, 6]
    let mut data = vec![1u8, 2, 3, 4, 5, 6];
    let mut img = FusableImage::new(&mut data, 2, 3, 1);

    let result = Rotate::rotate_90().execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    // After 90° clockwise:
    // [5, 3, 1]
    // [6, 4, 2]
    assert_eq!(rotated.width, 3);
    assert_eq!(rotated.height, 2);
    assert_eq!(rotated.data, &[5, 3, 1, 6, 4, 2]);
}

#[test]
fn test_rotate_180_values() {
    let mut data = vec![1u8, 2, 3, 4, 5, 6];
    let mut img = FusableImage::new(&mut data, 3, 2, 1);

    let result = Rotate::rotate_180().execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    // After 180°:
    // [6, 5, 4]
    // [3, 2, 1]
    assert_eq!(rotated.data, &[6, 5, 4, 3, 2, 1]);
}

#[test]
fn test_rotate_270_values() {
    let mut data = vec![1u8, 2, 3, 4, 5, 6];
    let mut img = FusableImage::new(&mut data, 2, 3, 1);

    let result = Rotate::rotate_270().execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    // After 270° clockwise (90° CCW):
    // [2, 4, 6]
    // [1, 3, 5]
    assert_eq!(rotated.width, 3);
    assert_eq!(rotated.height, 2);
    assert_eq!(rotated.data, &[2, 4, 6, 1, 3, 5]);
}

#[test]
fn test_rotate_90_rgb() {
    // 2x2 RGB:
    // [R0,G0,B0] [R1,G1,B1]
    // [R2,G2,B2] [R3,G3,B3]
    let mut data = vec![
        10, 20, 30,   // pixel 0
        40, 50, 60,   // pixel 1
        70, 80, 90,   // pixel 2
        100, 110, 120 // pixel 3
    ];
    let mut img = FusableImage::new(&mut data, 2, 2, 3);

    let result = Rotate::rotate_90().execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    assert_eq!(rotated.width, 2);
    assert_eq!(rotated.height, 2);
    assert_eq!(rotated.channels, 3);
    // Top row becomes right column: [70,80,90] [10,20,30]
    // Bottom row becomes left column: [100,110,120] [40,50,60]
    assert_eq!(
        rotated.data,
        &[70, 80, 90, 10, 20, 30, 100, 110, 120, 40, 50, 60]
    );
}

#[test]
fn test_rotate_180_square() {
    let mut data = vec![1u8, 2, 3, 4];
    let mut img = FusableImage::new(&mut data, 2, 2, 1);

    let result = Rotate::rotate_180().execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    assert_eq!(rotated.data, &[4, 3, 2, 1]);
}

#[test]
fn test_rotate_90_180_270_consistency() {
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];

    // Rotate 90° three times should equal 270°
    let mut data1 = data.clone();
    let mut img90 = FusableImage::new(&mut data1, 3, 3, 1);
    let r90 = Rotate::rotate_90().execute(&mut img90).unwrap();

    let mut data2 = r90.data.clone();
    let mut img180 = FusableImage::new(&mut data2, 3, 3, 1);
    let r180 = Rotate::rotate_90().execute(&mut img180).unwrap();

    let mut data3 = r180.data.clone();
    let mut img270 = FusableImage::new(&mut data3, 3, 3, 1);
    let r270_twice = Rotate::rotate_90().execute(&mut img270).unwrap();

    let mut data4 = data.clone();
    let mut img = FusableImage::new(&mut data4, 3, 3, 1);
    let r270_direct = Rotate::rotate_270().execute(&mut img).unwrap();

    assert_eq!(r270_twice.data, r270_direct.data);
}

#[test]
fn test_rotate_access_pattern() {
    // All rotations are OutOfPlace (allocate for speed)
    let r90 = Rotate::rotate_90();
    assert_eq!(r90.access(), AccessPattern::OutOfPlace);
    assert_eq!(r90.shape_effect(), ShapeEffect::Resize);

    let r180 = Rotate::rotate_180();
    assert_eq!(r180.access(), AccessPattern::OutOfPlace);
    assert_eq!(r180.shape_effect(), ShapeEffect::Resize);

    let r270 = Rotate::rotate_270();
    assert_eq!(r270.access(), AccessPattern::OutOfPlace);
    assert_eq!(r270.shape_effect(), ShapeEffect::Resize);
}
