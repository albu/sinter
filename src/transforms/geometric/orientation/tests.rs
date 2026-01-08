// Tests for Orientation and StructuralKernel

use crate::core::{FusableImage, Executable};
use super::Orientation;
use super::kernel::StructuralKernel;

#[test]
fn test_orientation_compose_rotations() {
    // Rot90 * Rot90 = Rot180
    assert_eq!(
        Orientation::Rot90.compose(Orientation::Rot90),
        Orientation::Rot180
    );

    // Rot90 * Rot180 = Rot270
    assert_eq!(
        Orientation::Rot90.compose(Orientation::Rot180),
        Orientation::Rot270
    );

    // Rot90 * Rot270 = Identity
    assert_eq!(
        Orientation::Rot90.compose(Orientation::Rot270),
        Orientation::Identity
    );

    // Rot180 * Rot180 = Identity
    assert_eq!(
        Orientation::Rot180.compose(Orientation::Rot180),
        Orientation::Identity
    );
}

#[test]
fn test_orientation_compose_flips() {
    // FlipH * FlipH = Identity
    assert_eq!(
        Orientation::FlipH.compose(Orientation::FlipH),
        Orientation::Identity
    );

    // FlipV * FlipV = Identity
    assert_eq!(
        Orientation::FlipV.compose(Orientation::FlipV),
        Orientation::Identity
    );

    // FlipH * FlipV = Rot180
    assert_eq!(
        Orientation::FlipH.compose(Orientation::FlipV),
        Orientation::Rot180
    );
}

#[test]
fn test_orientation_compose_flip_rotation() {
    // FlipH * Rot90 = Transverse
    assert_eq!(
        Orientation::FlipH.compose(Orientation::Rot90),
        Orientation::Transverse
    );

    // Rot90 * FlipH = Transpose
    assert_eq!(
        Orientation::Rot90.compose(Orientation::FlipH),
        Orientation::Transpose
    );
}

#[test]
fn test_output_size() {
    // Preserving orientations
    assert_eq!(Orientation::Identity.output_size(100, 50), (100, 50));
    assert_eq!(Orientation::Rot180.output_size(100, 50), (100, 50));
    assert_eq!(Orientation::FlipH.output_size(100, 50), (100, 50));
    assert_eq!(Orientation::FlipV.output_size(100, 50), (100, 50));

    // Swapping orientations
    assert_eq!(Orientation::Rot90.output_size(100, 50), (50, 100));
    assert_eq!(Orientation::Rot270.output_size(100, 50), (50, 100));
    assert_eq!(Orientation::Transpose.output_size(100, 50), (50, 100));
    assert_eq!(Orientation::Transverse.output_size(100, 50), (50, 100));
}

#[test]
fn test_structural_kernel_identity() {
    let mut data = vec![1u8, 2, 3, 4];
    let mut img = FusableImage::new(&mut data, 2, 2, 1);

    let kernel = StructuralKernel::identity();
    let result = kernel.execute(&mut img);

    assert!(result.is_none());
    assert_eq!(img.data, &[1, 2, 3, 4]);
}

#[test]
fn test_structural_kernel_rot90() {
    // 2x3:
    // [1, 2]
    // [3, 4]
    // [5, 6]
    let mut data = vec![1u8, 2, 3, 4, 5, 6];
    let mut img = FusableImage::new(&mut data, 2, 3, 1);

    let kernel = StructuralKernel::new(Orientation::Rot90);
    let result = kernel.execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    // After 90° CW: [5, 3, 1] / [6, 4, 2]
    assert_eq!(rotated.width, 3);
    assert_eq!(rotated.height, 2);
    assert_eq!(rotated.data, &[5, 3, 1, 6, 4, 2]);
}

#[test]
fn test_structural_kernel_rot180() {
    let mut data = vec![1u8, 2, 3, 4, 5, 6];
    let mut img = FusableImage::new(&mut data, 3, 2, 1);

    let kernel = StructuralKernel::new(Orientation::Rot180);
    let result = kernel.execute(&mut img);

    assert!(result.is_none());
    assert_eq!(img.data, &[6, 5, 4, 3, 2, 1]);
}

#[test]
fn test_structural_kernel_fliph() {
    let mut data = vec![1u8, 2, 3, 4];
    let mut img = FusableImage::new(&mut data, 2, 2, 1);

    let kernel = StructuralKernel::new(Orientation::FlipH);
    let result = kernel.execute(&mut img);

    assert!(result.is_none());
    assert_eq!(img.data, &[2, 1, 4, 3]);
}

#[test]
fn test_structural_kernel_flipv() {
    let mut data = vec![1u8, 2, 3, 4];
    let mut img = FusableImage::new(&mut data, 2, 2, 1);

    let kernel = StructuralKernel::new(Orientation::FlipV);
    let result = kernel.execute(&mut img);

    assert!(result.is_none());
    assert_eq!(img.data, &[3, 4, 1, 2]);
}

#[test]
fn test_structural_kernel_rot270() {
    let mut data = vec![1u8, 2, 3, 4, 5, 6];
    let mut img = FusableImage::new(&mut data, 2, 3, 1);

    let kernel = StructuralKernel::new(Orientation::Rot270);
    let result = kernel.execute(&mut img);

    assert!(result.is_some());
    let rotated = result.unwrap();
    // After 270° CW (90° CCW): [2, 4, 6] / [1, 3, 5]
    assert_eq!(rotated.width, 3);
    assert_eq!(rotated.height, 2);
    assert_eq!(rotated.data, &[2, 4, 6, 1, 3, 5]);
}

#[test]
fn test_dihedral_group_closure() {
    // Test that all orientations compose correctly
    let orientations = [
        Orientation::Identity,
        Orientation::Rot90,
        Orientation::Rot180,
        Orientation::Rot270,
        Orientation::FlipH,
        Orientation::FlipV,
        Orientation::Transpose,
        Orientation::Transverse,
    ];

    for &a in &orientations {
        for &b in &orientations {
            let result = a.compose(b);
            // Result should always be a valid orientation
            assert!(orientations.contains(&result));
        }
    }
}
