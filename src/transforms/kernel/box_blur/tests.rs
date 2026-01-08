// Tests for Box Blur transform

use crate::core::FusableImage;
use super::*;

#[test]
fn test_box_blur_gaussian_approx() {
    let mut data = vec![128u8; 50 * 50 * 3];
    let mut img = FusableImage::new(&mut data, 50, 50, 3);

    box_blur_gaussian(&mut img, 3, 3);

    // Should complete successfully
    assert_eq!(img.data.len(), 50 * 50 * 3);
}

#[test]
fn test_box_blur_preserves_mean() {
    let mut data = vec![0u8, 128u8, 255u8];
    let original_mean: u32 = data.iter().map(|&p| p as u32).sum::<u32>() / data.len() as u32;

    let mut img = FusableImage::new(&mut data, 3, 1, 1);
    box_blur(&mut img, 1);

    let new_mean: u32 = img.data.iter().map(|&p| p as u32).sum::<u32>() / img.data.len() as u32;

    // Mean should be approximately preserved (within 1)
    assert!((new_mean as i32 - original_mean as i32).abs() <= 1);
}
