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

#[test]
fn test_box_blur_constant_preservation_all_cases() {
    // 1. Tail columns: w=9, h=1, r=1
    let mut data = vec![137u8; 9 * 1 * 3];
    let mut img = FusableImage::new(&mut data, 9, 1, 3);
    box_blur(&mut img, 1);
    assert!(img.data.iter().all(|&p| p == 137), "Tail column w=9 h=1 failed");

    // 2. Narrow image w < 8: w=3, h=5, r=2
    let mut data = vec![137u8; 3 * 5 * 3];
    let mut img = FusableImage::new(&mut data, 3, 5, 3);
    box_blur(&mut img, 2);
    assert!(img.data.iter().all(|&p| p == 137), "Narrow image w=3 h=5 failed");

    // 3. Strip crossing: h=257 (exceeds 256 strip boundary)
    let mut data = vec![200u8; 17 * 257 * 3];
    let mut img = FusableImage::new(&mut data, 17, 257, 3);
    box_blur(&mut img, 3);
    assert!(img.data.iter().all(|&p| p == 200), "Strip crossing h=257 failed");
}

#[test]
fn test_box_blur_neon_matches_scalar_reference() {
    // Compare NEON RGB implementation against scalar implementation across tricky sizes
    for &(w, h, r) in &[
        (9, 1, 1),
        (3, 3, 1),
        (7, 11, 2),
        (15, 17, 3),
        (25, 260, 2), // crosses 256 rows
        (33, 40, 5),
    ] {
        let mut data_neon = Vec::with_capacity(w * h * 3);
        for i in 0..(w * h * 3) {
            data_neon.push(((i * 17 + 31) % 256) as u8);
        }
        let mut data_scalar = data_neon.clone();

        let mut img_neon = FusableImage::new(&mut data_neon, w, h, 3);
        box_blur(&mut img_neon, r);

        box_blur_impl(&mut data_scalar, w, h, 3, r);

        let mut max_diff = 0i32;
        for i in 0..data_neon.len() {
            let diff = (data_neon[i] as i32 - data_scalar[i] as i32).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff <= 1,
            "NEON vs scalar mismatch for w={}, h={}, r={}: max_diff={}",
            w, h, r, max_diff
        );
    }
}
