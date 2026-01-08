// Tests for LUT operations and fusion

use crate::core::{FusableImage, Executable};
use super::*;

// A simple test transform: invert values
#[derive(Debug, Clone, Copy, PartialEq)]
struct TestInvert;

impl LutOp for TestInvert {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for i in 0..256 {
            lut[i] = 255 - i as u8;
        }
        lut
    }
}

// A threshold transform (like Solarize)
#[derive(Debug, Clone, Copy, PartialEq)]
struct TestThreshold {
    threshold: u8,
}

impl LutOp for TestThreshold {
    fn build_lut(&self) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for i in 0..256 {
            let i = i as u8;
            lut[i as usize] = if i >= self.threshold { 255 - i } else { i };
        }
        lut
    }
}

// ===== LUT Executor Tests =====

#[test]
fn test_lut_executor_invert() {
    let mut data = vec![0u8, 128u8, 255u8];
    let mut img = FusableImage::new(&mut data, 3, 1, 1);

    TestInvert.execute_with_lut(&mut img);

    assert_eq!(img.data[0], 255);
    assert_eq!(img.data[1], 127);
    assert_eq!(img.data[2], 0);
}

#[test]
fn test_lut_executor_threshold() {
    let mut data = vec![0u8, 64u8, 127u8, 128u8, 191u8, 255u8];
    let mut img = FusableImage::new(&mut data, 6, 1, 1);

    TestThreshold { threshold: 128 }.execute_with_lut(&mut img);

    assert_eq!(img.data[0], 0);   // Below threshold
    assert_eq!(img.data[1], 64);  // Below threshold
    assert_eq!(img.data[2], 127); // Below threshold
    assert_eq!(img.data[3], 127); // At threshold (255-128=127)
    assert_eq!(img.data[4], 64);  // Above threshold (255-191=64)
    assert_eq!(img.data[5], 0);   // Above threshold (255-255=0)
}

#[test]
fn test_lut_executor_rgb() {
    let mut data = vec![255u8, 0u8, 128u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    TestInvert.execute_with_lut(&mut img);

    assert_eq!(img.data[0], 0);   // R: 255 -> 0
    assert_eq!(img.data[1], 255); // G: 0 -> 255
    assert_eq!(img.data[2], 127); // B: 128 -> 127
}

#[test]
fn test_lut_executor_large_image() {
    let mut data = vec![128u8; 256];
    let mut img = FusableImage::new(&mut data, 16, 16, 1);

    TestInvert.execute_with_lut(&mut img);

    // All pixels should be 127
    assert!(img.data.iter().all(|&p| p == 127));
}

#[test]
fn test_lut_executor_with_remainder() {
    // 35 pixels = 32 (8*4) + 3 remainder
    let mut data = vec![100u8; 35];
    let mut img = FusableImage::new(&mut data, 5, 7, 1);

    TestInvert.execute_with_lut(&mut img);

    // All pixels should be 155
    assert!(img.data.iter().all(|&p| p == 155));
}

#[test]
fn test_lut_build_identity() {
    // Identity LUT (no change)
    let mut data = vec![0u8, 128u8, 255u8];
    let mut img = FusableImage::new(&mut data, 3, 1, 1);

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Identity;
    impl LutOp for Identity {
        fn build_lut(&self) -> [u8; 256] {
            let mut lut = [0u8; 256];
            for i in 0..256 {
                lut[i] = i as u8;
            }
            lut
        }
    }

    Identity.execute_with_lut(&mut img);

    assert_eq!(img.data[0], 0);
    assert_eq!(img.data[1], 128);
    assert_eq!(img.data[2], 255);
}

#[test]
fn test_lut_build_clamp() {
    // LUT that clamps values to [64, 192]
    let mut data = vec![0u8, 64u8, 128u8, 192u8, 255u8];
    let mut img = FusableImage::new(&mut data, 5, 1, 1);

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Clamp;
    impl LutOp for Clamp {
        fn build_lut(&self) -> [u8; 256] {
            let mut lut = [0u8; 256];
            for i in 0..256 {
                lut[i] = (i as u8).clamp(64, 192);
            }
            lut
        }
    }

    Clamp.execute_with_lut(&mut img);

    assert_eq!(img.data[0], 64);   // 0 -> 64
    assert_eq!(img.data[1], 64);   // 64 -> 64
    assert_eq!(img.data[2], 128);  // 128 -> 128
    assert_eq!(img.data[3], 192);  // 192 -> 192
    assert_eq!(img.data[4], 192);  // 255 -> 192
}

#[test]
fn test_threshold_lut_correctness() {
    // Verify LUT is built correctly
    let op = TestThreshold { threshold: 128 };
    let lut = op.build_lut();

    // Check boundary conditions
    assert_eq!(lut[0], 0);
    assert_eq!(lut[127], 127);   // Below threshold
    assert_eq!(lut[128], 127);   // At threshold, inverted
    assert_eq!(lut[255], 0);     // Above threshold, inverted

    // Verify the formula: if i >= threshold: 255-i else: i
    for i in 0..256 {
        let expected = if i >= 128 { 255 - i as u8 } else { i as u8 };
        assert_eq!(lut[i], expected);
    }
}

#[test]
fn test_invert_lut_correctness() {
    let op = TestInvert;
    let lut = op.build_lut();

    // Verify LUT[i] = 255 - i
    for i in 0..256 {
        assert_eq!(lut[i], 255 - i as u8);
    }
}

// ===== LUT Fusion Tests =====

#[test]
fn test_fused_lut_executor_two_transforms() {
    let mut data = vec![200u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 1);

    // Apply: Invert -> Threshold(128)
    // 200 -> 55 (via Invert: 255-200=55)
    // 55 -> 55 (via Threshold: 55 < 128, unchanged)
    let ops: Vec<Box<dyn LutOp>> = vec![
        Box::new(TestInvert),
        Box::new(TestThreshold { threshold: 128 }),
    ];

    FusedLutExecutor::execute(&mut img, &ops);

    assert_eq!(img.data[0], 55);
}

#[test]
fn test_fused_lut_executor_three_transforms() {
    let mut data = vec![200u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 1);

    // Apply: Threshold(100) -> Invert -> Threshold(128)
    // 200 -> 55 (via Threshold: 200 >= 100, so 255-200=55)
    // 55 -> 200 (via Invert: 255-55=200)
    // 200 -> 55 (via Threshold: 200 >= 128, so 255-200=55)
    let ops: Vec<Box<dyn LutOp>> = vec![
        Box::new(TestThreshold { threshold: 100 }),
        Box::new(TestInvert),
        Box::new(TestThreshold { threshold: 128 }),
    ];

    FusedLutExecutor::execute(&mut img, &ops);

    assert_eq!(img.data[0], 55);
}

#[test]
fn test_fused_lut_vs_separate_application() {
    // Verify that fused LUT gives same result as separate applications
    let mut data1 = vec![0u8, 64u8, 127u8, 128u8, 191u8, 255u8];
    let mut data2 = data1.clone();

    // Apply transforms separately
    let mut img1 = FusableImage::new(&mut data1, 6, 1, 1);
    TestThreshold { threshold: 128 }.execute_with_lut(&mut img1);
    TestInvert.execute_with_lut(&mut img1);

    // Apply transforms fused
    let mut img2 = FusableImage::new(&mut data2, 6, 1, 1);
    let ops: Vec<Box<dyn LutOp>> = vec![
        Box::new(TestThreshold { threshold: 128 }),
        Box::new(TestInvert),
    ];
    FusedLutExecutor::execute(&mut img2, &ops);

    // Results should be identical
    assert_eq!(img1.data, img2.data);
}

#[test]
fn test_fused_lut_compose_correctness() {
    // Verify LUT composition directly
    let threshold = TestThreshold { threshold: 128 };
    let invert = TestInvert;

    let lut_threshold = threshold.build_lut();
    let lut_invert = invert.build_lut();

    // Compose: invert(threshold(i))
    let mut composed = [0u8; 256];
    for i in 0..=255u8 {
        composed[i as usize] = lut_invert[lut_threshold[i as usize] as usize];
    }

    // Check a few values manually
    // i=200: threshold gives 55 (200>=128, 255-200=55), invert gives 200 (255-55=200)
    assert_eq!(composed[200], 200);
    // i=50: threshold gives 50 (50<128), invert gives 205 (255-50=205)
    assert_eq!(composed[50], 205);
}

#[test]
fn test_fused_lut_empty_ops() {
    let mut data = vec![128u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 1);

    let ops: Vec<Box<dyn LutOp>> = vec![];
    FusedLutExecutor::execute(&mut img, &ops);

    // Should be unchanged
    assert_eq!(img.data[0], 128);
}

#[test]
fn test_fused_lut_single_op() {
    let mut data = vec![200u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 1);

    let ops: Vec<Box<dyn LutOp>> = vec![Box::new(TestInvert)];
    FusedLutExecutor::execute(&mut img, &ops);

    assert_eq!(img.data[0], 55);
}

#[test]
fn test_fused_lut_rgb_image() {
    let mut data = vec![200u8, 100u8, 50u8];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    // Apply Invert to all channels
    let ops: Vec<Box<dyn LutOp>> = vec![Box::new(TestInvert)];
    FusedLutExecutor::execute(&mut img, &ops);

    assert_eq!(img.data[0], 55);  // 200 -> 55
    assert_eq!(img.data[1], 155); // 100 -> 155
    assert_eq!(img.data[2], 205); // 50 -> 205
}

#[test]
fn test_fused_lut_large_image() {
    let mut data = vec![200u8; 256];
    let mut img = FusableImage::new(&mut data, 16, 16, 1);

    let ops: Vec<Box<dyn LutOp>> = vec![
        Box::new(TestThreshold { threshold: 128 }),
        Box::new(TestInvert),
    ];

    FusedLutExecutor::execute(&mut img, &ops);

    // All pixels should be 200:
    // 200 >= 128 -> 255-200=55 -> 255-55=200
    assert!(img.data.iter().all(|&p| p == 200));
}

// ===== FusedLut Transform Tests =====

#[test]
fn test_fused_lut_from_ops() {
    let ops: Vec<Box<dyn LutOp>> = vec![
        Box::new(TestThreshold { threshold: 128 }),
        Box::new(TestInvert),
    ];

    let fused = FusedLut::from_ops(&ops);

    // Verify the LUT is composed correctly
    // i=200: threshold gives 55 (200>=128, 255-200=55), invert gives 200 (255-55=200)
    assert_eq!(fused.lut[200], 200);
    // i=50: threshold gives 50 (50<128), invert gives 205 (255-50=205)
    assert_eq!(fused.lut[50], 205);
}

#[test]
fn test_fused_lut_identity_detection() {
    // An identity LUT should be detected
    let mut identity_lut = [0u8; 256];
    for i in 0..256 {
        identity_lut[i] = i as u8;
    }

    let fused = FusedLut::new(identity_lut);
    assert!(fused.is_identity());
}

#[test]
fn test_fused_lut_non_identity() {
    // A non-identity LUT should not be detected as identity
    let mut lut = [0u8; 256];
    for i in 0..256 {
        lut[i] = 255 - i as u8; // Invert
    }

    let fused = FusedLut::new(lut);
    assert!(!fused.is_identity());
}

#[test]
fn test_fused_lut_execute() {
    let mut data = vec![100u8, 200u8];
    let mut img = FusableImage::new(&mut data, 2, 1, 1);

    // Create a FusedLut that inverts values
    let mut lut = [0u8; 256];
    for i in 0..256 {
        lut[i] = 255 - i as u8;
    }
    let fused = FusedLut::new(lut);

    fused.execute(&mut img);

    assert_eq!(img.data[0], 155); // 255 - 100
    assert_eq!(img.data[1], 55);  // 255 - 200
}

#[test]
fn test_fused_lut_build_lut() {
    // FusedLut's build_lut should return itself
    let mut lut = [0u8; 256];
    for i in 0..256 {
        lut[i] = (i as u8).wrapping_add(1);
    }
    let fused = FusedLut::new(lut);

    let result = fused.build_lut();
    assert_eq!(result, lut);
}
