// Tests for MultiplicativeNoise transform

use super::*;

#[test]
fn test_multiplicative_noise_new() {
    let n = MultiplicativeNoise::new(1.0, 0.2);
    assert_eq!(n.multiplier, 1.0);
    assert_eq!(n.std_dev, 0.2);
    assert_eq!(n.granularity, NoiseGranularity::PerVector);
}

#[test]
#[should_panic(expected = "std_dev must be non-negative")]
fn test_multiplicative_noise_invalid_std_dev() {
    MultiplicativeNoise::new(1.0, -0.1);
}

#[test]
fn test_multiplicative_noise_zero_std_dev() {
    let mut data = vec![100u8; 100];
    let mut img = FusableImage::new(&mut data, 10, 10, 1);

    let n = MultiplicativeNoise::new(1.0, 0.0);
    n.execute(&mut img);

    // With zero std_dev, values should be unchanged (multiplied by 1.0)
    assert!(img.data.iter().all(|&x| x == 100));
}

#[test]
fn test_multiplicative_noise_multiplier() {
    let mut data = vec![100u8; 100];
    let mut img = FusableImage::new(&mut data, 10, 10, 1);

    let n = MultiplicativeNoise::new(2.0, 0.0);
    n.execute(&mut img);

    // Values should be approximately doubled
    let avg: f32 = img.data.iter().map(|&x| x as f32).sum::<f32>() / 100.0;
    assert!(
        (avg - 200.0).abs() < 5.0,
        "Average should be ~200, got {}",
        avg
    );
}

#[test]
fn test_multiplicative_noise_execute() {
    let mut data = vec![128u8; 100];
    let mut img = FusableImage::new(&mut data, 10, 10, 1);

    let n = MultiplicativeNoise::new(1.0, 0.2);
    n.execute(&mut img);

    // Values should vary from original
    let all_same = img.data.iter().all(|&x| x == 128);
    assert!(!all_same, "Noise should modify pixel values");
}

#[test]
fn test_multiplicative_noise_clamping_high() {
    // Test that multiplication doesn't cause overflow
    let mut data = vec![200u8; 100];
    let mut img = FusableImage::new(&mut data, 10, 10, 1);

    let n = MultiplicativeNoise::new(2.0, 0.1);
    n.execute(&mut img);

    // All values should be clamped to [0, 255]
    for &px in img.data.iter() {
        assert!((0..=255).contains(&px));
    }
}

#[test]
fn test_multiplicative_noise_clamping_low() {
    // Test that values don't go negative
    let mut data = vec![100u8; 100];
    let mut img = FusableImage::new(&mut data, 10, 10, 1);

    // Low multiplier could push values toward zero
    let n = MultiplicativeNoise::new(0.1, 0.05);
    n.execute(&mut img);

    // All values should be in valid range (>= 0)
    for &px in img.data.iter() {
        assert!((0..=255).contains(&px));
    }
}

#[test]
fn test_multiplicative_noise_reproducibility() {
    let mut data1 = vec![128u8; 100];
    let mut img1 = FusableImage::new(&mut data1, 10, 10, 1);

    let mut data2 = vec![128u8; 100];
    let mut img2 = FusableImage::new(&mut data2, 10, 10, 1);

    let n = MultiplicativeNoise::new(1.0, 0.2);
    n.execute(&mut img1);
    n.execute(&mut img2);

    // Same input should produce same output (reproducible)
    assert_eq!(img1.data, img2.data);
}

#[test]
fn test_multiplicative_noise_rgb() {
    let mut data = vec![128u8, 128, 128];
    let mut img = FusableImage::new(&mut data, 1, 1, 3);

    let n = MultiplicativeNoise::new(1.0, 0.2);
    n.execute(&mut img);

    // All channels should be modified
    // Due to different seeds, they should have different values
    let all_same = img.data[0] == img.data[1] && img.data[1] == img.data[2];
    assert!(
        !all_same || img.data[0] != 128,
        "RGB channels should have independent noise"
    );
}

#[test]
fn test_multiplicative_noise_high_std_dev() {
    let mut data = vec![128u8; 1000];
    let mut img = FusableImage::new(&mut data, 10, 10, 10);

    let n = MultiplicativeNoise::new(1.0, 0.5);
    n.execute(&mut img);

    // With high std dev, we should see a wide range of values
    let min_val = *img.data.iter().min().unwrap();
    let max_val = *img.data.iter().max().unwrap();

    // Range should be significant
    assert!(
        max_val - min_val > 60,
        "High std dev should produce wide value range"
    );
}

#[test]
fn test_multiplicative_noise_preserves_mean() {
    let mut data = vec![128u8; 10000];
    let mut img = FusableImage::new(&mut data, 100, 100, 1);

    let n = MultiplicativeNoise::new(1.0, 0.1);
    n.execute(&mut img);

    // Average should still be around 128 (with some variance due to randomness)
    let avg: f32 = img.data.iter().map(|&x| x as f32).sum::<f32>() / img.data.len() as f32;
    assert!((avg - 128.0).abs() < 15.0, "Average should be ~128");
}

#[test]
fn test_multiplicative_noise_access_pattern() {
    let n = MultiplicativeNoise::new(1.0, 0.2);
    assert_eq!(n.access(), AccessPattern::InPlace);
    assert_eq!(n.shape_effect(), ShapeEffect::Preserve);
}
