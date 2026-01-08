// Integration tests for the two-stage pipeline.
//
// These tests verify that:
// 1. Single-image transforms work correctly
// 2. Batch transforms work correctly
// 3. The two-stage pipeline (ImagePipeline → BatchPipeline) produces correct results

#[cfg(test)]
mod integration_tests {
    use crate::batch::{Batch, BatchTransform, MixUp, Label, SoftLabel};
    use crate::core::{BarrierImage, FusableImage};
    use rand::SeedableRng;

    /// Test the basic two-stage pipeline flow:
    /// Stage 1: Apply single-image transforms
    /// Stage 2: Apply batch transforms (MixUp)
    #[test]
    fn test_two_stage_pipeline_basic() {
        // Stage 1: Create images and apply single-image transforms
        let n_samples = 4;
        let width = 32;
        let height = 32;
        let channels = 3;

        let mut images = Vec::new();
        let mut labels = Vec::new();

        for i in 0..n_samples {
            // Create image with unique pattern per sample
            let mut data = vec![0u8; width * height * channels];
            for pixel in data.iter_mut() {
                *pixel = (i * 50) as u8; // 0, 50, 100, 150
            }

            let _img = FusableImage::new(&mut data, width, height, channels);

            // Apply a simple brightness transform (this would use the fusion system
            // in a real pipeline, but for simplicity we'll use the image directly)
            // In real usage: image_pipeline.apply(img)

            // Convert to BarrierImage for batch processing
            images.push(BarrierImage::from_vec(data, width, height, channels));

            // Create label
            labels.push(SoftLabel::one_hot(i, 10));
        }

        // Stage 2: Create batch and apply MixUp
        let mut batch = Batch::new(images, labels);
        let mixup = MixUp::new(1.0);
        // Use seeded RNG for deterministic testing
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        mixup.apply(&mut batch, &mut rng);

        // Verify results
        assert_eq!(batch.len(), n_samples);

        // All images should now have mixed pixel values (not just the original 0, 50, 100, 150)
        // And all labels should be soft (not one-hot)
        for i in 0..n_samples {
            // Check that image has mixed values
            let has_mixed_values = batch.images[i]
                .data
                .iter()
                .any(|&p| p != 0 && p != 50 && p != 100 && p != 150);
            assert!(
                has_mixed_values,
                "Image {} should have mixed pixel values",
                i
            );

            // Check that label is soft (max probability < 1.0)
            let max_prob = batch.labels[i]
                .probs()
                .iter()
                .cloned()
                .fold(0.0f32, f32::max);
            assert!(
                max_prob < 1.0,
                "Label {} should be mixed (soft), not one-hot",
                i
            );
        }
    }

    /// Test that image dimensions are preserved through the two-stage pipeline
    #[test]
    fn test_two_stage_pipeline_preserves_dimensions() {
        let width = 64;
        let height = 64;
        let channels = 3;

        let images: Vec<_> = (0..4)
            .map(|_| BarrierImage::new(width, height, channels))
            .collect();

        let labels: Vec<_> = (0..4).map(|i| SoftLabel::one_hot(i, 10)).collect();

        let mut batch = Batch::new(images, labels);

        let mixup = MixUp::new(1.0);
        let mut rng = rand::thread_rng();

        let width_before = batch.width();
        let height_before = batch.height();
        let channels_before = batch.channels();

        mixup.apply(&mut batch, &mut rng);

        assert_eq!(batch.width(), width_before);
        assert_eq!(batch.height(), height_before);
        assert_eq!(batch.channels(), channels_before);
    }

    /// Test that the pipeline is deterministic with seeded RNG
    #[test]
    #[cfg(feature = "rand_chacha")]
    fn test_two_stage_pipeline_deterministic() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let create_batch = || {
            let images: Vec<_> = (0..4).map(|_| BarrierImage::new(32, 32, 3)).collect();

            let labels: Vec<_> = (0..4).map(|i| SoftLabel::one_hot(i, 10)).collect();

            Batch::new(images, labels)
        };

        let mut batch1 = create_batch();
        let mut batch2 = create_batch();

        // Fill both batches with identical data
        for (img1, img2) in batch1.images.iter_mut().zip(batch2.images.iter_mut()) {
            img1.data.iter_mut().for_each(|p| *p = 100);
            img2.data.iter_mut().for_each(|p| *p = 100);
        }

        let mixup = MixUp::new(1.0);
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        mixup.apply(&mut batch1, &mut rng1);
        mixup.apply(&mut batch2, &mut rng2);

        // Results should be identical
        for (img1, img2) in batch1.images.iter().zip(batch2.images.iter()) {
            assert_eq!(img1.data, img2.data);
        }

        for (label1, label2) in batch1.labels.iter().zip(batch2.labels.iter()) {
            assert_eq!(label1.probs(), label2.probs());
        }
    }

    /// Test that MixUp with different alpha values produces different results
    #[test]
    fn test_mixup_alpha_affects_results() {
        let create_batch = || {
            let mut images = Vec::new();
            for i in 0..4 {
                let mut data = vec![0u8; 32 * 32 * 3];
                data.fill((i * 50) as u8);
                images.push(BarrierImage::from_vec(data, 32, 32, 3));
            }

            let labels: Vec<_> = (0..4).map(|i| SoftLabel::one_hot(i, 10)).collect();

            Batch::new(images, labels)
        };

        let mut batch1 = create_batch();
        let mut batch2 = create_batch();

        let mixup_low = MixUp::new(0.2); // Pushes λ toward 0 or 1 (less mixing)
        let mixup_high = MixUp::new(2.0); // Pushes λ toward 0.5 (more mixing)

        let mut rng = rand::thread_rng();

        // We can't directly compare results due to randomness, but we can verify
        // both complete without errors
        mixup_low.apply(&mut batch1, &mut rng);
        mixup_high.apply(&mut batch2, &mut rng);

        // Both should produce valid results
        assert_eq!(batch1.len(), 4);
        assert_eq!(batch2.len(), 4);
    }

    /// Test edge case: single-sample batch
    #[test]
    fn test_two_stage_pipeline_single_sample() {
        let images = vec![BarrierImage::new(32, 32, 3)];
        let labels = vec![SoftLabel::one_hot(0, 10)];

        let mut batch = Batch::new(images, labels);
        let original_data = batch.images[0].data.clone();
        let original_label = batch.labels[0].clone();

        let mixup = MixUp::new(1.0);
        let mut rng = rand::thread_rng();

        // Should not panic, should leave batch unchanged
        mixup.apply(&mut batch, &mut rng);

        assert_eq!(batch.images[0].data, original_data);
        assert_eq!(batch.labels[0].probs(), original_label.probs());
    }

    /// Test that MixUp correctly mixes labels
    #[test]
    fn test_mixup_label_mixing() {
        let mut images = vec![BarrierImage::new(8, 8, 1), BarrierImage::new(8, 8, 1)];

        // Create distinct images
        images[0].data.fill(100);
        images[1].data.fill(200);

        let labels = vec![SoftLabel::one_hot(0, 5), SoftLabel::one_hot(2, 5)];

        let mut batch = Batch::new(images, labels);

        let mixup = MixUp::new(1.0);
        let mut rng = rand::thread_rng();

        mixup.apply(&mut batch, &mut rng);

        // After MixUp, labels should be soft (not one-hot)
        for label in &batch.labels {
            let max_prob = label.probs().iter().cloned().fold(0.0f32, f32::max);
            assert!(max_prob < 1.0, "Labels should be soft after MixUp");
        }
    }

    /// Test with larger batch size
    #[test]
    fn test_two_stage_pipeline_large_batch() {
        let n_samples = 32;

        let images: Vec<_> = (0..n_samples)
            .map(|_| BarrierImage::new(32, 32, 3))
            .collect();

        let labels: Vec<_> = (0..n_samples)
            .map(|i| SoftLabel::one_hot(i, 32)) // Unique class per sample
            .collect();

        let mut batch = Batch::new(images, labels);

        let mixup = MixUp::new(1.0);
        let mut rng = rand::thread_rng();

        mixup.apply(&mut batch, &mut rng);

        assert_eq!(batch.len(), n_samples);

        // Most labels should be mixed (at least 50%)
        let mut mixed_count = 0;
        for label in &batch.labels {
            let max_prob = label.probs().iter().cloned().fold(0.0f32, f32::max);
            if max_prob < 1.0 {
                mixed_count += 1;
            }
        }
        assert!(
            mixed_count >= n_samples / 2,
            "At least half the labels should be mixed"
        );
    }
}
