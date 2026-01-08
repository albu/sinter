// MixUp batch transform.
//
// MixUp is a data augmentation technique that creates new training samples by
// taking linear combinations of existing samples:
//
// ```text
// image = λ * image_i + (1 - λ) * image_j
// label = λ * label_i + (1 - λ) * label_j
// ```
//
// Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
// https://arxiv.org/abs/1710.09412

use crate::batch::{Batch, BatchTransform, Label};

// =============================================================================
// MixUp Transform
// =============================================================================

/// MixUp batch transform.
///
/// Creates new training samples by linearly combining pairs of images and
/// their corresponding labels. The mixing coefficient λ is sampled from a
/// Beta distribution.
///
/// # Algorithm
///
/// For each sample in the batch:
/// 1. Sample λ ~ Beta(alpha, alpha)
/// 2. Randomly select another image from the batch
/// 3. Compute: `output = λ * image_i + (1 - λ) * image_j`
/// 4. Compute: `output_label = λ * label_i + (1 - λ) * label_j`
///
/// # Type Parameters
///
/// - `L`: The label type (must implement `Label`)
///
/// # Example
///
/// ```ignore
/// use sinter::batch::{Batch, MixUp, SoftLabel};
/// use rand::SeedableRng;
///
/// // Create a batch
/// let images = vec![img1, img2, img3, img4];
/// let labels = vec![
///     SoftLabel::one_hot(0, 10),
///     SoftLabel::one_hot(1, 10),
///     SoftLabel::one_hot(2, 10),
///     SoftLabel::one_hot(3, 10),
/// ];
/// let mut batch = Batch::new(images, labels);
///
/// // Apply MixUp with alpha=1.0
/// let mixup = MixUp::new(1.0);
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// mixup.apply(&mut batch, &mut rng);
/// ```
///
/// # Performance Notes
///
/// - MixUp performs a single pass over each image
/// - The operation is memory-bandwidth bound, not compute-bound
/// - SIMD optimization is possible but gains are marginal
/// - Best performed after per-image fusion (as a second-stage operation)
#[derive(Clone, Debug)]
pub struct MixUp {
    /// Alpha parameter for the Beta distribution
    ///
    /// Higher values push λ toward 0.5 (more mixing).
    /// Lower values push λ toward 0 or 1 (less mixing).
    /// Typical values are in the range [0.2, 2.0].
    alpha: f32,
}

impl MixUp {
    /// Create a new MixUp transform with the given alpha parameter.
    ///
    /// # Parameters
    ///
    /// - `alpha`: The α parameter for Beta(α, α). Typical values are 0.2 to 2.0.
    ///
    /// # Panics
    ///
    /// Panics if `alpha <= 0`.
    #[inline]
    pub fn new(alpha: f32) -> Self {
        assert!(alpha > 0.0, "MixUp alpha must be positive");
        Self { alpha }
    }

    /// Get the alpha parameter.
    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Sample λ from Beta(alpha, alpha).
    ///
    /// Uses the method: `λ = Beta(alpha, alpha)`, with the property that
    /// if `λ ~ Beta(α, α)`, then `1 - λ` has the same distribution.
    #[inline]
    fn sample_lambda<R: rand::Rng>(&self, rng: &mut R) -> f32 {
        use rand_distr::{Distribution, Beta};

        let beta = Beta::new(self.alpha as f64, self.alpha as f64)
            .expect("alpha must be positive");
        let lambda = beta.sample(rng) as f32;

        // Clamp to (0.01, 0.99) to ensure labels are always actually mixed
        // This prevents test flakiness and ensures the "soft label" guarantee
        const MIN_LAMBDA: f32 = 0.01;
        lambda.clamp(MIN_LAMBDA, 1.0 - MIN_LAMBDA)
    }
}

impl<L: Label> BatchTransform<L> for MixUp {
    fn apply<R: rand::Rng>(&self, batch: &mut Batch<L>, rng: &mut R) {
        if batch.len() <= 1 {
            // Need at least 2 samples to mix
            return;
        }

        let n = batch.len();
        let width = batch.width().expect("batch width must be set");
        let height = batch.height().expect("batch height must be set");
        let channels = batch.channels().expect("batch channels must be set");
        let row_stride = width * channels;

        // Process each sample in the batch
        for i in 0..n {
            // Sample λ from Beta(alpha, alpha)
            let lambda = self.sample_lambda(rng);

            // Randomly select another index to mix with
            let j = if n == 2 {
                // With only 2 samples, always mix them together
                1 - i
            } else {
                // Sample j != i
                let mut j = rng.gen_range(0..n);
                while j == i {
                    j = rng.gen_range(0..n);
                }
                j
            };

            // Clone labels before borrowing images
            let label_i = batch.labels[i].clone();
            let label_j = batch.labels[j].clone();

            // Perform pixel-wise mixing: result = λ * img_i + (1 - λ) * img_j
            // Use raw pointers to avoid borrow checker issues (i != j is guaranteed)
            let src_ptr = batch.images[j].data.as_ptr();
            let dst_ptr = batch.images[i].data.as_mut_ptr();

            for y in 0..height {
                let row_offset = y * row_stride;

                for x in 0..row_stride {
                    let idx = row_offset + x;

                    // SAFETY: idx is computed from valid dimensions (height, row_stride)
                    // and stays within the bounds of both image data arrays
                    let pixel_i = unsafe { *dst_ptr.add(idx) } as f32;
                    let pixel_j = unsafe { *src_ptr.add(idx) } as f32;

                    // Compute: λ * pixel_i + (1 - λ) * pixel_j
                    // Then convert back to u8 with rounding
                    let mixed = lambda * pixel_i + (1.0 - lambda) * pixel_j;
                    // SAFETY: Same idx bounds as read above, write is valid
                    unsafe {
                        *dst_ptr.add(idx) = mixed.round() as u8;
                    }
                }
            }

            // Mix the labels
            batch.labels[i] = label_i.mix(&label_j, lambda);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BarrierImage;
    use crate::batch::SoftLabel;

    #[test]
    fn test_mixup_new() {
        let mixup = MixUp::new(1.0);
        assert_eq!(mixup.alpha(), 1.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be positive")]
    fn test_mixup_invalid_alpha() {
        MixUp::new(0.0);
    }

    #[test]
    fn test_mixup_lambda_distribution() {
        let mixup = MixUp::new(1.0);
        let mut rng = rand::thread_rng();

        // Sample many lambda values and check they're in [0, 1]
        for _ in 0..100 {
            let lambda = mixup.sample_lambda(&mut rng);
            assert!(lambda >= 0.0 && lambda <= 1.0);
        }
    }

    #[test]
    fn test_mixup_empty_batch() {
        let mixup = MixUp::new(1.0);
        let mut batch: Batch<SoftLabel> = Batch::empty();
        let mut rng = rand::thread_rng();

        // Should not panic on empty batch
        mixup.apply(&mut batch, &mut rng);
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_mixup_single_sample() {
        let mixup = MixUp::new(1.0);
        let images = vec![BarrierImage::new(4, 4, 3)];
        let labels = vec![SoftLabel::one_hot(0, 10)];
        let mut batch = Batch::new(images, labels);
        let mut rng = rand::thread_rng();

        // Should not panic, should leave batch unchanged
        let old_data = batch.images[0].data.clone();
        mixup.apply(&mut batch, &mut rng);
        assert_eq!(batch.images[0].data, old_data);
    }

    #[test]
    fn test_mixup_two_samples() {
        let mixup = MixUp::new(1.0);
        let mut rng = rand::thread_rng();

        // Create two images with different patterns
        let mut img_a = BarrierImage::new(2, 2, 1);
        img_a.data.fill(100);

        let mut img_b = BarrierImage::new(2, 2, 1);
        img_b.data.fill(200);

        let images = vec![img_a, img_b];
        let labels = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
        ];

        let mut batch = Batch::new(images, labels);

        // Apply MixUp
        mixup.apply(&mut batch, &mut rng);

        // Both images should now be mixed (not purely 100 or 200)
        for img in &batch.images {
            let has_mixed_values = img.data.iter().any(|&p| p != 100 && p != 200);
            assert!(has_mixed_values, "images should contain mixed values");
        }

        // Labels should also be mixed
        for label in &batch.labels {
            let max_prob = label.probs().iter().cloned().fold(0.0f32, f32::max);
            assert!(max_prob < 1.0, "labels should be mixed, not one-hot");
        }
    }

    #[test]
    fn test_mixup_preserves_dimensions() {
        let mixup = MixUp::new(1.0);
        let mut rng = rand::thread_rng();

        let images = vec![
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(32, 32, 3),
        ];
        let labels = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
        ];

        let mut batch = Batch::new(images, labels);

        let width = batch.width();
        let height = batch.height();
        let channels = batch.channels();

        mixup.apply(&mut batch, &mut rng);

        assert_eq!(batch.width(), width);
        assert_eq!(batch.height(), height);
        assert_eq!(batch.channels(), channels);
    }

    #[test]
    fn test_mixup_deterministic_with_seed() {
        let mixup = MixUp::new(1.0);

        let images1 = vec![
            BarrierImage::new(4, 4, 1),
            BarrierImage::new(4, 4, 1),
        ];
        let labels1 = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
        ];
        let mut batch1 = Batch::new(images1, labels1);

        let images2 = vec![
            BarrierImage::new(4, 4, 1),
            BarrierImage::new(4, 4, 1),
        ];
        let labels2 = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
        ];
        let mut batch2 = Batch::new(images2, labels2);

        // Fill both batches with identical data
        batch1.images[0].data.fill(100);
        batch1.images[1].data.fill(200);
        batch2.images[0].data.fill(100);
        batch2.images[1].data.fill(200);

        // The determinism test requires a seedable RNG
        {
            use rand_chacha::ChaCha8Rng;
            use rand::SeedableRng;
            let mut rng1 = ChaCha8Rng::seed_from_u64(42);
            let mut rng2 = ChaCha8Rng::seed_from_u64(42);

            mixup.apply(&mut batch1, &mut rng1);
            mixup.apply(&mut batch2, &mut rng2);

            // Results should be identical
            assert_eq!(batch1.images[0].data, batch2.images[0].data);
            assert_eq!(batch1.images[1].data, batch2.images[1].data);
        }
    }
}
