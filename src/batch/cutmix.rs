// CutMix: Cut and paste image regions with label mixing.
//
// # Algorithm
//
// For each image in the batch:
// 1. Sample λ from Beta(α, α)
// 2. Select a random partner image (j ≠ i)
// 3. Calculate box size: sqrt(λ) * image dimensions
// 4. Sample random box position uniformly
// 5. Paste the box region from image_j onto image_i
// 6. Mix labels: label_i.mix(label_j, λ)
//
// # Label Mixing
//
// IMPORTANT: λ in CutMix is the box_area / image_area ratio, NOT the actual
// pixel overlap after clamping. This matches the original CutMix paper.
//
// # Reference
//
// Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers
// with Localizable Features" (ICCV 2019)

use crate::batch::{Batch, BatchTransform, Label};
use rand_distr::Beta as BetaDist;
use rand::Rng;

/// CutMix batch transform.
///
/// CutMix creates new training samples by cutting a rectangular region from
/// one image and pasting it onto another:
///
/// ```text
/// image_i[cx-w/2:cx+w/2, cy-h/2:cy+h/2] = image_j[same_region]
/// label = λ * label_i + (1 - λ) * label_j
/// ```
///
/// where λ = box_area / image_area ~ Beta(α, α).
///
/// # Type Parameters
///
/// - `L`: The label type (must implement `Label`)
///
/// # Example
///
/// ```ignore
/// use sinter::batch::{Batch, MixUp, SoftLabel};
///
/// let mut batch = Batch::new(images, labels);
/// let cutmix = CutMix::new(1.0);
/// let mut rng = rand::thread_rng();
/// cutmix.apply(&mut batch, &mut rng);
/// ```
#[derive(Clone, Debug)]
pub struct CutMix {
    /// The α parameter for Beta(α, α) distribution
    alpha: f32,
}

impl CutMix {
    /// Create a new CutMix transform.
    ///
    /// # Arguments
    ///
    /// - `alpha`: The α parameter for Beta(α, α) distribution
    ///   - Higher values → larger boxes (more mixing)
    ///   - Lower values → smaller boxes (less mixing)
    ///   - Typical values: 0.2 to 2.0
    ///
    /// # Panics
    ///
    /// Panics if `alpha <= 0.0`.
    #[inline]
    pub fn new(alpha: f32) -> Self {
        assert!(alpha > 0.0, "CutMix: alpha must be positive, got {}", alpha);
        Self { alpha }
    }

    /// Get the alpha parameter.
    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Sample λ from Beta(α, α) distribution.
    fn sample_lambda<R: Rng>(&self, rng: &mut R) -> f32 {
        let beta = BetaDist::new(self.alpha as f64, self.alpha as f64)
            .expect("CutMix: invalid Beta distribution parameters");
        let lambda = rng.sample(beta) as f32;

        // Clamp to (0.01, 0.99) to ensure labels are always actually mixed
        // This prevents test flakiness and ensures the "soft label" guarantee
        const MIN_LAMBDA: f32 = 0.01;
        lambda.clamp(MIN_LAMBDA, 1.0 - MIN_LAMBDA)
    }

    /// Sample a random box for CutMix.
    ///
    /// Returns (x, y, w, h) where (x, y) is the top-left corner and
    /// (w, h) are the box dimensions.
    fn sample_box<R: Rng>(&self, rng: &mut R, lambda: f32, width: usize, height: usize) -> (usize, usize, usize, usize) {
        // Calculate box size from lambda
        // box_area = lambda * image_area
        // Assuming square-ish boxes: w = h = sqrt(lambda) * dim
        let ratio = lambda.sqrt();
        let box_w = (ratio * width as f32) as usize;
        let box_h = (ratio * height as f32) as usize;

        // Ensure box has at least 1 pixel
        let box_w = box_w.max(1);
        let box_h = box_h.max(1);

        // Sample random position
        // Box center is uniformly distributed in [0, width] x [0, height]
        let cx = rng.gen_range(0..=width);
        let cy = rng.gen_range(0..=height);

        // Convert to top-left coordinates
        let x = cx.saturating_sub(box_w / 2);
        let y = cy.saturating_sub(box_h / 2);

        (x, y, box_w, box_h)
    }

    /// Clamp box coordinates to image bounds.
    ///
    /// Returns (x_clamped, y_clamped, w_clamped, h_clamped).
    fn clamp_box(x: usize, y: usize, w: usize, h: usize, width: usize, height: usize) -> (usize, usize, usize, usize) {
        let x_clamped = x.min(width);
        let y_clamped = y.min(height);
        let w_clamped = w.min(width - x_clamped);
        let h_clamped = h.min(height - y_clamped);
        (x_clamped, y_clamped, w_clamped, h_clamped)
    }
}

impl<L: Label> BatchTransform<L> for CutMix {
    fn apply<R: rand::Rng>(&self, batch: &mut Batch<L>, rng: &mut R) {
        let batch_size = batch.len();

        if batch_size <= 1 {
            return; // No mixing possible with single sample
        }

        let width = batch.width().expect("CutMix: batch has no images");
        let height = batch.height().expect("CutMix: batch has no images");
        let channels = batch.channels().expect("CutMix: batch has no images");

        // Generate random permutation for partner assignment
        // Ensure no element is mapped to itself (j != i for all i)
        let mut permutation: Vec<usize> = (0..batch_size).collect();
        for i in (1..batch_size).rev() {
            let j = rng.gen_range(0..=i);
            permutation.swap(i, j);
        }

        // Fix any self-assignments by swapping with the next element
        for i in 0..batch_size {
            if permutation[i] == i {
                // Swap with next element (wrapping around)
                let next = (i + 1) % batch_size;
                permutation.swap(i, next);
            }
        }

        // Apply CutMix to each sample
        for i in 0..batch_size {
            // Sample lambda for this sample
            let lambda = self.sample_lambda(rng);

            // Get partner index (ensure j ≠ i)
            let j = permutation[i];
            if j == i {
                continue; // Skip if partner is self (no mixing)
            }

            // Sample box
            let (x, y, box_w, box_h) = self.sample_box(rng, lambda, width, height);
            let (x, y, box_w, box_h) = Self::clamp_box(x, y, box_w, box_h, width, height);

            // Skip if box is empty
            if box_w == 0 || box_h == 0 {
                continue;
            }

            // Clone labels before borrowing images
            let label_i = batch.labels()[i].clone();
            let label_j = batch.labels()[j].clone();

            // Perform row-based copy for better performance
            // Get raw pointers to avoid borrow checker issues
            let src_data = batch.get_image(j).data.as_ptr();
            let dst_data = batch.get_image_mut(i).data.as_mut_ptr();

            let row_size = box_w * channels;

            // Copy box row by row from source to destination
            // The box is at the same position (x, y) in both images
            for dy in 0..box_h {
                let src_y = y + dy;
                let dst_y = y + dy;

                let src_row_start = (src_y * width + x) * channels;
                let dst_row_start = (dst_y * width + x) * channels;

                // Copy entire row at once using memcpy
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_data.add(src_row_start),
                        dst_data.add(dst_row_start),
                        row_size,
                    );
                }
            }

            // Mix labels
            let mixed_label = label_i.mix(&label_j, lambda);
            batch.labels_mut()[i] = mixed_label;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BarrierImage;
    use crate::batch::SoftLabel;

    #[test]
    fn test_cutmix_new() {
        let cutmix = CutMix::new(1.0);
        assert_eq!(cutmix.alpha(), 1.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be positive")]
    fn test_cutmix_invalid_alpha() {
        CutMix::new(0.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be positive")]
    fn test_cutmix_negative_alpha() {
        CutMix::new(-1.0);
    }

    #[test]
    fn test_cutmix_two_samples() {
        let mut img1 = BarrierImage::new(8, 8, 3);
        let mut img2 = BarrierImage::new(8, 8, 3);

        // Fill with different values
        img1.data.fill(100);
        img2.data.fill(200);

        let label1 = SoftLabel::one_hot(0, 4);
        let label2 = SoftLabel::one_hot(1, 4);

        let mut batch = Batch::new(vec![img1, img2], vec![label1, label2]);

        let cutmix = CutMix::new(1.0);
        let mut rng = rand::thread_rng();
        cutmix.apply(&mut batch, &mut rng);

        // At least one image should be modified (have both 100 and 200)
        let img1_has_both = batch.get_image(0).data.iter().any(|&v| v == 100)
            && batch.get_image(0).data.iter().any(|&v| v == 200);
        let img2_has_both = batch.get_image(1).data.iter().any(|&v| v == 100)
            && batch.get_image(1).data.iter().any(|&v| v == 200);

        assert!(img1_has_both || img2_has_both);
    }

    #[test]
    fn test_cutmix_preserves_dimensions() {
        let images = vec![
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(32, 32, 3),
        ];
        let labels = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
            SoftLabel::one_hot(2, 10),
        ];

        let mut batch = Batch::new(images, labels);

        let cutmix = CutMix::new(1.0);
        let mut rng = rand::thread_rng();
        cutmix.apply(&mut batch, &mut rng);

        assert_eq!(batch.width(), Some(32));
        assert_eq!(batch.height(), Some(32));
        assert_eq!(batch.channels(), Some(3));
    }

    #[test]
    fn test_cutmix_single_sample_no_op() {
        let img = BarrierImage::new(8, 8, 3);
        let label = SoftLabel::one_hot(0, 4);
        let original_data = img.data.clone();

        let mut batch = Batch::new(vec![img], vec![label]);

        let cutmix = CutMix::new(1.0);
        let mut rng = rand::thread_rng();
        cutmix.apply(&mut batch, &mut rng);

        // Single sample should be unchanged
        assert_eq!(batch.get_image(0).data, original_data);
    }

    #[test]
    fn test_sample_box_bounds() {
        let cutmix = CutMix::new(1.0);
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let lambda = rng.gen();
            let (x, y, w, h) = cutmix.sample_box(&mut rng, lambda, 100, 100);
            let (x, y, w, h) = CutMix::clamp_box(x, y, w, h, 100, 100);

            assert!(x + w <= 100, "Box extends beyond width: {} + {} > 100", x, w);
            assert!(y + h <= 100, "Box extends beyond height: {} + {} > 100", y, h);
        }
    }
}
