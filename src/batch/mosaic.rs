// Mosaic: Stitch 4 images into a single grid image.
//
// # Algorithm
//
// For each group of 4 images in the batch:
// 1. Divide the output image into 4 quadrants (2x2 grid)
// 2. Resize each source image to fit its quadrant
// 3. Stitch them together into a single output image
// 4. Concatenate the 4 label vectors
//
// # Batch Processing
//
// Mosaic processes groups of 4 consecutive images:
// - Input batch size must be divisible by 4
// - Output batch size is N/4 (each group becomes 1 mosaic)
// - If N is not divisible by 4, remaining samples are passed through unchanged
//
// # Label Handling
//
// Unlike MixUp/CutMix which mix labels, Mosaic concatenates them:
// - Output label has 4x the dimensions (for multi-label classification)
// - For single-label: becomes a [1, 1, 1, 1, 0, 0, 0, 0] style multi-hot vector
//
// # Reference
//
// YOLOv4: "Mosaic data augmentation" - combines 4 training images into one
// for improved detection of small objects and batch normalization statistics.

use crate::batch::{Batch, BatchTransform, Label};
use crate::core::BarrierImage;
use rand::Rng;

/// Mosaic batch transform.
///
/// Mosaic creates new training samples by stitching 4 images together
/// in a 2x2 grid layout:
///
/// ```text
/// ┌─────────┬─────────┐
/// │  img0   │  img1   │
/// │ (top-L) │ (top-R) │
/// ├─────────┼─────────┤
/// │  img2   │  img3   │
/// │ (btm-L) │ (btm-R) │
/// └─────────┴─────────┘
/// ```
///
/// Each source image is resized to half the output dimensions.
///
/// # Type Parameters
///
/// - `L`: The label type (must implement `Label`)
///
/// # Example
///
/// ```ignore
/// use sinter::batch::{Batch, Mosaic, SoftLabel};
///
/// let mut batch = Batch::new(images, labels);  // Must have N % 4 == 0
/// let mosaic = Mosaic::new();
/// let mut rng = rand::thread_rng();
/// mosaic.apply(&mut batch, &mut rng);
///
/// // Output batch has N/4 images, each with 4x dimension labels
/// ```
#[derive(Clone, Debug)]
pub struct Mosaic;

impl Mosaic {
    /// Create a new Mosaic transform.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Get the number of images that get combined into one output.
    #[inline]
    pub fn group_size(&self) -> usize {
        4
    }

    /// Process a group of 4 images into 1 mosaic image.
    fn process_group<L: Label>(
        &self,
        batch: &mut Batch<L>,
        group_idx: usize,
        width: usize,
        height: usize,
        channels: usize,
    ) where
        L: Clone,
    {
        let indices = [group_idx * 4, group_idx * 4 + 1, group_idx * 4 + 2, group_idx * 4 + 3];

        // Clone labels first (before borrowing images)
        let labels: Vec<L> = indices.iter().map(|&i| batch.labels()[i].clone()).collect();
        let concatenated_label = L::concatenate(&labels);

        // Get raw pointers to source images
        let src_ptrs: Vec<*const u8> = indices
            .iter()
            .map(|&i| batch.get_image(i).data.as_ptr())
            .collect();

        // Create output image (same size as input images)
        let half_w = width / 2;
        let half_h = height / 2;
        let mut output_data = vec![0u8; width * height * channels];

        // Define quadrant positions
        let quadrants = [
            (0, 0),           // Top-left
            (half_w, 0),      // Top-right
            (0, half_h),      // Bottom-left
            (half_w, half_h), // Bottom-right
        ];

        // Copy and resize each source image to its quadrant
        // Use row-based copying for better performance
        for (q_idx, &src_ptr) in src_ptrs.iter().enumerate() {
            let (dst_x, dst_y) = quadrants[q_idx];
            let row_stride_src = width * channels;
            let row_stride_dst = width * channels;
            let row_stride_quadrant = half_w * channels;

            // Nearest-neighbor resize with 2x downsampling
            for dy in 0..half_h {
                let src_y = dy * 2; // Scale by 2
                let dst_y = dst_y + dy;

                for dx in 0..half_w {
                    let src_x = dx * 2; // Scale by 2

                    // Copy entire pixel row (all channels) at once
                    let src_idx = (src_y * row_stride_src) + (src_x * channels);
                    let dst_idx = (dst_y * row_stride_dst) + ((dst_x + dx) * channels);

                    unsafe {
                        // Copy all channels for this pixel
                        for c in 0..channels {
                            *output_data.get_unchecked_mut(dst_idx + c) =
                                *src_ptr.add(src_idx + c);
                        }
                    }
                }
            }
        }

        // Update the first image in the group with the mosaic result
        let out_img = batch.get_image_mut(group_idx * 4);
        out_img.data.clear();
        out_img.data.extend_from_slice(&output_data);
        batch.labels_mut()[group_idx * 4] = concatenated_label;

        // Mark the other 3 images for removal (set to empty/zero)
        // They'll be handled by retain logic after processing
        for idx in &indices[1..] {
            let img = batch.get_image_mut(*idx);
            img.data.fill(0); // Clear the image
        }
    }
}

impl<L: Label> BatchTransform<L> for Mosaic
where
    L: Clone,
{
    fn apply<R: rand::Rng>(&self, batch: &mut Batch<L>, _rng: &mut R) {
        let batch_size = batch.len();

        if batch_size < 4 {
            return; // Need at least 4 images
        }

        let width = batch.width().expect("Mosaic: batch has no images");
        let height = batch.height().expect("Mosaic: batch has no images");
        let channels = batch.channels().expect("Mosaic: batch has no images");

        let num_groups = batch_size / 4;

        // Process each group of 4
        for group_idx in 0..num_groups {
            self.process_group(batch, group_idx, width, height, channels);
        }

        // Efficient batch compaction: keep only indices 0, 4, 8, 12, ...
        // This is O(N) instead of O(N²) from multiple remove() calls
        let keep_indices: Vec<usize> = (0..num_groups).map(|g| g * 4).collect();

        let mut new_images = Vec::with_capacity(num_groups);
        let mut new_labels = Vec::with_capacity(num_groups);

        for &idx in &keep_indices {
            new_images.push(batch.images[idx].clone());
            new_labels.push(batch.labels[idx].clone());
        }

        batch.images = new_images;
        batch.labels = new_labels;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::SoftLabel;

    #[test]
    fn test_mosaic_new() {
        let mosaic = Mosaic::new();
        assert_eq!(mosaic.group_size(), 4);
    }

    #[test]
    fn test_mosaic_four_images() {
        // Create 4 images with different values in each corner
        let mut img0 = BarrierImage::new(8, 8, 3);
        img0.data.fill(10);

        let mut img1 = BarrierImage::new(8, 8, 3);
        img1.data.fill(20);

        let mut img2 = BarrierImage::new(8, 8, 3);
        img2.data.fill(30);

        let mut img3 = BarrierImage::new(8, 8, 3);
        img3.data.fill(40);

        let label0 = SoftLabel::one_hot(0, 4);
        let label1 = SoftLabel::one_hot(1, 4);
        let label2 = SoftLabel::one_hot(2, 4);
        let label3 = SoftLabel::one_hot(3, 4);

        let mut batch = Batch::new(
            vec![img0, img1, img2, img3],
            vec![label0, label1, label2, label3],
        );

        let mosaic = Mosaic::new();
        let mut rng = rand::thread_rng();
        mosaic.apply(&mut batch, &mut rng);

        // Batch should have 1 image
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.width(), Some(8));
        assert_eq!(batch.height(), Some(8));

        // Label should have 16 dimensions (4 x 4)
        assert_eq!(batch.labels()[0].probs().len(), 16);

        // Check that label is concatenation of all 4
        let label = batch.labels()[0].probs();
        assert_eq!(label[0], 1.0); // First one-hot
        assert_eq!(label[5], 1.0); // Second one-hot (index 4+1)
        assert_eq!(label[10], 1.0); // Third one-hot (index 8+2)
        assert_eq!(label[15], 1.0); // Fourth one-hot (index 12+3)
    }

    #[test]
    fn test_mosaic_eight_images() {
        let images = vec![
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
        ];

        let labels = vec![
            SoftLabel::one_hot(0, 2),
            SoftLabel::one_hot(1, 2),
            SoftLabel::one_hot(0, 2),
            SoftLabel::one_hot(1, 2),
            SoftLabel::one_hot(0, 2),
            SoftLabel::one_hot(1, 2),
            SoftLabel::one_hot(0, 2),
            SoftLabel::one_hot(1, 2),
        ];

        let mut batch = Batch::new(images, labels);

        let mosaic = Mosaic::new();
        let mut rng = rand::thread_rng();
        mosaic.apply(&mut batch, &mut rng);

        // Batch should have 2 images
        assert_eq!(batch.len(), 2);

        // Each label should have 8 dimensions (4 x 2)
        assert_eq!(batch.labels()[0].probs().len(), 8);
        assert_eq!(batch.labels()[1].probs().len(), 8);
    }

    #[test]
    fn test_mosaic_three_samples_no_op() {
        let images = vec![
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
            BarrierImage::new(8, 8, 3),
        ];

        let labels = vec![
            SoftLabel::one_hot(0, 2),
            SoftLabel::one_hot(1, 2),
            SoftLabel::one_hot(0, 2),
        ];

        let batch_len = images.len();

        let mut batch = Batch::new(images, labels);

        let mosaic = Mosaic::new();
        let mut rng = rand::thread_rng();
        mosaic.apply(&mut batch, &mut rng);

        // Batch should be unchanged (less than 4 images)
        assert_eq!(batch.len(), batch_len);
    }

    #[test]
    fn test_mosaic_preserves_dimensions() {
        let images = vec![
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(32, 32, 3),
        ];

        let labels = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
            SoftLabel::one_hot(2, 10),
            SoftLabel::one_hot(3, 10),
        ];

        let mut batch = Batch::new(images, labels);

        let mosaic = Mosaic::new();
        let mut rng = rand::thread_rng();
        mosaic.apply(&mut batch, &mut rng);

        // Output image should have same dimensions
        assert_eq!(batch.width(), Some(32));
        assert_eq!(batch.height(), Some(32));
        assert_eq!(batch.channels(), Some(3));
    }
}
