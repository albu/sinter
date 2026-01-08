// Batch-level transforms (MixUp, CutMix, Mosaic, etc.).

pub mod label;
pub mod integration;
pub mod mixup;
pub mod cutmix;
pub mod mosaic;
pub mod pipeline;

// Re-export batch transforms
pub use label::{Label, SoftLabel, ClassIndex};
pub use mixup::MixUp;
pub use cutmix::CutMix;
pub use mosaic::Mosaic;
pub use pipeline::BatchPipeline;
//
// This module defines a separate execution domain for transforms that operate
// on multiple images simultaneously, along with their associated labels.
//
// ## Architectural Separation
//
// Batch transforms are fundamentally different from single-image transforms:
//
// - **Single-image transforms** (`Transform` trait): Operate on one image at a
//   time, support fusion (LUT, Matrix, Structural), and are optimized by the
//   compiler.
//
// - **Batch transforms** (`BatchTransform` trait): Operate on multiple images
//   together, require cross-image operations, and do not participate in fusion.
//
// ## Two-Stage Pipeline
//
// The recommended usage pattern is:
//
// ```ignore
// Stage 1: Apply per-image transforms with fusion
// let images = image_pipeline.apply_to_batch(&raw_images);
//
// Stage 2: Apply batch-level mixing
// let mut batch = Batch::new(images, labels);
// mixup.apply(&mut batch, &mut rng);
// ```
//
// This separation keeps the fusion optimizer pure while giving batch transforms
// full freedom to sample, mix, and rearrange data.

use crate::core::BarrierImage;
use std::fmt;

/// A batch of images and their associated labels.
///
/// # Invariants
///
/// - `images.len() == labels.len()`
/// - All images have the same dimensions and channel count
///
/// # Type Parameters
///
/// - `L`: The label type (must implement `Label`)
///
/// # Example
///
/// ```ignore
/// use sinter::batch::{Batch, SoftLabel};
///
/// // Create a batch from existing data
/// let images = vec![img1, img2, img3, img4];
/// let labels = vec![
///     SoftLabel::one_hot(0, 10),
///     SoftLabel::one_hot(1, 10),
///     SoftLabel::one_hot(2, 10),
///     SoftLabel::one_hot(3, 10),
/// ];
/// let batch = Batch::new(images, labels);
///
/// // Or start empty and push
/// let mut batch = Batch::empty();
/// batch.push(img1, label1);
/// ```
pub struct Batch<L: Label> {
    /// The images in this batch
    images: Vec<BarrierImage>,

    /// The labels corresponding to each image
    labels: Vec<L>,
}

impl<L: Label> Batch<L> {
    /// Create a new batch from existing images and labels.
    ///
    /// # Panics
    ///
    /// Panics if `images.len() != labels.len()`.
    ///
    /// # Panics
    ///
    /// Panics if images have inconsistent dimensions (not all same width,
    /// height, and channel count).
    #[inline]
    pub fn new(images: Vec<BarrierImage>, labels: Vec<L>) -> Self {
        assert_eq!(
            images.len(),
            labels.len(),
            "Batch: images and labels must have the same length"
        );

        if !images.is_empty() {
            Self::check_consistent(&images);
        }

        Self { images, labels }
    }

    /// Create an empty batch.
    #[inline]
    pub fn empty() -> Self {
        Self {
            images: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// Add an image-label pair to the batch.
    ///
    /// # Panics
    ///
    /// Panics if the image dimensions don't match the existing images in the
    /// batch (if any).
    #[inline]
    pub fn push(&mut self, image: BarrierImage, label: L) {
        if !self.images.is_empty() {
            Self::check_consistent_one(&self.images[0], &image);
        }
        self.images.push(image);
        self.labels.push(label);
    }

    /// Get the number of samples in the batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Returns `true` if the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Get a reference to the images.
    #[inline]
    pub fn images(&self) -> &[BarrierImage] {
        &self.images
    }

    /// Get a mutable reference to the images.
    #[inline]
    pub fn images_mut(&mut self) -> &mut [BarrierImage] {
        &mut self.images
    }

    /// Get a reference to the labels.
    #[inline]
    pub fn labels(&self) -> &[L] {
        &self.labels
    }

    /// Get a mutable reference to the labels.
    #[inline]
    pub fn labels_mut(&mut self) -> &mut [L] {
        &mut self.labels
    }

    /// Get the image width (all images have the same width).
    ///
    /// Returns `None` if the batch is empty.
    #[inline]
    pub fn width(&self) -> Option<usize> {
        self.images.first().map(|img| img.width)
    }

    /// Get the image height (all images have the same height).
    ///
    /// Returns `None` if the batch is empty.
    #[inline]
    pub fn height(&self) -> Option<usize> {
        self.images.first().map(|img| img.height)
    }

    /// Get the number of channels (all images have the same channel count).
    ///
    /// Returns `None` if the batch is empty.
    #[inline]
    pub fn channels(&self) -> Option<usize> {
        self.images.first().map(|img| img.channels)
    }

    /// Split the batch into separate images and labels vectors.
    ///
    /// This is useful when you want to consume the batch and take ownership
    /// of its components.
    #[inline]
    pub fn into_parts(self) -> (Vec<BarrierImage>, Vec<L>) {
        (self.images, self.labels)
    }

    /// Get a reference to the image at the given index (for Python bindings).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn get_image(&self, index: usize) -> &BarrierImage {
        &self.images[index]
    }

    /// Get a mutable reference to the image at the given index (for Python bindings).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn get_image_mut(&mut self, index: usize) -> &mut BarrierImage {
        &mut self.images[index]
    }

    /// Get a reference to the label at the given index (for Python bindings).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn get_label(&self, index: usize) -> &L {
        &self.labels[index]
    }

    /// Get image data slice at index (for Python bindings).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn image_data(&self, index: usize) -> &[u8] {
        &self.images[index].data
    }

    /// Retain only the samples for which the predicate returns `true`.
    ///
    /// This is useful for filtering batches based on label criteria.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Keep only samples where the max class is 0 or 1
    /// batch.retain(|label| label.argmax().map_or(false, |c| c < 2));
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&L) -> bool,
    {
        let mut to_remove = Vec::new();
        for (i, label) in self.labels.iter().enumerate() {
            if !f(label) {
                to_remove.push(i);
            }
        }

        // Remove in reverse order to preserve indices
        for i in to_remove.into_iter().rev() {
            self.images.remove(i);
            self.labels.remove(i);
        }
    }

    /// Shuffle the batch in-place using Fisher-Yates algorithm.
    ///
    /// # Parameters
    ///
    /// - `rng`: A random number generator implementing `rand::Rng`
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rand::SeedableRng;
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    /// batch.shuffle(&mut rng);
    /// ```
    pub fn shuffle<R: rand::Rng>(&mut self, rng: &mut R) {
        let len = self.len();
        if len <= 1 {
            return;
        }

        for i in (0..len).rev() {
            let j = rng.gen_range(0..=i);
            if i != j {
                self.images.swap(i, j);
                self.labels.swap(i, j);
            }
        }
    }

    /// Verify that all images in the batch have consistent dimensions.
    fn check_consistent(images: &[BarrierImage]) {
        let first = &images[0];
        for (i, img) in images.iter().enumerate().skip(1) {
            if img.width != first.width
                || img.height != first.height
                || img.channels != first.channels
            {
                panic!(
                    "Batch: image at index {} has inconsistent dimensions: \
                     expected {}x{}x{}, got {}x{}x{}",
                    i, first.width, first.height, first.channels,
                    img.width, img.height, img.channels
                );
            }
        }
    }

    /// Verify that a single image matches the expected dimensions.
    fn check_consistent_one(expected: &BarrierImage, actual: &BarrierImage) {
        if actual.width != expected.width
            || actual.height != expected.height
            || actual.channels != expected.channels
        {
            panic!(
                "Batch::push: image has inconsistent dimensions: \
                 expected {}x{}x{}, got {}x{}x{}",
                expected.width, expected.height, expected.channels,
                actual.width, actual.height, actual.channels
            );
        }
    }
}

impl<L: Label> fmt::Debug for Batch<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Batch")
            .field("len", &self.len())
            .field(
                "image_size",
                &self.width().zip(self.height()).zip(self.channels()),
            )
            .finish()
    }
}

// =============================================================================
// BatchTransform Trait
// =============================================================================

/// A transform that operates on an entire batch of images and labels.
///
/// Unlike `Transform`, which operates on single images and supports fusion,
/// `BatchTransform` operates on multiple images simultaneously and typically
/// involves:
///
/// - Random sampling across the batch
/// - Cross-image operations (mixing, cutting, pasting)
/// - Label transformations
///
/// # Examples
///
/// - **MixUp**: Linear blending of image pairs with corresponding label mixing
/// - **CutMix**: Replacing a rectangular region with content from another image
/// - **Mosaic**: Combining 4 images into a grid layout
///
/// # Design Notes
///
/// Batch transforms deliberately do NOT support fusion. They run after the
/// per-image pipeline has already optimized all single-image operations.
pub trait BatchTransform<L: Label> {
    /// Apply this transform to the batch in place.
    ///
    /// # Parameters
    ///
    /// - `batch`: The batch to transform (modified in place)
    /// - `rng`: A random number generator for stochastic operations
    ///
    /// # Implementation Notes
    ///
    /// Implementations should:
    /// - Modify `batch` in place (avoid allocations where possible)
    /// - Use `rng` for all stochastic decisions
    /// - Ensure batch invariants are preserved (e.g., consistent image sizes)
    fn apply<R: rand::Rng>(&self, batch: &mut Batch<L>, rng: &mut R);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_new() {
        let images = vec![
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(32, 32, 3),
        ];
        let labels = vec![SoftLabel::one_hot(0, 10), SoftLabel::one_hot(1, 10)];

        let batch = Batch::new(images.clone(), labels.clone());
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.width(), Some(32));
        assert_eq!(batch.height(), Some(32));
        assert_eq!(batch.channels(), Some(3));
    }

    #[test]
    #[should_panic(expected = "images and labels must have the same length")]
    fn test_batch_new_mismatched_lengths() {
        let images = vec![BarrierImage::new(32, 32, 3)];
        let labels = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
        ];
        Batch::new(images, labels);
    }

    #[test]
    #[should_panic(expected = "inconsistent dimensions")]
    fn test_batch_new_inconsistent_dimensions() {
        let images = vec![
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(64, 64, 3),
        ];
        let labels = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
        ];
        Batch::new(images, labels);
    }

    #[test]
    fn test_batch_empty() {
        let batch: Batch<SoftLabel> = Batch::empty();
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());
        assert_eq!(batch.width(), None);
        assert_eq!(batch.height(), None);
        assert_eq!(batch.channels(), None);
    }

    #[test]
    fn test_batch_push() {
        let mut batch = Batch::empty();
        batch.push(BarrierImage::new(32, 32, 3), SoftLabel::one_hot(0, 10));
        batch.push(BarrierImage::new(32, 32, 3), SoftLabel::one_hot(1, 10));

        assert_eq!(batch.len(), 2);
    }

    #[test]
    #[should_panic(expected = "inconsistent dimensions")]
    fn test_batch_push_inconsistent() {
        let mut batch = Batch::empty();
        batch.push(BarrierImage::new(32, 32, 3), SoftLabel::one_hot(0, 10));
        batch.push(BarrierImage::new(64, 64, 3), SoftLabel::one_hot(1, 10));
    }

    #[test]
    fn test_batch_into_parts() {
        let images = vec![
            BarrierImage::new(32, 32, 3),
            BarrierImage::new(32, 32, 3),
        ];
        let labels = vec![
            SoftLabel::one_hot(0, 10),
            SoftLabel::one_hot(1, 10),
        ];

        let batch = Batch::new(images.clone(), labels.clone());
        let (out_images, out_labels) = batch.into_parts();

        assert_eq!(out_images.len(), 2);
        assert_eq!(out_labels.len(), 2);
    }

    #[test]
    fn test_batch_retain() {
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
        batch.retain(|l| l.argmax().map_or(false, |c| c < 2));

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.labels()[0].argmax(), Some(0));
        assert_eq!(batch.labels()[1].argmax(), Some(1));
    }
}
