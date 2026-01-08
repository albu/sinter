// BatchPipeline - chains multiple batch transforms
//
// This provides a clean API for applying multiple batch transforms in sequence,
// similar to how Compose works for single-image transforms.

use crate::batch::{BatchTransform, Batch, MixUp, CutMix, Mosaic, SoftLabel};
use std::fmt;
use rand::Rng;

/// Enum representing all possible batch transforms
///
/// This is needed because BatchTransform is not object-safe (cannot be made into a trait object).
pub enum BatchTransformEnum {
    MixUp(MixUp),
    CutMix(CutMix),
    Mosaic(Mosaic),
}

impl BatchTransform<SoftLabel> for BatchTransformEnum {
    fn apply<R: rand::Rng>(&self, batch: &mut Batch<SoftLabel>, rng: &mut R) {
        match self {
            BatchTransformEnum::MixUp(t) => t.apply(batch, rng),
            BatchTransformEnum::CutMix(t) => t.apply(batch, rng),
            BatchTransformEnum::Mosaic(t) => t.apply(batch, rng),
        }
    }
}

impl Clone for BatchTransformEnum {
    fn clone(&self) -> Self {
        match self {
            BatchTransformEnum::MixUp(t) => BatchTransformEnum::MixUp(t.clone()),
            BatchTransformEnum::CutMix(t) => BatchTransformEnum::CutMix(t.clone()),
            BatchTransformEnum::Mosaic(t) => BatchTransformEnum::Mosaic(t.clone()),
        }
    }
}

/// A pipeline of batch transforms
///
/// Applies multiple batch transforms in sequence. This is the batch-level
/// equivalent of Compose for single-image transforms.
///
/// # Seeding and Reproducibility
///
/// Use `set_seed()` to enable deterministic behavior in Python:
///
/// ```ignore
/// # Python example
/// pipeline = BatchPipeline([MixUp(1.0), CutMix(1.0)])
/// pipeline.set_seed(42)
/// mixed_images, mixed_labels = pipeline.apply(images, labels)
/// ```
///
/// For Rust, use a seeded RNG directly:
/// ```ignore
/// use rand::SeedableRng;
///
/// let pipeline = BatchPipeline::new()
///     .add_mixup(MixUp::new(1.0))
///     .add_cutmix(CutMix::new(1.0));
///
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// pipeline.apply(&mut batch, &mut rng);
/// ```
///
/// # Example (non-deterministic)
/// ```ignore
/// use sinter::batch::{BatchPipeline, MixUp, CutMix, SoftLabel};
///
/// let pipeline = BatchPipeline::new()
///     .add(MixUp::new(1.0))
///     .add(CutMix::new(1.0));
///
/// let mut batch = Batch::new(images, labels);
/// let mut rng = rand::thread_rng();
/// pipeline.apply(&mut batch, &mut rng);
/// ```
pub struct BatchPipeline {
    transforms: Vec<BatchTransformEnum>,
    /// Optional seed for deterministic RNG
    seed: Option<u64>,
}

impl BatchPipeline {
    /// Create a new empty batch pipeline
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            seed: None,
        }
    }

    /// Set the seed for deterministic RNG behavior
    ///
    /// Once set, `apply_deterministic()` will use a seeded RNG.
    /// Use `clear_seed()` to revert to non-deterministic behavior.
    ///
    /// # Example
    /// ```ignore
    /// let mut pipeline = BatchPipeline::new()
    ///     .add_mixup(MixUp::new(1.0));
    /// pipeline.set_seed(42);
    /// ```
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
    }

    /// Clear the seed, reverting to non-deterministic RNG
    ///
    /// # Example
    /// ```ignore
    /// pipeline.clear_seed();
    /// ```
    pub fn clear_seed(&mut self) {
        self.seed = None;
    }

    /// Get the current seed, if set
    ///
    /// # Example
    /// ```ignore
    /// if let Some(seed) = pipeline.seed() {
    ///     println!("Using seed: {}", seed);
    /// }
    /// ```
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Add a MixUp transform to the pipeline
    pub fn add_mixup(mut self, mixup: MixUp) -> Self {
        self.transforms.push(BatchTransformEnum::MixUp(mixup));
        self
    }

    /// Add a CutMix transform to the pipeline
    pub fn add_cutmix(mut self, cutmix: CutMix) -> Self {
        self.transforms.push(BatchTransformEnum::CutMix(cutmix));
        self
    }

    /// Add a Mosaic transform to the pipeline
    pub fn add_mosaic(mut self, mosaic: Mosaic) -> Self {
        self.transforms.push(BatchTransformEnum::Mosaic(mosaic));
        self
    }

    /// Apply all transforms in the pipeline to a batch
    ///
    /// # Arguments
    /// - `batch`: The batch to transform (modified in place)
    /// - `rng`: A random number generator for stochastic operations
    ///
    /// # Example
    /// ```ignore
    /// let mut batch = Batch::new(images, labels);
    /// let mut rng = rand::thread_rng();
    /// pipeline.apply(&mut batch, &mut rng);
    /// ```
    pub fn apply<R: rand::Rng>(&self, batch: &mut Batch<SoftLabel>, rng: &mut R) {
        for transform in &self.transforms {
            transform.apply(batch, rng);
        }
    }

    /// Get the number of transforms in the pipeline
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Get iterator over the transforms
    pub fn iter(&self) -> impl Iterator<Item = &BatchTransformEnum> {
        self.transforms.iter()
    }
}

impl Default for BatchPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BatchPipeline {
    fn clone(&self) -> Self {
        Self {
            transforms: self.transforms.clone(),
            seed: self.seed,
        }
    }
}

impl fmt::Debug for BatchPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchPipeline")
            .field("num_transforms", &self.transforms.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BarrierImage;
    use crate::batch::SoftLabel;

    #[test]
    fn test_batch_pipeline_new() {
        let pipeline: BatchPipeline = BatchPipeline::new();
        assert_eq!(pipeline.len(), 0);
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_batch_pipeline_add() {
        let pipeline: BatchPipeline = BatchPipeline::new()
            .add_mixup(MixUp::new(1.0))
            .add_cutmix(CutMix::new(1.0));

        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn test_batch_pipeline_apply() {
        let images: Vec<_> = (0..4)
            .map(|_| BarrierImage::new(32, 32, 3))
            .collect();

        let labels: Vec<_> = (0..4)
            .map(|i| SoftLabel::one_hot(i, 10))
            .collect();

        let mut batch = Batch::new(images, labels);

        let pipeline: BatchPipeline = BatchPipeline::new()
            .add_mixup(MixUp::new(1.0));

        let mut rng = rand::thread_rng();
        pipeline.apply(&mut batch, &mut rng);

        // Pipeline should have modified the batch
        assert_eq!(batch.len(), 4);

        // Labels should be mixed (soft)
        for label in &batch.labels {
            let max_prob = label.probs().iter().cloned().fold(0.0f32, f32::max);
            assert!(max_prob < 1.0, "Labels should be soft after MixUp");
        }
    }

    #[test]
    fn test_batch_pipeline_multiple_transforms() {
        let images: Vec<_> = (0..8)
            .map(|_| BarrierImage::new(64, 64, 3))
            .collect();

        let labels: Vec<_> = (0..8)
            .map(|i| SoftLabel::one_hot(i, 10))
            .collect();

        let mut batch = Batch::new(images, labels);
        let original_batch_size = batch.len();

        let pipeline: BatchPipeline = BatchPipeline::new()
            .add_mixup(MixUp::new(1.0))
            .add_cutmix(CutMix::new(1.0));

        let mut rng = rand::thread_rng();
        pipeline.apply(&mut batch, &mut rng);

        // MixUp and CutMix preserve batch size
        assert_eq!(batch.len(), original_batch_size);

        // Labels should be mixed
        let mixed_count = batch.labels.iter()
            .filter(|l| l.probs().iter().cloned().fold(0.0f32, f32::max) < 1.0)
            .count();
        assert!(mixed_count > 0, "At least some labels should be mixed");
    }

    #[test]
    fn test_batch_pipeline_with_mosaic() {
        let images: Vec<_> = (0..4)
            .map(|_| BarrierImage::new(32, 32, 3))
            .collect();

        let labels: Vec<_> = (0..4)
            .map(|i| SoftLabel::one_hot(i % 2, 2))
            .collect();

        let mut batch = Batch::new(images, labels);

        let pipeline: BatchPipeline = BatchPipeline::new()
            .add_mosaic(Mosaic::new());

        let mut rng = rand::thread_rng();
        pipeline.apply(&mut batch, &mut rng);

        // Mosaic reduces batch size by 4x
        assert_eq!(batch.len(), 1);
        // And concatenates labels
        assert_eq!(batch.labels[0].probs().len(), 8); // 2 * 4
    }

    #[test]
    fn test_batch_pipeline_clone() {
        let pipeline: BatchPipeline = BatchPipeline::new()
            .add_mixup(MixUp::new(1.0))
            .add_cutmix(CutMix::new(1.0));

        let cloned = pipeline.clone();
        assert_eq!(cloned.len(), pipeline.len());
    }

    #[test]
    fn test_batch_pipeline_seed() {
        let mut pipeline: BatchPipeline = BatchPipeline::new()
            .add_mixup(MixUp::new(1.0));
        assert_eq!(pipeline.seed(), None);

        pipeline.set_seed(42);
        assert_eq!(pipeline.seed(), Some(42));

        pipeline.clear_seed();
        assert_eq!(pipeline.seed(), None);
    }

    #[test]
    fn test_batch_pipeline_deterministic() {
        // Create two identical batches
        let images1: Vec<_> = (0..4).map(|_| {
            let mut img = BarrierImage::new(32, 32, 3);
            img.data.fill(100);
            img
        }).collect();

        let images2: Vec<_> = (0..4).map(|_| {
            let mut img = BarrierImage::new(32, 32, 3);
            img.data.fill(100);
            img
        }).collect();

        let labels1: Vec<_> = (0..4).map(|i| SoftLabel::one_hot(i, 10)).collect();
        let labels2: Vec<_> = (0..4).map(|i| SoftLabel::one_hot(i, 10)).collect();

        let mut batch1 = Batch::new(images1, labels1);
        let mut batch2 = Batch::new(images2, labels2);

        let pipeline: BatchPipeline = BatchPipeline::new().add_mixup(MixUp::new(1.0));

        // Apply with same seed - using external RNG
        use rand_chacha::ChaCha8Rng;
        use rand::SeedableRng;
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        pipeline.apply(&mut batch1, &mut rng1);
        pipeline.apply(&mut batch2, &mut rng2);

        // Results should be identical
        assert_eq!(batch1.images[0].data, batch2.images[0].data);
        assert_eq!(batch1.labels[0].probs(), batch2.labels[0].probs());
    }

    #[test]
    fn test_batch_pipeline_clone_preserves_seed() {
        let mut pipeline: BatchPipeline = BatchPipeline::new()
            .add_mixup(MixUp::new(1.0));
        pipeline.set_seed(42);

        let cloned = pipeline.clone();
        assert_eq!(cloned.seed(), Some(42));
    }
}
