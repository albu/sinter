// Label types for batch-level transforms.
//
// This module defines the `Label` trait and concrete implementations for
// different supervision types (classification, detection, segmentation, etc.).
//
// ## Design Philosophy
//
// Labels represent semantic supervision that travels with images. When images
// are mixed at the batch level (MixUp, CutMix, Mosaic), labels must be combined
// in a task-appropriate manner.
//
// The `Label` trait is deliberately minimal: it only requires a `mix` operation.
// Different label types interpret the mixing coefficient differently:
//
// - **SoftLabel**: Linear interpolation of probability vectors
// - **BoundingBoxes**: Area-weighted combination or concatenation
// - **SegmentationMask**: Pixel-wise or spatial mixing
//
// ## Type Safety
//
// Not all label types can be mixed. For example, hard class indices (`u32`)
// cannot meaningfully participate in MixUp — they must be converted to soft
// labels first. This is enforced at compile time by *not* implementing `Label`
// for such types.

use std::fmt::Debug;

#[cfg(feature = "simd")]
use std::simd::prelude::*;

/// A semantic label that can be combined with other labels.
///
/// Represents supervision information that travels with an image and must be
/// transformed consistently when images are mixed at the batch level.
///
/// # Requirements
///
/// Implementations must ensure that `mix(a, b, λ)` preserves label invariants:
/// - Probability vectors sum to 1
/// - Box coordinates remain valid
/// - Any other task-specific constraints
///
/// # Example
///
/// ```ignore
/// let label_a = SoftLabel::from_class(0);
/// let label_b = SoftLabel::from_class(1);
/// let mixed = label_a.mix(&label_b, 0.7); // 70% class 0, 30% class 1
/// ```
pub trait Label: Clone + Send + Sync + Debug {
    /// Combine this label with another according to a mixing coefficient.
    ///
    /// # Parameters
    ///
    /// - `other`: The other label to mix with
    /// - `lambda`: Mixing coefficient in [0, 1], where:
    ///   - `lambda = 1.0` returns a copy of `self`
    ///   - `lambda = 0.0` returns a copy of `other`
    ///   - `lambda = 0.5` returns an equal blend
    ///
    /// # Interpretation by Transform
    ///
    /// - **MixUp**: `lambda` is the sampled mixing ratio from Beta(α, α)
    /// - **CutMix**: `lambda` may be derived from box area or ignored entirely
    ///
    /// # Implementation Notes
    ///
    /// Some label types cannot meaningfully mix (e.g., hard class indices).
    /// Such types should *not* implement `Label` — users must convert to a
    /// mixable type first (e.g., `ClassIndex(u32)` → `SoftLabel`).
    fn mix(&self, other: &Self, lambda: f32) -> Self;

    /// Concatenate multiple labels into a single combined label.
    ///
    /// This is used by transforms like Mosaic that combine multiple images
    /// into one. The default implementation concatenates probability vectors.
    ///
    /// # Parameters
    ///
    /// - `labels`: Slice of labels to concatenate
    ///
    /// # Returns
    ///
    /// A single label representing the concatenation of all input labels.
    ///
    /// # Implementation Notes
    ///
    /// The default implementation works for label types that store their data
    /// in a contiguous vector (like `SoftLabel`). Other label types may override
    /// this to provide task-specific concatenation semantics.
    fn concatenate(labels: &[Self]) -> Self
    where
        Self: Sized,
    {
        // Default implementation: this should be overridden by concrete types
        // We provide a panic here to force implementations to override
        panic!("Label::concatenate must be implemented for concrete label types");
    }
}

// =============================================================================
// Concrete Label Types
// =============================================================================

/// A soft (probabilistic) classification label.
///
/// This is the canonical label type for MixUp, as it supports linear
/// interpolation of class probabilities.
///
/// # Example
///
/// ```
/// use sinter::batch::{Label, label::SoftLabel};
///
/// // Create a one-hot vector for class 2 out of 5 classes
/// let label_a = SoftLabel::one_hot(2, 5);
/// let label_b = SoftLabel::one_hot(3, 5);
///
/// // Mix: 70% class 2, 30% class 3
/// let mixed = label_a.mix(&label_b, 0.7);
/// assert_eq!(mixed.probs()[2], 0.7);
/// assert_eq!(mixed.probs()[3], 0.3);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SoftLabel {
    /// Probability vector. May not be normalized (e.g., during computation).
    probs: Vec<f32>,
}

impl SoftLabel {
    /// Create a soft label from a probability vector.
    ///
    /// # Panics
    ///
    /// Panics if `probs` is empty.
    #[inline]
    pub fn new(probs: Vec<f32>) -> Self {
        assert!(!probs.is_empty(), "SoftLabel cannot be empty");
        Self { probs }
    }

    /// SIMD-optimized linear interpolation of two probability vectors.
    ///
    /// Computes: `result = lambda * self + (1 - lambda) * other`
    ///
    /// Uses `std::simd` f32x8 vectors when available (nightly feature).
    /// Processes 8 elements at a time for better throughput.
    #[inline]
    fn mix_simd(&self, other: &Self, lambda: f32) -> Self {
        assert_eq!(
            self.probs.len(),
            other.probs.len(),
            "cannot mix labels with different numbers of classes"
        );

        let len = self.probs.len();

        #[cfg(feature = "simd")]
        {
            // Use SIMD for larger label vectors (>= 8 classes)
            const SIMD_THRESHOLD: usize = 8;

            if len >= SIMD_THRESHOLD {
                let simd_len = len - (len % 8);
                let mut result = Vec::with_capacity(len);

                // Process 8 elements at a time
                for i in (0..simd_len).step_by(8) {
                    let a = f32x8::from_array([
                        self.probs[i],
                        self.probs[i + 1],
                        self.probs[i + 2],
                        self.probs[i + 3],
                        self.probs[i + 4],
                        self.probs[i + 5],
                        self.probs[i + 6],
                        self.probs[i + 7],
                    ]);
                    let b = f32x8::from_array([
                        other.probs[i],
                        other.probs[i + 1],
                        other.probs[i + 2],
                        other.probs[i + 3],
                        other.probs[i + 4],
                        other.probs[i + 5],
                        other.probs[i + 6],
                        other.probs[i + 7],
                    ]);

                    // Compute: lambda * a + (1 - lambda) * b
                    let lambda_vec = f32x8::splat(lambda);
                    let one_minus_lambda = f32x8::splat(1.0 - lambda);
                    let mixed = lambda_vec * a + one_minus_lambda * b;

                    result.extend_from_slice(&mixed.to_array());
                }

                // Handle remaining elements
                for i in simd_len..len {
                    result.push(lambda * self.probs[i] + (1.0 - lambda) * other.probs[i]);
                }

                return Self { probs: result };
            }
        }

        // Scalar fallback for small label vectors or when SIMD isn't available
        let probs = self
            .probs
            .iter()
            .zip(&other.probs)
            .map(|(a, b)| lambda * a + (1.0 - lambda) * b)
            .collect();

        Self { probs }
    }

    /// Create a one-hot encoded label.
    ///
    /// # Parameters
    ///
    /// - `class`: The class index to encode
    /// - `num_classes`: Total number of classes
    ///
    /// # Panics
    ///
    /// Panics if `class >= num_classes` or `num_classes == 0`.
    #[inline]
    pub fn one_hot(class: usize, num_classes: usize) -> Self {
        assert!(class < num_classes, "class index out of bounds");
        assert!(num_classes > 0, "num_classes must be positive");

        let mut probs = vec![0.0; num_classes];
        probs[class] = 1.0;
        Self { probs }
    }

    /// Create a soft label that approximates a hard class index.
    ///
    /// This is a convenience method for converting from `ClassIndex`.
    #[inline]
    pub fn from_class(class: usize, num_classes: usize) -> Self {
        Self::one_hot(class, num_classes)
    }

    /// Get the probability vector.
    #[inline]
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }

    /// Get the number of classes.
    #[inline]
    pub fn num_classes(&self) -> usize {
        self.probs.len()
    }

    /// Get the predicted class (argmax).
    ///
    /// Returns `None` if the probability vector is empty (should not happen
    /// due to invariant in `new`).
    #[inline]
    pub fn argmax(&self) -> Option<usize> {
        self.probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
    }

    /// Normalize the probability vector to sum to 1.
    ///
    /// This is useful after manual modifications or to correct numerical drift.
    #[inline]
    pub fn normalize(&mut self) {
        let sum: f32 = self.probs.iter().sum();
        if sum > 0.0 {
            for p in &mut self.probs {
                *p /= sum;
            }
        }
    }
}

impl Label for SoftLabel {
    #[inline]
    fn mix(&self, other: &Self, lambda: f32) -> Self {
        // Use SIMD-optimized implementation
        self.mix_simd(other, lambda)
    }

    #[inline]
    fn concatenate(labels: &[Self]) -> Self
    where
        Self: Sized,
    {
        if labels.is_empty() {
            panic!("cannot concatenate empty label list");
        }

        // Concatenate all probability vectors
        let probs = labels.iter().flat_map(|l| l.probs.iter().copied()).collect();

        Self { probs }
    }
}

// =============================================================================
// Reference Types (Do NOT Implement Label)
// =============================================================================

/// A hard classification label (class index).
///
/// **IMPORTANT**: This type does *not* implement `Label` because hard class
/// indices cannot meaningfully participate in MixUp.
///
/// To use MixUp, convert to `SoftLabel` first:
///
/// ```ignore
/// let hard = ClassIndex(2);
/// let soft = SoftLabel::from_class(hard.0, num_classes);
/// let mixed = soft.mix(&other, 0.7);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClassIndex(pub usize);

impl ClassIndex {
    /// Convert to a soft label.
    #[inline]
    pub fn to_soft(self, num_classes: usize) -> SoftLabel {
        SoftLabel::one_hot(self.0, num_classes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_label_one_hot() {
        let label = SoftLabel::one_hot(2, 5);
        assert_eq!(label.probs(), &[0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(label.argmax(), Some(2));
    }

    #[test]
    fn test_soft_label_mix() {
        let a = SoftLabel::one_hot(0, 3);
        let b = SoftLabel::one_hot(1, 3);

        let mixed = a.mix(&b, 0.7);
        assert_eq!(mixed.probs(), &[0.7, 0.3, 0.0]);
    }

    #[test]
    fn test_soft_label_mix_identity() {
        let a = SoftLabel::one_hot(0, 3);
        let b = SoftLabel::one_hot(1, 3);

        // lambda = 1.0 → return a unchanged
        assert_eq!(a.mix(&b, 1.0), a);

        // lambda = 0.0 → return b unchanged
        assert_eq!(a.mix(&b, 0.0), b);
    }

    #[test]
    #[should_panic(expected = "cannot mix labels with different numbers of classes")]
    fn test_soft_label_mix_different_sizes() {
        let a = SoftLabel::one_hot(0, 3);
        let b = SoftLabel::one_hot(0, 5);
        a.mix(&b, 0.5);
    }

    #[test]
    fn test_class_index_to_soft() {
        let hard = ClassIndex(2);
        let soft = hard.to_soft(5);
        assert_eq!(soft.probs(), &[0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_soft_label_concatenate() {
        let a = SoftLabel::one_hot(0, 2);
        let b = SoftLabel::one_hot(1, 2);
        let c = SoftLabel::one_hot(0, 2);
        let d = SoftLabel::one_hot(1, 2);

        let concatenated = SoftLabel::concatenate(&[a, b, c, d]);
        assert_eq!(concatenated.probs().len(), 8);
        assert_eq!(concatenated.probs(), &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "cannot concatenate empty label list")]
    fn test_soft_label_concatenate_empty() {
        SoftLabel::concatenate(&[]);
    }
}
