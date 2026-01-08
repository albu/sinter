// Core traits and enums
//
// This module defines the foundational types and traits that enable
// the compiler to reason about transform fusion and safe in-place execution.

use super::image::{FusableImage, BarrierImage};

/// How a transform accesses memory
///
/// This is critical for the planner to determine:
/// 1. Can transforms be fused?
/// 2. Is in-place execution safe?
/// 3. Are intermediate buffers needed?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Reads and writes the same buffer (mutation)
    ///
    /// Example: Brightness, Contrast, Normalize
    /// These are the primary candidates for fusion
    InPlace,

    /// Needs a separate output buffer
    ///
    /// Example: Transforms that cannot safely mutate in-place
    OutOfPlace,
}

/// How a transform affects image shape
///
/// The planner uses this to insert barriers between
/// shape-preserving and shape-changing operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeEffect {
    /// Height, Width, Channels all unchanged
    ///
    /// These transforms can be freely fused
    Preserve,

    /// Image dimensions change
    ///
    /// Example: Resize, Scale
    Resize,

    /// A subset of the image is extracted
    ///
    /// Example: RandomCrop, CenterCrop
    Crop,

    /// Image is extended with padding
    ///
    /// Example: Pad, PadIfNeeded
    Pad,
}

/// Reordering rules for the optimizer's canonicalization phase
///
/// This encodes which transforms can be safely reordered during the
/// geometric hoisting pass. The rule is based on algebraic properties,
/// not heuristics.
///
/// # The Only Safe Reordering
///
/// Per-pixel photometric transforms commute with geometric coordinate transforms:
///
/// ```text
/// P(f(x)) = f(P(x))
/// ```
///
/// Where:
/// - `P` = per-pixel photometric operation (LUT, matrix, pixel-wise math)
/// - `f` = bijective coordinate remapping (flip, rotate, transpose)
///
/// This holds because geometric ops only change *where* a pixel is read from,
/// while photometric ops only change *what* the pixel value is.
///
/// # Critical: Photometric Ops Do NOT Commute With Each Other
///
/// ```text
/// Brightness ∘ Contrast ≠ Contrast ∘ Brightness
/// Gamma ∘ Saturation ≠ Saturation ∘ Gamma
/// ```
///
/// They only commute with geometry, not with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReorderRule {
    /// Per-pixel photometric operations that commute with geometry
    ///
    /// These can be hoisted across geometric transforms during canonicalization.
    /// Examples: Brightness, Contrast, Gamma, LUT ops, Matrix ops, RGBShift
    ///
    /// Order among photometric ops is preserved - they are only reordered
    /// relative to geometric transforms, not relative to each other.
    CommutesWithGeometry,

    /// Coordinate remapping operations (geometric transforms)
    ///
    /// These can be composed via group operations (D4 group for flips/rotates)
    /// but cannot be reordered arbitrarily relative to each other.
    /// Examples: HorizontalFlip, VerticalFlip, Rotate, Transpose
    Geometry,

    /// Hard barriers that cannot be reordered
    ///
    /// These either:
    /// - Change shape/channels (ToGray, Resize, Crop, Pad)
    /// - Use neighborhoods (Blur, Sharpen, Convolve)
    /// - Introduce randomness (Noise)
    /// - Need global state (Histogram operations)
    /// - Are otherwise incompatible with reordering
    Barrier,
}

/// Semantic transform descriptor
///
/// This trait describes WHAT a transform does, not HOW to execute it.
/// The planner uses this information to build optimized execution plans.
///
/// No execution logic here - purely declarative.
pub trait Transform: std::any::Any + Send + Sync {
    /// Declare how this transform accesses memory
    fn access(&self) -> AccessPattern;

    /// Declare how this transform affects image shape
    fn shape_effect(&self) -> ShapeEffect;

    /// Get this transform as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get this transform as an Executable for zero-RTTI execution
    ///
    /// This uses vtable dispatch instead of RTTI for single-transform nodes.
    /// Returns `Some(&dyn Executable)` if this transform implements Executable.
    ///
    /// Default implementation returns None. Override by returning `Some(this)`.
    fn as_executable(&self) -> Option<&dyn Executable> {
        None
    }

    /// Get this transform as a LabelTransform for coordinate mapping
    ///
    /// This enables the transform to be applied to bounding boxes and keypoints.
    /// Default implementation returns None. Override by returning `Some(this)`.
    fn as_label_transform(&self) -> Option<&dyn LabelTransform> {
        None
    }

    /// Declare how this transform can be reordered during canonicalization
    ///
    /// Default is `Barrier` (cannot be reordered). Override for:
    /// - Per-pixel photometric ops → `CommutesWithGeometry`
    /// - Geometric coordinate ops → `Geometry`
    ///
    /// See `ReorderRule` documentation for the algebraic rules.
    fn reorder_rule(&self) -> ReorderRule {
        ReorderRule::Barrier
    }
}

/// Trait for transforms that map spatial coordinates
///
/// This is implemented by geometric transforms (Rotate, Resize, Crop, etc.)
/// to support bounding box and keypoint transformations.
pub trait LabelTransform: Transform {
    /// Transform a single 2D point (x, y)
    ///
    /// # Arguments
    /// - `point`: (x, y) coordinates in pixels
    /// - `image_size`: (width, height) of the image BEFORE this transform
    ///
    /// # Returns
    /// - `Some((x', y'))`: New coordinates
    /// - `None`: Point is outside valid area / clipped
    fn map_point(&self, point: (f32, f32), image_size: (u32, u32)) -> Option<(f32, f32)>;

    /// Transform a bounding box (x, y, w, h)
    ///
    /// # Arguments
    /// - `bbox`: [x, y, w, h] in pixels
    /// - `image_size`: (width, height) of the image BEFORE this transform
    ///
    /// # Returns
    /// - `Some([x', y', w', h'])`: New bounding box
    /// - `None`: Box is fully outside valid area / clipped
    fn map_bbox(&self, bbox: [f32; 4], image_size: (u32, u32)) -> Option<[f32; 4]>;
}

/// Trait for transforms that can be executed on a FusableImage
///
/// This is implemented by:
/// - Photometric transforms (via PixelOp)
/// - Geometric transforms (via direct apply method)
///
/// The executor uses this trait to run the optimized execution plan.
///
/// # Return value
/// - `None`: Transform modified the image in-place (most common)
/// - `Some(BarrierImage)`: Transform allocated a new buffer (Resize, Crop, etc.)
pub trait Executable: Transform + Send + Sync {
    /// Execute this transform on the given image
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage>;
}

/// Helper: Check if a transform can be fused into a single-pass block
///
/// Fusion is possible when:
/// - Transform is InPlace (no buffer allocation needed)
/// - Shape is Preserved (all pixels map 1:1)
pub fn is_fuseable<T: Transform + ?Sized>(transform: &T) -> bool {
    transform.access() == AccessPattern::InPlace
        && transform.shape_effect() == ShapeEffect::Preserve
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyInPlacePreserve;
    impl Transform for DummyInPlacePreserve {
        fn access(&self) -> AccessPattern {
            AccessPattern::InPlace
        }
        fn shape_effect(&self) -> ShapeEffect {
            ShapeEffect::Preserve
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    struct DummyOutOfPlace;
    impl Transform for DummyOutOfPlace {
        fn access(&self) -> AccessPattern {
            AccessPattern::OutOfPlace
        }
        fn shape_effect(&self) -> ShapeEffect {
            ShapeEffect::Preserve
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    struct DummyResize;
    impl Transform for DummyResize {
        fn access(&self) -> AccessPattern {
            AccessPattern::InPlace
        }
        fn shape_effect(&self) -> ShapeEffect {
            ShapeEffect::Resize
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_is_fuseable() {
        let t1 = DummyInPlacePreserve;
        let t2 = DummyOutOfPlace;
        let t3 = DummyResize;

        assert!(is_fuseable(&t1));
        assert!(!is_fuseable(&t2));
        assert!(!is_fuseable(&t3));
    }
}
