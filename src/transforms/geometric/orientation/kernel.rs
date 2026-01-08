// Structural Kernel - Geometric Transform Fusion
//
// Fuses multiple geometric transforms (flip, rotate, transpose) into a single operation.
// This is the result of composing transforms via the D4 group.
//
// # Architecture: Delegate to Dedicated Transforms
//
// StructuralKernel DOES NOT reimplement geometric operations. Instead, it DELEGATES
// to the dedicated transform implementations (Transpose, Rotate, FlipH, FlipV).
//
// This eliminates code duplication and ensures we always use the optimized implementations.
//
// # Why This Matters
//
// Previously, this file had duplicate implementations that could get out of sync
// with the dedicated transforms. For example, Transpose was reimplemented with slow
// scalar loops while the dedicated Transpose used NEON SIMD. This made fused operations
// 8x slower than running transforms separately!
//
// # Current Implementation Strategy
//
// - OutOfPlace operations (Rot90, Rot270, Transpose, Transverse): Delegate to dedicated transforms
// - InPlace operations (Rot180, FlipH, FlipV): Direct implementations (can't delegate due to BarrierImage allocation)

use super::Orientation;
use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};

// Import dedicated transforms for delegation
use crate::transforms::geometric::{HorizontalFlip, Rotate, RotateAngle, Transpose, VerticalFlip};

/// Unified Structural Kernel for geometric transforms
///
/// Fuses multiple geometric operations into a single efficient pass.
/// - Reads pixels once
/// - Writes to final transposed/flipped position
///
/// Total memory traffic: 1 read + 1 write (instead of N reads + N writes)
pub struct StructuralKernel {
    orientation: Orientation,
}

impl Transform for StructuralKernel {
    fn access(&self) -> AccessPattern {
        if self.orientation.preserves_size() {
            AccessPattern::InPlace
        } else {
            AccessPattern::OutOfPlace
        }
    }

    fn shape_effect(&self) -> ShapeEffect {
        if self.orientation.preserves_size() {
            ShapeEffect::Preserve
        } else {
            ShapeEffect::Resize
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for StructuralKernel {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        match self.orientation {
            Orientation::Identity => None, // No-op
            Orientation::Rot90 => Rotate::new(RotateAngle::Rotate90).execute(image),
            Orientation::Rot270 => Rotate::new(RotateAngle::Rotate270).execute(image),
            Orientation::Transpose => Transpose.execute(image),
            Orientation::Transverse => {
                // Transverse = Rot180(Transpose(x))
                // First transpose, then rotate 180
                if let Some(mut transposed) = Transpose.execute(image) {
                    let mut transposed_view = transposed.as_fusable();
                    self.apply_rot180(&mut transposed_view);
                    Some(transposed)
                } else {
                    None // Shouldn't happen
                }
            }
            // InPlace operations - direct implementation (can't delegate without extra allocation)
            Orientation::Rot180 | Orientation::FlipH | Orientation::FlipV => {
                self.apply_in_place(image);
                None
            }
        }
    }
}

impl StructuralKernel {
    /// Create a new structural kernel
    pub fn new(orientation: Orientation) -> Self {
        Self { orientation }
    }

    /// Create identity (no-op) kernel
    pub fn identity() -> Self {
        Self::new(Orientation::Identity)
    }

    /// Apply in-place geometric transformations
    ///
    /// Note: These operations (Rot180, FlipH, FlipV) preserve dimensions and
    /// modify the image in-place, so they can't delegate to the dedicated transforms
    /// which would allocate new buffers.
    fn apply_in_place(&self, image: &mut FusableImage) {
        match self.orientation {
            Orientation::Identity => unreachable!(),
            Orientation::Rot180 => self.apply_rot180(image),
            Orientation::FlipH => self.apply_fliph(image),
            Orientation::FlipV => self.apply_flipv(image),
            _ => unreachable!("Only in-place operations should call this"),
        }
    }

    // Rotation 180° (preserves dimensions, InPlace)
    fn apply_rot180(&self, image: &mut FusableImage) {
        // Rotate 180: reverse pixel order (not byte order)
        // This matches Rotate::execute(RotateAngle::Rotate180)
        let pixel_count = (image.width * image.height) as usize;
        let channels = image.channels;
        let ptr = image.data.as_mut_ptr();
        let mut left = 0;
        let mut right = pixel_count - 1;

        while left < right {
            unsafe {
                std::ptr::swap_nonoverlapping(
                    ptr.add(left * channels),
                    ptr.add(right * channels),
                    channels,
                );
            }
            left += 1;
            right -= 1;
        }
    }

    // Horizontal flip (preserves dimensions, InPlace)
    fn apply_fliph(&self, image: &mut FusableImage) {
        let w = image.width as usize;
        let h = image.height as usize;
        let c = image.channels as usize;
        let row_stride = w * c;
        let data_ptr = image.data.as_mut_ptr();

        for y in 0..h {
            let row_start = y * row_stride;
            let mut left = row_start;
            let mut right = row_start + (w - 1) * c;

            while left < right {
                unsafe {
                    std::ptr::swap_nonoverlapping(data_ptr.add(left), data_ptr.add(right), c);
                }
                left += c;
                right -= c;
            }
        }
    }

    // Vertical flip (preserves dimensions, InPlace)
    fn apply_flipv(&self, image: &mut FusableImage) {
        let w = image.width as usize;
        let h = image.height as usize;
        let c = image.channels as usize;
        let row_stride = w * c;
        let data_ptr = image.data.as_mut_ptr();

        for y in 0..(h / 2) {
            let top_row = y * row_stride;
            let bottom_row = (h - 1 - y) * row_stride;

            unsafe {
                std::ptr::swap_nonoverlapping(
                    data_ptr.add(top_row),
                    data_ptr.add(bottom_row),
                    row_stride,
                );
            }
        }
    }
}
