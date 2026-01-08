// Horizontal flip (left-right mirror)
//
// Reverses each row of the image in-place.
// This is an INDEX transform - no pixel math, just memory rearrangement.

#[cfg(target_arch = "aarch64")]
mod neon;

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, LabelTransform, ReorderRule,
    ShapeEffect, Transform,
};

/// Horizontal flip (left-right mirror)
///
/// Reverses each row of the image in-place.
///
/// Zero-copy, in-place, cache-friendly.
///
/// # Example
/// For a 3x3 image with values 1-9:
/// ```text
/// 1 2 3      3 2 1
/// 4 5 6  ->  6 5 4
/// 7 8 9      9 8 7
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HorizontalFlip;

impl HorizontalFlip {
    /// Create a new HorizontalFlip transform
    pub fn new() -> Self {
        Self
    }

    /// Apply horizontal flip to an image (optimized with SIMD)
    ///
    /// Zero-copy, in-place, cache-friendly.
    /// - Grayscale: Uses NEON to reverse 16/32-byte chunks
    /// - RGB/other: Falls back to per-pixel swaps (less efficient but correct)
    pub fn apply(&self, image: &mut FusableImage) {
        let w = image.width;
        let h = image.height;
        let c = image.channels;
        let row_stride = w * c;
        let data_ptr = image.data.as_mut_ptr();

        // Use SIMD-optimized path for grayscale
        if c == 1 {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                for y in 0..h {
                    let row_start = y * row_stride;
                    neon::horizontal_flip_gray_neon(data_ptr.add(row_start), w);
                }
                return;
            }
        }

        // Scalar fallback for non-grayscale or non-SIMD platforms
        for y in 0..h {
            let row_start = y * row_stride;
            let mut left = row_start;
            let mut right = row_start + (w - 1) * c;

            // Swap pixels symmetrically within the row
            while left < right {
                unsafe {
                    std::ptr::swap_nonoverlapping(data_ptr.add(left), data_ptr.add(right), c);
                }
                left += c;
                right -= c;
            }
        }
    }
}

impl Default for HorizontalFlip {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for HorizontalFlip {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_executable(&self) -> Option<&dyn crate::core::Executable> {
        Some(self)
    }

    fn as_label_transform(&self) -> Option<&dyn LabelTransform> {
        Some(self)
    }

    fn reorder_rule(&self) -> ReorderRule {
        ReorderRule::Geometry
    }
}

impl LabelTransform for HorizontalFlip {
    fn map_point(&self, point: (f32, f32), image_size: (u32, u32)) -> Option<(f32, f32)> {
        let (x, y) = point;
        let (w, _h) = image_size;
        // Flip x coordinate: new_x = width - x
        Some((w as f32 - x, y))
    }

    fn map_bbox(&self, bbox: [f32; 4], image_size: (u32, u32)) -> Option<[f32; 4]> {
        let [x, y, w, h] = bbox;
        let (img_w, _img_h) = image_size;
        // Flip bbox horizontally: new_x = width - (x + w)
        let new_x = img_w as f32 - (x + w);

        // Check if the flipped box is within bounds
        if new_x < 0.0 || new_x + w > img_w as f32 || y < 0.0 || y + h > _img_h as f32 {
            return None;
        }

        Some([new_x, y, w, h])
    }
}

impl Executable for HorizontalFlip {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        self.apply(image);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizontal_flip_creation() {
        let flip = HorizontalFlip::new();
        assert_eq!(flip, HorizontalFlip);
    }

    #[test]
    fn test_horizontal_flip_default() {
        let flip = HorizontalFlip::default();
        assert_eq!(flip, HorizontalFlip);
    }

    #[test]
    fn test_horizontal_flip_grayscale() {
        // Image:
        // 1 2 3
        // 4 5 6
        let mut data = vec![1u8, 2, 3, 4, 5, 6];
        let mut img = FusableImage::new(&mut data, 3, 2, 1);

        let flip = HorizontalFlip::new();
        flip.apply(&mut img);

        // Expected:
        // 3 2 1
        // 6 5 4
        assert_eq!(img.data, &[3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn test_horizontal_flip_rgb() {
        // 2x2 RGB image:
        // R G B   R G B
        // 1 2 3   4 5 6
        // 7 8 9   10 11 12
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut img = FusableImage::new(&mut data, 2, 2, 3);

        let flip = HorizontalFlip::new();
        flip.apply(&mut img);

        // After flip:
        // 4 5 6   1 2 3
        // 10 11 12  7 8 9
        assert_eq!(img.data, &[4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]);
    }

    #[test]
    fn test_horizontal_flip_odd_width() {
        // 3x3 with odd width (middle column stays):
        // 1 2 3      3 2 1
        // 4 5 6  ->  6 5 4
        // 7 8 9      9 8 7
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let flip = HorizontalFlip::new();
        flip.apply(&mut img);

        assert_eq!(img.data, &[3, 2, 1, 6, 5, 4, 9, 8, 7]);
    }
}
