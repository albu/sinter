// Vertical flip (up-down mirror)
//
// Reverses the rows of the image in-place.

use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform, ReorderRule, LabelTransform};

/// Vertical flip (up-down mirror)
///
/// Reverses the rows of the image in-place.
///
/// # Example
/// For a 3x3 image with values 1-9:
/// ```text
/// 1 2 3      7 8 9
/// 4 5 6  ->  4 5 6
/// 7 8 9      1 2 3
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerticalFlip;

impl VerticalFlip {
    /// Create a new VerticalFlip transform
    pub fn new() -> Self {
        Self
    }

    /// Apply vertical flip to an image (optimized)
    pub fn apply(&self, image: &mut FusableImage) {
        let row_stride = image.row_stride();
        let half_height = image.height / 2;

        // Use unsafe to get two mutable references to non-overlapping regions
        let data_ptr = image.data.as_mut_ptr();

        for row in 0..half_height {
            let top_row_start = row * row_stride;
            let bottom_row_start = (image.height - 1 - row) * row_stride;

            unsafe {
                // Create two mutable slices for the rows to swap
                // These regions don't overlap, so this is safe
                let top_row = std::slice::from_raw_parts_mut(data_ptr.add(top_row_start), row_stride);
                let bottom_row = std::slice::from_raw_parts_mut(data_ptr.add(bottom_row_start), row_stride);

                // Swap entire rows at once
                top_row.swap_with_slice(bottom_row);
            }
        }
    }
}

impl Default for VerticalFlip {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for VerticalFlip {
    fn access(&self) -> AccessPattern {
        // InPlace: mutates the buffer directly
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        // Preserve: dimensions don't change
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

impl LabelTransform for VerticalFlip {
    fn map_point(&self, point: (f32, f32), image_size: (u32, u32)) -> Option<(f32, f32)> {
        let (x, y) = point;
        let (_w, h) = image_size;
        // Flip y coordinate: new_y = height - y
        Some((x, h as f32 - y))
    }

    fn map_bbox(&self, bbox: [f32; 4], image_size: (u32, u32)) -> Option<[f32; 4]> {
        let [x, y, w, h_box] = bbox;
        let (_w, h_img) = image_size;
        // Flip bbox vertically: new_y = height - (y + h)
        Some([x, h_img as f32 - (y + h_box), w, h_box])
    }
}

impl Executable for VerticalFlip {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        self.apply(image);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertical_flip_creation() {
        let flip = VerticalFlip::new();
        assert_eq!(flip, VerticalFlip);
    }

    #[test]
    fn test_vertical_flip_default() {
        let flip = VerticalFlip::default();
        assert_eq!(flip, VerticalFlip);
    }

    #[test]
    fn test_vertical_flip_grayscale() {
        // Image:
        // 1 2 3
        // 4 5 6
        let mut data = vec![1u8, 2, 3, 4, 5, 6];
        let mut img = FusableImage::new(&mut data, 3, 2, 1);

        let flip = VerticalFlip::new();
        flip.apply(&mut img);

        // Expected:
        // 4 5 6
        // 1 2 3
        assert_eq!(img.data, &[4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn test_vertical_flip_rgb() {
        // 2x2 RGB image:
        // R G B   R G B
        // 1 2 3   4 5 6
        // 7 8 9   10 11 12
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut img = FusableImage::new(&mut data, 2, 2, 3);

        let flip = VerticalFlip::new();
        flip.apply(&mut img);

        // After flip:
        // 7 8 9   10 11 12
        // 1 2 3   4 5 6
        assert_eq!(img.data, &[7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_vertical_flip_odd_height() {
        // 3x3 with odd height (middle row stays):
        // 1 2 3      7 8 9
        // 4 5 6  ->  4 5 6
        // 7 8 9      1 2 3
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let flip = VerticalFlip::new();
        flip.apply(&mut img);

        assert_eq!(img.data, &[7, 8, 9, 4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn test_both_flips_sequentially() {
        // Apply horizontal then vertical flip
        // 1 2       2 1       4 3
        // 3 4  ->   4 3   ->  2 1
        let mut data = vec![1u8, 2, 3, 4];
        let mut img = FusableImage::new(&mut data, 2, 2, 1);

        super::super::horizontal_flip::HorizontalFlip::new().apply(&mut img);
        VerticalFlip::new().apply(&mut img);

        // Should be equivalent to 180-degree rotation
        assert_eq!(img.data, &[4, 3, 2, 1]);
    }

    #[test]
    fn test_flip_is_inplace() {
        use crate::core::{AccessPattern, ShapeEffect, Transform};

        let v_flip = VerticalFlip::new();

        // Should be InPlace + Preserve
        assert_eq!(v_flip.access(), AccessPattern::InPlace);
        assert_eq!(v_flip.shape_effect(), ShapeEffect::Preserve);
    }
}
