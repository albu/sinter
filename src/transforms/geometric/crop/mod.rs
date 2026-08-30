// Crop transform
//
// Crops a rectangular region from the image.

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, LabelTransform, ShapeEffect, Transform,
};

#[cfg(target_arch = "aarch64")]
mod neon;

/// Crop transform
///
/// Crops a rectangular region from the image.
///
/// # Parameters
/// - `x`: Left coordinate of crop region (must be >= 0)
/// - `y`: Top coordinate of crop region (must be >= 0)
/// - `width`: Width of crop region (must fit within image)
/// - `height`: Height of crop region (must fit within image)
///
/// # Panics
/// Panics if crop region is invalid or extends beyond image bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Crop {
    /// Create a new Crop transform
    ///
    /// # Panics
    /// Panics if:
    /// - x or y is negative (won't happen with u32)
    /// - width or height is zero
    /// - crop region extends beyond image bounds
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        assert!(width > 0, "width must be positive");
        assert!(height > 0, "height must be positive");
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl Transform for Crop {
    fn access(&self) -> AccessPattern {
        AccessPattern::OutOfPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Crop
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }

    fn as_label_transform(&self) -> Option<&dyn LabelTransform> {
        Some(self)
    }
}

impl LabelTransform for Crop {
    fn map_point(&self, point: (f32, f32), _image_size: (u32, u32)) -> Option<(f32, f32)> {
        let (x, y) = point;
        // Translate to crop coordinates
        let new_x = x - self.x as f32;
        let new_y = y - self.y as f32;

        // Check if inside crop region
        if new_x >= 0.0 && new_x < self.width as f32 && new_y >= 0.0 && new_y < self.height as f32 {
            Some((new_x, new_y))
        } else {
            None
        }
    }

    fn map_bbox(&self, bbox: [f32; 4], _image_size: (u32, u32)) -> Option<[f32; 4]> {
        let [x, y, w, h] = bbox;
        let crop_x = self.x as f32;
        let crop_y = self.y as f32;
        let crop_w = self.width as f32;
        let crop_h = self.height as f32;

        // Intersection logic
        let ix1 = x.max(crop_x);
        let iy1 = y.max(crop_y);
        let ix2 = (x + w).min(crop_x + crop_w);
        let iy2 = (y + h).min(crop_y + crop_h);

        // Check if intersection is valid (non-empty)
        if ix1 >= ix2 || iy1 >= iy2 {
            return None;
        }

        // Return new box relative to crop origin
        Some([ix1 - crop_x, iy1 - crop_y, ix2 - ix1, iy2 - iy1])
    }
}

impl Executable for Crop {
    #[cfg(target_arch = "aarch64")]
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Validate crop region (convert usize to u32 for comparison)
        let img_width = image.width as u32;
        let img_height = image.height as u32;

        assert!(
            self.x + self.width <= img_width,
            "crop region extends beyond image width: {} + {} > {}",
            self.x,
            self.width,
            img_width
        );
        assert!(
            self.y + self.height <= img_height,
            "crop region extends beyond image height: {} + {} > {}",
            self.y,
            self.height,
            img_height
        );

        let cropped_width = self.width as usize;
        let cropped_height = self.height as usize;
        let channels = image.channels;
        let stride = image.width * channels;
        let x_offset = self.x as usize * channels;
        let y_start = self.y as usize;

        let len = cropped_width * cropped_height * channels;
        let mut cropped_data = Vec::<u8>::with_capacity(len);
        unsafe {
            cropped_data.set_len(len);
        }

        // Use NEON SIMD for RGB and grayscale when crop width is sufficient
        if cropped_width >= 16 && (channels == 3 || channels == 1) {
            unsafe {
                neon::crop_neon_simd(
                    &image.data,
                    &mut cropped_data,
                    stride,
                    x_offset,
                    y_start,
                    cropped_width,
                    cropped_height,
                    channels,
                );
            }
        } else {
            // Scalar fallback for small crops or other channel counts
            crop_scalar(
                &image.data,
                &mut cropped_data,
                stride,
                x_offset,
                y_start,
                cropped_width,
                cropped_height,
                channels,
            );
        }

        Some(BarrierImage::from_vec(
            cropped_data,
            cropped_width,
            cropped_height,
            channels,
        ))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        // Validate crop region (convert usize to u32 for comparison)
        let img_width = image.width as u32;
        let img_height = image.height as u32;

        assert!(
            self.x + self.width <= img_width,
            "crop region extends beyond image width: {} + {} > {}",
            self.x,
            self.width,
            img_width
        );
        assert!(
            self.y + self.height <= img_height,
            "crop region extends beyond image height: {} + {} > {}",
            self.y,
            self.height,
            img_height
        );

        let cropped_width = self.width as usize;
        let cropped_height = self.height as usize;
        let channels = image.channels;
        let stride = image.width * channels;
        let x_offset = self.x as usize * channels;
        let y_start = self.y as usize;

        let mut cropped_data = vec![0u8; cropped_width * cropped_height * channels];

        crop_scalar(
            &image.data,
            &mut cropped_data,
            stride,
            x_offset,
            y_start,
            cropped_width,
            cropped_height,
            channels,
        );

        Some(BarrierImage::from_vec(
            cropped_data,
            cropped_width,
            cropped_height,
            channels,
        ))
    }
}

// ============================================================================
// Scalar fallback (used for non-AArch64 or small crops)
// ============================================================================

/// Scalar crop implementation
fn crop_scalar(
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    x_offset: usize,
    y_start: usize,
    cropped_width: usize,
    cropped_height: usize,
    channels: usize,
) {
    let row_bytes = cropped_width * channels;

    for row in 0..cropped_height {
        let src_row_start = (y_start + row) * src_stride + x_offset;
        let dst_row_start = row * row_bytes;
        let src_row_end = src_row_start + row_bytes;

        dst[dst_row_start..dst_row_start + row_bytes]
            .copy_from_slice(&src[src_row_start..src_row_end]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_new() {
        let c = Crop::new(10, 20, 100, 50);
        assert_eq!(c.x, 10);
        assert_eq!(c.y, 20);
        assert_eq!(c.width, 100);
        assert_eq!(c.height, 50);
    }

    #[test]
    #[should_panic(expected = "width must be positive")]
    fn test_crop_zero_width() {
        Crop::new(0, 0, 0, 10);
    }

    #[test]
    #[should_panic(expected = "height must be positive")]
    fn test_crop_zero_height() {
        Crop::new(0, 0, 10, 0);
    }

    #[test]
    fn test_crop_full_image() {
        // Crop the entire image
        let mut data = vec![1u8, 2, 3, 4, 5, 6]; // 2x1 RGB
        let mut img = FusableImage::new(&mut data, 2, 1, 3);

        let result = Crop::new(0, 0, 2, 1).execute(&mut img);

        assert!(result.is_some());
        let cropped = result.unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 1);
        assert_eq!(cropped.data, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_crop_partial() {
        // 3x3 image, crop center 1x1
        // [1, 2, 3]
        // [4, 5, 6]
        // [7, 8, 9]
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let result = Crop::new(1, 1, 1, 1).execute(&mut img);

        assert!(result.is_some());
        let cropped = result.unwrap();
        assert_eq!(cropped.width, 1);
        assert_eq!(cropped.height, 1);
        assert_eq!(cropped.data, &[5]);
    }

    #[test]
    fn test_crop_top_left() {
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let result = Crop::new(0, 0, 2, 2).execute(&mut img);

        assert!(result.is_some());
        let cropped = result.unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        assert_eq!(cropped.data, &[1, 2, 4, 5]);
    }

    #[test]
    fn test_crop_rgb() {
        // 2x2 RGB:
        // [R1,G1,B1] [R2,G2,B2]
        // [R3,G3,B3] [R4,G4,B4]
        let mut data = vec![
            10, 20, 30, // pixel 0
            40, 50, 60, // pixel 1
            70, 80, 90, // pixel 2
            100, 110, 120, // pixel 3
        ];
        let mut img = FusableImage::new(&mut data, 2, 2, 3);

        // Crop top-left 1x1
        let result = Crop::new(0, 0, 1, 1).execute(&mut img);

        assert!(result.is_some());
        let cropped = result.unwrap();
        assert_eq!(cropped.width, 1);
        assert_eq!(cropped.height, 1);
        assert_eq!(cropped.channels, 3);
        assert_eq!(cropped.data, &[10, 20, 30]);
    }

    #[test]
    fn test_crop_bottom_right() {
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        let result = Crop::new(1, 1, 2, 2).execute(&mut img);

        assert!(result.is_some());
        let cropped = result.unwrap();
        assert_eq!(cropped.data, &[5, 6, 8, 9]);
    }

    #[test]
    #[should_panic(expected = "crop region extends beyond image width")]
    fn test_crop_out_of_bounds_width() {
        let mut data = vec![1u8; 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Crop::new(2, 0, 2, 1).execute(&mut img); // 2 + 2 = 4 > 3
    }

    #[test]
    #[should_panic(expected = "crop region extends beyond image height")]
    fn test_crop_out_of_bounds_height() {
        let mut data = vec![1u8; 9];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        Crop::new(0, 2, 1, 2).execute(&mut img); // 2 + 2 = 4 > 3
    }

    #[test]
    fn test_crop_access_pattern() {
        let c = Crop::new(0, 0, 10, 10);
        assert_eq!(c.access(), AccessPattern::OutOfPlace);
        assert_eq!(c.shape_effect(), ShapeEffect::Crop);
    }

    #[test]
    fn test_crop_single_row() {
        let mut data = vec![1u8, 2, 3, 4, 5];
        let mut img = FusableImage::new(&mut data, 5, 1, 1);

        let result = Crop::new(1, 0, 3, 1).execute(&mut img);

        assert!(result.is_some());
        let cropped = result.unwrap();
        assert_eq!(cropped.width, 3);
        assert_eq!(cropped.height, 1);
        assert_eq!(cropped.data, &[2, 3, 4]);
    }

    #[test]
    fn test_crop_single_column() {
        let mut data = vec![1u8, 2, 3, 4, 5];
        let mut img = FusableImage::new(&mut data, 1, 5, 1);

        let result = Crop::new(0, 1, 1, 3).execute(&mut img);

        assert!(result.is_some());
        let cropped = result.unwrap();
        assert_eq!(cropped.width, 1);
        assert_eq!(cropped.height, 3);
        assert_eq!(cropped.data, &[2, 3, 4]);
    }
}
