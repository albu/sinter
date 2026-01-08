// Transpose transform
//
// Transposes the image (swaps x and y axes).
//
// Uses tiled NEON SIMD for optimal performance:
// - 8x8 tiles processed entirely in registers
// - For RGB: deinterleave (vld3) → transpose per-channel → reinterleave (vst3)
// - Same efficient pipeline as Rotate90 but without the flip

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, LabelTransform, ShapeEffect, Transform,
};

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(not(target_arch = "aarch64"))]
mod neon;

/// Transpose transform
///
/// Transposes the image by swapping x and y axes.
/// Equivalent to rotating 90° counter-clockwise then flipping horizontally.
///
/// # Parameters
/// - None
///
/// # Notes
/// - Allocates a new buffer (OutOfPlace)
/// - Swaps width and height
/// - Position (x, y) maps to (y, x) in the transposed image
/// - Uses NEON SIMD for RGB images (8x8 tiled transpose)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transpose;

impl Transpose {
    /// Create a new Transpose transform
    pub fn new() -> Self {
        Self
    }
}

impl Default for Transpose {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Transpose {
    fn access(&self) -> AccessPattern {
        AccessPattern::OutOfPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Resize // Swaps dimensions
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

impl LabelTransform for Transpose {
    fn map_point(&self, point: (f32, f32), _image_size: (u32, u32)) -> Option<(f32, f32)> {
        let (x, y) = point;
        Some((y, x))
    }

    fn map_bbox(&self, bbox: [f32; 4], _image_size: (u32, u32)) -> Option<[f32; 4]> {
        let [x, y, w, h] = bbox;
        Some([y, x, h, w])
    }
}

impl Executable for Transpose {
    #[cfg(target_arch = "aarch64")]
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let new_width = image.height;
        let new_height = image.width;
        let channels = image.channels;
        let src_stride = image.width * channels;
        let dst_stride = new_width * channels;

        let mut transposed_data = vec![0u8; new_width * new_height * channels];

        // Use optimized SIMD path for RGB and grayscale
        if channels == 3 {
            unsafe {
                neon::transpose_rgb_tiled(
                    &image.data,
                    &mut transposed_data,
                    image.width,
                    image.height,
                    src_stride,
                    dst_stride,
                );
            }
        } else if channels == 1 {
            // Grayscale NEON path
            unsafe {
                neon::transpose_gray_tiled(
                    &image.data,
                    &mut transposed_data,
                    image.width,
                    image.height,
                    src_stride,
                    dst_stride,
                );
            }
        } else {
            // Scalar fallback for other channel counts
            transpose_scalar(
                &image.data,
                &mut transposed_data,
                image.width,
                image.height,
                channels,
                src_stride,
                dst_stride,
            );
        }

        Some(BarrierImage::from_vec(
            transposed_data,
            new_width,
            new_height,
            channels,
        ))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let new_width = image.height;
        let new_height = image.width;
        let channels = image.channels;
        let src_stride = image.width * channels;
        let dst_stride = new_width * channels;

        let mut transposed_data = vec![0u8; new_width * new_height * channels];

        transpose_scalar(
            &image.data,
            &mut transposed_data,
            image.width,
            image.height,
            channels,
            src_stride,
            dst_stride,
        );

        Some(BarrierImage::from_vec(
            transposed_data,
            new_width,
            new_height,
            channels,
        ))
    }
}

// ============================================================================
// Scalar fallback (used for non-RGB or non-ARM platforms)
// ============================================================================

/// Scalar transpose with tiled processing for cache locality
fn transpose_scalar(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    // Use 16x16 tiles for better cache locality
    let tile_size = width.min(height).min(16);

    for y0 in (0..height).step_by(tile_size) {
        for x0 in (0..width).step_by(tile_size) {
            let y_max = (y0 + tile_size).min(height);
            let x_max = (x0 + tile_size).min(width);

            for y in y0..y_max {
                for x in x0..x_max {
                    let src_idx = y * src_stride + x * channels;
                    let dst_idx = x * dst_stride + y * channels;

                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr().add(src_idx),
                            dst.as_mut_ptr().add(dst_idx),
                            channels,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
