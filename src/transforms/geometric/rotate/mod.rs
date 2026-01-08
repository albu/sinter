// Rotate transform
//
// Rotates image by multiples of 90 degrees.
//
// OPTIMIZATION: Uses tiled NEON SIMD transpose for 90°/270° rotations.
// The NEON implementation is faster than OpenCV and enables geometric fusion.

#[cfg(target_arch = "aarch64")]
mod neon;
mod tests;

use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform, ReorderRule, LabelTransform};

// Re-export rotation primitives for use by StructuralKernel
#[cfg(target_arch = "aarch64")]
pub use neon::{rotate_90_cw_neon, rotate_270_cw_neon};

/// Rotation angle (multiples of 90 degrees)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RotateAngle {
    Rotate90,
    Rotate180,
    Rotate270,
}

/// Rotate transform
///
/// Rotates the image by 90, 180, or 270 degrees clockwise.
/// These rotations can be done efficiently without interpolation.
///
/// # Parameters
/// - `angle`: Rotation angle (90, 180, or 270 degrees)
///
/// # Notes
/// - Allocates a new buffer (OutOfPlace) for speed (linear memory access)
/// - For 90° and 270° rotations, width and height are swapped
/// - For 180° rotation, uses copy+reverse which is faster than in-place flips
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotate {
    pub angle: RotateAngle,
}

impl Rotate {
    /// Create a new Rotate transform
    pub fn new(angle: RotateAngle) -> Self {
        Self { angle }
    }

    /// Create a 90 degree rotation
    pub fn rotate_90() -> Self {
        Self::new(RotateAngle::Rotate90)
    }

    /// Create a 180 degree rotation
    pub fn rotate_180() -> Self {
        Self::new(RotateAngle::Rotate180)
    }

    /// Create a 270 degree rotation
    pub fn rotate_270() -> Self {
        Self::new(RotateAngle::Rotate270)
    }
}

impl Transform for Rotate {
    fn access(&self) -> AccessPattern {
        // All rotations allocate for speed (linear memory access patterns)
        AccessPattern::OutOfPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        // 90° and 270° swap dimensions (H×W → W×H)
        // 180° preserves dimensions but we still return Resize for consistency
        ShapeEffect::Resize
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

impl LabelTransform for Rotate {
    fn map_point(&self, point: (f32, f32), image_size: (u32, u32)) -> Option<(f32, f32)> {
        let (x, y) = point;
        let (w, h) = image_size;
        
        match self.angle {
            RotateAngle::Rotate90 => {
                // (x, y) -> (h - y, x)
                Some((h as f32 - y, x))
            }
            RotateAngle::Rotate180 => {
                // (x, y) -> (w - x, h - y)
                Some((w as f32 - x, h as f32 - y))
            }
            RotateAngle::Rotate270 => {
                // (x, y) -> (y, w - x)
                Some((y, w as f32 - x))
            }
        }
    }

    fn map_bbox(&self, bbox: [f32; 4], image_size: (u32, u32)) -> Option<[f32; 4]> {
        let [x, y, w_box, h_box] = bbox;
        let (w, h) = image_size;
        
        match self.angle {
            RotateAngle::Rotate90 => {
                // New x is distance from bottom edge (which becomes left edge)
                // New y is old x (left edge becomes top edge)
                // W and H swap
                Some([h as f32 - (y + h_box), x, h_box, w_box])
            }
            RotateAngle::Rotate180 => {
                // Both axes flip
                Some([w as f32 - (x + w_box), h as f32 - (y + h_box), w_box, h_box])
            }
            RotateAngle::Rotate270 => {
                // New x is old y
                // New y is distance from right edge
                // W and H swap
                Some([y, w as f32 - (x + w_box), h_box, w_box])
            }
        }
    }
}

impl Executable for Rotate {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let (new_width, new_height) = match self.angle {
            RotateAngle::Rotate90 | RotateAngle::Rotate270 => (image.height, image.width),
            RotateAngle::Rotate180 => (image.width, image.height),
        };

        let channels = image.channels;
        let mut rotated_data = vec![0u8; (new_width * new_height * channels) as usize];

        match self.angle {
            RotateAngle::Rotate90 => {
                #[cfg(target_arch = "aarch64")]
                {
                    // Use tiled NEON SIMD transpose for cache-friendly access
                    unsafe {
                        rotate_90_cw_neon(
                            &image.data,
                            &mut rotated_data,
                            image.width,
                            image.height,
                            channels,
                        );
                    }
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    // Scalar fallback
                    for y in 0..image.height {
                        for x in 0..image.width {
                            let src_idx = (y * image.width + x) * channels;
                            let dst_idx = (x * new_width + (image.height - 1 - y)) * channels;
                            for c in 0..channels {
                                rotated_data[dst_idx + c] = image.data[src_idx + c];
                            }
                        }
                    }
                }
            }
            RotateAngle::Rotate180 => {
                // Rotate 180°: copy then reverse pixel order (not byte order)
                // Use swap_nonoverlapping for efficient whole-pixel swaps
                rotated_data.copy_from_slice(&image.data);

                let pixel_count = rotated_data.len() / channels;
                let ptr = rotated_data.as_mut_ptr();
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
            RotateAngle::Rotate270 => {
                #[cfg(target_arch = "aarch64")]
                {
                    // Use tiled NEON SIMD transpose for cache-friendly access
                    unsafe {
                        rotate_270_cw_neon(
                            &image.data,
                            &mut rotated_data,
                            image.width,
                            image.height,
                            channels,
                        );
                    }
                }

                #[cfg(not(target_arch = "aarch64"))]
                {
                    // Scalar fallback
                    for y in 0..image.height {
                        for x in 0..image.width {
                            let src_idx = (y * image.width + x) * channels;
                            let dst_idx = ((image.width - 1 - x) * new_width + y) * channels;
                            for c in 0..channels {
                                rotated_data[dst_idx + c] = image.data[src_idx + c];
                            }
                        }
                    }
                }
            }
        }

        Some(BarrierImage::from_vec(rotated_data, new_width, new_height, channels))
    }
}
