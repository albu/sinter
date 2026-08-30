// Pad transform
//
// Adds padding around the image.

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, LabelTransform, ShapeEffect, Transform,
};

#[cfg(target_arch = "aarch64")]
mod neon;

/// Padding mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadMode {
    /// Fill with constant value
    Constant(u8),
    /// Replicate edge pixels
    Replicate,
    /// Reflect padding (mirror)
    Reflect,
    /// Wrap padding (tile image)
    Wrap,
}

/// Pad transform
///
/// Adds padding around the image.
///
/// # Parameters
/// - `top`: Padding pixels at top
/// - `bottom`: Padding pixels at bottom
/// - `left`: Padding pixels at left
/// - `right`: Padding pixels at right
/// - `mode`: Padding mode (constant fill, edge replication, or reflection)
///
/// # Notes
/// - Allocates a new buffer (OutOfPlace)
/// - Increases image dimensions
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pad {
    pub top: u32,
    pub bottom: u32,
    pub left: u32,
    pub right: u32,
    pub mode: PadMode,
}

impl Pad {
    /// Create a new Pad transform
    pub fn new(top: u32, bottom: u32, left: u32, right: u32, mode: PadMode) -> Self {
        Self {
            top,
            bottom,
            left,
            right,
            mode,
        }
    }

    /// Create symmetric padding (same on all sides)
    pub fn symmetric(padding: u32, mode: PadMode) -> Self {
        Self::new(padding, padding, padding, padding, mode)
    }

    /// Create padding with constant fill value
    pub fn with_fill(top: u32, bottom: u32, left: u32, right: u32, fill_value: u8) -> Self {
        Self::new(top, bottom, left, right, PadMode::Constant(fill_value))
    }
}

impl Transform for Pad {
    fn access(&self) -> AccessPattern {
        AccessPattern::OutOfPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Resize // Actually expands, but closest fit
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

impl LabelTransform for Pad {
    fn map_point(&self, point: (f32, f32), _image_size: (u32, u32)) -> Option<(f32, f32)> {
        let (x, y) = point;
        // Shift by left/top padding
        Some((x + self.left as f32, y + self.top as f32))
    }

    fn map_bbox(&self, bbox: [f32; 4], _image_size: (u32, u32)) -> Option<[f32; 4]> {
        let [x, y, w, h] = bbox;
        Some([x + self.left as f32, y + self.top as f32, w, h])
    }
}

impl Executable for Pad {
    #[cfg(target_arch = "aarch64")]
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let new_width = image.width as u32 + self.left + self.right;
        let new_height = image.height as u32 + self.top + self.bottom;
        let channels = image.channels;
        let stride = image.width * channels;

        let len = new_width as usize * new_height as usize * channels;
        let mut padded_data = Vec::with_capacity(len);
        unsafe { padded_data.set_len(len); }

        match self.mode {
            PadMode::Constant(fill) => {
                // Use NEON SIMD for fill when buffer is large enough
                if padded_data.len() >= 16 && (channels == 3 || channels == 1) {
                    unsafe {
                        neon::pad_constant_neon(
                            &mut padded_data,
                            &image.data,
                            new_width as usize,
                            image.width,
                            image.height,
                            self.top as usize,
                            self.left as usize,
                            stride,
                            fill,
                            channels,
                        );
                    }
                } else {
                    pad_constant_scalar(
                        &mut padded_data,
                        &image.data,
                        new_width as usize,
                        image.width,
                        image.height,
                        self.top as usize,
                        self.left as usize,
                        stride,
                        fill,
                        channels,
                    );
                }
            }
            PadMode::Replicate => {
                pad_replicate_scalar(
                    &mut padded_data,
                    &image.data,
                    new_width as usize,
                    image.width,
                    image.height,
                    self.top as usize,
                    self.left as usize,
                    channels,
                );
            }
            PadMode::Reflect => {
                // Use NEON-optimized version for grayscale on ARM64
                if channels == 1 {
                    unsafe {
                        neon::pad_reflect_neon(
                            &mut padded_data,
                            &image.data,
                            new_width as usize,
                            new_height as usize,
                            image.width,
                            image.height,
                            self.top as usize,
                            self.left as usize,
                            channels,
                        );
                    }
                } else {
                    // Fallback to scalar for RGB/other
                    pad_reflect_scalar(
                        &mut padded_data,
                        &image.data,
                        new_width as usize,
                        image.width,
                        image.height,
                        self.top as usize,
                        self.left as usize,
                        channels,
                    );
                }
            }
            PadMode::Wrap => {
                // Wrap padding - tile the image
                pad_wrap_scalar(
                    &mut padded_data,
                    &image.data,
                    new_width as usize,
                    image.width,
                    image.height,
                    self.top as usize,
                    self.left as usize,
                    channels,
                );
            }
        }

        Some(BarrierImage::from_vec(
            padded_data,
            new_width as usize,
            new_height as usize,
            channels,
        ))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let new_width = image.width as u32 + self.left + self.right;
        let new_height = image.height as u32 + self.top + self.bottom;
        let channels = image.channels;
        let stride = image.width * channels;

        let len = new_width as usize * new_height as usize * channels;
        let mut padded_data = Vec::with_capacity(len);
        unsafe { padded_data.set_len(len); }

        match self.mode {
            PadMode::Constant(fill) => {
                pad_constant_scalar(
                    &mut padded_data,
                    &image.data,
                    new_width as usize,
                    image.width,
                    image.height,
                    self.top as usize,
                    self.left as usize,
                    stride,
                    fill,
                    channels,
                );
            }
            PadMode::Replicate => {
                pad_replicate_scalar(
                    &mut padded_data,
                    &image.data,
                    new_width as usize,
                    image.width,
                    image.height,
                    self.top as usize,
                    self.left as usize,
                    channels,
                );
            }
            PadMode::Reflect => {
                pad_reflect_scalar(
                    &mut padded_data,
                    &image.data,
                    new_width as usize,
                    image.width,
                    image.height,
                    self.top as usize,
                    self.left as usize,
                    channels,
                );
            }
            PadMode::Wrap => {
                // Wrap padding - tile the image
                pad_wrap_scalar(
                    &mut padded_data,
                    &image.data,
                    new_width as usize,
                    image.width,
                    image.height,
                    self.top as usize,
                    self.left as usize,
                    channels,
                );
            }
        }

        Some(BarrierImage::from_vec(
            padded_data,
            new_width as usize,
            new_height as usize,
            channels,
        ))
    }
}

// ============================================================================
// Scalar fallback implementations
// ============================================================================

/// Pad with constant value (scalar)
///
/// Optimized to avoid double-write: only fills padding regions, then copies image data.
/// This is critical for cache-resident images (512x512 fits in L2 cache).
pub(crate) fn pad_constant_scalar(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    src_stride: usize,
    fill: u8,
    channels: usize,
) {
    let dst_stride = new_width * channels;
    let new_height = dst.len() / dst_stride;
    let bottom = top + src_height;
    let right = left + src_width;

    // Fill top padding region (entire rows)
    if top > 0 {
        for y in 0..top {
            let row_start = y * dst_stride;
            dst[row_start..row_start + dst_stride].fill(fill);
        }
    }

    // Fill middle section: left padding + image + right padding
    for y in top..bottom {
        let row_start = y * dst_stride;
        let image_start = row_start + left * channels;

        // Fill left padding
        if left > 0 {
            dst[row_start..image_start].fill(fill);
        }

        // Copy image data (this region is NOT pre-filled, avoiding double-write)
        let src_row_start = (y - top) * src_stride;
        let src_row_end = src_row_start + src_stride;
        dst[image_start..image_start + src_stride]
            .copy_from_slice(&src[src_row_start..src_row_end]);

        // Fill right padding
        let image_end = image_start + src_stride;
        let row_end = row_start + dst_stride;
        if image_end < row_end {
            dst[image_end..row_end].fill(fill);
        }
    }

    // Fill bottom padding region (entire rows)
    if bottom < new_height {
        for y in bottom..new_height {
            let row_start = y * dst_stride;
            dst[row_start..row_start + dst_stride].fill(fill);
        }
    }
}

/// Pad with edge replication (scalar)
fn pad_replicate_scalar(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    channels: usize,
) {
    let iw = src_width as i32;
    let ih = src_height as i32;
    let pw = new_width as i32;
    let ph = (dst.len() / (new_width * channels)) as i32;
    let top_i = top as i32;
    let left_i = left as i32;

    for py in 0..ph {
        for px in 0..pw {
            // Map padded coordinates to original image coordinates
            let src_x = (px - left_i).clamp(0, iw - 1);
            let src_y = (py - top_i).clamp(0, ih - 1);

            let src_idx = (src_y * iw + src_x) as usize * channels;
            let dst_idx = (py * pw + px) as usize * channels;

            for c in 0..channels {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }
}

/// Pad with wrap (tile) - scalar implementation
fn pad_wrap_scalar(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    channels: usize,
) {
    let iw = src_width as i32;
    let ih = src_height as i32;
    let pw = new_width as i32;
    let ph = (dst.len() / (new_width * channels)) as i32;
    let top_i = top as i32;
    let left_i = left as i32;

    for py in 0..ph {
        for px in 0..pw {
            // Map padded coordinates to original image coordinates using wrap (modulo)
            let src_x = ((px - left_i).rem_euclid(iw)) as usize;
            let src_y = ((py - top_i).rem_euclid(ih)) as usize;

            let src_idx = (src_y * src_width + src_x) * channels;
            let dst_idx = (py * pw + px) as usize * channels;

            for c in 0..channels {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }
}

/// Pad with reflection (scalar - always available as fallback)
pub(crate) fn pad_reflect_scalar(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    channels: usize,
) {
    let iw = src_width as i32;
    let ih = src_height as i32;
    let pw = new_width as i32;
    let ph = (dst.len() / (new_width * channels)) as i32;
    let top_i = top as i32;
    let left_i = left as i32;

    for py in 0..ph {
        for px in 0..pw {
            // Map padded coordinates to original image coordinates with reflection
            let ox = px - left_i;
            let oy = py - top_i;

            // Reflect coordinates
            let mut src_x = if ox < 0 {
                -ox - 1
            } else if ox >= iw {
                2 * iw - ox - 1
            } else {
                ox
            };
            let mut src_y = if oy < 0 {
                -oy - 1
            } else if oy >= ih {
                2 * ih - oy - 1
            } else {
                oy
            };

            // Clamp to valid range (handle edge cases)
            src_x = src_x.clamp(0, iw - 1);
            src_y = src_y.clamp(0, ih - 1);

            let src_idx = (src_y * iw + src_x) as usize * channels;
            let dst_idx = (py * pw + px) as usize * channels;

            for c in 0..channels {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }
}

#[cfg(test)]
mod tests;
