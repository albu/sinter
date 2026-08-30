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

/// Fast slice-based padding for Replicate, Wrap, and Reflect modes
fn pad_fast_slice(
    dst: &mut [u8],
    src: &[u8],
    new_width: usize,
    src_width: usize,
    src_height: usize,
    top: usize,
    left: usize,
    channels: usize,
    mode: PadMode,
) {
    let iw = src_width as i32;
    let ih = src_height as i32;
    let new_height = dst.len() / (new_width * channels);
    let dst_stride = new_width * channels;
    let src_stride = src_width * channels;

    let map_coord = |coord: i32, max_dim: i32| -> usize {
        let mapped = match mode {
            PadMode::Constant(_) => 0,
            PadMode::Replicate => coord.clamp(0, max_dim - 1),
            PadMode::Wrap => coord.rem_euclid(max_dim),
            PadMode::Reflect => {
                let m = if coord < 0 {
                    -coord - 1
                } else if coord >= max_dim {
                    2 * max_dim - coord - 1
                } else {
                    coord
                };
                m.clamp(0, max_dim - 1)
            }
        };
        mapped as usize
    };

    let mut x_map = Vec::<usize>::with_capacity(new_width);
    for px in 0..new_width {
        let ox = px as i32 - left as i32;
        let sx = map_coord(ox, iw);
        x_map.push(sx * channels);
    }

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    // 1. Fill interior rows (top .. top + src_height)
    for y in 0..src_height {
        let dst_y = top + y;
        let d_row = unsafe { dst_ptr.add(dst_y * dst_stride) };
        let s_row = unsafe { src_ptr.add(y * src_stride) };

        // Left border
        for px in 0..left {
            let sx_byte = unsafe { *x_map.get_unchecked(px) };
            for c in 0..channels {
                unsafe { *d_row.add(px * channels + c) = *s_row.add(sx_byte + c); }
            }
        }

        // Center copy
        unsafe {
            std::ptr::copy_nonoverlapping(s_row, d_row.add(left * channels), src_stride);
        }

        // Right border
        for px in (left + src_width)..new_width {
            let sx_byte = unsafe { *x_map.get_unchecked(px) };
            for c in 0..channels {
                unsafe { *d_row.add(px * channels + c) = *s_row.add(sx_byte + c); }
            }
        }
    }

    // 2. Fill top and bottom rows by row copy from filled interior
    for dst_y in 0..top {
        let oy = dst_y as i32 - top as i32;
        let sy = map_coord(oy, ih);
        let src_row_in_dst = unsafe { dst_ptr.add((top + sy) * dst_stride) };
        let target_row = unsafe { dst_ptr.add(dst_y * dst_stride) };
        unsafe {
            std::ptr::copy_nonoverlapping(src_row_in_dst, target_row, dst_stride);
        }
    }

    for dst_y in (top + src_height)..new_height {
        let oy = dst_y as i32 - top as i32;
        let sy = map_coord(oy, ih);
        let src_row_in_dst = unsafe { dst_ptr.add((top + sy) * dst_stride) };
        let target_row = unsafe { dst_ptr.add(dst_y * dst_stride) };
        unsafe {
            std::ptr::copy_nonoverlapping(src_row_in_dst, target_row, dst_stride);
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
    pad_fast_slice(dst, src, new_width, src_width, src_height, top, left, channels, PadMode::Replicate);
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
    pad_fast_slice(dst, src, new_width, src_width, src_height, top, left, channels, PadMode::Wrap);
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
    pad_fast_slice(dst, src, new_width, src_width, src_height, top, left, channels, PadMode::Reflect);
}

#[cfg(test)]
mod tests;
