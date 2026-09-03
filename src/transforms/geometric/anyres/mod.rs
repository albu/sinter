// Dynamic Tiling / AnyRes transform for modern Vision-Language Models (VLMs)
//
// Slices an arbitrary-aspect-ratio image into an optimal grid of standard tiles
// (e.g. 448x448 or 384x384) plus an optional global downsampled thumbnail.
//
// Used by modern VLMs including LLaVA-NeXT, Qwen2-VL, InternVL 2.0, and Llama 3.2 Vision.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::geometric::resize::{Resize, ResizeInterpolation};

#[derive(Debug, Clone, PartialEq)]
pub struct AnyRes {
    pub tile_size: u32,
    pub max_tiles: usize,
    pub include_thumbnail: bool,
    pub interpolation: ResizeInterpolation,
}

impl AnyRes {
    pub fn new(tile_size: u32, max_tiles: usize, include_thumbnail: bool, interpolation: ResizeInterpolation) -> Self {
        assert!(tile_size > 0, "tile_size must be positive, got {}", tile_size);
        assert!(max_tiles > 0, "max_tiles must be positive, got {}", max_tiles);
        Self {
            tile_size,
            max_tiles,
            include_thumbnail,
            interpolation,
        }
    }

    /// Select optimal (columns, rows) grid that best fits image aspect ratio within max_tiles budget
    pub fn select_best_grid(&self, orig_w: u32, orig_h: u32) -> (u32, u32) {
        let orig_ar = orig_w as f64 / orig_h as f64;
        let mut best_grid = (1, 1);
        let mut min_error = f64::MAX;

        for m in 1..=self.max_tiles as u32 {
            for n in 1..=self.max_tiles as u32 {
                if (m * n) as usize <= self.max_tiles {
                    let target_ar = m as f64 / n as f64;
                    let error = (orig_ar.ln() - target_ar.ln()).abs();
                    if error < min_error - 1e-5
                        || ((error - min_error).abs() <= 1e-5 && (m * n) > (best_grid.0 * best_grid.1))
                    {
                        min_error = error;
                        best_grid = (m, n);
                    }
                }
            }
        }
        best_grid
    }

    /// Execute AnyRes tiling on an image, returning a stacked 4D buffer of [num_tiles, S, S, C]
    pub fn execute_tiling(&self, image: &FusableImage) -> (usize, usize, usize, usize, Vec<u8>) {
        let (orig_w, orig_h, channels) = (image.width, image.height, image.channels);
        let (cols, rows) = self.select_best_grid(orig_w as u32, orig_h as u32);
        let s = self.tile_size as usize;

        let grid_w = (cols as usize) * s;
        let grid_h = (rows as usize) * s;

        // 1. Resize full image to match the grid resolution
        let resize_op = Resize::with_interpolation(grid_w, grid_h, self.interpolation);
        let mut image_copy = image.data.to_vec();
        let mut fusable_copy = FusableImage::new(&mut image_copy, orig_w, orig_h, channels);
        let resized = resize_op.execute(&mut fusable_copy);
        let resized_data = match &resized {
            Some(b) => &b.data,
            None => &image_copy,
        };

        let num_grid_tiles = (cols * rows) as usize;
        let total_tiles = if self.include_thumbnail { num_grid_tiles + 1 } else { num_grid_tiles };
        let tile_len = s * s * channels;
        let mut out_data = vec![0u8; total_tiles * tile_len];

        // 2. Slice the grid into non-overlapping tiles of size (s, s, channels)
        let grid_row_stride = grid_w * channels;
        let tile_row_bytes = s * channels;

        for r in 0..(rows as usize) {
            for c in 0..(cols as usize) {
                let tile_idx = r * (cols as usize) + c;
                let tile_offset = tile_idx * tile_len;

                for y in 0..s {
                    let src_y = r * s + y;
                    let src_start = src_y * grid_row_stride + c * tile_row_bytes;
                    let src_end = src_start + tile_row_bytes;
                    let dst_start = tile_offset + y * tile_row_bytes;
                    let dst_end = dst_start + tile_row_bytes;
                    out_data[dst_start..dst_end].copy_from_slice(&resized_data[src_start..src_end]);
                }
            }
        }

        // 3. Generate global thumbnail if requested
        if self.include_thumbnail {
            let thumb_op = Resize::with_interpolation(s, s, self.interpolation);
            let mut thumb_copy = image.data.to_vec();
            let mut thumb_fusable = FusableImage::new(&mut thumb_copy, orig_w, orig_h, channels);
            let thumb = thumb_op.execute(&mut thumb_fusable);
            let thumb_data = match &thumb {
                Some(b) => &b.data,
                None => &thumb_copy,
            };

            let thumb_offset = num_grid_tiles * tile_len;
            out_data[thumb_offset..thumb_offset + tile_len].copy_from_slice(&thumb_data[..tile_len]);
        }

        (total_tiles, s, s, channels, out_data)
    }
}

impl Transform for AnyRes {
    fn access(&self) -> AccessPattern {
        AccessPattern::OutOfPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Resize
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for AnyRes {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let (total_tiles, h, w, c, data) = self.execute_tiling(image);
        Some(BarrierImage::from_vec(data, w, total_tiles * h, c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_selection() {
        let anyres = AnyRes::new(448, 6, true, ResizeInterpolation::Bilinear);

        // 16:9 widescreen (e.g. 1920x1080) -> AR ~ 1.77 -> best fits (2, 1) AR = 2.0 (closer than 3:2=1.5)
        let (cols, rows) = anyres.select_best_grid(1920, 1080);
        assert!((cols * rows) <= 6);
        assert_eq!((cols, rows), (2, 1));

        // 9:16 vertical (e.g. 1080x1920) -> AR ~ 0.56 -> best fits (1, 2)
        let (cols, rows) = anyres.select_best_grid(1080, 1920);
        assert_eq!((cols, rows), (1, 2));

        // 1:1 square (e.g. 1000x1000) -> AR = 1.0 -> best fits (2, 2)
        let (cols, rows) = anyres.select_best_grid(1000, 1000);
        assert_eq!((cols, rows), (2, 2));
    }

    #[test]
    fn test_tiling_execution() {
        let anyres = AnyRes::new(64, 4, true, ResizeInterpolation::Nearest);
        let mut data = vec![128u8; 100 * 200 * 3];
        let fusable = FusableImage::new(&mut data, 200, 100, 3);

        let (total_tiles, h, w, c, out) = anyres.execute_tiling(&fusable);
        assert_eq!(h, 64);
        assert_eq!(w, 64);
        assert_eq!(c, 3);
        // (2, 1) grid + 1 thumbnail = 3 tiles
        assert_eq!(total_tiles, 3);
        assert_eq!(out.len(), 3 * 64 * 64 * 3);
    }
}
