// Resize transform
//
// Changes image dimensions with configurable interpolation.

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, LabelTransform, ShapeEffect, Transform,
};

#[cfg(target_arch = "aarch64")]
mod neon;

/// Interpolation method for resize
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeInterpolation {
    /// Nearest-neighbor interpolation (fastest, lower quality)
    #[default]
    Nearest,
    /// Bilinear interpolation (good balance of quality and speed, like OpenCV INTER_LINEAR)
    Bilinear,
    /// Bicubic interpolation (higher quality, slower) - not yet implemented, falls back to bilinear
    Bicubic,
    /// Lanczos4 interpolation (highest quality, slowest) - not yet implemented, falls back to bilinear
    Lanczos4,
}

impl ResizeInterpolation {
    /// Convert from i32 (for Python binding compatibility)
    /// 0 = Nearest, 1 = Bilinear, 2 = Bicubic, 3 = Lanczos4
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(ResizeInterpolation::Nearest),
            1 => Some(ResizeInterpolation::Bilinear),
            2 => Some(ResizeInterpolation::Bicubic),
            3 => Some(ResizeInterpolation::Lanczos4),
            _ => None,
        }
    }

    /// Convert to i32 (for Python binding compatibility)
    pub fn to_i32(self) -> i32 {
        match self {
            ResizeInterpolation::Nearest => 0,
            ResizeInterpolation::Bilinear => 1,
            ResizeInterpolation::Bicubic => 2,
            ResizeInterpolation::Lanczos4 => 3,
        }
    }

    /// Convert from string (for Python binding compatibility)
    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_lowercase().as_str() {
            "nearest" => Some(ResizeInterpolation::Nearest),
            "bilinear" => Some(ResizeInterpolation::Bilinear),
            "bicubic" => Some(ResizeInterpolation::Bicubic),
            "lanczos4" | "lanczos" => Some(ResizeInterpolation::Lanczos4),
            _ => None,
        }
    }

    /// Convert to string (for Python binding compatibility)
    pub fn to_str(self) -> &'static str {
        match self {
            ResizeInterpolation::Nearest => "nearest",
            ResizeInterpolation::Bilinear => "bilinear",
            ResizeInterpolation::Bicubic => "bicubic",
            ResizeInterpolation::Lanczos4 => "lanczos4",
        }
    }
}

/// Resize transform
///
/// Changes image dimensions with configurable interpolation method.
///
/// This is a BARRIER transform - it allocates a new buffer because
/// the output size differs from the input size.
///
/// # Parameters
/// - `new_width`: Target width in pixels
/// - `new_height`: Target height in pixels
/// - `interpolation`: Interpolation method (default: Nearest)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resize {
    pub new_width: usize,
    pub new_height: usize,
    pub interpolation: ResizeInterpolation,
}

impl Resize {
    /// Create a new Resize transform with default interpolation (Nearest)
    pub fn new(new_width: usize, new_height: usize) -> Self {
        assert!(new_width > 0, "new_width must be positive");
        assert!(new_height > 0, "new_height must be positive");
        Self {
            new_width,
            new_height,
            interpolation: ResizeInterpolation::default(),
        }
    }

    /// Create a new Resize transform with specified interpolation
    pub fn with_interpolation(
        new_width: usize,
        new_height: usize,
        interpolation: ResizeInterpolation,
    ) -> Self {
        assert!(new_width > 0, "new_width must be positive");
        assert!(new_height > 0, "new_height must be positive");
        Self {
            new_width,
            new_height,
            interpolation,
        }
    }

    /// Apply resize to an image
    ///
    /// Returns a new BarrierImage with the resized data. The original image
    /// is not modified.
    pub fn apply_owned(&self, image: &FusableImage) -> BarrierImage {
        let old_width = image.width;
        let old_height = image.height;
        let channels = image.channels;

        // Allocate new buffer
        let new_size = self.new_width * self.new_height * channels;
        let mut new_data = Vec::with_capacity(new_size);
        unsafe { new_data.set_len(new_size); }

        // Use platform-optimized path
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon::resize_neon(
                &image.data,
                &mut new_data,
                old_width,
                old_height,
                self.new_width,
                self.new_height,
                channels,
                self.interpolation,
            );
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Scalar fallback for other platforms
            match self.interpolation {
                ResizeInterpolation::Nearest => {
                    for dy in 0..self.new_height {
                        let sy = (dy * old_height) / self.new_height;
                        let sy = sy.min(old_height - 1);
                        for dx in 0..self.new_width {
                            let sx = (dx * old_width) / self.new_width;
                            let sx = sx.min(old_width - 1);
                            let src_idx = (sy * old_width + sx) * channels;
                            let dst_idx = (dy * self.new_width + dx) * channels;
                            for c in 0..channels {
                                new_data[dst_idx + c] = image.data[src_idx + c];
                            }
                        }
                    }
                }
                ResizeInterpolation::Bilinear
                | ResizeInterpolation::Bicubic
                | ResizeInterpolation::Lanczos4 => {
                    let x_scale = old_width as f32 / self.new_width as f32;
                    let y_scale = old_height as f32 / self.new_height as f32;

                    for dy in 0..self.new_height {
                        let y_src = (dy as f32 + 0.5) * y_scale - 0.5;
                        let y0_f = y_src.floor();
                        let y0 = if y_src < 0.0 { 0 } else { (y0_f as usize).min(old_height - 1) };
                        let y1 = (y0 + 1).min(old_height - 1);
                        let fy = if y_src < 0.0 || y_src >= (old_height - 1) as f32 { 0.0 } else { y_src - y0_f };

                        for dx in 0..self.new_width {
                            let x_src = (dx as f32 + 0.5) * x_scale - 0.5;
                            let x0_f = x_src.floor();
                            let x0 = if x_src < 0.0 { 0 } else { (x0_f as usize).min(old_width - 1) };
                            let x1 = (x0 + 1).min(old_width - 1);
                            let fx = if x_src < 0.0 || x_src >= (old_width - 1) as f32 { 0.0 } else { x_src - x0_f };

                            for c in 0..channels {
                                let i00 = image.data[((y0 * old_width + x0) * channels) + c] as f32;
                                let i10 = image.data[((y0 * old_width + x1) * channels) + c] as f32;
                                let i01 = image.data[((y1 * old_width + x0) * channels) + c] as f32;
                                let i11 = image.data[((y1 * old_width + x1) * channels) + c] as f32;

                                let val = i00 * (1.0 - fx) * (1.0 - fy)
                                    + i10 * fx * (1.0 - fy)
                                    + i01 * (1.0 - fx) * fy
                                    + i11 * fx * fy;
                                new_data[(dy * self.new_width + dx) * channels + c] = (val.round() as i32).clamp(0, 255) as u8;
                            }
                        }
                    }
                }
            }
        }

        BarrierImage {
            data: new_data,
            f32_data: None,
            width: self.new_width,
            height: self.new_height,
            channels,
            stride: self.new_width * channels,
            alignment: 0,
        }
    }

    /// Apply resize and subsequent LUT transform in a single execution step
    pub fn apply_with_lut(
        &self,
        image: &FusableImage,
        luts_3c: Option<&[[u8; 256]; 3]>,
        lut_1c: &[u8; 256],
    ) -> BarrierImage {
        let old_width = image.width;
        let old_height = image.height;
        let channels = image.channels;
        let new_size = self.new_width * self.new_height * channels;
        let mut new_data = Vec::with_capacity(new_size);
        unsafe { new_data.set_len(new_size); }

        let luts = if let Some(l) = luts_3c {
            *l
        } else {
            [*lut_1c, *lut_1c, *lut_1c]
        };

        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon::resize_with_lut_neon(
                &image.data,
                &mut new_data,
                old_width,
                old_height,
                self.new_width,
                self.new_height,
                channels,
                self.interpolation,
                &luts,
            );
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut barrier = self.apply_owned(image);
            let mut view = FusableImage::new(&mut barrier.data, self.new_width, self.new_height, barrier.channels);
            if let Some(luts) = luts_3c {
                if barrier.channels == 3 {
                    crate::transforms::runtime::lut::LutExecutor::apply_rgb_luts(&mut view, luts);
                } else {
                    crate::transforms::runtime::lut::LutExecutor::apply(&mut view, &luts[0]);
                }
            } else {
                crate::transforms::runtime::lut::LutExecutor::apply(&mut view, lut_1c);
            }
            return barrier;
        }

        BarrierImage {
            data: new_data,
            f32_data: None,
            width: self.new_width,
            height: self.new_height,
            channels,
            stride: self.new_width * channels,
            alignment: 0,
        }
    }
}



impl Transform for Resize {
    fn access(&self) -> AccessPattern {
        // OutOfPlace: requires new buffer due to size change
        AccessPattern::OutOfPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        // Resize: explicitly changes dimensions
        ShapeEffect::Resize
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

impl LabelTransform for Resize {
    fn map_point(&self, point: (f32, f32), image_size: (u32, u32)) -> Option<(f32, f32)> {
        let (x, y) = point;
        let (w, h) = image_size;

        let scale_x = self.new_width as f32 / w as f32;
        let scale_y = self.new_height as f32 / h as f32;

        Some((x * scale_x, y * scale_y))
    }

    fn map_bbox(&self, bbox: [f32; 4], image_size: (u32, u32)) -> Option<[f32; 4]> {
        let [x, y, w_box, h_box] = bbox;
        let (w, h) = image_size;

        let scale_x = self.new_width as f32 / w as f32;
        let scale_y = self.new_height as f32 / h as f32;

        Some([x * scale_x, y * scale_y, w_box * scale_x, h_box * scale_y])
    }
}

impl Executable for Resize {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        Some(self.apply_owned(image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_creation() {
        let resize = Resize::new(100, 200);
        assert_eq!(resize.new_width, 100);
        assert_eq!(resize.new_height, 200);
    }

    #[test]
    fn test_resize_upscale() {
        // 2x2 -> 4x4 upscale
        // 1 2      1 1 2 2
        // 3 4  ->  1 1 2 2
        //          3 3 4 4
        //          3 3 4 4
        let mut data = vec![1u8, 2, 3, 4];
        let img = FusableImage::new(&mut data[..], 2, 2, 1);

        let resize = Resize::new(4, 4);
        let owned = resize.apply_owned(&img);

        assert_eq!(owned.width, 4);
        assert_eq!(owned.height, 4);
        assert_eq!(owned.channels, 1);
        // Check corners
        assert_eq!(owned.data[0], 1); // top-left
        assert_eq!(owned.data[3], 2); // top-right
        assert_eq!(owned.data[12], 3); // bottom-left
        assert_eq!(owned.data[15], 4); // bottom-right
    }

    #[test]
    fn test_resize_downscale() {
        // 4x4 -> 2x2 downscale
        // 1  2  3  4
        // 5  6  7  8
        // 9  10 11 12
        // 13 14 15 16
        //      |
        //      v
        // 1  3
        // 9  11  (nearest-neighbor: (0,0)->(0,0), (1,0)->(2,0), (0,1)->(0,2), (1,1)->(2,2))
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let img = FusableImage::new(&mut data[..], 4, 4, 1);

        let resize = Resize::new(2, 2);
        let owned = resize.apply_owned(&img);

        assert_eq!(owned.width, 2);
        assert_eq!(owned.height, 2);
        assert_eq!(owned.data[0], 1); // top-left corner (from input 0,0)
        assert_eq!(owned.data[1], 3); // top-right corner (from input 2,0)
        assert_eq!(owned.data[2], 9); // bottom-left corner (from input 0,2)
        assert_eq!(owned.data[3], 11); // bottom-right corner (from input 2,2)
    }

    #[test]
    fn test_resize_rgb() {
        // 2x1 RGB -> 4x2 RGB
        // R,G,B R,G,B
        // 10,20,30 40,50,60
        let mut data = vec![10u8, 20, 30, 40, 50, 60];
        let img = FusableImage::new(&mut data[..], 2, 1, 3);

        let resize = Resize::new(4, 2);
        let owned = resize.apply_owned(&img);

        assert_eq!(owned.width, 4);
        assert_eq!(owned.height, 2);
        assert_eq!(owned.channels, 3);
        assert_eq!(owned.data.len(), 4 * 2 * 3);

        // Check first pixel has correct RGB values
        assert_eq!(owned.data[0], 10);
        assert_eq!(owned.data[1], 20);
        assert_eq!(owned.data[2], 30);
    }

    #[test]
    fn test_resize_same_size() {
        // 2x2 -> 2x2 (no change)
        let mut data = vec![1u8, 2, 3, 4];
        let img = FusableImage::new(&mut data[..], 2, 2, 1);

        let resize = Resize::new(2, 2);
        let owned = resize.apply_owned(&img);

        assert_eq!(owned.width, 2);
        assert_eq!(owned.height, 2);
        assert_eq!(owned.data, &[1u8, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "new_width must be positive")]
    fn test_resize_zero_width() {
        Resize::new(0, 100);
    }

    #[test]
    #[should_panic(expected = "new_height must be positive")]
    fn test_resize_zero_height() {
        Resize::new(100, 0);
    }

    #[test]
    fn test_resize_is_barrier() {
        use crate::core::{AccessPattern, ShapeEffect, Transform};

        let resize = Resize::new(100, 200);

        // Should be OutOfPlace + Resize (a barrier)
        assert_eq!(resize.access(), AccessPattern::OutOfPlace);
        assert_eq!(resize.shape_effect(), ShapeEffect::Resize);
    }

    #[test]
    fn test_resize_in_plan() {
        use crate::exec_ir::Optimizer;
        use crate::sampled_ir::ops::{Interpolation, SampledImageOp};
        use crate::sampled_ir::Plan;

        // Create a plan: Brightness -> Resize -> Contrast
        let plan = Plan::from_ops(vec![
            SampledImageOp::Brightness { delta: 10.0 },
            SampledImageOp::Resize {
                width: 50,
                height: 50,
                interpolation: Interpolation::Nearest,
            },
            SampledImageOp::Contrast { factor: 1.2 },
        ]);

        // Optimize - should create fused nodes with a barrier at Resize
        let exec_plan = Optimizer::new().optimize(plan);

        // Should have at least 2 nodes (possibly 3)
        assert!(!exec_plan.is_empty());
    }
}
