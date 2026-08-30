// Affine geometric transform
//
// Applies affine transformation: scaling, rotation, translation, shearing.
// OPTIMIZATION: Fast Q16.16 fixed-point coordinate stepper and bundled RGB bilinear interpolation.

mod interpolation;
#[cfg(target_arch = "aarch64")]
mod neon;
mod rust_impl;
mod tests;

use crate::core::{
    AccessPattern, BarrierImage, Executable, FusableImage, LabelTransform, ShapeEffect, Transform,
};

/// Interpolation method for affine transform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AffineInterpolation {
    /// Nearest-neighbor interpolation (fastest, for masks/labels)
    #[default]
    Nearest,
    /// Bilinear interpolation (good quality for images)
    Bilinear,
}

impl AffineInterpolation {
    /// Convert from i32 (for Python binding compatibility)
    /// 0 = Nearest, 1 = Bilinear
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(AffineInterpolation::Nearest),
            1 => Some(AffineInterpolation::Bilinear),
            _ => None,
        }
    }

    /// Convert to i32 (for Python binding compatibility)
    pub fn to_i32(self) -> i32 {
        match self {
            AffineInterpolation::Nearest => 0,
            AffineInterpolation::Bilinear => 1,
        }
    }

    /// Convert from string (for Python binding compatibility)
    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_lowercase().as_str() {
            "nearest" => Some(AffineInterpolation::Nearest),
            "bilinear" => Some(AffineInterpolation::Bilinear),
            _ => None,
        }
    }

    /// Convert to string (for Python binding compatibility)
    pub fn to_str(self) -> &'static str {
        match self {
            AffineInterpolation::Nearest => "nearest",
            AffineInterpolation::Bilinear => "bilinear",
        }
    }
}

/// Affine transform parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineParams {
    /// Scale factors (scale_x, scale_y)
    pub scale: (f32, f32),
    /// Rotation in degrees
    pub rotate: f32,
    /// Translation in pixels (translate_x, translate_y)
    pub translate: (f32, f32),
    /// Shear factors (shear_x, shear_y)
    pub shear: (f32, f32),
}

impl Default for AffineParams {
    fn default() -> Self {
        Self {
            scale: (1.0, 1.0),
            rotate: 0.0,
            translate: (0.0, 0.0),
            shear: (0.0, 0.0),
        }
    }
}

/// Border mode for affine transform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AffineBorderMode {
    /// Constant border with specified value (0-255)
    Constant { value: u8 },
    /// Reflect: mirror reflection (fedcba|abcdefgh|hgfedcb)
    Reflect,
    /// Replicate: repeat edge pixel (aaaaa|abcdefgh|hhhhh)
    #[default]
    Replicate,
    /// Wrap: wrap around (bcdefgh|abcdefgh|abcdefg)
    Wrap,
}

/// Affine transform
///
/// Applies a 2D affine transformation combining scaling, rotation,
/// translation, and shearing. Uses configurable interpolation for resampling.
///
/// # Parameters
/// - `params`: Affine transformation parameters
/// - `output_size`: Optional (width, height) for output. If None, uses input size.
/// - `interpolation`: Interpolation method (default: Nearest for masks, use Bilinear for images)
/// - `border_mode`: How to handle out-of-bounds pixels (default: Replicate)
///
/// # Notes
/// - Allocates a new buffer (OutOfPlace)
#[derive(Debug, Clone, PartialEq)]
pub struct Affine {
    pub params: AffineParams,
    pub output_size: Option<(usize, usize)>,
    pub interpolation: AffineInterpolation,
    pub border_mode: AffineBorderMode,
}

impl Affine {
    /// Create a new Affine transform
    pub fn new(params: AffineParams) -> Self {
        Self {
            params,
            output_size: None,
            interpolation: AffineInterpolation::default(),
            border_mode: AffineBorderMode::default(),
        }
    }

    /// Create a new Affine transform with specified output size
    pub fn with_output_size(params: AffineParams, width: usize, height: usize) -> Self {
        Self {
            params,
            output_size: Some((width, height)),
            interpolation: AffineInterpolation::default(),
            border_mode: AffineBorderMode::default(),
        }
    }

    /// Create a new Affine transform with specified interpolation
    pub fn with_interpolation(params: AffineParams, interpolation: AffineInterpolation) -> Self {
        Self {
            params,
            output_size: None,
            interpolation,
            border_mode: AffineBorderMode::default(),
        }
    }

    /// Create a new Affine transform with specified output size and interpolation
    pub fn with_output_size_and_interpolation(
        params: AffineParams,
        width: usize,
        height: usize,
        interpolation: AffineInterpolation,
    ) -> Self {
        Self {
            params,
            output_size: Some((width, height)),
            interpolation,
            border_mode: AffineBorderMode::default(),
        }
    }

    /// Create a new Affine transform with specified border mode
    pub fn with_border_mode(params: AffineParams, border_mode: AffineBorderMode) -> Self {
        Self {
            params,
            output_size: None,
            interpolation: AffineInterpolation::default(),
            border_mode,
        }
    }

    /// Create a new Affine transform with all parameters specified
    pub fn with_all(
        params: AffineParams,
        width: usize,
        height: usize,
        interpolation: AffineInterpolation,
        border_mode: AffineBorderMode,
    ) -> Self {
        let output_size = if width > 0 && height > 0 {
            Some((width, height))
        } else {
            None
        };
        Self {
            params,
            output_size,
            interpolation,
            border_mode,
        }
    }

    /// Build the inverse affine transformation matrix
    ///
    /// Returns 3x3 matrix in row-major order for inverse mapping:
    /// [a, b, c]
    /// [d, e, f]
    /// [0, 0, 1]
    pub(super) fn build_inverse_matrix(&self, in_width: usize, in_height: usize) -> [f32; 6] {
        let sx = self.params.scale.0;
        let sy = self.params.scale.1;
        let angle = self.params.rotate.to_radians();
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let tx = self.params.translate.0;
        let ty = self.params.translate.1;
        let shx = self.params.shear.0.to_radians().tan();
        let shy = self.params.shear.1.to_radians().tan();

        let cx = (in_width.saturating_sub(1)) as f32 / 2.0;
        let cy = (in_height.saturating_sub(1)) as f32 / 2.0;

        let det_shear = 1.0 - shx * shy;
        let inv_det_shear = if det_shear.abs() > 1e-6 { 1.0 / det_shear } else { 1.0 };

        let inv_sx = if sx.abs() > 1e-6 { 1.0 / sx } else { 1.0 };
        let inv_sy = if sy.abs() > 1e-6 { 1.0 / sy } else { 1.0 };

        let a = (cos_a + sin_a * shy) * inv_det_shear * inv_sx;
        let b = (-cos_a * shx - sin_a) * inv_det_shear * inv_sx;
        let d = (sin_a - cos_a * shy) * inv_det_shear * inv_sy;
        let e = (-sin_a * shx + cos_a) * inv_det_shear * inv_sy;

        let target_x = cx + tx;
        let target_y = cy + ty;
        let c = cx - (a * target_x + b * target_y);
        let f = cy - (d * target_x + e * target_y);

        [a, b, c, d, e, f]
    }

    /// Build the forward affine transformation matrix
    ///
    /// Returns 3x3 matrix in row-major order for forward mapping:
    /// [a, b, c]
    /// [d, e, f]
    /// [0, 0, 1]
    fn build_forward_matrix(&self) -> [f32; 6] {
        let cx = self.params.scale.0;
        let cy = self.params.scale.1;
        let angle = self.params.rotate.to_radians();
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let tx = self.params.translate.0;
        let ty = self.params.translate.1;

        // M = T * R * S
        // [cx*cos, -cy*sin, tx]
        // [cx*sin,  cy*cos, ty]

        [cx * cos_a, -cy * sin_a, tx, cx * sin_a, cy * cos_a, ty]
    }
}

impl Transform for Affine {
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

    fn as_label_transform(&self) -> Option<&dyn LabelTransform> {
        Some(self)
    }
}

impl LabelTransform for Affine {
    fn map_point(&self, point: (f32, f32), image_size: (u32, u32)) -> Option<(f32, f32)> {
        let (x, y) = point;
        let m = self.build_forward_matrix();
        let nx = m[0] * x + m[1] * y + m[2];
        let ny = m[3] * x + m[4] * y + m[5];

        // Output dimensions
        let (w, h) = self
            .output_size
            .unwrap_or((image_size.0 as usize, image_size.1 as usize));

        if nx >= 0.0 && nx < w as f32 && ny >= 0.0 && ny < h as f32 {
            Some((nx, ny))
        } else {
            None
        }
    }

    fn map_bbox(&self, bbox: [f32; 4], image_size: (u32, u32)) -> Option<[f32; 4]> {
        let [x, y, w, h] = bbox;
        let m = self.build_forward_matrix();

        // Map 4 corners
        let corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)];

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for (cx, cy) in corners {
            let nx = m[0] * cx + m[1] * cy + m[2];
            let ny = m[3] * cx + m[4] * cy + m[5];

            if nx < min_x {
                min_x = nx;
            }
            if nx > max_x {
                max_x = nx;
            }
            if ny < min_y {
                min_y = ny;
            }
            if ny > max_y {
                max_y = ny;
            }
        }

        // Output dimensions
        let (out_w, out_h) = self
            .output_size
            .unwrap_or((image_size.0 as usize, image_size.1 as usize));
        let max_w = out_w as f32;
        let max_h = out_h as f32;

        // Clip to image bounds
        let ix1 = min_x.max(0.0);
        let iy1 = min_y.max(0.0);
        let ix2 = max_x.min(max_w);
        let iy2 = max_y.min(max_h);

        if ix1 >= ix2 || iy1 >= iy2 {
            return None;
        }

        Some([ix1, iy1, ix2 - ix1, iy2 - iy1])
    }
}

impl Executable for Affine {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        #[cfg(target_arch = "aarch64")]
        {
            Some(neon::execute_neon(self, image))
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Some(rust_impl::execute_rust(self, image))
        }
    }
}

// Re-export interpolation helpers for tests
pub(crate) use interpolation::{bilinear_interpolate, nearest_interpolate};
