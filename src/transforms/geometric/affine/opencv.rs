// OpenCV backend for affine transforms
//
// Uses OpenCV's highly optimized warp_affine implementation.

use crate::core::FusableImage;
use crate::transforms::geometric::affine::{Affine, AffineInterpolation, AffineBorderMode};
use crate::core::BarrierImage;

/// Execute using OpenCV backend (when available)
pub(super) fn execute_with_opencv(
    affine: &Affine,
    image: &FusableImage,
) -> Result<BarrierImage, opencv::Error> {
    use opencv::core::{Mat, CV_MAKETYPE, CV_8U, CV_32F, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP};
    use opencv::imgproc::warp_affine;

    let (out_width, out_height) = affine.output_size.unwrap_or((image.width, image.height));
    let channels = image.channels;
    let cv_type = CV_MAKETYPE(CV_8U, channels as i32);

    // Build the 2x3 affine matrix for OpenCV
    let [a, b, c, d, e, f] = affine.build_inverse_matrix();
    let mut mat_data = vec![a as f32, b as f32, c as f32, d as f32, e as f32, f as f32];

    let mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(
            2, 3,
            CV_MAKETYPE(CV_32F, 1),
            mat_data.as_mut_ptr() as *mut std::ffi::c_void,
        )?
    };

    // Convert image data to OpenCV Mat
    let src = unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(
            image.height as i32,
            image.width as i32,
            cv_type,
            image.data.as_ptr() as *const std::ffi::c_void as *mut std::ffi::c_void,
        )?
    };

    // Create output buffer and wrap it
    let mut transformed_data = vec![0u8; out_width * out_height * channels];

    let mut dst = unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(
            out_height as i32,
            out_width as i32,
            cv_type,
            transformed_data.as_mut_ptr() as *mut std::ffi::c_void,
        )?
    };

    // Map interpolation mode to OpenCV constant
    let interpolation = match affine.interpolation {
        AffineInterpolation::Nearest => opencv::imgproc::INTER_NEAREST,
        AffineInterpolation::Bilinear => opencv::imgproc::INTER_LINEAR,
    };

    // Map border mode to OpenCV constant and value
    let (border_mode, border_value) = match affine.border_mode {
        AffineBorderMode::Constant { value } => {
            (BORDER_CONSTANT, opencv::core::Scalar::new(value as f64, value as f64, value as f64, 0.0))
        }
        AffineBorderMode::Reflect => (BORDER_REFLECT, opencv::core::Scalar::default()),
        AffineBorderMode::Replicate => (BORDER_REPLICATE, opencv::core::Scalar::default()),
        AffineBorderMode::Wrap => (BORDER_WRAP, opencv::core::Scalar::default()),
    };

    // Apply affine transform using OpenCV
    warp_affine(
        &src,
        &mut dst,
        &mat,
        opencv::core::Size::new(out_width as i32, out_height as i32),
        interpolation,
        border_mode,
        border_value,
    )?;

    Ok(BarrierImage::from_vec(transformed_data, out_width, out_height, channels))
}
