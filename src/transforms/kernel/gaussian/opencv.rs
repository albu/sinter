// OpenCV backend for Gaussian blur with sigma
//
// Feature-gated implementation using OpenCV's highly optimized GaussianBlur.

use crate::core::FusableImage;

#[cfg(feature = "opencv")]
use opencv::{
    core::{Mat, MatTraitConst, Size, CV_8U, CV_MAKETYPE},
    imgproc,
};

/// OpenCV implementation with zero-copy data wrapping
#[cfg(feature = "opencv")]
pub fn execute_opencv(image: &mut FusableImage, sigma_x: f32, sigma_y: f32) -> opencv::Result<()> {
    let rows = image.height as i32;
    let cols = image.width as i32;
    let channels = image.channels as i32;
    let cv_type = CV_MAKETYPE(CV_8U, channels);

    // Calculate kernel size the same way OpenCV does: round(6*sigma) to nearest odd
    // Use sigma_x for both (isotropic blur)
    let ksize_raw = (6.0 * sigma_x as f64).round() as i32;
    let ksize = if ksize_raw % 2 == 0 {
        ksize_raw + 1
    } else {
        ksize_raw
    };
    let ksize = Size::new(ksize, ksize);

    unsafe {
        let src_mat = Mat::new_rows_cols_with_data_unsafe_def(
            rows,
            cols,
            cv_type,
            image.data.as_mut_ptr() as *mut std::ffi::c_void,
        )?;
        let mut dst_mat = Mat::new_rows_cols_with_data_unsafe_def(
            rows,
            cols,
            cv_type,
            image.data.as_mut_ptr() as *mut std::ffi::c_void,
        )?;

        // sigmaX and sigmaY are used directly (not auto-calculated)
        // Using explicit value 4 for BORDER_DEFAULT/BORDER_REFLECT_101 to avoid import ambiguity if any
        // Note: AlgorithmHint required by new opencv-rust
        imgproc::gaussian_blur(
            &src_mat,
            &mut dst_mat,
            ksize,
            sigma_x as f64,
            sigma_y as f64,
            4,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    }

    Ok(())
}
