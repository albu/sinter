// OpenCV backend for median blur
//
// Feature-gated implementation using OpenCV's highly optimized medianBlur.

use crate::core::FusableImage;

#[cfg(feature = "opencv")]
use opencv::{core::{Mat, MatTraitConst, CV_MAKETYPE, CV_8U}, imgproc};

/// OpenCV implementation with proper buffer allocation
///
/// CRITICAL: OpenCV's medianBlur does NOT support in-place operation.
/// The src and dst Mats must point to different memory regions.
/// We allocate a temporary buffer, then copy the result back.
#[cfg(feature = "opencv")]
pub fn execute_opencv(image: &mut FusableImage, kernel_size: i32) -> opencv::Result<()> {
    use std::time::Instant;

    let total_start = Instant::now();

    let rows = image.height as i32;
    let cols = image.width as i32;
    let channels = image.channels as i32;
    let cv_type = CV_MAKETYPE(CV_8U, channels);
    let data_size = (rows * cols * channels) as usize;

    unsafe {
        let mat_start = Instant::now();
        // Wrap source image (zero-copy, API requires *mut even for read-only)
        let src_mat = Mat::new_rows_cols_with_data_unsafe_def(
            rows, cols, cv_type,
            image.data.as_ptr() as *mut std::ffi::c_void,
        )?;

        // Allocate temporary buffer for output
        let mut temp_buffer = vec![0u8; data_size];
        let mut dst_mat = Mat::new_rows_cols_with_data_unsafe_def(
            rows, cols, cv_type,
            temp_buffer.as_mut_ptr() as *mut std::ffi::c_void,
        )?;
        let mat_time = mat_start.elapsed();

        let blur_start = Instant::now();
        // Run median blur with separate src and dst buffers
        imgproc::median_blur(&src_mat, &mut dst_mat, kernel_size)?;
        let blur_time = blur_start.elapsed();

        let copy_start = Instant::now();
        // Copy result back to original image
        image.data.copy_from_slice(&temp_buffer);
        let copy_time = copy_start.elapsed();

        let total_time = total_start.elapsed();

        // PROFILING DISABLED
        // if total_time.as_micros() > 50 {
        //     eprintln!("PROFILING: execute_opencv total={:?} mat={:?} blur={:?} copy={:?}",
        //         total_time, mat_time, blur_time, copy_time);
        // }
    }

    Ok(())
}
