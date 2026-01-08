// Python image conversion utilities
//
// Internal utilities for converting between numpy arrays and BarrierImage.
// NOTE: This is kept for internal use only. The public API is Compose.apply().

use crate::core::BarrierImage;
use super::types::{numpy_to_barrier_image, barrier_image_to_numpy};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::PyArray3;

/// Convert a numpy array (HWC format) to a BarrierImage
///
/// # Arguments
/// - `arr`: A 3D numpy array with shape (height, width, channels)
///
/// # Returns
/// A BarrierImage owning the converted data
///
/// # Performance
/// This creates a COPY of the numpy array. Use Compose.apply() for zero-copy operation.
#[cfg(feature = "python")]
pub fn numpy_array_to_barrier_image(arr: &PyArray3<u8>) -> PyResult<BarrierImage> {
    numpy_to_barrier_image(arr)
}

/// Convert a BarrierImage to a numpy array
///
/// # Arguments
/// - `py`: Python token
/// - `img`: The BarrierImage to convert
///
/// # Returns
/// A 3D numpy array with shape (height, width, channels)
#[cfg(feature = "python")]
pub fn barrier_image_to_numpy_array<'py>(
    py: Python<'py>,
    img: &BarrierImage,
) -> PyResult<&'py PyArray3<u8>> {
    barrier_image_to_numpy(py, img)
}
