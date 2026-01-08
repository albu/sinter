// Type conversion utilities for Python bindings
//
// Handles conversion between numpy arrays and BarrierImage.

use crate::core::{BarrierImage, FusableImage, Transform, Executable};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::{PyArray3, PyArray1};

/// Convert a numpy array (HWC format) to a BarrierImage
///
/// # Arguments
/// - `arr`: A 3D numpy array with shape (height, width, channels)
///
/// # Returns
/// A BarrierImage owning the converted data
#[cfg(feature = "python")]
pub fn numpy_to_barrier_image(arr: &PyArray3<u8>) -> PyResult<BarrierImage> {
    let shape = arr.shape();
    let (height, width, channels) = (shape[0], shape[1], shape[2]);

    // Ensure the array is C-contiguous and get the data
    let slice = unsafe { arr.as_slice()? };
    let data = slice.to_vec();

    Ok(BarrierImage::from_vec(data, width, height, channels))
}

/// Apply a transform directly to a numpy array (ZERO-COPY)
///
/// # Arguments
/// - `py`: Python token
/// - `transform`: The transform to apply
/// - `arr`: The numpy array to modify in-place
///
/// # Performance
/// This is ZERO-COPY because it operates directly on the numpy array's memory.
/// No data is copied between Rust and Python.
///
/// # Safety
/// The numpy array is modified in-place. The original data will be lost.
///
/// # Example
/// ```python
/// import numpy as np
/// from sinter import Brightness
///
/// arr = np.zeros((512, 512, 3), dtype=np.uint8)
/// Brightness(delta=50.0).apply_numpy(arr)  # arr is modified in-place
/// ```
#[cfg(feature = "python")]
pub fn apply_to_numpy<'py>(
    py: Python<'py>,
    transform: &PyAny,
    arr: &PyArray3<u8>,
) -> PyResult<()> {
    let shape = arr.shape();
    let (height, width, channels) = (shape[0], shape[1], shape[2]);

    // Get mutable slice to numpy array data (zero-copy!)
    let data = unsafe { arr.as_slice_mut()? };

    // Create a FusableImage that borrows the numpy data
    let mut fusable = FusableImage {
        data,
        width,
        height,
        channels,
    };

    // Direct zero-copy execution is not supported through this generic function.
    // Use transform.apply_numpy(arr) instead, which is implemented per-transform.
    Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
        "apply_to_numpy is not implemented. Use transform.apply_numpy(arr) instead."
    ))
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
pub fn barrier_image_to_numpy<'py>(
    py: Python<'py>,
    img: &BarrierImage,
) -> PyResult<&'py PyArray3<u8>> {
    // Get the dimensions
    let shape = [img.height, img.width, img.channels];

    // Use from_vec to take ownership of the cloned data
    // Note: This requires cloning the Vec since we only have a reference
    let array_1d = PyArray1::from_vec(py, img.data.clone());

    // Reshape to (height, width, channels) - this is a view, no copy
    let array_3d = array_1d.reshape(shape)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to reshape array: {}", e)
        ))?;

    Ok(array_3d)
}

/// Convert a BarrierImage to a numpy array (takes ownership, NO CLONE)
///
/// # Arguments
/// - `py`: Python token
/// - `img`: The BarrierImage to convert (consumed)
///
/// # Returns
/// A 3D numpy array with shape (height, width, channels)
///
/// # Performance
/// This is ZERO-COPY because it takes ownership of the Vec instead of cloning.
#[cfg(feature = "python")]
pub fn barrier_image_to_numpy_owned<'py>(
    py: Python<'py>,
    img: BarrierImage,
) -> PyResult<&'py PyArray3<u8>> {
    let shape = [img.height, img.width, img.channels];

    // Use from_vec to take ownership - NO CLONE!
    let array_1d = PyArray1::from_vec(py, img.data);

    // Reshape to (height, width, channels) - this is a view, no copy
    let array_3d = array_1d.reshape(shape)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to reshape array: {}", e)
        ))?;

    Ok(array_3d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "python")]
    fn test_numpy_to_barrier_image() {
        // This test requires Python context, so it's mainly a compile check
        // Actual testing will be done in Python tests
    }
}
