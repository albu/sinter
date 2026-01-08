// PyTorch tensor utilities for Python bindings
//
// This module provides utilities for working with PyTorch tensors
// through numpy interop. PyTorch's tensor.numpy() is zero-copy for CPU tensors.

use crate::core::{BarrierImage, FusableImage};
use super::types::barrier_image_to_numpy;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::PyArray3;

/// Apply a transform to a PyTorch tensor in-place (zero-copy)
///
/// This function applies an sinter Compose pipeline directly to a PyTorch tensor's
/// memory without copying. The tensor is modified in-place.
///
/// # Performance
/// ZERO-COPY for CPU tensors - operates directly on tensor's memory.
/// No data is copied between PyTorch and Rust.
///
/// # Arguments
///
/// * `tensor` - PyTorch tensor (must be CPU, HWC, uint8, WILL BE MODIFIED)
/// * `compose` - Compose pipeline to apply
///
/// # Returns
///
/// None (modifies tensor in-place)
///
/// # Example
///
/// ```python
/// import torch
/// from sinter import Compose, Brightness, apply_to_tensor_inplace
///
/// tensor = torch.zeros((100, 100, 3), dtype=torch.uint8)
/// pipeline = Compose([Brightness(delta=50.0)])
/// apply_to_tensor_inplace(tensor, pipeline)
/// ```
#[cfg(feature = "python")]
#[pyfunction]
pub fn apply_to_tensor_inplace(tensor: &PyAny, compose: &PyAny) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        // Convert tensor to numpy (ZERO-COPY for CPU tensors)
        let numpy_array = tensor.call_method0("numpy")?;

        // Convert to PyArray3<u8>
        let arr = numpy_array.downcast::<PyArray3<u8>>()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Tensor must be uint8 dtype with 3 dimensions (HWC)"
            ))?;

        // Call compose.apply() on the numpy array
        compose.call_method1("apply", (arr,))?;

        Ok(())
    })
}
