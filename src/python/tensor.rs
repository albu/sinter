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

        // Call compose.apply(arr, inplace=True) on the numpy array
        let initial_shape = arr.shape().to_vec();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("inplace", true)?;
        let result_obj = compose.call_method("apply", (arr,), Some(kwargs))?;

        if let Ok(res_arr) = result_obj.downcast::<PyArray3<u8>>() {
            if res_arr.shape() != initial_shape.as_slice() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "apply_to_tensor_inplace requires shape-preserving transforms; shape changed from {:?} to {:?}",
                    initial_shape, res_arr.shape()
                )));
            }
        }

        Ok(())
    })
}

/// Check if a PyAny object is a PyTorch tensor
#[cfg(feature = "python")]
pub fn is_torch_tensor(obj: &PyAny) -> bool {
    if let Ok(type_name) = obj.get_type().name() {
        if type_name == "Tensor" {
            return true;
        }
    }
    obj.hasattr("numpy").unwrap_or(false)
        && obj.hasattr("dim").unwrap_or(false)
        && obj.hasattr("is_cuda").unwrap_or(false)
}

/// Process a PyTorch tensor with a Sinter executor callback
///
/// Handles CHW / HWC layout, contiguous checking, CPU validation, and returns a matching PyTorch tensor.
#[cfg(feature = "python")]
pub fn handle_torch_tensor<'py, F>(
    tensor: &'py PyAny,
    inplace: Option<bool>,
    py: Python<'py>,
    exec_numpy: F,
) -> PyResult<&'py PyAny>
where
    F: FnOnce(&'py PyAny, Option<bool>, Python<'py>) -> PyResult<&'py PyAny>,
{
    if let Ok(is_cuda) = tensor.getattr("is_cuda") {
        if is_cuda.extract::<bool>().unwrap_or(false) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sinter operates on CPU memory; pass a CPU tensor (e.g., tensor.cpu())",
            ));
        }
    }

    let dtype_str = tensor.getattr("dtype")?.to_string();
    if !dtype_str.contains("uint8") && !dtype_str.contains("Byte") {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "PyTorch tensor must have uint8 dtype for image transforms, got {}",
            dtype_str
        )));
    }

    let shape: Vec<usize> = tensor.getattr("shape")?.extract()?;
    let is_inplace = inplace.unwrap_or(false);

    if shape.len() == 3 {
        let (c, h, w) = (shape[0], shape[1], shape[2]);
        // Check if layout is CHW (channels first, e.g. 1 or 3 channels)
        if (c == 1 || c == 3 || c == 4) && (h > 4 || w > 4) {
            // Permute CHW -> HWC
            let hwc_tensor = tensor
                .call_method1("permute", ((1, 2, 0),))?
                .call_method0("contiguous")?;
            let numpy_arr = hwc_tensor.call_method0("numpy")?;
            let out_numpy = exec_numpy(numpy_arr, Some(is_inplace), py)?;
            let torch_mod = py.import("torch")?;
            let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
            let chw_res = res_tensor
                .call_method1("permute", ((2, 0, 1),))?
                .call_method0("contiguous")?;
            Ok(chw_res)
        } else {
            // HWC layout
            let cont_tensor = if is_inplace {
                tensor
            } else {
                tensor.call_method0("contiguous")?
            };
            let numpy_arr = cont_tensor.call_method0("numpy")?;
            let out_numpy = exec_numpy(numpy_arr, Some(is_inplace), py)?;
            let torch_mod = py.import("torch")?;
            let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
            Ok(res_tensor)
        }
    } else if shape.len() == 2 {
        let cont_tensor = if is_inplace {
            tensor
        } else {
            tensor.call_method0("contiguous")?
        };
        let numpy_arr = cont_tensor.call_method0("numpy")?;
        let out_numpy = exec_numpy(numpy_arr, Some(is_inplace), py)?;
        let torch_mod = py.import("torch")?;
        let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
        Ok(res_tensor)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "PyTorch tensor must be 2D (H, W) or 3D (C, H, W) / (H, W, C), got shape {:?}",
            shape
        )))
    }
}
