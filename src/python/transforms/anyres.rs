// Python bindings for AnyRes / Dynamic Tiling

use crate::core::FusableImage;
use crate::python::tensor::is_torch_tensor;
use crate::transforms::geometric::anyres::AnyRes;
use crate::transforms::geometric::resize::ResizeInterpolation;
use numpy::{PyArray1, PyArray3};
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass(name = "AnyRes")]
pub struct PyAnyRes {
    pub inner: AnyRes,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAnyRes {
    #[new]
    #[pyo3(signature = (tile_size=448, max_tiles=6, include_thumbnail=true, interpolation="bilinear"))]
    fn new(
        tile_size: u32,
        max_tiles: usize,
        include_thumbnail: bool,
        interpolation: &str,
    ) -> PyResult<Self> {
        let interp = match interpolation.to_lowercase().as_str() {
            "nearest" => ResizeInterpolation::Nearest,
            "bilinear" => ResizeInterpolation::Bilinear,
            "bicubic" => ResizeInterpolation::Bicubic,
            "lanczos4" => ResizeInterpolation::Lanczos4,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown interpolation '{}'; choose 'nearest', 'bilinear', 'bicubic', or 'lanczos4'",
                    interpolation
                )))
            }
        };

        Ok(Self {
            inner: AnyRes::new(tile_size, max_tiles, include_thumbnail, interp),
        })
    }

    /// Select optimal (columns, rows) grid for the given image resolution
    #[pyo3(signature = (width, height))]
    fn select_grid(&self, width: u32, height: u32) -> (u32, u32) {
        self.inner.select_best_grid(width, height)
    }

    #[pyo3(signature = (image))]
    fn __call__<'py>(&self, image: &'py PyAny, py: Python<'py>) -> PyResult<&'py PyAny> {
        self.apply(image, py)
    }

    #[pyo3(signature = (image))]
    fn apply<'py>(&self, image: &'py PyAny, py: Python<'py>) -> PyResult<&'py PyAny> {
        // Case 1: PyTorch Tensor [C, H, W] or [H, W, C]
        if is_torch_tensor(image) {
            let shape: Vec<usize> = image.getattr("shape")?.extract()?;
            if shape.len() == 3 {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                if (c == 1 || c == 3 || c == 4) && (h > 4 || w > 4) {
                    // CHW layout -> permute to HWC, apply, permute back to [N, C, S, S]
                    let hwc = image.call_method1("permute", ((1, 2, 0),))?.call_method0("contiguous")?;
                    let numpy_hwc = hwc.call_method0("numpy")?;
                    let out_numpy = self.apply(numpy_hwc, py)?;
                    let torch_mod = py.import("torch")?;
                    let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
                    let res_chw = res_tensor.call_method1("permute", ((0, 3, 1, 2),))?.call_method0("contiguous")?;
                    return Ok(res_chw);
                } else {
                    // HWC layout
                    let cont = image.call_method0("contiguous")?;
                    let numpy_hwc = cont.call_method0("numpy")?;
                    let out_numpy = self.apply(numpy_hwc, py)?;
                    let torch_mod = py.import("torch")?;
                    let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
                    return Ok(res_tensor);
                }
            }
        }

        // Case 2: NumPy PyArray3<u8> (H, W, C)
        if let Ok(arr3) = image.downcast::<PyArray3<u8>>() {
            let shape = arr3.shape();
            let (h, w, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            let slice = unsafe { arr3.as_slice_mut()? };
            let fusable = FusableImage::new(slice, w, h, c);

            let (total_tiles, s_h, s_w, channels, out_data) = py.allow_threads(|| {
                self.inner.execute_tiling(&fusable)
            });

            let arr1 = PyArray1::from_vec(py, out_data);
            let arr4 = arr1.reshape([total_tiles, s_h, s_w, channels]).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to reshape AnyRes output tensor: {}",
                    e
                ))
            })?;
            return Ok(arr4.as_ref());
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected 3D NumPy array (H, W, C) or PyTorch tensor (C, H, W) for AnyRes",
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "AnyRes(tile_size={}, max_tiles={}, include_thumbnail={})",
            self.inner.tile_size, self.inner.max_tiles, self.inner.include_thumbnail
        )
    }
}
