use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use numpy::PyArray1;
use crate::codec::{imread, imread_crop, read_header, imread_bytes, imread_crop_bytes, read_header_bytes};

/// Read an image from disk with native pure-Rust SIMD decoding directly to an RGB8 NumPy array.
///
/// If `crop=(x, y, w, h)` is provided, uses native MCU block-skipping to only decode
/// and run IDCT on blocks overlapping the crop window (2x to 10x faster).
#[pyfunction]
#[pyo3(signature = (path, crop=None))]
pub fn py_imread<'py>(
    py: Python<'py>,
    path: &str,
    crop: Option<(usize, usize, usize, usize)>,
) -> PyResult<&'py PyAny> {
    let img = py.allow_threads(|| {
        if let Some((x, y, w, h)) = crop {
            imread_crop(path, x, y, w, h)
        } else {
            imread(path)
        }
    }).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))?;

    let shape = [img.height, img.width, img.channels];
    let array_1d = PyArray1::from_vec(py, img.data);
    let array_3d = array_1d.reshape(shape)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to reshape: {}", e)))?;

    Ok(array_3d.as_ref())
}

/// Decode in-memory JPEG bytes directly to an RGB8 NumPy array.
#[pyfunction]
#[pyo3(signature = (bytes, crop=None))]
pub fn py_decode_jpeg<'py>(
    py: Python<'py>,
    bytes: &[u8],
    crop: Option<(usize, usize, usize, usize)>,
) -> PyResult<&'py PyAny> {
    let img = py.allow_threads(|| {
        if let Some((x, y, w, h)) = crop {
            imread_crop_bytes(bytes, x, y, w, h)
        } else {
            imread_bytes(bytes)
        }
    }).map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;

    let shape = [img.height, img.width, img.channels];
    let array_1d = PyArray1::from_vec(py, img.data);
    let array_3d = array_1d.reshape(shape)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("Failed to reshape: {}", e)))?;

    Ok(array_3d.as_ref())
}

/// Rapidly extract (width, height, channels) from file header without decoding pixels.
#[pyfunction]
pub fn py_read_header(path: &str) -> PyResult<(usize, usize, usize)> {
    read_header(path).map_err(|e| PyErr::new::<PyIOError, _>(e.to_string()))
}

/// Rapidly extract (width, height, channels) from in-memory byte slice header.
#[pyfunction]
pub fn py_read_header_bytes(bytes: &[u8]) -> PyResult<(usize, usize, usize)> {
    read_header_bytes(bytes).map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))
}
