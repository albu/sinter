// Mosaic Python bindings
//
// Mosaic creates new training samples by stitching 4 images together
// in a 2x2 grid layout:
// ```text
// ┌─────────┬─────────┐
// │  img0   │  img1   │
// │ (top-L) │ (top-R) │
// ├─────────┼─────────┤
// │  img2   │  img3   │
// │ (btm-L) │ (btm-R) │
// └─────────┴─────────┘
// ```

use crate::batch::{Mosaic as RustMosaic, BatchTransform};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::{PyArray4, PyArray2};

/// Mosaic batch transform - Python wrapper
///
/// # Python Example
/// ```python
/// from sinter import Mosaic
/// import numpy as np
///
/// # Batch of 8 images (will produce 2 mosaic outputs)
/// images = np.random.randint(0, 255, (8, 256, 256, 3), dtype=np.uint8)
///
/// # Labels: each is 10-dimensional
/// labels = np.eye(10, dtype=np.float32)[:8]
///
/// # Create Mosaic transform
/// mosaic = Mosaic()
///
/// # Apply - returns 2 images with 40-dimensional labels
/// mixed_images, mixed_labels = mosaic.apply(images, labels)
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "Mosaic")]
pub struct PyMosaic {
    inner: RustMosaic,
}

#[cfg(feature = "python")]
impl PyMosaic {
    pub fn from_inner(inner: RustMosaic) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &RustMosaic {
        &self.inner
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMosaic {
    /// Create a new Mosaic transform.
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: RustMosaic::new(),
        })
    }

    /// Apply Mosaic to a batch of images and labels.
    ///
    /// # Arguments
    /// - `images`: 4D numpy array with shape (N, H, W, C), dtype=np.uint8
    /// - `labels`: 2D numpy array with shape (N, num_classes), dtype=np.float32
    ///
    /// # Returns
    /// - `mixed_images`: 4D numpy array with shape (N/4, H, W, C)
    /// - `mixed_labels`: 2D numpy array with shape (N/4, num_classes*4)
    ///
    /// # Notes
    /// - Batch size must be divisible by 4
    /// - Remaining samples (if N % 4 != 0) are dropped
    ///
    /// # Performance
    /// - **Images**: New allocation (mosaic creates new images)
    /// - **Labels**: New allocation (concatenation)
    /// - **GIL**: Released during execution
    fn apply<'py>(
        &self,
        py: Python<'py>,
        images: &PyArray4<u8>,
        labels: &PyArray2<f32>,
    ) -> PyResult<(&'py PyArray4<u8>, &'py PyArray2<f32>)> {
        // Clone the inner value to move into the closure (thread-safe)
        let mosaic = self.inner.clone();
        super::apply_batch_transform(
            py,
            images,
            labels,
            |batch| {
                let mut rng = rand::thread_rng();
                mosaic.apply(batch, &mut rng);
            },
            true, // batch size may change
        )
    }

    fn __repr__(&self) -> String {
        "Mosaic()".to_string()
    }

    /// Get the group size (always 4 for Mosaic).
    #[getter]
    fn group_size(&self) -> usize {
        self.inner.group_size()
    }
}
