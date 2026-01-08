// CutMix Python bindings
//
// CutMix creates new training samples by cutting a rectangular region from
// one image and pasting it onto another:
// ```text
// image_i[box_region] = image_j[box_region]
// label = λ * label_i + (1 - λ) * label_j
// ```
//
// where λ = box_area / image_area ~ Beta(α, α).

use crate::batch::{CutMix as RustCutMix, BatchTransform};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::{PyArray4, PyArray2};

/// CutMix batch transform - Python wrapper
///
/// # Python Example
/// ```python
/// from sinter import CutMix
/// import numpy as np
///
/// # Batch of 4 images: (N, H, W, C) format
/// images = np.random.randint(0, 255, (4, 256, 256, 3), dtype=np.uint8)
///
/// # Soft labels: (N, num_classes) format
/// labels = np.array([
///     [1.0, 0.0, 0.0, 0.0],  # class 0
///     [0.0, 1.0, 0.0, 0.0],  # class 1
///     [0.0, 0.0, 1.0, 0.0],  # class 2
///     [0.0, 0.0, 0.0, 1.0],  # class 3
/// ], dtype=np.float32)
///
/// # Create CutMix transform
/// cutmix = CutMix(alpha=1.0)
///
/// # Apply (images modified in-place, labels are new)
/// mixed_images, mixed_labels = cutmix.apply(images, labels)
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "CutMix")]
pub struct PyCutMix {
    inner: std::sync::Arc<RustCutMix>,
}

#[cfg(feature = "python")]
impl PyCutMix {
    pub fn from_inner(inner: std::sync::Arc<RustCutMix>) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &std::sync::Arc<RustCutMix> {
        &self.inner
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCutMix {
    /// Create a new CutMix transform
    ///
    /// # Arguments
    /// - `alpha`: The α parameter for Beta(α, α) distribution
    ///   - Higher values → larger boxes (more mixing)
    ///   - Lower values → smaller boxes (less mixing)
    ///   - Typical values: 0.2 to 2.0
    #[new]
    fn new(alpha: f32) -> PyResult<Self> {
        if alpha <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be positive"
            ));
        }
        Ok(Self {
            inner: std::sync::Arc::new(RustCutMix::new(alpha)),
        })
    }

    /// Apply CutMix to a batch of images and labels
    ///
    /// # Arguments
    /// - `images`: 4D numpy array with shape (N, H, W, C), dtype=np.uint8
    /// - `labels`: 2D numpy array with shape (N, num_classes), dtype=np.float32
    ///
    /// # Returns
    /// - `mixed_images`: 4D numpy array (same object, modified in-place)
    /// - `mixed_labels`: 2D numpy array (new object, mixed label vectors)
    ///
    /// # Performance
    /// - **Images**: Modified in-place (zero-copy view into numpy memory)
    /// - **Labels**: New allocation (mixing requires new vectors)
    /// - **GIL**: Released during execution
    fn apply<'py>(
        &self,
        py: Python<'py>,
        images: &PyArray4<u8>,
        labels: &PyArray2<f32>,
    ) -> PyResult<(&'py PyArray4<u8>, &'py PyArray2<f32>)> {
        // Clone the Arc to move into the closure (thread-safe)
        let cutmix = self.inner.clone();
        super::apply_batch_transform(
            py,
            images,
            labels,
            |batch| {
                let mut rng = rand::thread_rng();
                cutmix.apply(batch, &mut rng);
            },
            false, // batch size unchanged
        )
    }

    fn __repr__(&self) -> String {
        format!("CutMix(alpha={})", self.inner.alpha())
    }

    /// Get the alpha parameter
    #[getter]
    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }
}
