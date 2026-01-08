// MixUp Python bindings
//
// MixUp creates new training samples by linearly combining pairs of images:
// ```text
// image = λ * image_i + (1 - λ) * image_j
// label = λ * label_i + (1 - λ) * label_j
// ```

use crate::batch::{MixUp as RustMixUp, BatchTransform};
use crate::batch::label::SoftLabel;
use crate::core::BarrierImage;
use crate::batch::Batch;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::{PyArray4, PyArray2};

/// MixUp batch transform - Python wrapper
///
/// # Python Example
/// ```python
/// from sinter import MixUp
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
/// # Create MixUp transform
/// mixup = MixUp(alpha=1.0)
///
/// # Apply (images modified in-place, labels are new)
/// mixed_images, mixed_labels = mixup.apply(images, labels)
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "MixUp")]
pub struct PyMixUp {
    inner: std::sync::Arc<RustMixUp>,
}

#[cfg(feature = "python")]
impl PyMixUp {
    pub fn from_inner(inner: std::sync::Arc<RustMixUp>) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &std::sync::Arc<RustMixUp> {
        &self.inner
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyMixUp {
    /// Create a new MixUp transform
    ///
    /// # Arguments
    /// - `alpha`: The α parameter for Beta(α, α) distribution
    ///   - Higher values → λ near 0.5 (more mixing)
    ///   - Lower values → λ near 0 or 1 (less mixing)
    ///   - Typical values: 0.2 to 2.0
    #[new]
    fn new(alpha: f32) -> PyResult<Self> {
        if alpha <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be positive"
            ));
        }
        Ok(Self {
            inner: std::sync::Arc::new(RustMixUp::new(alpha)),
        })
    }

    /// Apply MixUp to a batch of images and labels
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
        let mixup = self.inner.clone();
        super::apply_batch_transform(
            py,
            images,
            labels,
            |batch| {
                let mut rng = rand::thread_rng();
                mixup.apply(batch, &mut rng);
            },
            false, // batch size unchanged
        )
    }

    fn __repr__(&self) -> String {
        format!("MixUp(alpha={})", self.inner.alpha())
    }

    /// Get the alpha parameter
    #[getter]
    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }
}
