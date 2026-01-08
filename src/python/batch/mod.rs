// Batch-level transform Python bindings (MixUp, CutMix, Mosaic)
//
// This module provides zero-copy Python bindings for batch transforms.

mod mixup;
mod cutmix;
mod mosaic;

pub use mixup::PyMixUp;
pub use cutmix::PyCutMix;
pub use mosaic::PyMosaic;

use crate::batch::{Batch, BatchPipeline, MixUp, CutMix, Mosaic};
use crate::batch::label::SoftLabel;
use crate::core::BarrierImage;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;
#[cfg(feature = "python")]
use numpy::{PyArray3, PyArray4, PyArray1, PyArray2};
#[cfg(feature = "python")]
use std::sync::Arc;

// =============================================================================
// Shared utilities
// =============================================================================

/// Shared implementation for applying batch transforms
///
/// This function handles the common pattern of:
/// 1. Validating input shapes
/// 2. Converting numpy arrays to Batch format
/// 3. Releasing GIL during transform execution
/// 4. Converting results back to numpy arrays
#[cfg(feature = "python")]
fn apply_batch_transform<'py>(
    py: Python<'py>,
    images: &PyArray4<u8>,
    labels: &PyArray2<f32>,
    transform_fn: impl FnOnce(&mut Batch<SoftLabel>) + Send,
    batch_size_may_change: bool,
) -> PyResult<(&'py PyArray4<u8>, &'py PyArray2<f32>)> {
    let img_shape = images.shape();
    let (batch_size, height, width, channels) = (
        img_shape[0] as usize,
        img_shape[1] as usize,
        img_shape[2] as usize,
        img_shape[3] as usize,
    );

    let label_shape = labels.shape();
    let (label_batch_size, num_classes) = (
        label_shape[0] as usize,
        label_shape[1] as usize,
    );

    if batch_size != label_batch_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("images and labels must have same batch size: got {} images and {} labels",
                batch_size, label_batch_size)
        ));
    }

    if batch_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "batch size cannot be empty"
        ));
    }

    // Borrow the 4D images array as a mutable slice
    let images_slice = unsafe { images.as_slice_mut()? };

    // Borrow the labels array as a slice
    let labels_slice = unsafe { labels.as_slice()? };

    // Convert to format expected by Batch
    // Images: Need to extract each (H, W, C) slice from the (N, H, W, C) array
    let image_slice_size = height * width * channels;

    let mut barrier_images = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let start = i * image_slice_size;
        let end = start + image_slice_size;
        let img_data = &images_slice[start..end];

        // Copy the image data (could be optimized to zero-copy)
        barrier_images.push(BarrierImage::from_vec(img_data.to_vec(), width, height, channels));
    }

    // Convert labels to SoftLabel
    let mut soft_labels = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let start = i * num_classes;
        let end = start + num_classes;
        let label_data = &labels_slice[start..end];
        soft_labels.push(SoftLabel::new(label_data.to_vec()));
    }

    // Create Batch
    let mut batch = Batch::new(barrier_images, soft_labels);

    // Release GIL during transform execution
    py.allow_threads(|| {
        transform_fn(&mut batch);
    });

    // Check if batch size changed (e.g., due to Mosaic)
    let out_batch_size = batch.len();
    let out_num_classes = batch.labels()[0].probs().len();

    // Create output numpy arrays
    if out_batch_size == batch_size && out_num_classes == num_classes {
        // Batch size unchanged - write back to original images array
        let images_slice = unsafe { images.as_slice_mut()? };
        for i in 0..batch_size {
            let start = i * image_slice_size;
            let end = start + image_slice_size;
            let img_slice = &mut images_slice[start..end];
            img_slice.copy_from_slice(batch.image_data(i));
        }

        // Create new numpy array for labels
        let output_labels: Vec<f32> = batch.labels()
            .iter()
            .flat_map(|label| label.probs().iter().copied())
            .collect();

        let labels_1d = PyArray1::from_vec(py, output_labels);
        let labels_array = labels_1d.reshape([batch_size, num_classes])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to reshape labels: {}", e)
            ))?;

        unsafe {
            Ok((
                std::mem::transmute::<&PyArray4<u8>, &'py PyArray4<u8>>(images),
                std::mem::transmute::<&PyArray2<f32>, &'py PyArray2<f32>>(labels_array),
            ))
        }
    } else {
        // Batch size changed - create new arrays
        let output_images: Vec<u8> = batch.images()
            .iter()
            .flat_map(|img| img.data.iter().copied())
            .collect();

        let output_labels: Vec<f32> = batch.labels()
            .iter()
            .flat_map(|label| label.probs().iter().copied())
            .collect();

        let images_1d = PyArray1::from_vec(py, output_images);
        let images_array = images_1d.reshape([out_batch_size, height, width, channels])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to reshape images: {}", e)
            ))?;

        let labels_array = PyArray1::from_vec(py, output_labels)
            .reshape([out_batch_size, out_num_classes])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to reshape labels: {}", e)
            ))?;

        unsafe {
            Ok((
                std::mem::transmute::<&PyArray4<u8>, &'py PyArray4<u8>>(images_array),
                std::mem::transmute::<&PyArray2<f32>, &'py PyArray2<f32>>(labels_array),
            ))
        }
    }
}

// =============================================================================
// BatchPipeline
// =============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "BatchPipeline")]
pub struct PyBatchPipeline {
    inner: BatchPipeline,
    /// Seed value (0 means no seed)
    seed: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBatchPipeline {
    /// Create a new BatchPipeline from a list of batch transforms
    ///
    /// # Arguments
    /// - `transforms`: List of batch transform objects (MixUp, CutMix, Mosaic)
    ///
    /// # Example
    /// ```python
    /// pipeline = BatchPipeline([
    ///     MixUp(alpha=1.0),
    ///     CutMix(alpha=1.0),
    /// ])
    /// ```
    #[new]
    fn new(transforms: &PyAny) -> PyResult<Self> {
        let py = transforms.py();

        // Extract Python list
        let py_list = transforms.extract::<Vec<PyObject>>()?;

        let mut pipeline = BatchPipeline::new();

        for (i, item) in py_list.iter().enumerate() {
            // Try to downcast to each transform type
            if let Ok(mixup) = item.extract::<PyRef<PyMixUp>>(py) {
                pipeline = pipeline.add_mixup((**mixup.inner()).clone());
            } else if let Ok(cutmix) = item.extract::<PyRef<PyCutMix>>(py) {
                pipeline = pipeline.add_cutmix((**cutmix.inner()).clone());
            } else if let Ok(mosaic) = item.extract::<PyRef<PyMosaic>>(py) {
                pipeline = pipeline.add_mosaic(mosaic.inner().clone());
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    format!("transform at index {} is not a valid batch transform (expected MixUp, CutMix, or Mosaic)", i)
                ));
            }
        }

        Ok(PyBatchPipeline {
            inner: pipeline,
            seed: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Set the seed for deterministic RNG behavior
    ///
    /// Once set, all `apply()` calls will use the seeded RNG.
    ///
    /// # Arguments
    /// - `seed`: Seed value for deterministic behavior
    ///
    /// # Example
    /// ```python
    /// pipeline = BatchPipeline([MixUp(1.0)])
    /// pipeline.set_seed(42)
    /// result1 = pipeline.apply(images1, labels1)
    /// result2 = pipeline.apply(images2, labels2)
    /// # Results are deterministic based on seed
    /// ```
    fn set_seed(&self, seed: u64) -> PyResult<()> {
        // Store 0 as "no seed", so if seed is 0, use 1 instead (unlikely conflict)
        let stored = if seed == 0 { 1 } else { seed };
        self.seed.store(stored, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Clear the seed, reverting to non-deterministic RNG
    ///
    /// # Example
    /// ```python
    /// pipeline.clear_seed()
    /// ```
    fn clear_seed(&self) -> PyResult<()> {
        self.seed.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Get the current seed, if set
    ///
    /// Returns None if no seed is set.
    ///
    /// # Example
    /// ```python
    /// seed = pipeline.seed()
    /// if seed is not None:
    ///     print(f"Using seed: {seed}")
    /// ```
    fn seed(&self) -> PyResult<Option<u64>> {
        let s = self.seed.load(std::sync::atomic::Ordering::Relaxed);
        Ok(if s == 0 { None } else { Some(s) })
    }

    /// Apply the batch pipeline to images and labels
    ///
    /// # Arguments
    /// - `images`: 4D numpy array with shape (N, H, W, C), dtype=np.uint8
    /// - `labels`: 2D numpy array with shape (N, num_classes), dtype=np.float32
    ///
    /// # Returns
    /// - `mixed_images`: 4D numpy array (may be modified in-place or new, depending on transforms)
    /// - `mixed_labels`: 2D numpy array (new object)
    ///
    /// # Performance
    /// - **Images**: May be modified in-place or new (depends on transforms)
    /// - **Labels**: New allocation (mixing requires new vectors)
    /// - **GIL**: Released during execution
    ///
    /// # Example
    /// ```python
    /// pipeline = BatchPipeline([MixUp(1.0), CutMix(1.0)])
    /// mixed_images, mixed_labels = pipeline.apply(images, labels)
    /// ```
    fn apply<'py>(
        &self,
        py: Python<'py>,
        images: &numpy::PyArray4<u8>,
        labels: &numpy::PyArray2<f32>,
    ) -> PyResult<(&'py numpy::PyArray4<u8>, &'py numpy::PyArray2<f32>)> {
        let img_shape = images.shape();
        let (batch_size, height, width, channels) = (
            img_shape[0] as usize,
            img_shape[1] as usize,
            img_shape[2] as usize,
            img_shape[3] as usize,
        );

        let label_shape = labels.shape();
        let (label_batch_size, num_classes) = (
            label_shape[0] as usize,
            label_shape[1] as usize,
        );

        if batch_size != label_batch_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("images and labels must have same batch size: got {} images and {} labels",
                    batch_size, label_batch_size)
            ));
        }

        if batch_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch size cannot be empty"
            ));
        }

        // Borrow the 4D images array as a mutable slice
        let images_slice = unsafe { images.as_slice_mut()? };

        // Borrow the labels array as a slice
        let labels_slice = unsafe { labels.as_slice()? };

        // Convert to format expected by Batch
        let image_slice_size = height * width * channels;

        let mut barrier_images = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * image_slice_size;
            let end = start + image_slice_size;
            let img_data = &images_slice[start..end];
            barrier_images.push(BarrierImage::from_vec(img_data.to_vec(), width, height, channels));
        }

        // Convert labels to SoftLabel
        let mut soft_labels = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * num_classes;
            let end = start + num_classes;
            let label_data = &labels_slice[start..end];
            soft_labels.push(SoftLabel::new(label_data.to_vec()));
        }

        // Create Batch
        let mut batch = Batch::new(barrier_images, soft_labels);

        // Check if seed is set for deterministic behavior (lock-free atomic read)
        let seed_value = self.seed.load(std::sync::atomic::Ordering::Relaxed);

        // Release GIL during batch pipeline execution
        py.allow_threads(|| {
            if seed_value != 0 {
                // Use seeded RNG for deterministic behavior
                use rand::SeedableRng;
                let mut rng = ChaCha8Rng::seed_from_u64(seed_value);
                self.inner.apply(&mut batch, &mut rng);
            } else {
                // Use thread_rng for non-deterministic behavior
                let mut rng = rand::thread_rng();
                self.inner.apply(&mut batch, &mut rng);
            }
        });

        // Check if batch size changed (e.g., due to Mosaic)
        let out_batch_size = batch.len();
        let out_num_classes = batch.labels()[0].probs().len();

        // Create output numpy arrays
        if out_batch_size == batch_size && out_num_classes == num_classes {
            // Batch size unchanged - write back to original images array
            let images_slice = unsafe { images.as_slice_mut()? };
            for i in 0..batch_size {
                let start = i * image_slice_size;
                let end = start + image_slice_size;
                let img_slice = &mut images_slice[start..end];
                img_slice.copy_from_slice(batch.image_data(i));
            }

            // Create new numpy array for labels
            let output_labels: Vec<f32> = batch.labels()
                .iter()
                .flat_map(|label| label.probs().iter().copied())
                .collect();

            let labels_1d = numpy::PyArray1::from_vec(py, output_labels);
            let labels_array = labels_1d.reshape([batch_size, num_classes])
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Failed to reshape labels: {}", e)
                ))?;

            unsafe {
                Ok((
                    std::mem::transmute::<&numpy::PyArray4<u8>, &'py numpy::PyArray4<u8>>(images),
                    std::mem::transmute::<&numpy::PyArray2<f32>, &'py numpy::PyArray2<f32>>(labels_array),
                ))
            }
        } else {
            // Batch size changed - create new arrays
            let output_images: Vec<u8> = batch.images()
                .iter()
                .flat_map(|img| img.data.iter().copied())
                .collect();

            let output_labels: Vec<f32> = batch.labels()
                .iter()
                .flat_map(|label| label.probs().iter().copied())
                .collect();

            let images_1d = numpy::PyArray1::from_vec(py, output_images);
            let images_array = images_1d.reshape([out_batch_size, height, width, channels])
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Failed to reshape images: {}", e)
                ))?;

            let labels_array = numpy::PyArray1::from_vec(py, output_labels)
                .reshape([out_batch_size, out_num_classes])
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Failed to reshape labels: {}", e)
                ))?;

            unsafe {
                Ok((
                    std::mem::transmute::<&numpy::PyArray4<u8>, &'py numpy::PyArray4<u8>>(images_array),
                    std::mem::transmute::<&numpy::PyArray2<f32>, &'py numpy::PyArray2<f32>>(labels_array),
                ))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("BatchPipeline(num_transforms={})", self.inner.len())
    }

    /// Get the number of transforms in the pipeline
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
}
