// PyCompose - pipeline of transforms with unified API
//
// This module provides the Compose class which represents a pipeline
// of transforms with support for images, bboxes, keypoints, and masks.

use crate::sampling::RandomImageNode;
use crate::sampling::RandomImageProgram;

#[cfg(feature = "python")]
use numpy::{PyArray2, PyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "python")]
use super::super::sampled::PySampledImageProgram;
#[cfg(feature = "python")]
use super::random::extract_node;

/// Compose - pipeline of transforms
#[cfg(feature = "python")]
#[pyclass(name = "Compose")]
pub struct PyCompose {
    pub inner: RandomImageProgram,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCompose {
    #[new]
    #[pyo3(signature = (transforms=None))]
    fn new(transforms: Option<&PyAny>) -> PyResult<Self> {
        let mut program = RandomImageProgram::new();

        if let Some(transforms_list) = transforms {
            if let Ok(iter) = transforms_list.iter() {
                for item_result in iter {
                    let item = item_result?;
                    let node = extract_node(item)?;
                    program.add(node);
                }
            }
        }

        Ok(PyCompose { inner: program })
    }

    /// Apply the pipeline to an image and optional targets
    ///
    /// # Arguments
    /// - `image`: Input image (H, W, C) numpy array
    /// - `bboxes`: Optional bounding boxes (N, 4+) numpy array
    /// - `keypoints`: Optional keypoints (N, 2) or (N, 3) numpy array
    /// - `masks`: Optional masks (H, W) or (H, W, C) numpy array
    /// - `bbox_format`: Format string for bboxes (default: "xywh")
    /// - `keypoint_format`: Format string for keypoints (default: "xy")
    /// - `seed`: Optional random seed for reproducibility (default: random)
    /// - `inplace`: Whether to mutate in-place (default: False, safe copy)
    ///
    /// # Returns
    /// Dictionary with keys:
    /// - "image": Transformed image
    /// - "bboxes": Transformed bounding boxes (if input provided)
    /// - "keypoints": Transformed keypoints (if input provided)
    /// - "masks": Transformed masks (if input provided)
    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, bbox_format="xywh", keypoint_format="xy", seed=None, inplace=None))]
    pub(crate) fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&PyArray2<f32>>,
        keypoints: Option<&PyArray2<f32>>,
        masks: Option<&PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        seed: Option<u64>,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let seed_value = seed.unwrap_or_else(rand::random);
        let sampled_inner = self.inner.sample_with_seed(seed_value);
        let sampled = PySampledImageProgram {
            inner: sampled_inner,
        };
        sampled.__call__(
            image,
            bboxes,
            keypoints,
            masks,
            bbox_format,
            keypoint_format,
            inplace,
            py,
        )
    }

    /// Apply with new random parameters each call (backward compatible)
    ///
    /// This is a convenience method that returns only the transformed image.
    /// Default `inplace=False` ensures input arrays are never mutated.
    #[pyo3(signature = (array, inplace=None, seed=None))]
    pub(crate) fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        seed: Option<u64>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let seed_value = seed.unwrap_or_else(rand::random);
        let sampled_inner = self.inner.sample_with_seed(seed_value);
        let sampled = PySampledImageProgram {
            inner: sampled_inner,
        };

        sampled.apply(array, inplace, py)
    }

    /// Apply this pipeline to a batch of images in parallel across CPU cores
    ///
    /// Generates distinct random seeds for each image in the batch to ensure statistical independence.
    #[pyo3(signature = (images, inplace=None, num_threads=None, seed=None))]
    pub fn apply_batch<'py>(
        &self,
        images: &'py PyAny,
        inplace: Option<bool>,
        num_threads: Option<usize>,
        seed: Option<u64>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let base_seed = seed.unwrap_or_else(rand::random);
        let inner = self.inner.clone();
        crate::python::batch::parallel_apply_batch(py, images, inplace, num_threads, move |idx| {
            inner.sample_with_seed(base_seed.wrapping_add(idx as u64))
        })
    }

    /// Sample with seed for deterministic reuse
    fn sample_with_seed(&self, seed: u64) -> PyResult<PySampledImageProgram> {
        let sampled = self.inner.sample_with_seed(seed);
        Ok(PySampledImageProgram { inner: sampled })
    }

    /// Export the compiled execution plan as a Mermaid flowchart markdown string
    #[pyo3(signature = (seed=None, direction="LR"))]
    fn to_mermaid(&self, seed: Option<u64>, direction: &str) -> String {
        let seed_val = seed.unwrap_or(42);
        let sampled = self.inner.sample_with_seed(seed_val);
        sampled.to_mermaid(Some(direction))
    }

    /// Visualize the compiled execution plan graph (prints and returns Mermaid markdown)
    #[pyo3(signature = (seed=None, direction="LR"))]
    fn visualize(&self, seed: Option<u64>, direction: &str) -> String {
        let mermaid = self.to_mermaid(seed, direction);
        println!("{}", mermaid);
        mermaid
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Compose(num_transforms={})", self.inner.len())
    }
}
