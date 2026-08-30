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
    /// - `bboxes`: Optional bounding boxes (N, 4) numpy array
    /// - `keypoints`: Optional keypoints (N, 2) numpy array
    /// - `masks`: Optional masks (H, W) or (H, W, N) numpy array
    /// - `bbox_format`: Format string for bboxes (default: "xywh")
    /// - `keypoint_format`: Format string for keypoints (default: "xy")
    /// - `seed`: Optional random seed for reproducibility (default: random)
    ///
    /// # Returns
    /// Dictionary with keys:
    /// - "image": Transformed image
    /// - "bboxes": Transformed bounding boxes (if input provided)
    /// - "keypoints": Transformed keypoints (if input provided)
    /// - "masks": Transformed masks (if input provided)
    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, bbox_format="xywh", keypoint_format="xy", seed=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&PyArray2<f32>>,
        keypoints: Option<&PyArray2<f32>>,
        masks: Option<&PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        seed: Option<u64>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        use numpy::{PyArray2, PyArray3};
        use pyo3::types::{PyDict, PyList};

        // 1. Sample the pipeline (randomness resolution)
        let seed_value = seed.unwrap_or_else(rand::random);
        let sampled_inner = self.inner.sample_with_seed(seed_value);
        let sampled = PySampledImageProgram {
            inner: sampled_inner,
        };

        // 2. Apply to image (always required)
        let transformed_image_array = sampled.apply(image, py)?;

        // 3. Create result dictionary
        let result_dict = PyDict::new(py);
        result_dict.set_item("image", transformed_image_array)?;

        // Helper to get (width, height)
        let get_image_size = || -> PyResult<(u32, u32)> {
            if let Ok(arr3) = image.downcast::<PyArray3<u8>>() {
                let s = arr3.shape();
                Ok((s[1] as u32, s[0] as u32))
            } else if let Ok(arr2) = image.downcast::<PyArray2<u8>>() {
                let s = arr2.shape();
                Ok((s[1] as u32, s[0] as u32))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Expected 2D or 3D uint8 numpy array",
                ))
            }
        };

        // 4. Transform and add bboxes if present
        if let Some(bbox_array) = bboxes {
            let image_size = get_image_size()?;
            let transformed_bboxes =
                sampled.apply_to_bboxes(bbox_array, image_size, bbox_format, None, py)?;
            result_dict.set_item("bboxes", transformed_bboxes)?;
        }

        // 5. Transform and add keypoints if present
        if let Some(kpt_array) = keypoints {
            let image_size = get_image_size()?;
            let transformed_keypoints =
                sampled.apply_to_keypoints(kpt_array, image_size, keypoint_format, None, py)?;
            result_dict.set_item("keypoints", transformed_keypoints)?;
        }

        // 6. Transform and add masks if present
        if let Some(mask_array) = masks {
            let image_size = get_image_size()?;
            let transformed_masks = sampled.apply_to_masks(mask_array, image_size, py)?;
            result_dict.set_item("masks", transformed_masks)?;
        }

        Ok(result_dict.into())
    }

    /// Apply with new random parameters each call (backward compatible)
    ///
    /// This is a convenience method that returns only the transformed image.
    /// For batch processing with bboxes/keypoints/masks, use `__call__` instead.
    fn apply<'py>(&self, array: &'py PyAny, py: Python<'py>) -> PyResult<&'py PyAny> {
        // Use random seed
        let seed = rand::random();
        let sampled_inner = self.inner.sample_with_seed(seed);
        let sampled = PySampledImageProgram {
            inner: sampled_inner,
        };

        sampled.apply(array, py)
    }

    /// Sample with seed for deterministic reuse
    fn sample_with_seed(&self, seed: u64) -> PyResult<PySampledImageProgram> {
        let sampled = self.inner.sample_with_seed(seed);
        Ok(PySampledImageProgram { inner: sampled })
    }

    fn __repr__(&self) -> String {
        format!("Compose(num_transforms={})", self.inner.len())
    }
}
