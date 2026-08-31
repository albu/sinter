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
    pub transforms: Vec<PyObject>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCompose {
    #[new]
    #[pyo3(signature = (transforms=None))]
    fn new(transforms: Option<&PyAny>, py: Python) -> PyResult<Self> {
        let mut program = RandomImageProgram::new();
        let mut py_transforms = Vec::new();

        if let Some(transforms_list) = transforms {
            if let Ok(iter) = transforms_list.iter() {
                for item_result in iter {
                    let item = item_result?;
                    let node = extract_node(item)?;
                    program.add(node);
                    py_transforms.push(item.to_object(py));
                }
            }
        }

        Ok(PyCompose {
            inner: program,
            transforms: py_transforms,
        })
    }

    fn __len__(&self) -> usize {
        self.transforms.len().max(self.inner.len())
    }

    fn __getitem__(&self, index: &PyAny, py: Python) -> PyResult<PyObject> {
        if let Ok(idx) = index.extract::<isize>() {
            let len = self.transforms.len() as isize;
            let actual_idx = if idx < 0 { len + idx } else { idx };
            if actual_idx < 0 || actual_idx >= len {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Index out of bounds",
                ));
            }
            return Ok(self.transforms[actual_idx as usize].clone_ref(py));
        }
        if let Ok(slice) = index.downcast::<pyo3::types::PySlice>() {
            let list = PyList::new(py, &self.transforms);
            let sliced_obj = list.as_ref().get_item(slice)?;
            let new_compose = Self::new(Some(sliced_obj), py)?;
            return Ok(new_compose.into_py(py));
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Indices must be integers or slices",
        ))
    }

    fn __add__(&self, other: &PyAny, py: Python) -> PyResult<Self> {
        let mut combined = self.transforms.clone();
        if let Ok(other_compose) = other.extract::<PyRef<Self>>() {
            combined.extend(other_compose.transforms.clone());
        } else if let Ok(iter) = other.iter() {
            for item in iter {
                combined.push(item?.to_object(py));
            }
        } else {
            combined.push(other.to_object(py));
        }
        let list = PyList::new(py, &combined);
        Self::new(Some(list), py)
    }

    fn __radd__(&self, other: &PyAny, py: Python) -> PyResult<Self> {
        let mut combined = Vec::new();
        if let Ok(other_compose) = other.extract::<PyRef<Self>>() {
            combined.extend(other_compose.transforms.clone());
        } else if let Ok(iter) = other.iter() {
            for item in iter {
                combined.push(item?.to_object(py));
            }
        } else {
            combined.push(other.to_object(py));
        }
        combined.extend(self.transforms.clone());
        let list = PyList::new(py, &combined);
        Self::new(Some(list), py)
    }

    fn __iter__(slf: PyRef<Self>, py: Python) -> PyResult<PyObject> {
        let list = PyList::new(py, &slf.transforms);
        list.call_method0("__iter__").map(|iter| iter.to_object(py))
    }

    /// Sample with optional seed (default: random)
    #[pyo3(signature = (seed=None))]
    fn sample(&self, seed: Option<u64>) -> PyResult<PySampledImageProgram> {
        let seed_value = seed.unwrap_or_else(rand::random);
        let sampled = self.inner.sample_with_seed(seed_value);
        Ok(PySampledImageProgram { inner: sampled })
    }

    fn __repr__(&self, py: Python) -> String {
        if self.transforms.is_empty() {
            return format!("Compose(num_transforms={})", self.inner.len());
        }
        let mut lines = Vec::new();
        for t in &self.transforms {
            if let Ok(r) = t.as_ref(py).repr() {
                lines.push(format!("    {}", r));
            }
        }
        format!("Compose([\n{}\n])", lines.join(",\n"))
    }

    /// Explain the execution plan optimization directly from Compose
    #[pyo3(signature = (seed=None))]
    fn explain(&self, seed: Option<u64>) -> String {
        let seed_value = seed.unwrap_or_else(rand::random);
        let sampled = PySampledImageProgram {
            inner: self.inner.sample_with_seed(seed_value),
        };
        sampled.explain()
    }

    /// One-line summary of fusion directly from Compose
    #[pyo3(signature = (seed=None))]
    fn summary(&self, seed: Option<u64>) -> String {
        let seed_value = seed.unwrap_or_else(rand::random);
        self.inner.sample_with_seed(seed_value).summary()
    }

    /// Serialize sampled program to JSON
    #[pyo3(signature = (seed=None))]
    fn to_json(&self, seed: Option<u64>) -> PyResult<String> {
        let seed_value = seed.unwrap_or_else(rand::random);
        self.inner
            .sample_with_seed(seed_value)
            .to_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Apply the pipeline to an image and optional targets
    ///
    /// # Arguments
    /// - `image`: Input image (H, W, C) numpy array or torch.Tensor
    /// - `bboxes`: Optional bounding boxes (N, 4+) numpy array, torch.Tensor, or list
    /// - `keypoints`: Optional keypoints (N, 2) numpy array, torch.Tensor, or list
    /// - `masks`: Optional masks (H, W) or (H, W, C) numpy array or torch.Tensor
    /// - `mask`: Singular alias for `masks`
    /// - `bbox_format`: Format string for bboxes (default: "xywh", also accepts "pascal_voc", "coco", "albumentations", "yolo")
    /// - `keypoint_format`: Format string for keypoints (default: "xy")
    /// - `seed`: Optional random seed for reproducibility (default: random)
    /// - `inplace`: Whether to mutate in-place (default: False, safe copy)
    /// - `labels`: Not a separate argument — classification labels ride as
    ///   extra bbox columns (e.g. an N×5 array with the class id in column 5).
    ///
    /// # Returns
    /// Dictionary with keys:
    /// - "image": Transformed image
    /// - "bboxes": Transformed bounding boxes (if input provided)
    /// - "keypoints": Transformed keypoints (if input provided)
    /// - "masks" / "mask": Transformed mask(s) (if input provided)
    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", seed=None, inplace=None, labels=None))]
    pub(crate) fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        seed: Option<u64>,
        inplace: Option<bool>,
        labels: Option<&'py PyAny>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        if labels.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "labels is not a separate argument: classification labels ride as extra \
                 bbox columns (pass an N×5 array with the class id in column 5, e.g. \
                 bboxes=np.array([[x, y, w, h, class_id]], dtype=np.float32))",
            ));
        }
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
            mask,
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
}
