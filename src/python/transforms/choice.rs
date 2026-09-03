use crate::sampling::{Dist, RandomImageNode};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::super::distributions::format_dist;
use super::random::{apply_node_to_image, apply_node_to_targets, extract_node, parse_p_dist};

/// Identity - no-op transform that returns the input unchanged
#[cfg(feature = "python")]
#[pyclass(name = "Identity")]
#[derive(Clone)]
pub struct PyIdentity {
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyIdentity {
    #[new]
    #[pyo3(signature = (p=None))]
    pub fn new(p: Option<&PyAny>) -> PyResult<Self> {
        Ok(Self {
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox=None, keypoint=None, bbox_format="xywh", keypoint_format="xy", inplace=None, **kwargs))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox: Option<&'py PyAny>,
        keypoint: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        kwargs: Option<&'py PyDict>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let effective_bboxes = bboxes.or(bbox);
        let effective_keypoints = keypoints.or(keypoint);
        let node = RandomImageNode::Identity;
        let res = apply_node_to_targets(
            node,
            self.p.clone(),
            image,
            effective_bboxes,
            effective_keypoints,
            masks,
            mask,
            bbox_format,
            keypoint_format,
            inplace,
            py,
        )?;
        if let Some(extra) = kwargs {
            if let Ok(dict) = res.extract::<&PyDict>(py) {
                for (k, v) in extra.iter() {
                    dict.set_item(k, v)?;
                }
            }
        }
        Ok(res)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::Identity;
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!("Identity(p={})", format_dist(&self.p))
    }
}

/// Choice - select exactly one candidate transform (unambiguous replacement for OneOf)
#[cfg(feature = "python")]
#[pyclass(name = "Choice")]
#[derive(Clone)]
pub struct PyChoice {
    pub transforms: Vec<PyObject>,
    pub weights: Option<Vec<f32>>,
    pub p: Dist,
}

impl PyChoice {
    pub fn to_node(&self, py: Python) -> PyResult<RandomImageNode> {
        let mut children = Vec::with_capacity(self.transforms.len());
        for t in &self.transforms {
            let child_any = t.as_ref(py);
            let node = extract_node(child_any)?;
            children.push(node);
        }
        Ok(RandomImageNode::OneOf {
            children,
            weights: self.weights.clone(),
        })
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyChoice {
    #[new]
    #[pyo3(signature = (transforms, weights=None, p=None))]
    pub fn new(
        transforms: &PyAny,
        weights: Option<Vec<f32>>,
        p: Option<&PyAny>,
        py: Python,
    ) -> PyResult<Self> {
        let iter = transforms.iter()?;
        let mut py_transforms = Vec::new();
        for item in iter {
            let t = item?;
            // Verify child transform is extractable
            let _ = extract_node(t)?;
            py_transforms.push(t.to_object(py));
        }

        if py_transforms.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Choice requires at least one transform",
            ));
        }

        if let Some(ref w) = weights {
            if w.len() != py_transforms.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "weights length ({}) must match transforms length ({})",
                    w.len(),
                    py_transforms.len()
                )));
            }
            let sum: f32 = w.iter().sum();
            if sum <= 0.0 || w.iter().any(|&x| x < 0.0 || x.is_nan()) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "weights must be non-negative and sum to a positive value",
                ));
            }
        }

        Ok(Self {
            transforms: py_transforms,
            weights,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox=None, keypoint=None, bbox_format="xywh", keypoint_format="xy", inplace=None, **kwargs))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox: Option<&'py PyAny>,
        keypoint: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        kwargs: Option<&'py PyDict>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let effective_bboxes = bboxes.or(bbox);
        let effective_keypoints = keypoints.or(keypoint);
        let node = self.to_node(py)?;
        let res = apply_node_to_targets(
            node,
            self.p.clone(),
            image,
            effective_bboxes,
            effective_keypoints,
            masks,
            mask,
            bbox_format,
            keypoint_format,
            inplace,
            py,
        )?;
        if let Some(extra) = kwargs {
            if let Ok(dict) = res.extract::<&PyDict>(py) {
                for (k, v) in extra.iter() {
                    dict.set_item(k, v)?;
                }
            }
        }
        Ok(res)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = self.to_node(py)?;
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __len__(&self) -> usize {
        self.transforms.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Choice(num_transforms={}, p={})",
            self.transforms.len(),
            format_dist(&self.p)
        )
    }
}
