// PySampledImageProgram wrapper

use crate::core::FusableImage;
use crate::exec_ir::Optimizer;
use crate::sampled_ir::{Plan, SampledImageProgram, IR_VERSION};
use crate::labels::{BBoxFormat, KeypointArray, KeypointFormat};

/// Fail with a clean ValueError (instead of a Rust panic that pyo3 surfaces as
/// a PanicException and kills DataLoader workers) when any Crop barrier in the
/// optimized plan extends beyond the image.
fn validate_crop_bounds(
    exec_plan: &crate::exec_ir::ExecPlan,
    width: usize,
    height: usize,
) -> PyResult<()> {
    use crate::exec_ir::ExecNodeKind;
    use crate::sampled_ir::ops::SampledImageOp;
    for node in &exec_plan.nodes {
        if let ExecNodeKind::Barrier(SampledImageOp::Crop {
            x, y, width: w, height: h,
        }) = &node.kind
        {
            let (x, y, w, h) = (*x as usize, *y as usize, *w as usize, *h as usize);
            if x + w > width || y + h > height {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "crop region ({}, {}, {}x{}) exceeds image size ({}x{}); \
                     sample x within [0, {}] and y within [0, {}]",
                    x,
                    y,
                    w,
                    h,
                    width,
                    height,
                    width.saturating_sub(w),
                    height.saturating_sub(h)
                )));
            }
        } else if let ExecNodeKind::Barrier(SampledImageOp::RandomCrop {
            width: w, height: h, ..
        }) = &node.kind
        {
            let (w, h) = (*w as usize, *h as usize);
            if w > width || h > height {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "RandomCrop window {}x{} exceeds image size ({}x{}); \
                     the window must fit inside the image",
                    w, h, width, height
                )));
            }
        }
    }

    // Normalize produces float32 output and terminates the pipeline: no
    // transform can run after it.
    let mut normalize_seen = false;
    for node in &exec_plan.nodes {
        let has_normalize = match &node.kind {
            ExecNodeKind::Barrier(SampledImageOp::Normalize { .. }) => true,
            ExecNodeKind::Fused(ops) => {
                ops.iter().any(|op| matches!(op, SampledImageOp::Normalize { .. }))
            }
            _ => false,
        };
        if has_normalize {
            normalize_seen = true;
        } else if normalize_seen {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Normalize produces float32 output and must be the last transform \
                 in the pipeline",
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, PyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use std::path::PathBuf;

/// _SampledImageProgram - deterministic, serializable transform program (internal)
///
/// This is the output of the sampling phase and input to the optimizer.
/// All parameters are fixed, all randomness resolved.
///
/// # API
///
/// This is returned by `Compose.sample_with_seed()` and represents a fully
/// sampled, optimizable, serializable image transformation pipeline.
#[cfg(feature = "python")]
#[pyclass(name = "SampledImageProgram")]
pub struct PySampledImageProgram {
    pub inner: SampledImageProgram,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySampledImageProgram {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("SampledImageProgram(ops={})", self.inner.len())
    }

    fn __getitem__(&self, index: isize) -> PyResult<String> {
        let len = self.inner.len() as isize;
        let actual_idx = if index < 0 { len + index } else { index };
        if actual_idx < 0 || actual_idx >= len {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"));
        }
        let op = &self.inner.ops[actual_idx as usize];
        Ok(format!("{:?}", op))
    }

    fn __iter__(slf: PyRef<Self>, py: Python) -> PyResult<PyObject> {
        let ops_list: Vec<String> = slf.inner.iter().map(|op| format!("{:?}", op)).collect();
        let list = PyList::new(py, ops_list);
        list.call_method0("__iter__").map(|iter| iter.to_object(py))
    }

    /// Number of operations in the program
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Is the program empty?
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get IR version
    fn version(&self) -> u32 {
        self.inner.version
    }

    /// Serialize to JSON (for inspection)
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Deserialize from JSON
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let program = SampledImageProgram::from_json(json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner: program })
    }

    /// Serialize to bytes (binary format)
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        self.inner
            .to_bytes()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Deserialize from bytes
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let program = SampledImageProgram::from_bytes(data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner: program })
    }

    /// Save to file
    fn save(&self, path: &str) -> PyResult<()> {
        let path_buf = PathBuf::from(path);
        self.inner
            .save(&path_buf)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Load from file
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let path_buf = PathBuf::from(path);
        let program = SampledImageProgram::load(&path_buf)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner: program })
    }

    /// Get a summary of the program
    pub(crate) fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Convert to dictionary representation (for Python inspection)
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("version", self.inner.version)?;
            dict.set_item("num_ops", self.inner.len())?;

            let ops: Vec<String> = self.inner.iter().map(|op| op.name().to_string()).collect();
            dict.set_item("ops", ops)?;

            Ok(dict.into())
        })
    }

    /// Get the execution plan after fusion optimization
    ///
    /// Returns a dictionary with information about how transforms are fused:
    /// - "num_ops": number of input transforms
    /// - "num_nodes": number of execution nodes after fusion
    /// - "fusion_ratio": percentage of transforms fused (0.0 to 1.0)
    /// - "nodes": list of node types and their sizes
    ///
    /// # Example
    /// ```python
    /// sampled = pipeline.sample_with_seed(42)
    /// plan = sampled.execution_plan()
    /// print(f"{plan['num_ops']} transforms → {plan['num_nodes']} execution nodes")
    /// ```
    fn execution_plan(&self) -> PyResult<PyObject> {
        use crate::exec_ir::ExecNodeKind;
        use crate::exec_ir::Optimizer;
        use crate::sampled_ir::Plan;

        // Convert to plan and optimize
        let plan = self.inner.to_plan();
        let exec_plan = Optimizer::new().optimize(plan);

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let num_ops = self.inner.len();
            let num_nodes = exec_plan.len();
            let fusion_ratio = if num_ops > 0 {
                (num_ops - num_nodes) as f64 / num_ops as f64
            } else {
                0.0
            };

            dict.set_item("num_ops", num_ops)?;
            dict.set_item("num_nodes", num_nodes)?;
            dict.set_item("fusion_ratio", fusion_ratio)?;

            // Build node info (just type and count)
            let nodes_list = pyo3::types::PyList::empty(py);
            for node in &exec_plan.nodes {
                let node_dict = PyDict::new(py);
                match &node.kind {
                    ExecNodeKind::Fused(ops) => {
                        node_dict.set_item("type", "Fused")?;
                        node_dict.set_item("count", ops.len())?;
                    }
                    ExecNodeKind::Barrier(_) => {
                        node_dict.set_item("type", "Barrier")?;
                        node_dict.set_item("count", 1)?;
                    }
                }
                nodes_list.append(node_dict)?;
            }
            dict.set_item("nodes", nodes_list)?;

            Ok(dict.into())
        })
    }

    /// Get a human-readable explanation of the execution plan
    ///
    /// Shows how transforms are fused for efficient execution.
    ///
    /// # Example
    /// ```python
    /// sampled = pipeline.sample_with_seed(42)
    /// print(sampled.explain())
    /// ```
    pub(crate) fn explain(&self) -> String {
        use crate::exec_ir::ExecNodeKind;
        use crate::exec_ir::Optimizer;
        use crate::sampled_ir::Plan;

        // Convert to plan and optimize
        let plan = self.inner.to_plan();
        let exec_plan = Optimizer::new().optimize(plan);

        let num_ops = self.inner.len();
        let num_nodes = exec_plan.len();
        let fusion_pct = if num_ops > 0 {
            ((num_ops - num_nodes) as f64 / num_ops as f64 * 100.0) as i32
        } else {
            0
        };

        let mut result = String::new();
        result.push_str("Execution Plan:\n");
        result.push_str(&format!(
            "  {} transforms → {} execution nodes ({}% fusion)\n",
            num_ops, num_nodes, fusion_pct
        ));
        result.push_str("\n");

        for (i, node) in exec_plan.nodes.iter().enumerate() {
            match &node.kind {
                ExecNodeKind::Fused(ops) => {
                    result.push_str(&format!("Node {}: Fused({} ops)\n", i + 1, ops.len()));
                }
                ExecNodeKind::Barrier(_) => {
                    result.push_str(&format!("Node {}: Barrier\n", i + 1));
                }
            }
        }

        result
    }

    /// Export the compiled execution plan as a Mermaid flowchart markdown string
    #[pyo3(signature = (direction="LR"))]
    fn to_mermaid(&self, direction: &str) -> String {
        self.inner.to_mermaid(Some(direction))
    }

    /// Visualize the compiled execution plan graph (prints and returns Mermaid markdown)
    #[pyo3(signature = (direction="LR"))]
    fn visualize(&self, direction: &str) -> String {
        let mermaid = self.inner.to_mermaid(Some(direction));
        println!("{}", mermaid);
        mermaid
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    /// Apply this sampled program to an image and optional targets
    ///
    /// # Arguments
    /// - `image`: Input image (H, W, C) numpy array
    /// - `bboxes`: Optional bounding boxes (N, 4+) numpy array
    /// - `keypoints`: Optional keypoints (N, 2) or (N, 3) numpy array
    /// - `masks`: Optional masks (H, W) or (H, W, C) numpy array
    /// - `bbox_format`: Format string for bboxes (default: "xywh")
    /// - `keypoint_format`: Format string for keypoints (default: "xy")
    /// - `inplace`: Whether to mutate in-place (default: False, safe copy)
    ///
    /// # Returns
    /// Dictionary with keys: "image", and optionally "bboxes", "keypoints", "masks"
    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None, optimize=None))]
    pub(crate) fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        optimize: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let transformed_image_array = self.apply(image, inplace, optimize, py)?;

        let result_dict = PyDict::new(py);
        result_dict.set_item("image", transformed_image_array)?;

        let get_image_size = || -> PyResult<(u32, u32)> {
            if let Ok(arr3) = image.downcast::<PyArray3<u8>>() {
                let s = arr3.shape();
                Ok((s[1] as u32, s[0] as u32))
            } else if let Ok(arr2) = image.downcast::<PyArray2<u8>>() {
                let s = arr2.shape();
                Ok((s[1] as u32, s[0] as u32))
            } else if crate::python::tensor::is_torch_tensor(image) {
                let shape: Vec<usize> = image.getattr("shape")?.extract()?;
                if shape.len() == 3 {
                    if (shape[0] == 1 || shape[0] == 3 || shape[0] == 4) && (shape[1] > 4 || shape[2] > 4) {
                        // CHW layout: (channels, height, width)
                        Ok((shape[2] as u32, shape[1] as u32))
                    } else {
                        // HWC layout: (height, width, channels)
                        Ok((shape[1] as u32, shape[0] as u32))
                    }
                } else if shape.len() == 2 {
                    Ok((shape[1] as u32, shape[0] as u32))
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unexpected tensor shape: {:?}",
                        shape
                    )))
                }
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Expected 2D or 3D uint8 numpy array or torch.Tensor",
                ))
            }
        };

        if let Some(bbox_obj) = bboxes {
            let image_size = get_image_size()?;
            let (bbox_array, is_torch, is_list) = extract_2d_coords(bbox_obj, 4, py)?;
            if bbox_array.shape()[0] == 0 {
                if is_list {
                    result_dict.set_item("bboxes", PyList::empty(py))?;
                } else if is_torch {
                    let torch_mod = py.import("torch")?;
                    let empty_tensor = torch_mod.call_method1("from_numpy", (bbox_array,))?;
                    result_dict.set_item("bboxes", empty_tensor)?;
                } else {
                    result_dict.set_item("bboxes", bbox_array)?;
                }
            } else {
                let transformed_bboxes =
                    self.apply_to_bboxes(bbox_array, image_size, bbox_format, None, py)?;
                if is_list {
                    result_dict.set_item("bboxes", transformed_bboxes.call_method0("tolist")?)?;
                } else if is_torch {
                    let torch_mod = py.import("torch")?;
                    let res_tensor = torch_mod.call_method1("from_numpy", (transformed_bboxes,))?;
                    result_dict.set_item("bboxes", res_tensor)?;
                } else {
                    result_dict.set_item("bboxes", transformed_bboxes)?;
                }
            }
        }

        if let Some(kpt_obj) = keypoints {
            let image_size = get_image_size()?;
            let (kpt_array, is_torch, is_list) = extract_2d_coords(kpt_obj, 2, py)?;
            if kpt_array.shape()[0] == 0 {
                if is_list {
                    result_dict.set_item("keypoints", PyList::empty(py))?;
                } else if is_torch {
                    let torch_mod = py.import("torch")?;
                    let empty_tensor = torch_mod.call_method1("from_numpy", (kpt_array,))?;
                    result_dict.set_item("keypoints", empty_tensor)?;
                } else {
                    result_dict.set_item("keypoints", kpt_array)?;
                }
            } else {
                let transformed_keypoints =
                    self.apply_to_keypoints(kpt_array, image_size, keypoint_format, None, py)?;
                if is_list {
                    result_dict.set_item("keypoints", transformed_keypoints.call_method0("tolist")?)?;
                } else if is_torch {
                    let torch_mod = py.import("torch")?;
                    let res_tensor = torch_mod.call_method1("from_numpy", (transformed_keypoints,))?;
                    result_dict.set_item("keypoints", res_tensor)?;
                } else {
                    result_dict.set_item("keypoints", transformed_keypoints)?;
                }
            }
        }

        let mask_target = if masks.is_some() { masks } else { mask };
        if let Some(mask_array) = mask_target {
            let image_size = get_image_size()?;
            let transformed_masks = self.apply_to_masks(mask_array, image_size, inplace, py)?;
            if mask.is_some() && masks.is_none() {
                result_dict.set_item("mask", transformed_masks)?;
            } else {
                result_dict.set_item("masks", transformed_masks)?;
            }
        }

        Ok(result_dict.into())
    }

    /// Apply this sampled program to an image (numpy array or torch.Tensor)
    ///
    /// # Arguments
    /// - `array`: numpy array or torch.Tensor of shape (H, W, C), (C, H, W) or (H, W) with dtype uint8
    /// - `inplace`: bool, default False. If True, modifies array in-place without copying.
    ///
    /// # Returns
    /// A transformed numpy array or torch.Tensor matching the input type and layout
    #[pyo3(signature = (array, inplace=None, optimize=None))]
    pub(crate) fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        optimize: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        if crate::python::tensor::is_torch_tensor(array) {
            return crate::python::tensor::handle_torch_tensor(array, inplace, py, |np_arr, inp, p| {
                self.apply(np_arr, inp, optimize, p)
            });
        }

        use numpy::{PyArray2, PyArray3};

        // CHW numpy guard: a 3D array whose first dim is a channel count and
        // whose last dim is not is almost certainly channels-first. The HWC
        // path would silently treat it as (H=channels, W, C=width) and corrupt
        // the data, so fail loudly instead.
        if let Ok(arr3) = array.downcast::<PyArray3<u8>>() {
            let sh = arr3.shape();
            let (h, w, c) = (sh[0], sh[1], sh[2]);
            if (h == 1 || h == 3 || h == 4) && !(c == 1 || c == 3 || c == 4) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "detected channels-first numpy array with shape ({}, {}, {}); \
                     sinter's numpy path expects (H, W, C). Pass a torch.Tensor \
                     for CHW, or transpose with np.transpose(img, (1, 2, 0))",
                    h, w, c
                )));
            }
        }

        // Convert SampledImageProgram to Plan
        let plan = self.inner.to_plan();

        // Optimize the plan (or keep unoptimized if optimize=False)
        let exec_plan = if optimize.unwrap_or(true) {
            Optimizer::new().optimize(plan)
        } else {
            plan.to_unoptimized_exec_plan()
        };

        let is_inplace = inplace.unwrap_or(false);
        let is_c_contiguous = array
            .getattr("flags")
            .and_then(|f| f.getattr("c_contiguous"))
            .and_then(|c| c.extract::<bool>())
            .unwrap_or(true);

        if is_inplace && !is_c_contiguous {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cannot mutate non-contiguous array with inplace=True; use inplace=False to allow defensive copy",
            ));
        }

        if is_inplace && exec_plan.mutates_input() {
            let is_writeable = array
                .getattr("flags")
                .and_then(|f| f.getattr("writeable"))
                .and_then(|w| w.extract::<bool>())
                .unwrap_or(true);
            if !is_writeable {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot mutate read-only array with inplace=True; use inplace=False to create a defensive copy",
                ));
            }
        }

        let needs_copy = (!is_inplace && exec_plan.mutates_input()) || !is_c_contiguous;
        let working_array = if needs_copy {
            array.call_method0("copy")?
        } else {
            array
        };

        if let Ok(array3) = working_array.downcast::<PyArray3<u8>>() {
            let shape = array3.shape();
            let (height, width, channels) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            validate_crop_bounds(&exec_plan, width, height)?;
            let slice = unsafe { array3.as_slice_mut()? };
            let mut fusable_img = FusableImage::new(slice, width, height, channels);

            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_img));

            match result {
                Some(new_barrier) => {
                    let arr = crate::python::types::barrier_image_to_numpy_owned(py, new_barrier)?;
                    Ok(arr)
                }
                None => Ok(array3.as_ref()),
            }
        } else if let Ok(array2) = working_array.downcast::<PyArray2<u8>>() {
            let shape = array2.shape();
            let (height, width, channels) = (shape[0] as usize, shape[1] as usize, 1);
            validate_crop_bounds(&exec_plan, width, height)?;
            let slice = unsafe { array2.as_slice_mut()? };
            let mut fusable_img = FusableImage::new(slice, width, height, channels);

            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_img));

            match result {
                Some(new_barrier) => {
                    let (new_h, new_w, new_c) = (new_barrier.height, new_barrier.width, new_barrier.channels);
                    if let Some(f32_data) = new_barrier.f32_data {
                        let array_1d = numpy::PyArray1::from_vec(py, f32_data);
                        if new_c == 1 {
                            let array_2d = array_1d.reshape([new_h, new_w]).map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                    "Failed to reshape array: {}",
                                    e
                                ))
                            })?;
                            Ok(array_2d.as_ref())
                        } else {
                            let array_3d = array_1d.reshape([new_h, new_w, new_c]).map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                    "Failed to reshape array: {}",
                                    e
                                ))
                            })?;
                            Ok(array_3d.as_ref())
                        }
                    } else {
                        let array_1d = numpy::PyArray1::from_vec(py, new_barrier.data);
                        if new_c == 1 {
                            let array_2d = array_1d.reshape([new_h, new_w]).map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                    "Failed to reshape array: {}",
                                    e
                                ))
                            })?;
                            Ok(array_2d.as_ref())
                        } else {
                            let array_3d = array_1d.reshape([new_h, new_w, new_c]).map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                    "Failed to reshape array: {}",
                                    e
                                ))
                            })?;
                            Ok(array_3d.as_ref())
                        }
                    }
                }
                None => Ok(array2.as_ref()),
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a 2D or 3D uint8 numpy array or torch.Tensor for 'image'",
            ))
        }
    }

    /// Apply this sampled program to a batch of images in parallel across CPU cores
    #[pyo3(signature = (images, inplace=None, num_threads=None))]
    pub fn apply_batch<'py>(
        &self,
        images: &'py PyAny,
        inplace: Option<bool>,
        num_threads: Option<usize>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let prog = self.inner.clone();
        crate::python::batch::parallel_apply_batch(py, images, inplace, num_threads, move |_| {
            prog.clone()
        })
    }

    /// Apply the program to bounding boxes (supports N, 4+ column arrays)
    ///
    /// Preserves extra payload columns (such as class labels or confidence scores)
    /// attached after coordinates.
    #[pyo3(signature = (bboxes, image_size, format="xywh", format_out=None))]
    pub(crate) fn apply_to_bboxes<'py>(
        &self,
        bboxes: &numpy::PyArray2<f32>,
        image_size: (u32, u32),
        format: &str,
        format_out: Option<&str>,
        py: Python<'py>,
    ) -> PyResult<&'py numpy::PyArray2<f32>> {
        use crate::labels::BBoxFormat;

        // Parse format string
        let bbox_format = parse_bbox_format(format)?;
        let bbox_format_out = match format_out {
            Some(f) => parse_bbox_format(f)?,
            None => bbox_format,
        };

        let shape = bboxes.shape();
        let num_boxes = shape[0] as usize;
        let cols = shape[1] as usize;

        if cols < 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "bboxes must have at least 4 columns (x, y, w, h), got shape {:?}",
                shape
            )));
        }

        let slice = unsafe { bboxes.as_slice()? };
        let mut with_extra = Vec::with_capacity(num_boxes);
        for i in 0..num_boxes {
            let row = &slice[i * cols..(i + 1) * cols];
            let box_coords = [row[0], row[1], row[2], row[3]];
            let internal_box = bbox_format.to_internal(box_coords, image_size.0, image_size.1);
            let extra = row[4..].to_vec();
            with_extra.push((internal_box, extra));
        }

        let (transformed, (final_w, final_h)) =
            self.inner.apply_to_bboxes_with_extra(with_extra, image_size);

        let mut output_flat = Vec::with_capacity(transformed.len() * cols);
        for (internal_box, extra) in transformed {
            let out_box = bbox_format_out.from_internal(internal_box, final_w, final_h);
            output_flat.extend_from_slice(&out_box);
            output_flat.extend_from_slice(&extra);
        }

        let out_count = if cols > 0 {
            output_flat.len() / cols
        } else {
            0
        };
        let array1d = numpy::PyArray1::from_vec(py, output_flat);
        let result = array1d.reshape((out_count, cols)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to reshape bboxes: {}",
                e
            ))
        })?;

        Ok(result)
    }

    /// Apply the program to keypoints (zero-copy with format support)
    ///
    /// Supports N, core+ column arrays (extra payload columns such as class
    /// ids or scores pass through unchanged, matching apply_to_bboxes).
    #[pyo3(signature = (keypoints, image_size, format="xy", format_out=None))]
    pub(crate) fn apply_to_keypoints<'py>(
        &self,
        keypoints: &numpy::PyArray2<f32>,
        image_size: (u32, u32),
        format: &str,
        format_out: Option<&str>,
        py: Python<'py>,
    ) -> PyResult<&'py numpy::PyArray2<f32>> {
        use crate::labels::KeypointFormat;

        let kpt_format = parse_keypoint_format(format)?;
        let kpt_format_out = match format_out {
            Some(f) => parse_keypoint_format(f)?,
            None => kpt_format,
        };

        let shape = keypoints.shape();
        let num = shape[0] as usize;
        let cols = shape[1] as usize;
        let core = kpt_format.len();

        if cols < core {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "keypoints must have at least {} columns for format '{}', got shape {:?}",
                core, format, shape
            )));
        }

        let slice = unsafe { keypoints.as_slice()? };
        let mut with_extra = Vec::with_capacity(num);
        for i in 0..num {
            let row = &slice[i * cols..(i + 1) * cols];
            let internal = kpt_format.to_internal(&row[..core], image_size.0, image_size.1);
            let extra = row[core..].to_vec();
            with_extra.push((internal, extra));
        }

        let xy_only: Vec<(f32, f32)> = with_extra.iter().map(|&((x, y, _), _)| (x, y)).collect();
        let visibilities: Vec<u8> = with_extra.iter().map(|&((_, _, v), _)| v).collect();
        let (transformed_xy, (final_w, final_h)) =
            self.inner.apply_to_keypoints(xy_only, image_size);

        let out_cols = kpt_format_out.len() + cols.saturating_sub(core);
        let mut output_flat = Vec::with_capacity(transformed_xy.len() * out_cols);
        for (i, ((x, y), v)) in transformed_xy.into_iter().zip(visibilities.into_iter()).enumerate() {
            let mut row = kpt_format_out.from_internal(x, y, v, final_w, final_h);
            row.extend_from_slice(&with_extra[i].1);
            output_flat.extend_from_slice(&row);
        }

        let out_count = if out_cols > 0 { output_flat.len() / out_cols } else { 0 };
        let array1d = numpy::PyArray1::from_vec(py, output_flat);
        let result = array1d.reshape((out_count, out_cols)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to reshape keypoints: {}",
                e
            ))
        })?;

        Ok(result)
    }

    /// Apply the program to classification labels
    fn apply_to_labels<'py>(
        &self,
        labels: &numpy::PyArray1<i32>,
        image_size: (u32, u32),
        py: Python<'py>,
    ) -> PyResult<&'py numpy::PyArray1<i32>> {
        let slice = unsafe { labels.as_slice()? };
        let labels_vec = slice.to_vec();
        let transformed = self.inner.apply_to_labels(labels_vec, image_size);
        let result = numpy::PyArray1::from_vec(py, transformed);
        Ok(result)
    }

    /// Apply the program to segmentation masks
    ///
    /// Applies ONLY geometric transformations using Nearest-Neighbor interpolation,
    /// ensuring segmentation integer class labels are never altered by photometric ops.
    #[pyo3(signature = (mask, image_size, inplace=None))]
    pub(crate) fn apply_to_masks<'py>(
        &self,
        mask: &'py PyAny,
        image_size: (u32, u32),
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        if crate::python::tensor::is_torch_tensor(mask) {
            return crate::python::tensor::handle_torch_tensor(mask, inplace, py, |np_arr, inp, p| {
                self.apply_to_masks(np_arr, image_size, inp, p)
            });
        }

        let _ = image_size;
        use numpy::{PyArray2, PyArray3};

        // Convert only geometric ops to Plan (nearest-neighbor interpolation for masks)
        let geom_prog = self.inner.geometric_program();
        let plan = geom_prog.to_plan();
        let exec_plan = Optimizer::new().optimize(plan);

        let is_inplace = inplace.unwrap_or(false);
        let is_c_contiguous = mask
            .getattr("flags")
            .and_then(|f| f.getattr("c_contiguous"))
            .and_then(|c| c.extract::<bool>())
            .unwrap_or(true);

        if is_inplace && !is_c_contiguous {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cannot mutate non-contiguous mask with inplace=True; use inplace=False to allow defensive copy",
            ));
        }

        if is_inplace && exec_plan.mutates_input() {
            let is_writeable = mask
                .getattr("flags")
                .and_then(|f| f.getattr("writeable"))
                .and_then(|w| w.extract::<bool>())
                .unwrap_or(true);
            if !is_writeable {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot mutate read-only mask with inplace=True; use inplace=False to create a defensive copy",
                ));
            }
        }

        let needs_copy = (!is_inplace && exec_plan.mutates_input()) || !is_c_contiguous;
        let working_mask = if needs_copy {
            mask.call_method0("copy")?
        } else {
            mask
        };

        if let Ok(array2) = working_mask.downcast::<PyArray2<u8>>() {
            let shape = array2.shape();
            let (height, width) = (shape[0] as usize, shape[1] as usize);
            let slice = unsafe { array2.as_slice_mut()? };
            let mut fusable_mask = FusableImage::new(slice, width, height, 1);

            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_mask));

            match result {
                Some(new_barrier) => {
                    let flat = &new_barrier.data;
                    let (new_h, new_w, new_c) =
                        (new_barrier.height, new_barrier.width, new_barrier.channels);

                    let stride = new_w * new_c;
                    let channel_0: Vec<u8> = (0..new_h)
                        .flat_map(|y| {
                            let row_start = y * stride;
                            &flat[row_start..row_start + new_w]
                        })
                        .copied()
                        .collect();

                    let array1d = numpy::PyArray1::from_vec(py, channel_0);
                    let result = array1d.reshape((new_h, new_w)).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Failed to reshape mask: {}",
                            e
                        ))
                    })?;
                    Ok(result.as_ref())
                }
                None => Ok(array2),
            }
        } else if let Ok(array3) = working_mask.downcast::<PyArray3<u8>>() {
            let shape = array3.shape();
            let (height, width, channels) =
                (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            let slice = unsafe { array3.as_slice_mut()? };
            let mut fusable_mask = FusableImage::new(slice, width, height, channels);

            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_mask));

            match result {
                Some(new_barrier) => {
                    crate::python::types::barrier_image_to_numpy_owned(py, new_barrier)
                        .map(|arr| arr)
                }
                None => Ok(array3),
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Mask must be a 2D or 3D numpy array with dtype uint8",
            ))
        }
    }
}

/// Parse bbox format string
fn parse_bbox_format(s: &str) -> PyResult<BBoxFormat> {
    let clean = s.trim().to_lowercase().replace('-', "_");
    match clean.as_str() {
        "xyxy" | "pascal_voc" | "pascalvoc" => Ok(BBoxFormat::Xyxy),
        "xywh" | "coco" => Ok(BBoxFormat::Xywh),
        "cxcywh" | "center_xywh" => Ok(BBoxFormat::Cxcywh),
        "rel_xyxy" | "albumentations" => Ok(BBoxFormat::RelXyxy),
        "rel_xywh" => Ok(BBoxFormat::RelXywh),
        "rel_cxcywh" | "yolo" => Ok(BBoxFormat::RelCxcywh),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown bbox format: '{}'. Supported formats: 'xyxy' ('pascal_voc'), 'xywh' ('coco'), 'cxcywh', 'rel_xyxy' ('albumentations'), 'rel_xywh', 'rel_cxcywh' ('yolo')",
            s
        ))),
    }
}

/// Parse keypoint format string
fn parse_keypoint_format(s: &str) -> PyResult<KeypointFormat> {
    let clean = s.trim().to_lowercase().replace('-', "_");
    match clean.as_str() {
        "xy" => Ok(KeypointFormat::Xy),
        "xyv" | "xy_visibility" => Ok(KeypointFormat::Xyv),
        "rel_xy" => Ok(KeypointFormat::RelXy),
        "rel_xyv" => Ok(KeypointFormat::RelXyv),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown keypoint format: '{}'. Supported formats: 'xy', 'xyv', 'rel_xy', 'rel_xyv'",
            s
        ))),
    }
}

/// Extract 2D coordinate array from numpy ndarray, torch.Tensor, or Python list/sequence
fn extract_2d_coords<'py>(
    obj: &'py PyAny,
    min_cols: usize,
    py: Python<'py>,
) -> PyResult<(&'py numpy::PyArray2<f32>, bool, bool)> {
    if let Ok(arr) = obj.downcast::<numpy::PyArray2<f32>>() {
        return Ok((arr, false, false));
    }
    if let Ok(arr) = obj.downcast::<numpy::PyArray2<f64>>() {
        let np = py.import("numpy")?;
        let f32_arr = np.call_method1("asarray", (arr, "float32"))?.downcast::<numpy::PyArray2<f32>>()?;
        return Ok((f32_arr, false, false));
    }
    if let Ok(arr) = obj.downcast::<numpy::PyArray2<i64>>() {
        let np = py.import("numpy")?;
        let f32_arr = np.call_method1("asarray", (arr, "float32"))?.downcast::<numpy::PyArray2<f32>>()?;
        return Ok((f32_arr, false, false));
    }
    if let Ok(arr) = obj.downcast::<numpy::PyArray2<i32>>() {
        let np = py.import("numpy")?;
        let f32_arr = np.call_method1("asarray", (arr, "float32"))?.downcast::<numpy::PyArray2<f32>>()?;
        return Ok((f32_arr, false, false));
    }
    if crate::python::tensor::is_torch_tensor(obj) {
        let np_obj = obj.call_method0("numpy")?;
        let np = py.import("numpy")?;
        let f32_arr = np.call_method1("asarray", (np_obj, "float32"))?.downcast::<numpy::PyArray2<f32>>()?;
        return Ok((f32_arr, true, false));
    }
    if let Ok(seq) = obj.downcast::<pyo3::types::PySequence>() {
        if seq.len()? == 0 {
            let empty = numpy::PyArray2::<f32>::zeros(py, [0, min_cols], false);
            return Ok((empty, false, true));
        }
        let np = py.import("numpy")?;
        let arr_obj = np.call_method1("asarray", (obj, "float32"))?;
        if let Ok(arr2) = arr_obj.downcast::<numpy::PyArray2<f32>>() {
            return Ok((arr2, false, true));
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected 2D numpy array, torch.Tensor, or sequence of coordinates, got {}",
        obj.get_type().name()?
    )))
}
