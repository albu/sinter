// PySampledImageProgram wrapper

use crate::core::FusableImage;
use crate::exec_ir::Optimizer;
use crate::sampled_ir::{Plan, SampledImageProgram, IR_VERSION};
use crate::labels::{BBoxFormat, KeypointArray, KeypointFormat};

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, PyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use std::path::PathBuf;

/// _SampledImageProgram - deterministic, serializable transform program (internal)
///
/// This is the output of the sampling phase and input to the optimizer.
/// All parameters are fixed, all randomness resolved.
///
/// # Internal API
///
/// This is returned by `Compose.sample_with_seed()` and should not
/// be constructed directly by users.
#[cfg(feature = "python")]
#[pyclass(name = "_SampledImageProgram")]
pub struct PySampledImageProgram {
    pub inner: SampledImageProgram,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySampledImageProgram {
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
    fn summary(&self) -> String {
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
    fn explain(&self) -> String {
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

    fn __repr__(&self) -> String {
        format!(
            "_SampledImageProgram(version={}, ops={})",
            self.inner.version,
            self.inner.len()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.len()
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
    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    pub(crate) fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&PyArray2<f32>>,
        keypoints: Option<&PyArray2<f32>>,
        masks: Option<&PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let transformed_image_array = self.apply(image, inplace, py)?;

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

        if let Some(bbox_array) = bboxes {
            let image_size = get_image_size()?;
            let transformed_bboxes =
                self.apply_to_bboxes(bbox_array, image_size, bbox_format, None, py)?;
            result_dict.set_item("bboxes", transformed_bboxes)?;
        }

        if let Some(kpt_array) = keypoints {
            let image_size = get_image_size()?;
            let transformed_keypoints =
                self.apply_to_keypoints(kpt_array, image_size, keypoint_format, None, py)?;
            result_dict.set_item("keypoints", transformed_keypoints)?;
        }

        if let Some(mask_array) = masks {
            let image_size = get_image_size()?;
            let transformed_masks = self.apply_to_masks(mask_array, image_size, inplace, py)?;
            result_dict.set_item("masks", transformed_masks)?;
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
    #[pyo3(signature = (array, inplace=None))]
    pub(crate) fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        if crate::python::tensor::is_torch_tensor(array) {
            return crate::python::tensor::handle_torch_tensor(array, inplace, py, |np_arr, inp, p| {
                self.apply(np_arr, inp, p)
            });
        }

        use numpy::{PyArray2, PyArray3};

        // Convert SampledImageProgram to Plan
        let plan = self.inner.to_plan();

        // Optimize the plan
        let exec_plan = Optimizer::new().optimize(plan);

        let is_inplace = inplace.unwrap_or(false);
        let needs_copy = !is_inplace && exec_plan.mutates_input();
        let working_array = if needs_copy {
            array.call_method0("copy")?
        } else {
            array
        };

        if let Ok(array3) = working_array.downcast::<PyArray3<u8>>() {
            let shape = array3.shape();
            let (height, width, channels) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            let slice = unsafe { array3.as_slice_mut()? };
            let mut fusable_img = FusableImage::new(slice, width, height, channels);

            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_img));

            match result {
                Some(new_barrier) => {
                    let arr = crate::python::types::barrier_image_to_numpy_owned(py, new_barrier)?;
                    Ok(arr.as_ref())
                }
                None => Ok(array3.as_ref()),
            }
        } else if let Ok(array2) = working_array.downcast::<PyArray2<u8>>() {
            let shape = array2.shape();
            let (height, width, channels) = (shape[0] as usize, shape[1] as usize, 1);
            let slice = unsafe { array2.as_slice_mut()? };
            let mut fusable_img = FusableImage::new(slice, width, height, channels);

            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_img));

            match result {
                Some(new_barrier) => {
                    let (new_h, new_w, new_c) = (new_barrier.height, new_barrier.width, new_barrier.channels);
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

        let slice = unsafe { keypoints.as_slice()? };
        let kpt_array = KeypointArray::from_slice(slice, kpt_format, image_size.0, image_size.1)
            .with_output_format(kpt_format_out);

        let internal_kpts = kpt_array.to_vec_internal();
        let visibilities: Vec<u8> = internal_kpts.iter().map(|&(_, _, v)| v).collect();
        let xy_only: Vec<(f32, f32)> = internal_kpts.iter().map(|&(x, y, _)| (x, y)).collect();
        let (transformed_xy, (final_w, final_h)) =
            self.inner.apply_to_keypoints(xy_only, image_size);

        let transformed: Vec<(f32, f32, u8)> = transformed_xy
            .into_iter()
            .zip(visibilities.into_iter())
            .map(|((x, y), v)| (x, y, v))
            .collect();

        let owned = crate::labels::KeypointArrayOwned::from_internal(transformed, final_w, final_h)
            .with_output_format(kpt_format_out);
        let output = owned.to_output();
        let out_stride = kpt_format_out.len();

        let flat: Vec<f32> = output.iter().flatten().copied().collect();
        let array1d = numpy::PyArray1::from_vec(py, flat);
        let result = array1d.reshape((output.len(), out_stride)).map_err(|e| {
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
        let needs_copy = !is_inplace && exec_plan.mutates_input();
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
                        .map(|arr| arr.as_ref())
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
    match s {
        "xyxy" => Ok(BBoxFormat::Xyxy),
        "xywh" => Ok(BBoxFormat::Xywh),
        "cxcywh" => Ok(BBoxFormat::Cxcywh),
        "rel_xyxy" => Ok(BBoxFormat::RelXyxy),
        "rel_xywh" => Ok(BBoxFormat::RelXywh),
        "rel_cxcywh" => Ok(BBoxFormat::RelCxcywh),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown bbox format: {}. Expected: xyxy, xywh, cxcywh, rel_xyxy, rel_xywh, rel_cxcywh",
            s
        ))),
    }
}

/// Parse keypoint format string
fn parse_keypoint_format(s: &str) -> PyResult<KeypointFormat> {
    match s {
        "xy" => Ok(KeypointFormat::Xy),
        "xyv" => Ok(KeypointFormat::Xyv),
        "rel_xy" => Ok(KeypointFormat::RelXy),
        "rel_xyv" => Ok(KeypointFormat::RelXyv),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown keypoint format: {}. Expected: xy, xyv, rel_xy, rel_xyv",
            s
        ))),
    }
}
