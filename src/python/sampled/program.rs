// PySampledImageProgram wrapper

use crate::core::FusableImage;
use crate::exec_ir::Optimizer;
use crate::sampled_ir::{Plan, SampledImageProgram, IR_VERSION};
use crate::labels::{BBoxFormat, KeypointFormat};

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

    /// Apply this sampled program to an image
    ///
    /// # Arguments
    /// - `array`: numpy array of shape (H, W, C) with dtype uint8
    ///
    /// # Returns
    /// A numpy array with the transformed image
    ///
    /// # Example
    /// ```python
    /// sampled = plan.sample_with_seed(42)
    /// result = sampled.apply(image_array)
    /// ```
    /// Apply this sampled program to an image
    ///
    /// # Arguments
    /// - `array`: numpy array of shape (H, W, C) with dtype uint8
    ///
    /// # Returns
    /// A numpy array with the transformed image
    ///
    /// # Example
    /// ```python
    /// sampled = plan.sample_with_seed(42)
    /// result = sampled.apply(image_array)
    /// ```
    #[pyo3(name = "apply")]
    pub(crate) fn apply<'py>(
        &self,
        array: &'py PyAny,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        use numpy::{PyArray2, PyArray3};

        // Convert SampledImageProgram to Plan
        let plan = self.inner.to_plan();

        // Optimize the plan
        let exec_plan = Optimizer::new().optimize(plan);

        if let Ok(array3) = array.downcast::<PyArray3<u8>>() {
            // Get shape information
            let shape = array3.shape();
            let (height, width, channels) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);

            // Get mutable slice from numpy array
            let slice = unsafe { array3.as_slice_mut()? };

            // Create a FusableImage that borrows the numpy data
            let mut fusable_img = FusableImage::new(slice, width, height, channels);

            // Release GIL during Rust execution - allows other Python threads to run!
            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_img));

            match result {
                Some(new_barrier) => {
                    let arr = crate::python::types::barrier_image_to_numpy_owned(py, new_barrier)?;
                    Ok(arr.as_ref())
                }
                None => Ok(array3.as_ref()),
            }
        } else if let Ok(array2) = array.downcast::<PyArray2<u8>>() {
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
                "Expected a 2D or 3D uint8 numpy array for 'image'",
            ))
        }
    }

    /// Apply the program to bounding boxes (zero-copy with format support)
    ///
    /// # Arguments
    /// - `bboxes`: 2D numpy array of shape (N, 4) with dtype float32
    /// - `image_size`: Tuple of (width, height) integers
    /// - `format`: Optional format string ("xyxy", "xywh", "cxcywh",
    ///   "rel_xyxy", "rel_xywh", "rel_cxcywh"). Default: "xywh"
    /// - `format_out`: Optional output format. Default: same as input
    ///
    /// # Returns
    /// 2D numpy array of transformed bounding boxes in the specified output format
    ///
    /// # Example
    /// ```python
    /// sampled = compose.sample_with_seed(42)
    ///
    /// # Input in xywh format
    /// bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
    /// result = sampled.apply_to_bboxes(bboxes, (100, 100))
    ///
    /// # Input in xyxy format
    /// bboxes = np.array([[10, 20, 40, 60]], dtype=np.float32)
    /// result = sampled.apply_to_bboxes(bboxes, (100, 100), format="xyxy")
    ///
    /// # Convert to normalized output
    /// result = sampled.apply_to_bboxes(bboxes, (100, 100), format_out="rel_xywh")
    /// ```
    #[pyo3(signature = (bboxes, image_size, format="xywh", format_out=None))]
    pub(crate) fn apply_to_bboxes<'py>(
        &self,
        bboxes: &numpy::PyArray2<f32>,
        image_size: (u32, u32),
        format: &str,
        format_out: Option<&str>,
        py: Python<'py>,
    ) -> PyResult<&'py numpy::PyArray2<f32>> {
        use crate::labels::{BBoxArray, BBoxFormat};

        // Parse format string
        let bbox_format = parse_bbox_format(format)?;
        let bbox_format_out = match format_out {
            Some(f) => parse_bbox_format(f)?,
            None => bbox_format,
        };

        // Get zero-copy view of the numpy array
        let slice = unsafe { bboxes.as_slice()? };

        // Wrap in zero-copy BBoxArray
        let bbox_array = BBoxArray::from_slice(slice, bbox_format, image_size.0, image_size.1)
            .with_output_format(bbox_format_out);

        // Apply transform and get owned result
        let internal_bboxes = bbox_array.to_vec_internal();
        let (transformed, (final_w, final_h)) =
            self.inner.apply_to_bboxes(internal_bboxes, image_size);

        // Convert to owned format and create output array
        let owned = crate::labels::BBoxArrayOwned::from_internal(transformed, final_w, final_h)
            .with_output_format(bbox_format_out);
        let output = owned.to_output();

        // Flatten to 2D array using PyArray1 then reshape
        let flat: Vec<f32> = output.iter().flatten().copied().collect();
        let array1d = numpy::PyArray1::from_vec(py, flat);
        let result = array1d
            .reshape((output.len(), bbox_format_out.len()))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to reshape bboxes: {}",
                    e
                ))
            })?;

        Ok(result)
    }

    /// Apply the program to keypoints (zero-copy with format support)
    ///
    /// # Arguments
    /// - `keypoints`: 2D numpy array of shape (N, 2) or (N, 3) with dtype float32
    /// - `image_size`: Tuple of (width, height) integers
    /// - `format`: Optional format string ("xy", "xyv", "rel_xy", "rel_xyv"). Default: "xy"
    /// - `format_out`: Optional output format. Default: same as input
    ///
    /// # Returns
    /// 2D numpy array of transformed keypoints in the specified output format
    ///
    /// # Visibility Values (for xyv formats)
    /// - 0: Not visible (outside image or occluded)
    /// - 1: Occluded
    /// - 2: Visible
    ///
    /// # Example
    /// ```python
    /// sampled = compose.sample_with_seed(42)
    ///
    /// # Input in xy format
    /// kpts = np.array([[10, 20], [30, 40]], dtype=np.float32)
    /// result = sampled.apply_to_keypoints(kpts, (100, 100))
    ///
    /// # Input with visibility
    /// kpts = np.array([[10, 20, 2], [30, 40, 1]], dtype=np.float32)
    /// result = sampled.apply_to_keypoints(kpts, (100, 100), format="xyv")
    ///
    /// # Convert to normalized output
    /// result = sampled.apply_to_keypoints(kpts, (100, 100), format_out="rel_xy")
    /// ```
    #[pyo3(signature = (keypoints, image_size, format="xy", format_out=None))]
    pub(crate) fn apply_to_keypoints<'py>(
        &self,
        keypoints: &numpy::PyArray2<f32>,
        image_size: (u32, u32),
        format: &str,
        format_out: Option<&str>,
        py: Python<'py>,
    ) -> PyResult<&'py numpy::PyArray2<f32>> {
        use crate::labels::{KeypointArray, KeypointFormat};

        // Parse format string
        let kpt_format = parse_keypoint_format(format)?;
        let kpt_format_out = match format_out {
            Some(f) => parse_keypoint_format(f)?,
            None => kpt_format,
        };

        // Get zero-copy view of the numpy array
        let slice = unsafe { keypoints.as_slice()? };

        // Wrap in zero-copy KeypointArray
        let kpt_array = KeypointArray::from_slice(slice, kpt_format, image_size.0, image_size.1)
            .with_output_format(kpt_format_out);

        // Apply transform and get owned result
        // Note: We need to preserve visibility values separately since
        // SampledImageProgram::apply_to_keypoints only accepts (x, y) tuples
        let internal_kpts = kpt_array.to_vec_internal();
        let visibilities: Vec<u8> = internal_kpts.iter().map(|&(_, _, v)| v).collect();
        let xy_only: Vec<(f32, f32)> = internal_kpts.iter().map(|&(x, y, _)| (x, y)).collect();
        let (transformed_xy, (final_w, final_h)) =
            self.inner.apply_to_keypoints(xy_only, image_size);

        // Reattach visibility values
        let transformed: Vec<(f32, f32, u8)> = transformed_xy
            .into_iter()
            .zip(visibilities.into_iter())
            .map(|((x, y), v)| (x, y, v))
            .collect();

        // Convert to owned format and create output array
        let owned = crate::labels::KeypointArrayOwned::from_internal(transformed, final_w, final_h)
            .with_output_format(kpt_format_out);
        let output = owned.to_output();

        // Find the output stride (may vary if format differs)
        let out_stride = kpt_format_out.len();

        // Flatten to 2D array using PyArray1 then reshape
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
    ///
    /// # Arguments
    /// - `labels`: 1D numpy array of class labels (int32)
    /// - `image_size`: Tuple of (width, height) integers
    ///
    /// # Returns
    /// 1D numpy array of class labels (unchanged for most transforms)
    ///
    /// # Note
    /// Most geometric transforms do not affect classification labels.
    /// This method returns the labels unchanged. Future transforms like
    /// MixUp or CutMix may need special handling.
    ///
    /// # Example
    /// ```python
    /// sampled = compose.sample_with_seed(42)
    /// labels = np.array([0, 1, 2], dtype=np.int32)
    /// result = sampled.apply_to_labels(labels, (100, 100))
    /// ```
    fn apply_to_labels<'py>(
        &self,
        labels: &numpy::PyArray1<i32>,
        image_size: (u32, u32),
        py: Python<'py>,
    ) -> PyResult<&'py numpy::PyArray1<i32>> {
        // Get zero-copy view of the numpy array
        let slice = unsafe { labels.as_slice()? };
        let labels_vec = slice.to_vec();

        // Apply transform (pass-through for most transforms)
        let transformed = self.inner.apply_to_labels(labels_vec, image_size);

        // Create output array
        let result = numpy::PyArray1::from_vec(py, transformed);
        Ok(result)
    }

    /// Apply the program to segmentation masks
    ///
    /// # Arguments
    /// - `mask`: 2D numpy array (H, W) or 3D array (H, W, 1) with dtype uint8 or uint32
    /// - `image_size`: Tuple of (width, height) integers - should match mask dimensions
    ///
    /// # Returns
    /// Transformed mask as a 2D or 3D numpy array (same shape as input)
    ///
    /// # Note
    /// Masks are transformed using the same geometric operations as images.
    /// For resize/rotate operations, this may use interpolation which could
    /// introduce artifacts. For strict label preservation, consider using
    /// nearest-neighbor interpolation (future enhancement).
    ///
    /// # Example
    /// ```python
    /// sampled = compose.sample_with_seed(42)
    /// mask = np.zeros((100, 100), dtype=np.uint8)
    /// mask[20:60, 30:70] = 1  # Object mask
    /// result = sampled.apply_to_masks(mask, (100, 100))
    /// ```
    pub(crate) fn apply_to_masks<'py>(
        &self,
        mask: &'py PyAny,
        image_size: (u32, u32),
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        use numpy::{PyArray2, PyArray3};

        // Convert SampledImageProgram to Plan
        let plan = self.inner.to_plan();

        // Optimize the plan
        let exec_plan = Optimizer::new().optimize(plan);

        // For 2D masks (H, W)
        if let Ok(array2) = mask.downcast::<PyArray2<u8>>() {
            let shape = array2.shape();
            let (height, width) = (shape[0] as usize, shape[1] as usize);

            // Get mutable slice from numpy array
            let slice = unsafe { array2.as_slice_mut()? };

            // Create a FusableImage (single channel)
            let mut fusable_mask = FusableImage::new(slice, width, height, 1);

            // Release GIL during Rust execution
            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_mask));

            match result {
                Some(new_barrier) => {
                    // Transform allocated a new buffer
                    // For now, return the first channel as 2D array
                    let flat = &new_barrier.data;
                    let (new_h, new_w, new_c) =
                        (new_barrier.height, new_barrier.width, new_barrier.channels);

                    // Return as 2D array (take first channel if multi-channel)
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
                None => {
                    // In-place transform - return original array (same as apply method does)
                    // SAFETY: array2 has lifetime 'py from the input parameter, matching the GIL lifetime
                    Ok(array2)
                }
            }
        } else if let Ok(array3) = mask.downcast::<PyArray3<u8>>() {
            // For 3D masks (H, W, C)
            let shape = array3.shape();
            let (height, width, channels) =
                (shape[0] as usize, shape[1] as usize, shape[2] as usize);

            // Get mutable slice from numpy array
            let slice = unsafe { array3.as_slice_mut()? };

            // Create a FusableImage
            let mut fusable_mask = FusableImage::new(slice, width, height, channels);

            // Release GIL during Rust execution
            let result = py.allow_threads(|| exec_plan.execute(&mut fusable_mask));

            match result {
                Some(new_barrier) => {
                    // Transform allocated a new buffer
                    crate::python::types::barrier_image_to_numpy_owned(py, new_barrier)
                        .map(|arr| arr.as_ref())
                }
                None => {
                    // In-place transform - return original array
                    // SAFETY: array3 has lifetime 'py from the input parameter, matching the GIL lifetime
                    Ok(array3)
                }
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
