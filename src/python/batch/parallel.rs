// Parallel multi-image batch execution using Rayon
//
// Drops the GIL and executes compiled fused plans in parallel across CPU cores.

use crate::core::FusableImage;
use crate::exec_ir::Optimizer;
use crate::python::tensor::is_torch_tensor;
use crate::sampled_ir::SampledImageProgram;
use numpy::{PyArray1, PyArray3, PyArray4};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySequence};
use rayon::prelude::*;

/// Apply a sequence of sampled programs in parallel to a batch of images
#[cfg(feature = "python")]
pub fn parallel_apply_batch<'py, F>(
    py: Python<'py>,
    images: &'py PyAny,
    inplace: Option<bool>,
    num_threads: Option<usize>,
    get_program: F,
) -> PyResult<&'py PyAny>
where
    F: Fn(usize) -> SampledImageProgram + Sync + Send,
{
    let is_inplace = inplace.unwrap_or(false);

    // Case 1: 4D PyTorch Tensor (N, C, H, W) or (N, H, W, C)
    if is_torch_tensor(images) {
        if let Ok(is_cuda) = images.getattr("is_cuda") {
            if is_cuda.extract::<bool>().unwrap_or(false) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Sinter operates on CPU memory; pass a CPU tensor (e.g., tensor.cpu())",
                ));
            }
        }

        let shape: Vec<usize> = images.getattr("shape")?.extract()?;
        if shape.len() == 4 {
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let _ = n;
            if (c == 1 || c == 3 || c == 4) && (h > 4 || w > 4) {
                // Layout is NCHW -> permute to NHWC
                let nhwc_tensor = images
                    .call_method1("permute", ((0, 2, 3, 1),))?
                    .call_method0("contiguous")?;
                let numpy_4d = nhwc_tensor.call_method0("numpy")?;
                let out_numpy = parallel_apply_batch(py, numpy_4d, Some(is_inplace), num_threads, get_program)?;
                let torch_mod = py.import("torch")?;
                let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
                let nchw_res = res_tensor
                    .call_method1("permute", ((0, 3, 1, 2),))?
                    .call_method0("contiguous")?;
                return Ok(nchw_res);
            } else {
                // Layout is NHWC
                let cont_tensor = if is_inplace {
                    images
                } else {
                    images.call_method0("contiguous")?
                };
                let numpy_4d = cont_tensor.call_method0("numpy")?;
                let out_numpy = parallel_apply_batch(py, numpy_4d, Some(is_inplace), num_threads, get_program)?;
                let torch_mod = py.import("torch")?;
                let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
                return Ok(res_tensor);
            }
        }
    }

    // Case 2: 4D numpy array (N, H, W, C)
    if let Ok(arr4) = images.downcast::<PyArray4<u8>>() {
        let shape = arr4.shape();
        let (batch_size, height, width, channels) = (
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
        );

        if batch_size == 0 {
            return Ok(arr4.as_ref());
        }

        let is_c_contiguous = images
            .getattr("flags")
            .and_then(|f| f.getattr("c_contiguous"))
            .and_then(|c| c.extract::<bool>())
            .unwrap_or(true);

        if is_inplace {
            if !is_c_contiguous {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot mutate non-contiguous batch with inplace=True; use inplace=False to allow defensive copy",
                ));
            }
            let is_writeable = images
                .getattr("flags")
                .and_then(|f| f.getattr("writeable"))
                .and_then(|w| w.extract::<bool>())
                .unwrap_or(true);
            if !is_writeable {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot mutate read-only batch with inplace=True; use inplace=False to create a defensive copy",
                ));
            }
        }

        let working_array = if is_inplace && is_c_contiguous {
            arr4
        } else {
            let copied = images.call_method0("copy")?;
            copied.downcast::<PyArray4<u8>>()?
        };

        let slice = unsafe { working_array.as_slice_mut()? };
        let img_slice_len = height * width * channels;

        let execute_fn = || {
            let chunks: Vec<&mut [u8]> = slice.chunks_exact_mut(img_slice_len).collect();
            let barriers: Vec<Option<crate::core::BarrierImage>> = chunks
                .into_par_iter()
                .enumerate()
                .map(|(idx, chunk)| {
                    let prog = get_program(idx);
                    let plan = prog.to_plan();
                    let exec_plan = Optimizer::new().optimize(plan);
                    let mut fusable = FusableImage::new(chunk, width, height, channels);
                    exec_plan.execute(&mut fusable)
                })
                .collect();
            barriers
        };

        let barriers = match num_threads {
            Some(n) if n > 0 => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                py.allow_threads(|| pool.install(execute_fn))
            }
            _ => py.allow_threads(execute_fn),
        };

        // If barriers were returned, check if all have the same shape
        let (first_h, first_w, first_c) = match &barriers[0] {
            Some(b) => (b.height, b.width, b.channels),
            None => (height, width, channels),
        };

        let same_shape = barriers.iter().all(|maybe_b| match maybe_b {
            Some(b) => b.height == first_h && b.width == first_w && b.channels == first_c,
            None => height == first_h && width == first_w && channels == first_c,
        });

        if same_shape {
            let item_len = first_h * first_w * first_c;
            let mut out_vec = Vec::with_capacity(batch_size * item_len);
            for (i, maybe_b) in barriers.into_iter().enumerate() {
                if let Some(b) = maybe_b {
                    out_vec.extend_from_slice(&b.data);
                } else {
                    let start = i * img_slice_len;
                    let end = start + img_slice_len;
                    out_vec.extend_from_slice(&slice[start..end]);
                }
            }
            let arr1 = PyArray1::from_vec(py, out_vec);
            let arr4_out = arr1.reshape([batch_size, first_h, first_w, first_c]).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to reshape 4D batch array: {}",
                    e
                ))
            })?;
            return Ok(arr4_out.as_ref());
        } else {
            // Ragged batch (different output dimensions) -> return list
            let result_list = PyList::empty(py);
            for (i, maybe_b) in barriers.into_iter().enumerate() {
                if let Some(b) = maybe_b {
                    let item_arr = crate::python::types::barrier_image_to_numpy_owned(py, b)?;
                    result_list.append(item_arr)?;
                } else {
                    let start = i * img_slice_len;
                    let end = start + img_slice_len;
                    let chunk_vec = slice[start..end].to_vec();
                    let arr1 = PyArray1::from_vec(py, chunk_vec);
                    let item_arr = arr1.reshape([height, width, channels]).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Failed to reshape array: {}",
                            e
                        ))
                    })?;
                    result_list.append(item_arr)?;
                }
            }
            return Ok(result_list.as_ref());
        }
    }

    // Case 3: Python list/sequence of images
    if let Ok(seq) = images.downcast::<PySequence>() {
        let len = seq.len()?;
        if len == 0 {
            return Ok(PyList::empty(py).as_ref());
        }

        let mut py_items = Vec::with_capacity(len);
        for i in 0..len {
            py_items.push(seq.get_item(i)?);
        }

        let mut results = Vec::with_capacity(len);
        for (idx, item) in py_items.iter().enumerate() {
            let prog = get_program(idx);
            let py_prog = crate::python::sampled::PySampledImageProgram { inner: prog };
            let transformed = py_prog.apply(item, Some(is_inplace), None, py)?;
            results.push(transformed);
        }

        let result_list = PyList::new(py, results);
        return Ok(result_list.as_ref());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected 4D numpy array, 4D torch.Tensor, or sequence of images for apply_batch",
    ))
}

/// Apply a single sampled program consistently across all frames of a video clip [T, H, W, C] or [T, C, H, W]
#[cfg(feature = "python")]
pub fn parallel_apply_video_clip<'py>(
    py: Python<'py>,
    video: &'py PyAny,
    sampled_prog: &SampledImageProgram,
    inplace: Option<bool>,
    num_threads: Option<usize>,
) -> PyResult<&'py PyAny> {
    let is_inplace = inplace.unwrap_or(false);

    // Case 1: 4D PyTorch Tensor (T, C, H, W) or (T, H, W, C)
    if is_torch_tensor(video) {
        if let Ok(is_cuda) = video.getattr("is_cuda") {
            if is_cuda.extract::<bool>().unwrap_or(false) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Sinter operates on CPU memory; pass a CPU tensor (e.g., tensor.cpu())",
                ));
            }
        }

        let shape: Vec<usize> = video.getattr("shape")?.extract()?;
        if shape.len() == 4 {
            let (_t, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            if (c == 1 || c == 3 || c == 4) && (h > 4 || w > 4) {
                // Layout is TCHW -> permute to THWC
                let thwc_tensor = video
                    .call_method1("permute", ((0, 2, 3, 1),))?
                    .call_method0("contiguous")?;
                let numpy_4d = thwc_tensor.call_method0("numpy")?;
                let out_numpy = parallel_apply_video_clip(py, numpy_4d, sampled_prog, Some(is_inplace), num_threads)?;
                let torch_mod = py.import("torch")?;
                let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
                let tchw_res = res_tensor
                    .call_method1("permute", ((0, 3, 1, 2),))?
                    .call_method0("contiguous")?;
                return Ok(tchw_res);
            } else {
                // Layout is THWC
                let cont_tensor = if is_inplace {
                    video
                } else {
                    video.call_method0("contiguous")?
                };
                let numpy_4d = cont_tensor.call_method0("numpy")?;
                let out_numpy = parallel_apply_video_clip(py, numpy_4d, sampled_prog, Some(is_inplace), num_threads)?;
                let torch_mod = py.import("torch")?;
                let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
                return Ok(res_tensor);
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 4D PyTorch tensor for video clip, got shape {:?}",
                shape
            )));
        }
    }

    // Case 2: 4D numpy array (T, H, W, C)
    if let Ok(arr4) = video.downcast::<PyArray4<u8>>() {
        let shape = arr4.shape();
        let (num_frames, height, width, channels) = (
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
        );

        if num_frames == 0 {
            return Ok(arr4.as_ref());
        }

        let is_c_contiguous = video
            .getattr("flags")
            .and_then(|f| f.getattr("c_contiguous"))
            .and_then(|c| c.extract::<bool>())
            .unwrap_or(true);

        if is_inplace {
            if !is_c_contiguous {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot mutate non-contiguous video clip with inplace=True; use inplace=False to allow defensive copy",
                ));
            }
            let is_writeable = video
                .getattr("flags")
                .and_then(|f| f.getattr("writeable"))
                .and_then(|w| w.extract::<bool>())
                .unwrap_or(true);
            if !is_writeable {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cannot mutate read-only video clip with inplace=True; use inplace=False to create a defensive copy",
                ));
            }
        }

        // Optimize plan ONCE for the entire clip
        let plan = sampled_prog.to_plan();
        let exec_plan = Optimizer::new().optimize(plan);

        let working_array = if is_inplace && is_c_contiguous {
            arr4
        } else {
            let copied = video.call_method0("copy")?;
            copied.downcast::<PyArray4<u8>>()?
        };

        let slice = unsafe { working_array.as_slice_mut()? };
        let img_slice_len = height * width * channels;

        let execute_fn = || {
            let chunks: Vec<&mut [u8]> = slice.chunks_exact_mut(img_slice_len).collect();
            let barriers: Vec<Option<crate::core::BarrierImage>> = chunks
                .into_par_iter()
                .map(|chunk| {
                    let mut fusable = FusableImage::new(chunk, width, height, channels);
                    exec_plan.execute(&mut fusable)
                })
                .collect();
            barriers
        };

        let barriers = match num_threads {
            Some(n) if n > 0 => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                py.allow_threads(|| pool.install(execute_fn))
            }
            _ => py.allow_threads(execute_fn),
        };

        let (first_h, first_w, first_c) = match &barriers[0] {
            Some(b) => (b.height, b.width, b.channels),
            None => (height, width, channels),
        };

        let item_len = first_h * first_w * first_c;
        let mut out_vec = Vec::with_capacity(num_frames * item_len);
        for (i, maybe_b) in barriers.into_iter().enumerate() {
            if let Some(b) = maybe_b {
                out_vec.extend_from_slice(&b.data);
            } else {
                let start = i * img_slice_len;
                let end = start + img_slice_len;
                out_vec.extend_from_slice(&slice[start..end]);
            }
        }
        let arr1 = PyArray1::from_vec(py, out_vec);
        let arr4_out = arr1.reshape([num_frames, first_h, first_w, first_c]).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to reshape 4D video clip array: {}",
                e
            ))
        })?;
        return Ok(arr4_out.as_ref());
    }

    // Case 3: Python list/sequence of frames
    if let Ok(seq) = video.downcast::<PySequence>() {
        let len = seq.len()?;
        if len == 0 {
            return Ok(PyList::empty(py).as_ref());
        }

        let py_prog = crate::python::sampled::PySampledImageProgram {
            inner: sampled_prog.clone(),
        };

        let mut results = Vec::with_capacity(len);
        for i in 0..len {
            let item = seq.get_item(i)?;
            let transformed = py_prog.apply(item, Some(is_inplace), None, py)?;
            results.push(transformed);
        }

        let result_list = PyList::new(py, results);
        return Ok(result_list.as_ref());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected 4D numpy array, 4D torch.Tensor, or sequence of frames for video clip",
    ))
}

/// Apply video augmentation to a batch of video clips [B, T, C, H, W], [B, T, H, W, C], or list of clips
#[cfg(feature = "python")]
pub fn parallel_apply_video_batch<'py, F>(
    py: Python<'py>,
    videos: &'py PyAny,
    inplace: Option<bool>,
    num_threads: Option<usize>,
    get_clip_program: F,
) -> PyResult<&'py PyAny>
where
    F: Fn(usize) -> SampledImageProgram + Sync + Send,
{
    let is_inplace = inplace.unwrap_or(false);

    // Case 1: 5D PyTorch Tensor (B, T, C, H, W) or (B, T, H, W, C)
    if is_torch_tensor(videos) {
        if let Ok(is_cuda) = videos.getattr("is_cuda") {
            if is_cuda.extract::<bool>().unwrap_or(false) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Sinter operates on CPU memory; pass a CPU tensor (e.g., tensor.cpu())",
                ));
            }
        }

        let shape: Vec<usize> = videos.getattr("shape")?.extract()?;
        if shape.len() == 5 {
            let (_b, _t, c, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
            if (c == 1 || c == 3 || c == 4) && (h > 4 || w > 4) {
                // Layout is BTCHW -> permute to BTHWC
                let bthwc_tensor = videos
                    .call_method1("permute", ((0, 1, 3, 4, 2),))?
                    .call_method0("contiguous")?;
                let numpy_5d = bthwc_tensor.call_method0("numpy")?;
                let out_numpy = parallel_apply_video_batch(py, numpy_5d, Some(is_inplace), num_threads, get_clip_program)?;
                let torch_mod = py.import("torch")?;
                let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
                let btchw_res = res_tensor
                    .call_method1("permute", ((0, 1, 4, 2, 3),))?
                    .call_method0("contiguous")?;
                return Ok(btchw_res);
            } else {
                // Layout is BTHWC
                let cont_tensor = if is_inplace {
                    videos
                } else {
                    videos.call_method0("contiguous")?
                };
                let numpy_5d = cont_tensor.call_method0("numpy")?;
                let out_numpy = parallel_apply_video_batch(py, numpy_5d, Some(is_inplace), num_threads, get_clip_program)?;
                let torch_mod = py.import("torch")?;
                let res_tensor = torch_mod.call_method1("from_numpy", (out_numpy,))?;
                return Ok(res_tensor);
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 5D PyTorch tensor for video batch, got shape {:?}",
                shape
            )));
        }
    }

    // Case 2: 5D numpy array (B, T, H, W, C) via numpy shape check
    if let Ok(shape_obj) = videos.getattr("shape") {
        if let Ok(shape) = shape_obj.extract::<Vec<usize>>() {
            if shape.len() == 5 {
                let batch_size = shape[0];
                let mut out_clips = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let clip_item = videos.get_item(b)?;
                    let prog = get_clip_program(b);
                    let transformed_clip = parallel_apply_video_clip(py, clip_item, &prog, Some(is_inplace), num_threads)?;
                    out_clips.push(transformed_clip);
                }

                let np_mod = py.import("numpy")?;
                let stacked = np_mod.call_method1("stack", (PyList::new(py, out_clips), 0))?;
                return Ok(stacked);
            }
        }
    }

    // Case 3: Sequence / List of 4D video clips
    if let Ok(seq) = videos.downcast::<PySequence>() {
        let len = seq.len()?;
        let mut results = Vec::with_capacity(len);
        for b in 0..len {
            let clip = seq.get_item(b)?;
            let prog = get_clip_program(b);
            let transformed = parallel_apply_video_clip(py, clip, &prog, Some(is_inplace), num_threads)?;
            results.push(transformed);
        }
        let result_list = PyList::new(py, results);
        return Ok(result_list.as_ref());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected 5D numpy array, 5D torch.Tensor, or sequence of 4D clips for apply_video_batch",
    ))
}
