// Python wrappers for transforms with distribution support
//
// This module provides the **unified API** for sinter.
// All transforms accept either plain values (constant) or distributions.

use crate::core::FusableImage;
use crate::exec_ir::Optimizer;
use crate::sampled_ir::ops::{BorderMode, EdgeMethod, EmbossDirection, Interpolation, PadMode, RotateAngle};
use crate::sampled_ir::Plan;
use crate::sampling::{Dist, RandomImageNode, RandomImageProgram};

use super::super::distributions::{format_dist, parse_distribution};
use super::super::enums::{
    parse_border_mode, parse_edge_method, parse_emboss_direction, parse_interpolation, parse_pad_mode,
    parse_rotate_angle,
};
use super::super::sampled::PySampledImageProgram;
use super::compose::PyCompose;
use super::helpers::maybe_wrap;

#[cfg(feature = "python")]
use numpy::{PyArray2, PyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

// ============================================================================
// Helper application functions
// ============================================================================

#[cfg(feature = "python")]
fn apply_node_to_targets<'py>(
    node: RandomImageNode,
    p: Dist,
    image: &'py PyAny,
    bboxes: Option<&'py PyAny>,
    keypoints: Option<&'py PyAny>,
    masks: Option<&'py PyAny>,
    mask: Option<&'py PyAny>,
    bbox_format: &str,
    keypoint_format: &str,
    inplace: Option<bool>,
    py: Python<'py>,
) -> PyResult<PyObject> {
    // Image-only calls return the transformed array directly (parity with
    // .apply()); passing any label target switches to the dict form, like
    // Compose.__call__.
    if bboxes.is_none() && keypoints.is_none() && masks.is_none() && mask.is_none() {
        return Ok(apply_node_to_image(node, p, image, inplace, py)?.to_object(py));
    }

    let mut prog = RandomImageProgram::new();
    prog.add(maybe_wrap(node, p));
    let compose = PyCompose {
        inner: prog,
        transforms: Vec::new(),
    };
    compose.__call__(
        image,
        bboxes,
        keypoints,
        masks,
        mask,
        bbox_format,
        keypoint_format,
        None,
        inplace,
        None, // labels
        py,
    )
}

#[cfg(feature = "python")]
fn apply_node_to_image<'py>(
    node: RandomImageNode,
    p: Dist,
    array: &'py PyAny,
    inplace: Option<bool>,
    py: Python<'py>,
) -> PyResult<&'py PyAny> {
    let mut prog = RandomImageProgram::new();
    prog.add(maybe_wrap(node, p));
    let compose = PyCompose {
        inner: prog,
        transforms: Vec::new(),
    };
    compose.apply(array, inplace, None, py)
}

// ============================================================================
// Macros to reduce boilerplate
// ============================================================================

/// Parse a tuple distribution (x, y) from Python
#[cfg(feature = "python")]
fn parse_tuple_dist(val: Option<&PyAny>, default: f32) -> PyResult<(Dist, Dist)> {
    if let Some(v) = val {
        if let Ok((x, y)) = v.extract::<(f32, f32)>() {
            return Ok((Dist::constant(x), Dist::constant(y)));
        }
        if let Ok(s) = v.extract::<f32>() {
            return Ok((Dist::constant(s), Dist::constant(s)));
        }
        let dist = parse_distribution(v)?;
        Ok((dist.clone(), dist))
    } else {
        Ok((Dist::constant(default), Dist::constant(default)))
    }
}

/// Parse a distribution with default value
#[cfg(feature = "python")]
fn parse_dist_with_default(val: Option<&PyAny>, default: f32) -> PyResult<Dist> {
    if let Some(v) = val {
        if let Ok(f) = v.extract::<f32>() {
            return Ok(Dist::constant(f));
        }
        parse_distribution(v)
    } else {
        Ok(Dist::constant(default))
    }
}

/// Parse the probability distribution (defaults to 1.0)
#[cfg(feature = "python")]
fn parse_p_dist(p: Option<&PyAny>) -> PyResult<Dist> {
    Ok(if let Some(val) = p {
        parse_distribution(val)?
    } else {
        Dist::constant(1.0)
    })
}

/// Parse an array of distributions from Python
#[cfg(feature = "python")]
fn parse_array_dist(value: &PyAny, len: usize) -> PyResult<[Dist; 4]> {
    if let Ok(iter) = value.iter() {
        let mut dists = [Dist::constant(0.0); 4];
        for (i, item_result) in iter.enumerate() {
            if i >= len {
                break;
            }
            dists[i] = parse_distribution(item_result?)?;
        }
        return Ok(dists);
    }

    let dist = parse_distribution(value)?;
    Ok([dist.clone(), dist.clone(), dist.clone(), dist])
}

fn parse_channel_shuffle_order(val: Option<&PyAny>) -> PyResult<u8> {
    if let Some(v) = val {
        if let Ok(num) = v.extract::<u8>() {
            if num > 5 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "order must be 0-5, got {}",
                    num
                )));
            }
            return Ok(num);
        }
        if let Ok(s) = v.extract::<&str>() {
            let clean = s.trim().to_uppercase();
            match clean.as_str() {
                "RGB" => return Ok(0),
                "RBG" => return Ok(1),
                "GRB" => return Ok(2),
                "GBR" => return Ok(3),
                "BRG" => return Ok(4),
                "BGR" => return Ok(5),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown channel shuffle order '{}'. Expected one of: 'RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR' (or 0-5)",
                        s
                    )))
                }
            }
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "order must be int (0-5) or str ('RGB', 'BGR', etc.), got {}",
            v.get_type().name()?
        )))
    } else {
        Ok(0)
    }
}

macro_rules! define_basic_transforms {
    (
        $( $(#[$meta:meta])* $name:ident ( $py_struct:ident, $py_name:literal ) { $( $field:ident : $default_expr:expr ),* } => $node_gen:expr ),*
    ) => {
        $(
            $(#[$meta])*
            #[cfg(feature = "python")]
            #[pyclass(name = $py_name)]
            pub struct $py_struct {
                $( pub $field: Dist, )*
                pub p: Dist,
            }

            #[cfg(feature = "python")]
            #[pymethods]
            impl $py_struct {
                #[new]
                #[pyo3(signature = ( $( $field=None, )* p=None ))]
                fn new( $( $field: Option<&PyAny>, )* p: Option<&PyAny> ) -> PyResult<Self> {
                    Ok(Self {
                        $(
                            $field: match $field {
                                Some(v) => parse_distribution(v)?,
                                None => Dist::constant($default_expr),
                            },
                        )*
                        p: parse_p_dist(p)?,
                    })
                }

                #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
                fn __call__<'py>(
                    &self,
                    image: &'py PyAny,
                    bboxes: Option<&'py PyAny>,
                    keypoints: Option<&'py PyAny>,
                    masks: Option<&'py PyAny>,
                    mask: Option<&'py PyAny>,
                    bbox_format: &str,
                    keypoint_format: &str,
                    inplace: Option<bool>,
                    py: Python<'py>,
                ) -> PyResult<PyObject> {
                    let gen = $node_gen;
                    let node = gen(self);
                    apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
                }

                #[pyo3(signature = (array, inplace=None))]
                fn apply<'py>(
                    &self,
                    array: &'py PyAny,
                    inplace: Option<bool>,
                    py: Python<'py>,
                ) -> PyResult<&'py PyAny> {
                    let gen = $node_gen;
                    let node = gen(self);
                    apply_node_to_image(node, self.p.clone(), array, inplace, py)
                }

                fn __repr__(&self) -> String {
                    let mut s = String::from($py_name);
                    s.push('(');
                    $(
                        s.push_str(concat!(stringify!($field), "="));
                        s.push_str(&format_dist(&self.$field));
                        s.push_str(", ");
                    )*
                    s.push_str("p=");
                    s.push_str(&format_dist(&self.p));
                    s.push(')');
                    s
                }
            }
        )*

        #[cfg(feature = "python")]
        fn extract_basic_node(item: &PyAny) -> PyResult<Option<RandomImageNode>> {
            $(
                if let Ok(obj) = item.extract::<PyRef<$py_struct>>() {
                    let node = $node_gen(&obj);
                    return Ok(Some(maybe_wrap(node, obj.p.clone())));
                }
            )*
            Ok(None)
        }
    }
}

// ============================================================================
// Transform Classes
// ============================================================================

define_basic_transforms! {
    #[doc = "Flip horizontally (mirror left-right)."]
    HorizontalFlip(PyHorizontalFlip, "HorizontalFlip") {} => |_| RandomImageNode::HorizontalFlip,
    #[doc = "Flip vertically (mirror top-bottom)."]
    VerticalFlip(PyVerticalFlip, "VerticalFlip") {} => |_| RandomImageNode::VerticalFlip,
    #[doc = "Transpose axes (swap width and height)."]
    Transpose(PyTranspose, "Transpose") {} => |_| RandomImageNode::Transpose,
    #[doc = "Invert pixel values (255 - x)."]
    Invert(PyInvert, "Invert") {} => |_| RandomImageNode::Invert,
    #[doc = "Convert RGB to grayscale (single channel)."]
    ToGray(PyToGray, "ToGray") {} => |_| RandomImageNode::ToGray,
    #[doc = "Apply a sepia color matrix."]
    ToSepia(PyToSepia, "ToSepia") {} => |_| RandomImageNode::ToSepia,
    #[doc = "Convert grayscale to RGB (channel replication)."]
    ToRGB(PyToRGB, "ToRGB") {} => |_| RandomImageNode::ToRGB,
    #[doc = "Histogram equalization (per channel)."]
    Equalize(PyEqualize, "Equalize") {} => |_| RandomImageNode::Equalize,
    #[doc = "Stretch contrast to the full range, optionally ignoring extreme outliers (cutoff 0.0-0.5)."]
    AutoContrast(PyAutoContrast, "AutoContrast") { cutoff: 0.0 } => |obj: &PyAutoContrast| RandomImageNode::AutoContrast { cutoff: obj.cutoff.clone() },

    #[doc = "Add a constant delta to pixel values."]
    Brightness(PyBrightness, "Brightness") { delta: 0.0 } => |obj: &PyBrightness| RandomImageNode::Brightness { delta: obj.delta.clone() },
    #[doc = "Scale contrast by a factor."]
    Contrast(PyContrast, "Contrast") { factor: 1.0 } => |obj: &PyContrast| RandomImageNode::Contrast { factor: obj.factor.clone() },
    #[doc = "Reduce the number of bits per channel (posterize)."]
    Posterize(PyPosterize, "Posterize") { bits: 4.0 } => |obj: &PyPosterize| RandomImageNode::Posterize { bits: obj.bits.clone() },
    #[doc = "Solarize: invert pixels above a threshold."]
    Solarize(PySolarize, "Solarize") { threshold: 128.0 } => |obj: &PySolarize| RandomImageNode::Solarize { threshold: obj.threshold.clone() },
    #[doc = "Apply a gamma correction."]
    Gamma(PyGamma, "Gamma") { gamma: 1.0 } => |obj: &PyGamma| RandomImageNode::Gamma { gamma: obj.gamma.clone() },
    #[doc = "Sharpen the image (3x3 laplacian blend, strength 0-1)."]
    Sharpen(PySharpen, "Sharpen") { strength: 0.5 } => |obj: &PySharpen| RandomImageNode::Sharpen { strength: obj.strength.clone() },

    #[doc = "Multiply each pixel by a random factor (speckle noise)."]
    MultiplicativeNoise(PyMultiplicativeNoise, "MultiplicativeNoise") { multiplier: 1.0 } => |obj: &PyMultiplicativeNoise| RandomImageNode::MultiplicativeNoise { multiplier: obj.multiplier.clone() },
    #[doc = "Replace random pixels with salt (255) or pepper (0)."]
    SaltAndPepper(PySaltAndPepper, "SaltAndPepper") { amount: 0.05, salt_vs_pepper: 0.5 } => |obj: &PySaltAndPepper| RandomImageNode::SaltAndPepper { amount: obj.amount.clone(), salt_vs_pepper: obj.salt_vs_pepper.clone() },

    #[doc = "Adjust color temperature (Kelvin-like tint)."]
    ColorTemperature(PyColorTemperature, "ColorTemperature") { temperature: 0.0 } => |obj: &PyColorTemperature| RandomImageNode::ColorTemperature { temperature: obj.temperature.clone() },
    #[doc = "Scale the R/G/B channels independently."]
    ColorBalance(PyColorBalance, "ColorBalance") { r_scale: 1.0, g_scale: 1.0, b_scale: 1.0 } => |obj: &PyColorBalance| RandomImageNode::ColorBalance { r_scale: obj.r_scale.clone(), g_scale: obj.g_scale.clone(), b_scale: obj.b_scale.clone() }
}

// ============================================================================
// Discrete Parameter Transforms Macro
// ============================================================================

macro_rules! define_discrete_transforms {
    (
        $( $(#[$meta:meta])* $name:ident ( $py_struct:ident, $py_name:literal ) { $field:ident : $type:ty = $default:expr, values: $valid_values:expr } => $node_gen:expr ),*
    ) => {
        $(
            $(#[$meta])*
            #[cfg(feature = "python")]
            #[pyclass(name = $py_name)]
            pub struct $py_struct {
                pub $field: $type,
                pub p: Dist,
            }

            #[cfg(feature = "python")]
            #[pymethods]
            impl $py_struct {
                #[new]
                #[pyo3(signature = ( $field=None, p=None ))]
                fn new( $field: Option<$type>, p: Option<&PyAny> ) -> PyResult<Self> {
                    let val = $field.unwrap_or($default);
                    let valid = $valid_values;
                    if !valid.contains(&val) {
                         return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                             "{} must be one of {:?}, got {}",
                             stringify!($field), valid, val
                         )));
                    }
                    Ok(Self {
                        $field: val,
                        p: parse_p_dist(p)?,
                    })
                }

                #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
                fn __call__<'py>(
                    &self,
                    image: &'py PyAny,
                    bboxes: Option<&'py PyAny>,
                    keypoints: Option<&'py PyAny>,
                    masks: Option<&'py PyAny>,
                    mask: Option<&'py PyAny>,
                    bbox_format: &str,
                    keypoint_format: &str,
                    inplace: Option<bool>,
                    py: Python<'py>,
                ) -> PyResult<PyObject> {
                    let gen = $node_gen;
                    let node = gen(self);
                    apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
                }

                #[pyo3(signature = (array, inplace=None))]
                fn apply<'py>(
                    &self,
                    array: &'py PyAny,
                    inplace: Option<bool>,
                    py: Python<'py>,
                ) -> PyResult<&'py PyAny> {
                    let gen = $node_gen;
                    let node = gen(self);
                    apply_node_to_image(node, self.p.clone(), array, inplace, py)
                }

                fn __repr__(&self) -> String {
                    format!("{}({}={}, p={})", $py_name, stringify!($field), self.$field, format_dist(&self.p))
                }
            }
        )*

        #[cfg(feature = "python")]
        fn extract_discrete_node(item: &PyAny) -> PyResult<Option<RandomImageNode>> {
            $(
                if let Ok(obj) = item.extract::<PyRef<$py_struct>>() {
                    let node = $node_gen(&obj);
                    return Ok(Some(maybe_wrap(node, obj.p.clone())));
                }
            )*
            Ok(None)
        }
    }
}

define_discrete_transforms! {
    #[doc = "Gaussian blur with a fixed kernel size (3, 5, 7, 13, 21, 31)."]
    GaussianBlur(PyGaussianBlur, "GaussianBlur") { kernel_size: u32 = 3, values: [3, 5, 7, 13, 21, 31] } => |obj: &PyGaussianBlur| RandomImageNode::GaussianBlur { kernel_size: obj.kernel_size },
    #[doc = "Median blur with a 3x3 or 5x5 kernel."]
    MedianBlur(PyMedianBlur, "MedianBlur") { kernel_size: u32 = 3, values: [3, 5] } => |obj: &PyMedianBlur| RandomImageNode::MedianBlur { kernel_size: obj.kernel_size }
}

/// ChannelShuffle - shuffle RGB channels
#[cfg(feature = "python")]
#[pyclass(name = "ChannelShuffle")]
pub struct PyChannelShuffle {
    pub order: u8, // 0-5 representing different shuffle patterns
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyChannelShuffle {
    #[new]
    #[pyo3(signature = (order=None, p=None))]
    fn new(order: Option<&PyAny>, p: Option<&PyAny>) -> PyResult<Self> {
        let order_val = parse_channel_shuffle_order(order)?;
        Ok(Self {
            order: order_val,
            p: parse_p_dist(p)?,
        })
    }

    #[staticmethod]
    fn rgb() -> PyResult<Self> {
        Ok(Self {
            order: 0,
            p: Dist::constant(1.0),
        })
    }
    #[staticmethod]
    fn rbg() -> PyResult<Self> {
        Ok(Self {
            order: 1,
            p: Dist::constant(1.0),
        })
    }
    #[staticmethod]
    fn grb() -> PyResult<Self> {
        Ok(Self {
            order: 2,
            p: Dist::constant(1.0),
        })
    }
    #[staticmethod]
    fn gbr() -> PyResult<Self> {
        Ok(Self {
            order: 3,
            p: Dist::constant(1.0),
        })
    }
    #[staticmethod]
    fn brg() -> PyResult<Self> {
        Ok(Self {
            order: 4,
            p: Dist::constant(1.0),
        })
    }
    #[staticmethod]
    fn bgr() -> PyResult<Self> {
        Ok(Self {
            order: 5,
            p: Dist::constant(1.0),
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let node = RandomImageNode::ChannelShuffle {
            order: permutations[(self.order as usize) % 6],
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let node = RandomImageNode::ChannelShuffle {
            order: permutations[(self.order as usize) % 6],
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        let names = ["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"];
        format!(
            "ChannelShuffle(order='{}', p={})",
            names[self.order as usize],
            format_dist(&self.p)
        )
    }
}

fn parse_hsv_scale_dist(val: &PyAny, is_shift: bool) -> PyResult<Dist> {
    let dist = parse_distribution(val)?;
    if is_shift {
        match dist {
            Dist::Constant(v) => {
                let scale = if v.abs() > 1.0 { 1.0 + v / 255.0 } else { 1.0 + v };
                Ok(Dist::constant(scale.max(0.0)))
            }
            Dist::Uniform { min, max } => {
                let s_min = if min.abs() > 1.0 || max.abs() > 1.0 { 1.0 + min / 255.0 } else { 1.0 + min };
                let s_max = if min.abs() > 1.0 || max.abs() > 1.0 { 1.0 + max / 255.0 } else { 1.0 + max };
                Ok(Dist::uniform(s_min.max(0.0), s_max.max(0.0)))
            }
            Dist::Normal { mu, sigma } => {
                let s_mu = if mu.abs() > 1.0 || sigma > 1.0 { 1.0 + mu / 255.0 } else { 1.0 + mu };
                let s_sigma = if mu.abs() > 1.0 || sigma > 1.0 { sigma / 255.0 } else { sigma };
                Ok(Dist::normal(s_mu.max(0.0), s_sigma))
            }
            Dist::UniformInt { min, max } => {
                let s_min = 1.0 + (min as f32) / 255.0;
                let s_max = 1.0 + (max as f32) / 255.0;
                Ok(Dist::uniform(s_min.max(0.0), s_max.max(0.0)))
            }
            _ => Ok(dist),
        }
    } else {
        match dist {
            Dist::Uniform { min, max } if min < 0.0 => {
                let s_min = if min.abs() > 1.0 || max.abs() > 1.0 { 1.0 + min / 255.0 } else { 1.0 + min };
                let s_max = if min.abs() > 1.0 || max.abs() > 1.0 { 1.0 + max / 255.0 } else { 1.0 + max };
                Ok(Dist::uniform(s_min.max(0.0), s_max.max(0.0)))
            }
            _ => Ok(dist),
        }
    }
}

/// HueSaturationValue - adjust hue, saturation, and value
#[cfg(feature = "python")]
#[pyclass(name = "HueSaturationValue")]
pub struct PyHueSaturationValue {
    pub hue_shift: Dist,
    pub saturation_scale: Dist,
    pub value_scale: Dist,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyHueSaturationValue {
    #[new]
    #[pyo3(signature = (
        hue_shift=None,
        saturation_scale=None,
        value_scale=None,
        hue_shift_limit=None,
        sat_shift_limit=None,
        val_shift_limit=None,
        sat_shift=None,
        val_shift=None,
        hue=None,
        sat=None,
        val=None,
        p=None
    ))]
    fn new(
        hue_shift: Option<&PyAny>,
        saturation_scale: Option<&PyAny>,
        value_scale: Option<&PyAny>,
        hue_shift_limit: Option<&PyAny>,
        sat_shift_limit: Option<&PyAny>,
        val_shift_limit: Option<&PyAny>,
        sat_shift: Option<&PyAny>,
        val_shift: Option<&PyAny>,
        hue: Option<&PyAny>,
        sat: Option<&PyAny>,
        val: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let h = hue_shift
            .or(hue_shift_limit)
            .or(hue);
        let s = saturation_scale
            .or(sat_shift_limit)
            .or(sat_shift)
            .or(sat);
        let v = value_scale
            .or(val_shift_limit)
            .or(val_shift)
            .or(val);

        let h_dist = match h {
            Some(val) => parse_distribution(val)?,
            None => Dist::constant(0.0),
        };
        let is_s_shift = sat_shift_limit.is_some() || sat_shift.is_some();
        let is_v_shift = val_shift_limit.is_some() || val_shift.is_some();

        let s_dist = match s {
            Some(val) => parse_hsv_scale_dist(val, is_s_shift)?,
            None => Dist::constant(1.0),
        };
        let v_dist = match v {
            Some(val) => parse_hsv_scale_dist(val, is_v_shift)?,
            None => Dist::constant(1.0),
        };

        Ok(Self {
            hue_shift: h_dist,
            saturation_scale: s_dist,
            value_scale: v_dist,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::HueSaturationValue {
            hue_shift: self.hue_shift.clone(),
            saturation_scale: self.saturation_scale.clone(),
            value_scale: self.value_scale.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::HueSaturationValue {
            hue_shift: self.hue_shift.clone(),
            saturation_scale: self.saturation_scale.clone(),
            value_scale: self.value_scale.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "HueSaturationValue(hue_shift={}, saturation_scale={}, value_scale={}, p={})",
            format_dist(&self.hue_shift),
            format_dist(&self.saturation_scale),
            format_dist(&self.value_scale),
            format_dist(&self.p)
        )
    }
}

/// RGBShift - shift each RGB channel
#[cfg(feature = "python")]
#[pyclass(name = "RGBShift")]
pub struct PyRGBShift {
    pub r_shift: Dist,
    pub g_shift: Dist,
    pub b_shift: Dist,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRGBShift {
    #[new]
    #[pyo3(signature = (
        r_shift=None,
        g_shift=None,
        b_shift=None,
        r_shift_limit=None,
        g_shift_limit=None,
        b_shift_limit=None,
        r=None,
        g=None,
        b=None,
        p=None
    ))]
    fn new(
        r_shift: Option<&PyAny>,
        g_shift: Option<&PyAny>,
        b_shift: Option<&PyAny>,
        r_shift_limit: Option<&PyAny>,
        g_shift_limit: Option<&PyAny>,
        b_shift_limit: Option<&PyAny>,
        r: Option<&PyAny>,
        g: Option<&PyAny>,
        b: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let r_in = r_shift.or(r_shift_limit).or(r);
        let g_in = g_shift.or(g_shift_limit).or(g);
        let b_in = b_shift.or(b_shift_limit).or(b);

        let r_dist = match r_in {
            Some(val) => parse_distribution(val)?,
            None => Dist::constant(0.0),
        };
        let g_dist = match g_in {
            Some(val) => parse_distribution(val)?,
            None => Dist::constant(0.0),
        };
        let b_dist = match b_in {
            Some(val) => parse_distribution(val)?,
            None => Dist::constant(0.0),
        };

        Ok(Self {
            r_shift: r_dist,
            g_shift: g_dist,
            b_shift: b_dist,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::RGBShift {
            r_shift: self.r_shift.clone(),
            g_shift: self.g_shift.clone(),
            b_shift: self.b_shift.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::RGBShift {
            r_shift: self.r_shift.clone(),
            g_shift: self.g_shift.clone(),
            b_shift: self.b_shift.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "RGBShift(r_shift={}, g_shift={}, b_shift={}, p={})",
            format_dist(&self.r_shift),
            format_dist(&self.g_shift),
            format_dist(&self.b_shift),
            format_dist(&self.p)
        )
    }
}

/// GaussNoise - add Gaussian noise
#[cfg(feature = "python")]
#[pyclass(name = "GaussNoise")]
pub struct PyGaussNoise {
    pub mean: Dist,
    pub std: Dist,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGaussNoise {
    #[new]
    #[pyo3(signature = (
        mean=None,
        std=None,
        mu=None,
        sigma=None,
        var_limit=None,
        p=None
    ))]
    fn new(
        mean: Option<&PyAny>,
        std: Option<&PyAny>,
        mu: Option<&PyAny>,
        sigma: Option<&PyAny>,
        var_limit: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let mean_in = mean.or(mu);
        let std_in = std.or(sigma);

        let mean_dist = match mean_in {
            Some(val) => parse_distribution(val)?,
            None => Dist::constant(0.0),
        };

        let std_dist = if let Some(val) = std_in {
            parse_distribution(val)?
        } else if let Some(vl) = var_limit {
            if let Ok((min_v, max_v)) = vl.extract::<(f32, f32)>() {
                Dist::uniform(min_v.max(0.0).sqrt(), max_v.max(0.0).sqrt())
            } else if let Ok(v) = vl.extract::<f32>() {
                Dist::uniform(0.0, v.max(0.0).sqrt())
            } else {
                parse_distribution(vl)?
            }
        } else {
            Dist::constant(10.0)
        };

        Ok(Self {
            mean: mean_dist,
            std: std_dist,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::GaussNoise {
            mean: self.mean.clone(),
            std: self.std.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::GaussNoise {
            mean: self.mean.clone(),
            std: self.std.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussNoise(mean={}, std={}, p={})",
            format_dist(&self.mean),
            format_dist(&self.std),
            format_dist(&self.p)
        )
    }
}

/// Crop - crop a rectangular region
#[cfg(feature = "python")]
#[pyclass(name = "Crop")]
pub struct PyCrop {
    pub x: Dist,
    pub y: Dist,
    pub width: Dist,
    pub height: Dist,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCrop {
    #[new]
    #[pyo3(signature = (
        x=None,
        y=None,
        width=None,
        height=None,
        x_min=None,
        y_min=None,
        x_max=None,
        y_max=None,
        w=None,
        h=None,
        p=None
    ))]
    fn new(
        x: Option<&PyAny>,
        y: Option<&PyAny>,
        width: Option<&PyAny>,
        height: Option<&PyAny>,
        x_min: Option<&PyAny>,
        y_min: Option<&PyAny>,
        x_max: Option<&PyAny>,
        y_max: Option<&PyAny>,
        w: Option<&PyAny>,
        h: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let x_in = x.or(x_min);
        let y_in = y.or(y_min);
        let w_in = width.or(w);
        let h_in = height.or(h);

        let x_dist = match x_in {
            Some(val) => parse_distribution(val)?,
            None => Dist::constant(0.0),
        };
        let y_dist = match y_in {
            Some(val) => parse_distribution(val)?,
            None => Dist::constant(0.0),
        };

        let width_dist = if let Some(val) = w_in {
            parse_distribution(val)?
        } else if let (Some(xmin), Some(xmax)) = (x_min, x_max) {
            let x0 = xmin.extract::<f32>().unwrap_or(0.0);
            let x1 = xmax.extract::<f32>().unwrap_or(100.0);
            Dist::constant((x1 - x0).max(1.0))
        } else {
            Dist::constant(100.0)
        };

        let height_dist = if let Some(val) = h_in {
            parse_distribution(val)?
        } else if let (Some(ymin), Some(ymax)) = (y_min, y_max) {
            let y0 = ymin.extract::<f32>().unwrap_or(0.0);
            let y1 = ymax.extract::<f32>().unwrap_or(100.0);
            Dist::constant((y1 - y0).max(1.0))
        } else {
            Dist::constant(100.0)
        };

        Ok(Self {
            x: x_dist,
            y: y_dist,
            width: width_dist,
            height: height_dist,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::Crop {
            x: self.x.clone(),
            y: self.y.clone(),
            width: self.width.clone(),
            height: self.height.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::Crop {
            x: self.x.clone(),
            y: self.y.clone(),
            width: self.width.clone(),
            height: self.height.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Crop(x={}, y={}, width={}, height={}, p={})",
            format_dist(&self.x),
            format_dist(&self.y),
            format_dist(&self.width),
            format_dist(&self.height),
            format_dist(&self.p)
        )
    }
}

// ============================================================================
// Manual Transform Classes
// ============================================================================

/// Rotate - rotate by 90, 180, or 270 degrees
/// Rotate by a fixed angle (90/180/270) or a random one via a distribution.
#[cfg(feature = "python")]
#[pyclass(name = "Rotate")]
pub struct PyRotate {
    pub angle: RotateAngle,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRotate {
    #[new]
    #[pyo3(signature = (angle=None, p=None))]
    fn new(angle: Option<&PyAny>, p: Option<&PyAny>) -> PyResult<Self> {
        let angle_val = if let Some(a) = angle {
            parse_rotate_angle(a)?
        } else {
            RotateAngle::Rotate90
        };
        Ok(Self {
            angle: angle_val,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::Rotate { angle: self.angle };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::Rotate { angle: self.angle };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        let angle_str = match self.angle {
            RotateAngle::Rotate90 => "90",
            RotateAngle::Rotate180 => "180",
            RotateAngle::Rotate270 => "270",
        };
        format!("Rotate(angle={}, p={})", angle_str, format_dist(&self.p))
    }
}

/// Resize - resize to specific dimensions
/// Resize to a target width/height (nearest or bilinear).
#[cfg(feature = "python")]
#[pyclass(name = "Resize")]
pub struct PyResize {
    pub width: u32,
    pub height: u32,
    pub interpolation: Interpolation,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyResize {
    #[new]
    #[pyo3(signature = (width, height, interpolation=None, p=None))]
    fn new(
        width: u32,
        height: u32,
        interpolation: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "width and height must be positive",
            ));
        }
        let interp = if let Some(i) = interpolation {
            parse_interpolation(i)?
        } else {
            Interpolation::Bilinear
        };
        Ok(Self {
            width,
            height,
            interpolation: interp,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::Resize {
            width: self.width,
            height: self.height,
            interpolation: self.interpolation,
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::Resize {
            width: self.width,
            height: self.height,
            interpolation: self.interpolation,
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        let interp_str = match self.interpolation {
            Interpolation::Nearest => "nearest",
            Interpolation::Bilinear => "bilinear",
            Interpolation::Bicubic => "bicubic",
            Interpolation::Lanczos4 => "lanczos4",
        };
        format!(
            "Resize(width={}, height={}, interpolation='{}', p={})",
            self.width,
            self.height,
            interp_str,
            format_dist(&self.p)
        )
    }
}

/// Pad - pad image
/// Pad the image on each side (constant, reflect, replicate, or wrap).
#[cfg(feature = "python")]
#[pyclass(name = "Pad")]
pub struct PyPad {
    pub top: Dist,
    pub bottom: Dist,
    pub left: Dist,
    pub right: Dist,
    pub mode: PadMode,
    pub value: u8,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPad {
    #[new]
    #[pyo3(signature = (top=None, bottom=None, left=None, right=None, mode=None, value=None, p=None))]
    fn new(
        top: Option<&PyAny>,
        bottom: Option<&PyAny>,
        left: Option<&PyAny>,
        right: Option<&PyAny>,
        mode: Option<&PyAny>,
        value: Option<u8>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let top_dist = parse_dist_with_default(top, 0.0)?;
        let bottom_dist = parse_dist_with_default(bottom, 0.0)?;
        let left_dist = parse_dist_with_default(left, 0.0)?;
        let right_dist = parse_dist_with_default(right, 0.0)?;
        let prob_dist = parse_p_dist(p)?;

        let pad_mode = if let Some(m) = mode {
            parse_pad_mode(m)?
        } else {
            PadMode::Reflect
        };

        let val = value.unwrap_or(match pad_mode {
            PadMode::Constant { value: v } => v,
            _ => 0,
        });

        Ok(Self {
            top: top_dist,
            bottom: bottom_dist,
            left: left_dist,
            right: right_dist,
            mode: pad_mode,
            value: val,
            p: prob_dist,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::Pad {
            top: self.top.clone(),
            bottom: self.bottom.clone(),
            left: self.left.clone(),
            right: self.right.clone(),
            mode: self.mode,
            value: self.value,
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::Pad {
            top: self.top.clone(),
            bottom: self.bottom.clone(),
            left: self.left.clone(),
            right: self.right.clone(),
            mode: self.mode,
            value: self.value,
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        let mode_str = match self.mode {
            PadMode::Constant { value } => format!("constant({})", value),
            PadMode::Reflect => "reflect".to_string(),
            PadMode::Replicate => "replicate".to_string(),
            PadMode::Wrap => "wrap".to_string(),
        };
        format!(
            "Pad(top={}, bottom={}, left={}, right={}, mode='{}', p={})",
            format_dist(&self.top),
            format_dist(&self.bottom),
            format_dist(&self.left),
            format_dist(&self.right),
            mode_str,
            format_dist(&self.p)
        )
    }
}

/// Affine - affine transformation (scale, rotate, translate, shear)
/// Affine transform: scale, rotate, translate, shear with bilinear/nearest
/// interpolation and a configurable border mode.
#[cfg(feature = "python")]
#[pyclass(name = "Affine")]
pub struct PyAffine {
    pub scale: (Dist, Dist),     // (scale_x, scale_y)
    pub rotate: Dist,            // rotation in degrees
    pub translate: (Dist, Dist), // (translate_x, translate_y) in pixels
    pub shear: (Dist, Dist),     // (shear_x, shear_y)
    pub interpolation: Interpolation,
    pub border_mode: BorderMode,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAffine {
    #[new]
    #[pyo3(signature = (scale=None, rotate=None, translate=None, shear=None, interpolation=None, border_mode=None, p=None))]
    fn new(
        scale: Option<&PyAny>,
        rotate: Option<&PyAny>,
        translate: Option<&PyAny>,
        shear: Option<&PyAny>,
        interpolation: Option<&PyAny>,
        border_mode: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let interp_enum = if let Some(i) = interpolation {
            parse_interpolation(i)?
        } else {
            Interpolation::Bilinear
        };

        let border_enum = if let Some(b) = border_mode {
            parse_border_mode(b)?
        } else {
            BorderMode::Constant { value: 0 }
        };

        let scale_dist = parse_tuple_dist(scale, 1.0)?;
        let rotate_dist = parse_dist_with_default(rotate, 0.0)?;
        let translate_dist = parse_tuple_dist(translate, 0.0)?;
        let shear_dist = parse_tuple_dist(shear, 0.0)?;
        let prob_dist = parse_p_dist(p)?;

        Ok(Self {
            scale: scale_dist,
            rotate: rotate_dist,
            translate: translate_dist,
            shear: shear_dist,
            interpolation: interp_enum,
            border_mode: border_enum,
            p: prob_dist,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::Affine {
            scale: self.scale.clone(),
            rotate: self.rotate.clone(),
            translate: self.translate.clone(),
            shear: self.shear.clone(),
            interpolation: self.interpolation,
            border_mode: self.border_mode,
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::Affine {
            scale: self.scale.clone(),
            rotate: self.rotate.clone(),
            translate: self.translate.clone(),
            shear: self.shear.clone(),
            interpolation: self.interpolation,
            border_mode: self.border_mode,
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        let border_str = match self.border_mode {
            BorderMode::Constant { value } => format!("constant({})", value),
            BorderMode::Reflect => "reflect".to_string(),
            BorderMode::Replicate => "replicate".to_string(),
            BorderMode::Wrap => "wrap".to_string(),
        };
        let interp_str = match self.interpolation {
            Interpolation::Nearest => "nearest",
            Interpolation::Bilinear => "bilinear",
            Interpolation::Bicubic => "bicubic",
            Interpolation::Lanczos4 => "lanczos4",
        };
        format!(
            "Affine(scale=({}, {}), rotate={}, translate=({}, {}), shear=({}, {}), interpolation='{}', border_mode='{}', p={})",
            format_dist(&self.scale.0),
            format_dist(&self.scale.1),
            format_dist(&self.rotate),
            format_dist(&self.translate.0),
            format_dist(&self.translate.1),
            format_dist(&self.shear.0),
            format_dist(&self.shear.1),
            interp_str,
            border_str,
            format_dist(&self.p)
        )
    }
}

/// Normalize - normalize pixel values with mean and standard deviation
#[cfg(feature = "python")]
#[pyclass(name = "Normalize")]
pub struct PyNormalize {
    pub mean: Dist,
    pub std: Dist,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyNormalize {
    #[new]
    #[pyo3(signature = (mean=None, std=None, p=None))]
    fn new(mean: Option<&PyAny>, std: Option<&PyAny>, p: Option<&PyAny>) -> PyResult<Self> {
        for (v, name) in [(mean, "mean"), (std, "std")] {
            if let Some(v) = v {
                if v.extract::<(f64, f64, f64)>().is_ok()
                    || v.extract::<(f64, f64, f64, f64)>().is_ok()
                {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Normalize.{}: per-channel values are not supported. Normalize applies a \
                         single scalar mean/std through a uint8 LUT (output stays in 0..255); pass \
                         scalars like Normalize(mean=0.5, std=0.25). True per-channel float \
                         normalization is not yet supported.",
                        name
                    )));
                }
            }
        }
        let mean_dist = parse_dist_with_default(mean, 0.0)?;
        let std_dist = parse_dist_with_default(std, 1.0)?;
        if let Dist::Constant(v) = &std_dist {
            if *v <= 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "std must be positive",
                ));
            }
        }
        let p_dist = parse_p_dist(p)?;
        Ok(Self {
            mean: mean_dist,
            std: std_dist,
            p: p_dist,
        })
    }

    #[staticmethod]
    fn standard() -> PyResult<Self> {
        Ok(Self {
            mean: Dist::constant(0.0),
            std: Dist::constant(1.0),
            p: Dist::constant(1.0),
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::Normalize {
            mean: self.mean.clone(),
            std: self.std.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::Normalize {
            mean: self.mean.clone(),
            std: self.std.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Normalize(mean={}, std={}, p={})",
            format_dist(&self.mean),
            format_dist(&self.std),
            format_dist(&self.p)
        )
    }
}

/// ColorTint - apply color tint
/// Tint the image toward a target color.
#[cfg(feature = "python")]
#[pyclass(name = "ColorTint")]
pub struct PyColorTint {
    pub tint: [Dist; 4],
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyColorTint {
    #[new]
    #[pyo3(signature = (tint=None, p=None))]
    fn new(tint: Option<&PyAny>, p: Option<&PyAny>) -> PyResult<Self> {
        let tint_val = if let Some(t) = tint {
            parse_array_dist(t, 4)?
        } else {
            [Dist::constant(0.0), Dist::constant(0.0), Dist::constant(0.0), Dist::constant(0.0)]
        };
        Ok(Self {
            tint: tint_val,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::ColorTint {
            tint: self.tint.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::ColorTint {
            tint: self.tint.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "ColorTint(tint=({}, {}, {}, {}), p={})",
            format_dist(&self.tint[0]),
            format_dist(&self.tint[1]),
            format_dist(&self.tint[2]),
            format_dist(&self.tint[3]),
            format_dist(&self.p)
        )
    }
}

/// CoarseDropout - dropout rectangular regions
/// Drop rectangular regions (set to zero).
#[cfg(feature = "python")]
#[pyclass(name = "CoarseDropout")]
pub struct PyCoarseDropout {
    pub holes: Dist,
    pub hole_size: (Dist, Dist),
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCoarseDropout {
    #[new]
    #[pyo3(signature = (holes=None, hole_size=None, p=None))]
    fn new(holes: Option<&PyAny>, hole_size: Option<&PyAny>, p: Option<&PyAny>) -> PyResult<Self> {
        let holes_dist = parse_dist_with_default(holes, 8.0)?;
        let size_dist = if let Some(hs) = hole_size {
            if let Ok(iter) = hs.iter() {
                let size_vec: Vec<_> = iter.collect::<Result<Vec<_>, _>>()?;
                if size_vec.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "hole_size must be a tuple of 2 values, got {}",
                        size_vec.len()
                    )));
                }
                let h_dist = parse_distribution(&size_vec[0])?;
                let w_dist = parse_distribution(&size_vec[1])?;
                (h_dist, w_dist)
            } else if let Ok(s) = hs.extract::<f32>() {
                (Dist::constant(s), Dist::constant(s))
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "hole_size must be a tuple of 2 values or a number",
                ));
            }
        } else {
            (Dist::constant(0.08), Dist::constant(0.08))
        };
        Ok(Self {
            holes: holes_dist,
            hole_size: size_dist,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::CoarseDropout {
            holes: self.holes.clone(),
            hole_size: self.hole_size.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::CoarseDropout {
            holes: self.holes.clone(),
            hole_size: self.hole_size.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "CoarseDropout(holes={}, hole_size=({}, {}), p={})",
            format_dist(&self.holes),
            format_dist(&self.hole_size.0),
            format_dist(&self.hole_size.1),
            format_dist(&self.p)
        )
    }
}

/// GridDropout - grid-based dropout
/// Drop grid-aligned rectangular regions.
#[cfg(feature = "python")]
#[pyclass(name = "GridDropout")]
pub struct PyGridDropout {
    pub ratio: Dist,
    pub unit_size: Dist,
    pub holes: Dist,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGridDropout {
    #[new]
    #[pyo3(signature = (ratio=None, unit_size=None, holes=None, p=None))]
    fn new(
        ratio: Option<&PyAny>,
        unit_size: Option<&PyAny>,
        holes: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let ratio_dist = parse_dist_with_default(ratio, 0.5)?;
        let unit_dist = parse_dist_with_default(unit_size, 16.0)?;
        let holes_dist = parse_dist_with_default(holes, 4.0)?;
        Ok(Self {
            ratio: ratio_dist,
            unit_size: unit_dist,
            holes: holes_dist,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::GridDropout {
            ratio: self.ratio.clone(),
            unit_size: self.unit_size.clone(),
            holes: self.holes.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::GridDropout {
            ratio: self.ratio.clone(),
            unit_size: self.unit_size.clone(),
            holes: self.holes.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "GridDropout(ratio={}, unit_size={}, holes={}, p={})",
            format_dist(&self.ratio),
            format_dist(&self.unit_size),
            format_dist(&self.holes),
            format_dist(&self.p)
        )
    }
}

/// GaussianBlurSigma - Sigma-agnostic Gaussian blur
/// Sigma-based Gaussian blur with exact or fast quality.
#[cfg(feature = "python")]
#[pyclass(name = "GaussianBlurSigma")]
pub struct PyGaussianBlurSigma {
    pub sigma: f32,
    pub quality: String, // "Exact" or "Fast"
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGaussianBlurSigma {
    #[new]
    #[pyo3(signature = (sigma=1.0, quality="Exact", p=None))]
    fn new(sigma: f32, quality: Option<&str>, p: Option<&PyAny>) -> PyResult<Self> {
        let quality_str = quality.unwrap_or("Exact");
        if !matches!(quality_str, "Exact" | "Fast") {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "quality must be 'Exact' or 'Fast', got '{}'",
                quality_str
            )));
        }

        let sigma_val = sigma.max(0.1);

        Ok(Self {
            sigma: sigma_val,
            quality: quality_str.to_string(),
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::GaussianBlurSigma { sigma: self.sigma };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::GaussianBlurSigma { sigma: self.sigma };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussianBlurSigma(sigma={}, quality='{}', p={})",
            self.sigma,
            self.quality,
            format_dist(&self.p)
        )
    }
}

/// Emboss - emboss effect (blend-based)
/// Emboss effect with a directional kernel and alpha/strength blend.
#[cfg(feature = "python")]
#[pyclass(name = "Emboss")]
pub struct PyEmboss {
    pub direction: EmbossDirection,
    pub alpha: Dist,
    pub strength: Dist,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEmboss {
    #[new]
    #[pyo3(signature = (direction=None, alpha=None, strength=None, p=None))]
    fn new(
        direction: Option<&PyAny>,
        alpha: Option<&PyAny>,
        strength: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let dir_val = if let Some(d) = direction {
            parse_emboss_direction(d)?
        } else {
            EmbossDirection::TopLeft
        };
        let alpha_dist = parse_dist_with_default(alpha, 0.5)?;
        let strength_dist = parse_dist_with_default(strength, 0.5)?;
        Ok(Self {
            direction: dir_val,
            alpha: alpha_dist,
            strength: strength_dist,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::Emboss {
            direction: self.direction,
            alpha: self.alpha.clone(),
            strength: self.strength.clone(),
        };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::Emboss {
            direction: self.direction,
            alpha: self.alpha.clone(),
            strength: self.strength.clone(),
        };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        let dir_str = match self.direction {
            EmbossDirection::TopLeft => "TOP_LEFT",
            EmbossDirection::Top => "TOP",
            EmbossDirection::TopRight => "TOP_RIGHT",
            EmbossDirection::Right => "RIGHT",
            EmbossDirection::BottomRight => "BOTTOM_RIGHT",
            EmbossDirection::Bottom => "BOTTOM",
            EmbossDirection::BottomLeft => "BOTTOM_LEFT",
            EmbossDirection::Left => "LEFT",
        };
        format!(
            "Emboss(direction='{}', alpha={}, strength={}, p={})",
            dir_str,
            format_dist(&self.alpha),
            format_dist(&self.strength),
            format_dist(&self.p)
        )
    }
}

/// EdgeDetection - edge detection with various methods
/// Edge detection (laplacian or sobel-style).
#[cfg(feature = "python")]
#[pyclass(name = "EdgeDetection")]
pub struct PyEdgeDetection {
    pub method: EdgeMethod,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEdgeDetection {
    #[new]
    #[pyo3(signature = (method=None, p=None))]
    fn new(method: Option<&PyAny>, p: Option<&PyAny>) -> PyResult<Self> {
        let method_val = if let Some(m) = method {
            parse_edge_method(m)?
        } else {
            EdgeMethod::Sobel
        };
        Ok(Self {
            method: method_val,
            p: parse_p_dist(p)?,
        })
    }

    #[pyo3(signature = (image, bboxes=None, keypoints=None, masks=None, mask=None, bbox_format="xywh", keypoint_format="xy", inplace=None))]
    fn __call__<'py>(
        &self,
        image: &'py PyAny,
        bboxes: Option<&'py PyAny>,
        keypoints: Option<&'py PyAny>,
        masks: Option<&'py PyAny>,
        mask: Option<&'py PyAny>,
        bbox_format: &str,
        keypoint_format: &str,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let node = RandomImageNode::EdgeDetection { method: self.method };
        apply_node_to_targets(node, self.p.clone(), image, bboxes, keypoints, masks, mask, bbox_format, keypoint_format, inplace, py)
    }

    #[pyo3(signature = (array, inplace=None))]
    fn apply<'py>(
        &self,
        array: &'py PyAny,
        inplace: Option<bool>,
        py: Python<'py>,
    ) -> PyResult<&'py PyAny> {
        let node = RandomImageNode::EdgeDetection { method: self.method };
        apply_node_to_image(node, self.p.clone(), array, inplace, py)
    }

    fn __repr__(&self) -> String {
        let method_str = match self.method {
            EdgeMethod::Sobel => "SOBEL",
            EdgeMethod::Prewitt => "PREWITT",
            EdgeMethod::Laplacian => "LAPLACIAN",
            EdgeMethod::Canny => "CANNY",
        };
        format!("EdgeDetection(method='{}', p={})", method_str, format_dist(&self.p))
    }
}

// ============================================================================
// Extract Node Dispatch
// ============================================================================

macro_rules! extract_manual_node {
    (
        $item:expr,
        $(
            $py_struct:ident => $node_gen:expr
        ),* $(,)?
    ) => {{
        $(
            if let Ok(obj) = $item.extract::<PyRef<$py_struct>>() {
                let node = $node_gen(&obj);
                return Ok(maybe_wrap(node, obj.p.clone()));
            }
        )*
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "transform must be a sinter transform type, got {}",
            $item.get_type().name()?
        )))
    }};
}

/// Extract a RandomImageNode from a Python object
#[cfg(feature = "python")]
pub(crate) fn extract_node(item: &PyAny) -> PyResult<RandomImageNode> {
    // 1. Try basic transforms (generated by macro)
    if let Some(node) = extract_basic_node(item)? {
        return Ok(node);
    }

    // 2. Try discrete transforms (generated by macro)
    if let Some(node) = extract_discrete_node(item)? {
        return Ok(node);
    }

    // 3. Try manual transforms using macro dispatch
    extract_manual_node!(
        item,
        PyRotate => |obj: &PyRotate| RandomImageNode::Rotate { angle: obj.angle },
        PyResize => |obj: &PyResize| RandomImageNode::Resize {
            width: obj.width,
            height: obj.height,
            interpolation: obj.interpolation,
        },
        PyPad => |obj: &PyPad| RandomImageNode::Pad {
            top: obj.top.clone(),
            bottom: obj.bottom.clone(),
            left: obj.left.clone(),
            right: obj.right.clone(),
            mode: obj.mode,
            value: obj.value,
        },
        PyAffine => |obj: &PyAffine| RandomImageNode::Affine {
            scale: obj.scale.clone(),
            rotate: obj.rotate.clone(),
            translate: obj.translate.clone(),
            shear: obj.shear.clone(),
            interpolation: obj.interpolation,
            border_mode: obj.border_mode,
        },
        PyNormalize => |obj: &PyNormalize| RandomImageNode::Normalize {
            mean: obj.mean.clone(),
            std: obj.std.clone(),
        },
        PyColorTint => |obj: &PyColorTint| RandomImageNode::ColorTint {
            tint: obj.tint.clone(),
        },
        PyChannelShuffle => |obj: &PyChannelShuffle| {
            let permutations = [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ];
            let idx = (obj.order as usize) % 6;
            RandomImageNode::ChannelShuffle { order: permutations[idx] }
        },
        PyCoarseDropout => |obj: &PyCoarseDropout| RandomImageNode::CoarseDropout {
            holes: obj.holes.clone(),
            hole_size: obj.hole_size.clone(),
        },
        PyGridDropout => |obj: &PyGridDropout| RandomImageNode::GridDropout {
            ratio: obj.ratio.clone(),
            unit_size: obj.unit_size.clone(),
            holes: obj.holes.clone(),
        },
        PyGaussianBlurSigma => |obj: &PyGaussianBlurSigma| RandomImageNode::GaussianBlurSigma {
            sigma: obj.sigma,
        },
        PyEmboss => |obj: &PyEmboss| RandomImageNode::Emboss {
            direction: obj.direction,
            alpha: obj.alpha.clone(),
            strength: obj.strength.clone(),
        },
        PyHueSaturationValue => |obj: &PyHueSaturationValue| RandomImageNode::HueSaturationValue {
            hue_shift: obj.hue_shift.clone(),
            saturation_scale: obj.saturation_scale.clone(),
            value_scale: obj.value_scale.clone(),
        },
        PyRGBShift => |obj: &PyRGBShift| RandomImageNode::RGBShift {
            r_shift: obj.r_shift.clone(),
            g_shift: obj.g_shift.clone(),
            b_shift: obj.b_shift.clone(),
        },
        PyGaussNoise => |obj: &PyGaussNoise| RandomImageNode::GaussNoise {
            mean: obj.mean.clone(),
            std: obj.std.clone(),
        },
        PyCrop => |obj: &PyCrop| RandomImageNode::Crop {
            x: obj.x.clone(),
            y: obj.y.clone(),
            width: obj.width.clone(),
            height: obj.height.clone(),
        },
        PyEdgeDetection => |obj: &PyEdgeDetection| RandomImageNode::EdgeDetection {
            method: obj.method,
        },
    )
}
