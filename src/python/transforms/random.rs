// Python wrappers for transforms with distribution support
//
// This module provides the **unified API** for sinter.
// All transforms accept either plain values (constant) or distributions.

use crate::core::FusableImage;
use crate::exec_ir::Optimizer;
use crate::sampled_ir::ops::{EdgeMethod, EmbossDirection, Interpolation, PadMode, RotateAngle};
use crate::sampled_ir::Plan;
use crate::sampling::{Dist, RandomImageNode, RandomImageProgram};

use super::super::distributions::parse_distribution;
use super::super::sampled::PySampledImageProgram;
use super::helpers::maybe_wrap;

#[cfg(feature = "python")]
use numpy::{PyArray2, PyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

// ============================================================================
// Macros to reduce boilerplate
// ============================================================================

/// Parse a tuple distribution (x, y) from Python
///
/// Accepts:
/// - None → (default, default)
/// - Single value → (value, value)
/// - Tuple (x, y) → (x, y)
/// - Distribution → (dist.clone(), dist)
#[cfg(feature = "python")]
fn parse_tuple_dist(val: Option<&PyAny>, default: f32) -> PyResult<(Dist, Dist)> {
    if let Some(v) = val {
        // Try tuple first
        if let Ok((x, y)) = v.extract::<(f32, f32)>() {
            return Ok((Dist::constant(x), Dist::constant(y)));
        }
        // Try single value
        if let Ok(s) = v.extract::<f32>() {
            return Ok((Dist::constant(s), Dist::constant(s)));
        }
        // Parse as distribution
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

/// Format a probability distribution value for __repr__
///
/// Returns:
/// - "0.5" for Constant(0.5)
/// - "Bernoulli(0.3)" for Bernoulli { p: 0.3 }
/// - "<dist>" for other distributions
#[cfg(feature = "python")]
fn format_p_value(dist: &Dist) -> String {
    match dist {
        Dist::Constant(v) => format!("{}", v),
        Dist::Bernoulli { p } => format!("Bernoulli({})", p),
        _ => "<dist>".to_string(),
    }
}

/// Parse an array of distributions from Python
///
/// Accepts:
/// - A list/tuple of values → each parsed as a distribution
/// - A single value → applied to all array elements
///
/// # Example
/// ```ignore
/// let tint = parse_array_dist(value, 4)?;  // Returns [Dist; 4]
/// ```
#[cfg(feature = "python")]
fn parse_array_dist(value: &PyAny, len: usize) -> PyResult<[Dist; 4]> {
    // Try to iterate (list/tuple)
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

    // Single value applied to all elements
    let dist = parse_distribution(value)?;
    Ok([dist.clone(), dist.clone(), dist.clone(), dist])
}

macro_rules! define_basic_transforms {
    (
        $( $name:ident ( $py_struct:ident, $py_name:literal ) { $( $field:ident ),* } => $node_gen:expr ),*
    ) => {
        $(
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
                #[pyo3(signature = ( $( $field, )* p=None ))]
                fn new( $( $field: &PyAny, )* p: Option<&PyAny> ) -> PyResult<Self> {
                    Ok(Self {
                        $( $field: parse_distribution($field)?, )*
                        p: parse_p_dist(p)?,
                    })
                }

                fn __repr__(&self) -> String {
                    let mut s = String::from($py_name);
                    s.push('(');
                    $(
                        s.push_str(concat!(stringify!($field), "=<dist>, "));
                    )*
                    s.push_str("p=<dist>)");
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
    HorizontalFlip(PyHorizontalFlip, "HorizontalFlip") {} => |_| RandomImageNode::HorizontalFlip,
    VerticalFlip(PyVerticalFlip, "VerticalFlip") {} => |_| RandomImageNode::VerticalFlip,
    Transpose(PyTranspose, "Transpose") {} => |_| RandomImageNode::Transpose,
    Invert(PyInvert, "Invert") {} => |_| RandomImageNode::Invert,
    ToGray(PyToGray, "ToGray") {} => |_| RandomImageNode::ToGray,
    ToSepia(PyToSepia, "ToSepia") {} => |_| RandomImageNode::ToSepia,
    ToRGB(PyToRGB, "ToRGB") {} => |_| RandomImageNode::ToRGB,
    Equalize(PyEqualize, "Equalize") {} => |_| RandomImageNode::Equalize,
    AutoContrast(PyAutoContrast, "AutoContrast") {} => |_| RandomImageNode::AutoContrast,

    Brightness(PyBrightness, "Brightness") { delta } => |obj: &PyBrightness| RandomImageNode::Brightness { delta: obj.delta.clone() },
    Contrast(PyContrast, "Contrast") { factor } => |obj: &PyContrast| RandomImageNode::Contrast { factor: obj.factor.clone() },
    Posterize(PyPosterize, "Posterize") { bits } => |obj: &PyPosterize| RandomImageNode::Posterize { bits: obj.bits.clone() },
    Solarize(PySolarize, "Solarize") { threshold } => |obj: &PySolarize| RandomImageNode::Solarize { threshold: obj.threshold.clone() },
    Gamma(PyGamma, "Gamma") { gamma } => |obj: &PyGamma| RandomImageNode::Gamma { gamma: obj.gamma.clone() },
    Sharpen(PySharpen, "Sharpen") { strength } => |obj: &PySharpen| RandomImageNode::Sharpen { strength: obj.strength.clone() },

    GaussNoise(PyGaussNoise, "GaussNoise") { mean, std } => |obj: &PyGaussNoise| RandomImageNode::GaussNoise { mean: obj.mean.clone(), std: obj.std.clone() },
    MultiplicativeNoise(PyMultiplicativeNoise, "MultiplicativeNoise") { multiplier } => |obj: &PyMultiplicativeNoise| RandomImageNode::MultiplicativeNoise { multiplier: obj.multiplier.clone() },
    SaltAndPepper(PySaltAndPepper, "SaltAndPepper") { amount, salt_vs_pepper } => |obj: &PySaltAndPepper| RandomImageNode::SaltAndPepper { amount: obj.amount.clone(), salt_vs_pepper: obj.salt_vs_pepper.clone() },

    RGBShift(PyRGBShift, "RGBShift") { r_shift, g_shift, b_shift } => |obj: &PyRGBShift| RandomImageNode::RGBShift { r_shift: obj.r_shift.clone(), g_shift: obj.g_shift.clone(), b_shift: obj.b_shift.clone() },
    HueSaturationValue(PyHueSaturationValue, "HueSaturationValue") { hue_shift, saturation_scale, value_scale } => |obj: &PyHueSaturationValue| RandomImageNode::HueSaturationValue { hue_shift: obj.hue_shift.clone(), saturation_scale: obj.saturation_scale.clone(), value_scale: obj.value_scale.clone() },

    ColorTemperature(PyColorTemperature, "ColorTemperature") { temperature } => |obj: &PyColorTemperature| RandomImageNode::ColorTemperature { temperature: obj.temperature.clone() },
    ColorBalance(PyColorBalance, "ColorBalance") { r_scale, g_scale, b_scale } => |obj: &PyColorBalance| RandomImageNode::ColorBalance { r_scale: obj.r_scale.clone(), g_scale: obj.g_scale.clone(), b_scale: obj.b_scale.clone() },

    Crop(PyCrop, "Crop") { x, y, width, height } => |obj: &PyCrop| RandomImageNode::Crop { x: obj.x.clone(), y: obj.y.clone(), width: obj.width.clone(), height: obj.height.clone() }
}

// ============================================================================
// Discrete Parameter Transforms Macro
// ============================================================================

macro_rules! define_discrete_transforms {
    (
        $( $name:ident ( $py_struct:ident, $py_name:literal ) { $field:ident : $type:ty $(= $default:expr)?, values: $valid_values:expr } => $node_gen:expr ),*
    ) => {
        $(
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
                #[pyo3(signature = ( $field $(= $default)?, p=None ))]
                fn new( $field: $type, p: Option<&PyAny> ) -> PyResult<Self> {
                    let valid = $valid_values;
                    if !valid.contains(&$field) {
                         return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                             "{} must be one of {:?}, got {}",
                             stringify!($field), valid, $field
                         )));
                    }
                    Ok(Self {
                        $field,
                        p: parse_p_dist(p)?,
                    })
                }

                fn __repr__(&self) -> String {
                    format!(concat!($py_name, "(", stringify!($field), "={}, p=<dist>)"), self.$field)
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
    GaussianBlur(PyGaussianBlur, "GaussianBlur") { kernel_size: u32 = 3, values: [3, 5, 7, 13, 21, 31] } => |obj: &PyGaussianBlur| RandomImageNode::GaussianBlur { kernel_size: obj.kernel_size },
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
    #[pyo3(signature = (order, p=None))]
    fn new(order: u8, p: Option<&PyAny>) -> PyResult<Self> {
        if order > 5 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "order must be 0-5, got {}",
                order
            )));
        }
        Ok(Self {
            order,
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

    fn __repr__(&self) -> String {
        let names = ["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"];
        format!(
            "ChannelShuffle(order={}, p=<dist>)",
            names[self.order as usize]
        )
    }
}

// ============================================================================
// Manual Transform Classes
// ============================================================================

/// Rotate - rotate by 90, 180, or 270 degrees
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
    #[pyo3(signature = (angle, p=None))]
    fn new(angle: crate::python::enums::PyRotateAngle, p: Option<&PyAny>) -> PyResult<Self> {
        Ok(Self {
            angle: angle.inner,
            p: parse_p_dist(p)?,
        })
    }

    fn __repr__(&self) -> String {
        let angle_str = match self.angle {
            RotateAngle::Rotate90 => "90°",
            RotateAngle::Rotate180 => "180°",
            RotateAngle::Rotate270 => "270°",
        };
        format!("Rotate(angle={}, p={})", angle_str, format_p_value(&self.p))
    }
}

/// Resize - resize to specific dimensions
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
        interpolation: Option<crate::python::enums::PyInterpolation>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "width and height must be positive",
            ));
        }
        Ok(Self {
            width,
            height,
            interpolation: interpolation
                .map(|i| i.inner)
                .unwrap_or(Interpolation::Bilinear),
            p: parse_p_dist(p)?,
        })
    }

    fn __repr__(&self) -> String {
        let interp_str = match self.interpolation {
            Interpolation::Nearest => "nearest",
            Interpolation::Bilinear => "bilinear",
            Interpolation::Bicubic => "bicubic",
            Interpolation::Lanczos4 => "lanczos4",
        };
        format!(
            "Resize(width={}, height={}, interpolation={}, p={})",
            self.width,
            self.height,
            interp_str,
            format_p_value(&self.p)
        )
    }
}

/// Pad - pad image
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
    #[pyo3(signature = (top, bottom, left, right, mode, p=None))]
    fn new(
        top: &PyAny,
        bottom: &PyAny,
        left: &PyAny,
        right: &PyAny,
        mode: crate::python::enums::PyPadMode,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let top_dist = parse_distribution(top)?;
        let bottom_dist = parse_distribution(bottom)?;
        let left_dist = parse_distribution(left)?;
        let right_dist = parse_distribution(right)?;
        let prob_dist = parse_p_dist(p)?;

        // Extract value from mode if it's Constant
        let value = match mode.inner {
            PadMode::Constant { value } => value,
            _ => 0,
        };

        Ok(Self {
            top: top_dist,
            bottom: bottom_dist,
            left: left_dist,
            right: right_dist,
            mode: mode.inner,
            value,
            p: prob_dist,
        })
    }

    fn __repr__(&self) -> String {
        let mode_str = match self.mode {
            PadMode::Constant { value } => format!("constant(value={})", value),
            PadMode::Reflect => "reflect".to_string(),
            PadMode::Replicate => "replicate".to_string(),
            PadMode::Wrap => "wrap".to_string(),
        };
        format!(
            "Pad(top=<dist>, bottom=<dist>, left=<dist>, right=<dist>, mode={}, p=<dist>)",
            mode_str
        )
    }
}

/// Affine - affine transformation (scale, rotate, translate, shear)
#[cfg(feature = "python")]
#[pyclass(name = "Affine")]
pub struct PyAffine {
    pub scale: (Dist, Dist),     // (scale_x, scale_y)
    pub rotate: Dist,            // rotation in degrees
    pub translate: (Dist, Dist), // (translate_x, translate_y) in pixels
    pub shear: (Dist, Dist),     // (shear_x, shear_y)
    pub interpolation: Interpolation,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAffine {
    #[new]
    #[pyo3(signature = (scale=None, rotate=None, translate=None, shear=None, interpolation=None, p=None))]
    fn new(
        scale: Option<&PyAny>,
        rotate: Option<&PyAny>,
        translate: Option<&PyAny>,
        shear: Option<&PyAny>,
        interpolation: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        // Parse interpolation - default to Bilinear
        let interp_enum = if let Some(interp_val) = interpolation {
            interp_val
                .extract::<crate::python::enums::PyInterpolation>()?
                .inner
        } else {
            Interpolation::Bilinear
        };

        // Parse parameters using helper functions
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
            p: prob_dist,
        })
    }

    fn __repr__(&self) -> String {
        let interp_str = match self.interpolation {
            Interpolation::Nearest => "nearest",
            Interpolation::Bilinear => "bilinear",
            Interpolation::Bicubic => "bicubic",
            Interpolation::Lanczos4 => "lanczos4",
        };
        format!("Affine(scale=<dist>, rotate=<dist>, translate=<dist>, shear=<dist>, interpolation={}, p=<dist>)", interp_str)
    }
}

/// Normalize - normalize with mean/std
#[cfg(feature = "python")]
#[pyclass(name = "Normalize")]
pub struct PyNormalize {
    pub mean: Dist,
    pub std: Dist,
    pub p: Dist,
}

#[cfg(feature = "python")]
#[cfg(feature = "python")]
#[pymethods]
impl PyNormalize {
    #[new]
    #[pyo3(signature = (mean, std, p=None))]
    fn new(mean: &PyAny, std: &PyAny, p: Option<&PyAny>) -> PyResult<Self> {
        let mean_dist = parse_distribution(mean)?;
        let std_dist = parse_distribution(std)?;
        if let Dist::Constant(v) = &std_dist {
            if *v == 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "std must be non-zero",
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

    fn __repr__(&self) -> String {
        format!("Normalize(mean=<dist>, std=<dist>, p=<dist>)")
    }
}

/// ColorTint - apply color tint
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
    #[pyo3(signature = (tint, p=None))]
    fn new(tint: &PyAny, p: Option<&PyAny>) -> PyResult<Self> {
        Ok(Self {
            tint: parse_array_dist(tint, 4)?,
            p: parse_p_dist(p)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("ColorTint(tint=<dist>, p=<dist>)")
    }
}

/// CoarseDropout - dropout rectangular regions
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
    #[pyo3(signature = (holes, hole_size, p=None))]
    fn new(holes: &PyAny, hole_size: &PyAny, p: Option<&PyAny>) -> PyResult<Self> {
        let holes_dist = parse_distribution(holes)?;
        let size_dist = if let Ok(iter) = hole_size.iter() {
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
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "hole_size must be a tuple of 2 values",
            ));
        };
        Ok(Self {
            holes: holes_dist,
            hole_size: size_dist,
            p: parse_p_dist(p)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("CoarseDropout(holes=<dist>, hole_size=<dist>, p=<dist>)")
    }
}

/// GridDropout - grid-based dropout
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
    #[pyo3(signature = (ratio, unit_size, holes, p=None))]
    fn new(ratio: &PyAny, unit_size: &PyAny, holes: &PyAny, p: Option<&PyAny>) -> PyResult<Self> {
        let ratio_dist = parse_distribution(ratio)?;
        let unit_dist = parse_distribution(unit_size)?;
        let holes_dist = parse_distribution(holes)?;
        Ok(Self {
            ratio: ratio_dist,
            unit_size: unit_dist,
            holes: holes_dist,
            p: parse_p_dist(p)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("GridDropout(ratio=<dist>, unit_size=<dist>, holes=<dist>, p=<dist>)")
    }
}

/// GaussianBlurSigma - Sigma-agnostic Gaussian blur (NEW)
/// Uses exact Gaussian kernels generated from sigma
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

    fn __repr__(&self) -> String {
        format!(
            "GaussianBlurSigma(sigma={}, quality={}, p=<dist>)",
            self.sigma, self.quality
        )
    }
}

/// Emboss - emboss effect (blend-based, compatible with albumentations)
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
    #[pyo3(signature = (direction, alpha=None, strength=None, p=None))]
    fn new(
        direction: crate::python::enums::PyEmbossDirection,
        alpha: Option<&PyAny>,
        strength: Option<&PyAny>,
        p: Option<&PyAny>,
    ) -> PyResult<Self> {
        let alpha_dist = parse_dist_with_default(alpha, 0.5)?;
        let strength_dist = parse_dist_with_default(strength, 0.5)?;
        Ok(Self {
            direction: direction.inner,
            alpha: alpha_dist,
            strength: strength_dist,
            p: parse_p_dist(p)?,
        })
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
            "Emboss(direction={}, alpha=<dist>, strength=<dist>, p=<dist>)",
            dir_str
        )
    }
}

/// EdgeDetection - edge detection with various methods
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
    #[pyo3(signature = (method, p=None))]
    fn new(method: crate::python::enums::PyEdgeMethod, p: Option<&PyAny>) -> PyResult<Self> {
        Ok(Self {
            method: method.inner,
            p: parse_p_dist(p)?,
        })
    }

    fn __repr__(&self) -> String {
        let method_str = match self.method {
            EdgeMethod::Sobel => "SOBEL",
            EdgeMethod::Prewitt => "PREWITT",
            EdgeMethod::Laplacian => "LAPLACIAN",
            EdgeMethod::Canny => "CANNY",
        };
        format!("EdgeDetection(method={}, p=<dist>)", method_str)
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
        PyEdgeDetection => |obj: &PyEdgeDetection| RandomImageNode::EdgeDetection {
            method: obj.method,
        },
    )
}
