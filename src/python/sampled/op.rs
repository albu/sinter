// PySampledImageOp wrapper and constructors

use crate::sampled_ir::ops::{Interpolation, RotateAngle};
use crate::sampled_ir::SampledImageOp;

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ============================================================================
// Individual op constructors (exposed to Python via ops submodule)
// ============================================================================

/// Create a Brightness operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn brightness(delta: f32) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Brightness { delta },
    })
}

/// Create a Contrast operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn contrast(factor: f32) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Contrast { factor },
    })
}

/// Create a Gamma operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn gamma(gamma: f32) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Gamma { gamma },
    })
}

/// Create a Saturation operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn saturation(factor: f32) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Saturation { factor },
    })
}

/// Create an Invert operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn invert() -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Invert,
    })
}

/// Create a ToGray operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn to_gray() -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::ToGray,
    })
}

/// Create a ToSepia operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn to_sepia() -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::ToSepia,
    })
}

/// Create a Posterize operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn posterize(bits: u8) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Posterize { bits },
    })
}

/// Create a Solarize operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn solarize(threshold: u8) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Solarize { threshold },
    })
}

/// Create an Equalize operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn equalize() -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Equalize,
    })
}

/// Create a HueSaturationValue operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn hue_saturation_value(
    hue_shift: i32,
    saturation_scale: f32,
    value_scale: f32,
) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::HueSaturationValue {
            hue_shift,
            saturation_scale,
            value_scale,
        },
    })
}

/// Create an RGBShift operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn rgb_shift(r_shift: i32, g_shift: i32, b_shift: i32) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::RGBShift {
            r_shift,
            g_shift,
            b_shift,
        },
    })
}

/// Create a Normalize operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn normalize(mean: [f32; 3], std: [f32; 3]) -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Normalize { mean, std },
    })
}

/// Create a HorizontalFlip operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn horizontal_flip() -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::HorizontalFlip,
    })
}

/// Create a VerticalFlip operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn vertical_flip() -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::VerticalFlip,
    })
}

/// Create a Transpose operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn transpose() -> PyResult<PySampledImageOp> {
    Ok(PySampledImageOp {
        inner: SampledImageOp::Transpose,
    })
}

/// Create a Rotate operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn rotate(angle: i32) -> PyResult<PySampledImageOp> {
    let rotate_angle = match angle {
        90 => RotateAngle::Rotate90,
        180 => RotateAngle::Rotate180,
        270 => RotateAngle::Rotate270,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "angle must be 90, 180, or 270, got {}",
                angle
            )))
        }
    };
    Ok(PySampledImageOp {
        inner: SampledImageOp::Rotate {
            angle: rotate_angle,
        },
    })
}

/// Create a Resize operation
#[cfg(feature = "python")]
#[pyfunction]
pub fn resize(
    width: u32,
    height: u32,
    interpolation: Option<String>,
) -> PyResult<PySampledImageOp> {
    let interp = match interpolation.as_deref() {
        Some("bilinear") | None => Interpolation::Bilinear,
        Some("nearest") => Interpolation::Nearest,
        Some("bicubic") => Interpolation::Bicubic,
        Some(other) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown interpolation: {}, must be: bilinear, nearest, bicubic",
                other
            )))
        }
    };
    Ok(PySampledImageOp {
        inner: SampledImageOp::Resize {
            width,
            height,
            interpolation: interp,
        },
    })
}

// ============================================================================
// PySampledImageOp wrapper
// ============================================================================

/// Wrapper for SampledImageOp (internal)
#[cfg(feature = "python")]
#[pyclass(name = "_SampledImageOp")]
pub struct PySampledImageOp {
    pub inner: SampledImageOp,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySampledImageOp {
    fn __repr__(&self) -> String {
        format!("_SampledImageOp({})", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Get the name of this operation
    fn name(&self) -> String {
        self.inner.name().to_string()
    }
}
