// Python enums for transform parameters
//
// Exposes Rust enums to Python for type-safe transform construction.

#[cfg(feature = "python")]
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::sampled_ir::ops::{EdgeMethod, EmbossDirection, Interpolation, PadMode, RotateAngle};

// =============================================================================
// RotateAngle
// =============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "RotateAngle")]
#[derive(Debug, Clone, Copy)]
pub struct PyRotateAngle {
    pub inner: RotateAngle,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRotateAngle {
    #[classattr]
    fn ROTATE_90() -> PyRotateAngle {
        PyRotateAngle {
            inner: RotateAngle::Rotate90,
        }
    }

    #[classattr]
    fn ROTATE_180() -> PyRotateAngle {
        PyRotateAngle {
            inner: RotateAngle::Rotate180,
        }
    }

    #[classattr]
    fn ROTATE_270() -> PyRotateAngle {
        PyRotateAngle {
            inner: RotateAngle::Rotate270,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            RotateAngle::Rotate90 => "RotateAngle.ROTATE_90",
            RotateAngle::Rotate180 => "RotateAngle.ROTATE_180",
            RotateAngle::Rotate270 => "RotateAngle.ROTATE_270",
        }
        .to_string()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// =============================================================================
// Interpolation
// =============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "Interpolation")]
#[derive(Debug, Clone, Copy)]
pub struct PyInterpolation {
    pub inner: Interpolation,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyInterpolation {
    #[classattr]
    fn NEAREST() -> PyInterpolation {
        PyInterpolation {
            inner: Interpolation::Nearest,
        }
    }

    #[classattr]
    fn BILINEAR() -> PyInterpolation {
        PyInterpolation {
            inner: Interpolation::Bilinear,
        }
    }

    #[classattr]
    fn BICUBIC() -> PyInterpolation {
        PyInterpolation {
            inner: Interpolation::Bicubic,
        }
    }

    #[classattr]
    fn LANCZOS4() -> PyInterpolation {
        PyInterpolation {
            inner: Interpolation::Lanczos4,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Interpolation::Nearest => "Interpolation.NEAREST",
            Interpolation::Bilinear => "Interpolation.BILINEAR",
            Interpolation::Bicubic => "Interpolation.BICUBIC",
            Interpolation::Lanczos4 => "Interpolation.LANCZOS4",
        }
        .to_string()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// =============================================================================
// PadMode
// =============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "PadMode")]
#[derive(Debug, Clone, Copy)]
pub struct PyPadMode {
    pub inner: PadMode,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPadMode {
    #[classattr]
    fn REFLECT() -> PyPadMode {
        PyPadMode {
            inner: PadMode::Reflect,
        }
    }

    #[classattr]
    fn REPLICATE() -> PyPadMode {
        PyPadMode {
            inner: PadMode::Replicate,
        }
    }

    #[classattr]
    fn WRAP() -> PyPadMode {
        PyPadMode {
            inner: PadMode::Wrap,
        }
    }

    #[staticmethod]
    fn constant(value: u8) -> PyPadMode {
        PyPadMode {
            inner: PadMode::Constant { value },
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            PadMode::Constant { value } => format!("PadMode.constant({})", value),
            PadMode::Reflect => "PadMode.REFLECT".to_string(),
            PadMode::Replicate => "PadMode.REPLICATE".to_string(),
            PadMode::Wrap => "PadMode.WRAP".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// =============================================================================
// EmbossDirection
// =============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "EmbossDirection")]
#[derive(Debug, Clone, Copy)]
pub struct PyEmbossDirection {
    pub inner: EmbossDirection,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEmbossDirection {
    #[classattr]
    fn TOP_LEFT() -> PyEmbossDirection {
        PyEmbossDirection {
            inner: EmbossDirection::TopLeft,
        }
    }

    #[classattr]
    fn TOP() -> PyEmbossDirection {
        PyEmbossDirection {
            inner: EmbossDirection::Top,
        }
    }

    #[classattr]
    fn TOP_RIGHT() -> PyEmbossDirection {
        PyEmbossDirection {
            inner: EmbossDirection::TopRight,
        }
    }

    #[classattr]
    fn RIGHT() -> PyEmbossDirection {
        PyEmbossDirection {
            inner: EmbossDirection::Right,
        }
    }

    #[classattr]
    fn BOTTOM_RIGHT() -> PyEmbossDirection {
        PyEmbossDirection {
            inner: EmbossDirection::BottomRight,
        }
    }

    #[classattr]
    fn BOTTOM() -> PyEmbossDirection {
        PyEmbossDirection {
            inner: EmbossDirection::Bottom,
        }
    }

    #[classattr]
    fn BOTTOM_LEFT() -> PyEmbossDirection {
        PyEmbossDirection {
            inner: EmbossDirection::BottomLeft,
        }
    }

    #[classattr]
    fn LEFT() -> PyEmbossDirection {
        PyEmbossDirection {
            inner: EmbossDirection::Left,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            EmbossDirection::TopLeft => "EmbossDirection.TOP_LEFT",
            EmbossDirection::Top => "EmbossDirection.TOP",
            EmbossDirection::TopRight => "EmbossDirection.TOP_RIGHT",
            EmbossDirection::Right => "EmbossDirection.RIGHT",
            EmbossDirection::BottomRight => "EmbossDirection.BOTTOM_RIGHT",
            EmbossDirection::Bottom => "EmbossDirection.BOTTOM",
            EmbossDirection::BottomLeft => "EmbossDirection.BOTTOM_LEFT",
            EmbossDirection::Left => "EmbossDirection.LEFT",
        }
        .to_string()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// =============================================================================
// EdgeMethod
// =============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "EdgeMethod")]
#[derive(Debug, Clone, Copy)]
pub struct PyEdgeMethod {
    pub inner: EdgeMethod,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEdgeMethod {
    #[classattr]
    fn SOBEL() -> PyEdgeMethod {
        PyEdgeMethod {
            inner: EdgeMethod::Sobel,
        }
    }

    #[classattr]
    fn PREWITT() -> PyEdgeMethod {
        PyEdgeMethod {
            inner: EdgeMethod::Prewitt,
        }
    }

    #[classattr]
    fn LAPLACIAN() -> PyEdgeMethod {
        PyEdgeMethod {
            inner: EdgeMethod::Laplacian,
        }
    }

    #[classattr]
    fn CANNY() -> PyEdgeMethod {
        PyEdgeMethod {
            inner: EdgeMethod::Canny,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            EdgeMethod::Sobel => "EdgeMethod.SOBEL",
            EdgeMethod::Prewitt => "EdgeMethod.PREWITT",
            EdgeMethod::Laplacian => "EdgeMethod.LAPLACIAN",
            EdgeMethod::Canny => "EdgeMethod.CANNY",
        }
        .to_string()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// =============================================================================
// Flexible Enum Parsers (accept PyClass, string, or int)
// =============================================================================

/// Parse RotateAngle from PyRotateAngle, int, or string
pub fn parse_rotate_angle(val: &PyAny) -> PyResult<RotateAngle> {
    if let Ok(obj) = val.extract::<PyRef<PyRotateAngle>>() {
        return Ok(obj.inner);
    }
    if let Ok(num) = val.extract::<i32>() {
        match num {
            90 => return Ok(RotateAngle::Rotate90),
            180 => return Ok(RotateAngle::Rotate180),
            270 => return Ok(RotateAngle::Rotate270),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Rotate angle must be 90, 180, or 270 degrees, got {}",
                    num
                )))
            }
        }
    }
    if let Ok(s) = val.extract::<&str>() {
        let clean = s.trim().to_lowercase();
        match clean.as_str() {
            "90" | "rotate90" | "rotate_90" => return Ok(RotateAngle::Rotate90),
            "180" | "rotate180" | "rotate_180" => return Ok(RotateAngle::Rotate180),
            "270" | "rotate270" | "rotate_270" => return Ok(RotateAngle::Rotate270),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown rotate angle '{}'. Expected: 90, 180, 270",
                    s
                )))
            }
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected RotateAngle, int (90, 180, 270), or str ('90', '180', '270'), got {}",
        val.get_type().name()?
    )))
}

/// Parse Interpolation from PyInterpolation or string
pub fn parse_interpolation(val: &PyAny) -> PyResult<Interpolation> {
    if let Ok(obj) = val.extract::<PyRef<PyInterpolation>>() {
        return Ok(obj.inner);
    }
    if let Ok(s) = val.extract::<&str>() {
        let clean = s.trim().to_lowercase();
        match clean.as_str() {
            "nearest" => return Ok(Interpolation::Nearest),
            "bilinear" | "linear" => return Ok(Interpolation::Bilinear),
            "bicubic" | "cubic" => return Ok(Interpolation::Bicubic),
            "lanczos4" | "lanczos" => return Ok(Interpolation::Lanczos4),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown interpolation mode '{}'. Expected: 'nearest', 'bilinear', 'bicubic', 'lanczos4'",
                    s
                )))
            }
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected Interpolation or str ('nearest', 'bilinear', 'bicubic', 'lanczos4'), got {}",
        val.get_type().name()?
    )))
}

/// Parse PadMode from PyPadMode, string, or (string, value)
pub fn parse_pad_mode(val: &PyAny) -> PyResult<PadMode> {
    if let Ok(obj) = val.extract::<PyRef<PyPadMode>>() {
        return Ok(obj.inner);
    }
    if let Ok(s) = val.extract::<&str>() {
        let clean = s.trim().to_lowercase();
        match clean.as_str() {
            "reflect" => return Ok(PadMode::Reflect),
            "replicate" | "edge" => return Ok(PadMode::Replicate),
            "wrap" => return Ok(PadMode::Wrap),
            "constant" => return Ok(PadMode::Constant { value: 0 }),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown pad mode '{}'. Expected: 'reflect', 'replicate', 'wrap', 'constant'",
                    s
                )))
            }
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected PadMode or str ('reflect', 'replicate', 'wrap', 'constant'), got {}",
        val.get_type().name()?
    )))
}

/// Parse EmbossDirection from PyEmbossDirection or string
pub fn parse_emboss_direction(val: &PyAny) -> PyResult<EmbossDirection> {
    if let Ok(obj) = val.extract::<PyRef<PyEmbossDirection>>() {
        return Ok(obj.inner);
    }
    if let Ok(s) = val.extract::<&str>() {
        let clean = s.trim().to_lowercase().replace('-', "_");
        match clean.as_str() {
            "top_left" | "topleft" => return Ok(EmbossDirection::TopLeft),
            "top" => return Ok(EmbossDirection::Top),
            "top_right" | "topright" => return Ok(EmbossDirection::TopRight),
            "right" => return Ok(EmbossDirection::Right),
            "bottom_right" | "bottomright" => return Ok(EmbossDirection::BottomRight),
            "bottom" => return Ok(EmbossDirection::Bottom),
            "bottom_left" | "bottomleft" => return Ok(EmbossDirection::BottomLeft),
            "left" => return Ok(EmbossDirection::Left),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown emboss direction '{}'",
                    s
                )))
            }
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected EmbossDirection or str, got {}",
        val.get_type().name()?
    )))
}

/// Parse EdgeMethod from PyEdgeMethod or string
pub fn parse_edge_method(val: &PyAny) -> PyResult<EdgeMethod> {
    if let Ok(obj) = val.extract::<PyRef<PyEdgeMethod>>() {
        return Ok(obj.inner);
    }
    if let Ok(s) = val.extract::<&str>() {
        let clean = s.trim().to_lowercase();
        match clean.as_str() {
            "sobel" => return Ok(EdgeMethod::Sobel),
            "prewitt" => return Ok(EdgeMethod::Prewitt),
            "laplacian" => return Ok(EdgeMethod::Laplacian),
            "canny" => return Ok(EdgeMethod::Canny),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown edge method '{}'. Expected: 'sobel', 'prewitt', 'laplacian', 'canny'",
                    s
                )))
            }
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected EdgeMethod or str ('sobel', 'prewitt', 'laplacian', 'canny'), got {}",
        val.get_type().name()?
    )))
}

/// Parse BorderMode from PyPadMode, string, or int
pub fn parse_border_mode(val: &PyAny) -> PyResult<crate::sampled_ir::ops::BorderMode> {
    use crate::sampled_ir::ops::BorderMode;
    if let Ok(obj) = val.extract::<PyRef<PyPadMode>>() {
        return Ok(match obj.inner {
            PadMode::Constant { value } => BorderMode::Constant { value },
            PadMode::Reflect => BorderMode::Reflect,
            PadMode::Replicate => BorderMode::Replicate,
            PadMode::Wrap => BorderMode::Wrap,
        });
    }
    if let Ok(s) = val.extract::<&str>() {
        let clean = s.trim().to_lowercase();
        match clean.as_str() {
            "reflect" => return Ok(BorderMode::Reflect),
            "replicate" | "nearest" => return Ok(BorderMode::Replicate),
            "wrap" => return Ok(BorderMode::Wrap),
            "constant" => return Ok(BorderMode::Constant { value: 0 }),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown border mode '{}'. Expected: 'reflect', 'replicate', 'wrap', 'constant'",
                    s
                )))
            }
        }
    }
    if let Ok(v) = val.extract::<u8>() {
        return Ok(BorderMode::Constant { value: v });
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Expected BorderMode or str ('reflect', 'replicate', 'wrap', 'constant'), got {}",
        val.get_type().name()?
    )))
}
