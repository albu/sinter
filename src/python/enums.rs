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
