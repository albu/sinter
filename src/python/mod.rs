// Python bindings for sinter
//
// This module provides PyO3 bindings to expose sinter functionality.

#[cfg(feature = "python")]
pub mod batch;
#[cfg(feature = "python")]
pub mod distributions;
#[cfg(feature = "python")]
pub mod enums;
#[cfg(feature = "python")]
pub mod image;
#[cfg(feature = "python")]
pub mod sampled;
#[cfg(feature = "python")]
pub mod tensor;
#[cfg(feature = "python")]
pub mod transforms;
#[cfg(feature = "python")]
pub mod types;

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ============================================================================
// Registration Helper Macro
// ============================================================================

macro_rules! register_classes {
    ($module:expr, $($class:ty),+ $(,)?) => {
        $(
            $module.add_class::<$class>()?;
        )+
    };
}

// Non-unified transforms (not yet migrated to distribution API)
// NOTE: All transforms have been migrated to the unified API or removed
// No non-unified transforms remain

// Unified API transforms (with distribution support)
#[cfg(feature = "python")]
use transforms::{
    PyAffine, PyAutoContrast, PyBrightness, PyChannelShuffle, PyCoarseDropout, PyColorBalance,
    PyColorTemperature, PyColorTint, PyCompose, PyContrast, PyCrop, PyEdgeDetection, PyEmboss,
    PyEqualize, PyGamma, PyGaussNoise, PyGaussianBlur, PyGaussianBlurSigma, PyGridDropout,
    PyHorizontalFlip, PyHueSaturationValue, PyInvert, PyMedianBlur, PyMultiplicativeNoise,
    PyNormalize, PyPad, PyPosterize, PyRGBShift, PyResize, PyRotate, PySaltAndPepper, PySharpen,
    PySolarize, PyToGray, PyToRGB, PyToSepia, PyTranspose, PyVerticalFlip,
};

// Distribution types
#[cfg(feature = "python")]
use distributions::{PyBernoulli, PyConstant, PyNormal, PyUniform, PyUniformInt};

// Enums
#[cfg(feature = "python")]
use enums::{PyEdgeMethod, PyEmbossDirection, PyInterpolation, PyPadMode, PyRotateAngle};

// Other
#[cfg(feature = "python")]
use batch::{PyBatchPipeline, PyCutMix, PyMixUp, PyMosaic};
#[cfg(feature = "python")]
use sampled::PySampledImageProgram;
#[cfg(feature = "python")]
use tensor::apply_to_tensor_inplace;

/// Python module for sinter
#[pymodule]
fn sinter(_py: Python, m: &PyModule) -> PyResult<()> {

    // Distribution types
    register_classes!(
        m,
        PyConstant,
        PyUniform,
        PyUniformInt,
        PyBernoulli,
        PyNormal,
    );

    // ===== UNIFIED API TRANSFORMS (with distribution support) =====

    // Geometric transforms
    register_classes!(
        m,
        PyHorizontalFlip,
        PyVerticalFlip,
        PyTranspose,
        PyRotate,
        PyResize,
        PyCrop,
        PyPad,
        PyAffine,
    );

    // Photometric transforms (parameter-based)
    register_classes!(
        m,
        PyInvert,
        PyBrightness,
        PyContrast,
        PyPosterize,
        PySolarize,
        PyGamma,
        PyNormalize,
        PyEqualize,
        PyAutoContrast,
        PyToGray,
        PyToSepia,
        PyToRGB,
    );

    // Noise transforms
    register_classes!(m, PyGaussNoise, PyMultiplicativeNoise, PySaltAndPepper,);

    // Color transforms
    register_classes!(
        m,
        PyRGBShift,
        PyHueSaturationValue,
        PyColorTemperature,
        PyColorTint,
        PyColorBalance,
        PyChannelShuffle,
    );

    // Dropout transforms
    register_classes!(m, PyCoarseDropout, PyGridDropout,);

    // Kernel transforms
    register_classes!(
        m,
        PyGaussianBlur,
        PyGaussianBlurSigma,
        PyMedianBlur,
        PySharpen,
        PyEmboss,
        PyEdgeDetection,
    );

    // Main compose
    m.add_class::<PyCompose>()?;

    // ===== ENUMS =====
    register_classes!(
        m,
        PyRotateAngle,
        PyInterpolation,
        PyPadMode,
        PyEmbossDirection,
        PyEdgeMethod,
    );

    // All transforms have been migrated to unified API - no non-unified transforms remain

    // Sampled IR
    m.add_class::<PySampledImageProgram>()?;
    m.add("SampledImageProgram", m.getattr("_SampledImageProgram")?)?;

    // Batch transforms
    register_classes!(m, PyMixUp, PyCutMix, PyMosaic, PyBatchPipeline,);

    // PyTorch integration
    m.add_function(wrap_pyfunction!(apply_to_tensor_inplace, m)?)?;

    Ok(())
}
