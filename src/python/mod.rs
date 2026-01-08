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

/// Initialize OpenCV to run single-threaded for consistent benchmarking
#[cfg(feature = "opencv")]
fn init_opencv_single_threaded() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = opencv::core::set_num_threads(1);
    });
}

/// Python module for sinter
#[pymodule]
fn sinter(_py: Python, m: &PyModule) -> PyResult<()> {
    #[cfg(feature = "opencv")]
    init_opencv_single_threaded();

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

    // Batch transforms
    register_classes!(m, PyMixUp, PyCutMix, PyMosaic, PyBatchPipeline,);

    // PyTorch integration
    m.add_function(wrap_pyfunction!(apply_to_tensor_inplace, m)?)?;

    // OpenCV control
    #[cfg(feature = "opencv")]
    {
        m.add_function(wrap_pyfunction!(set_opencv_num_threads, m)?)?;
        m.add_function(wrap_pyfunction!(get_opencv_num_threads, m)?)?;
    }

    Ok(())
}

/// Set the number of threads OpenCV uses
#[cfg(feature = "opencv")]
#[pyfunction]
fn set_opencv_num_threads(_py: Python, num: i32) -> PyResult<()> {
    opencv::core::set_num_threads(num);
    Ok(())
}

/// Get the number of threads OpenCV is configured to use
#[cfg(feature = "opencv")]
#[pyfunction]
fn get_opencv_num_threads(_py: Python) -> PyResult<i32> {
    match opencv::core::get_num_threads() {
        Ok(threads) => Ok(threads),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to get OpenCV num threads: {}",
            e
        ))),
    }
}
