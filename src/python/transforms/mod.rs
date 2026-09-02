// Python wrappers for transforms
//
// This module provides Python-accessible transform classes.
// All transforms have been migrated to the unified API (random.rs).

use crate::core::Transform;

#[cfg(feature = "python")]
use pyo3::prelude::*;

mod random;
mod compose;
pub mod choice;
mod helpers;

// Unified API (with distribution support) - exports ALL transforms
pub use random::{
    PyHorizontalFlip, PyVerticalFlip, PyTranspose, PyRotate, PyResize, PyCrop, PyRandomCrop, PyPad,
    PyAffine,
    PyInvert,
    PyBrightness, PyContrast, PyPosterize, PySolarize,
    PyGamma, PyNormalize, PyEqualize, PyAutoContrast, PyToGray, PyToSepia, PyToRGB,
    PyGaussNoise, PyMultiplicativeNoise, PySaltAndPepper,
    PyRGBShift, PyHueSaturationValue,
    PyColorTemperature, PyColorTint, PyColorBalance, PyChannelShuffle,
    PyCoarseDropout, PyGridDropout,
    PyGaussianBlur, PyGaussianBlurSigma, PyMedianBlur, PySharpen, PyEmboss, PyEdgeDetection,
};
pub use compose::PyCompose;
pub use choice::{PyChoice, PyIdentity};

/// Trait for extracting the inner transform from Python wrappers
#[cfg(feature = "python")]
pub trait PyTransformExtract {
    fn as_transform(&self) -> Box<dyn Transform>;
}
