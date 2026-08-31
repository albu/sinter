// Python bindings for distribution types
//
// This module provides Python classes for the Dist enum, allowing users
// to specify distributions for transform parameters.
//
// # Example
//
// ```python
// from sinter import Compose, Brightness, Uniform, Bernoulli, Constant
//
// # All of these work:
// Compose([
//     Brightness(delta=50),                     # implicit Constant
//     Brightness(delta=Constant(50)),            # explicit Constant
//     Brightness(delta=Uniform(-30, 30)),        # Uniform
//     HorizontalFlip(p=Bernoulli(0.5)),          # Bernoulli
// ])
// ```

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::sampling::Dist;

// =============================================================================
// Constant
// =============================================================================

/// A constant value (no sampling)
///
/// This is the default when you pass a plain number to a transform.
/// You can also use it explicitly for clarity.
#[cfg(feature = "python")]
#[pyclass(name = "Constant")]
#[derive(Clone, Debug)]
pub struct PyConstant {
    pub value: f32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyConstant {
    #[new]
    fn new(value: f32) -> Self {
        Self { value }
    }

    fn __repr__(&self) -> String {
        format!("Constant({})", self.value)
    }

    fn __eq__(&self, other: &PyAny) -> PyResult<bool> {
        if let Ok(other_const) = other.extract::<PyRef<PyConstant>>() {
            Ok(self.value == other_const.value)
        } else {
            Ok(false)
        }
    }
}

// =============================================================================
// Uniform
// =============================================================================

/// Uniform distribution over [min, max]
///
/// # Example
/// ```python
/// Brightness(delta=Uniform(-30, 30))  # Samples uniformly from [-30, 30]
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "Uniform")]
#[derive(Clone, Debug)]
pub struct PyUniform {
    pub min: f32,
    pub max: f32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyUniform {
    #[new]
    fn new(min: f32, max: f32) -> PyResult<Self> {
        if min >= max {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Uniform: min ({}) must be less than max ({})", min, max)
            ));
        }
        Ok(Self { min, max }
        )
    }

    fn __repr__(&self) -> String {
        format!("Uniform({}, {})", self.min, self.max)
    }
}

// =============================================================================
// UniformInt
// =============================================================================

/// Uniform distribution over integers [min, max]
///
/// # Example
/// ```python
/// Posterize(bits=UniformInt(1, 8))  # Samples uniformly from [1, 8]
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "UniformInt")]
#[derive(Clone, Debug)]
pub struct PyUniformInt {
    pub min: i32,
    pub max: i32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyUniformInt {
    #[new]
    fn new(min: i32, max: i32) -> PyResult<Self> {
        if min >= max {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("UniformInt: min ({}) must be less than max ({})", min, max)
            ));
        }
        Ok(Self { min, max }
        )
    }

    fn __repr__(&self) -> String {
        format!("UniformInt({}, {})", self.min, self.max)
    }
}

// =============================================================================
// Bernoulli
// =============================================================================

/// Bernoulli distribution (probability)
///
/// Used for probability-based transforms like HorizontalFlip, Invert, etc.
///
/// # Example
/// ```python
/// HorizontalFlip(p=Bernoulli(0.5))  # 50% chance of flipping
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "Bernoulli")]
#[derive(Clone, Debug)]
pub struct PyBernoulli {
    pub p: f32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBernoulli {
    #[new]
    fn new(p: f32) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Bernoulli: p must be in [0, 1], got {}", p)
            ));
        }
        Ok(Self { p }
        )
    }

    fn __repr__(&self) -> String {
        format!("Bernoulli({})", self.p)
    }
}

// =============================================================================
// Normal
// =============================================================================

/// Normal (Gaussian) distribution
///
/// # Example
/// ```python
/// Brightness(delta=Normal(0, 30))  # Mean=0, Std=30
/// ```
#[cfg(feature = "python")]
#[pyclass(name = "Normal")]
#[derive(Clone, Debug)]
pub struct PyNormal {
    pub mu: f32,
    pub sigma: f32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyNormal {
    #[new]
    fn new(mu: f32, sigma: f32) -> PyResult<Self> {
        if sigma < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Normal: sigma must be non-negative, got {}", sigma)
            ));
        }
        Ok(Self { mu, sigma }
        )
    }

    fn __repr__(&self) -> String {
        format!("Normal({}, {})", self.mu, self.sigma)
    }
}

// =============================================================================
// Conversion helper
// =============================================================================

/// Format a Dist for readable __repr__ output
pub fn format_dist(dist: &Dist) -> String {
    match dist {
        Dist::Constant(v) => format!("{}", v),
        Dist::Uniform { min, max } => format!("Uniform({}, {})", min, max),
        Dist::UniformInt { min, max } => format!("UniformInt({}, {})", min, max),
        Dist::Bernoulli { p } => format!("Bernoulli({})", p),
        Dist::Normal { mu, sigma } => format!("Normal({}, {})", mu, sigma),
    }
}

/// Parse a Python value to a Dist
///
/// This helper function attempts to convert a Python value to a Dist enum.
/// It tries:
/// 1. PyConstant -> Dist::Constant
/// 2. PyUniform -> Dist::Uniform
/// 3. PyUniformInt -> Dist::UniformInt
/// 4. PyBernoulli -> Dist::Bernoulli
/// 5. PyNormal -> Dist::Normal
/// 6. (min, max) tuple or list of 2 numbers -> Dist::Uniform
/// 7. Plain f32/i32 -> Dist::Constant (implicit conversion)
#[cfg(feature = "python")]
pub fn parse_distribution(value: &PyAny) -> PyResult<Dist> {
    // Try distribution objects first
    if let Ok(c) = value.extract::<PyRef<PyConstant>>() {
        return Ok(Dist::Constant(c.value));
    }
    if let Ok(u) = value.extract::<PyRef<PyUniform>>() {
        return Ok(Dist::Uniform { min: u.min, max: u.max });
    }
    if let Ok(u) = value.extract::<PyRef<PyUniformInt>>() {
        return Ok(Dist::UniformInt { min: u.min, max: u.max });
    }
    if let Ok(b) = value.extract::<PyRef<PyBernoulli>>() {
        return Ok(Dist::Bernoulli { p: b.p });
    }
    if let Ok(n) = value.extract::<PyRef<PyNormal>>() {
        return Ok(Dist::Normal { mu: n.mu, sigma: n.sigma });
    }

    // Try tuple (min, max)
    if let Ok((min, max)) = value.extract::<(f32, f32)>() {
        if min >= max {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Uniform: min ({}) must be less than max ({})", min, max)
            ));
        }
        return Ok(Dist::Uniform { min, max });
    }

    // Try list [min, max]
    if let Ok(vec) = value.extract::<Vec<f32>>() {
        if vec.len() == 2 {
            let min = vec[0];
            let max = vec[1];
            if min >= max {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Uniform: min ({}) must be less than max ({})", min, max)
                ));
            }
            return Ok(Dist::Uniform { min, max });
        }
    }

    // Implicit: plain number becomes Constant
    if let Ok(v) = value.extract::<f32>() {
        return Ok(Dist::Constant(v));
    }
    if let Ok(v) = value.extract::<i32>() {
        return Ok(Dist::Constant(v as f32));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        format!(
            "Expected a distribution (Constant, Uniform, Bernoulli, Normal), a (min, max) tuple, or a number, got {}",
            value.get_type().name()?
        )
    ))
}
