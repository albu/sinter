// PySampledImageProgramBuilder wrapper

use super::op::PySampledImageOp;
use crate::sampled_ir::SampledImageProgramBuilder;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Builder for SampledImageProgram
#[cfg(feature = "python")]
#[pyclass(name = "_SampledImageProgramBuilder")]
pub struct PySampledImageProgramBuilder {
    pub inner: SampledImageProgramBuilder,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySampledImageProgramBuilder {
    /// Add an operation to the builder
    /// Note: This returns a new builder (Python-compatible pattern)
    fn add(&self, op: &PySampledImageOp) -> PyResult<PySampledImageProgramBuilder> {
        // Clone the builder and add the op (not efficient but Python-compatible)
        let mut new_builder = self.inner.clone();
        new_builder = new_builder.add(op.inner.clone());
        Ok(PySampledImageProgramBuilder { inner: new_builder })
    }

    /// Build the final program
    fn build(&self) -> PyResult<super::program::PySampledImageProgram> {
        Ok(super::program::PySampledImageProgram {
            inner: self.inner.clone().build(),
        })
    }

    fn __repr__(&self) -> String {
        // We can't access .ops.len() directly since it's private
        format!("_SampledImageProgramBuilder(...)")
    }
}
