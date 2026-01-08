// Python bindings for Sampled IR (SampledImageProgram)
//
// Provides Python-accessible classes for the deterministic, serializable
// intermediate representation that is produced after sampling.
//
// NOTE: This type is internal (prefixed with _) and should not be
// used directly by users. It is returned by Compose.sample_with_seed().

mod program;

pub use program::PySampledImageProgram;
