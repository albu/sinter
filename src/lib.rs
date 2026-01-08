// Sinter - Compiled Image Augmentation Engine
// Copyright (c) 2025 Sinter contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

pub mod batch;
pub mod core;
pub mod exec_ir;
pub mod labels;
pub mod sampled_ir;
pub mod sampling;
pub mod transforms;

// Python bindings (optional feature)
#[cfg(feature = "python")]
pub mod python;

// Re-export core types for convenience
pub use core::{
    is_fuseable, AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform,
};

// Re-export batch types (including label types)
pub use batch::{Batch, BatchTransform, MixUp, Label, SoftLabel, ClassIndex};

// Re-export ExecIR types
pub use exec_ir::{ExecNode, ExecPlan, Optimizer};

// Re-export sampling types
pub use sampling::{
    Bernoulli, RandomImageNode, RandomImageProgram, Rng, SamplingContext, Uniform, UniformInt,
};

// Re-export sampled IR types
pub use sampled_ir::{
    SampledBatchOp, SampledBatchProgram, SampledImageOp, SampledImageProgram, IR_VERSION, Plan,
};

// Note: Macros are exported at crate root via #[macro_export]
// Use them directly: `use crate::random_atomic_op;` or just `random_atomic_op!`

// Re-export transforms
pub use transforms::{
    execute_fused, Brightness, Contrast, FusedExecutor, HorizontalFlip, Normalize, PixelOp, Resize,
    VerticalFlip,
};
