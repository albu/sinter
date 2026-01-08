// Execution IR (ExecIR)
//
// This module defines:
// 1. ExecIR - the optimized representation after fusion
// 2. Fusion strategies - the decision tree for transform fusion
// 3. The optimizer that transforms Plan → ExecPlan
// 4. Execution of optimized plans

mod nodes;
mod execution;
pub mod fusion;
pub mod optimizer;

#[cfg(test)]
mod tests;

// Re-export for convenience
pub use nodes::{ExecNode, ExecNodeKind, ExecPlan, FusionStats, FastKernel};
pub use optimizer::{Optimizer, BlockStats, FusionStrategy, OptimizerDebug, print_stats};
