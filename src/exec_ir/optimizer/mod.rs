// Optimizer
//
// Transforms a Plan into an optimized ExecPlan by applying fusion rules.

mod canonicalize;
#[cfg(test)]
mod fusion_tests;
mod type_conversions;
mod stats;

use crate::core::{BarrierImage, FusableImage};
use crate::exec_ir::nodes::{ExecNode, ExecNodeKind, ExecPlan, FastKernel};
use crate::exec_ir::fusion::{fuse_transform_block, FusionResult};
use crate::sampled_ir::{Plan, SampledImageOp};
use crate::transforms::{FusedLut, LutOp, Resize, ToGray};
pub use type_conversions::{
    is_geometric_transform, is_geometric_transform_sampled, try_as_lut_op, try_as_lut_op_sampled,
    try_as_matrix_op_internal, try_as_matrix_op_sampled,
};
pub use stats::{print_stats, BlockStats, FusionStrategy, OptimizerDebug};

use canonicalize::canonicalize;

/// Optimizer / Planner
///
/// Transforms a Plan into an optimized ExecPlan by applying fusion rules.
pub struct Optimizer {
    debug: OptimizerDebug,
    stats: Vec<BlockStats>,
}

impl Optimizer {
    /// Create a new optimizer
    pub fn new() -> Self {
        Self {
            debug: OptimizerDebug::None,
            stats: Vec::new(),
        }
    }

    /// Create a new optimizer with debug output enabled
    pub fn with_debug(mut self) -> Self {
        self.debug = OptimizerDebug::Verbose;
        self
    }

    /// Get the fusion statistics from the last optimization
    pub fn stats(&self) -> &[BlockStats] {
        &self.stats
    }

    /// Print a summary of fusion statistics
    pub fn print_stats(&self) {
        stats::print_stats(&self.stats);
    }

    /// Optimize a Plan into an ExecPlan
    ///
    /// # Fusion Rules
    ///
    /// The optimizer applies a decision tree to select the best fusion strategy:
    ///
    /// 1. **Structural Fusion**: Geometric transforms (Rotate, Flip) combined
    ///    with LUT transforms are fused into a single StructuralKernel with
    ///    photometric lifting.
    ///
    /// 2. **Matrix Fusion**: Consecutive MatrixOp transforms (ToSepia, Saturation, etc.)
    ///    are composed into a single FusedMatrix by multiplying their 3x3 matrices.
    ///
    /// 3. **LUT Fusion**: Consecutive LUT transforms (Invert, Solarize, Posterize,
    ///    Brightness, Contrast, Normalize) are composed into a single FusedLut.
    ///
    /// 4. **General Fusion**: Consecutive InPlace + Preserve transforms are fused
    ///    into a single Fused node.
    ///
    /// 5. **Barriers**: Any transform that is not InPlace + Preserve becomes a Barrier.
    ///
    /// # Example
    ///
    /// ```text
    /// Input: ToSepia → Saturation(0.7) → Resize
    ///
    /// Output: Fused(FusedMatrix { matrix: M_sat × M_sepia }) → Barrier(Resize)
    /// ```
    pub fn optimize(&mut self, plan: Plan) -> ExecPlan {
        self.stats.clear();

        if matches!(self.debug, OptimizerDebug::Verbose) {
            println!("\n=== Optimizer: Starting ===");
        }

        let mut exec_nodes = Vec::new();
        let mut current_fused: Vec<SampledImageOp> = Vec::new();

        let mut ops = plan.ops;
        canonicalize::hoist_crops(&mut ops);

        let mut i = 0;
        while i < ops.len() {
            let op = &ops[i];

            // Special check: Resize followed by contiguous LUT ops
            if let SampledImageOp::Resize { width, height, interpolation } = op {
                let new_width = *width as usize;
                let new_height = *height as usize;
                use crate::transforms::geometric::resize::ResizeInterpolation;
                let interp = match interpolation {
                    crate::sampled_ir::ops::Interpolation::Nearest => ResizeInterpolation::Nearest,
                    crate::sampled_ir::ops::Interpolation::Bilinear => ResizeInterpolation::Bilinear,
                    crate::sampled_ir::ops::Interpolation::Bicubic => ResizeInterpolation::Bicubic,
                    crate::sampled_ir::ops::Interpolation::Lanczos4 => ResizeInterpolation::Lanczos4,
                };


                let mut lut_count = 0;
                while i + 1 + lut_count < ops.len() && ops[i + 1 + lut_count].is_lut_op() {
                    lut_count += 1;
                }

                if lut_count >= 1 {
                    self.flush_fused_block(&mut exec_nodes, &mut current_fused);

                    let mut group = Vec::with_capacity(1 + lut_count);
                    group.push(op.clone());
                    for j in 0..lut_count {
                        group.push(ops[i + 1 + j].clone());
                    }

                    let fused_lut = FusedLut::from_sampled_ops(&group[1..]);
                    let luts_3c = fused_lut.luts_3c;
                    let lut_1c = fused_lut.lut;
                    let resize = Resize::with_interpolation(new_width, new_height, interp);

                    exec_nodes.push(ExecNode::with_kernel_kind(
                        ExecNodeKind::Fused(group),
                        crate::exec_ir::nodes::KernelKind::ResizeWithLut {
                            resize,
                            luts_3c: luts_3c.map(Box::new),
                            lut_1c,
                        },
                    ));
                    self.stats.push(BlockStats {
                        input_count: 1 + lut_count,
                        output_count: 1,
                        strategy: FusionStrategy::Lut,
                    });

                    i += 1 + lut_count;
                    continue;
                }
            }

            // Check if this op can be fused
            let is_geometric = is_geometric_transform_sampled(op);

            if !op.is_fuseable() && !is_geometric {
                // True barrier (Crop, etc.) - flush current block
                self.flush_fused_block(&mut exec_nodes, &mut current_fused);
                exec_nodes.push(ExecNode::barrier(op.clone()));
                self.stats.push(BlockStats {
                    input_count: 1,
                    output_count: 1,
                    strategy: FusionStrategy::None,
                });
                i += 1;
                continue;
            }

            if matches!(self.debug, OptimizerDebug::Verbose) {
                println!("Op {}: Adding to fuseable block", i);
            }
            current_fused.push(op.clone());
            i += 1;
        }

        // Flush remaining fused block
        self.flush_fused_block(&mut exec_nodes, &mut current_fused);

        ExecPlan::from_nodes_with_stats(exec_nodes, self.stats.clone())
    }


    /// Flush a fused block of transforms, applying all fusion optimizations
    fn flush_fused_block(
        &mut self,
        exec_nodes: &mut Vec<ExecNode>,
        fused: &mut Vec<SampledImageOp>,
    ) {
        if fused.is_empty() {
            return;
        }

        let input_count = fused.len();

        // Phase 2: Canonicalization - geometric hoisting
        // This runs BEFORE fusion to ensure blocks are in canonical form
        canonicalize(fused);

        // Try fusion on the canonicalized block
        match fuse_transform_block(fused) {
            FusionResult::Success(nodes) => {
                let output_count = nodes.len();
                if output_count == 0 {
                    // Empty nodes means transforms canceled to identity
                    if matches!(self.debug, OptimizerDebug::Verbose) {
                        println!("  -> Identity transform, skipping");
                    }
                    self.stats.push(BlockStats {
                        input_count,
                        output_count: 0,
                        strategy: FusionStrategy::Identity,
                    });
                } else {
                    if matches!(self.debug, OptimizerDebug::Verbose) {
                        println!("  -> Fused into {} execution nodes", output_count);
                    }
                    exec_nodes.extend(nodes);

                    let strategy = self.detect_strategy(fused);
                    self.stats.push(BlockStats {
                        input_count,
                        output_count,
                        strategy,
                    });
                }
            }
            FusionResult::NotApplicable => {
                // No fusion strategy applied - create individual nodes
                if matches!(self.debug, OptimizerDebug::Verbose) {
                    println!("  -> Not fusable, creating individual nodes");
                }
                for op in fused.drain(..) {
                    exec_nodes.push(ExecNode::fused(vec![op]));
                }

                self.stats.push(BlockStats {
                    input_count,
                    output_count: input_count,
                    strategy: FusionStrategy::None,
                });
            }
        }

        fused.clear();
    }

    /// Detect which fusion strategy was used based on the transforms
    fn detect_strategy(&self, fused: &[SampledImageOp]) -> FusionStrategy {
        use crate::sampled_ir::SampledImageOp::*;

        // Check for Structural Fusion (geometric + LUT)
        let has_geometric = fused
            .iter()
            .any(|t| matches!(t, HorizontalFlip | VerticalFlip | Transpose | Rotate { .. }));
        let has_lut = fused.iter().any(|t| try_as_lut_op_sampled(t).is_some());

        if has_geometric && has_lut {
            return FusionStrategy::Structural;
        }

        // Geometric-only fusion (all transforms are geometric, no LUT)
        let all_geometric = fused.iter().all(|t| is_geometric_transform_sampled(t));
        if all_geometric && !fused.is_empty() {
            return FusionStrategy::Geometric;
        }

        // Check if all are MatrixOp
        let all_matrix = fused.iter().all(|t| try_as_matrix_op_sampled(t).is_some());
        if all_matrix && !fused.is_empty() {
            return FusionStrategy::Matrix;
        }

        // Check if all are LUT ops
        let all_lut = fused.iter().all(|t| try_as_lut_op_sampled(t).is_some());
        if all_lut && !fused.is_empty() {
            return FusionStrategy::Lut;
        }

        // Default to general fusion
        FusionStrategy::General
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let opt = Optimizer::new();
        assert_eq!(opt.stats.len(), 0);
    }

    #[test]
    fn test_optimizer_with_debug() {
        let opt = Optimizer::new().with_debug();
        assert!(matches!(opt.debug, OptimizerDebug::Verbose));
    }
}
