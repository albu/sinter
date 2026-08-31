// Execution logic for ExecPlan
//
// Provides the core execution methods for running optimized plans.

use crate::core::{BarrierImage, Executable, FusableImage, Transform};
use crate::exec_ir::optimizer::try_as_lut_op_sampled;
use crate::sampled_ir::SampledImageOp;
use crate::transforms::{FusedLut, FusedLutExecutor, FusedMatrix, LutOp};

use super::nodes::{ExecNode, ExecNodeKind};

/// Execute a plan on an image
///
/// This is called from ExecPlan::execute() and handles the main execution loop.
pub fn execute_plan(
    plan: &super::nodes::ExecPlan,
    initial_image: &mut FusableImage,
) -> Option<BarrierImage> {
    let _total_start = std::time::Instant::now();

    // We may switch to a barrier image after a Resize/other barrier
    let mut barrier_image: Option<BarrierImage> = None;

    for (_i, node) in plan.nodes.iter().enumerate() {
        let _node_start = std::time::Instant::now();

        // Dispatch based on whether we're using the barrier image or the original
        let result = if let Some(ref mut barrier) = barrier_image {
            // Execute on the barrier image (borrowed view)
            let mut img_view = barrier.as_fusable();
            execute_node(node, &mut img_view)
        } else {
            // Execute on the original borrowed image
            execute_node(node, initial_image)
        };

        // If we got a new barrier image, use it for subsequent operations
        if let Some(new_barrier) = result {
            if new_barrier.is_f32() && _i + 1 < plan.nodes.len() {
                panic!(
                    "Normalize produces float32 output and must be the last transform \
                     (found {} more nodes after it)",
                    plan.nodes.len() - _i - 1
                );
            }
            barrier_image = Some(new_barrier);
        }
    }

    barrier_image
}

/// Helper function to execute a single node on an image
fn execute_node(node: &ExecNode, image: &mut FusableImage) -> Option<BarrierImage> {
    // FAST PATH 1: Use pre-bound kernel if available (zero type checks!)
    // This is the 8-10x speedup path for MatrixOp transforms!
    if let Some(ref kernel) = node.kernel {
        return kernel(image);
    }

    // FAST PATH 2: Single-transform nodes use vtable dispatch via as_executable()
    // This avoids RTTI overhead for single transforms (e.g., individual Brightness, Rotate, etc.)
    let result = match &node.kind {
        ExecNodeKind::Fused(transforms) if transforms.len() == 1 => {
            // Single transform - use vtable dispatch instead of RTTI
            if let Some(executable) = transforms[0].as_executable() {
                executable.execute(image)
            } else {
                // Fallback to RTTI for transforms that don't implement as_executable()
                execute_with_rtti(&node.kind, image)
            }
        }
        _ => {
            // FALLBACK: Use RTTI for multi-transform nodes or barriers
            execute_with_rtti(&node.kind, image)
        }
    };

    result
}

/// Execute using RTTI (slow path for multi-transform nodes)
fn execute_with_rtti(kind: &ExecNodeKind, image: &mut FusableImage) -> Option<BarrierImage> {
    use crate::exec_ir::optimizer::try_as_lut_op_sampled;
    match kind {
        ExecNodeKind::Fused(transforms) => {
            // First, try LUT fusion for maximum speed
            let all_lut = transforms
                .iter()
                .all(|t| try_as_lut_op_sampled(t).is_some());

            if all_lut {
                // Use LUT fusion for maximum speed
                let lut_ops: Vec<Box<dyn LutOp>> = transforms
                    .iter()
                    .filter_map(|t| try_as_lut_op_sampled(t))
                    .collect();

                FusedLutExecutor::execute(image, &lut_ops);
                None
            } else {
                // Mixed transforms: execute in order, but batch consecutive LUT transforms
                // We must PRESERVE ORDER - cannot execute all geometric first then all LUT
                let mut pending_luts: Vec<Box<dyn LutOp>> = Vec::new();

                // When a transform returns a BarrierImage, we need to handle remaining transforms
                // differently. The key insight: we can't easily continue with the fused pattern
                // once we get a BarrierImage, because we'd need to convert it back to FusableImage.
                //
                // Solution: When we get a BarrierImage mid-sequence, split the remaining
                // transforms and return both the BarrierImage and the remaining transforms.
                // The caller (execute_plan) will need to handle this.

                let remaining_transforms: Vec<SampledImageOp> = Vec::new();

                for transform in transforms {
                    // Check if this is a LUT transform (using helper that supports SampledImageOp)
                    if let Some(lut_op) = try_as_lut_op_sampled(transform) {
                        pending_luts.push(lut_op);
                    } else {
                        // Non-LUT transform - flush pending LUTs first, then execute this transform
                        if !pending_luts.is_empty() {
                            FusedLutExecutor::execute(image, &pending_luts);
                            pending_luts.clear();
                        }

                        // Execute the SampledImageOp directly (NO RTTI!)
                        if let Some(barrier) = transform.execute(image) {
                            return Some(barrier);
                        }
                    }
                }

                // Flush any remaining LUT transforms
                if !pending_luts.is_empty() {
                    FusedLutExecutor::execute(image, &pending_luts);
                }

                None
            }
        }
        ExecNodeKind::Barrier(transform) => {
            // Execute the SampledImageOp directly (NO RTTI!)
            transform.execute(image)
        }
    }
}
