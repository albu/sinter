// Execution logic for ExecPlan
//
// Provides the core execution methods for running optimized plans.

use crate::core::{BarrierImage, FusableImage};

/// Execute a plan on an image
///
/// This is called from ExecPlan::execute() and handles the main execution loop.
pub fn execute_plan(
    plan: &super::nodes::ExecPlan,
    initial_image: &mut FusableImage,
) -> Option<BarrierImage> {
    // We may switch to a barrier image after a Resize/other barrier
    let mut barrier_image: Option<BarrierImage> = None;

    for (_i, node) in plan.nodes.iter().enumerate() {

        // Dispatch based on whether we're using the barrier image or the original
        let result = if let Some(ref mut barrier) = barrier_image {
            // Execute on the barrier image (borrowed view)
            let mut img_view = barrier.as_fusable();
            node.execute(&mut img_view)
        } else {
            // Execute on the original borrowed image
            node.execute(initial_image)
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

