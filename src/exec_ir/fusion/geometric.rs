// Strategy 0: Geometric-Only Fusion
//
// Composes pure geometric sequences into a single operation using D4 group
// composition. This eliminates redundant geometric transformations.

use super::FusionResult;
use crate::core::Transform;
use crate::core::{Executable, FusableImage};
use crate::exec_ir::nodes::{FastKernel, ExecNode, ExecNodeKind};
use crate::sampled_ir::SampledImageOp;
use crate::transforms::{Orientation, StructuralKernel};

/// Strategy 0: Geometric-Only Fusion
///
/// Composes pure geometric sequences into a single operation using D4 group
/// composition. This eliminates redundant geometric transformations.
///
/// # Examples
/// ```text
/// [Rot90, Rot90]              → Rot180
/// [Rot90, Rot90, Rot90]        → Rot270
/// [Rot90, Rot90, Rot90, Rot90] → Identity (skip!)
/// [FlipH, FlipH]               → Identity (skip!)
/// [FlipH, FlipV]               → Rot180
/// [FlipH, Rot90, FlipV]        → Transpose
/// ```
///
/// # Why This Matters
///
/// Without fusion, each geometric operation touches memory. For example:
/// - Rot90 allocates a new buffer (OutOfPlace)
/// - Rot180 allocates another buffer
/// - Total: 2 allocations, 4 memory operations
///
/// With fusion:
/// - Compose Rot90 + Rot90 → Rot180
/// - Single allocation, 2 memory operations
///
/// For Identity results (e.g., FlipH + FlipH), we eliminate all operations!
pub(crate) fn try_geometric_only_fusion(fused: &mut Vec<SampledImageOp>) -> FusionResult {
    use crate::sampled_ir::ops::RotateAngle as SampledRotateAngle;

    // Step 1: Check if ALL transforms are geometric
    for op in fused.iter() {
        match op {
            SampledImageOp::HorizontalFlip
            | SampledImageOp::VerticalFlip
            | SampledImageOp::Transpose
            | SampledImageOp::Rotate { .. } => {
                // Geometric - continue
            }
            _ => {
                // Non-geometric SampledImageOp found - can't use this strategy
                return FusionResult::NotApplicable;
            }
        }
    }

    // Need at least one transform
    if fused.is_empty() {
        return FusionResult::NotApplicable;
    }

    // Step 2: Compose all geometric transforms using D4 group
    let mut composed_orientation = Orientation::Identity;

    for op in fused.iter() {
        match op {
            SampledImageOp::HorizontalFlip => {
                composed_orientation = composed_orientation.compose(Orientation::FlipH);
            }
            SampledImageOp::VerticalFlip => {
                composed_orientation = composed_orientation.compose(Orientation::FlipV);
            }
            SampledImageOp::Transpose => {
                composed_orientation = composed_orientation.compose(Orientation::Transpose);
            }
            SampledImageOp::Rotate { angle } => {
                let orientation = match angle {
                    SampledRotateAngle::Rotate90 => Orientation::Rot90,
                    SampledRotateAngle::Rotate180 => Orientation::Rot180,
                    SampledRotateAngle::Rotate270 => Orientation::Rot270,
                };
                composed_orientation = composed_orientation.compose(orientation);
            }
            _ => {
                // Should not reach here since we checked above
                return FusionResult::NotApplicable;
            }
        }
    }

    // Step 3: Check result
    if composed_orientation == Orientation::Identity {
        // Geometric sequence composes to identity - return empty nodes
        return FusionResult::Success(vec![]);
    }

    // Step 4: Create StructuralKernel (pure geometric, no LUT)
    let structural_kernel = StructuralKernel::new(composed_orientation);

    // Bind fast-path kernel - use Executable trait directly
    let orientation_for_kernel = composed_orientation;
    let kernel: FastKernel = Box::new(
        move |image: &mut FusableImage| -> Option<crate::core::BarrierImage> {
            let kernel = StructuralKernel::new(orientation_for_kernel);
            Executable::execute(&kernel, image)
        },
    );

    // Store the ORIGINAL transforms in the ExecNode for counting
    let original_transforms = std::mem::take(fused);

    let node = ExecNode::with_kernel(ExecNodeKind::Fused(original_transforms), kernel);

    FusionResult::Success(vec![node])
}
