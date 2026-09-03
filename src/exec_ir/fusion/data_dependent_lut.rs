// Strategy: Data-Dependent LUT Fusion
//
// Fuses data-dependent LUT transforms (Equalize, AutoContrast) with subsequent regular LUT transforms.
//
// Data-dependent LUT ops need to see the image data first to build their LUT.
// They CANNOT fuse with transforms BEFORE them, but CAN fuse with transforms AFTER them.

use super::FusionResult;
use crate::core::{BarrierImage, Executable, FusableImage};
use crate::exec_ir::nodes::{FastKernel, ExecNode, ExecNodeKind};
use crate::exec_ir::optimizer::try_as_lut_op_sampled;
use crate::sampled_ir::SampledImageOp;
use crate::transforms::{runtime::lut::LutExecutor, AutoContrast, Equalize, FusedLut, LutOp};

/// Try to fuse data-dependent LUT transforms with subsequent regular LUT transforms
///
/// # Algorithm
/// 1. Check if the first transform is a data-dependent LUT op (Equalize, AutoContrast)
/// 2. Check if there are subsequent regular LUT ops
/// 3. Create a kernel that applies data-dependent op first, then fused LUT of subsequent ops
///
/// # Example
/// Input: [Equalize, Posterize, Solarize]
///
/// Output: Single ExecNode with kernel:
///   - Apply Equalize (builds LUT from image data)
///   - Apply fused LUT of [Posterize, Solarize]
///
/// # Arguments
/// * `fused` - Mutable reference to vector of transforms (will be drained)
///
/// # Returns
/// * `FusionResult::Success(nodes)` - Fused nodes
/// * `FusionResult::NotApplicable` - First op is not data-dependent LUT
pub(crate) fn try_data_dependent_lut_fusion(fused: &mut Vec<SampledImageOp>) -> FusionResult {
    if fused.is_empty() {
        return FusionResult::NotApplicable;
    }

    // Step 1: Check if the first transform is a data-dependent LUT op
    let first_is_data_dependent = fused[0].is_data_dependent_lut_op();
    if !first_is_data_dependent {
        return FusionResult::NotApplicable;
    }

    // Step 2: Find all subsequent regular LUT ops
    let lut_end = find_contiguous_lut_group(&fused[1..]);

    if lut_end == 0 {
        // Only the data-dependent op, no subsequent LUT ops to fuse
        // Return NotApplicable so it becomes an individual node
        return FusionResult::NotApplicable;
    }

    // Step 3: Drain the group (data-dependent op + subsequent LUT ops)
    let group_end = 1 + lut_end;
    let lut_group: Vec<SampledImageOp> = fused.drain(0..group_end).collect();

    // Step 4: Separate the data-dependent op from subsequent LUT ops
    let data_dependent_op = &lut_group[0];
    // Step 4: Fuse the subsequent LUT ops directly from SampledImageOp
    let fused_lut = FusedLut::from_sampled_ops(&lut_group[1..]);
    let luts_3c = fused_lut.luts_3c;
    let lut_1c = fused_lut.lut;

    // Step 5: Create node with concrete KernelKind
    let kernel_kind = match data_dependent_op {
        SampledImageOp::Equalize => crate::exec_ir::nodes::KernelKind::EqualizeWithLut {
            luts_3c: luts_3c.map(Box::new),
            lut_1c,
        },
        SampledImageOp::AutoContrast { cutoff_low, .. } => {
            crate::exec_ir::nodes::KernelKind::AutoContrastWithLut {
                cutoff: *cutoff_low,
                luts_3c: luts_3c.map(Box::new),
                lut_1c,
            }
        }
        _ => crate::exec_ir::nodes::KernelKind::FusedLut {
            luts_3c: luts_3c.map(Box::new),
            lut_1c,
        },
    };

    let node = ExecNode::with_kernel_kind(ExecNodeKind::Fused(lut_group), kernel_kind);

    FusionResult::Success(vec![node])
}

/// Find the end of a contiguous group of regular LUT ops
/// Returns the length of the group (relative to start)
fn find_contiguous_lut_group(slice: &[SampledImageOp]) -> usize {
    for (i, op) in slice.iter().enumerate() {
        if !op.is_lut_op() {
            return i;
        }
    }
    slice.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_dependent_lut_fusion_basic() {
        let mut ops = vec![
            SampledImageOp::Equalize,
            SampledImageOp::Brightness { delta: 10.0 },
            SampledImageOp::Contrast { factor: 1.2 },
        ];

        let result = try_data_dependent_lut_fusion(&mut ops);

        // Should succeed and fuse all 3 ops
        match result {
            FusionResult::Success(nodes) => {
                assert_eq!(nodes.len(), 1);
                if let ExecNodeKind::Fused(fused_ops) = &nodes[0].kind {
                    assert_eq!(fused_ops.len(), 3);
                } else {
                    panic!("Expected Fused node");
                }
            }
            FusionResult::NotApplicable => {
                panic!("Expected Success, got NotApplicable");
            }
        }

        // Ops should be drained
        assert_eq!(ops.len(), 0);
    }

    #[test]
    fn test_data_dependent_lut_fusion_no_subsequent() {
        let mut ops = vec![SampledImageOp::Equalize];

        let result = try_data_dependent_lut_fusion(&mut ops);

        // Should return NotApplicable (no subsequent LUT ops to fuse)
        assert!(matches!(result, FusionResult::NotApplicable));

        // Ops should not be drained
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn test_data_dependent_lut_fusion_non_lut_after() {
        let mut ops = vec![SampledImageOp::Equalize, SampledImageOp::HorizontalFlip];

        let result = try_data_dependent_lut_fusion(&mut ops);

        // Should return NotApplicable (HorizontalFlip is not a LUT op)
        assert!(matches!(result, FusionResult::NotApplicable));
    }

    #[test]
    fn test_data_dependent_lut_fusion_empty() {
        let mut ops: Vec<SampledImageOp> = vec![];

        let result = try_data_dependent_lut_fusion(&mut ops);

        assert!(matches!(result, FusionResult::NotApplicable));
    }

    #[test]
    fn test_data_dependent_lut_fusion_first_not_data_dependent() {
        let mut ops = vec![
            SampledImageOp::Brightness { delta: 10.0 },
            SampledImageOp::Contrast { factor: 1.2 },
        ];

        let result = try_data_dependent_lut_fusion(&mut ops);

        // Should return NotApplicable (first op is not data-dependent)
        assert!(matches!(result, FusionResult::NotApplicable));
    }
}
