// Extractive Fusion - Fuse contiguous homogeneous groups
//
// This is the new primary fusion strategy that replaces the all-or-nothing
// approach. It extracts what it CAN fuse from any block.

use super::FusionResult;
use crate::core::{BarrierImage, FusableImage};
use crate::exec_ir::nodes::{FastKernel, ExecNode, ExecNodeKind};
use crate::exec_ir::optimizer::{
    is_geometric_transform_sampled, try_as_lut_op_sampled, try_as_matrix_op_sampled,
};
use crate::sampled_ir::SampledImageOp;
use crate::transforms::{
    runtime::lut::LutExecutor, runtime::matrix::MatrixExecutor, FusedLut, FusedMatrix, LutOp, MatrixOp,
    ToGray,
};


/// Extractive Fusion - Fuse contiguous homogeneous groups
///
/// Instead of all-or-nothing, this strategy extracts what it CAN fuse:
///
/// # Example
/// Input: [Solarize, Contrast, Brightness, Gamma, Posterize, Saturation]
///
/// Output:
///   - FusedLut(Solarize + Contrast + Brightness + Posterize)
///   - Gamma (individual - neither LUT nor Matrix)
///   - Saturation (individual - only 1 Matrix op)
///
/// # Algorithm
/// 1. Scan the block left-to-right
/// 2. Extract contiguous groups of LUT transforms → FusedLut
/// 3. Extract contiguous groups of Matrix transforms → FusedMatrix
/// 4. Compose geometric transforms
/// 5. Leave everything else as individual nodes
pub(crate) fn try_extractive_fusion(fused: &mut Vec<SampledImageOp>) -> FusionResult {
    if fused.is_empty() {
        return FusionResult::NotApplicable;
    }

    // Check for geometric-only block (special case: can compose via D4 group)
    let all_geometric = fused.iter().all(|t| is_geometric_transform_sampled(t));
    if all_geometric && fused.len() >= 2 {
        // Multiple geometric transforms - try to compose them
        return super::geometric::try_geometric_only_fusion(fused);
    }

    // General case: extract contiguous groups
    let mut nodes = Vec::new();
    let mut start = 0;

    while start < fused.len() {
        // Try to extract a geometric group FIRST (before LUT/Matrix)
        let geom_group_len =
            find_contiguous_group_end(&fused[start..], |t| is_geometric_transform_sampled(t));
        if geom_group_len >= 2 {
            // Drain the geometric group and try to compose them
            let geom_group: Vec<SampledImageOp> =
                fused.drain(start..start + geom_group_len).collect();

            // Try geometric fusion on the group
            // Make a mutable copy for the fusion function
            let mut geom_vec = geom_group;
            match super::geometric::try_geometric_only_fusion(&mut geom_vec) {
                FusionResult::Success(geom_nodes) => {
                    if !geom_nodes.is_empty() {
                        nodes.extend(geom_nodes);
                    }
                    // If geom_nodes is empty, the transforms canceled to identity - skip them
                }
                FusionResult::NotApplicable => {
                    // Failed to compose - create individual nodes with fast kernels
                    for t in geom_vec {
                        nodes.push(create_single_transform_node(t));
                    }
                }
            }
            continue;
        }

        // Try data-dependent LUT fusion (Equalize, AutoContrast with subsequent LUT ops)
        if fused
            .get(start)
            .map_or(false, |t| t.is_data_dependent_lut_op())
        {
            match super::data_dependent_lut::try_data_dependent_lut_fusion(fused) {
                FusionResult::Success(dd_nodes) => {
                    nodes.extend(dd_nodes);
                    continue;
                }
                FusionResult::NotApplicable => {
                    // Data-dependent LUT fusion failed, fall through to regular handling
                }
            }
        }

        // Try to extract a LUT group followed by ToGray (pointwise fusion)
        let lut_count = find_contiguous_group_end(&fused[start..], |t| t.is_lut_op());
        let has_trailing_to_gray = matches!(fused.get(start + lut_count), Some(SampledImageOp::ToGray));

        if has_trailing_to_gray && lut_count >= 1 {
            let total_len = lut_count + 1;
            let group: Vec<SampledImageOp> = fused.drain(start..start + total_len).collect();
            let fused_lut = FusedLut::from_sampled_ops(&group[..lut_count]);
            let luts_3c = fused_lut
                .luts_3c
                .unwrap_or([fused_lut.lut, fused_lut.lut, fused_lut.lut]);
            nodes.push(ExecNode::with_kernel_kind(
                ExecNodeKind::Fused(group),
                crate::exec_ir::nodes::KernelKind::LutToGray {
                    luts_3c: Box::new(luts_3c),
                },
            ));
            continue;
        }

        // Try to extract a regular LUT group
        let lut_end = start + find_contiguous_group_end(&fused[start..], |t| t.is_lut_op());
        if lut_end - start >= 2 {
            // Drain the LUT group and fuse it directly with zero heap allocations
            let lut_group: Vec<SampledImageOp> = fused.drain(start..lut_end).collect();
            let fused_lut = FusedLut::from_sampled_ops(&lut_group);
            if !fused_lut.is_identity() {
                nodes.push(ExecNode::with_kernel_kind(
                    ExecNodeKind::Fused(lut_group),
                    crate::exec_ir::nodes::KernelKind::FusedLut {
                        luts_3c: fused_lut.luts_3c.map(Box::new),
                        lut_1c: fused_lut.lut,
                    },
                ));
            }
            continue;
        }

        // Try to extract a Matrix group
        let matrix_end = start
            + find_contiguous_group_end(&fused[start..], |t| try_as_matrix_op_sampled(t).is_some());
        if matrix_end - start >= 2 {
            // Drain the Matrix group and fuse it
            let matrix_group: Vec<SampledImageOp> = fused.drain(start..matrix_end).collect();
            let matrix_ops: Vec<Box<dyn MatrixOp>> = matrix_group
                .iter()
                .filter_map(|t| try_as_matrix_op_sampled(t))
                .collect();

            if !matrix_ops.is_empty() {
                let refs: Vec<&dyn MatrixOp> = matrix_ops.iter().map(|b| b.as_ref()).collect();
                let fused_matrix = FusedMatrix::from_matrix_ops(&refs);
                nodes.push(ExecNode::with_kernel_kind(
                    ExecNodeKind::Fused(matrix_group),
                    crate::exec_ir::nodes::KernelKind::FusedMatrix(fused_matrix),
                ));
            }
            continue;
        }

        // Single transform - create individual node with static enum kernel
        let transform = fused.remove(start);
        let kernel_kind = crate::exec_ir::nodes::KernelKind::Single(transform.clone());
        nodes.push(ExecNode::with_kernel_kind(
            ExecNodeKind::Fused(vec![transform]),
            kernel_kind,
        ));
        // Don't increment start - we removed an element
    }

    if nodes.is_empty() {
        FusionResult::NotApplicable
    } else {
        FusionResult::Success(nodes)
    }
}

/// Create a single-transform ExecNode using static enum dispatch
fn create_single_transform_node(transform: SampledImageOp) -> ExecNode {
    let kernel_kind = crate::exec_ir::nodes::KernelKind::Single(transform.clone());
    ExecNode::with_kernel_kind(ExecNodeKind::Fused(vec![transform]), kernel_kind)
}

/// Find the end of a contiguous group where predicate is true
/// Returns the length of the group (relative to start)
fn find_contiguous_group_end(
    slice: &[SampledImageOp],
    predicate: impl Fn(&SampledImageOp) -> bool,
) -> usize {
    for (i, op) in slice.iter().enumerate() {
        if !predicate(op) {
            return i;
        }
    }
    slice.len()
}
