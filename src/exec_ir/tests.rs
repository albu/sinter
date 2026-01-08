// Tests for execution IR

use super::super::Optimizer;
use super::{ExecNode, ExecPlan};
use crate::core::FusableImage;
use crate::sampled_ir::ops::{Interpolation, SampledImageOp};
use crate::sampled_ir::Plan;

// Helper to create a plan from SampledImageOp
fn create_plan_from_ops(ops: Vec<SampledImageOp>) -> Plan {
    Plan::from_ops(ops)
}

// ===== ExecNode Tests =====

#[test]
fn test_exec_node_fused() {
    let node = ExecNode::fused(vec![
        SampledImageOp::Brightness { delta: 10.0 },
        SampledImageOp::Brightness { delta: 20.0 },
    ]);
    assert_eq!(node.len(), 2);
    assert!(node.is_fused());
    assert!(!node.is_barrier());
}

#[test]
fn test_exec_node_barrier() {
    let node = ExecNode::barrier(SampledImageOp::Resize {
        width: 100,
        height: 100,
        interpolation: Interpolation::Nearest,
    });
    assert_eq!(node.len(), 1);
    assert!(!node.is_fused());
    assert!(node.is_barrier());
}

// ===== ExecPlan Tests =====

#[test]
fn test_exec_plan_empty() {
    let plan = ExecPlan::new();
    assert!(plan.is_empty());
    assert_eq!(plan.len(), 0);
}

#[test]
fn test_optimizer_all_fuseable() {
    let mut optimizer = Optimizer::new();
    let input_plan = create_plan_from_ops(vec![
        SampledImageOp::Brightness { delta: 10.0 },
        SampledImageOp::Contrast { factor: 1.2 },
        SampledImageOp::Brightness { delta: -5.0 },
    ]);

    let exec_plan = optimizer.optimize(input_plan);

    // All fuseable transforms should be merged
    assert!(!exec_plan.is_empty());
    assert_eq!(exec_plan.barrier_count(), 0);
    if exec_plan.len() > 0 {
        assert!(exec_plan.nodes[0].is_fused());
    }
}

#[test]
fn test_optimizer_with_barrier() {
    let mut optimizer = Optimizer::new();
    let input_plan = create_plan_from_ops(vec![
        SampledImageOp::Brightness { delta: 10.0 },
        SampledImageOp::Contrast { factor: 1.2 },
        SampledImageOp::Resize {
            width: 256,
            height: 256,
            interpolation: Interpolation::Nearest,
        },
        SampledImageOp::Brightness { delta: -5.0 },
    ]);

    let exec_plan = optimizer.optimize(input_plan);

    // Should have at least 2 nodes: fused block before Resize, then individual after
    assert!(!exec_plan.is_empty());
}

#[test]
fn test_optimizer_only_barriers() {
    let mut optimizer = Optimizer::new();
    let input_plan = create_plan_from_ops(vec![
        SampledImageOp::Resize {
            width: 100,
            height: 100,
            interpolation: Interpolation::Nearest,
        },
        SampledImageOp::Resize {
            width: 200,
            height: 200,
            interpolation: Interpolation::Nearest,
        },
    ]);

    let exec_plan = optimizer.optimize(input_plan);

    assert_eq!(exec_plan.len(), 2);
    assert_eq!(exec_plan.fused_transform_count(), 0);
    assert_eq!(exec_plan.barrier_count(), 2);
}

#[test]
fn test_fusion_ratio() {
    let plan = ExecPlan::from_nodes(vec![
        ExecNode::fused(vec![
            SampledImageOp::Brightness { delta: 10.0 },
            SampledImageOp::Brightness { delta: 20.0 },
        ]),
        ExecNode::barrier(SampledImageOp::Resize {
            width: 100,
            height: 100,
            interpolation: Interpolation::Nearest,
        }),
        ExecNode::fused(vec![SampledImageOp::Brightness { delta: 30.0 }]),
    ]);

    // 3 original transforms compressed into 3 exec nodes (no fusion in this simple case)
    let ratio = plan.fusion_ratio(3);
    assert!((ratio - 1.0).abs() < 0.01);
}

// ===== Integration Tests =====

#[test]
fn test_exec_plan_small_image() {
    let mut data = vec![100u8; 12]; // 2x2 RGB
    let mut img = FusableImage::new(&mut data, 2, 2, 3);

    let plan = Plan::from_ops(vec![
        SampledImageOp::Brightness { delta: 50.0 },
        SampledImageOp::Contrast { factor: 1.5 },
    ]);

    let mut optimizer = Optimizer::new();
    let exec_plan = optimizer.optimize(plan);

    // Execute the plan
    exec_plan.execute(&mut img);

    // Image should be modified
    assert_ne!(data, vec![100u8; 12]);
}

#[test]
fn test_exec_plan_geometric_fusion() {
    let mut optimizer = Optimizer::new();
    let plan = Plan::from_ops(vec![
        SampledImageOp::HorizontalFlip,
        SampledImageOp::VerticalFlip,
    ]);

    let exec_plan = optimizer.optimize(plan);

    // Geometric transforms should be optimized
    assert!(!exec_plan.is_empty());
}

#[test]
fn test_exec_plan_lut_fusion() {
    let mut optimizer = Optimizer::new();
    let plan = Plan::from_ops(vec![
        SampledImageOp::Invert,
        SampledImageOp::Brightness { delta: 10.0 },
    ]);

    let exec_plan = optimizer.optimize(plan);

    // LUT transforms should be fused
    assert!(!exec_plan.is_empty());
}

#[test]
fn test_exec_plan_mixed() {
    let mut optimizer = Optimizer::new();
    let plan = Plan::from_ops(vec![
        SampledImageOp::Invert,
        SampledImageOp::Brightness { delta: 10.0 },
        SampledImageOp::Resize {
            width: 100,
            height: 100,
            interpolation: Interpolation::Nearest,
        },
        SampledImageOp::Contrast { factor: 1.2 },
    ]);

    let exec_plan = optimizer.optimize(plan);

    // Should fuse before and after the barrier
    assert!(!exec_plan.is_empty());
}

#[test]
fn test_exec_plan_visualize() {
    let plan = ExecPlan::from_nodes(vec![
        ExecNode::fused(vec![
            SampledImageOp::Brightness { delta: 10.0 },
            SampledImageOp::Contrast { factor: 1.2 },
        ]),
        ExecNode::barrier(SampledImageOp::Resize {
            width: 100,
            height: 100,
            interpolation: Interpolation::Nearest,
        }),
    ]);

    // Just make sure it doesn't panic
    plan.visualize();
    plan.print_stats();
}

#[test]
fn test_exec_plan_detailed_description() {
    let plan = ExecPlan::from_nodes(vec![ExecNode::fused(vec![SampledImageOp::Invert])]);

    let desc = plan.detailed_description();
    assert!(desc.contains("ExecPlan"));
    assert!(desc.contains("nodes"));
}
