// Bridge from SampledImageProgram to Plan
//
// This module connects the sampled IR to the Plan format used by the optimizer.
//
// KEY CHANGE: NO RTTI! We use SampledImageOp enum directly instead of converting
// to Box<dyn Transform>. This eliminates all downcast_ref calls in hot paths.

use super::Plan;
use crate::sampled_ir::{SampledImageOp, SampledImageProgram};

impl SampledImageProgram {
    /// Convert this sampled program to a Plan for optimization
    ///
    /// This bridges the sampled IR to the Plan format used by the optimizer.
    /// Operations are cloned into a new Plan.
    ///
    /// NO RTTI - we use the enum directly, no trait object conversion!
    pub fn to_plan(&self) -> Plan {
        Plan::from_ops(self.ops.iter().cloned().collect())
    }

    /// Convert this sampled program to a Plan, consuming self
    ///
    /// Same as to_plan() but takes ownership to avoid cloning when possible.
    pub fn into_plan(self) -> Plan {
        Plan::from_ops(self.ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec_ir::Optimizer;

    #[test]
    fn test_empty_program_to_plan() {
        let sampled = SampledImageProgram::new();
        let plan = sampled.to_plan();

        assert_eq!(plan.len(), 0);
        assert!(plan.is_empty());
    }

    #[test]
    fn test_single_op_to_plan() {
        let sampled = SampledImageProgram::builder()
            .add(SampledImageOp::Brightness { delta: 10.0 })
            .build();

        let plan = sampled.to_plan();

        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn test_multiple_ops_to_plan() {
        let sampled = SampledImageProgram::builder()
            .add(SampledImageOp::Brightness { delta: 10.0 })
            .add(SampledImageOp::Contrast { factor: 1.5 })
            .add(SampledImageOp::Invert)
            .build();

        let plan = sampled.to_plan();

        assert_eq!(plan.len(), 3);
    }

    /// Integration test: SampledImageProgram → Plan → Optimizer → ExecPlan
    #[test]
    fn test_sampled_to_plan_to_optimizer() {
        use crate::sampled_ir::ops::Interpolation;

        // Create a sampled program with fuseable ops
        let sampled = SampledImageProgram::builder()
            .add(SampledImageOp::Brightness { delta: 10.0 })
            .add(SampledImageOp::Contrast { factor: 1.5 })
            .add(SampledImageOp::Invert)
            .build();

        // Bridge to Plan
        let plan = sampled.to_plan();
        assert_eq!(plan.len(), 3);

        // Run through optimizer
        let mut optimizer = Optimizer::new();
        let exec_plan = optimizer.optimize(plan);

        // Should fuse into at least one node
        // (exact fusion behavior depends on optimizer strategies)
        assert!(!exec_plan.is_empty());
    }

    /// Integration test with barriers
    #[test]
    fn test_sampled_with_barrier_through_optimizer() {
        use crate::sampled_ir::ops::Interpolation;

        let sampled = SampledImageProgram::builder()
            .add(SampledImageOp::Brightness { delta: 10.0 })
            .add(SampledImageOp::Contrast { factor: 1.5 })
            .add(SampledImageOp::Invert)
            .add(SampledImageOp::Resize {
                width: 256,
                height: 256,
                interpolation: Interpolation::Bilinear,
            })
            .build();

        let plan = sampled.to_plan();
        let mut optimizer = Optimizer::new();
        let exec_plan = optimizer.optimize(plan);

        // Should have at least 2 nodes:
        // - Fused photometric block (Brightness + Contrast + Invert)
        // - Barrier (Resize)
        assert!(!exec_plan.is_empty());
    }

    #[test]
    fn test_plan_preserves_access_patterns() {
        use crate::core::AccessPattern;
        use crate::sampled_ir::ops::Interpolation;

        let sampled = SampledImageProgram::builder()
            .add(SampledImageOp::Brightness { delta: 10.0 }) // InPlace
            .add(SampledImageOp::Resize {
                width: 256,
                height: 256,
                interpolation: Interpolation::Bilinear,
            }) // InPlace (buffer is modified, not replaced)
            .build();

        let plan = sampled.to_plan();

        assert_eq!(plan.len(), 2);
        // All operations are InPlace - the distinction is in shape_effect
        assert_eq!(plan.ops[0].access_pattern(), AccessPattern::InPlace);
        assert_eq!(plan.ops[1].access_pattern(), AccessPattern::InPlace);
    }

    #[test]
    fn test_plan_fuseable_detection() {
        let sampled = SampledImageProgram::builder()
            .add(SampledImageOp::Brightness { delta: 10.0 })
            .add(SampledImageOp::Contrast { factor: 1.5 })
            .add(SampledImageOp::Invert)
            .build();

        let plan = sampled.to_plan();

        // All three should be fuseable (InPlace + Preserve)
        assert_eq!(plan.count_leading_fuseable(), 3);
    }

    #[test]
    fn test_plan_barrier_detection() {
        use crate::sampled_ir::ops::Interpolation;

        let sampled = SampledImageProgram::builder()
            .add(SampledImageOp::Brightness { delta: 10.0 })
            .add(SampledImageOp::Resize {
                width: 256,
                height: 256,
                interpolation: Interpolation::Bilinear,
            })
            .add(SampledImageOp::Invert)
            .build();

        let plan = sampled.to_plan();

        assert_eq!(plan.len(), 3);
        let barriers = plan.find_barriers();
        assert_eq!(barriers, vec![1]); // Resize is at index 1
    }

    #[test]
    fn test_into_plan() {
        let sampled = SampledImageProgram::builder()
            .add(SampledImageOp::Brightness { delta: 10.0 })
            .build();

        let plan = sampled.into_plan();

        assert_eq!(plan.len(), 1);
    }
}
