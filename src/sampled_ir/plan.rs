// Transform IR (Intermediate Representation)
//
// This module defines the intermediate representation used after
// parameter sampling and before optimization.
//
// This is the input to the optimizer/planner.

use crate::sampled_ir::SampledImageOp;

/// A Plan is the IR representation of a Compose
///
/// After parameter sampling, a Compose becomes a Plan.
/// This is the input to the optimizer.
///
/// # Example
///
/// A Compose with transforms:
///
/// ```text
/// Compose([
///     Brightness(delta=0.1),
///     Contrast(factor=1.2),
///     Resize(256, 256),
/// ])
/// ```
///
/// Becomes a Plan with ops:
///
/// ```text
/// Plan([
///     Brightness,
///     Contrast,
///     Resize,
/// ])
/// ```
#[derive(Debug, Clone)]
pub struct Plan {
    /// Ordered sequence of transform operations
    pub ops: Vec<SampledImageOp>,
}

impl Plan {
    /// Create a new empty Plan
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Create a Plan from a vector of operations
    pub fn from_ops(ops: Vec<SampledImageOp>) -> Self {
        Self { ops }
    }

    /// Add an operation to the plan
    pub fn add_op(&mut self, op: SampledImageOp) {
        self.ops.push(op);
    }

    /// Number of operations in the plan
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Is the plan empty?
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Get an iterator over the operations
    pub fn iter(&self) -> impl Iterator<Item = &SampledImageOp> {
        self.ops.iter()
    }

    /// Analyze the plan for fusion opportunities
    ///
    /// Returns the number of consecutive fuseable ops from the start
    pub fn count_leading_fuseable(&self) -> usize {
        self.ops.iter().take_while(|op| op.is_fuseable()).count()
    }

    /// Find all fusion barriers in the plan
    ///
    /// Returns indices of operations that break fusion chains
    pub fn find_barriers(&self) -> Vec<usize> {
        self.ops
            .iter()
            .enumerate()
            .filter(|(_, op)| !op.is_fuseable())
            .map(|(i, _)| i)
            .collect()
    }
}

impl Default for Plan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampled_ir::ops::{Interpolation, RotateAngle, SampledImageOp};

    #[test]
    fn test_plan_creation() {
        let plan = Plan::new();
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn test_plan_add_ops() {
        let mut plan = Plan::new();
        plan.add_op(SampledImageOp::Brightness { delta: 10.0 });
        plan.add_op(SampledImageOp::Resize {
            width: 100,
            height: 100,
            interpolation: Interpolation::Nearest,
        });

        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());
    }

    #[test]
    fn test_count_leading_fuseable() {
        let mut plan = Plan::new();

        // Empty plan
        assert_eq!(plan.count_leading_fuseable(), 0);

        // All fuseable
        plan.add_op(SampledImageOp::Brightness { delta: 10.0 });
        plan.add_op(SampledImageOp::Brightness { delta: 20.0 });
        plan.add_op(SampledImageOp::Brightness { delta: 30.0 });
        assert_eq!(plan.count_leading_fuseable(), 3);

        // With barrier
        let mut plan2 = Plan::new();
        plan2.add_op(SampledImageOp::Brightness { delta: 10.0 });
        plan2.add_op(SampledImageOp::Brightness { delta: 20.0 });
        plan2.add_op(SampledImageOp::Resize {
            width: 100,
            height: 100,
            interpolation: Interpolation::Nearest,
        });
        plan2.add_op(SampledImageOp::Brightness { delta: 30.0 });
        assert_eq!(plan2.count_leading_fuseable(), 2);
    }

    #[test]
    fn test_find_barriers() {
        let mut plan = Plan::new();
        plan.add_op(SampledImageOp::Brightness { delta: 10.0 });
        plan.add_op(SampledImageOp::Resize {
            width: 100,
            height: 100,
            interpolation: Interpolation::Nearest,
        });
        plan.add_op(SampledImageOp::Brightness { delta: 20.0 });
        plan.add_op(SampledImageOp::Resize {
            width: 200,
            height: 200,
            interpolation: Interpolation::Nearest,
        });

        let barriers = plan.find_barriers();
        assert_eq!(barriers, vec![1, 3]);
    }

    #[test]
    fn test_op_is_fuseable() {
        let fuseable = SampledImageOp::Brightness { delta: 10.0 };
        let barrier = SampledImageOp::Resize {
            width: 100,
            height: 100,
            interpolation: Interpolation::Nearest,
        };

        assert!(fuseable.is_fuseable());
        assert!(!barrier.is_fuseable());
    }

    #[test]
    fn test_from_ops() {
        let plan = Plan::from_ops(vec![
            SampledImageOp::Brightness { delta: 10.0 },
            SampledImageOp::Brightness { delta: 20.0 },
            SampledImageOp::Resize {
                width: 100,
                height: 100,
                interpolation: Interpolation::Nearest,
            },
        ]);

        assert_eq!(plan.len(), 3);
        assert_eq!(plan.count_leading_fuseable(), 2);
    }
}
