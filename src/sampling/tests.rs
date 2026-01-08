// Tests for sampling module

use super::distributions::{Bernoulli, Dist, Uniform};
use super::sampling_nodes::{RandomImageNode, RandomImageProgram, SamplingContext};
use super::traits::Rng;
use crate::core::{AccessPattern, ShapeEffect};
use crate::sampled_ir::SampledImageOp;

// Simple test RNG that returns predictable values
struct TestRng {
    value: f32,
}

impl Rng for TestRng {
    fn random_f32(&mut self) -> f32 {
        let v = self.value;
        self.value += 0.1;
        v
    }

    fn random_i32(&mut self, upper: i32) -> i32 {
        let v = (self.value as i32) % upper;
        self.value += 1.0;
        v.max(0)
    }
}

#[test]
fn test_bernoulli_true() {
    let mut rng = TestRng { value: 0.0 };
    let bernoulli = Bernoulli::new(0.5);
    assert!(bernoulli.sample(&mut rng));
}

#[test]
fn test_bernoulli_false() {
    let mut rng = TestRng { value: 0.9 };
    let bernoulli = Bernoulli::new(0.5);
    assert!(!bernoulli.sample(&mut rng));
}

#[test]
fn test_uniform() {
    let mut rng = TestRng { value: 0.0 };
    let uniform = Uniform::new(-10.0, 10.0);
    assert_eq!(uniform.sample(&mut rng), -10.0);
}

#[test]
fn test_brightness_sampling() {
    // Leaf transforms now ALWAYS emit (no p field)
    let mut rng = TestRng { value: 0.5 };
    let mut ctx = SamplingContext::new(&mut rng, 0, 0);
    let node = RandomImageNode::Brightness {
        delta: Dist::uniform(-20.0, 20.0),
    };

    let mut ops = Vec::new();
    node.sample(&mut ctx, &mut ops);

    assert_eq!(ops.len(), 1);
    match &ops[0] {
        SampledImageOp::Brightness { delta } => {
            assert!(*delta >= -20.0 && *delta <= 20.0);
        }
        _ => panic!("Expected Brightness"),
    }
}

#[test]
fn test_maybe_enabled() {
    // Activation is now via Maybe wrapper
    let mut rng = TestRng { value: 0.0 };
    let mut ctx = SamplingContext::new(&mut rng, 0, 0);
    let node = RandomImageNode::Maybe {
        child: Box::new(RandomImageNode::Invert),
        p: Dist::bernoulli(0.5),
    };

    let mut ops = Vec::new();
    node.sample(&mut ctx, &mut ops);

    assert_eq!(ops.len(), 1);
}

#[test]
fn test_maybe_disabled() {
    let mut rng = TestRng { value: 0.9 };
    let mut ctx = SamplingContext::new(&mut rng, 0, 0);
    let node = RandomImageNode::Maybe {
        child: Box::new(RandomImageNode::Invert),
        p: Dist::bernoulli(0.5),
    };

    let mut ops = Vec::new();
    node.sample(&mut ctx, &mut ops);

    assert_eq!(ops.len(), 0);
}

#[test]
fn test_sampling_nodes_program_sampling() {
    let mut program = RandomImageProgram::new();
    // Invert with always-active p (Constant(1.0)) -> no Maybe wrapper
    program.add(RandomImageNode::Invert);
    // Invert with never-active p (Constant(0.0)) -> Maybe wrapper that skips
    program.add(RandomImageNode::Maybe {
        child: Box::new(RandomImageNode::Invert),
        p: Dist::constant(0.0),
    });

    let sampled = program.sample_with_seed(42);
    // Only the first Invert should apply
    assert_eq!(sampled.ops.len(), 1);
}

#[test]
fn test_sampling_nodes_node_all() {
    // Sequential is now All
    let node = RandomImageNode::All {
        children: vec![
            RandomImageNode::Maybe {
                child: Box::new(RandomImageNode::Invert),
                p: Dist::bernoulli(0.5),
            },
            RandomImageNode::Brightness {
                delta: Dist::uniform(-10.0, 10.0),
            },
        ],
    };

    let mut rng = TestRng { value: 0.0 };
    let mut ctx = SamplingContext::new(&mut rng, 0, 0);
    let mut ops = Vec::new();
    node.sample(&mut ctx, &mut ops);

    // Invert with p=0.5 and rng.value=0.0 should apply
    // Brightness always applies (no Maybe wrapper)
    assert_eq!(ops.len(), 2);
}
