// Tests for RandomImageNode sampling

use super::*;
use crate::sampling::Dist;

// Mock RNG that returns predictable values
struct MockRng {
    f32_value: f32,
    i32_values: Vec<i32>,
    i32_index: usize,
}

impl MockRng {
    fn new(f32_value: f32, i32_values: Vec<i32>) -> Self {
        Self {
            f32_value,
            i32_values,
            i32_index: 0,
        }
    }
}

impl Rng for MockRng {
    fn random_f32(&mut self) -> f32 {
        self.f32_value
    }

    fn random_i32(&mut self, _upper: i32) -> i32 {
        if self.i32_index < self.i32_values.len() {
            let val = self.i32_values[self.i32_index];
            self.i32_index += 1;
            val
        } else {
            0
        }
    }
}

#[test]
fn test_horizontal_flip_always_emits() {
    // Without Maybe, leaf transforms ALWAYS emit
    let node = RandomImageNode::HorizontalFlip;
    let mut rng = MockRng::new(0.5, vec![]);
    let mut ctx = SamplingContext::new(&mut rng, 42, 0);
    let mut out = Vec::new();

    node.sample(&mut ctx, &mut out);

    assert_eq!(out.len(), 1);
    match &out[0] {
        SampledImageOp::HorizontalFlip => {}
        _ => panic!("Expected HorizontalFlip"),
    }
}

#[test]
fn test_maybe_with_low_probability_skips() {
    let node = RandomImageNode::Maybe {
        child: Box::new(RandomImageNode::HorizontalFlip),
        p: Dist::bernoulli(0.1),
    };
    let mut rng = MockRng::new(0.5, vec![]); // 50% > 10% threshold
    let mut ctx = SamplingContext::new(&mut rng, 42, 0);
    let mut out = Vec::new();

    node.sample(&mut ctx, &mut out);

    assert_eq!(out.len(), 0);
}

#[test]
fn test_maybe_with_high_probability_applies() {
    let node = RandomImageNode::Maybe {
        child: Box::new(RandomImageNode::HorizontalFlip),
        p: Dist::bernoulli(0.9),
    };
    let mut rng = MockRng::new(0.3, vec![]); // 30% < 90% threshold
    let mut ctx = SamplingContext::new(&mut rng, 42, 0);
    let mut out = Vec::new();

    node.sample(&mut ctx, &mut out);

    assert_eq!(out.len(), 1);
}

#[test]
fn test_brightness_samples() {
    let node = RandomImageNode::Brightness {
        delta: Dist::uniform(-30.0, 30.0),
    };
    let mut rng = MockRng::new(0.5, vec![]);
    let mut ctx = SamplingContext::new(&mut rng, 42, 0);
    let mut out = Vec::new();

    node.sample(&mut ctx, &mut out);

    assert_eq!(out.len(), 1);
    match &out[0] {
        SampledImageOp::Brightness { delta } => {
            assert!(*delta >= -30.0 && *delta <= 30.0);
        }
        _ => panic!("Expected Brightness"),
    }
}

#[test]
fn test_all_applies_all() {
    let node = RandomImageNode::All {
        children: vec![
            RandomImageNode::Brightness {
                delta: Dist::uniform(-30.0, 30.0),
            },
            RandomImageNode::Contrast {
                factor: Dist::uniform(0.8, 1.2),
            },
        ],
    };
    let mut rng = MockRng::new(0.5, vec![]);
    let mut ctx = SamplingContext::new(&mut rng, 42, 0);
    let mut out = Vec::new();

    node.sample(&mut ctx, &mut out);

    assert_eq!(out.len(), 2);
}

#[test]
fn test_oneof_selects_one() {
    let node = RandomImageNode::OneOf {
        children: vec![
            RandomImageNode::HorizontalFlip,
            RandomImageNode::VerticalFlip,
        ],
    };
    let mut rng = MockRng::new(0.5, vec![0]); // Select index 0
    let mut ctx = SamplingContext::new(&mut rng, 42, 0);
    let mut out = Vec::new();

    node.sample(&mut ctx, &mut out);

    assert_eq!(out.len(), 1);
    match &out[0] {
        SampledImageOp::HorizontalFlip => {}
        _ => panic!("Expected HorizontalFlip"),
    }
}

#[test]
fn test_someof_selects_k() {
    let node = RandomImageNode::SomeOf {
        children: vec![
            RandomImageNode::Invert,
            RandomImageNode::Brightness {
                delta: Dist::uniform(-10.0, 10.0),
            },
            RandomImageNode::Contrast {
                factor: Dist::uniform(0.9, 1.1),
            },
        ],
        n: Dist::constant(2.0),
    };
    let mut rng = MockRng::new(0.5, vec![1, 0]); // k=2, shuffle indices
    let mut ctx = SamplingContext::new(&mut rng, 42, 0);
    let mut out = Vec::new();

    node.sample(&mut ctx, &mut out);

    assert_eq!(out.len(), 2);
}

#[test]
fn test_sampling_nodes_program() {
    let mut program = RandomImageProgram::new();
    program.add(RandomImageNode::Maybe {
        child: Box::new(RandomImageNode::HorizontalFlip),
        p: Dist::bernoulli(0.5),
    });
    program.add(RandomImageNode::Brightness {
        delta: Dist::uniform(-30.0, 30.0),
    });

    let sampled = program.sample_with_seed(42);

    // At least Brightness should be present
    assert!(!sampled.ops.is_empty());
}
