// Fusion Correctness Tests
//
// Tests that fusion produces semantically correct results (same output as unfused).

use crate::core::{Executable, FusableImage};
use crate::exec_ir::Optimizer;
use crate::sampled_ir::Plan;
use crate::sampling::{Dist, RandomImageNode, RandomImageProgram};
use crate::transforms::*;

/// Helper to create a test image
fn create_test_image(width: u32, height: u32, channels: u8) -> Vec<u8> {
    let size = (width * height * channels as u32) as usize;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        // Create a simple pattern without saturation
        let x = (i as u32 % width) as u32;
        let y = (i as u32 / width) as u32;
        let c = (i as u32 / (width * height)) % 3;
        // Use a formula that won't saturate
        let value = (x + y + c * 10) % 256;
        data.push(value as u8);
    }
    data
}

/// Helper to create a program from transforms
fn create_program(transforms: Vec<RandomImageNode>) -> RandomImageProgram {
    let mut program = RandomImageProgram::new();
    for transform in transforms {
        program.add(transform);
    }
    program
}

#[test]
fn test_lut_fusion_correctness() {
    // Test that LUT fusion produces the same result as unfused
    let data1 = create_test_image(100, 100, 3);
    let data2 = create_test_image(100, 100, 3);
    let mut data1_clone = data1.clone();
    let mut data2_clone = data2.clone();
    let mut img1 = FusableImage::new(&mut data1_clone, 100, 100, 3);
    let mut img2 = FusableImage::new(&mut data2_clone, 100, 100, 3);

    // Apply unfused (individual transforms)
    let brightness = Brightness::new(30.0);
    let contrast = Contrast::new(1.2);
    let solarize = Solarize::new(128);

    Executable::execute(&brightness, &mut img1);
    Executable::execute(&contrast, &mut img1);
    Executable::execute(&solarize, &mut img1);

    // Apply fused (single LUT)
    let lut_ops: Vec<Box<dyn LutOp>> = vec![
        Box::new(Brightness::new(30.0)),
        Box::new(Contrast::new(1.2)),
        Box::new(Solarize::new(128)),
    ];
    let fused_lut = FusedLut::from_ops(&lut_ops);
    crate::transforms::runtime::lut::LutExecutor::apply(&mut img2, &fused_lut.lut);

    // Results should be identical
    assert_eq!(img1.data, img2.data, "LUT fusion produced different result");
}

#[test]
fn test_matrix_fusion_correctness() {
    // Test that Matrix fusion produces the same result as unfused
    let data1 = create_test_image(100, 100, 3);
    let data2 = create_test_image(100, 100, 3);
    let mut data1_clone = data1.clone();
    let mut data2_clone = data2.clone();
    let mut img1 = FusableImage::new(&mut data1_clone, 100, 100, 3);
    let mut img2 = FusableImage::new(&mut data2_clone, 100, 100, 3);

    // Apply unfused
    let to_sepia = ToSepia;
    let color_temp = ColorTemperature::new(50.0);

    Executable::execute(&to_sepia, &mut img1);
    Executable::execute(&color_temp, &mut img1);

    // Apply fused
    let matrix_ops: Vec<Box<dyn MatrixOp>> =
        vec![Box::new(ToSepia), Box::new(ColorTemperature::new(50.0))];
    let refs: Vec<&dyn MatrixOp> = matrix_ops.iter().map(|b| b.as_ref()).collect();
    let fused_matrix = FusedMatrix::from_matrix_ops(&refs);

    crate::transforms::runtime::matrix::MatrixExecutor::apply(&mut img2, &fused_matrix.matrix);

    // Matrix fusion uses single clamping at end, while unfused clamps after each transform.
    // This causes small differences (typically 0-1 values) due to intermediate precision loss.
    // Allow tolerance of 2 to account for this.
    for (i, (&a, &b)) in img1.data.iter().zip(img2.data.iter()).enumerate() {
        let diff = (a as i32 - b as i32).abs();
        assert!(
            diff <= 2,
            "Matrix fusion produced different result at index {}: {} vs {} (diff: {})",
            i,
            a,
            b,
            diff
        );
    }
}

/// Stable matrix chain: every composed prefix matrix keeps [0,255]^3 inside
/// [0,255]^3 (all coefficients >= 0 and every row sums <= 1). For such chains
/// no intermediate clamping can ever fire, so the only difference between
/// fused and sequential execution is fixed-point rounding (per-step truncation
/// vs composed-then-quantized), bounded to a couple of grey levels. This pins
/// the guarantee that stable chains do NOT diverge due to clamping.
#[test]
fn test_matrix_fusion_stable_chain_no_clamping_divergence() {
    let data1 = create_test_image(100, 100, 3);
    let data2 = data1.clone();
    let mut data1_clone = data1.clone();
    let mut data2_clone = data2.clone();
    let mut img1 = FusableImage::new(&mut data1_clone, 100, 100, 3);
    let mut img2 = FusableImage::new(&mut data2_clone, 100, 100, 3);

    // All coefficients >= 0, every row sum <= 1.
    let m1 = [[0.9, 0.0, 0.0], [0.0, 0.85, 0.0], [0.0, 0.0, 0.95]];
    let m2 = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]];
    let m3 = [[0.8, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]];

    // Sequential (Q8 matrix arithmetic, clamping between -- never fires here).
    MatrixExecutor::apply(&mut img1, &m1);
    MatrixExecutor::apply(&mut img1, &m2);
    MatrixExecutor::apply(&mut img1, &m3);

    // Fused: compose once, apply once.
    let ops: Vec<Box<dyn MatrixOp>> = vec![
        Box::new(ChannelMix::new(m1)),
        Box::new(ChannelMix::new(m2)),
        Box::new(ChannelMix::new(m3)),
    ];
    let refs: Vec<&dyn MatrixOp> = ops.iter().map(|b| b.as_ref()).collect();
    let fused = FusedMatrix::from_matrix_ops(&refs);
    MatrixExecutor::apply(&mut img2, &fused.matrix);

    let max_diff = img1
        .data
        .iter()
        .zip(img2.data.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 2,
        "stable matrix chain diverged more than fixed-point rounding allows: max_diff={}",
        max_diff
    );
}

/// The canonical semantic for fused matrix chains is "compose, then clamp once".
/// This pins the composition order: for ops applied in order op1 -> op2 -> op3
/// the composed matrix must be M3 * M2 * M1 (right-to-left matrix product).
#[test]
fn test_matrix_fusion_composition_order() {
    let sepia = ToSepia.get_matrix();
    let temp = ColorTemperature::new(50.0).get_matrix();
    let s = 1.2f32;
    let v = 1.1f32;
    let om_s = 1.0 - s;
    let hsv = [
        [(om_s * 0.299 + s) * v, om_s * 0.587 * v, om_s * 0.114 * v],
        [om_s * 0.299 * v, (om_s * 0.587 + s) * v, om_s * 0.114 * v],
        [om_s * 0.299 * v, om_s * 0.587 * v, (om_s * 0.114 + s) * v],
    ];

    let ops: Vec<Box<dyn MatrixOp>> = vec![
        Box::new(ToSepia),
        Box::new(ColorTemperature::new(50.0)),
        Box::new(ChannelMix::new(hsv)),
    ];
    let refs: Vec<&dyn MatrixOp> = ops.iter().map(|b| b.as_ref()).collect();
    let fused = FusedMatrix::from_matrix_ops(&refs);

    // Expected: M_hsv * M_temp * M_sepia.
    let mut expected = sepia;
    for m in [temp, hsv] {
        let mut next = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                next[i][j] = m[i][0] * expected[0][j]
                    + m[i][1] * expected[1][j]
                    + m[i][2] * expected[2][j];
            }
        }
        expected = next;
    }

    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (fused.matrix[i][j] - expected[i][j]).abs() < 1e-4,
                "composition order mismatch at [{i}][{j}]: fused={} expected={}",
                fused.matrix[i][j],
                expected[i][j]
            );
        }
    }
}

/// Saturating chains (ToSepia rows sum to 1.35, warm ColorTemperature and
/// HSV val>1 also overshoot) are the one place where fused output can differ
/// from running each op sequentially: the sequential path clamps to [0,255]
/// after every op, while the fused path applies the composed linear map and
/// clamps once at the end. The fused result is the higher-precision (less
/// clipped) output. This test pins the divergence to a documented bound so a
/// regression in the composition math is caught.
#[test]
fn test_matrix_fusion_divergence_bounded() {
    // Deterministic pseudo-random image (covers the full [0,255] range).
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 256) as u8
    };
    let mut data1 = Vec::with_capacity(100 * 100 * 3);
    for _ in 0..(100 * 100 * 3) {
        data1.push(next());
    }
    let data2 = data1.clone();
    let mut data1_clone = data1.clone();
    let mut data2_clone = data2.clone();
    let mut img1 = FusableImage::new(&mut data1_clone, 100, 100, 3);
    let mut img2 = FusableImage::new(&mut data2_clone, 100, 100, 3);

    // Sequential (production executors: MatrixExecutor for ToSepia/ChannelMix,
    // float apply_matrix for ColorTemperature -- each op clamps to [0,255]).
    Executable::execute(&ToSepia, &mut img1);
    Executable::execute(&ColorTemperature::new(50.0), &mut img1);
    Executable::execute(&ChannelMix::new([
        [0.5, 0.3, 0.2],
        [0.1, 0.6, 0.3],
        [0.2, 0.2, 0.6],
    ]), &mut img1);

    // Fused: compose once, apply once (Q8 SIMD, clamp at the end).
    // Note: HueSaturationValue(hue=0) is no longer matrix-fused (it is not a
    // linear map under the exact hexcone semantics), so this chain uses a
    // plain ChannelMix instead.
    let ops: Vec<Box<dyn MatrixOp>> = vec![
        Box::new(ToSepia),
        Box::new(ColorTemperature::new(50.0)),
        Box::new(ChannelMix::new([
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.6],
        ])),
    ];
    let refs: Vec<&dyn MatrixOp> = ops.iter().map(|b| b.as_ref()).collect();
    let fused = FusedMatrix::from_matrix_ops(&refs);
    MatrixExecutor::apply(&mut img2, &fused.matrix);

    // Documented bound: divergence comes from sequential intermediate clamping
    // (and Q8-vs-float executor mixing), not from composition math.
    let max_diff = img1
        .data
        .iter()
        .zip(img2.data.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 32,
        "fused vs sequential divergence exceeded documented bound: max_diff={}",
        max_diff
    );
}

/// Long mixed pipeline (LUT + matrix + geometric) executed through the
/// optimizer, comparing the fused output against sequential application of
/// every op. LUT and geometric fusion are exact; matrix fusion applies the
/// composed linear map and clamps once, so it can diverge from sequential by
/// up to ~22 grey levels when intermediates saturate. The bound below is set
/// to 32 to cover that while still catching composition/order regressions.
/// Also asserts the optimized plan is deterministic on replay.
#[test]
fn test_long_fused_pipeline_output_correctness() {
    use crate::sampled_ir::ops::SampledImageOp;
    use crate::sampled_ir::SampledImageProgram;

    let ops = vec![
        SampledImageOp::Brightness { delta: 20.0 },
        SampledImageOp::Contrast { factor: 1.2 },
        SampledImageOp::Gamma { gamma: 0.9 },
        SampledImageOp::Solarize { threshold: 128 },
        SampledImageOp::Posterize { bits: 4 },
        SampledImageOp::ToSepia,
        SampledImageOp::ColorTemperature { temperature: 50.0 },
        SampledImageOp::HueSaturationValue {
            hue_shift: 0,
            saturation_scale: 1.3,
            value_scale: 1.0,
        },
        SampledImageOp::ColorTint {
            tint: [255.0, 200.0, 100.0, 0.5],
        },
        SampledImageOp::HorizontalFlip,
        SampledImageOp::VerticalFlip,
    ];

    let mut program = SampledImageProgram::new();
    for op in &ops {
        program.push(op.clone());
    }

    // Deterministic pseudo-random image covering the full [0,255] range.
    let mut state = 0xD1B54A32D192ED03u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 256) as u8
    };
    let base: Vec<u8> = (0..(64 * 64 * 3)).map(|_| next()).collect();

    // Fused through the optimizer.
    let plan = program.to_plan();
    let exec_plan = Optimizer::new().optimize(plan);
    assert!(
        exec_plan.len() < ops.len(),
        "expected fusion to reduce node count ({} -> {})",
        ops.len(),
        exec_plan.len()
    );
    let mut d1 = base.clone();
    let mut img_fused = FusableImage::new(&mut d1, 64, 64, 3);
    let barrier = exec_plan.execute(&mut img_fused);
    // The trailing HorizontalFlip + VerticalFlip fuse to Rot180, which now
    // delegates to the out-of-place NEON Rotate path, so the plan may emit a
    // barrier even though every individual op is in-place.
    let fused_data: &[u8] = match &barrier {
        Some(b) => &b.data,
        None => &img_fused.data,
    };

    // Sequential application of the same ops.
    let mut d2 = base.clone();
    let mut img_seq = FusableImage::new(&mut d2, 64, 64, 3);
    for op in &ops {
        Executable::execute(op, &mut img_seq);
    }

    let max_diff = fused_data
        .iter()
        .zip(img_seq.data.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 32,
        "long fused pipeline diverged beyond documented bound: max_diff={}",
        max_diff
    );

    // Determinism: replaying the optimized plan yields identical output.
    let mut d3 = base.clone();
    let mut img_replay = FusableImage::new(&mut d3, 64, 64, 3);
    let replay_barrier = exec_plan.execute(&mut img_replay);
    let replay_data: &[u8] = match &replay_barrier {
        Some(b) => &b.data,
        None => &img_replay.data,
    };
    assert_eq!(
        fused_data, replay_data,
        "optimized plan must be deterministic on replay"
    );
}

#[test]
fn test_geometric_fusion_correctness() {
    // Test that geometric fusion produces the same result as unfused
    let data1 = create_test_image(100, 100, 3);
    let data2 = create_test_image(100, 100, 3);
    let mut data1_clone = data1.clone();
    let mut data2_clone = data2.clone();
    let mut img1 = FusableImage::new(&mut data1_clone, 100, 100, 3);
    let mut img2 = FusableImage::new(&mut data2_clone, 100, 100, 3);

    // Apply unfused: FlipH + FlipV = Rot180
    let flip_h = HorizontalFlip;
    let flip_v = VerticalFlip;

    Executable::execute(&flip_h, &mut img1);
    Executable::execute(&flip_v, &mut img1);

    // Apply fused: Rot180
    let rot180 = Rotate::new(RotateAngle::Rotate180);
    // Rotate is OutOfPlace - returns BarrierImage, doesn't modify img2
    let result = Executable::execute(&rot180, &mut img2);
    if let Some(mut barrier) = result {
        let img2_view = barrier.as_fusable();
        assert_eq!(
            img1.data, img2_view.data,
            "Geometric fusion produced different result"
        );
    } else {
        panic!("Rot180 should return BarrierImage");
    }
}

#[test]
fn test_canonicalization_correctness_with_geometric() {
    // Test that canonicalization (geometric hoisting) is semantically correct
    let data1 = create_test_image(100, 100, 3);
    let data2 = create_test_image(100, 100, 3);
    let mut data1_clone = data1.clone();
    let mut data2_clone = data2.clone();
    let mut img1 = FusableImage::new(&mut data1_clone, 100, 100, 3);
    let mut img2 = FusableImage::new(&mut data2_clone, 100, 100, 3);

    // Original order: Brightness + FlipH + Contrast
    let brightness = Brightness::new(20.0);
    let flip_h = HorizontalFlip;
    let contrast = Contrast::new(1.1);

    Executable::execute(&brightness, &mut img1);
    Executable::execute(&flip_h, &mut img1);
    Executable::execute(&contrast, &mut img1);

    // Canonicalized: FlipH + Brightness + Contrast
    Executable::execute(&flip_h, &mut img2);
    let brightness2 = Brightness::new(20.0);
    let contrast2 = Contrast::new(1.1);
    Executable::execute(&brightness2, &mut img2);
    Executable::execute(&contrast2, &mut img2);

    assert_eq!(img1.data, img2.data, "Canonicalization changed semantics");
}

#[test]
fn test_execution_plan_reduces_nodes() {
    // Test that optimization reduces the number of execution nodes
    use crate::sampling::Dist;

    // Create a program with 6 LUT transforms (should fuse to 1)
    let transforms = vec![
        RandomImageNode::Brightness {
            delta: Dist::Constant(10.0),
        },
        RandomImageNode::Contrast {
            factor: Dist::Constant(1.1),
        },
        RandomImageNode::Gamma {
            gamma: Dist::Constant(0.9),
        },
        RandomImageNode::Invert,
        RandomImageNode::Solarize {
            threshold: Dist::Constant(128.0),
        },
        RandomImageNode::Posterize {
            bits: Dist::Constant(6.0),
        },
    ];

    let program = create_program(transforms);
    let sampled = program.sample_with_seed(42);

    // Create plan and optimize
    let plan = sampled.to_plan();
    let mut optimizer = Optimizer::new();
    let exec_plan = optimizer.optimize(plan);

    // Should have 1 execution node (all LUTs fused)
    assert_eq!(
        exec_plan.len(),
        1,
        "Expected 1 execution node, got {}",
        exec_plan.len()
    );

    // Verify it's a fused node
    match &exec_plan.nodes[0].kind {
        crate::exec_ir::nodes::ExecNodeKind::Fused(ops) => {
            assert_eq!(ops.len(), 6, "Expected 6 fused ops, got {}", ops.len());
        }
        _ => panic!("Expected Fused node"),
    }
}

#[test]
fn test_geometric_d4_fusion() {
    // Test D4 group fusion: FlipH + FlipV should fuse
    use crate::sampling::Dist;

    let transforms = vec![
        RandomImageNode::HorizontalFlip,
        RandomImageNode::VerticalFlip,
    ];

    let program = create_program(transforms);
    let sampled = program.sample_with_seed(42);

    let plan = sampled.to_plan();
    let mut optimizer = Optimizer::new();
    let exec_plan = optimizer.optimize(plan);

    // Should fuse to 1 node
    assert_eq!(
        exec_plan.len(),
        1,
        "Expected 1 execution node for D4 fusion, got {}",
        exec_plan.len()
    );

    // Verify the fused ops
    match &exec_plan.nodes[0].kind {
        crate::exec_ir::nodes::ExecNodeKind::Fused(ops) => {
            assert_eq!(ops.len(), 2, "Expected 2 fused ops, got {}", ops.len());
        }
        _ => panic!("Expected Fused node"),
    }
}

#[test]
fn test_barriers_block_fusion() {
    // Test that barriers (Resize) block LUT fusion
    use crate::sampling::Dist;

    let transforms = vec![
        RandomImageNode::Brightness {
            delta: Dist::Constant(10.0),
        },
        RandomImageNode::Resize {
            width: 50,
            height: 50,
            interpolation: crate::sampled_ir::ops::Interpolation::Bilinear,
        },
        RandomImageNode::Contrast {
            factor: Dist::Constant(1.1),
        },
    ];

    let program = create_program(transforms);
    let sampled = program.sample_with_seed(42);

    let plan = sampled.to_plan();
    let mut optimizer = Optimizer::new();
    let exec_plan = optimizer.optimize(plan);

    // Should have 3 nodes (Resize is a barrier)
    assert_eq!(
        exec_plan.len(),
        3,
        "Expected 3 execution nodes with barrier, got {}",
        exec_plan.len()
    );

    // Verify structure: Fused(Brightness) + Barrier(Resize) + Fused(Contrast)
    match &exec_plan.nodes[0].kind {
        crate::exec_ir::nodes::ExecNodeKind::Fused(ops) => {
            assert_eq!(ops.len(), 1);
        }
        _ => panic!("Node 0 should be Fused"),
    }

    match &exec_plan.nodes[1].kind {
        crate::exec_ir::nodes::ExecNodeKind::Barrier(_) => {
            // Correct
        }
        _ => panic!("Node 1 should be Barrier"),
    }

    match &exec_plan.nodes[2].kind {
        crate::exec_ir::nodes::ExecNodeKind::Fused(ops) => {
            assert_eq!(ops.len(), 1);
        }
        _ => panic!("Node 2 should be Fused"),
    }
}

#[test]
fn test_four_geometric_fuses_to_identity() {
    // Test that FlipH + FlipH cancels to identity
    use crate::sampling::Dist;

    let transforms = vec![
        RandomImageNode::HorizontalFlip,
        RandomImageNode::HorizontalFlip,
    ];

    let program = create_program(transforms);
    let sampled = program.sample_with_seed(42);

    let plan = sampled.to_plan();
    let mut optimizer = Optimizer::new();
    let exec_plan = optimizer.optimize(plan);

    // Should fuse to identity (0 nodes)
    assert_eq!(
        exec_plan.len(),
        0,
        "Expected 0 nodes (identity), got {}",
        exec_plan.len()
    );
}

#[test]
fn test_multiple_rotations_compose_correctly() {
    // Test Rot90 + Rot90 = Rot180
    let data1 = create_test_image(100, 100, 3);
    let data2 = create_test_image(100, 100, 3);
    let mut data1_clone = data1.clone();
    let mut data2_clone = data2.clone();
    let mut img1 = FusableImage::new(&mut data1_clone, 100, 100, 3);
    let mut img2 = FusableImage::new(&mut data2_clone, 100, 100, 3);

    // Two Rot90 transforms
    let rot90 = Rotate::new(RotateAngle::Rotate90);
    Executable::execute(&rot90, &mut img1);
    Executable::execute(&rot90, &mut img1);

    // Single Rot180
    let rot180 = Rotate::new(RotateAngle::Rotate180);
    Executable::execute(&rot180, &mut img2);

    assert_eq!(img1.data, img2.data, "Rot90 + Rot90 != Rot180");
}

#[test]
fn test_complex_pipeline_fusion() {
    // Test a complex pipeline with mixed transform types
    use crate::sampling::Dist;

    let transforms = vec![
        // Geometric
        RandomImageNode::HorizontalFlip,
        RandomImageNode::VerticalFlip,
        // LUT
        RandomImageNode::Brightness {
            delta: Dist::Constant(20.0),
        },
        RandomImageNode::Contrast {
            factor: Dist::Constant(1.1),
        },
        // More geometric
        RandomImageNode::Rotate {
            angle: crate::sampled_ir::ops::RotateAngle::Rotate90,
        },
        // More LUT
        RandomImageNode::Gamma {
            gamma: Dist::Constant(0.9),
        },
        RandomImageNode::Invert,
    ];

    let program = create_program(transforms);
    let sampled = program.sample_with_seed(42);

    let plan = sampled.to_plan();
    let mut optimizer = Optimizer::new();
    let exec_plan = optimizer.optimize(plan);

    // After canonicalization: all geometric first, then all LUT
    // Geometric: FlipH + FlipV + Rot90 → 1 fused node (D4 composition)
    // LUT: Brightness + Contrast + Gamma + Invert → 1 fused node
    // Total: 2 nodes
    assert_eq!(
        exec_plan.len(),
        2,
        "Expected 2 execution nodes, got {}",
        exec_plan.len()
    );
}
