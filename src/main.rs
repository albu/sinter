// Sinter - Compiled Image Augmentation Engine
// Step 1-4: Full Pipeline Demo with ExecPlan Executor

use sinter::{
    FusableImage, Plan, Optimizer,
    SampledImageOp,
};

fn main() {
    println!("Sinter - Steps 1-4: Full Pipeline Demo");
    println!("================================================");
    println!();

    // Step 1: Create an image
    println!("Step 1: Create Image");
    println!("  Creating 4x4 grayscale image with value 128...");
    let mut data = vec![128u8; 4 * 4 * 1];
    let mut img = FusableImage::new(&mut data, 4, 4, 1);
    println!("  Created: {}x{}x{} ({} pixels)", img.width, img.height, img.channels, img.pixel_count());
    println!();

    // Step 2: Build a Plan with transforms
    println!("Step 2: Build Plan with Transforms");
    let mut plan = Plan::new();
    plan.add_op(SampledImageOp::Brightness { delta: 32.0 });
    plan.add_op(SampledImageOp::Contrast { factor: 1.5 });
    println!("  Plan: Brightness(+32) → Contrast(1.5)");
    println!("  Transforms: {}", plan.len());
    println!();

    // Step 3: Optimize - fuse the transforms
    println!("Step 3: Optimize (Fusion)");
    let mut optimizer = Optimizer::new();
    let exec_plan = optimizer.optimize(plan);
    println!("  Execution nodes: {}", exec_plan.len());
    println!("  Fused transforms: {}", exec_plan.fused_transform_count());
    println!("  Barriers: {}", exec_plan.barrier_count());
    println!("  Fusion ratio: {:.2}x", exec_plan.fusion_ratio(2));
    println!();

    // Step 4: Execute the optimized plan
    println!("Step 4: Execute Optimized Plan");
    println!("  Original pixel value: 128");
    let _owned = exec_plan.execute(&mut img);
    println!("  After execution: {} (first pixel)", img.data[0]);
    println!();

    println!("Summary:");
    println!("  ✓ Single-pass execution (no intermediate buffers)");
    println!("  ✓ Zero-copy in-place mutation");
    println!("  ✓ All transforms fused into one loop");
    println!();

    // Demonstrate with a flip (barrier)
    println!("Bonus: With Horizontal Flip");
    let mut data2 = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let mut img2 = FusableImage::new(&mut data2, 4, 2, 1);
    println!("  Before: {:?}", img2.data);

    let mut plan2 = Plan::new();
    plan2.add_op(SampledImageOp::Brightness { delta: 10.0 });
    plan2.add_op(SampledImageOp::HorizontalFlip);
    plan2.add_op(SampledImageOp::Contrast { factor: 1.2 });

    let exec_plan2 = optimizer.optimize(plan2);
    let _owned2 = exec_plan2.execute(&mut img2);
    println!("  After Brightness(+10) → HFlip → Contrast(1.2): {:?}", img2.data);
    println!();

    println!("Run `cargo test` for all tests.");
}
