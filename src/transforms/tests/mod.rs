mod correctness;
#[test]
fn test_gaussian_through_optimizer() {
    use crate::core::FusableImage;
    use crate::exec_ir::Optimizer;
    use crate::sampled_ir::ops::SampledImageOp;
    use crate::sampled_ir::SampledImageProgram;

    let (w, h) = (8usize, 8usize);
    let mut data: Vec<u8> = (0..w * h * 3).map(|i| i as u8).collect();
    let original = data.clone();

    let mut program = SampledImageProgram::new();
    program.push(SampledImageOp::GaussianBlur { kernel_size: 3, sigma: 0.0 });
    let plan = program.to_plan();
    let exec_plan = Optimizer::new().optimize(plan);
    let mut img = FusableImage::new(&mut data, w, h, 3);
    exec_plan.execute(&mut img);

    eprintln!("optimized nodes: {}", exec_plan.len());
    let shift = (0..data.len()).filter(|&i| data[i] != original[i]).count();
    eprintln!("changed bytes: {}/{}", shift, data.len());
    eprintln!("out[0..6] = {:?}", &data[0..6]);
    eprintln!("in [0..6] = {:?}", &original[0..6]);
}
#[test]
fn test_gaussian_through_optimizer2() {
    use crate::core::{Executable, FusableImage};
    use crate::exec_ir::Optimizer;
    use crate::sampled_ir::ops::SampledImageOp;
    use crate::sampled_ir::SampledImageProgram;
    use crate::transforms::kernel::gaussian_blur::{GaussianBlur, KernelSize};

    let (w, h) = (8usize, 8usize);
    let mut d1: Vec<u8> = (0..w * h * 3).map(|i| i as u8).collect();
    let mut d2 = d1.clone();

    // Direct execution
    let mut img1 = FusableImage::new(&mut d1, w, h, 3);
    GaussianBlur::with_kernel_size(KernelSize::Size3).execute(&mut img1);

    // Optimizer path
    let mut program = SampledImageProgram::new();
    program.push(SampledImageOp::GaussianBlur { kernel_size: 3, sigma: 0.0 });
    let plan = program.to_plan();
    let exec_plan = Optimizer::new().optimize(plan);
    let mut img2 = FusableImage::new(&mut d2, w, h, 3);
    exec_plan.execute(&mut img2);

    let mm = d1.iter().zip(d2.iter()).filter(|(a, b)| a != b).count();
    eprintln!("direct vs optimizer: {} differing bytes of {}", mm, d1.len());
    eprintln!("direct   out[0..6] = {:?}", &d1[0..6]);
    eprintln!("optimize out[0..6] = {:?}", &d2[0..6]);
    eprintln!("input    in [0..6] = {:?}", &(0..w*h*3).map(|i| i as u8).collect::<Vec<u8>>()[0..6]);
    assert_eq!(d1, d2, "optimizer path differs from direct: {} bytes", mm);
}
