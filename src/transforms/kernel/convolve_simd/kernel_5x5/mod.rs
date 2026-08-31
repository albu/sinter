// 5x5 kernel convolution implementations (Gaussian [1 4 6 4 1] / 16)
//
// Provides both 1D horizontal/vertical passes and separable implementation.

use crate::core::FusableImage;

mod neon;

// Re-export the NEON functions
pub(crate) use neon::{
    convolve_1d_horizontal_neon_5,
    convolve_1d_vertical_neon_5,
    convolve_separable_neon_5,
};

/// A/B experiment: where does the 5x5 RGB gap vs cv2 (~0.80-0.86x) live?
///
/// Variant "fused": the production kernel — H pass into a 5-row L1 ring
/// buffer, V pass emitted straight back into `data` (u8 intermediate, >> 4
/// per pass).
///
/// Variant "twopass": the standalone horizontal + vertical kernels —
/// arithmetically IDENTICAL (same >> 4 roundings), but the intermediate is a
/// full frame that is fully written and fully re-read. If twopass matches
/// fused, the ring layout is vindicated and the gap is in the pass bodies;
/// if twopass wins, the ring is the problem.
///
/// Run with: cargo test --release gauss5_ab --ignored -- --nocapture
#[cfg(all(test, target_arch = "aarch64"))]
mod gauss5_ab {
    use super::*;
    use std::time::Instant;

    fn random_rgb(width: usize, height: usize) -> Vec<u8> {
        let mut data = vec![0u8; width * height * 3];
        let mut x: u64 = 0x9E3779B97F4A7C15;
        for v in data.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *v = (x >> 32) as u8;
        }
        data
    }

    fn bench(name: &str, f: &mut dyn FnMut(), runs: usize) -> f64 {
        // Warmup
        for _ in 0..5 {
            f();
        }
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..runs {
                f();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / runs as f64;
            best = best.min(ms);
        }
        println!("  {name:>10}: {best:.4} ms");
        best
    }

    #[test]
    #[ignore = "manual A/B: cargo test --release gauss5_ab --ignored -- --nocapture"]
    fn gauss5_ab_ring_vs_twopass() {
        for &(w, h) in &[(1024usize, 1024usize), (2048, 2048)] {
            println!("{w}x{h} RGB 5x5 [1 4 6 4 1]:");
            let src = random_rgb(w, h);

            let mut d_fused = src.clone();
            let mut img_fused = FusableImage::new(&mut d_fused, w, h, 3);
            let t_fused = bench("fused", &mut || unsafe {
                convolve_separable_neon_5(&mut img_fused, &[], 16);
            }, 20);

            let mut d_tp = src.clone();
            let mut img_tp = FusableImage::new(&mut d_tp, w, h, 3);
            let kernel = [1i32, 4, 6, 4, 1];
            let t_tp = bench("twopass", &mut || unsafe {
                convolve_1d_horizontal_neon_5(&mut img_tp, &kernel, 16);
                convolve_1d_vertical_neon_5(&mut img_tp, &kernel, 16);
            }, 20);

            assert_eq!(d_fused, d_tp, "variants disagree — comparison invalid");
            println!("  → twopass/fused = {:.3}  (ring {})",
                t_tp / t_fused,
                if t_tp < t_fused { "LOSES" } else { "vindicated" });
        }
    }
}
