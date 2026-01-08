// ARM NEON LUT executor implementations
//
// Provides vqtbl4q_u8 based LUT lookup for ARM platforms.

use crate::core::FusableImage;
use std::arch::aarch64::{
    vld1q_u8, vst1q_u8, vld1q_u8_x4,
    vdupq_n_u8,
    vqtbl4q_u8, vbslq_u8, vcgeq_u8, vsubq_u8, vorrq_u8,
};

/// ARM NEON implementation using vqtbl4q_u8 - the M1 Pro secret weapon
///
/// ARM's vqtbl4q_u8 can lookup into a 64-byte table (4 registers) in ONE instruction.
/// For 256-byte LUT, we do 4 parallel lookups and merge based on index ranges.
///
/// Efficiency: 4 table lookups + 3 compares + 3 selects per 16 pixels (10 instructions)
///
/// Performance: Target ~12-18 GB/s on M1 Pro (4-5x faster than scalar)
#[inline]
pub(crate) unsafe fn apply_neon_vqtbl(image: &mut FusableImage, lut: &[u8; 256]) {
    let data = &mut image.data;
    let len = data.len();

    // Use 4-way interleaved version ONLY for very large images where ILP matters
    // For small/medium images, the 1-way version is faster due to lower overhead
    const INTERLEAVE_THRESHOLD: usize = 768 * 768 * 3;  // Only for 768x768+ images

    if len >= INTERLEAVE_THRESHOLD {
        apply_neon_vqtbl_x4(image, lut);
        return;
    }

    let chunks = len / 16;

    // Pre-load LUT into 4 sets of 4 registers (256 bytes total)
    let lut_ptr = lut.as_ptr();
    let table0 = vld1q_u8_x4(lut_ptr);           // bytes 0-63
    let table1 = vld1q_u8_x4(lut_ptr.add(64));    // bytes 64-127
    let table2 = vld1q_u8_x4(lut_ptr.add(128));   // bytes 128-191
    let table3 = vld1q_u8_x4(lut_ptr.add(192));   // bytes 192-255

    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 input pixels (indices)
        let indices = vld1q_u8(data.as_ptr().add(offset) as *const u8);

        // Perform 4 parallel 64-byte table lookups
        let res0 = vqtbl4q_u8(table0, indices);
        let res1 = vqtbl4q_u8(table1, vsubq_u8(indices, vdupq_n_u8(64)));
        let res2 = vqtbl4q_u8(table2, vsubq_u8(indices, vdupq_n_u8(128)));
        let res3 = vqtbl4q_u8(table3, vsubq_u8(indices, vdupq_n_u8(192)));

        // Create masks for each range
        let mask1 = vcgeq_u8(indices, vdupq_n_u8(64));
        let mask2 = vcgeq_u8(indices, vdupq_n_u8(128));
        let mask3 = vcgeq_u8(indices, vdupq_n_u8(192));

        // Merge results using bitwise select
        let mut result = vbslq_u8(mask1, res1, res0);
        result = vbslq_u8(mask2, res2, result);
        result = vbslq_u8(mask3, res3, result);

        // Store result
        vst1q_u8(data.as_mut_ptr().add(offset) as *mut u8, result);
    }

    // Handle remaining pixels with scalar loop
    for i in (chunks * 16)..len {
        data[i] = lut[data[i] as usize];
    }
}

/// ARM NEON 4-way interleaved implementation - "final boss" version
///
/// Processes 64 pixels per loop iteration to fully saturate the M1 Pro's:
/// - 4 NEON execution units
/// - All 32 NEON registers (using ~28-30)
/// - Hardware prefetcher with cache-line-aligned access
///
/// The key insight: vqtbl4q_u8 has ~3 cycle latency. By having 16 lookups "in flight"
/// (4 batches × 4 phases), we hide this latency completely.
///
/// Expected: 10-15 GB/s (3-4x faster than scalar)
#[inline]
pub(crate) unsafe fn apply_neon_vqtbl_x4(image: &mut FusableImage, lut: &[u8; 256]) {
    let data = &mut image.data;
    let len = data.len();

    // Pre-load LUT into 16 registers (4 sets of 4 registers = 256 bytes)
    let lut_ptr = lut.as_ptr();
    let t0 = vld1q_u8_x4(lut_ptr);           // bytes 0-63
    let t1 = vld1q_u8_x4(lut_ptr.add(64));    // bytes 64-127
    let t2 = vld1q_u8_x4(lut_ptr.add(128));   // bytes 128-191
    let t3 = vld1q_u8_x4(lut_ptr.add(192));   // bytes 192-255

    // Pre-create constants
    let v64 = vdupq_n_u8(64);
    let v128 = vdupq_n_u8(128);
    let v192 = vdupq_n_u8(192);

    let mut i = 0;

    // Main loop: 64 pixels per iteration (4 batches of 16)
    while i + 63 < len {
        // Load 4 vectors (64 pixels) using gathered loads
        let idx_a = vld1q_u8(data.as_ptr().add(i) as *const u8);
        let idx_b = vld1q_u8(data.as_ptr().add(i + 16) as *const u8);
        let idx_c = vld1q_u8(data.as_ptr().add(i + 32) as *const u8);
        let idx_d = vld1q_u8(data.as_ptr().add(i + 48) as *const u8);

        // Phase 1: Lookups for bytes 0-63
        let ra0 = vqtbl4q_u8(t0, idx_a);
        let rb0 = vqtbl4q_u8(t0, idx_b);
        let rc0 = vqtbl4q_u8(t0, idx_c);
        let rd0 = vqtbl4q_u8(t0, idx_d);

        // Phase 2: Lookups for bytes 64-127
        let ra1 = vqtbl4q_u8(t1, vsubq_u8(idx_a, v64));
        let rb1 = vqtbl4q_u8(t1, vsubq_u8(idx_b, v64));
        let rc1 = vqtbl4q_u8(t1, vsubq_u8(idx_c, v64));
        let rd1 = vqtbl4q_u8(t1, vsubq_u8(idx_d, v64));

        // Phase 3: Lookups for bytes 128-191
        let ra2 = vqtbl4q_u8(t2, vsubq_u8(idx_a, v128));
        let rb2 = vqtbl4q_u8(t2, vsubq_u8(idx_b, v128));
        let rc2 = vqtbl4q_u8(t2, vsubq_u8(idx_c, v128));
        let rd2 = vqtbl4q_u8(t2, vsubq_u8(idx_d, v128));

        // Phase 4: Lookups for bytes 192-255
        let ra3 = vqtbl4q_u8(t3, vsubq_u8(idx_a, v192));
        let rb3 = vqtbl4q_u8(t3, vsubq_u8(idx_b, v192));
        let rc3 = vqtbl4q_u8(t3, vsubq_u8(idx_c, v192));
        let rd3 = vqtbl4q_u8(t3, vsubq_u8(idx_d, v192));

        // Final merge: Use bitwise select with masks (not OR!)
        // Masks for each range
        let mask_a1 = vcgeq_u8(idx_a, v64);
        let mask_a2 = vcgeq_u8(idx_a, v128);
        let mask_a3 = vcgeq_u8(idx_a, v192);
        let mut res_a = vbslq_u8(mask_a1, ra1, ra0);
        res_a = vbslq_u8(mask_a2, ra2, res_a);
        res_a = vbslq_u8(mask_a3, ra3, res_a);

        let mask_b1 = vcgeq_u8(idx_b, v64);
        let mask_b2 = vcgeq_u8(idx_b, v128);
        let mask_b3 = vcgeq_u8(idx_b, v192);
        let mut res_b = vbslq_u8(mask_b1, rb1, rb0);
        res_b = vbslq_u8(mask_b2, rb2, res_b);
        res_b = vbslq_u8(mask_b3, rb3, res_b);

        let mask_c1 = vcgeq_u8(idx_c, v64);
        let mask_c2 = vcgeq_u8(idx_c, v128);
        let mask_c3 = vcgeq_u8(idx_c, v192);
        let mut res_c = vbslq_u8(mask_c1, rc1, rc0);
        res_c = vbslq_u8(mask_c2, rc2, res_c);
        res_c = vbslq_u8(mask_c3, rc3, res_c);

        let mask_d1 = vcgeq_u8(idx_d, v64);
        let mask_d2 = vcgeq_u8(idx_d, v128);
        let mask_d3 = vcgeq_u8(idx_d, v192);
        let mut res_d = vbslq_u8(mask_d1, rd1, rd0);
        res_d = vbslq_u8(mask_d2, rd2, res_d);
        res_d = vbslq_u8(mask_d3, rd3, res_d);

        // Store all 4 results
        vst1q_u8(data.as_mut_ptr().add(i) as *mut u8, res_a);
        vst1q_u8(data.as_mut_ptr().add(i + 16) as *mut u8, res_b);
        vst1q_u8(data.as_mut_ptr().add(i + 32) as *mut u8, res_c);
        vst1q_u8(data.as_mut_ptr().add(i + 48) as *mut u8, res_d);

        i += 64;
    }

    // Handle remaining pixels (less than 64)
    if i < len {
        let remaining = len - i;

        // Handle up to 48 pixels in one batch
        if remaining >= 48 {
            let idx_a = vld1q_u8(data.as_ptr().add(i) as *const u8);
            let idx_b = vld1q_u8(data.as_ptr().add(i + 16) as *const u8);
            let idx_c = vld1q_u8(data.as_ptr().add(i + 32) as *const u8);

            let ra0 = vqtbl4q_u8(t0, idx_a);
            let rb0 = vqtbl4q_u8(t0, idx_b);
            let rc0 = vqtbl4q_u8(t0, idx_c);

            let ra1 = vqtbl4q_u8(t1, vsubq_u8(idx_a, v64));
            let rb1 = vqtbl4q_u8(t1, vsubq_u8(idx_b, v64));
            let rc1 = vqtbl4q_u8(t1, vsubq_u8(idx_c, v64));

            let ra2 = vqtbl4q_u8(t2, vsubq_u8(idx_a, v128));
            let rb2 = vqtbl4q_u8(t2, vsubq_u8(idx_b, v128));
            let rc2 = vqtbl4q_u8(t2, vsubq_u8(idx_c, v128));

            let ra3 = vqtbl4q_u8(t3, vsubq_u8(idx_a, v192));
            let rb3 = vqtbl4q_u8(t3, vsubq_u8(idx_b, v192));
            let rc3 = vqtbl4q_u8(t3, vsubq_u8(idx_c, v192));

            let res_a = vorrq_u8(vorrq_u8(ra0, ra1), vorrq_u8(ra2, ra3));
            let res_b = vorrq_u8(vorrq_u8(rb0, rb1), vorrq_u8(rb2, rb3));
            let res_c = vorrq_u8(vorrq_u8(rc0, rc1), vorrq_u8(rc2, rc3));

            vst1q_u8(data.as_mut_ptr().add(i) as *mut u8, res_a);
            vst1q_u8(data.as_mut_ptr().add(i + 16) as *mut u8, res_b);
            vst1q_u8(data.as_mut_ptr().add(i + 32) as *mut u8, res_c);

            i += 48;
        }

        // Handle up to 32 pixels
        if i < len && len - i >= 32 {
            let idx_a = vld1q_u8(data.as_ptr().add(i) as *const u8);
            let idx_b = vld1q_u8(data.as_ptr().add(i + 16) as *const u8);

            let ra0 = vqtbl4q_u8(t0, idx_a);
            let rb0 = vqtbl4q_u8(t0, idx_b);

            let ra1 = vqtbl4q_u8(t1, vsubq_u8(idx_a, v64));
            let rb1 = vqtbl4q_u8(t1, vsubq_u8(idx_b, v64));

            let ra2 = vqtbl4q_u8(t2, vsubq_u8(idx_a, v128));
            let rb2 = vqtbl4q_u8(t2, vsubq_u8(idx_b, v128));

            let ra3 = vqtbl4q_u8(t3, vsubq_u8(idx_a, v192));
            let rb3 = vqtbl4q_u8(t3, vsubq_u8(idx_b, v192));

            let res_a = vorrq_u8(vorrq_u8(ra0, ra1), vorrq_u8(ra2, ra3));
            let res_b = vorrq_u8(vorrq_u8(rb0, rb1), vorrq_u8(rb2, rb3));

            vst1q_u8(data.as_mut_ptr().add(i) as *mut u8, res_a);
            vst1q_u8(data.as_mut_ptr().add(i + 16) as *mut u8, res_b);

            i += 32;
        }

        // Handle up to 16 pixels
        if i < len && len - i >= 16 {
            let indices = vld1q_u8(data.as_ptr().add(i) as *const u8);

            let res0 = vqtbl4q_u8(t0, indices);
            let res1 = vqtbl4q_u8(t1, vsubq_u8(indices, v64));
            let res2 = vqtbl4q_u8(t2, vsubq_u8(indices, v128));
            let res3 = vqtbl4q_u8(t3, vsubq_u8(indices, v192));

            let result = vorrq_u8(vorrq_u8(res0, res1), vorrq_u8(res2, res3));

            vst1q_u8(data.as_mut_ptr().add(i) as *mut u8, result);

            i += 16;
        }

        // Handle remaining with scalar
        while i < len {
            data[i] = lut[data[i] as usize];
            i += 1;
        }
    }
}
