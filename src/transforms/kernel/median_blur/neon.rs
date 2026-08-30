#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::sorting_network::median9;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vcas(a: &mut uint8x16_t, b: &mut uint8x16_t) {
    let min = vminq_u8(*a, *b);
    let max = vmaxq_u8(*a, *b);
    *a = min;
    *b = max;
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sort3(
    a: uint8x16_t,
    b: uint8x16_t,
    c: uint8x16_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    let min_ab = vminq_u8(a, b);
    let max_ab = vmaxq_u8(a, b);
    let min_all = vminq_u8(min_ab, c);
    let max_temp = vmaxq_u8(min_ab, c);
    let mid_all = vminq_u8(max_ab, max_temp);
    let max_all = vmaxq_u8(max_ab, c);
    (min_all, mid_all, max_all)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mid3(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    let min_ab = vminq_u8(a, b);
    let max_ab = vmaxq_u8(a, b);
    vminq_u8(max_ab, vmaxq_u8(min_ab, c))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn median_of_3_sorted_columns(
    min0: uint8x16_t,
    mid0: uint8x16_t,
    max0: uint8x16_t,
    min1: uint8x16_t,
    mid1: uint8x16_t,
    max1: uint8x16_t,
    min2: uint8x16_t,
    mid2: uint8x16_t,
    max2: uint8x16_t,
) -> uint8x16_t {
    let max_min = vmaxq_u8(vmaxq_u8(min0, min1), min2);
    let min_max = vminq_u8(vminq_u8(max0, max1), max2);
    let mid_mid = mid3(mid0, mid1, mid2);
    mid3(max_min, mid_mid, min_max)
}

/// Apply 3x3 median filter using ARM NEON column-cache sorting network
#[cfg(target_arch = "aarch64")]
pub unsafe fn apply_median_blur_3x3_neon(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
) {
    if width < 3 || height < 3 {
        super::sorting_network::apply_median_blur_3x3_scalar(data, width, height, channels);
        return;
    }

    let len = data.len();
    let mut output = Vec::<u8>::with_capacity(len);
    output.set_len(len);
    let stride = width * channels;

    // 1. Top and bottom border rows (scalar)
    for y in [0, height - 1] {
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(height - 1);
        let row_curr = y * stride;
        let row_prev = y_prev * stride;
        let row_next = y_next * stride;

        for x in 0..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);
            for c in 0..channels {
                let p = [
                    data[row_prev + x_prev * channels + c],
                    data[row_prev + x * channels + c],
                    data[row_prev + x_next * channels + c],
                    data[row_curr + x_prev * channels + c],
                    data[row_curr + x * channels + c],
                    data[row_curr + x_next * channels + c],
                    data[row_next + x_prev * channels + c],
                    data[row_next + x * channels + c],
                    data[row_next + x_next * channels + c],
                ];
                output[row_curr + x * channels + c] = median9(p);
            }
        }
    }

    // 2. Interior rows using fast column-cache (interleaved across all channels)
    let step = channels;
    let row_bytes = width * channels;

    for y in 1..(height - 1) {
        let prev_ptr = data.as_ptr().add((y - 1) * stride);
        let curr_ptr = data.as_ptr().add(y * stride);
        let next_ptr = data.as_ptr().add((y + 1) * stride);
        let out_ptr = output.as_mut_ptr().add(y * stride);

        // Left border pixel (x=0)
        for c in 0..channels {
            let p_left = [
                *prev_ptr.add(c), *prev_ptr.add(c), *prev_ptr.add(step + c),
                *curr_ptr.add(c), *curr_ptr.add(c), *curr_ptr.add(step + c),
                *next_ptr.add(c), *next_ptr.add(c), *next_ptr.add(step + c),
            ];
            *out_ptr.add(c) = median9(p_left);
        }

        let mut byte_idx = step;
        let simd_end = row_bytes.saturating_sub(16 + step);

        while byte_idx + 32 <= simd_end + 16 {
            let p00_a = vld1q_u8(prev_ptr.add(byte_idx - step));
            let p01_a = vld1q_u8(prev_ptr.add(byte_idx));
            let p02_a = vld1q_u8(prev_ptr.add(byte_idx + step));

            let p10_a = vld1q_u8(curr_ptr.add(byte_idx - step));
            let p11_a = vld1q_u8(curr_ptr.add(byte_idx));
            let p12_a = vld1q_u8(curr_ptr.add(byte_idx + step));

            let p20_a = vld1q_u8(next_ptr.add(byte_idx - step));
            let p21_a = vld1q_u8(next_ptr.add(byte_idx));
            let p22_a = vld1q_u8(next_ptr.add(byte_idx + step));

            let (c0_min_a, c0_mid_a, c0_max_a) = sort3(p00_a, p10_a, p20_a);
            let (c1_min_a, c1_mid_a, c1_max_a) = sort3(p01_a, p11_a, p21_a);
            let (c2_min_a, c2_mid_a, c2_max_a) = sort3(p02_a, p12_a, p22_a);

            let res_a = median_of_3_sorted_columns(
                c0_min_a, c0_mid_a, c0_max_a,
                c1_min_a, c1_mid_a, c1_max_a,
                c2_min_a, c2_mid_a, c2_max_a,
            );

            let p00_b = vld1q_u8(prev_ptr.add(byte_idx + 16 - step));
            let p01_b = vld1q_u8(prev_ptr.add(byte_idx + 16));
            let p02_b = vld1q_u8(prev_ptr.add(byte_idx + 16 + step));

            let p10_b = vld1q_u8(curr_ptr.add(byte_idx + 16 - step));
            let p11_b = vld1q_u8(curr_ptr.add(byte_idx + 16));
            let p12_b = vld1q_u8(curr_ptr.add(byte_idx + 16 + step));

            let p20_b = vld1q_u8(next_ptr.add(byte_idx + 16 - step));
            let p21_b = vld1q_u8(next_ptr.add(byte_idx + 16));
            let p22_b = vld1q_u8(next_ptr.add(byte_idx + 16 + step));

            let (c0_min_b, c0_mid_b, c0_max_b) = sort3(p00_b, p10_b, p20_b);
            let (c1_min_b, c1_mid_b, c1_max_b) = sort3(p01_b, p11_b, p21_b);
            let (c2_min_b, c2_mid_b, c2_max_b) = sort3(p02_b, p12_b, p22_b);

            let res_b = median_of_3_sorted_columns(
                c0_min_b, c0_mid_b, c0_max_b,
                c1_min_b, c1_mid_b, c1_max_b,
                c2_min_b, c2_mid_b, c2_max_b,
            );

            vst1q_u8(out_ptr.add(byte_idx), res_a);
            vst1q_u8(out_ptr.add(byte_idx + 16), res_b);

            byte_idx += 32;
        }

        while byte_idx <= simd_end {
            let p00 = vld1q_u8(prev_ptr.add(byte_idx - step));
            let p01 = vld1q_u8(prev_ptr.add(byte_idx));
            let p02 = vld1q_u8(prev_ptr.add(byte_idx + step));

            let p10 = vld1q_u8(curr_ptr.add(byte_idx - step));
            let p11 = vld1q_u8(curr_ptr.add(byte_idx));
            let p12 = vld1q_u8(curr_ptr.add(byte_idx + step));

            let p20 = vld1q_u8(next_ptr.add(byte_idx - step));
            let p21 = vld1q_u8(next_ptr.add(byte_idx));
            let p22 = vld1q_u8(next_ptr.add(byte_idx + step));

            let (c0_min, c0_mid, c0_max) = sort3(p00, p10, p20);
            let (c1_min, c1_mid, c1_max) = sort3(p01, p11, p21);
            let (c2_min, c2_mid, c2_max) = sort3(p02, p12, p22);

            let res = median_of_3_sorted_columns(
                c0_min, c0_mid, c0_max,
                c1_min, c1_mid, c1_max,
                c2_min, c2_mid, c2_max,
            );
            vst1q_u8(out_ptr.add(byte_idx), res);
            byte_idx += 16;
        }

        // Right remainder pixels in the row
        let x_start = byte_idx / channels;
        for x in x_start..width {
            let x_prev = x.saturating_sub(1);
            let x_next = (x + 1).min(width - 1);
            for c in 0..channels {
                let p = [
                    *prev_ptr.add(x_prev * channels + c),
                    *prev_ptr.add(x * channels + c),
                    *prev_ptr.add(x_next * channels + c),
                    *curr_ptr.add(x_prev * channels + c),
                    *curr_ptr.add(x * channels + c),
                    *curr_ptr.add(x_next * channels + c),
                    *next_ptr.add(x_prev * channels + c),
                    *next_ptr.add(x * channels + c),
                    *next_ptr.add(x_next * channels + c),
                ];
                *out_ptr.add(x * channels + c) = median9(p);
            }
        }
    }

    data.copy_from_slice(&output);
}

/// Median of 25 via a 25-input odd-even mergesort network (Batcher).
///
/// Derived from the 32-input network by dropping the no-op comparators against
/// the constant-max padding lanes; p12 holds the true median after sorting.
/// Verified against the 0-1 lemma and random inputs.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vmedian25_sortnet(
    mut p0: uint8x16_t, mut p1: uint8x16_t, mut p2: uint8x16_t, mut p3: uint8x16_t, mut p4: uint8x16_t,
    mut p5: uint8x16_t, mut p6: uint8x16_t, mut p7: uint8x16_t, mut p8: uint8x16_t, mut p9: uint8x16_t,
    mut p10: uint8x16_t, mut p11: uint8x16_t, mut p12: uint8x16_t, mut p13: uint8x16_t, mut p14: uint8x16_t,
    mut p15: uint8x16_t, mut p16: uint8x16_t, mut p17: uint8x16_t, mut p18: uint8x16_t, mut p19: uint8x16_t,
    mut p20: uint8x16_t, mut p21: uint8x16_t, mut p22: uint8x16_t, mut p23: uint8x16_t, mut p24: uint8x16_t,
) -> uint8x16_t {
    vcas(&mut p0, &mut p1);
    vcas(&mut p2, &mut p3);
    vcas(&mut p0, &mut p2);
    vcas(&mut p1, &mut p3);
    vcas(&mut p1, &mut p2);
    vcas(&mut p4, &mut p5);
    vcas(&mut p6, &mut p7);
    vcas(&mut p4, &mut p6);
    vcas(&mut p5, &mut p7);
    vcas(&mut p5, &mut p6);
    vcas(&mut p0, &mut p4);
    vcas(&mut p2, &mut p6);
    vcas(&mut p2, &mut p4);
    vcas(&mut p1, &mut p5);
    vcas(&mut p3, &mut p7);
    vcas(&mut p3, &mut p5);
    vcas(&mut p1, &mut p2);
    vcas(&mut p3, &mut p4);
    vcas(&mut p5, &mut p6);
    vcas(&mut p8, &mut p9);
    vcas(&mut p10, &mut p11);
    vcas(&mut p8, &mut p10);
    vcas(&mut p9, &mut p11);
    vcas(&mut p9, &mut p10);
    vcas(&mut p12, &mut p13);
    vcas(&mut p14, &mut p15);
    vcas(&mut p12, &mut p14);
    vcas(&mut p13, &mut p15);
    vcas(&mut p13, &mut p14);
    vcas(&mut p8, &mut p12);
    vcas(&mut p10, &mut p14);
    vcas(&mut p10, &mut p12);
    vcas(&mut p9, &mut p13);
    vcas(&mut p11, &mut p15);
    vcas(&mut p11, &mut p13);
    vcas(&mut p9, &mut p10);
    vcas(&mut p11, &mut p12);
    vcas(&mut p13, &mut p14);
    vcas(&mut p0, &mut p8);
    vcas(&mut p4, &mut p12);
    vcas(&mut p4, &mut p8);
    vcas(&mut p2, &mut p10);
    vcas(&mut p6, &mut p14);
    vcas(&mut p6, &mut p10);
    vcas(&mut p2, &mut p4);
    vcas(&mut p6, &mut p8);
    vcas(&mut p10, &mut p12);
    vcas(&mut p1, &mut p9);
    vcas(&mut p5, &mut p13);
    vcas(&mut p5, &mut p9);
    vcas(&mut p3, &mut p11);
    vcas(&mut p7, &mut p15);
    vcas(&mut p7, &mut p11);
    vcas(&mut p3, &mut p5);
    vcas(&mut p7, &mut p9);
    vcas(&mut p11, &mut p13);
    vcas(&mut p1, &mut p2);
    vcas(&mut p3, &mut p4);
    vcas(&mut p5, &mut p6);
    vcas(&mut p7, &mut p8);
    vcas(&mut p9, &mut p10);
    vcas(&mut p11, &mut p12);
    vcas(&mut p13, &mut p14);
    vcas(&mut p16, &mut p17);
    vcas(&mut p18, &mut p19);
    vcas(&mut p16, &mut p18);
    vcas(&mut p17, &mut p19);
    vcas(&mut p17, &mut p18);
    vcas(&mut p20, &mut p21);
    vcas(&mut p22, &mut p23);
    vcas(&mut p20, &mut p22);
    vcas(&mut p21, &mut p23);
    vcas(&mut p21, &mut p22);
    vcas(&mut p16, &mut p20);
    vcas(&mut p18, &mut p22);
    vcas(&mut p18, &mut p20);
    vcas(&mut p17, &mut p21);
    vcas(&mut p19, &mut p23);
    vcas(&mut p19, &mut p21);
    vcas(&mut p17, &mut p18);
    vcas(&mut p19, &mut p20);
    vcas(&mut p21, &mut p22);
    vcas(&mut p16, &mut p24);
    vcas(&mut p20, &mut p24);
    vcas(&mut p18, &mut p20);
    vcas(&mut p22, &mut p24);
    vcas(&mut p19, &mut p21);
    vcas(&mut p17, &mut p18);
    vcas(&mut p19, &mut p20);
    vcas(&mut p21, &mut p22);
    vcas(&mut p23, &mut p24);
    vcas(&mut p0, &mut p16);
    vcas(&mut p8, &mut p24);
    vcas(&mut p8, &mut p16);
    vcas(&mut p4, &mut p20);
    vcas(&mut p12, &mut p20);
    vcas(&mut p4, &mut p8);
    vcas(&mut p12, &mut p16);
    vcas(&mut p20, &mut p24);
    vcas(&mut p2, &mut p18);
    vcas(&mut p10, &mut p18);
    vcas(&mut p6, &mut p22);
    vcas(&mut p14, &mut p22);
    vcas(&mut p6, &mut p10);
    vcas(&mut p14, &mut p18);
    vcas(&mut p2, &mut p4);
    vcas(&mut p6, &mut p8);
    vcas(&mut p10, &mut p12);
    vcas(&mut p14, &mut p16);
    vcas(&mut p18, &mut p20);
    vcas(&mut p22, &mut p24);
    vcas(&mut p1, &mut p17);
    vcas(&mut p9, &mut p17);
    vcas(&mut p5, &mut p21);
    vcas(&mut p13, &mut p21);
    vcas(&mut p5, &mut p9);
    vcas(&mut p13, &mut p17);
    vcas(&mut p3, &mut p19);
    vcas(&mut p11, &mut p19);
    vcas(&mut p7, &mut p23);
    vcas(&mut p15, &mut p23);
    vcas(&mut p7, &mut p11);
    vcas(&mut p15, &mut p19);
    vcas(&mut p3, &mut p5);
    vcas(&mut p7, &mut p9);
    vcas(&mut p11, &mut p13);
    vcas(&mut p15, &mut p17);
    vcas(&mut p19, &mut p21);
    vcas(&mut p1, &mut p2);
    vcas(&mut p3, &mut p4);
    vcas(&mut p5, &mut p6);
    vcas(&mut p7, &mut p8);
    vcas(&mut p9, &mut p10);
    vcas(&mut p11, &mut p12);
    vcas(&mut p13, &mut p14);
    vcas(&mut p15, &mut p16);
    vcas(&mut p17, &mut p18);
    vcas(&mut p19, &mut p20);
    vcas(&mut p21, &mut p22);
    vcas(&mut p23, &mut p24);
    p12
}

/// Apply 5x5 median filter using a vectorized odd-even mergesort network.
#[cfg(target_arch = "aarch64")]
pub unsafe fn apply_median_blur_5x5_neon(
    data: &mut [u8],
    width: usize,
    height: usize,
    channels: usize,
) {
    if width < 5 || height < 5 {
        super::histogram::apply_median_blur_5x5(&mut crate::core::FusableImage::new(
            data, width, height, channels,
        ));
        return;
    }

    let stride = width * channels;
    let step = if channels == 3 { 3 } else { 1 };
    let row_len = stride;
    let mut output = vec![0u8; data.len()];

    // Border rows (top 2, bottom 2): direct clamped read.
    for y in (0..2).chain(height - 2..height) {
        for x in 0..width {
            for c in 0..channels {
                let mut win = [0u8; 25];
                let mut idx = 0;
                for dy in -2i32..=2 {
                    let sy = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                    for dx in -2i32..=2 {
                        let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        win[idx] = data[sy * stride + sx * channels + c];
                        idx += 1;
                    }
                }
                win.sort_unstable();
                output[y * stride + x * channels + c] = win[12];
            }
        }
    }

    // Five row buffers for rows y-2 .. y+2 (interior only).
    let mut rows: Vec<Vec<u8>> = vec![vec![0u8; row_len]; 5];
    for (i, dy) in [-2i32, -1, 0, 1, 2].iter().enumerate() {
        let sy = (2i32 + dy) as usize;
        rows[i].copy_from_slice(&data[sy * stride..(sy + 1) * stride]);
    }

    for y in 2..height - 2 {
        let out_row = y * stride;
        let ptrs = [
            rows[0].as_ptr(),
            rows[1].as_ptr(),
            rows[2].as_ptr(),
            rows[3].as_ptr(),
            rows[4].as_ptr(),
        ];

        // Left border columns (x = 0, 1): clamped read from the row buffers.
        for x in 0..2 {
            for c in 0..channels {
                let mut win = [0u8; 25];
                let mut idx = 0;
                for r in 0..5 {
                    for dx in -2i32..=2 {
                        let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        win[idx] = *ptrs[r].add(sx * channels + c);
                        idx += 1;
                    }
                }
                win.sort_unstable();
                output[out_row + x * channels + c] = win[12];
            }
        }

        // SIMD interior: 16 pixels at a time.
        let mut byte_idx = 2 * step;
        let simd_end = row_len.saturating_sub(16 + 2 * step);
        while byte_idx <= simd_end {
            let s = step;
            let s2 = 2 * step;
            let p0 = vld1q_u8(ptrs[0].add(byte_idx - s2));
            let p1 = vld1q_u8(ptrs[0].add(byte_idx - s));
            let p2 = vld1q_u8(ptrs[0].add(byte_idx));
            let p3 = vld1q_u8(ptrs[0].add(byte_idx + s));
            let p4 = vld1q_u8(ptrs[0].add(byte_idx + s2));
            let p5 = vld1q_u8(ptrs[1].add(byte_idx - s2));
            let p6 = vld1q_u8(ptrs[1].add(byte_idx - s));
            let p7 = vld1q_u8(ptrs[1].add(byte_idx));
            let p8 = vld1q_u8(ptrs[1].add(byte_idx + s));
            let p9 = vld1q_u8(ptrs[1].add(byte_idx + s2));
            let p10 = vld1q_u8(ptrs[2].add(byte_idx - s2));
            let p11 = vld1q_u8(ptrs[2].add(byte_idx - s));
            let p12 = vld1q_u8(ptrs[2].add(byte_idx));
            let p13 = vld1q_u8(ptrs[2].add(byte_idx + s));
            let p14 = vld1q_u8(ptrs[2].add(byte_idx + s2));
            let p15 = vld1q_u8(ptrs[3].add(byte_idx - s2));
            let p16 = vld1q_u8(ptrs[3].add(byte_idx - s));
            let p17 = vld1q_u8(ptrs[3].add(byte_idx));
            let p18 = vld1q_u8(ptrs[3].add(byte_idx + s));
            let p19 = vld1q_u8(ptrs[3].add(byte_idx + s2));
            let p20 = vld1q_u8(ptrs[4].add(byte_idx - s2));
            let p21 = vld1q_u8(ptrs[4].add(byte_idx - s));
            let p22 = vld1q_u8(ptrs[4].add(byte_idx));
            let p23 = vld1q_u8(ptrs[4].add(byte_idx + s));
            let p24 = vld1q_u8(ptrs[4].add(byte_idx + s2));

            let res = vmedian25_sortnet(
                p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                p10, p11, p12, p13, p14, p15, p16, p17, p18, p19,
                p20, p21, p22, p23, p24,
            );
            vst1q_u8(output.as_mut_ptr().add(out_row + byte_idx), res);
            byte_idx += 16;
        }

        // Right border columns: everything after the last SIMD chunk.
        let x_start = byte_idx / channels;
        for x in x_start..width {
            for c in 0..channels {
                let mut win = [0u8; 25];
                let mut idx = 0;
                for r in 0..5 {
                    for dx in -2i32..=2 {
                        let sx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        win[idx] = *ptrs[r].add(sx * channels + c);
                        idx += 1;
                    }
                }
                win.sort_unstable();
                output[out_row + x * channels + c] = win[12];
            }
        }

        // Prepare row buffers for the next row (y+1): rows become y-1..y+3.
        if y + 1 < height - 2 {
            rows.rotate_left(1);
            rows[4].copy_from_slice(&data[(y + 3) * stride..(y + 4) * stride]);
        }
    }

    data.copy_from_slice(&output);
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use std::arch::aarch64::*;

    fn median25_scalar(vals: &[u8; 25]) -> u8 {
        let mut v = *vals;
        v.sort_unstable();
        v[12]
    }

    /// Direct check of the 25-input odd-even mergesort network against the
    /// true median, lane by lane, on pseudo-random inputs.
    #[test]
    fn test_vmedian25_sortnet_correct() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 256) as u8
        };

        unsafe {
            for _trial in 0..300 {
                let mut lanes = [0u8; 16];
                let mut vecs = Vec::with_capacity(25);
                for _k in 0..25 {
                    for i in 0..16 {
                        lanes[i] = next();
                    }
                    vecs.push(vld1q_u8(lanes.as_ptr()));
                }
                let res = vmedian25_sortnet(
                    vecs[0], vecs[1], vecs[2], vecs[3], vecs[4],
                    vecs[5], vecs[6], vecs[7], vecs[8], vecs[9],
                    vecs[10], vecs[11], vecs[12], vecs[13], vecs[14],
                    vecs[15], vecs[16], vecs[17], vecs[18], vecs[19],
                    vecs[20], vecs[21], vecs[22], vecs[23], vecs[24],
                );
                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), res);
                for i in 0..16 {
                    let mut win = [0u8; 25];
                    for k in 0..25 {
                        let mut tmp = [0u8; 16];
                        vst1q_u8(tmp.as_mut_ptr(), vecs[k]);
                        win[k] = tmp[i];
                    }
                    let expected = median25_scalar(&win);
                    assert_eq!(
                        out[i],
                        expected,
                        "vmedian25_sortnet wrong at lane {} trial {}: got {} expected {}",
                        i,
                        _trial,
                        out[i],
                        expected
                    );
                }
            }
        }
    }
}
