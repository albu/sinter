/// NEON-optimized box blur using 2-pass separable sliding window (AArch64 only)
///
/// Architecture:
/// - Horizontal pass: Sliding sum across each row (O(1) per pixel), storing horizontal averages in temp buffer
/// - Vertical pass: NEON SIMD with sliding window down each column (O(1) per pixel)
/// - Exact 32-bit fixed point multipliers to guarantee 100% constant image preservation

use std::arch::aarch64::*;

pub(in crate::transforms::kernel) fn box_blur_impl_neon(data: &mut [u8], width: usize, height: usize, radius: usize) {
    if width == 0 || height == 0 || radius == 0 {
        return;
    }

    let mut temp_buffer = vec![0u8; width * height * 3];

    // Pre-compute horizontal fixed-point multipliers
    let mut mul_at = vec![0u64; width];
    for x in 0..width {
        let h_left = x.saturating_sub(radius);
        let h_right = (x + radius).min(width - 1);
        let h_width = (h_right - h_left + 1) as u64;
        mul_at[x] = ((1u64 << 32) + h_width / 2) / h_width;
    }

    // Pre-compute vertical fixed-point multipliers
    let mut mul_v = vec![0u64; height];
    for y in 0..height {
        let v_top = y.saturating_sub(radius);
        let v_bottom = (y + radius).min(height - 1);
        let v_height = (v_bottom - v_top + 1) as u64;
        mul_v[y] = ((1u64 << 32) + v_height / 2) / v_height;
    }

    // ============================================================
    // PASS 1: HORIZONTAL (O(1) sliding sum per pixel)
    // ============================================================
    for y in 0..height {
        let row_offset = y * width * 3;
        for c in 0..3 {
            let ch_offset = row_offset + c;
            let mut sum: u32 = 0;
            let init_w = radius.min(width - 1);
            for x in 0..=init_w {
                sum += data[ch_offset + x * 3] as u32;
            }

            for x in 0..width {
                let mul = mul_at[x];
                let avg = (((sum as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                temp_buffer[ch_offset + x * 3] = avg;

                let x_in = x + radius + 1;
                let x_out = x.saturating_sub(radius);

                if x_in < width {
                    sum += data[ch_offset + x_in * 3] as u32;
                }
                if x >= radius {
                    sum -= data[ch_offset + x_out * 3] as u32;
                }
            }
        }
    }

    // ============================================================
    // PASS 2: VERTICAL (NEON SIMD with sliding window)
    // ============================================================
    for c in 0..3 {
        let mut x = 0;

        // Process 8 columns at a time with NEON
        while x + 8 <= width {
            unsafe {
                let v_bottom = radius.min(height - 1);
                let mut acc_lo = vdupq_n_u32(0);
                let mut acc_hi = vdupq_n_u32(0);

                for img_row in 0..=v_bottom {
                    let base_ptr = temp_buffer.as_ptr().add(img_row * width * 3 + x * 3 + c);
                    let v8 = load_u8x8_strided(base_ptr, 3);
                    let v16 = vmovl_u8(v8);
                    acc_lo = vaddw_u16(acc_lo, vget_low_u16(v16));
                    acc_hi = vaddw_u16(acc_hi, vget_high_u16(v16));
                }

                for out_y in 0..height {
                    let mul = mul_v[out_y];
                    let s0 = vgetq_lane_u32(acc_lo, 0);
                    let s1 = vgetq_lane_u32(acc_lo, 1);
                    let s2 = vgetq_lane_u32(acc_lo, 2);
                    let s3 = vgetq_lane_u32(acc_lo, 3);
                    let s4 = vgetq_lane_u32(acc_hi, 0);
                    let s5 = vgetq_lane_u32(acc_hi, 1);
                    let s6 = vgetq_lane_u32(acc_hi, 2);
                    let s7 = vgetq_lane_u32(acc_hi, 3);

                    let out_ptr = data.as_mut_ptr().add((out_y * width + x) * 3 + c);
                    *out_ptr.offset(0) = (((s0 as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                    *out_ptr.offset(3) = (((s1 as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                    *out_ptr.offset(6) = (((s2 as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                    *out_ptr.offset(9) = (((s3 as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                    *out_ptr.offset(12) = (((s4 as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                    *out_ptr.offset(15) = (((s5 as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                    *out_ptr.offset(18) = (((s6 as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                    *out_ptr.offset(21) = (((s7 as u64 * mul) + (1u64 << 31)) >> 32) as u8;

                    let row_leaving = out_y.saturating_sub(radius);
                    let row_entering = out_y + radius + 1;

                    if row_entering < height {
                        let base_ptr = temp_buffer.as_ptr().add(row_entering * width * 3 + x * 3 + c);
                        let v8 = load_u8x8_strided(base_ptr, 3);
                        let v16 = vmovl_u8(v8);
                        acc_lo = vaddw_u16(acc_lo, vget_low_u16(v16));
                        acc_hi = vaddw_u16(acc_hi, vget_high_u16(v16));
                    }

                    if out_y >= radius {
                        let base_ptr = temp_buffer.as_ptr().add(row_leaving * width * 3 + x * 3 + c);
                        let v8 = load_u8x8_strided(base_ptr, 3);
                        let v16 = vmovl_u8(v8);
                        acc_lo = vsubw_u16(acc_lo, vget_low_u16(v16));
                        acc_hi = vsubw_u16(acc_hi, vget_high_u16(v16));
                    }
                }
            }
            x += 8;
        }

        // Scalar tail for remaining columns (< 8 columns or narrow widths)
        for col in x..width {
            let v_bottom = radius.min(height - 1);
            let mut acc: u32 = 0;
            for img_row in 0..=v_bottom {
                acc += temp_buffer[img_row * width * 3 + col * 3 + c] as u32;
            }

            for out_y in 0..height {
                let mul = mul_v[out_y];
                let out_val = (((acc as u64 * mul) + (1u64 << 31)) >> 32) as u8;
                data[(out_y * width + col) * 3 + c] = out_val;

                let row_leaving = out_y.saturating_sub(radius);
                let row_entering = out_y + radius + 1;

                if row_entering < height {
                    acc += temp_buffer[row_entering * width * 3 + col * 3 + c] as u32;
                }
                if out_y >= radius {
                    acc -= temp_buffer[row_leaving * width * 3 + col * 3 + c] as u32;
                }
            }
        }
    }
}

/// NEON helper: Load 8 u8 values with stride (for interleaved RGB)
#[inline(always)]
unsafe fn load_u8x8_strided(base: *const u8, stride: isize) -> uint8x8_t {
    let arr = [
        *base.offset(0),
        *base.offset(stride),
        *base.offset(2 * stride),
        *base.offset(3 * stride),
        *base.offset(4 * stride),
        *base.offset(5 * stride),
        *base.offset(6 * stride),
        *base.offset(7 * stride),
    ];
    vld1_u8(arr.as_ptr())
}
