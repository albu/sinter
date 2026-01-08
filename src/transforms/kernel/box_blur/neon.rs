/// NEON-optimized box blur using strip-mined fusion with O(1) vertical sliding window (AArch64 only)
///
/// Architecture:
/// - Strip height keeps temp buffer in L2 cache
/// - Horizontal pass: scalar sliding sum (O(1) per pixel)
/// - Vertical pass: NEON SIMD with SLIDING WINDOW (O(1) per pixel, not O(R))
///
/// Key insight: Vertical pass must use sliding window, not re-sum all rows for each pixel

use std::arch::aarch64::*;

pub(in crate::transforms::kernel) fn box_blur_impl_neon(data: &mut [u8], width: usize, height: usize, radius: usize) {
    // STRIP_HEIGHT: Process image in horizontal strips of this height.
    // - 256 rows * 1920 cols * 3 channels = ~1.5 MB (fits in L2 cache on most CPUs)
    // - Smaller values increase cache efficiency but add loop overhead
    // - Larger values may cause cache misses, reducing performance
    const STRIP_HEIGHT: usize = 256;

    // Temporary buffer for horizontal sums (u16 fits since max horizontal sum = 255 * (2R+1))
    // For R=15: 255 * 31 = 7905, well within u16::MAX = 65535
    let temp_h = STRIP_HEIGHT + 2 * radius;
    let mut temp_buffer = vec![0u16; width * 3 * temp_h];

    // Process image in strips
    let mut y_base = 0;
    while y_base < height {
        let strip_h = (height - y_base).min(STRIP_HEIGHT);

        // Pre-compute fixed-point multipliers and window sizes for all x positions
        // This eliminates per-pixel branch prediction and calculation overhead
        let mut window_size_at = vec![0usize; width];
        let mut mul_at = vec![0u64; width];
        for x in 0..width {
            let h_left = x.saturating_sub(radius);
            let h_right = (x + radius).min(width - 1);
            let h_width = (h_right - h_left + 1) as usize;
            window_size_at[x] = h_width;
            mul_at[x] = ((1u64 << 32) + h_width as u64 / 2) / h_width as u64;
        }

        // ============================================================
        // PASS 1: HORIZONTAL (NEON Row-Parallel, O(1) per pixel)
        // Process 4 rows simultaneously using uint32x4_t accumulators
        // Key: vaddq_u32/vsubq_u32 updates all 4 rows in 1 cycle!
        // ============================================================

        let mut dy = 0;
        while dy + 3 < strip_h {
            // Process 4 rows at once (dy, dy+1, dy+2, dy+3)
            let img_y = [
                (y_base + dy).min(height - 1),
                (y_base + dy + 1).min(height - 1),
                (y_base + dy + 2).min(height - 1),
                (y_base + dy + 3).min(height - 1),
            ];

            let temp_row_base = [
                dy * width * 3,
                (dy + 1) * width * 3,
                (dy + 2) * width * 3,
                (dy + 3) * width * 3,
            ];

            // Process each channel independently
            for c in 0..3 {
                unsafe {
                    // SAFETY: All pointer arithmetic is bounded by:
                    // - img_y values are clamped to (0, height-1)
                    // - Offsets (c and x*3) are within stride bounds
                    // - temp_row_base is computed from valid indices (dy, width)
                    // Base pointers for the 4 rows (channel c offset)
                    let row_ptrs = [
                        data.as_ptr().add(img_y[0] * width * 3 + c),
                        data.as_ptr().add(img_y[1] * width * 3 + c),
                        data.as_ptr().add(img_y[2] * width * 3 + c),
                        data.as_ptr().add(img_y[3] * width * 3 + c),
                    ];

                    let dst_ptrs = [
                        temp_buffer.as_mut_ptr().add(temp_row_base[0] + c),
                        temp_buffer.as_mut_ptr().add(temp_row_base[1] + c),
                        temp_buffer.as_mut_ptr().add(temp_row_base[2] + c),
                        temp_buffer.as_mut_ptr().add(temp_row_base[3] + c),
                    ];

                    // === WARMUP: Initialize SIMD accumulator ===
                    // acc holds [sum_row0, sum_row1, sum_row2, sum_row3]
                    let mut acc = vdupq_n_u32(0);

                    // Sum first (radius+1) pixels for each row
                    for i in 0..=radius.min(width - 1) {
                        let idx = i * 3;

                        // Load 4 values (one from each row) into vector
                        let mut v_in = vdupq_n_u32(0);
                        v_in = vsetq_lane_u32(*row_ptrs[0].add(idx) as u32, v_in, 0);
                        v_in = vsetq_lane_u32(*row_ptrs[1].add(idx) as u32, v_in, 1);
                        v_in = vsetq_lane_u32(*row_ptrs[2].add(idx) as u32, v_in, 2);
                        v_in = vsetq_lane_u32(*row_ptrs[3].add(idx) as u32, v_in, 3);

                        // SIMD add: updates all 4 accumulators in 1 cycle
                        acc = vaddq_u32(acc, v_in);
                    }

                    // === HOT LOOP: Slide window across all 4 rows in parallel ===
                    for x in 0..width {
                        // A. NORMALIZE & STORE (extract lanes, multiply, store)
                        let mul = mul_at[x] as u64; // Pre-computed, no division!

                        let s0 = vgetq_lane_u32(acc, 0);
                        let s1 = vgetq_lane_u32(acc, 1);
                        let s2 = vgetq_lane_u32(acc, 2);
                        let s3 = vgetq_lane_u32(acc, 3);

                        // Fixed-point with rounding: ((sum * mul) + (1 << 31)) >> 32
                        *dst_ptrs[0].add(x * 3) = (((s0 as u64 * mul) + (1u64 << 31)) >> 32) as u16;
                        *dst_ptrs[1].add(x * 3) = (((s1 as u64 * mul) + (1u64 << 31)) >> 32) as u16;
                        *dst_ptrs[2].add(x * 3) = (((s2 as u64 * mul) + (1u64 << 31)) >> 32) as u16;
                        *dst_ptrs[3].add(x * 3) = (((s3 as u64 * mul) + (1u64 << 31)) >> 32) as u16;

                        // B. UPDATE ACCUMULATORS (The Speedup!)
                        // Prepare vectors for entering and leaving pixels
                        let mut v_enter = vdupq_n_u32(0);
                        let mut v_leave = vdupq_n_u32(0);

                        // Load entering pixel (right side of window)
                        let x_in = x + radius + 1;
                        if x_in < width {
                            let idx = x_in * 3;
                            v_enter = vsetq_lane_u32(*row_ptrs[0].add(idx) as u32, v_enter, 0);
                            v_enter = vsetq_lane_u32(*row_ptrs[1].add(idx) as u32, v_enter, 1);
                            v_enter = vsetq_lane_u32(*row_ptrs[2].add(idx) as u32, v_enter, 2);
                            v_enter = vsetq_lane_u32(*row_ptrs[3].add(idx) as u32, v_enter, 3);
                        }

                        // Load leaving pixel (left side of window)
                        if x >= radius {
                            let x_out = x - radius;
                            let idx = x_out * 3;
                            v_leave = vsetq_lane_u32(*row_ptrs[0].add(idx) as u32, v_leave, 0);
                            v_leave = vsetq_lane_u32(*row_ptrs[1].add(idx) as u32, v_leave, 1);
                            v_leave = vsetq_lane_u32(*row_ptrs[2].add(idx) as u32, v_leave, 2);
                            v_leave = vsetq_lane_u32(*row_ptrs[3].add(idx) as u32, v_leave, 3);
                        }

                        // SIMD math: Update all 4 rows in 1 cycle each!
                        acc = vaddq_u32(acc, v_enter);
                        acc = vsubq_u32(acc, v_leave);
                    }
                }
            }

            dy += 4;
        }

        // Handle remaining rows (1-3 rows) with scalar fallback
        while dy < strip_h {
            let img_y = (y_base + dy).min(height - 1);
            let temp_row_base = dy * width * 3;

            for c in 0..3 {
                let row_offset = img_y * width * 3 + c;

                let mut sum: u32 = 0;
                for x in 0..=radius.min(width - 1) {
                    sum += data[row_offset + x * 3] as u32;
                }

                for x in 0..width {
                    // Use pre-computed multiplier for scalar fallback
                    let mul = mul_at[x];
                    // Fixed-point division with proper rounding: ((sum * mul) + (1 << 31)) >> 32
                    let avg = (((sum as u64 * mul) + (1u64 << 31)) >> 32) as u16;
                    temp_buffer[temp_row_base + x * 3 + c] = avg;

                    let x_in = x + radius + 1;
                    let x_out = x.saturating_sub(radius);

                    if x_in < width {
                        sum += data[row_offset + x_in * 3] as u32;
                    }
                    if x >= radius {
                        sum -= data[row_offset + x_out * 3] as u32;
                    }
                }
            }

            dy += 1;
        }

        // ============================================================
        // PASS 2: VERTICAL (NEON SIMD with SLIDING WINDOW, O(1) per pixel)
        // ============================================================
        for c in 0..3 {
            let mut x = 0;

            // SIMD: Process 8 pixels at a time (using two u32x4 accumulators)
            while x + 8 <= width {
                unsafe {
                    // === PROLOGUE: WARMUP ACCUMULATOR (O(R) once per column) ===
                    // For the FIRST pixel in this strip, compute its actual vertical window range
                    let first_out_y = y_base;
                    let v_top = first_out_y.saturating_sub(radius);
                    let v_bottom = (first_out_y + radius).min(height - 1);

                    let mut acc_lo = vdupq_n_u32(0);
                    let mut acc_hi = vdupq_n_u32(0);

                    // Sum only the rows that are in the actual vertical window
                    for img_row in v_top..=v_bottom {
                        let temp_row = img_row - y_base;  // Map to temp buffer index
                        let base_ptr = temp_buffer.as_ptr().add(temp_row * width * 3 + x * 3 + c);
                        let v16 = load_u16x8_strided(base_ptr, 3);

                        let v16_lo = vget_low_u16(v16);
                        let v16_hi = vget_high_u16(v16);
                        acc_lo = vaddw_u16(acc_lo, v16_lo);
                        acc_hi = vaddw_u16(acc_hi, v16_hi);
                    }

                    // === STEP 2: SLIDE VERTICALLY DOWN THE STRIP (O(1) per row) ===
                    for dy in 0..strip_h {
                        let out_y = y_base + dy;

                        // Normalize and store current accumulator
                        let v0 = vgetq_lane_u32(acc_lo, 0);
                        let v1 = vgetq_lane_u32(acc_lo, 1);
                        let v2 = vgetq_lane_u32(acc_lo, 2);
                        let v3 = vgetq_lane_u32(acc_lo, 3);
                        let v4 = vgetq_lane_u32(acc_hi, 0);
                        let v5 = vgetq_lane_u32(acc_hi, 1);
                        let v6 = vgetq_lane_u32(acc_hi, 2);
                        let v7 = vgetq_lane_u32(acc_hi, 3);

                        // Compute ACTUAL window height for each pixel (handles edges)
                        // For separable box blur: horizontal pass divided by h_width,
                        // so vertical pass only needs to divide by v_height
                        let mut v_heights = [0u32; 8];
                        for i in 0..8 {
                            let v_top = out_y.saturating_sub(radius);
                            let v_bottom = (out_y + radius).min(height - 1);
                            v_heights[i] = (v_bottom - v_top + 1) as u32;
                        }

                        // Normalize each pixel by vertical height only
                        let out_vals = [
                            (((v0 * get_reciprocal(v_heights[0])) + (1u32 << 15)) >> 16) as u8,
                            (((v1 * get_reciprocal(v_heights[1])) + (1u32 << 15)) >> 16) as u8,
                            (((v2 * get_reciprocal(v_heights[2])) + (1u32 << 15)) >> 16) as u8,
                            (((v3 * get_reciprocal(v_heights[3])) + (1u32 << 15)) >> 16) as u8,
                            (((v4 * get_reciprocal(v_heights[4])) + (1u32 << 15)) >> 16) as u8,
                            (((v5 * get_reciprocal(v_heights[5])) + (1u32 << 15)) >> 16) as u8,
                            (((v6 * get_reciprocal(v_heights[6])) + (1u32 << 15)) >> 16) as u8,
                            (((v7 * get_reciprocal(v_heights[7])) + (1u32 << 15)) >> 16) as u8,
                        ];

                        // Store 8 pixels (stride-3 for interleaved RGB)
                        let out_ptr = data.as_mut_ptr().add((out_y * width + x) * 3 + c);
                        *out_ptr.offset(0) = out_vals[0];
                        *out_ptr.offset(3) = out_vals[1];
                        *out_ptr.offset(6) = out_vals[2];
                        *out_ptr.offset(9) = out_vals[3];
                        *out_ptr.offset(12) = out_vals[4];
                        *out_ptr.offset(15) = out_vals[5];
                        *out_ptr.offset(18) = out_vals[6];
                        *out_ptr.offset(21) = out_vals[7];

                        // === SLIDE WINDOW: add row entering, subtract row leaving ===
                        // Rows are relative to out_y (current output row in image)
                        let row_leaving = out_y.saturating_sub(radius);   // Row at top of current window
                        let row_entering = out_y + radius + 1;             // Row entering bottom of window

                        // Add entering row (if valid in image)
                        if row_entering < height {
                            let temp_row = row_entering - y_base;  // Map to temp buffer
                            let base_ptr = temp_buffer.as_ptr().add(temp_row * width * 3 + x * 3 + c);
                            let v_in = load_u16x8_strided(base_ptr, 3);
                            let v_in_lo = vget_low_u16(v_in);
                            let v_in_hi = vget_high_u16(v_in);
                            acc_lo = vaddw_u16(acc_lo, v_in_lo);
                            acc_hi = vaddw_u16(acc_hi, v_in_hi);
                        }

                        // Subtract leaving row (if valid in image)
                        if out_y >= radius {
                            let temp_row = row_leaving - y_base;  // Map to temp buffer
                            let base_ptr = temp_buffer.as_ptr().add(temp_row * width * 3 + x * 3 + c);
                            let v_out = load_u16x8_strided(base_ptr, 3);
                            let v_out_lo = vget_low_u16(v_out);
                            let v_out_hi = vget_high_u16(v_out);
                            acc_lo = vsubw_u16(acc_lo, v_out_lo);
                            acc_hi = vsubw_u16(acc_hi, v_out_hi);
                        }
                    }
                }
                x += 8;
            }

            // Scalar tail for remaining columns
            for x in x..width {
                // Warmup accumulator: compute actual vertical window for first pixel
                let first_out_y = y_base;
                let v_top = first_out_y.saturating_sub(radius);
                let v_bottom = (first_out_y + radius).min(height - 1);

                let mut acc: u32 = 0;
                for img_row in v_top..=v_bottom {
                    let temp_row = img_row - y_base;
                    acc += temp_buffer[temp_row * width * 3 + x * 3 + c] as u32;
                }

                // Slide vertically
                for dy in 0..strip_h {
                    let out_y = y_base + dy;

                    // Normalize
                    let h_left = x.saturating_sub(radius);
                    let h_right = (x + radius).min(width - 1);
                    let h_width = h_right - h_left + 1;

                    let v_top = out_y.saturating_sub(radius);
                    let v_bottom = (out_y + radius).min(height - 1);
                    let v_height = v_bottom - v_top + 1;

                    let area = (h_width * v_height) as u32;
                    let recip = get_reciprocal(area);
                    let out_val = (((acc * recip) + (1u32 << 15)) >> 16) as u8;
                    data[(out_y * width + x) * 3 + c] = out_val;

                    // Slide window: rows are relative to out_y (image row)
                    let row_leaving = out_y.saturating_sub(radius);
                    let row_entering = out_y + radius + 1;

                    // Add entering row (if valid)
                    if row_entering < height {
                        let temp_row = row_entering - y_base;
                        acc += temp_buffer[temp_row * width * 3 + x * 3 + c] as u32;
                    }

                    // Subtract leaving row (if valid)
                    if out_y >= radius {
                        let temp_row = row_leaving - y_base;
                        acc -= temp_buffer[temp_row * width * 3 + x * 3 + c] as u32;
                    }
                }
            }
        }

        y_base += STRIP_HEIGHT;
    }
}

/// NEON helper: Load 8 u16 values with stride (for interleaved RGB)
///
/// # Safety
/// Caller must ensure that:
/// - base points to valid memory for at least (7 * stride + 1) u16 elements
/// - stride is the appropriate byte distance (3 for RGB interleaved data)
#[inline(always)]
unsafe fn load_u16x8_strided(base: *const u16, stride: isize) -> uint16x8_t {
    // Load low 4 and high 4 separately, then combine
    let mut low = [0u16; 4];
    let mut high = [0u16; 4];

    // SAFETY: Caller guarantees base has enough memory for all offsets
    for i in 0..4 {
        low[i] = *base.offset(i as isize * stride);
        high[i] = *base.offset((i + 4) as isize * stride);
    }

    // SAFETY: transmute is safe here because [u16; 4] and uint16x4_t have the same
    // size and memory layout (both are 4 16-bit values)
    let low_vec = std::mem::transmute::<[u16; 4], uint16x4_t>(low);
    let high_vec = std::mem::transmute::<[u16; 4], uint16x4_t>(high);

    vcombine_u16(low_vec, high_vec)
}

/// NEON helper: Get fixed-point reciprocal for a given window size
/// Returns reciprocal in Q16 fixed-point format for division
/// Formula: result = (value * recip + rounding) >> 16
#[inline(always)]
fn get_reciprocal(window_size: u32) -> u32 {
    // Fixed-point reciprocal: round((1 << 16) / window_size)
    // For proper rounding: result = (value * recip + (1 << 15)) >> 16
    ((1u32 << 16) + window_size / 2) / window_size
}
