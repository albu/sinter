// Perreault & Hébert (2007) vectorized coarse/fine sliding column-histogram
//
// Maintains 16-bin coarse and fine histograms per column.
// Single instruction horizontal updates (vaddq_u8 / vsubq_u8 in ARM NEON SIMD).
// Mathematically exact to OpenCV (0% error), O(1) per pixel.
//
// Note: Reference histogram implementations used for validation in unit tests.
#![allow(dead_code)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::core::FusableImage;

/// Apply 3x3 median blur using Huang's sliding histogram algorithm (fallback)
pub fn apply_median_blur_3x3_huang(image: &mut FusableImage) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    if height < 3 || width < 3 {
        return;
    }

    if channels == 3 {
        apply_median_blur_3x3_huang_rgb_scalar(data, width, height);
    } else {
        apply_median_blur_3x3_huang_grayscale(data, width, height);
    }
}

/// Apply 5x5 median blur using vectorized sliding column-histogram algorithm.
pub fn apply_median_blur_5x5_huang(image: &mut FusableImage) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    if width == 0 || height == 0 {
        return;
    }

    let mut output = vec![0u8; data.len()];
    let stride = width * channels;

    let pad_w = width + 4;
    let mut row_buf = vec![0u8; pad_w];
    let mut row_buf_out = vec![0u8; pad_w];
    let mut row_buf_in = vec![0u8; pad_w];

    // col_coarse: pad_w * 16 contiguous bytes
    let mut col_coarse = vec![0u8; pad_w * 16];
    // col_fine: 16 * pad_w * 16 contiguous bytes (bin * pad_w * 16 + col * 16 + f)
    let mut col_fine = vec![0u8; 16 * pad_w * 16];

    #[inline(always)]
    fn fill_padded_row(
        data: &[u8],
        sy: usize,
        c: usize,
        width: usize,
        channels: usize,
        stride: usize,
        row_buf: &mut [u8],
    ) {
        let row_start = sy * stride + c;
        let first_val = data[row_start];
        row_buf[0] = first_val;
        row_buf[1] = first_val;
        for x in 0..width {
            row_buf[x + 2] = data[row_start + x * channels];
        }
        let last_val = data[row_start + (width - 1) * channels];
        row_buf[width + 2] = last_val;
        row_buf[width + 3] = last_val;
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn prefix_sum_u8(v: uint8x16_t) -> uint8x16_t {
        let zero = vdupq_n_u8(0);
        let s1 = vaddq_u8(v, vextq_u8(zero, v, 15));
        let s2 = vaddq_u8(s1, vextq_u8(zero, s1, 14));
        let s4 = vaddq_u8(s2, vextq_u8(zero, s2, 12));
        vaddq_u8(s4, vextq_u8(zero, s4, 8))
    }

    let bin_stride = pad_w * 16;

    for c in 0..channels {
        col_coarse.fill(0);
        col_fine.fill(0);

        let coarse_ptr = col_coarse.as_mut_ptr();
        let fine_ptr_base = col_fine.as_mut_ptr();

        // Initialize column histograms for y = 0
        for dy in -2i32..=2i32 {
            let sy = dy.clamp(0, height as i32 - 1) as usize;
            fill_padded_row(data, sy, c, width, channels, stride, &mut row_buf);
            for col in 0..pad_w {
                let v = row_buf[col] as usize;
                let b = v >> 4;
                let f = v & 0x0F;
                unsafe {
                    *coarse_ptr.add(col * 16 + b) += 1;
                    *fine_ptr_base.add(b * bin_stride + col * 16 + f) += 1;
                }
            }
        }

        for y in 0..height {
            if y > 0 {
                let sy_out = (y as i32 - 3).clamp(0, height as i32 - 1) as usize;
                let sy_in = (y as i32 + 2).clamp(0, height as i32 - 1) as usize;
                if sy_out != sy_in {
                    fill_padded_row(data, sy_out, c, width, channels, stride, &mut row_buf_out);
                    fill_padded_row(data, sy_in, c, width, channels, stride, &mut row_buf_in);
                    for col in 0..pad_w {
                        let vo = row_buf_out[col] as usize;
                        let bo = vo >> 4;
                        let fo = vo & 0x0F;

                        let vi = row_buf_in[col] as usize;
                        let bi = vi >> 4;
                        let fi = vi & 0x0F;

                        unsafe {
                            *coarse_ptr.add(col * 16 + bo) -= 1;
                            *fine_ptr_base.add(bo * bin_stride + col * 16 + fo) -= 1;

                            *coarse_ptr.add(col * 16 + bi) += 1;
                            *fine_ptr_base.add(bi * bin_stride + col * 16 + fi) += 1;
                        }
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                let mut win_coarse_vec = vaddq_u8(
                    vaddq_u8(
                        vld1q_u8(coarse_ptr),
                        vld1q_u8(coarse_ptr.add(16)),
                    ),
                    vaddq_u8(
                        vaddq_u8(
                            vld1q_u8(coarse_ptr.add(32)),
                            vld1q_u8(coarse_ptr.add(48)),
                        ),
                        vld1q_u8(coarse_ptr.add(64)),
                    ),
                );

                let out_row_ptr = output.as_mut_ptr().add(y * stride + c);

                for x in 0..width {
                    if x > 0 {
                        let c_in = vld1q_u8(coarse_ptr.add((x + 4) * 16));
                        let c_out = vld1q_u8(coarse_ptr.add((x - 1) * 16));
                        win_coarse_vec = vsubq_u8(vaddq_u8(win_coarse_vec, c_in), c_out);
                    }

                    let prefix = prefix_sum_u8(win_coarse_vec);
                    let cmp = vcgtq_u8(vdupq_n_u8(13), prefix);
                    let target_b = vaddlvq_u8(vshrq_n_u8(cmp, 7)) as usize;

                    let mut prefix_arr = [0u8; 16];
                    vst1q_u8(prefix_arr.as_mut_ptr(), prefix);
                    let acc_before = if target_b > 0 { prefix_arr[target_b - 1] } else { 0 };

                    let b_fine_ptr = fine_ptr_base.add(target_b * bin_stride + x * 16);
                    let f0 = vld1q_u8(b_fine_ptr);
                    let f1 = vld1q_u8(b_fine_ptr.add(16));
                    let f2 = vld1q_u8(b_fine_ptr.add(32));
                    let f3 = vld1q_u8(b_fine_ptr.add(48));
                    let f4 = vld1q_u8(b_fine_ptr.add(64));
                    let win_fine_vec = vaddq_u8(
                        vaddq_u8(f0, f1),
                        vaddq_u8(vaddq_u8(f2, f3), f4),
                    );

                    let prefix_f = prefix_sum_u8(win_fine_vec);
                    let cmp_f = vcgtq_u8(vdupq_n_u8(13 - acc_before), prefix_f);
                    let target_f = vaddlvq_u8(vshrq_n_u8(cmp_f, 7)) as u8;

                    *out_row_ptr.add(x * channels) = ((target_b as u8) << 4) | target_f;
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut win_coarse = [0u8; 16];
                for col in 0..5 {
                    for b in 0..16 {
                        win_coarse[b] += col_coarse[col * 16 + b];
                    }
                }

                for x in 0..width {
                    if x > 0 {
                        let in_off = (x + 4) * 16;
                        let out_off = (x - 1) * 16;
                        for b in 0..16 {
                            win_coarse[b] = win_coarse[b] + col_coarse[in_off + b] - col_coarse[out_off + b];
                        }
                    }

                    let mut acc = 0u8;
                    let mut target_b = 0usize;
                    let mut acc_before = 0u8;
                    for b in 0..16 {
                        let count = win_coarse[b];
                        if acc + count >= 13 {
                            target_b = b;
                            acc_before = acc;
                            break;
                        }
                        acc += count;
                    }

                    let mut win_fine = [0u8; 16];
                    let b_offset = target_b * bin_stride;
                    for dx in 0..5 {
                        let col_off = b_offset + (x + dx) * 16;
                        for f in 0..16 {
                            win_fine[f] += col_fine[col_off + f];
                        }
                    }

                    let mut acc_f = acc_before;
                    let mut target_f = 0u8;
                    for f in 0..16 {
                        let count = win_fine[f];
                        if acc_f + count >= 13 {
                            target_f = f as u8;
                            break;
                        }
                        acc_f += count;
                    }

                    output[y * stride + x * channels + c] = ((target_b as u8) << 4) | target_f;
                }
            }
        }
    }

    data.copy_from_slice(&output);
}

fn apply_median_blur_3x3_huang_grayscale(data: &mut [u8], width: usize, height: usize) {
    let mut output = vec![0u8; data.len()];

    for y in 0..height {
        for x in 0..width {
            let mut hist = [0u16; 256];
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let py = (y as i32 + dy).clamp(0, (height - 1) as i32) as usize;
                    let px = (x as i32 + dx).clamp(0, (width - 1) as i32) as usize;
                    let val = data[py * width + px];
                    hist[val as usize] += 1;
                }
            }

            let mut count = 0u16;
            let target = 5;
            for i in 0..256 {
                count += hist[i];
                if count >= target {
                    output[y * width + x] = i as u8;
                    break;
                }
            }
        }
    }

    data.copy_from_slice(&output);
}

fn apply_median_blur_3x3_huang_rgb_scalar(data: &mut [u8], width: usize, height: usize) {
    let mut output = vec![0u8; data.len()];

    for c in 0..3 {
        let mut hist = [0u16; 256];
        let mut median_lt = 0;
        let mut median_gt = 0;
        let mut median = 0u8;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let py = dy.clamp(0, (height - 1) as i32) as usize;
                let px = dx.clamp(0, (width - 1) as i32) as usize;
                let val = data[(py * width + px) * 3 + c];
                hist[val as usize] += 1;
            }
        }

        find_initial_median(&hist, &mut median, &mut median_lt, &mut median_gt, 9);
        output[c] = median;

        for x in 1..width {
            slide_window_horizontal_rgb(data, width, height, x, 0, c, &mut hist, &mut median_lt, &mut median_gt, &mut median);
            output[x * 3 + c] = median;
        }

        for y in 1..height {
            slide_window_vertical_rgb(data, width, height, 0, y, c, &mut hist, &mut median_lt, &mut median_gt, &mut median);
            output[(y * width) * 3 + c] = median;

            for x in 1..width {
                slide_window_horizontal_rgb(data, width, height, x, y, c, &mut hist, &mut median_lt, &mut median_gt, &mut median);
                output[(y * width + x) * 3 + c] = median;
            }
        }
    }

    data.copy_from_slice(&output);
}

fn slide_window_horizontal_rgb(
    data: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    c: usize,
    hist: &mut [u16; 256],
    median_lt: &mut u16,
    median_gt: &mut u16,
    median: &mut u8,
) {
    for dy in -1i32..=1 {
        let py = (y as i32 + dy).clamp(0, (height - 1) as i32) as usize;
        let px_out = (x as i32 - 2).clamp(0, (width - 1) as i32) as usize;
        let px_in = (x as i32 + 1).clamp(0, (width - 1) as i32) as usize;

        if px_out != px_in {
            let val_out = data[(py * width + px_out) * 3 + c];
            update_histogram_remove(hist, val_out, median_lt, median_gt, *median);
        }

        let val_in = data[(py * width + px_in) * 3 + c];
        update_histogram_add(hist, val_in, median_lt, median_gt, *median);
    }

    update_median(hist, median, median_lt, median_gt, 9);
}

fn slide_window_vertical_rgb(
    data: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    c: usize,
    hist: &mut [u16; 256],
    median_lt: &mut u16,
    median_gt: &mut u16,
    median: &mut u8,
) {
    for dx in -1i32..=1 {
        let px = (x as i32 + dx).clamp(0, (width - 1) as i32) as usize;
        let py_out = (y as i32 - 2).clamp(0, (height - 1) as i32) as usize;
        let py_in = (y as i32 + 1).clamp(0, (height - 1) as i32) as usize;

        if py_out != py_in {
            let val_out = data[(py_out * width + px) * 3 + c];
            update_histogram_remove(hist, val_out, median_lt, median_gt, *median);
        }

        let val_in = data[(py_in * width + px) * 3 + c];
        update_histogram_add(hist, val_in, median_lt, median_gt, *median);
    }

    update_median(hist, median, median_lt, median_gt, 9);
}

/// Find initial median from histogram
///
/// Walks from 0 upward to find the median value.
/// Also computes counts below and above median.
fn find_initial_median(
    hist: &[u16; 256],
    median: &mut u8,
    median_lt: &mut u16,
    median_gt: &mut u16,
    window_size: u16,
) {
    let mut count = 0u16;
    let target = window_size / 2 + 1; // For 9 pixels, median is 5th

    for i in 0..256 {
        count += hist[i];
        if count >= target {
            *median = i as u8;
            break;
        }
    }

    // Compute counts below and above median
    *median_lt = hist[..*median as usize].iter().sum();
    *median_gt = hist[(*median as usize + 1)..].iter().sum();
}

/// Update histogram when removing a value
///
/// Updates histogram and median_lt/median_gt counts.
fn update_histogram_remove(
    hist: &mut [u16; 256],
    val: u8,
    median_lt: &mut u16,
    median_gt: &mut u16,
    median: u8,
) {
    // Only decrement if the histogram value is positive
    if hist[val as usize] > 0 {
        hist[val as usize] -= 1;

        if val < median {
            *median_lt = median_lt.saturating_sub(1);
        } else if val > median {
            *median_gt = median_gt.saturating_sub(1);
        }
    }
}

/// Update histogram when adding a value
///
/// Updates histogram and median_lt/median_gt counts.
fn update_histogram_add(
    hist: &mut [u16; 256],
    val: u8,
    median_lt: &mut u16,
    median_gt: &mut u16,
    median: u8,
) {
    hist[val as usize] += 1;

    if val < median {
        *median_lt += 1;
    } else if val > median {
        *median_gt += 1;
    }
}

/// Update median from current histogram state
///
/// Walks the histogram from the current median position to find the new median.
/// This is O(1) amortized because the median moves slowly.
fn update_median(
    hist: &[u16; 256],
    median: &mut u8,
    median_lt: &mut u16,
    median_gt: &mut u16,
    window_size: u16,
) {
    let target = window_size / 2 + 1;

    // If median_lt is too small, move median up
    while *median_lt >= target {
        *median_gt += hist[*median as usize];
        *median = median.saturating_sub(1);
        *median_lt -= hist[*median as usize];
    }

    // If median_lt + hist[median] < target, move median down
    while *median_lt + hist[*median as usize] < target {
        *median_lt += hist[*median as usize];
        *median += 1;
        *median_gt = median_gt.saturating_sub(hist[*median as usize]);
    }

    // Recompute median_gt for correctness
    let total = *median_lt as u32 + hist[*median as usize] as u32;
    *median_gt = if total <= window_size as u32 {
        window_size - *median_lt - hist[*median as usize]
    } else {
        0
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force exact 5x5 median reference (per-pixel sort of the window).
    fn reference_median_5x5(data: &[u8], width: usize, height: usize, channels: usize) -> Vec<u8> {
        let mut out = vec![0u8; data.len()];
        let stride = width * channels;
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let mut win = Vec::with_capacity(25);
                    for dy in -2i32..=2 {
                        let yy = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                        for dx in -2i32..=2 {
                            let xx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                            win.push(data[yy * stride + xx * channels + c]);
                        }
                    }
                    win.sort_unstable();
                    out[y * stride + x * channels + c] = win[win.len() / 2];
                }
            }
        }
        out
    }

    #[test]
    fn test_median_5x5_exact_matches_reference() {
        for (w, h, ch) in [(1usize, 1usize, 3usize), (2, 2, 1), (5, 5, 1), (16, 13, 3), (32, 32, 3)] {
            let mut data: Vec<u8> = (0..w * h * ch)
                .map(|i| ((i as u64 * 2654435761) % 256) as u8)
                .collect();
            let expected = reference_median_5x5(&data, w, h, ch);

            let mut img = crate::core::FusableImage::new(&mut data, w, h, ch);
            apply_median_blur_5x5_huang(&mut img);

            let mut mismatches = 0usize;
            let mut max_diff = 0i32;
            for i in 0..data.len() {
                let diff = (data[i] as i32 - expected[i] as i32).abs();
                if diff > 0 {
                    mismatches += 1;
                    max_diff = max_diff.max(diff);
                    if w == 2 && h == 2 && mismatches <= 4 {
                        eprintln!(
                            "  idx={} (x={}, y={}): got={} expected={}",
                            i,
                            i % w,
                            i / w,
                            data[i],
                            expected[i]
                        );
                    }
                }
            }
            assert_eq!(
                mismatches,
                0,
                "5x5 median mismatch for {}x{}x{}: {} mismatches, max_diff={}",
                w,
                h,
                ch,
                mismatches,
                max_diff
            );
        }
    }

    #[test]
    fn test_median_5x5_salt_pepper_exact() {
        // A single isolated 0 in a 5x5 block of 128s must become 128 exactly.
        let mut data = vec![128u8; 25 * 3];
        data[12 * 3] = 0; // center pixel, R channel
        let mut img = crate::core::FusableImage::new(&mut data, 5, 5, 3);
        apply_median_blur_5x5_huang(&mut img);
        assert_eq!(img.data[12 * 3], 128);
        assert_eq!(img.data[12 * 3 + 1], 128);
        assert_eq!(img.data[12 * 3 + 2], 128);
    }

    #[test]
    fn test_median_5x5_grayscale_large_exact() {
        // Larger grayscale image (the tiny grayscale cases miss a bug).
        let (w, h) = (64usize, 64usize);
        let mut data: Vec<u8> = (0..w * h)
            .map(|i| ((i as u64 * 2654435761) % 256) as u8)
            .collect();
        let expected = reference_median_5x5(&data, w, h, 1);

        let mut img = crate::core::FusableImage::new(&mut data, w, h, 1);
        apply_median_blur_5x5_huang(&mut img);

        let mut mismatches = 0usize;
        let mut max_diff = 0i32;
        let mut first = None;
        for i in 0..data.len() {
            let diff = (data[i] as i32 - expected[i] as i32).abs();
            if diff > 0 {
                mismatches += 1;
                max_diff = max_diff.max(diff);
                if first.is_none() {
                    first = Some((i, data[i], expected[i]));
                }
            }
        }
        if let Some((i, got, exp)) = first {
            eprintln!("  first mismatch idx={} (x={}, y={}): got={} expected={}", i, i % w, i / w, got, exp);
        }
        assert_eq!(
            mismatches,
            0,
            "5x5 grayscale {}x{} mismatch: {} mismatches, max_diff={}",
            w,
            h,
            mismatches,
            max_diff
        );
    }

    #[test]
    fn test_median_blur_3x3_huang_constant() {
        let mut data = vec![128u8; 9 * 3]; // 3x3 RGB
        let mut img = crate::core::FusableImage::new(&mut data, 3, 3, 3);

        apply_median_blur_3x3_huang(&mut img);

        // All pixels should remain 128
        assert!(img.data.iter().all(|&x| x == 128));
    }

    #[test]
    fn test_median_blur_3x3_huang_salt_pepper() {
        let mut data = vec![
            128u8, 128u8, 128u8,  128u8, 128u8, 128u8,  128u8, 128u8, 128u8,
            128u8, 128u8, 128u8,  0u8, 0u8, 0u8,         128u8, 128u8, 128u8,
            128u8, 128u8, 128u8,  128u8, 128u8, 128u8,  128u8, 128u8, 128u8,
        ];
        let mut img = crate::core::FusableImage::new(&mut data, 3, 3, 3);

        apply_median_blur_3x3_huang(&mut img);

        // Center pixel should be ~128 (salt-pepper removed)
        let center_idx = 4 * 3 + 1; // Center pixel, G channel
        assert!((img.data[center_idx] as i16 - 128).abs() < 30);
    }

    #[test]
    fn test_median_blur_3x3_huang_larger_image() {
        let mut data = vec![128u8; 100 * 3]; // 10x10 RGB
        // Add some salt-pepper noise
        data[10 * 3] = 0;
        data[50 * 3 + 1] = 255;
        data[90 * 3 + 2] = 0;

        let mut img = crate::core::FusableImage::new(&mut data, 10, 10, 3);

        apply_median_blur_3x3_huang(&mut img);

        // Salt-pepper should be reduced
        assert_ne!(img.data[10 * 3], 0);
        assert_ne!(img.data[50 * 3 + 1], 255);
        assert_ne!(img.data[90 * 3 + 2], 0);
    }

    #[test]
    fn test_find_initial_median() {
        let mut hist = [0u16; 256];
        hist[100] = 9; // All 9 pixels are value 100

        let mut median = 0u8;
        let mut median_lt = 0;
        let mut median_gt = 0;

        find_initial_median(&hist, &mut median, &mut median_lt, &mut median_gt, 9);

        assert_eq!(median, 100);
        assert_eq!(median_lt, 0);
        assert_eq!(median_gt, 0);
    }

    #[test]
    fn test_update_median() {
        // Histogram: values 0-8 each appear once
        let mut hist = [0u16; 256];
        for i in 0..9 {
            hist[i] = 1;
        }

        let mut median = 4u8;
        let mut median_lt = 4;
        let mut median_gt = 4;

        update_median(&hist, &mut median, &mut median_lt, &mut median_gt, 9);

        // Median should be 4 (5th element in 0-8)
        assert_eq!(median, 4);
    }
}
