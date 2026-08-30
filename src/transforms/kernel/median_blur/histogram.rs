// Exact 5x5 median blur via a sliding column-histogram with median-position
// tracking.
//
// Maintains a per-column fine histogram (16 bins per coarse bin) plus the
// 5-column window's coarse and fine histograms. Instead of recomputing prefix
// sums for every pixel (which was ~18 ms at 1024x1024), the median position is
// tracked incrementally with a below-count (`count_lt`), so each pixel costs
// only the 5 outgoing + 5 incoming value updates plus an amortized O(1) median
// walk. Bit-exact with OpenCV `medianBlur(ksize=5)` (replicate borders).

use crate::core::FusableImage;

/// Fill a padded row buffer (2-pixel replicate border on each side) for
/// channel `c`.
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
    let first = data[row_start];
    row_buf[0] = first;
    row_buf[1] = first;
    for x in 0..width {
        row_buf[x + 2] = data[row_start + x * channels];
    }
    let last = data[row_start + (width - 1) * channels];
    row_buf[width + 2] = last;
    row_buf[width + 3] = last;
}

/// Rebuild the window's fine histogram (16 bins inside one coarse bin) for the
/// window at padded-column offset `x` (columns x..x+4) from the per-column
/// fine histograms.
#[inline(always)]
fn rebuild_fine(
    win_fine: &mut [u8; 16],
    col_fine: &[u8],
    bin: usize,
    x: usize,
    bin_stride: usize,
) {
    let base = bin * bin_stride + x * 16;
    for f in 0..16 {
        win_fine[f] = col_fine[base + f]
            + col_fine[base + 16 + f]
            + col_fine[base + 32 + f]
            + col_fine[base + 48 + f]
            + col_fine[base + 64 + f];
    }
}

/// Apply exact 5x5 median blur using a sliding column histogram with
/// median-position tracking.
pub fn apply_median_blur_5x5(image: &mut FusableImage) {
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
    let bin_stride = pad_w * 16;

    // Per-column fine histograms: col_fine[bin * bin_stride + col * 16 + f].
    // Only the fine histograms are kept; the window coarse histogram is built
    // directly from the 5x5 window and updated per pixel.
    let mut col_fine = vec![0u8; 16 * pad_w * 16];
    // Five padded row buffers representing rows y-2 .. y+2.
    let mut rows: Vec<Vec<u8>> = vec![vec![0u8; pad_w]; 5];
    let mut row_out = vec![0u8; pad_w];
    let mut row_in = vec![0u8; pad_w];

    let init_dy = [-2i32, -1, 0, 1, 2];

    for c in 0..channels {
        col_fine.fill(0);

        // Initial rows for y = 0 (clamped, so border rows repeat).
        for (i, dy) in init_dy.iter().enumerate() {
            let sy = (*dy).clamp(0, height as i32 - 1) as usize;
            fill_padded_row(data, sy, c, width, channels, stride, &mut rows[i]);
        }
        // Initialize per-column fine histograms from the initial rows.
        for col in 0..pad_w {
            for r in 0..5 {
                let v = rows[r][col] as usize;
                col_fine[(v >> 4) * bin_stride + col * 16 + (v & 0x0F)] += 1;
            }
        }

        for y in 0..height {
            if y > 0 {
                let sy_out = (y as i32 - 3).clamp(0, height as i32 - 1) as usize;
                let sy_in = (y as i32 + 2).clamp(0, height as i32 - 1) as usize;
                if sy_out != sy_in {
                    fill_padded_row(data, sy_out, c, width, channels, stride, &mut row_out);
                    fill_padded_row(data, sy_in, c, width, channels, stride, &mut row_in);
                    for col in 0..pad_w {
                        let vo = row_out[col] as usize;
                        col_fine[(vo >> 4) * bin_stride + col * 16 + (vo & 0x0F)] -= 1;
                        let vi = row_in[col] as usize;
                        col_fine[(vi >> 4) * bin_stride + col * 16 + (vi & 0x0F)] += 1;
                    }
                }
                // Rotate row buffers: rows become y-1 .. y+3.
                rows.rotate_left(1);
                fill_padded_row(data, sy_in, c, width, channels, stride, &mut rows[4]);
            }

            // Initialize the window at x = 0 (padded columns 0..4).
            let mut win_coarse = [0u8; 16];
            for col in 0..5 {
                for r in 0..5 {
                    let v = rows[r][col] as usize;
                    win_coarse[v >> 4] += 1;
                }
            }
            let mut acc = 0u16;
            let mut target_b = 0usize;
            let mut acc_before = 0u16;
            for b in 0..16 {
                let cnt = win_coarse[b] as u16;
                if acc + cnt >= 13 {
                    target_b = b;
                    acc_before = acc;
                    break;
                }
                acc += cnt;
            }
            let mut win_fine = [0u8; 16];
            rebuild_fine(&mut win_fine, &col_fine, target_b, 0, bin_stride);
            let mut acc_f = acc_before;
            let mut target_f = 0u8;
            for f in 0..16 {
                let cnt = win_fine[f] as u16;
                if acc_f + cnt >= 13 {
                    target_f = f as u8;
                    break;
                }
                acc_f += cnt;
            }
            let mut median = ((target_b as u8) << 4) | target_f;
            let mut count_lt = (acc_before
                + win_fine[..target_f as usize]
                    .iter()
                    .map(|&x| x as u16)
                    .sum::<u16>()) as u8;

            let out_row = y * stride + c;
            output[out_row] = median;

            for x in 1..width {
                // Slide the window from (x-1) to x: remove padded column x-1,
                // add padded column x+4.
                for r in 0..5 {
                    let v_out = rows[r][x - 1] as usize;
                    let b = v_out >> 4;
                    win_coarse[b] -= 1;
                    if b == target_b {
                        win_fine[v_out & 0x0F] -= 1;
                    }
                    if (v_out as u8) < median {
                        count_lt -= 1;
                    }

                    let v_in = rows[r][x + 4] as usize;
                    let b = v_in >> 4;
                    win_coarse[b] += 1;
                    if b == target_b {
                        win_fine[v_in & 0x0F] += 1;
                    }
                    if (v_in as u8) < median {
                        count_lt += 1;
                    }
                }

                // Adjust the median position (amortized O(1): the median moves
                // only a few steps per pixel on natural images).
                loop {
                    let mf = win_fine[(median & 0x0F) as usize];
                    if count_lt + mf < 13 {
                        count_lt += mf;
                        median = median.wrapping_add(1);
                        if (median & 0x0F) == 0 {
                            target_b = (median >> 4) as usize;
                            rebuild_fine(&mut win_fine, &col_fine, target_b, x, bin_stride);
                        }
                    } else {
                        break;
                    }
                }
                while count_lt >= 13 {
                    median = median.wrapping_sub(1);
                    if (median & 0x0F) == 0x0F {
                        target_b = (median >> 4) as usize;
                        rebuild_fine(&mut win_fine, &col_fine, target_b, x, bin_stride);
                    }
                    count_lt -= win_fine[(median & 0x0F) as usize];
                }

                output[out_row + x * channels] = median;
            }
        }
    }

    data.copy_from_slice(&output);
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
    fn test_histogram_5x5_exact_matches_reference() {
        for (w, h, ch) in [
            (1usize, 1usize, 3usize),
            (2, 2, 1),
            (3, 5, 1),
            (5, 5, 3),
            (16, 13, 3),
            (32, 32, 1),
            (64, 64, 3),
            (65, 63, 1),
        ] {
            let mut data: Vec<u8> = (0..w * h * ch)
                .map(|i| ((i as u64 * 2654435761) % 256) as u8)
                .collect();
            let expected = reference_median_5x5(&data, w, h, ch);

            let mut img = crate::core::FusableImage::new(&mut data, w, h, ch);
            apply_median_blur_5x5(&mut img);

            let mut mismatches = 0usize;
            let mut max_diff = 0i32;
            for i in 0..data.len() {
                let diff = (data[i] as i32 - expected[i] as i32).abs();
                if diff > 0 {
                    mismatches += 1;
                    max_diff = max_diff.max(diff);
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
}
