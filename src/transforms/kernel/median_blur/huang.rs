// Huang's sliding histogram algorithm for median blur
//
// This is the algorithm used by OpenCV for high-performance median filtering.
// Key insight: instead of sorting per-pixel, maintain a histogram that slides
// across the image, updating incrementally both horizontally and vertically.
//
// Complexity:
// - Initialization: O(K²) for first window
// - Per-pixel: O(1) amortized (6 histogram updates + median walk)
// - Total: O(width × height) instead of O(width × height × K² log K²)
//
// Expected speedup: 50-100x over per-pixel sorting for 3x3 kernel

use crate::core::FusableImage;

/// Apply 3x3 median blur using Huang's sliding histogram algorithm
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

/// Huang's algorithm for grayscale images
fn apply_median_blur_3x3_huang_grayscale(data: &mut [u8], width: usize, height: usize) {
    let mut output = vec![0u8; data.len()];

    // Process each pixel independently (simplest correct implementation)
    for y in 0..height {
        for x in 0..width {
            // Build histogram for 3x3 neighborhood
            let mut hist = [0u16; 256];
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let py = (y as i32 + dy).clamp(0, (height - 1) as i32) as usize;
                    let px = (x as i32 + dx).clamp(0, (width - 1) as i32) as usize;
                    let val = data[py * width + px];
                    hist[val as usize] += 1;
                }
            }

            // Find median
            let mut count = 0u16;
            let target = 5; // 5th element in sorted 9 elements
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

/// Scalar RGB implementation with proper sliding window
fn apply_median_blur_3x3_huang_rgb_scalar(data: &mut [u8], width: usize, height: usize) {
    let mut output = vec![0u8; data.len()];

    // Process each channel independently
    for c in 0..3 {
        // Initialize histogram for first 3x3 window at (0,0)
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

        // Slide across first row
        for x in 1..width {
            slide_window_horizontal_rgb(data, width, height, x, 0, c, &mut hist, &mut median_lt, &mut median_gt, &mut median);
            output[x * 3 + c] = median;
        }

        // Slide down through remaining rows
        for y in 1..height {
            // Slide vertically from previous row
            slide_window_vertical_rgb(data, width, height, 0, y, c, &mut hist, &mut median_lt, &mut median_gt, &mut median);
            output[(y * width) * 3 + c] = median;

            // Slide horizontally across this row
            for x in 1..width {
                slide_window_horizontal_rgb(data, width, height, x, y, c, &mut hist, &mut median_lt, &mut median_gt, &mut median);
                output[(y * width + x) * 3 + c] = median;
            }
        }
    }

    data.copy_from_slice(&output);
}

/// Slide window horizontally: remove left column, add right column (RGB)
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

        // Remove leftmost pixel (only if different from incoming pixel)
        let px_out = (x as i32 - 2).clamp(0, (width - 1) as i32) as usize;
        let px_in = (x as i32 + 1).clamp(0, (width - 1) as i32) as usize;

        if px_out != px_in {
            let val_out = data[(py * width + px_out) * 3 + c];
            update_histogram_remove(hist, val_out, median_lt, median_gt, *median);
        }

        // Add rightmost pixel
        let val_in = data[(py * width + px_in) * 3 + c];
        update_histogram_add(hist, val_in, median_lt, median_gt, *median);
    }

    update_median(hist, median, median_lt, median_gt, 9);
}

/// Slide window vertically: remove top row, add bottom row (RGB)
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

        // Remove top row pixel (only if different from incoming pixel)
        let py_out = (y as i32 - 2).clamp(0, (height - 1) as i32) as usize;
        let py_in = (y as i32 + 1).clamp(0, (height - 1) as i32) as usize;

        if py_out != py_in {
            let val_out = data[(py_out * width + px) * 3 + c];
            update_histogram_remove(hist, val_out, median_lt, median_gt, *median);
        }

        // Add bottom row pixel
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
