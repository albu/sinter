// Box Blur (StackBlur) - O(1) sliding sum approximation of Gaussian
//
// Applies box blur using a sliding sum (moving average) algorithm.
// By Central Limit Theorem, 3 passes of box blur approximates Gaussian.
//
// Algorithm:
// - Horizontal pass: Sliding sum across each row (O(1) per pixel)
// - Vertical pass: Sliding sum down each column (O(1) per pixel)
// - Repeat 3 times for Gaussian approximation
//
// Complexity: O(1) per pixel regardless of kernel size (2 ops: 1 add, 1 sub)
// vs discrete convolution: O(K) where K is kernel size

use crate::core::FusableImage;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Apply box blur with given radius
///
/// For Gaussian approximation, this should be called 3 times.
/// A radius of 3 (7x7 window) approximates sigma=1.0 Gaussian.
pub fn box_blur(image: &mut FusableImage, radius: usize) {
    if radius == 0 {
        return;
    }

    let width = image.width;
    let height = image.height;
    let channels = image.channels;

    #[cfg(target_arch = "aarch64")]
    {
        // Use optimized version for RGB images
        if channels == 3 {
            neon::box_blur_impl_neon(&mut image.data, width, height, radius);
            return;
        }
    }

    // Fallback to scalar
    box_blur_impl(&mut image.data, width, height, channels, radius);
}

/// Single-pass box blur (call 3 times for Gaussian approximation)
fn box_blur_impl(data: &mut [u8], width: usize, height: usize, channels: usize, radius: usize) {
    let mut temp = vec![0u32; data.len()];

    // Horizontal pass with sliding sum
    for y in 0..height {
        for c in 0..channels {
            let row_base = y * width * channels + c;
            let full_window_size = (radius * 2 + 1) as u32;

            // Initial window sum for first pixel (x=0)
            // Window spans from x=-radius to x=+radius, but we only include valid pixels
            let mut sum: u32 = 0;
            for x in 0..width.min(radius + 1) {
                sum += data[row_base + x * channels] as u32;
            }

            // Slide window across row
            for x in 0..width {
                // Compute actual window size for this position
                // At left edge: window is [0, min(x + radius, width-1)]
                // At right edge: window is [max(0, x - radius), width-1]
                let left = x.saturating_sub(radius);
                let right = (x + radius).min(width - 1);
                let actual_window_size = (right - left + 1) as u32;

                // Normalize and store (with rounding)
                let avg = (sum + actual_window_size / 2) / actual_window_size;
                temp[row_base + x * channels] = avg;

                // Update sliding sum for next position
                // Add pixel entering window (right side)
                if x + radius + 1 < width {
                    sum += data[row_base + (x + radius + 1) * channels] as u32;
                }
                // Subtract pixel leaving window (left side)
                if x >= radius {
                    sum -= data[row_base + (x - radius) * channels] as u32;
                }
            }
        }
    }

    // Vertical pass with sliding sum
    for x in 0..width {
        for c in 0..channels {
            let col_base = c;
            let full_window_size = (radius * 2 + 1) as u32;

            // Initial window sum for first pixel (y=0)
            let mut sum: u32 = 0;
            for y in 0..height.min(radius + 1) {
                sum += temp[(y * width + x) * channels + c];
            }

            // Slide window down column
            for y in 0..height {
                // Compute actual window size for this position
                let top = y.saturating_sub(radius);
                let bottom = (y + radius).min(height - 1);
                let actual_window_size = (bottom - top + 1) as u32;

                // Normalize and store
                let avg = (sum + actual_window_size / 2) / actual_window_size;
                data[(y * width + x) * channels + c] = avg.min(255) as u8;

                // Update sliding sum for next position
                // Add pixel entering window (bottom)
                if y + radius + 1 < height {
                    sum += temp[((y + radius + 1) * width + x) * channels + c];
                }
                // Subtract pixel leaving window (top)
                if y >= radius {
                    sum -= temp[((y - radius) * width + x) * channels + c];
                }
            }
        }
    }
}

/// Apply 3-pass box blur to approximate Gaussian
///
/// This is the "StackBlur" algorithm - multiple box blurs
/// approximate a Gaussian distribution via Central Limit Theorem.
pub fn box_blur_gaussian(image: &mut FusableImage, radius: usize, passes: usize) {
    for _ in 0..passes {
        box_blur(image, radius);
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(test)]
mod tests;
