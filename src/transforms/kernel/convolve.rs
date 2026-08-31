// Convolution utilities for kernel-based transforms
//
// These transforms require neighborhood access (convolution with a kernel),
// so they cannot be fused into the single-pass PixelOp executor.

use crate::core::FusableImage;

/// Apply a 3x3 convolution kernel to an image
///
/// The kernel is applied to each channel independently.
/// Border handling uses edge extension (replicate border pixels).
///
/// # Arguments
/// * `image` - The image to convolve
/// * `kernel` - 3x3 convolution kernel (row-major order)
/// * `scale` - Divisor for the kernel sum (to avoid float operations)
/// * `offset` - Value to add after convolution (typically 0)
pub fn convolve_3x3(image: &mut FusableImage, kernel: &[i32; 9], scale: i32, offset: i32) {
    super::convolve_2d::convolve_3x3_fast(image, kernel, scale, offset);
}

/// Apply a 5x5 convolution kernel to an image
///
/// The kernel is applied to each channel independently.
/// Border handling uses edge extension (replicate border pixels).
///
/// # Arguments
/// * `image` - The image to convolve
/// * `kernel` - 5x5 convolution kernel (row-major order)
/// * `scale` - Divisor for the kernel sum
/// * `offset` - Value to add after convolution
pub fn convolve_5x5(image: &mut FusableImage, kernel: &[i32; 25], scale: i32, offset: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;

    // Create output buffer
    let mut output = vec![0u8; data.len()];

    // Helper to get pixel value with edge extension
    let get_pixel = |data: &[u8], x: i32, y: i32, c: usize| -> u8 {
        let x_clamped = x.max(0).min(width as i32 - 1) as usize;
        let y_clamped = y.max(0).min(height as i32 - 1) as usize;
        data[(y_clamped * width as usize + x_clamped) * channels + c]
    };

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut sum: i32 = 0;

                // Apply 5x5 kernel
                for ky in 0..5 {
                    for kx in 0..5 {
                        let px = x as i32 + kx as i32 - 2;
                        let py = y as i32 + ky as i32 - 2;
                        let pixel = get_pixel(data, px, py, c) as i32;
                        sum += pixel * kernel[ky * 5 + kx];
                    }
                }

                // Apply scale and offset
                let value = (sum / scale).saturating_add(offset);
                output[(y * width + x) * channels + c] = value.clamp(0, 255) as u8;
            }
        }
    }

    // Copy output back to image
    data.copy_from_slice(&output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_3x3_identity() {
        // Identity kernel: should preserve the image
        let kernel = [0, 0, 0, 0, 1, 0, 0, 0, 0];
        let mut data = vec![100u8, 128u8, 150u8];
        let mut img = FusableImage::new(&mut data, 3, 1, 1);

        convolve_3x3(&mut img, &kernel, 1, 0);

        assert_eq!(img.data, &[100, 128, 150]);
    }

    #[test]
    fn test_convolve_3x3_blur() {
        // Box blur kernel
        let kernel = [1, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut data = vec![
            0u8, 255u8, 0u8,   // 0 255 0
            255u8, 128u8, 255u8, // 255 128 255
            0u8, 255u8, 0u8,    // 0 255 0
        ];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        convolve_3x3(&mut img, &kernel, 9, 0);

        // Center pixel: sum of all neighbors / 9
        // (0 + 255 + 0 + 255 + 128 + 255 + 0 + 255 + 0) / 9 = 1148 / 9 = 127
        assert_eq!(img.data[4], 127); // Center pixel
    }

    #[test]
    fn test_convolve_3x3_sharpen() {
        // Sharpen kernel
        let kernel = [0, -1, 0, -1, 5, -1, 0, -1, 0];
        let mut data = vec![
            128u8, 128u8, 128u8,
            128u8, 200u8, 128u8,
            128u8, 128u8, 128u8,
        ];
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        convolve_3x3(&mut img, &kernel, 1, 0);

        // Center pixel: -128 - 128 - 128 - 128 + 5*200 = 1000 - 512 = 488 -> 255 (clamped)
        // Actually: all neighbors are 128, so -128*4 + 1000 = 488, clamped to 255
        assert_eq!(img.data[4], 255);
    }

    #[test]
    fn test_convolve_3x3_rgb() {
        // Test that channels are processed independently
        let kernel = [0, 0, 0, 0, 2, 0, 0, 0, 0]; // 2x center pixel
        let mut data = vec![100u8, 128u8, 150u8]; // RGB pixel
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        convolve_3x3(&mut img, &kernel, 1, 0);

        // Each channel should be doubled (then clamped)
        assert_eq!(img.data, &[200, 255, 255]); // 200, 256->255, 300->255
    }
}

/// Apply a 1D horizontal convolution (for separable filters)
///
/// Used as first pass of separable convolution (e.g., Gaussian blur).
///
/// # Arguments
/// * `image` - The image to convolve
/// * `kernel` - 1D convolution kernel
/// * `scale` - Divisor for the kernel sum
pub fn convolve_1d_horizontal(image: &mut FusableImage, kernel: &[i32], scale: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;
    let k_radius = kernel.len() / 2;

    let mut output = vec![0u8; data.len()];

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                // Use i64 to prevent overflow with large kernel values (e.g., 31x31 Gaussian)
                let mut sum: i64 = 0;

                // Apply 1D kernel horizontally
                for k in 0..kernel.len() {
                    let px = (x as i32 + k as i32 - k_radius as i32).clamp(0, width as i32 - 1) as usize;
                    let pixel = data[(y * width + px) * channels + c] as i64;
                    sum += pixel * kernel[k] as i64;
                }

                output[(y * width + x) * channels + c] = ((sum / scale as i64).clamp(0, 255)) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

/// Apply a 1D vertical convolution (for separable filters)
///
/// Used as second pass of separable convolution (e.g., Gaussian blur).
///
/// # Arguments
/// * `image` - The image to convolve
/// * `kernel` - 1D convolution kernel
/// * `scale` - Divisor for the kernel sum
pub fn convolve_1d_vertical(image: &mut FusableImage, kernel: &[i32], scale: i32) {
    let width = image.width;
    let height = image.height;
    let channels = image.channels;
    let data = &mut image.data;
    let k_radius = kernel.len() / 2;

    let mut output = vec![0u8; data.len()];

    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                // Use i64 to prevent overflow with large kernel values (e.g., 31x31 Gaussian)
                let mut sum: i64 = 0;

                // Apply 1D kernel vertically
                for k in 0..kernel.len() {
                    let py = (y as i32 + k as i32 - k_radius as i32).clamp(0, height as i32 - 1) as usize;
                    let pixel = data[(py * width + x) * channels + c] as i64;
                    sum += pixel * kernel[k] as i64;
                }

                output[(y * width + x) * channels + c] = ((sum / scale as i64).clamp(0, 255)) as u8;
            }
        }
    }

    data.copy_from_slice(&output);
}

/// Apply a separable convolution (horizontal then vertical)
///
/// This is much faster than full 2D convolution for separable kernels like Gaussian.
/// For a KxK kernel: K^2 ops/pixel (full 2D) vs 2K ops/pixel (separable).
///
/// # Arguments
/// * `image` - The image to convolve
/// * `kernel` - 1D kernel (applied both horizontally and vertically)
/// * `scale` - Divisor for the kernel sum
pub fn convolve_separable(image: &mut FusableImage, kernel: &[i32], scale: i32) {
    // IMPORTANT: The image data is modified in-place, but we need to ensure
    // the vertical pass doesn't read from partially-written data.
    // The SIMD implementations handle this correctly by using separate buffers.

    #[cfg(target_arch = "aarch64")]
    {
        // Use SIMD-optimized version when available
        use super::convolve_simd;
        convolve_simd::convolve_separable_detect(image, kernel, scale);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Scalar fallback for other architectures
        convolve_1d_horizontal(image, kernel, scale);
        convolve_1d_vertical(image, kernel, scale);
    }
}
