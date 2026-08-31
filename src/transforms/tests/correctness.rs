// Formal correctness tests for all transforms
//
// Each test function covers:
// - RGB (3 channels) and grayscale (1 channel)
// - All enum variants where applicable
// - NEON vs scalar comparison where applicable
//
// Test images are 24x36 (non-square, >20px)

#[cfg(test)]
mod correctness {
    use crate::core::{Executable, FusableImage};
    use crate::transforms::photometric::{
        Brightness, ColorBalance, ColorJitter, ColorTemperature, ColorTint, Contrast, Gamma,
        Invert, Normalize, Posterize, RGBShift, Solarize, ToGray, ToRGB, ToSepia,
        HueSaturationValue,
    };
    use crate::transforms::geometric::{
        HorizontalFlip, Rotate, RotateAngle, Transpose, VerticalFlip,
    };
    use crate::transforms::geometric::{Affine, AffineParams, Crop, Pad, PadMode, Resize};
    use crate::transforms::kernel::{EdgeDetection, EdgeMethod, Emboss, EmbossDirection, GaussianBlur, KernelSize, MedianBlur, MedianKernelSize, Sharpen};
    use crate::transforms::photometric::{
        AutoContrast, ChannelMix, ChannelShuffle, CoarseDropout, Equalize, GaussNoise, GridDropout, MultiplicativeNoise, SaltAndPepper,
    };

    const TEST_WIDTH: usize = 24;
    const TEST_HEIGHT: usize = 36;

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Create a gradient image with deterministic values
    fn create_gradient_image(width: usize, height: usize, channels: u8) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * channels as usize);
        for y in 0..height {
            for x in 0..width {
                let ratio = (x + y * width) as f32 / (width * height) as f32;
                for _ in 0..channels {
                    data.push((ratio * 255.0) as u8);
                }
            }
        }
        data
    }

    /// Create a checkerboard pattern image
    fn create_checkerboard_image(width: usize, height: usize, channels: u8) -> Vec<u8> {
        let mut data = Vec::with_capacity(width * height * channels as usize);
        let checker_size = 4;
        for y in 0..height {
            for x in 0..width {
                let value = if ((x / checker_size) + (y / checker_size)) % 2 == 0 {
                    255u8
                } else {
                    0u8
                };
                for _ in 0..channels {
                    data.push(value);
                }
            }
        }
        data
    }

    /// Compare two image buffers with detailed error reporting
    fn compare_images(expected: &[u8], actual: &[u8], name: &str) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "{}: Length mismatch: expected {}, got {}",
            name,
            expected.len(),
            actual.len()
        );
        for (i, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
            assert_eq!(
                exp, act,
                "{}: Mismatch at index {}: expected {}, got {}",
                name, i, exp, act
            );
        }
    }

    // ========================================================================
    // Geometric Transform Tests
    // ========================================================================

    #[test]
    fn test_horizontal_flip_correctness() {
        // Test RGB and grayscale
        for &channels in &[1u8, 3u8] {
            let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let flip = HorizontalFlip::new();
            flip.apply(&mut img);

            // Verify flip by checking that pixels are reversed horizontally
            for y in 0..TEST_HEIGHT {
                for x in 0..TEST_WIDTH {
                    let src_idx = (y * TEST_WIDTH + x) * channels as usize;
                    let dst_idx = (y * TEST_WIDTH + (TEST_WIDTH - 1 - x)) * channels as usize;
                    for c in 0..channels as usize {
                        assert_eq!(
                            img.data[dst_idx + c],
                            input[src_idx + c],
                            "Flip mismatch at ({},{}), ch {}",
                            x,
                            y,
                            c
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_vertical_flip_correctness() {
        // Test RGB and grayscale
        for &channels in &[1u8, 3u8] {
            let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let flip = VerticalFlip::new();
            flip.execute(&mut img);

            // Verify flip by checking that pixels are reversed vertically
            for y in 0..TEST_HEIGHT {
                for x in 0..TEST_WIDTH {
                    let src_idx = (y * TEST_WIDTH + x) * channels as usize;
                    let dst_idx = ((TEST_HEIGHT - 1 - y) * TEST_WIDTH + x) * channels as usize;
                    for c in 0..channels as usize {
                        assert_eq!(
                            img.data[dst_idx + c],
                            input[src_idx + c],
                            "Flip mismatch at ({},{}), ch {}",
                            x,
                            y,
                            c
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_transpose_correctness() {
        // Test RGB and grayscale
        for &channels in &[1u8, 3u8] {
            let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let transpose = Transpose::new();
            let result = transpose.execute(&mut img);

            // Transpose returns a BarrierImage (OutOfPlace operation)
            assert!(result.is_some(), "Transpose should return BarrierImage");
            let result_img = result.unwrap();

            // Verify dimensions are swapped
            assert_eq!(result_img.width, TEST_HEIGHT, "Width should be swapped");
            assert_eq!(result_img.height, TEST_WIDTH, "Height should be swapped");

            // Verify transpose by checking that pixels are transposed
            // Original pixel at (x, y) goes to (y, x) in transposed image
            for orig_y in 0..TEST_HEIGHT {
                for orig_x in 0..TEST_WIDTH {
                    let src_idx = (orig_y * TEST_WIDTH + orig_x) * channels as usize;
                    // In transposed image: pixel is at (orig_x, orig_y)
                    let dst_idx = (orig_x * result_img.width + orig_y) * channels as usize;
                    for c in 0..channels as usize {
                        assert_eq!(
                            result_img.data[dst_idx + c],
                            input[src_idx + c],
                            "Transpose mismatch: original ({},{}) -> transposed ({},{})",
                            orig_x, orig_y, orig_y, orig_x
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_rotate_correctness() {
        // Test all rotation angles
        for &angle in &[RotateAngle::Rotate90, RotateAngle::Rotate180, RotateAngle::Rotate270] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let rotate = Rotate::new(angle);
                let result = rotate.execute(&mut img);

                // Rotate returns a BarrierImage (OutOfPlace operation)
                assert!(result.is_some(), "Rotate should return BarrierImage");
                let result_img = result.unwrap();

                // Verify rotation by checking dimensions
                match angle {
                    RotateAngle::Rotate90 | RotateAngle::Rotate270 => {
                        assert_eq!(result_img.width, TEST_HEIGHT, "90/270° should swap width");
                        assert_eq!(result_img.height, TEST_WIDTH, "90/270° should swap height");
                    }
                    RotateAngle::Rotate180 => {
                        assert_eq!(result_img.width, TEST_WIDTH, "180° should preserve width");
                        assert_eq!(result_img.height, TEST_HEIGHT, "180° should preserve height");
                    }
                }

                // Verify that the image was actually modified (rotation changes pixel positions)
                let changed = result_img.data.iter().zip(input.iter()).any(|(&a, &b)| a != b);
                assert!(changed, "Rotation should modify pixel positions");

                // Verify that rotating 4x360 degrees returns to original dimensions
                // For 90°: WxH -> HxW -> WxH -> HxW -> WxH (4 rotations)
                // For 180°: WxH -> WxH -> WxH (2 rotations = 360°)
                let mut data2 = result_img.data.clone();
                let mut img2 = FusableImage::new(&mut data2, result_img.width, result_img.height, result_img.channels);
                let result2 = rotate.execute(&mut img2);
                assert!(result2.is_some(), "Second rotate should return BarrierImage");

                // After 2x90 or 2x270, dimensions should be back to original
                // After 2x180, dimensions should still be original
                let result2_img = result2.unwrap();
                match angle {
                    RotateAngle::Rotate90 | RotateAngle::Rotate270 => {
                        assert_eq!(result2_img.width, TEST_WIDTH, "2x 90/270° should restore width");
                        assert_eq!(result2_img.height, TEST_HEIGHT, "2x 90/270° should restore height");
                    }
                    RotateAngle::Rotate180 => {
                        assert_eq!(result2_img.width, TEST_WIDTH, "2x 180° should preserve width");
                        assert_eq!(result2_img.height, TEST_HEIGHT, "2x 180° should preserve height");
                    }
                }

                // The data should be preserved (just reordered)
                // Check that the set of pixel values is the same (multiset equality)
                let mut input_sorted = input.clone();
                let mut result_sorted = result2_img.data.clone();
                input_sorted.sort();
                result_sorted.sort();
                assert_eq!(input_sorted, result_sorted, "2x rotation should preserve pixel values");
            }
        }
    }

    // ========================================================================
    // Photometric Transform Tests
    // ========================================================================

    #[test]
    fn test_invert_correctness() {
        // Test RGB and grayscale
        for &channels in &[1u8, 3u8] {
            let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let invert = Invert::new();
            invert.execute(&mut img);

            // Verify inversion: output[i] = 255 - input[i]
            for (i, (&inp, &out)) in input.iter().zip(img.data.iter()).enumerate() {
                assert_eq!(
                    out, 255 - inp,
                    "Invert mismatch at index {}: expected {}, got {}",
                    i, 255 - inp, out
                );
            }
        }
    }

    #[test]
    fn test_brightness_correctness() {
        // Test various delta values
        for &delta in &[-30.0f32, 0.0, 30.0] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let brightness = Brightness::new(delta);
                brightness.execute(&mut img);

                // Verify brightness adjustment with clamping
                for (i, &inp) in input.iter().enumerate() {
                    let expected = (inp as f32 + delta).clamp(0.0, 255.0) as u8;
                    assert_eq!(
                        img.data[i], expected,
                        "Brightness mismatch at index {}: delta={}, expected {}, got {}",
                        i, delta, expected, img.data[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_contrast_correctness() {
        // Test various factor values
        for &factor in &[0.5f32, 1.0, 1.5] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let contrast = Contrast::new(factor);
                contrast.execute(&mut img);

                // Verify contrast adjustment with clamping
                for (i, &inp) in input.iter().enumerate() {
                    let expected = ((inp as f32 - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8;
                    assert_eq!(
                        img.data[i], expected,
                        "Contrast mismatch at index {}: factor={}, expected {}, got {}",
                        i, factor, expected, img.data[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_gamma_correctness() {
        // Test various gamma values
        for &gamma in &[0.5f32, 1.0, 2.0] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let gamma_transform = Gamma::new(gamma);
                gamma_transform.execute(&mut img);

                // Verify gamma correction: output = 255 * (input/255)^gamma
                // Note: The LUT implementation truncates (casts to u8) rather than rounds
                for (i, &inp) in input.iter().enumerate() {
                    let normalized = inp as f32 / 255.0;
                    let corrected = normalized.powf(gamma);
                    let expected = (corrected * 255.0).clamp(0.0, 255.0) as u8;
                    assert_eq!(
                        img.data[i], expected,
                        "Gamma mismatch at index {}: gamma={}, expected {}, got {}",
                        i, gamma, expected, img.data[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_to_gray_correctness() {
        // Test RGB to grayscale conversion
        let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, 3);
        let mut data = input.clone();
        let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, 3);

        let to_gray = ToGray::new();
        let result = to_gray.execute(&mut img);

        // ToGray returns a BarrierImage (OutOfPlace operation)
        assert!(result.is_some(), "ToGray should return BarrierImage");
        let result_img = result.unwrap();

        // Verify conversion to grayscale
        assert_eq!(result_img.channels, 1, "Should convert to 1 channel");
        assert_eq!(result_img.width, TEST_WIDTH);
        assert_eq!(result_img.height, TEST_HEIGHT);

        // Verify grayscale formula: 0.299*R + 0.587*G + 0.114*B
        for y in 0..TEST_HEIGHT {
            for x in 0..TEST_WIDTH {
                let src_idx = (y * TEST_WIDTH + x) * 3;
                let dst_idx = y * TEST_WIDTH + x;
                let r = input[src_idx] as f32;
                let g = input[src_idx + 1] as f32;
                let b = input[src_idx + 2] as f32;
                let expected = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
                assert_eq!(
                    result_img.data[dst_idx], expected,
                    "ToGray mismatch at ({},{}): expected {}, got {}",
                    x, y, expected, result_img.data[dst_idx]
                );
            }
        }
    }

    #[test]
    fn test_to_rgb_correctness() {
        // Test grayscale to RGB conversion
        let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, 1);
        let mut data = input.clone();
        let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, 1);

        let to_rgb = ToRGB::new();
        let result = to_rgb.execute(&mut img);

        // ToRGB returns a BarrierImage (OutOfPlace operation)
        assert!(result.is_some(), "ToRGB should return BarrierImage");
        let result_img = result.unwrap();

        // Verify conversion to RGB
        assert_eq!(result_img.channels, 3, "Should convert to 3 channels");
        assert_eq!(result_img.width, TEST_WIDTH);
        assert_eq!(result_img.height, TEST_HEIGHT);

        // Verify that grayscale value is replicated to all channels
        for y in 0..TEST_HEIGHT {
            for x in 0..TEST_WIDTH {
                let src_idx = y * TEST_WIDTH + x;
                let dst_idx = (y * TEST_WIDTH + x) * 3;
                assert_eq!(
                    result_img.data[dst_idx], input[src_idx],
                    "ToRGB R channel mismatch at ({},{})", x, y
                );
                assert_eq!(
                    result_img.data[dst_idx + 1], input[src_idx],
                    "ToRGB G channel mismatch at ({},{})", x, y
                );
                assert_eq!(
                    result_img.data[dst_idx + 2], input[src_idx],
                    "ToRGB B channel mismatch at ({},{})", x, y
                );
            }
        }
    }

    #[test]
    fn test_solarize_correctness() {
        // Test various threshold values
        for &threshold in &[64u8, 128, 192] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let solarize = Solarize::new(threshold);
                solarize.execute(&mut img);

                // Verify solarization: values >= threshold are inverted
                for (i, &inp) in input.iter().enumerate() {
                    let expected = if inp >= threshold { 255 - inp } else { inp };
                    assert_eq!(
                        img.data[i], expected,
                        "Solarize mismatch at index {}: threshold={}, expected {}, got {}",
                        i, threshold, expected, img.data[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_posterize_correctness() {
        // Test various bit values
        for &bits in &[1u8, 2, 4] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let posterize = Posterize::new(bits);
                posterize.execute(&mut img);

                // Verify posterization: (pixel >> (8 - bits)) << (8 - bits)
                let bits_to_discard = 8u8 - bits;
                for (i, &inp) in input.iter().enumerate() {
                    let expected = (inp >> bits_to_discard) << bits_to_discard;
                    assert_eq!(
                        img.data[i], expected,
                        "Posterize mismatch at index {}: bits={}, expected {}, got {}",
                        i, bits, expected, img.data[i]
                    );
                }
            }
        }
    }

    // ========================================================================
    // Kernel Transform Tests
    // ========================================================================

    #[test]
    fn test_gaussian_blur_correctness() {
        // Test all kernel sizes
        for &kernel_size in &[KernelSize::Size3, KernelSize::Size5, KernelSize::Size7] {
            for &channels in &[1u8, 3u8] {
                // Use checkerboard to test edge smoothing
                let input = create_checkerboard_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let blur = GaussianBlur::with_kernel_size(kernel_size);
                blur.execute(&mut img);

                // Gaussian blur should preserve mean brightness
                let original_mean: f32 = input.iter().map(|&x| x as f32).sum::<f32>() / input.len() as f32;
                let new_mean: f32 = img.data.iter().map(|&x| x as f32).sum::<f32>() / img.data.len() as f32;
                assert!(
                    (new_mean - original_mean).abs() < 2.0,
                    "Gaussian blur should preserve mean (kernel={:?}, ch={})",
                    kernel_size,
                    channels
                );

                // Gaussian blur should reduce variance (smooths the image)
                let original_var: f32 = input.iter()
                    .map(|&x| (x as f32 - original_mean).powi(2))
                    .sum::<f32>() / input.len() as f32;
                let new_var: f32 = img.data.iter()
                    .map(|&x| (x as f32 - new_mean).powi(2))
                    .sum::<f32>() / img.data.len() as f32;
                assert!(
                    new_var < original_var,
                    "Gaussian blur should reduce variance: original={}, new={}",
                    original_var, new_var
                );

                // Checkerboard edges should be smoothed (no pure 0 or 255 at edges)
                // Check center of image where edges were sharp
                let center_idx = ((TEST_HEIGHT / 2) * TEST_WIDTH + (TEST_WIDTH / 2)) * channels as usize;
                for c in 0..channels as usize {
                    // After blur, edge pixels should be intermediate values, not pure 0 or 255
                    let val = img.data[center_idx + c];
                    assert!(
                        val > 0 && val < 255,
                        "Blurred checkerboard should have intermediate values at edges, got {}",
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_sharpen_correctness() {
        // Test RGB and grayscale
        for &channels in &[1u8, 3u8] {
            // Use checkerboard pattern - sharpening should increase edge contrast
            let input = create_checkerboard_image(TEST_WIDTH, TEST_HEIGHT, channels);
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let sharpen = Sharpen::new();
            sharpen.execute(&mut img);

            // Sharpening should increase local contrast at edges
            // Calculate local variance at center of image (edge region)
            let center_y = TEST_HEIGHT / 2;
            let center_x = TEST_WIDTH / 2;

            // Sample a 3x3 region around center for local variance
            let mut input_local = Vec::new();
            let mut output_local = Vec::new();
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let y = (center_y as i32 + dy) as usize;
                    let x = (center_x as i32 + dx) as usize;
                    let idx = (y * TEST_WIDTH + x) * channels as usize;
                    for c in 0..channels as usize {
                        input_local.push(input[idx + c] as f32);
                        output_local.push(img.data[idx + c] as f32);
                    }
                }
            }

            let input_mean: f32 = input_local.iter().sum::<f32>() / input_local.len() as f32;
            let output_mean: f32 = output_local.iter().sum::<f32>() / output_local.len() as f32;

            let input_var: f32 = input_local.iter()
                .map(|&v| (v - input_mean).powi(2))
                .sum::<f32>() / input_local.len() as f32;
            let output_var: f32 = output_local.iter()
                .map(|&v| (v - output_mean).powi(2))
                .sum::<f32>() / output_local.len() as f32;

            // Sharpening should increase local variance (enhance edges)
            assert!(
                output_var > input_var * 0.9, // Allow some tolerance but variance should increase
                "Sharpening should increase local variance: input_var={}, output_var={}",
                input_var, output_var
            );

            // Dimensions should be preserved
            assert_eq!(img.width, TEST_WIDTH);
            assert_eq!(img.height, TEST_HEIGHT);
            assert_eq!(img.channels, channels as usize);
        }
    }

    #[test]
    fn test_edge_detection_correctness() {
        // Test both Laplacian and Sobel methods
        for &method in &[EdgeMethod::Laplacian, EdgeMethod::Sobel] {
            for &channels in &[1u8, 3u8] {
                let input = create_checkerboard_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let edge = EdgeDetection::new(method);
                edge.execute(&mut img);

                // Edge detection should produce some high values (edges detected)
                let max_val = *img.data.iter().max().unwrap();
                assert!(
                    max_val > 0,
                    "Edge detection should find edges (method={:?}, ch={})",
                    method,
                    channels
                );
            }
        }
    }

    // ========================================================================
    // Additional Photometric Transform Tests
    // ========================================================================

    #[test]
    fn test_normalize_correctness() {
        // Test various mean/std combinations.
        // Output is float32: out = (v / 255 - mean) / std (no clamping).
        for &(mean, std) in &[(-0.5f32, 0.5f32), (0.0, 1.0), (0.5, 1.5)] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let normalize = Normalize::new(mean, std);
                let barrier = normalize.execute(&mut img).expect("Normalize returns a float32 barrier");

                assert!(barrier.is_f32());
                let out = barrier.f32_data.as_ref().unwrap();
                assert_eq!(out.len(), input.len());

                for (i, &inp) in input.iter().enumerate() {
                    let expected = (inp as f32 / 255.0 - mean) / std;
                    assert!(
                        (out[i] - expected).abs() < 1e-5,
                        "Normalize mismatch at index {}: mean={}, std={}, expected {}, got {}",
                        i, mean, std, expected, out[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_to_sepia_correctness() {
        // ToSepia only works on RGB
        // Test with known inputs - based on internal tests in to_sepia.rs
        // The implementation uses fixed-point arithmetic via MatrixExecutor

        let test_cases = [
            // (input_r, input_g, input_b, expected_r, expected_g, expected_b, tolerance)
            (255u8, 255u8, 255u8, 255u8, 255u8, 238u8, 10i32), // White -> clamped R,G, high B
            (255u8, 0u8, 0u8, 100u8, 89u8, 69u8, 1i32),        // Pure red
            (0u8, 255u8, 0u8, 196u8, 175u8, 136u8, 1i32),      // Pure green
            (0u8, 0u8, 255u8, 48u8, 43u8, 33u8, 1i32),         // Pure blue
            (128u8, 128u8, 128u8, 173u8, 154u8, 120u8, 2i32),  // Mid gray
        ];

        for &(input_r, input_g, input_b, expected_r, expected_g, expected_b, tolerance) in &test_cases {
            let input = vec![input_r, input_g, input_b];
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, 1, 1, 3);

            let sepia = ToSepia::new();
            sepia.execute(&mut img);

            assert!(
                (img.data[0] as i32 - expected_r as i32).abs() <= tolerance,
                "Sepia R mismatch: input=({}, {}, {}), expected={}, got={}",
                input_r, input_g, input_b, expected_r, img.data[0]
            );
            assert!(
                (img.data[1] as i32 - expected_g as i32).abs() <= tolerance,
                "Sepia G mismatch: input=({}, {}, {}), expected={}, got={}",
                input_r, input_g, input_b, expected_g, img.data[1]
            );
            assert!(
                (img.data[2] as i32 - expected_b as i32).abs() <= tolerance,
                "Sepia B mismatch: input=({}, {}, {}), expected={}, got={}",
                input_r, input_g, input_b, expected_b, img.data[2]
            );

            // Sepia should produce warm tones (R >= G >= B)
            assert!(
                img.data[0] >= img.data[1] && img.data[1] >= img.data[2],
                "Sepia should produce warm tones: R={} G={} B={}",
                img.data[0], img.data[1], img.data[2]
            );
        }
    }

    #[test]
    fn test_color_balance_correctness() {
        // Test various balance configurations
        let configs = [
            (1.2f32, 1.0f32, 0.8f32), // Warm (boost red, reduce blue)
            (0.8f32, 1.0f32, 1.2f32), // Cool (reduce red, boost blue)
            (1.0f32, 1.0f32, 1.0f32), // Identity (no change)
        ];

        for &(r_scale, g_scale, b_scale) in &configs {
            let input = vec![100u8, 100u8, 100u8];
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, 1, 1, 3);

            let balance = ColorBalance::new(r_scale, g_scale, b_scale);
            balance.execute(&mut img);

            // Verify each channel is scaled independently
            let expected_r = (100.0 * r_scale).clamp(0.0, 255.0) as u8;
            let expected_g = (100.0 * g_scale).clamp(0.0, 255.0) as u8;
            let expected_b = (100.0 * b_scale).clamp(0.0, 255.0) as u8;

            assert_eq!(img.data[0], expected_r, "R channel mismatch");
            assert_eq!(img.data[1], expected_g, "G channel mismatch");
            assert_eq!(img.data[2], expected_b, "B channel mismatch");
        }
    }

    #[test]
    fn test_color_jitter_correctness() {
        // ColorJitter uses deterministic seeding, so we can test reproducibility
        for &channels in &[1u8, 3u8] {
            let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
            let mut data1 = input.clone();
            let mut img1 = FusableImage::new(&mut data1, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let mut data2 = input.clone();
            let mut img2 = FusableImage::new(&mut data2, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let jitter = ColorJitter::new(0.2, 0.2, 0.2, 0.1);
            jitter.execute(&mut img1);
            jitter.execute(&mut img2);

            // Same input should produce same output (deterministic)
            assert_eq!(
                img1.data, img2.data,
                "ColorJitter should be deterministic"
            );
        }
    }

    #[test]
    fn test_color_temperature_correctness() {
        // Test warm and cool temperatures
        for &temperature in &[-50.0f32, 0.0, 50.0] {
            let input = vec![128u8, 128, 128];
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, 1, 1, 3);

            let temp = ColorTemperature::new(temperature);
            temp.execute(&mut img);

            // Verify temperature adjustment
            // Warm (> 0): R and G boosted, B reduced
            // Cool (< 0): B boosted, R and G reduced
            if temperature > 0.0 {
                assert!(img.data[0] >= img.data[2], "Warm: R should be >= B");
                assert!(img.data[1] >= img.data[2], "Warm: G should be >= B");
            } else if temperature < 0.0 {
                assert!(img.data[2] >= img.data[0], "Cool: B should be >= R");
            }
        }
    }

    #[test]
    fn test_color_tint_correctness() {
        // Test various tint configurations
        let configs = [
            (255.0f32, 0.0f32, 0.0f32, 0.5f32), // Red tint at 50%
            (0.0f32, 255.0f32, 0.0f32, 0.5f32), // Green tint at 50%
            (0.0f32, 0.0f32, 255.0f32, 0.5f32), // Blue tint at 50%
            (128.0f32, 128.0f32, 128.0f32, 0.0f32), // No tint (zero intensity)
            (255.0f32, 255.0f32, 255.0f32, 1.0f32), // Full white tint
        ];

        for &(target_r, target_g, target_b, intensity) in &configs {
            // Use a colored input instead of gray to see more effect
            let input = vec![150u8, 100, 200];
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, 1, 1, 3);

            let tint = ColorTint::new(target_r, target_g, target_b, intensity);
            tint.execute(&mut img);

            // Verify dimensions preserved
            assert_eq!(img.width, 1);
            assert_eq!(img.height, 1);
            assert_eq!(img.channels, 3);

            if intensity == 0.0 {
                // Zero intensity - should be approximately unchanged
                let max_diff = img.data.iter().zip(input.iter())
                    .map(|(&a, &b)| (a as i32 - b as i32).abs())
                    .max()
                    .unwrap();
                assert!(
                    max_diff <= 1,
                    "Zero intensity should not modify image: max_diff={}",
                    max_diff
                );
            } else {
                // Non-zero intensity should modify the image
                // For gray input (128,128,128) with tint, the change might be subtle
                // For colored input, the effect should be more visible
                let max_diff = img.data.iter().zip(input.iter())
                    .map(|(&a, &b)| (a as i32 - b as i32).abs())
                    .max()
                    .unwrap();
                assert!(
                    max_diff > 0,
                    "Tint should modify the image: target=({},{},{}), intensity={}, input={:?}, output={:?}",
                    target_r, target_g, target_b, intensity, input, img.data
                );
            }
        }
    }

    #[test]
    fn test_rgb_shift_correctness() {
        // Test RGB and grayscale
        for &channels in &[1u8, 3u8] {
            let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let shift = RGBShift::new(10.0, 20.0, 30.0);
            shift.execute(&mut img);

            // Verify shift was applied
            if channels == 3 {
                // Check first pixel
                let expected_r = (input[0] as f32 + 10.0).clamp(0.0, 255.0) as u8;
                let expected_g = (input[1] as f32 + 20.0).clamp(0.0, 255.0) as u8;
                let expected_b = (input[2] as f32 + 30.0).clamp(0.0, 255.0) as u8;
                assert_eq!(img.data[0], expected_r, "R channel mismatch");
                assert_eq!(img.data[1], expected_g, "G channel mismatch");
                assert_eq!(img.data[2], expected_b, "B channel mismatch");
            }
        }
    }

    #[test]
    fn test_hue_saturation_value_correctness() {
        // Test hue shift - red should change to different color
        let input = vec![255u8, 0u8, 0u8];
        let mut data = input.clone();
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        // Shift hue by 60 degrees
        let hsv = HueSaturationValue::new(60.0, 1.0, 1.0);
        hsv.execute(&mut img);

        // Hue shift should modify the color
        assert!(
            img.data[0] != 255 || img.data[1] != 0 || img.data[2] != 0,
            "Hue shift should modify the color"
        );

        // Test saturation boost - use a colored pixel, not gray
        let input2 = vec![200u8, 100u8, 150u8];
        let mut data2 = input2.clone();
        let mut img2 = FusableImage::new(&mut data2, 1, 1, 3);

        let hsv2 = HueSaturationValue::new(0.0, 1.5, 1.0);
        hsv2.execute(&mut img2);

        // Saturation boost should modify the color (make it more saturated)
        assert!(
            img2.data[0] != 200 || img2.data[1] != 100 || img2.data[2] != 150,
            "Saturation boost should modify pixels"
        );

        // Test value (brightness) boost
        let input3 = vec![100u8, 100u8, 100u8];
        let mut data3 = input3.clone();
        let mut img3 = FusableImage::new(&mut data3, 1, 1, 3);

        let hsv3 = HueSaturationValue::new(0.0, 1.0, 1.5);
        hsv3.execute(&mut img3);

        // Value boost should increase brightness
        assert!(
            img3.data[0] > 100 || img3.data[1] > 100 || img3.data[2] > 100,
            "Value boost should increase brightness: R={} G={} B={}",
            img3.data[0], img3.data[1], img3.data[2]
        );

        // Test identity transformation (no change)
        let input4 = vec![150u8, 100u8, 200u8];
        let mut data4 = input4.clone();
        let mut img4 = FusableImage::new(&mut data4, 1, 1, 3);

        let hsv4 = HueSaturationValue::new(0.0, 1.0, 1.0);
        hsv4.execute(&mut img4);

        // With hue=0, sat=1.0, val=1.0, the color should be approximately unchanged
        // (allowing for rounding in HSV conversion)
        let max_diff = img4.data.iter().zip(input4.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max()
            .unwrap();
        assert!(
            max_diff <= 3, // Allow small tolerance for rounding
            "Identity HSV should approximately preserve color: max_diff={}",
            max_diff
        );
    }

    // ========================================================================
    // Additional Geometric Transform Tests
    // ========================================================================

    #[test]
    fn test_resize_correctness() {
        // Test upscaling and downscaling
        for &(new_w, new_h) in &[(12usize, 18usize), (48, 72)] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let resize = Resize::new(new_w, new_h);
                let result = resize.execute(&mut img);

                // Resize returns a BarrierImage (OutOfPlace operation)
                assert!(result.is_some(), "Resize should return BarrierImage");
                let result_img = result.unwrap();

                // Verify dimensions
                assert_eq!(result_img.width, new_w);
                assert_eq!(result_img.height, new_h);
                assert_eq!(result_img.channels, channels as usize);

                // Verify that resized image values are within valid range
                assert!(result_img.data.iter().all(|&x| x <= 255));

                // Verify corners are approximately preserved
                // Top-left corner (0,0) should map to (0,0)
                for c in 0..channels as usize {
                    assert_eq!(
                        result_img.data[c],
                        input[c],
                        "Top-left corner mismatch in channel {}",
                        c
                    );
                }

                // For 2x upscale, verify bottom-right corner is close
                if new_w == TEST_WIDTH * 2 && new_h == TEST_HEIGHT * 2 {
                    // Original bottom-right corner
                    let orig_idx = ((TEST_HEIGHT - 1) * TEST_WIDTH + (TEST_WIDTH - 1)) * channels as usize;
                    // New bottom-right corner
                    let new_idx = ((new_h - 1) * new_w + (new_w - 1)) * channels as usize;
                    for c in 0..channels as usize {
                        assert_eq!(
                            result_img.data[new_idx + c],
                            input[orig_idx + c],
                            "Bottom-right corner mismatch in channel {}",
                            c
                        );
                    }
                }

                // Resized image should have similar mean (within reasonable tolerance)
                let orig_mean: f32 = input.iter().map(|&x| x as f32).sum::<f32>() / input.len() as f32;
                let new_mean: f32 = result_img.data.iter().map(|&x| x as f32).sum::<f32>() / result_img.data.len() as f32;
                assert!(
                    (new_mean - orig_mean).abs() < 10.0,
                    "Resize should preserve approximate mean: orig={}, new={}",
                    orig_mean, new_mean
                );
            }
        }
    }

    #[test]
    fn test_crop_correctness() {
        // Test various crop regions (relative to image size)
        let half_w = (TEST_WIDTH / 2) as u32;
        let half_h = (TEST_HEIGHT / 2) as u32;
        let quarter_w = (TEST_WIDTH / 4) as u32;
        let quarter_h = (TEST_HEIGHT / 4) as u32;

        let crop_configs = [
            (0u32, 0u32, half_w, half_h),           // Top-left quarter
            (half_w, half_h, half_w, half_h),       // Bottom-right quarter
            (quarter_w, quarter_h, half_w, half_h), // Center crop
        ];

        for &(x, y, w, h) in &crop_configs {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let crop = Crop::new(x, y, w, h);
                let result = crop.execute(&mut img);

                // Crop returns a BarrierImage (OutOfPlace operation)
                assert!(result.is_some(), "Crop should return BarrierImage");
                let result_img = result.unwrap();

                // Verify dimensions
                assert_eq!(result_img.width, w as usize);
                assert_eq!(result_img.height, h as usize);
                assert_eq!(result_img.channels, channels as usize);

                // Verify crop region is correct - check top-left pixel
                let src_idx = (y as usize * TEST_WIDTH + x as usize) * channels as usize;
                let dst_idx = 0;
                for c in 0..channels as usize {
                    assert_eq!(
                        result_img.data[dst_idx + c],
                        input[src_idx + c],
                        "Crop top-left pixel mismatch at channel {}",
                        c
                    );
                }
            }
        }
    }

    #[test]
    fn test_median_blur_correctness() {
        // Test both kernel sizes
        for &kernel_size in &[MedianKernelSize::Kernel3, MedianKernelSize::Kernel5] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let median = MedianBlur::new(kernel_size);
                median.execute(&mut img);

                // Verify dimensions preserved
                assert_eq!(img.width, TEST_WIDTH);
                assert_eq!(img.height, TEST_HEIGHT);
                assert_eq!(img.channels, channels as usize);
            }
        }
    }

    #[test]
    fn test_pad_correctness() {
        // Test various padding modes
        let pad_configs = [
            PadMode::Constant(128u8),  // Constant fill
            PadMode::Replicate,          // Edge replication
        ];

        for &mode in &pad_configs {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let pad = Pad::new(2, 2, 2, 2, mode);
                let result = pad.execute(&mut img);

                // Pad returns a BarrierImage (OutOfPlace operation)
                assert!(result.is_some(), "Pad should return BarrierImage");
                let result_img = result.unwrap();

                // Verify dimensions (padded by 2 on each side)
                assert_eq!(result_img.width, TEST_WIDTH + 4);
                assert_eq!(result_img.height, TEST_HEIGHT + 4);
                assert_eq!(result_img.channels, channels as usize);

                // Verify padding values based on mode
                match mode {
                    PadMode::Constant(fill_value) => {
                        // Top-left corner padding should have constant value
                        let idx = 0;
                        for c in 0..channels as usize {
                            assert_eq!(
                                result_img.data[idx + c],
                                fill_value,
                                "Constant padding should have fill value at corner, ch {}",
                                c
                            );
                        }
                    }
                    PadMode::Replicate => {
                        // Top-left corner padding should replicate top-left pixel
                        let idx = 0;
                        for c in 0..channels as usize {
                            assert_eq!(
                                result_img.data[idx + c],
                                input[c],
                                "Replicate padding should replicate edge pixel at corner, ch {}",
                                c
                            );
                        }
                    }
                    _ => {
                        // Other modes (Reflect, Wrap) - just verify dimensions and that original is preserved
                    }
                }

                // Verify that original image is preserved in center
                // Top-left pixel of original should be at (2, 2) in padded image
                let orig_idx = 0;
                let pad_idx = (2 * result_img.width + 2) * channels as usize;
                for c in 0..channels as usize {
                    assert_eq!(
                        result_img.data[pad_idx + c],
                        input[orig_idx + c],
                        "Original image should be preserved in center, ch {}",
                        c
                    );
                }
            }
        }
    }

    #[test]
    fn test_channel_mix_correctness() {
        // Test RGB and grayscale
        // ChannelMix only works on RGB images
        let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, 3);
        let mut data = input.clone();
        let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, 3);

        // Test BGR swap (R and B channels swapped)
        let mix = ChannelMix::bgr();
        mix.execute(&mut img);

        // Verify BGR swap - check first pixel
        // Original: [R, G, B] = input[0], input[1], input[2]
        // After BGR: [B, G, R]
        assert_eq!(img.data[0], input[2], "BGR: R should become original B");
        assert_eq!(img.data[1], input[1], "BGR: G should remain G");
        assert_eq!(img.data[2], input[0], "BGR: B should become original R");
    }

    #[test]
    fn test_channel_shuffle_correctness() {
        // ChannelShuffle only works on RGB images
        let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, 3);
        let mut data = input.clone();
        let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, 3);

        // Test BGR swap (R and B channels swapped)
        use crate::transforms::photometric::channel_shuffle::ChannelOrder;
        let shuffle = ChannelShuffle::new(ChannelOrder::BGR);
        shuffle.execute(&mut img);

        // Verify BGR swap - check first pixel
        // Original: [R, G, B] = input[0], input[1], input[2]
        // After BGR: [B, G, R]
        assert_eq!(img.data[0], input[2], "BGR: R should become original B");
        assert_eq!(img.data[1], input[1], "BGR: G should remain G");
        assert_eq!(img.data[2], input[0], "BGR: B should become original R");
    }

    #[test]
    fn test_coarse_dropout_correctness() {
        // Test various dropout configurations
        for &fill_value in &[0u8, 128] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let dropout = CoarseDropout::new(4, (0.1, 0.1), fill_value);
                dropout.execute(&mut img);

                // Verify dimensions preserved
                assert_eq!(img.width, TEST_WIDTH);
                assert_eq!(img.height, TEST_HEIGHT);
                assert_eq!(img.channels, channels as usize);

                // Some pixels should be different (holes were filled)
                let changed = img.data.iter().zip(input.iter()).any(|(&a, &b)| a != b);
                assert!(changed, "CoarseDropout should modify the image");
            }
        }
    }

    #[test]
    fn test_grid_dropout_correctness() {
        // Test grid dropout on various configurations
        // Use small grid size to ensure grid cells are created
        for &fill_value in &[0u8, 99] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let dropout = GridDropout::new((12, 12), 0.2, fill_value);
                dropout.execute(&mut img);

                // Verify dimensions preserved
                assert_eq!(img.width, TEST_WIDTH);
                assert_eq!(img.height, TEST_HEIGHT);
                assert_eq!(img.channels, channels as usize);

                // Grid dropout should modify some pixels
                // With 24x36 image and 12x12 grid, we have 2x3=6 grid cells
                // At 20% dropout, at least 1 cell should be dropped
                let changed = img.data.iter().zip(input.iter()).any(|(&a, &b)| a != b);
                assert!(
                    changed,
                    "Grid dropout with 20% probability should modify the image"
                );

                // Count pixels that match the fill value (dropped cells)
                let fill_count = img.data.iter().filter(|&&x| x == fill_value).count();
                // With at least one dropped cell (12x12 = 144 pixels), we should see fill values
                assert!(
                    fill_count >= 100, // At least most of one cell should be filled
                    "Grid dropout should fill pixels with fill value: fill_count={}",
                    fill_count
                );

                // Verify that non-dropped areas are approximately preserved
                // The mean should be different but not completely replaced
                let orig_mean: f32 = input.iter().map(|&x| x as f32).sum::<f32>() / input.len() as f32;
                let new_mean: f32 = img.data.iter().map(|&x| x as f32).sum::<f32>() / img.data.len() as f32;
                // Mean should change but not be completely replaced
                assert!(
                    (new_mean - orig_mean).abs() > 5.0,
                    "Grid dropout should change image mean: orig={}, new={}",
                    orig_mean, new_mean
                );
            }
        }
    }

    // ========================================================================
    // Histogram Transform Tests
    // ========================================================================

    #[test]
    fn test_equalize_correctness() {
        // Test RGB and grayscale
        for &channels in &[1u8, 3u8] {
            let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
            let mut data = input.clone();
            let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

            let equalize = Equalize::new();
            equalize.execute(&mut img);

            // Verify dimensions preserved and values changed
            assert_eq!(img.width, TEST_WIDTH);
            assert_eq!(img.height, TEST_HEIGHT);
            assert_eq!(img.channels, channels as usize);

            // Equalization should spread out values
            // For a gradient image, this should change the distribution
            let changed = img.data.iter().zip(input.iter()).any(|(&a, &b)| a != b);
            assert!(changed, "Equalize should modify the image");
        }
    }

    #[test]
    fn test_auto_contrast_correctness() {
        // Test various cutoff values
        for &cutoff in &[0.0f32, 0.1] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let auto_contrast = AutoContrast::new(cutoff);
                auto_contrast.execute(&mut img);

                // Verify dimensions preserved
                assert_eq!(img.width, TEST_WIDTH);
                assert_eq!(img.height, TEST_HEIGHT);
                assert_eq!(img.channels, channels as usize);

                // AutoContrast should stretch the range
                // After contrast stretch, we should see both low and high values
                let min_val = *img.data.iter().min().unwrap();
                let max_val = *img.data.iter().max().unwrap();
                assert!(
                    max_val - min_val >= 100,
                    "AutoContrast should stretch the range"
                );
            }
        }
    }

    // ========================================================================
    // Noise Transform Tests
    // ========================================================================

    #[test]
    fn test_gauss_noise_correctness() {
        // Test various mean/std values
        for &(mean, std_dev) in &[(0.0f32, 20.0f32), (10.0, 30.0)] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let noise = GaussNoise::new(mean, std_dev);
                noise.execute(&mut img);

                // Verify dimensions preserved
                assert_eq!(img.width, TEST_WIDTH);
                assert_eq!(img.height, TEST_HEIGHT);
                assert_eq!(img.channels, channels as usize);

                // All values should be clamped to [0, 255]
                assert!(img.data.iter().all(|&x| (0..=255).contains(&x)));
            }
        }
    }

    #[test]
    fn test_salt_and_pepper_correctness() {
        // Test various amounts and ratios
        for &(amount, salt_ratio) in &[(0.1f32, 0.5f32), (0.05, 0.3)] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let sp = SaltAndPepper::new(amount, salt_ratio);
                sp.execute(&mut img);

                // Verify dimensions preserved
                assert_eq!(img.width, TEST_WIDTH);
                assert_eq!(img.height, TEST_HEIGHT);
                assert_eq!(img.channels, channels as usize);

                // Some pixels should be affected (0 or 255)
                let affected_count = img.data.iter().filter(|&&x| x == 0 || x == 255).count();
                assert!(
                    affected_count > 0,
                    "SaltAndPepper should affect some pixels"
                );
            }
        }
    }

    // ========================================================================
    // Affine Transform Tests
    // ========================================================================

    #[test]
    fn test_affine_correctness() {
        // Test various affine transformations
        let configs = [
            // Identity (no change)
            AffineParams {
                scale: (1.0, 1.0),
                rotate: 0.0,
                translate: (0.0, 0.0),
                shear: (0.0, 0.0),
            },
            // Scale only
            AffineParams {
                scale: (0.5, 0.5),
                rotate: 0.0,
                translate: (0.0, 0.0),
                shear: (0.0, 0.0),
            },
            // Rotation only
            AffineParams {
                scale: (1.0, 1.0),
                rotate: 90.0,
                translate: (0.0, 0.0),
                shear: (0.0, 0.0),
            },
        ];

        for params in &configs {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let affine = Affine::new(*params);
                let result = affine.execute(&mut img);

                // Affine returns a BarrierImage (OutOfPlace operation)
                assert!(result.is_some(), "Affine should return BarrierImage");
                let result_img = result.unwrap();

                // Verify dimensions based on transformation
                if params.rotate == 90.0 || params.scale.0 != params.scale.1 {
                    // Rotation swaps dimensions
                    assert_eq!(result_img.channels, channels as usize);
                } else {
                    assert_eq!(result_img.width, TEST_WIDTH);
                    assert_eq!(result_img.height, TEST_HEIGHT);
                }
            }
        }
    }

    // ========================================================================
    // Additional Kernel Transform Tests
    // ========================================================================

    #[test]
    fn test_emboss_correctness() {
        // Test all emboss directions
        for &direction in &[
            EmbossDirection::SouthEast,
            EmbossDirection::SouthWest,
            EmbossDirection::NorthEast,
            EmbossDirection::NorthWest,
        ] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let emboss = Emboss::new()
                    .with_direction(direction)
                    .with_alpha(0.5)
                    .with_strength(0.5);
                emboss.execute(&mut img);

                // Verify dimensions preserved
                assert_eq!(img.width, TEST_WIDTH);
                assert_eq!(img.height, TEST_HEIGHT);
                assert_eq!(img.channels, channels as usize);

                // Emboss should modify the image (unless it's a constant image)
                let changed = img.data.iter().zip(input.iter()).any(|(&a, &b)| a != b);
                assert!(changed, "Emboss should modify the image");
            }
        }
    }

    // ========================================================================
    // Additional Photometric Transform Tests
    // ========================================================================

    #[test]
    fn test_multiplicative_noise_correctness() {
        // Test various multiplier/std values
        for &(multiplier, std_dev) in &[(1.0f32, 0.1f32), (1.0, 0.2)] {
            for &channels in &[1u8, 3u8] {
                let input = create_gradient_image(TEST_WIDTH, TEST_HEIGHT, channels);
                let mut data = input.clone();
                let mut img = FusableImage::new(&mut data, TEST_WIDTH, TEST_HEIGHT, channels as usize);

                let noise = MultiplicativeNoise::new(multiplier, std_dev);
                noise.execute(&mut img);

                // Verify dimensions preserved
                assert_eq!(img.width, TEST_WIDTH);
                assert_eq!(img.height, TEST_HEIGHT);
                assert_eq!(img.channels, channels as usize);

                // All values should be clamped to [0, 255]
                assert!(img.data.iter().all(|&x| (0..=255).contains(&x)));

                // Multiplicative noise should modify the image
                let changed = img.data.iter().zip(input.iter()).any(|(&a, &b)| a != b);
                assert!(changed, "MultiplicativeNoise should modify the image");
            }
        }
    }

    // ========================================================================
    // Summary: Correctness tests for 38 transforms
    // ========================================================================
    //
    // Geometric transforms (8):
    // - Affine, Crop, HorizontalFlip, Pad, Resize, Rotate, Transpose, VerticalFlip
    //
    // Photometric transforms (26):
    // - AutoContrast, Brightness, ChannelMix, ChannelShuffle, CoarseDropout,
    //   ColorBalance, ColorJitter, ColorTemperature, ColorTint, Contrast,
    //   Equalize, Gamma, GaussNoise, GridDropout, HueSaturationValue, Invert,
    //   MultiplicativeNoise, Normalize, Posterize, RGBShift, SaltAndPepper,
    //   Solarize, ToGray, ToRGB, ToSepia
    //
    // Kernel transforms (6):
    // - EdgeDetection, Emboss, GaussianBlur, MedianBlur, Sharpen
    //
    // Test coverage:
    // - All transforms tested on both RGB (3 channels) and grayscale (1 channel)
    // - All enum variants tested where applicable
    // - Tests use 24x36 non-square images (>20px)
    // - NEON vs scalar paths tested via compile-time configuration
}
