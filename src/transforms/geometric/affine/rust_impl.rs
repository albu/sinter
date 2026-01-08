// Rust fallback implementation for affine transforms
//
// Pure Rust implementation used when OpenCV is not available or as fallback.

use crate::core::FusableImage;
use crate::transforms::geometric::affine::{Affine, AffineInterpolation};
use crate::core::BarrierImage;
use super::{bilinear_interpolate, nearest_interpolate};

/// Execute using Rust implementation (fallback when OpenCV is not available)
pub(super) fn execute_rust(affine: &Affine, image: &FusableImage) -> BarrierImage {
    let (out_width, out_height) = affine.output_size.unwrap_or((image.width, image.height));
    let channels = image.channels;
    let mut transformed_data = vec![0u8; out_width * out_height * channels];

    // Build inverse transformation matrix
    let [a, b, c, d, e, f] = affine.build_inverse_matrix();

    // Apply inverse mapping: for each output pixel, find corresponding input pixel
    for y_out in 0..out_height {
        for x_out in 0..out_width {
            // Map output coordinates to input coordinates
            let x_in = a * x_out as f32 + b * y_out as f32 + c;
            let y_in = d * x_out as f32 + e * y_out as f32 + f;

            // Interpolate for each channel
            for ch in 0..channels {
                let out_idx = (y_out * out_width + x_out) * channels + ch;
                transformed_data[out_idx] = match affine.interpolation {
                    AffineInterpolation::Nearest => nearest_interpolate(
                        &image.data,
                        x_in,
                        y_in,
                        image.width,
                        image.height,
                        channels,
                        ch,
                    ),
                    AffineInterpolation::Bilinear => bilinear_interpolate(
                        &image.data,
                        x_in,
                        y_in,
                        image.width,
                        image.height,
                        channels,
                        ch,
                    ),
                };
            }
        }
    }

    BarrierImage::from_vec(transformed_data, out_width, out_height, channels)
}
