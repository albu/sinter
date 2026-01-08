// Channel Mix transform
//
// Applies a custom 3x3 RGB mixing matrix to an image.
// This is a general-purpose transform that can express any linear RGB color transformation.
//
// This is a 3x3 RGB matrix operation that can be fused with other MatrixOp transforms.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::runtime::matrix::{apply_matrix, MatrixOp};

/// ChannelMix transform
///
/// Applies a custom 3x3 RGB mixing matrix to an image. This is the most general
/// form of RGB matrix transformation and can express:
/// - Grayscale conversion
/// - Sepia tone
/// - Channel swapping
/// - Custom color grading
///
/// # Parameters
/// - `matrix`: A 3x3 matrix where each row defines the output channel
///   - Row 0: Output R = R*m[0][0] + G*m[0][1] + B*m[0][2]
///   - Row 1: Output G = R*m[1][0] + G*m[1][1] + B*m[1][2]
///   - Row 2: Output B = R*m[2][0] + G*m[2][1] + B*m[2][2]
///
/// # Example
/// ```text
/// // Swap R and B channels (BGR conversion)
/// let matrix = [
///     [0.0, 0.0, 1.0],  // R' = B
///     [0.0, 1.0, 0.0],  // G' = G
///     [1.0, 0.0, 0.0],  // B' = R
/// ];
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelMix {
    /// The 3x3 RGB mixing matrix
    pub matrix: [[f32; 3]; 3],
}

impl ChannelMix {
    /// Create a new ChannelMix transform with a custom matrix
    ///
    /// # Arguments
    /// * `matrix` - A 3x3 RGB mixing matrix
    pub fn new(matrix: [[f32; 3]; 3]) -> Self {
        Self { matrix }
    }

    /// Create a transform that swaps R and B channels (BGR conversion)
    ///
    /// This is useful for converting between RGB and BGR color formats.
    pub fn bgr() -> Self {
        Self {
            matrix: [
                [0.0, 0.0, 1.0], // R' = B
                [0.0, 1.0, 0.0], // G' = G
                [1.0, 0.0, 0.0], // B' = R
            ],
        }
    }

    /// Create a transform that swaps G and B channels
    pub fn swap_gb() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0], // R' = R
                [0.0, 0.0, 1.0], // G' = B
                [0.0, 1.0, 0.0], // B' = G
            ],
        }
    }

    /// Create a transform that swaps R and G channels
    pub fn swap_rg() -> Self {
        Self {
            matrix: [
                [0.0, 1.0, 0.0], // R' = G
                [1.0, 0.0, 0.0], // G' = R
                [0.0, 0.0, 1.0], // B' = B
            ],
        }
    }

    /// Create a vintage/warm look (boosts reds, reduces blues)
    pub fn vintage() -> Self {
        Self {
            matrix: [
                [1.2, 0.1, 0.0], // R' = 1.2*R + 0.1*G
                [0.0, 1.0, 0.1], // G' = G + 0.1*B
                [0.0, 0.0, 0.9], // B' = 0.9*B
            ],
        }
    }

    /// Create a cool look (boosts blues, reduces reds)
    pub fn cool() -> Self {
        Self {
            matrix: [
                [0.9, 0.0, 0.0], // R' = 0.9*R
                [0.0, 1.0, 0.0], // G' = G
                [0.1, 0.1, 1.2], // B' = 0.1*R + 0.1*G + 1.2*B
            ],
        }
    }
}

impl Default for ChannelMix {
    fn default() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

impl Transform for ChannelMix {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl MatrixOp for ChannelMix {
    fn get_matrix(&self) -> [[f32; 3]; 3] {
        self.matrix
    }
}

impl Executable for ChannelMix {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        if image.channels != 3 {
            return None;
        }
        apply_matrix(image, &self.matrix);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_mix_new() {
        let matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cm = ChannelMix::new(matrix);
        assert_eq!(cm.matrix, matrix);
    }

    #[test]
    fn test_channel_mix_default() {
        let cm = ChannelMix::default();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(cm.matrix, identity);
    }

    #[test]
    fn test_channel_mix_bgr() {
        let cm = ChannelMix::bgr();
        let expected = [
            [0.0, 0.0, 1.0], // R' = B
            [0.0, 1.0, 0.0], // G' = G
            [1.0, 0.0, 0.0], // B' = R
        ];
        assert_eq!(cm.matrix, expected);
    }

    #[test]
    fn test_channel_mix_execute_identity() {
        // Identity matrix should leave image unchanged
        let mut data = vec![100u8, 150, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        ChannelMix::new(identity).execute(&mut img);

        assert_eq!(img.data[0], 100);
        assert_eq!(img.data[1], 150);
        assert_eq!(img.data[2], 200);
    }

    #[test]
    fn test_channel_mix_execute_bgr() {
        // BGR swap: R and B should be swapped
        let mut data = vec![100u8, 150, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ChannelMix::bgr().execute(&mut img);

        // R and B swapped
        assert_eq!(img.data[0], 200); // R' = original B
        assert_eq!(img.data[1], 150); // G' = original G
        assert_eq!(img.data[2], 100); // B' = original R
    }

    #[test]
    fn test_channel_mix_execute_swap_gb() {
        // G and B swap
        let mut data = vec![100u8, 150, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ChannelMix::swap_gb().execute(&mut img);

        assert_eq!(img.data[0], 100); // R' = original R
        assert_eq!(img.data[1], 200); // G' = original B
        assert_eq!(img.data[2], 150); // B' = original G
    }

    #[test]
    fn test_channel_mix_execute_swap_rg() {
        // R and G swap
        let mut data = vec![100u8, 150, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ChannelMix::swap_rg().execute(&mut img);

        assert_eq!(img.data[0], 150); // R' = original G
        assert_eq!(img.data[1], 100); // G' = original R
        assert_eq!(img.data[2], 200); // B' = original B
    }

    #[test]
    fn test_channel_mix_grayscale_passthrough() {
        // Grayscale should be unchanged
        let mut data = vec![128u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 1);

        ChannelMix::bgr().execute(&mut img);

        assert_eq!(img.data[0], 128);
    }

    #[test]
    fn test_channel_mix_access_pattern() {
        let cm = ChannelMix::bgr();
        assert_eq!(cm.access(), AccessPattern::InPlace);
        assert_eq!(cm.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_channel_mix_vintage_look() {
        let mut data = vec![200u8, 200, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ChannelMix::vintage().execute(&mut img);

        // Red should be boosted (higher than G and B)
        assert!(img.data[0] >= img.data[1]);
        assert!(img.data[0] >= img.data[2]);
    }

    #[test]
    fn test_channel_mix_cool_look() {
        let mut data = vec![200u8, 200, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        ChannelMix::cool().execute(&mut img);

        // Blue should be boosted
        assert!(img.data[2] >= img.data[0]);
        assert!(img.data[2] >= img.data[1]);
    }
}
