// Edge detection transform
//
// Applies Laplacian or Sobel operators for edge detection.

use super::convolve::convolve_3x3;
use crate::core::{AccessPattern, Executable, FusableImage, ShapeEffect, Transform};

/// Edge detection method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeMethod {
    /// Laplacian operator - detects edges in all directions
    Laplacian,
    /// Sobel operator - detects edges with directional sensitivity
    Sobel,
}

/// Edge detection transform
///
/// Detects edges in the image using either Laplacian or Sobel operators.
///
/// **Laplacian**: Detects edges in all directions by computing the second derivative.
/// Produces thin edges and is sensitive to noise.
///
/// **Sobel**: Computes gradient magnitude using horizontal and vertical kernels.
/// More robust to noise, produces thicker edges.
///
/// # Parameters
/// - `method`: Edge detection method (Laplacian or Sobel)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeDetection {
    pub method: EdgeMethod,
}

impl EdgeDetection {
    /// Create a new EdgeDetection transform with Laplacian method
    pub fn laplacian() -> Self {
        Self {
            method: EdgeMethod::Laplacian,
        }
    }

    /// Create a new EdgeDetection transform with Sobel method
    pub fn sobel() -> Self {
        Self {
            method: EdgeMethod::Sobel,
        }
    }

    /// Create with custom method
    pub fn new(method: EdgeMethod) -> Self {
        Self { method }
    }
}

impl Transform for EdgeDetection {
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

impl Executable for EdgeDetection {
    fn execute(&self, image: &mut FusableImage) -> Option<crate::core::BarrierImage> {
        match self.method {
            EdgeMethod::Laplacian => self.apply_laplacian(image),
            EdgeMethod::Sobel => self.apply_sobel(image),
        }
        None
    }
}

impl EdgeDetection {
    /// Apply Laplacian edge detection
    ///
    /// Kernel:
    ///  0  1  0
    ///  1 -4  1
    ///  0  1  0
    fn apply_laplacian(&self, image: &mut FusableImage) {
        super::convolve_2d::apply_laplacian(image);
    }

    /// Apply Sobel edge detection
    ///
    /// Computes gradient magnitude using horizontal and vertical Sobel kernels:
    /// Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] (horizontal)
    /// Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] (vertical)
    /// Result = sqrt(Gx² + Gy²)
    fn apply_sobel(&self, image: &mut FusableImage) {
        let width = image.width;
        let height = image.height;
        let channels = image.channels;
        let data = &image.data;

        // Create output buffer
        let mut output = vec![0u8; data.len()];

        // Sobel kernels
        let gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]; // Horizontal gradient
        let gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1]; // Vertical gradient

        // Helper to get pixel value with edge extension
        let get_pixel = |data: &[u8], x: i32, y: i32, c: usize| -> u8 {
            let x_clamped = x.max(0).min(width as i32 - 1) as usize;
            let y_clamped = y.max(0).min(height as i32 - 1) as usize;
            data[(y_clamped * width as usize + x_clamped) * channels + c]
        };

        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let mut sum_x: i32 = 0;
                    let mut sum_y: i32 = 0;

                    // Apply 3x3 kernels
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let px = x as i32 + kx as i32 - 1;
                            let py = y as i32 + ky as i32 - 1;
                            let pixel = get_pixel(data, px, py, c) as i32;
                            sum_x += pixel * gx[ky * 3 + kx];
                            sum_y += pixel * gy[ky * 3 + kx];
                        }
                    }

                    // Compute gradient magnitude
                    let magnitude = ((sum_x * sum_x + sum_y * sum_y) as f64).sqrt() as i32;
                    output[(y * width + x) * channels + c] = magnitude.clamp(0, 255) as u8;
                }
            }
        }

        // Copy output back to image
        image.data.copy_from_slice(&output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_detection_laplacian() {
        let e = EdgeDetection::laplacian();
        assert_eq!(e.method, EdgeMethod::Laplacian);
    }

    #[test]
    fn test_edge_detection_sobel() {
        let e = EdgeDetection::sobel();
        assert_eq!(e.method, EdgeMethod::Sobel);
    }

    #[test]
    fn test_edge_detection_new() {
        let e = EdgeDetection::new(EdgeMethod::Laplacian);
        assert_eq!(e.method, EdgeMethod::Laplacian);
    }

    #[test]
    fn test_laplacian_constant() {
        // Constant image should produce zero (no edges)
        let mut data = vec![128u8; 9]; // 3x3 constant
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        EdgeDetection::laplacian().execute(&mut img);

        // All pixels should be 0 (no edges in constant image)
        assert!(img.data.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_laplacian_horizontal_edge() {
        // Image with horizontal edge
        // 0 0 0
        // 255 255 255
        let mut data = vec![0u8, 0u8, 0u8, 255u8, 255u8, 255u8];
        let mut img = FusableImage::new(&mut data, 3, 2, 1);

        EdgeDetection::laplacian().execute(&mut img);

        // Edge pixels should have high values
        // Check that at least some pixels detected the edge
        let edge_max = img.data.iter().cloned().max().unwrap();
        assert!(edge_max > 0, "Edge should be detected");
    }

    #[test]
    fn test_sobel_constant() {
        // Constant image should produce zero gradient
        let mut data = vec![128u8; 9]; // 3x3 constant
        let mut img = FusableImage::new(&mut data, 3, 3, 1);

        EdgeDetection::sobel().execute(&mut img);

        // All pixels should be 0 (no gradient in constant image)
        assert!(img.data.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_sobel_diagonal_edge() {
        // Image with diagonal edge
        // 255 0
        // 0 0
        let mut data = vec![255u8, 0u8, 0u8, 0u8];
        let mut img = FusableImage::new(&mut data, 2, 2, 1);

        EdgeDetection::sobel().execute(&mut img);

        // At least some pixels should detect the edge
        let max_val = *img.data.iter().max().unwrap();
        assert!(max_val > 0, "Edge should be detected");
    }

    #[test]
    fn test_edge_detection_rgb() {
        // Test RGB image
        let mut data = vec![
            100u8, 100u8, 100u8, 128u8, 128u8, 128u8, 150u8, 150u8, 150u8,
        ];
        let mut img = FusableImage::new(&mut data, 3, 1, 3);

        EdgeDetection::laplacian().execute(&mut img);

        // Each channel should be processed independently
        // For Laplacian, gradient image should have some variation
        assert_eq!(img.data.len(), 9);
    }

    #[test]
    fn test_sobel_vs_laplacian() {
        // Both methods should work on the same image
        let mut data1 = vec![0u8, 0u8, 0u8, 255u8, 255u8, 255u8];
        let mut data2 = data1.clone();

        let mut img1 = FusableImage::new(&mut data1, 3, 2, 1);
        let mut img2 = FusableImage::new(&mut data2, 3, 2, 1);

        EdgeDetection::laplacian().execute(&mut img1);
        EdgeDetection::sobel().execute(&mut img2);

        // Both should detect edges (not all zeros)
        let has_edge1 = img1.data.iter().any(|&p| p > 0);
        let has_edge2 = img2.data.iter().any(|&p| p > 0);

        assert!(has_edge1, "Laplacian should detect edge");
        assert!(has_edge2, "Sobel should detect edge");
    }
}
