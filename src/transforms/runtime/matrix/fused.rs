// Fused Matrix transform
//
// Represents a composition of multiple 3x3 RGB matrix operations into a single matrix.
// Created by the optimizer when it detects consecutive MatrixOp transforms.
//
// This is the output of matrix fusion optimization. For example:
// ToSepia → Saturation becomes FusedMatrix { matrix: M_saturation × M_tosepia }
//
// OPTIMIZATION: Uses pure Rust implementation with NEON SIMD,
// which is faster than OpenCV's cv::transform for this use case.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::runtime::matrix::{MatrixExecutor, MatrixOp};
use std::fmt;

/// Fused Matrix transform
///
/// Represents a composed 3x3 RGB matrix operation. This is created by the optimizer
/// when it detects consecutive transforms that implement MatrixOp (ToSepia, Saturation, etc.)
///
/// # Example
/// If you have: ToSepia → Saturation
/// The optimizer composes: M_combined = M_saturation × M_tosepia
/// And creates: FusedMatrix { matrix: M_combined }
///
/// Then execution only needs ONE pass over pixels instead of TWO.
#[derive(Debug, Clone, Copy)]
pub struct FusedMatrix {
    /// The composed 3x3 transformation matrix
    pub matrix: [[f32; 3]; 3],
}

impl FusedMatrix {
    /// Create a new FusedMatrix with the given matrix
    pub fn new(matrix: [[f32; 3]; 3]) -> Self {
        Self { matrix }
    }

    /// Create from a slice of matrix operations
    pub fn from_matrix_ops(ops: &[&dyn crate::transforms::runtime::matrix::MatrixOp]) -> Self {
        let matrix = crate::transforms::runtime::matrix::compose_matrices(ops);
        Self { matrix }
    }

    /// Get the matrix
    pub fn get_matrix(&self) -> [[f32; 3]; 3] {
        self.matrix
    }
}

impl Transform for FusedMatrix {
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

impl MatrixOp for FusedMatrix {
    fn get_matrix(&self) -> [[f32; 3]; 3] {
        self.matrix
    }
}

impl Executable for FusedMatrix {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        if image.channels != 3 {
            return None;
        }

        // Use pure Rust implementation with NEON SIMD (faster than OpenCV)
        MatrixExecutor::apply(image, &self.matrix);
        None
    }
}

impl fmt::Display for FusedMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FusedMatrix([[{:.3},{:.3},{:.3}],[{:.3},{:.3},{:.3}],[{:.3},{:.3},{:.3}]])",
            self.matrix[0][0],
            self.matrix[0][1],
            self.matrix[0][2],
            self.matrix[1][0],
            self.matrix[1][1],
            self.matrix[1][2],
            self.matrix[2][0],
            self.matrix[2][1],
            self.matrix[2][2],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_matrix_identity() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let mut data = vec![100u8, 150u8, 200u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        FusedMatrix::new(identity).execute(&mut img);

        assert_eq!(img.data[0], 100);
        assert_eq!(img.data[1], 150);
        assert_eq!(img.data[2], 200);
    }

    #[test]
    fn test_fused_matrix_sepia() {
        // Standard sepia matrix
        let sepia = [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ];

        let mut data = vec![255u8, 0u8, 0u8];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);

        FusedMatrix::new(sepia).execute(&mut img);

        assert!((img.data[0] as i32 - 100).abs() <= 1);
        assert!((img.data[1] as i32 - 89).abs() <= 1);
        assert!((img.data[2] as i32 - 69).abs() <= 1);
    }

    #[test]
    fn test_fused_matrix_access_pattern() {
        let fm = FusedMatrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(fm.access(), AccessPattern::InPlace);
        assert_eq!(fm.shape_effect(), ShapeEffect::Preserve);
    }

    #[test]
    fn test_fused_matrix_display() {
        let sepia = [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ];
        let fm = FusedMatrix::new(sepia);
        let s = format!("{}", fm);
        assert!(s.contains("FusedMatrix"));
    }
}
