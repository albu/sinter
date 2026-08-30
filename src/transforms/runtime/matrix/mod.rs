// RGB Matrix transform fusion
//
// Fuses multiple 3x3 matrix operations on RGB images into a single pass.
//
// Similar to LUT fusion but for operations that mix RGB channels:
// - ToSepia, Saturation, ColorTemperature, HueRotate, etc.
//
// Performance benefits:
// - Single pass instead of N passes
// - Cache-friendly sequential access
// - SIMD-friendly (can be optimized further)
//
// Algorithm:
// 1. Compose matrices: M_combined = M_n * ... * M_2 * M_1
// 2. Apply once: out = M_combined * in for each pixel

use crate::core::FusableImage;
use std::fmt;

mod executor;
pub mod fused;
mod tests;

pub use executor::MatrixExecutor;

/// Trait for transforms that can be expressed as 3x3 RGB matrix operations
///
/// Any transform where each output pixel is a linear combination of RGB inputs:
/// ```text
/// [R']   [m00 m01 m02] [R]
/// [G'] = [m10 m11 m12] * [G]
/// [B']   [m20 m21 m22] [B]
/// ```
///
/// Matrix is row-major: matrix[row][col]
pub trait MatrixOp: fmt::Debug {
    /// Get the 3x3 transformation matrix
    ///
    /// Returns row-major 3x3 matrix where:
    /// - row 0: R' = m[0][0]*R + m[0][1]*G + m[0][2]*B
    /// - row 1: G' = m[1][0]*R + m[1][1]*G + m[1][2]*B
    /// - row 2: B' = m[2][0]*R + m[2][1]*G + m[2][2]*B
    fn get_matrix(&self) -> [[f32; 3]; 3];

    /// Execute using matrix (default implementation)
    ///
    /// Transforms can override this for specialized behavior.
    fn execute_with_matrix(&self, image: &mut FusableImage) {
        let matrix = self.get_matrix();
        MatrixExecutor::apply(image, &matrix);
    }
}

/// Compose multiple matrix operations into a single matrix
///
/// For operations: op1 → op2 → op3
/// Combined matrix = M3 * M2 * M1
///
/// # Arguments
/// * `ops` - Slice of matrix operations to compose (applied in order)
///
/// # Returns
/// Single composed 3x3 matrix
pub fn compose_matrices(ops: &[&dyn MatrixOp]) -> [[f32; 3]; 3] {
    // Start with identity matrix
    let mut result = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Compose in reverse order (matrix multiplication is right-to-left)
    // If we want op1 → op2 → op3 (apply op1 first, then op2, then op3),
    // we compute M3 * M2 * M1 * I
    for op in ops.iter().rev() {
        let m = op.get_matrix();
        result = multiply_matrices(&result, &m);
    }

    result
}

/// Multiply two 3x3 matrices: C = A * B
fn multiply_matrices(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Apply a single 3x3 matrix transform to an RGB image
///
/// For each pixel:
/// ```text
/// [R']   [m00 m01 m02] [R]
/// [G'] = [m10 m11 m12] * [G]
/// [B']   [m20 m21 m22] [B]
/// ```
///
/// # Arguments
/// * `image` - RGB image to transform (must have 3 channels)
/// * `matrix` - 3x3 transformation matrix (row-major)
///
/// # Panics
/// Panics if image doesn't have exactly 3 channels
pub fn apply_matrix(image: &mut FusableImage, matrix: &[[f32; 3]; 3]) {
    assert_eq!(
        image.channels, 3,
        "Matrix transforms only work with RGB images (3 channels)"
    );
    MatrixExecutor::apply(image, matrix);
}
