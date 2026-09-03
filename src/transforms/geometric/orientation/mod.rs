// D4 Dihedral Group Orientation State
//
// Represents all 8 possible orientations from combinations of
// 90° rotations and horizontal/vertical flips.

mod kernel;
mod tests;

pub use kernel::StructuralKernel;

use crate::core::{AccessPattern, ShapeEffect, Transform, Executable, FusableImage, BarrierImage};

/// The 8 orientations of the D4 dihedral group
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// No transformation
    Identity,
    /// Rotate 90° clockwise
    Rot90,
    /// Rotate 180°
    Rot180,
    /// Rotate 270° clockwise (90° CCW)
    Rot270,
    /// Horizontal flip (left-right mirror)
    FlipH,
    /// Vertical flip (top-bottom mirror)
    FlipV,
    /// Transpose (across main diagonal): FlipH then Rot90
    Transpose,
    /// Transverse (across anti-diagonal): FlipV then Rot90
    Transverse,
}

impl Orientation {
    /// Compose two orientations (this * other)
    ///
    /// Uses the D4 group multiplication table.
    /// This allows us to combine multiple geometric ops into one state.
    #[inline]
    pub fn compose(self, other: Orientation) -> Orientation {
        // D4 group multiplication table
        // Rows: self, Columns: other
        match (self, other) {
            // Identity is neutral
            (Orientation::Identity, o) => o,
            (s, Orientation::Identity) => s,

            // Rotations compose (clockwise: Rot90 + Rot90 = Rot180, etc.)
            (Orientation::Rot90, Orientation::Rot90) => Orientation::Rot180,
            (Orientation::Rot90, Orientation::Rot180) => Orientation::Rot270,
            (Orientation::Rot90, Orientation::Rot270) => Orientation::Identity,
            (Orientation::Rot180, Orientation::Rot90) => Orientation::Rot270,
            (Orientation::Rot180, Orientation::Rot180) => Orientation::Identity,
            (Orientation::Rot180, Orientation::Rot270) => Orientation::Rot90,
            (Orientation::Rot270, Orientation::Rot90) => Orientation::Identity,
            (Orientation::Rot270, Orientation::Rot180) => Orientation::Rot90,
            (Orientation::Rot270, Orientation::Rot270) => Orientation::Rot180,

            // Flip compositions
            (Orientation::FlipH, Orientation::FlipH) => Orientation::Identity,
            (Orientation::FlipV, Orientation::FlipV) => Orientation::Identity,

            // FlipH with rotations
            (Orientation::FlipH, Orientation::Rot90) => Orientation::Transverse,
            (Orientation::FlipH, Orientation::Rot180) => Orientation::FlipV,
            (Orientation::FlipH, Orientation::Rot270) => Orientation::Transpose,
            (Orientation::Rot90, Orientation::FlipH) => Orientation::Transpose,
            (Orientation::Rot180, Orientation::FlipH) => Orientation::FlipV,
            (Orientation::Rot270, Orientation::FlipH) => Orientation::Transverse,

            // FlipV with rotations
            (Orientation::FlipV, Orientation::Rot90) => Orientation::Transpose,
            (Orientation::FlipV, Orientation::Rot180) => Orientation::FlipH,
            (Orientation::FlipV, Orientation::Rot270) => Orientation::Transverse,
            (Orientation::Rot90, Orientation::FlipV) => Orientation::Transverse,
            (Orientation::Rot180, Orientation::FlipV) => Orientation::FlipH,
            (Orientation::Rot270, Orientation::FlipV) => Orientation::Transpose,

            // Transpose compositions
            (Orientation::Transpose, Orientation::Transpose) => Orientation::Identity,
            (Orientation::Transverse, Orientation::Transverse) => Orientation::Identity,

            // Transpose with rotations (verified in image coordinates)
            (Orientation::Transpose, Orientation::Rot90) => Orientation::FlipH,
            (Orientation::Transpose, Orientation::Rot180) => Orientation::Transverse,
            (Orientation::Transpose, Orientation::Rot270) => Orientation::FlipV,
            (Orientation::Rot90, Orientation::Transpose) => Orientation::FlipV,
            (Orientation::Rot180, Orientation::Transpose) => Orientation::Transverse,
            (Orientation::Rot270, Orientation::Transpose) => Orientation::FlipH,

            // Transverse with rotations (verified in image coordinates)
            (Orientation::Transverse, Orientation::Rot90) => Orientation::FlipV,
            (Orientation::Transverse, Orientation::Rot180) => Orientation::Transpose,
            (Orientation::Transverse, Orientation::Rot270) => Orientation::FlipH,
            (Orientation::Rot90, Orientation::Transverse) => Orientation::FlipH,
            (Orientation::Rot180, Orientation::Transverse) => Orientation::Transpose,
            (Orientation::Rot270, Orientation::Transverse) => Orientation::FlipV,

            // Mixed flip combinations
            (Orientation::FlipH, Orientation::FlipV) => Orientation::Rot180,
            (Orientation::FlipV, Orientation::FlipH) => Orientation::Rot180,
            (Orientation::FlipH, Orientation::Transpose) => Orientation::Rot270,
            (Orientation::FlipH, Orientation::Transverse) => Orientation::Rot90,
            (Orientation::FlipV, Orientation::Transpose) => Orientation::Rot90,
            (Orientation::FlipV, Orientation::Transverse) => Orientation::Rot270,
            (Orientation::Transpose, Orientation::FlipH) => Orientation::Rot90,
            (Orientation::Transpose, Orientation::FlipV) => Orientation::Rot270,
            (Orientation::Transverse, Orientation::FlipH) => Orientation::Rot270,
            (Orientation::Transverse, Orientation::FlipV) => Orientation::Rot90,
            (Orientation::Transpose, Orientation::Transverse) => Orientation::Rot180,
            (Orientation::Transverse, Orientation::Transpose) => Orientation::Rot180,
        }
    }

    /// Get the output dimensions for this orientation
    #[inline]
    pub fn output_size(&self, width: u32, height: u32) -> (u32, u32) {
        match self {
            Orientation::Identity | Orientation::Rot180 |
            Orientation::FlipH | Orientation::FlipV => (width, height),
            Orientation::Rot90 | Orientation::Rot270 |
            Orientation::Transpose | Orientation::Transverse => (height, width),
        }
    }

    /// Check if this orientation preserves dimensions
    #[inline]
    pub fn preserves_size(&self) -> bool {
        matches!(self,
            Orientation::Identity | Orientation::Rot180 |
            Orientation::FlipH | Orientation::FlipV
        )
    }

    /// Check if this orientation swaps dimensions
    #[inline]
    pub fn swaps_dimensions(&self) -> bool {
        !self.preserves_size()
    }
}
