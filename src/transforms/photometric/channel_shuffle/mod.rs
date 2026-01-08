// Channel Shuffle transform
//
// Randomly permutes RGB channels. This is a useful data augmentation technique
// that helps models learn to be color-agnostic.
//
// This is a 3x3 RGB matrix operation (permutation matrix) that can be fused
// with other MatrixOp transforms.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use crate::transforms::runtime::matrix::MatrixOp;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Channel ordering for ChannelShuffle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelOrder {
    /// RGB order (no change)
    RGB,
    /// BGR order (swap R and B)
    BGR,
    /// GRB order (swap R and G)
    GRB,
    /// GBR order
    GBR,
    /// RBG order
    RBG,
    /// BRG order
    BRG,
}

impl ChannelOrder {
    /// Get all possible channel orders
    pub fn all() -> [ChannelOrder; 6] {
        [
            ChannelOrder::RGB,
            ChannelOrder::BGR,
            ChannelOrder::GRB,
            ChannelOrder::GBR,
            ChannelOrder::RBG,
            ChannelOrder::BRG,
        ]
    }

    /// Get the permutation matrix for this channel order
    fn permutation_matrix(&self) -> [[f32; 3]; 3] {
        match self {
            ChannelOrder::RGB => [
                [1.0, 0.0, 0.0], // R' = R
                [0.0, 1.0, 0.0], // G' = G
                [0.0, 0.0, 1.0], // B' = B
            ],
            ChannelOrder::BGR => [
                [0.0, 0.0, 1.0], // R' = B
                [0.0, 1.0, 0.0], // G' = G
                [1.0, 0.0, 0.0], // B' = R
            ],
            ChannelOrder::GRB => [
                [0.0, 1.0, 0.0], // R' = G
                [1.0, 0.0, 0.0], // G' = R
                [0.0, 0.0, 1.0], // B' = B
            ],
            ChannelOrder::GBR => [
                [0.0, 1.0, 0.0], // R' = G
                [0.0, 0.0, 1.0], // G' = B
                [1.0, 0.0, 0.0], // B' = R
            ],
            ChannelOrder::RBG => [
                [1.0, 0.0, 0.0], // R' = R
                [0.0, 0.0, 1.0], // G' = B
                [0.0, 1.0, 0.0], // B' = G
            ],
            ChannelOrder::BRG => [
                [0.0, 0.0, 1.0], // R' = B
                [1.0, 0.0, 0.0], // G' = R
                [0.0, 1.0, 0.0], // B' = G
            ],
        }
    }
}

/// ChannelShuffle transform
///
/// Permutes RGB channels according to a specified channel order.
/// This is useful for data augmentation to make models color-agnostic.
///
/// # Parameters
/// - `order`: The channel ordering to apply
///
/// # Algorithm
/// Uses a permutation matrix to reorder RGB channels.
///
/// # Example
/// ```text
/// ChannelShuffle(ChannelOrder::BGR): Converts RGB to BGR
/// ChannelShuffle(ChannelOrder::GRB): Swaps R and G
/// ```
///
/// # Data Augmentation Use Case
/// Random channel shuffling helps neural networks learn that color order
/// shouldn't affect the task (e.g., object recognition should work regardless
/// of whether the image is RGB or BGR).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelShuffle {
    pub order: ChannelOrder,
}

impl ChannelShuffle {
    /// Create a new ChannelShuffle transform with the specified order
    pub fn new(order: ChannelOrder) -> Self {
        Self { order }
    }

    /// Create RGB to BGR conversion
    pub fn bgr() -> Self {
        Self {
            order: ChannelOrder::BGR,
        }
    }

    /// Create a random channel shuffle
    ///
    /// Note: For deterministic random shuffling during training, use this
    /// with a random number generator. The randomness happens at creation time.
    pub fn random() -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::SystemTime;

        // Simple deterministic "random" based on time
        let mut hasher = DefaultHasher::new();
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut hasher);
        let idx = (hasher.finish() % 6) as usize;

        Self {
            order: ChannelOrder::all()[idx],
        }
    }
}

impl Default for ChannelShuffle {
    fn default() -> Self {
        Self {
            order: ChannelOrder::RGB,
        }
    }
}

impl Transform for ChannelShuffle {
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

impl MatrixOp for ChannelShuffle {
    fn get_matrix(&self) -> [[f32; 3]; 3] {
        self.order.permutation_matrix()
    }
}

impl Executable for ChannelShuffle {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        if image.channels != 3 {
            return None;
        }
        // Use specialized SIMD shuffle path (faster than general matrix multiply)
        apply_shuffle(image, self.order);
        None
    }
}

/// Apply channel shuffle using specialized SIMD paths
///
/// For permutations, we don't need any arithmetic - just reordering.
/// After vld3 de-interleaves RGB into separate lanes, we reorder the lanes.
fn apply_shuffle(image: &mut FusableImage, order: ChannelOrder) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::apply_shuffle_neon(image, order);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Fallback to scalar for other architectures
        apply_shuffle_scalar(image, order);
    }
}

/// Scalar fallback for non-ARM64 or remaining pixels
#[allow(dead_code)]
fn apply_shuffle_scalar(image: &mut FusableImage, order: ChannelOrder) {
    let len = image.data.len();
    apply_shuffle_scalar_range(&mut image.data, 0, len, order);
}

pub(crate) fn apply_shuffle_scalar_range(data: &mut [u8], mut offset: usize, end: usize, order: ChannelOrder) {
    while offset + 3 <= end {
        let r = data[offset];
        let g = data[offset + 1];
        let b = data[offset + 2];

        let (r_out, g_out, b_out) = match order {
            ChannelOrder::RGB => (r, g, b),
            ChannelOrder::BGR => (b, g, r),
            ChannelOrder::GRB => (g, r, b),
            ChannelOrder::GBR => (g, b, r),
            ChannelOrder::RBG => (r, b, g),
            ChannelOrder::BRG => (b, r, g),
        };

        data[offset] = r_out;
        data[offset + 1] = g_out;
        data[offset + 2] = b_out;

        offset += 3;
    }
}

#[cfg(test)]
mod tests;
