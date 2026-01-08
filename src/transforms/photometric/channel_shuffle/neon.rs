// NEON SIMD implementation for channel shuffle
//
// This module contains ARM64 NEON optimizations for channel permutation.
// For permutations, we don't need any arithmetic - just reordering.
// After vld3 de-interleaves RGB into separate lanes, we reorder the lanes.

use crate::transforms::photometric::channel_shuffle::ChannelOrder;
use crate::core::FusableImage;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Apply channel shuffle using NEON SIMD
#[cfg(target_arch = "aarch64")]
pub fn apply_shuffle_neon(image: &mut FusableImage, order: ChannelOrder) {
    let data = &mut image.data;
    let len = data.len();
    let chunks = len / 48; // 48 bytes = 16 RGB pixels

    unsafe {
        let mut offset = 0;

        for _ in 0..chunks {
            let src = data.as_ptr().add(offset) as *const u8;
            let dst = data.as_mut_ptr().add(offset) as *mut u8;

            // vld3: De-interleave 16 RGB pixels -> {R16, G16, B16}
            let rgb = vld3q_u8(src);

            // Reorder lanes based on permutation (NO arithmetic!)
            let shuffled = match order {
                ChannelOrder::RGB => {
                    // Identity - no change
                    rgb
                }
                ChannelOrder::BGR => {
                    // Swap R and B: (R,G,B) -> (B,G,R)
                    uint8x16x3_t(rgb.2, rgb.1, rgb.0)
                }
                ChannelOrder::GRB => {
                    // Swap R and G: (R,G,B) -> (G,R,B)
                    uint8x16x3_t(rgb.1, rgb.0, rgb.2)
                }
                ChannelOrder::GBR => {
                    // Rotate left: (R,G,B) -> (G,B,R)
                    uint8x16x3_t(rgb.1, rgb.2, rgb.0)
                }
                ChannelOrder::RBG => {
                    // Swap G and B: (R,G,B) -> (R,B,G)
                    uint8x16x3_t(rgb.0, rgb.2, rgb.1)
                }
                ChannelOrder::BRG => {
                    // Rotate right: (R,G,B) -> (B,R,G)
                    uint8x16x3_t(rgb.2, rgb.0, rgb.1)
                }
            };

            // vst3: Re-interleave and store 16 RGB pixels
            vst3q_u8(dst, shuffled);

            offset += 48;
        }

        // Handle remaining pixels with scalar fallback
        if chunks * 48 < len {
            super::apply_shuffle_scalar_range(data, offset, len, order);
        }
    }
}

/// Fallback implementation for non-ARM64 architectures
#[cfg(not(target_arch = "aarch64"))]
pub fn apply_shuffle_neon(image: &mut FusableImage, order: ChannelOrder) {
    // On non-ARM64 platforms, just use the scalar implementation
    let len = image.data.len();
    super::apply_shuffle_scalar_range(&mut image.data, 0, len, order);
}
