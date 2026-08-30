// NEON SIMD implementation for Grayscale to RGB
//
// Uses vst3_u8 for RGB interleaving (grayscale replication).

/// Convert grayscale to RGB using NEON SIMD
///
/// Processes 8 pixels at a time (24 bytes = 8 RGB pixels).
/// Uses vst3_u8 to interleave the replicated grayscale values efficiently.
#[cfg(target_arch = "aarch64")]
pub unsafe fn to_rgb_neon(src: &[u8], dst: &mut [u8], pixel_count: usize) {
    use std::arch::aarch64::*;

    let mut i = 0;

    // Process 16 pixels at a time (48 bytes)
    while i + 16 <= pixel_count {
        let gray = vld1q_u8(src.as_ptr().add(i));
        let rgb = uint8x16x3_t(gray, gray, gray);
        vst3q_u8(dst.as_mut_ptr().add(i * 3), rgb);
        i += 16;
    }

    // Process 8 pixels if available
    while i + 8 <= pixel_count {
        let gray = vld1_u8(src.as_ptr().add(i));
        let rgb = uint8x8x3_t(gray, gray, gray);
        vst3_u8(dst.as_mut_ptr().add(i * 3), rgb);
        i += 8;
    }

    // Handle remaining pixels scalar
    for j in i..pixel_count {
        let gray = src[j];
        dst[j * 3] = gray;
        dst[j * 3 + 1] = gray;
        dst[j * 3 + 2] = gray;
    }
}

/// Fallback for non-ARM64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn to_rgb_neon(src: &[u8], dst: &mut [u8], pixel_count: usize) {
    for i in 0..pixel_count {
        let gray = src[i];
        dst[i * 3] = gray;
        dst[i * 3 + 1] = gray;
        dst[i * 3 + 2] = gray;
    }
}
