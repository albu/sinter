// Matrix executor for RGB transforms
//
// Provides fused execution of multiple 3x3 matrix operations.

use super::{compose_matrices, MatrixOp};
use crate::core::FusableImage;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Executor for RGB matrix operations
pub struct MatrixExecutor;

impl MatrixExecutor {
    /// Apply a single matrix transform to an image
    ///
    /// # Arguments
    /// * `image` - RGB image to transform
    /// * `matrix` - 3x3 transformation matrix (row-major)
    #[cfg(target_arch = "aarch64")]
    pub fn apply(image: &mut FusableImage, matrix: &[[f32; 3]; 3]) {
        let data = &mut image.data;
        let len = data.len();

        // Convert to Q8.8 fixed-point (scale by 256)
        let m00 = (matrix[0][0] * 256.0) as i16;
        let m01 = (matrix[0][1] * 256.0) as i16;
        let m02 = (matrix[0][2] * 256.0) as i16;
        let m10 = (matrix[1][0] * 256.0) as i16;
        let m11 = (matrix[1][1] * 256.0) as i16;
        let m12 = (matrix[1][2] * 256.0) as i16;

        let m20 = (matrix[2][0] * 256.0) as i16;
        let m21 = (matrix[2][1] * 256.0) as i16;
        let m22 = (matrix[2][2] * 256.0) as i16;

        let chunks = len / 48; // 48 bytes = 16 RGB pixels

        unsafe {
            // Broadcast matrix coefficients into vectors
            let m0_vec = [vdupq_n_s16(m00), vdupq_n_s16(m01), vdupq_n_s16(m02)];
            let m1_vec = [vdupq_n_s16(m10), vdupq_n_s16(m11), vdupq_n_s16(m12)];
            let m2_vec = [vdupq_n_s16(m20), vdupq_n_s16(m21), vdupq_n_s16(m22)];

            let mut offset = 0;

            // Process 16 pixels (48 bytes) at a time
            for _ in 0..chunks {
                let src = data.as_ptr().add(offset) as *const u8;

                // vld3: De-interleave 16 RGB pixels -> {R16, G16, B16}
                let rgb = vld3q_u8(src);

                // Widen u8 -> u16 (low 8 pixels)
                let r_lo = vmovl_u8(vget_low_u8(rgb.0)); // uint16x8_t
                let g_lo = vmovl_u8(vget_low_u8(rgb.1));
                let b_lo = vmovl_u8(vget_low_u8(rgb.2));

                // Widen u8 -> u16 (high 8 pixels)
                let r_hi = vmovl_u8(vget_high_u8(rgb.0));
                let g_hi = vmovl_u8(vget_high_u8(rgb.1));
                let b_hi = vmovl_u8(vget_high_u8(rgb.2));

                // R' = m00*R + m01*G + m02*B (low 8 pixels)
                let mut r_acc_lo = vmull_s16(
                    vreinterpret_s16_u16(vget_low_u16(r_lo)),
                    vget_low_s16(m0_vec[0]),
                );
                r_acc_lo = vmlal_s16(
                    r_acc_lo,
                    vreinterpret_s16_u16(vget_low_u16(g_lo)),
                    vget_low_s16(m0_vec[1]),
                );
                r_acc_lo = vmlal_s16(
                    r_acc_lo,
                    vreinterpret_s16_u16(vget_low_u16(b_lo)),
                    vget_low_s16(m0_vec[2]),
                );

                let mut r_acc_lo_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_high_u16(r_lo)),
                    vget_high_s16(m0_vec[0]),
                );
                r_acc_lo_hi = vmlal_s16(
                    r_acc_lo_hi,
                    vreinterpret_s16_u16(vget_high_u16(g_lo)),
                    vget_high_s16(m0_vec[1]),
                );
                r_acc_lo_hi = vmlal_s16(
                    r_acc_lo_hi,
                    vreinterpret_s16_u16(vget_high_u16(b_lo)),
                    vget_high_s16(m0_vec[2]),
                );

                // G' = m10*R + m11*G + m12*B (low 8 pixels)
                let mut g_acc_lo = vmull_s16(
                    vreinterpret_s16_u16(vget_low_u16(r_lo)),
                    vget_low_s16(m1_vec[0]),
                );
                g_acc_lo = vmlal_s16(
                    g_acc_lo,
                    vreinterpret_s16_u16(vget_low_u16(g_lo)),
                    vget_low_s16(m1_vec[1]),
                );
                g_acc_lo = vmlal_s16(
                    g_acc_lo,
                    vreinterpret_s16_u16(vget_low_u16(b_lo)),
                    vget_low_s16(m1_vec[2]),
                );

                let mut g_acc_lo_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_high_u16(r_lo)),
                    vget_high_s16(m1_vec[0]),
                );
                g_acc_lo_hi = vmlal_s16(
                    g_acc_lo_hi,
                    vreinterpret_s16_u16(vget_high_u16(g_lo)),
                    vget_high_s16(m1_vec[1]),
                );
                g_acc_lo_hi = vmlal_s16(
                    g_acc_lo_hi,
                    vreinterpret_s16_u16(vget_high_u16(b_lo)),
                    vget_high_s16(m1_vec[2]),
                );

                // B' = m20*R + m21*G + m22*B (low 8 pixels)
                let mut b_acc_lo = vmull_s16(
                    vreinterpret_s16_u16(vget_low_u16(r_lo)),
                    vget_low_s16(m2_vec[0]),
                );
                b_acc_lo = vmlal_s16(
                    b_acc_lo,
                    vreinterpret_s16_u16(vget_low_u16(g_lo)),
                    vget_low_s16(m2_vec[1]),
                );
                b_acc_lo = vmlal_s16(
                    b_acc_lo,
                    vreinterpret_s16_u16(vget_low_u16(b_lo)),
                    vget_low_s16(m2_vec[2]),
                );

                let mut b_acc_lo_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_high_u16(r_lo)),
                    vget_high_s16(m2_vec[0]),
                );
                b_acc_lo_hi = vmlal_s16(
                    b_acc_lo_hi,
                    vreinterpret_s16_u16(vget_high_u16(g_lo)),
                    vget_high_s16(m2_vec[1]),
                );
                b_acc_lo_hi = vmlal_s16(
                    b_acc_lo_hi,
                    vreinterpret_s16_u16(vget_high_u16(b_lo)),
                    vget_high_s16(m2_vec[2]),
                );

                // R' = m00*R + m01*G + m02*B (high 8 pixels)
                let mut r_acc_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_low_u16(r_hi)),
                    vget_low_s16(m0_vec[0]),
                );
                r_acc_hi = vmlal_s16(
                    r_acc_hi,
                    vreinterpret_s16_u16(vget_low_u16(g_hi)),
                    vget_low_s16(m0_vec[1]),
                );
                r_acc_hi = vmlal_s16(
                    r_acc_hi,
                    vreinterpret_s16_u16(vget_low_u16(b_hi)),
                    vget_low_s16(m0_vec[2]),
                );

                let mut r_acc_hi_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_high_u16(r_hi)),
                    vget_high_s16(m0_vec[0]),
                );
                r_acc_hi_hi = vmlal_s16(
                    r_acc_hi_hi,
                    vreinterpret_s16_u16(vget_high_u16(g_hi)),
                    vget_high_s16(m0_vec[1]),
                );
                r_acc_hi_hi = vmlal_s16(
                    r_acc_hi_hi,
                    vreinterpret_s16_u16(vget_high_u16(b_hi)),
                    vget_high_s16(m0_vec[2]),
                );

                // G' = m10*R + m11*G + m12*B (high 8 pixels)
                let mut g_acc_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_low_u16(r_hi)),
                    vget_low_s16(m1_vec[0]),
                );
                g_acc_hi = vmlal_s16(
                    g_acc_hi,
                    vreinterpret_s16_u16(vget_low_u16(g_hi)),
                    vget_low_s16(m1_vec[1]),
                );
                g_acc_hi = vmlal_s16(
                    g_acc_hi,
                    vreinterpret_s16_u16(vget_low_u16(b_hi)),
                    vget_low_s16(m1_vec[2]),
                );

                let mut g_acc_hi_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_high_u16(r_hi)),
                    vget_high_s16(m1_vec[0]),
                );
                g_acc_hi_hi = vmlal_s16(
                    g_acc_hi_hi,
                    vreinterpret_s16_u16(vget_high_u16(g_hi)),
                    vget_high_s16(m1_vec[1]),
                );
                g_acc_hi_hi = vmlal_s16(
                    g_acc_hi_hi,
                    vreinterpret_s16_u16(vget_high_u16(b_hi)),
                    vget_high_s16(m1_vec[2]),
                );

                // B' = m20*R + m21*G + m22*B (high 8 pixels)
                let mut b_acc_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_low_u16(r_hi)),
                    vget_low_s16(m2_vec[0]),
                );
                b_acc_hi = vmlal_s16(
                    b_acc_hi,
                    vreinterpret_s16_u16(vget_low_u16(g_hi)),
                    vget_low_s16(m2_vec[1]),
                );
                b_acc_hi = vmlal_s16(
                    b_acc_hi,
                    vreinterpret_s16_u16(vget_low_u16(b_hi)),
                    vget_low_s16(m2_vec[2]),
                );

                let mut b_acc_hi_hi = vmull_s16(
                    vreinterpret_s16_u16(vget_high_u16(r_hi)),
                    vget_high_s16(m2_vec[0]),
                );
                b_acc_hi_hi = vmlal_s16(
                    b_acc_hi_hi,
                    vreinterpret_s16_u16(vget_high_u16(g_hi)),
                    vget_high_s16(m2_vec[1]),
                );
                b_acc_hi_hi = vmlal_s16(
                    b_acc_hi_hi,
                    vreinterpret_s16_u16(vget_high_u16(b_hi)),
                    vget_high_s16(m2_vec[2]),
                );

                // Shift right by 8 (divide by 256), saturate, narrow: i32 -> u8
                let r_out_0 = vqshrun_n_s32(r_acc_lo, 8);
                let r_out_1 = vqshrun_n_s32(r_acc_lo_hi, 8);
                let r_out_2 = vqshrun_n_s32(r_acc_hi, 8);
                let r_out_3 = vqshrun_n_s32(r_acc_hi_hi, 8);

                let g_out_0 = vqshrun_n_s32(g_acc_lo, 8);
                let g_out_1 = vqshrun_n_s32(g_acc_lo_hi, 8);
                let g_out_2 = vqshrun_n_s32(g_acc_hi, 8);
                let g_out_3 = vqshrun_n_s32(g_acc_hi_hi, 8);

                let b_out_0 = vqshrun_n_s32(b_acc_lo, 8);
                let b_out_1 = vqshrun_n_s32(b_acc_lo_hi, 8);
                let b_out_2 = vqshrun_n_s32(b_acc_hi, 8);
                let b_out_3 = vqshrun_n_s32(b_acc_hi_hi, 8);

                // Combine pairs: uint16x4_t -> uint16x8_t
                let r_out_lo = vcombine_u16(r_out_0, r_out_1);
                let r_out_hi = vcombine_u16(r_out_2, r_out_3);
                let g_out_lo = vcombine_u16(g_out_0, g_out_1);
                let g_out_hi = vcombine_u16(g_out_2, g_out_3);
                let b_out_lo = vcombine_u16(b_out_0, b_out_1);
                let b_out_hi = vcombine_u16(b_out_2, b_out_3);

                // Narrow: uint16x8_t -> uint8x8_t (SATURATING)
                let r_out_lo_u8 = vqmovn_u16(r_out_lo);
                let r_out_hi_u8 = vqmovn_u16(r_out_hi);
                let g_out_lo_u8 = vqmovn_u16(g_out_lo);
                let g_out_hi_u8 = vqmovn_u16(g_out_hi);
                let b_out_lo_u8 = vqmovn_u16(b_out_lo);
                let b_out_hi_u8 = vqmovn_u16(b_out_hi);

                // Combine to uint8x16_t
                let r_out = vcombine_u8(r_out_lo_u8, r_out_hi_u8);
                let g_out = vcombine_u8(g_out_lo_u8, g_out_hi_u8);
                let b_out = vcombine_u8(b_out_lo_u8, b_out_hi_u8);

                // Interleave RGB and store directly to memory
                let dst = data.as_mut_ptr().add(offset);
                vst3q_u8(dst, uint8x16x3_t(r_out, g_out, b_out));

                offset += 48;
            }

            // Handle remaining pixels (scalar fallback)
            for i in (chunks * 48)..len {
                if i % 3 != 0 {
                    continue;
                }
                let r = data[i] as f32;
                let g = data[i + 1] as f32;
                let b = data[i + 2] as f32;

                let out_r = (matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b)
                    .clamp(0.0, 255.0) as u8;
                let out_g = (matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b)
                    .clamp(0.0, 255.0) as u8;
                let out_b = (matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b)
                    .clamp(0.0, 255.0) as u8;

                data[i] = out_r;
                data[i + 1] = out_g;
                data[i + 2] = out_b;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn apply(image: &mut FusableImage, matrix: &[[f32; 3]; 3]) {
        // Scalar fallback for other architectures
        super::apply_matrix(image, matrix);
    }

    /// Execute multiple matrix operations in a single fused pass
    ///
    /// Composes all matrices into one, then applies once.
    ///
    /// # Arguments
    /// * `image` - RGB image to transform
    /// * `ops` - Slice of matrix operations to apply in order
    ///
    /// # Example
    /// ```ignore
    /// let ops: &[&dyn MatrixOp] = &[&tosepia, &saturation];
    /// MatrixExecutor::execute_fused(image, ops);
    /// ```
    pub fn execute_fused(image: &mut FusableImage, ops: &[&dyn MatrixOp]) {
        if ops.is_empty() {
            return;
        }

        if ops.len() == 1 {
            // Single op - no composition needed
            let matrix = ops[0].get_matrix();
            Self::apply(image, &matrix);
        } else {
            // Multiple ops - compose into single matrix
            let combined = compose_matrices(ops);
            Self::apply(image, &combined);
        }
    }
}
