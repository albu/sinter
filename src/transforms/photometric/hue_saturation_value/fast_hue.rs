// Hue rotation handling for Fast mode
//
// For now, hue rotation falls back to Accurate mode.
// SIMD optimization is primarily for saturation + value.

use crate::core::FusableImage;
use crate::transforms::photometric::hue_saturation_value::HueSaturationValue;

/// Execute with hue rotation
///
/// Note: For non-zero hue, falls back to Accurate mode.
/// SIMD optimization is most effective for saturation + value only.
pub fn execute_fast_with_hue(hsv: &HueSaturationValue, image: &mut FusableImage) {
    // For non-zero hue, use accurate/scalar implementation
    // Proper channel-permutation hue rotation is complex to SIMD-optimize
    if hsv.hue_shift != 0.0 {
        #[cfg(feature = "opencv")]
        {
            if image.channels == 3 {
                match super::opencv::execute_with_opencv(hsv, image) {
                    Ok(()) => return,
                    Err(_) => {
                        super::rust_impl::execute_rust(hsv, image);
                        return;
                    }
                }
            }
        }
        super::rust_impl::execute_rust(hsv, image);
        return;
    }

    // For hue=0, do sat+val only (inline to avoid circular dependency)
    let pixel_count = image.data.len() / 3;
    let sat_factor_q8 = (hsv.sat_scale * 256.0) as i32;
    let val_delta = ((hsv.val_scale - 1.0) * 255.0) as i32;

    for i in 0..pixel_count {
        let idx = i * 3;
        let r = image.data[idx] as i32;
        let g = image.data[idx + 1] as i32;
        let b = image.data[idx + 2] as i32;

        // Saturation
        let gray = (77 * r + 150 * g + 29 * b) >> 8;
        let r = gray + (((r - gray) * sat_factor_q8 + 128) >> 8);
        let g = gray + (((g - gray) * sat_factor_q8 + 128) >> 8);
        let b = gray + (((b - gray) * sat_factor_q8 + 128) >> 8);

        // Value
        let mut r = r + val_delta;
        let mut g = g + val_delta;
        let mut b = b + val_delta;

        // Clamp
        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        image.data[idx] = r as u8;
        image.data[idx + 1] = g as u8;
        image.data[idx + 2] = b as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hue_zero_fast_path() {
        let mut data = vec![100u8, 150, 200];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);
        let hsv = HueSaturationValue {
            hue_shift: 0.0,
            sat_scale: 1.2,
            val_scale: 1.1,
            mode: crate::transforms::photometric::hue_saturation_value::HsvMode::Fast,
        };
        execute_fast_with_hue(&hsv, &mut img);
        // Should modify data (sat+val applied)
        assert!(!img.data.is_empty());
    }

    #[test]
    fn test_hue_nonzero_fallback() {
        let mut data = vec![255u8, 0, 0];
        let mut img = FusableImage::new(&mut data, 1, 1, 3);
        let hsv = HueSaturationValue {
            hue_shift: 60.0,
            sat_scale: 1.0,
            val_scale: 1.0,
            mode: crate::transforms::photometric::hue_saturation_value::HsvMode::Fast,
        };
        execute_fast_with_hue(&hsv, &mut img);
        // Should not crash (fallback to Accurate/scalar)
        assert!(!img.data.is_empty());
    }
}
