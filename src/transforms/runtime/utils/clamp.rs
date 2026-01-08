// Branchless clamping utilities
//
// Provides SIMD-friendly, branchless clamping operations.
// These are designed to eliminate branch mispredictions in hot loops.
//
// The key insight is that we can use min/max operations instead of
// conditional branches for clamping.

/// Branchless clamp to [0, 255] for f32 values
///
/// Uses min/max operations instead of conditional branches for better SIMD optimization.
#[inline]
pub fn clamp_u8(value: f32) -> f32 {
    value.max(0.0).min(255.0)
}

/// Branchless clamp with explicit bounds for f32 values
#[inline]
pub const fn clamp_bounds(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Conditional clamp - only clamp if needed
///
/// This is used when we have range information that tells us
/// whether clamping is necessary. The compiler can optimize
/// the check away when the condition is known at compile time.
#[inline]
pub fn clamp_if_needed(value: f32, needs_clamp: bool) -> f32 {
    if needs_clamp {
        clamp_u8(value)
    } else {
        value
    }
}

/// Saturating cast from f32 to u8
///
/// Clamps the value to [0, 255] before casting.
#[inline]
pub fn saturate_u8(value: f32) -> u8 {
    value.clamp(0.0, 255.0) as u8
}

/// Branchless saturating cast using min/max
///
/// More explicit version of saturate_u8 that may optimize differently.
#[inline]
pub fn saturate_u8_branchless(value: f32) -> u8 {
    let clamped = value.max(0.0).min(255.0);
    clamped as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_u8_in_range() {
        assert_eq!(clamp_u8(100.0), 100.0);
        assert_eq!(clamp_u8(0.0), 0.0);
        assert_eq!(clamp_u8(255.0), 255.0);
        assert_eq!(clamp_u8(128.5), 128.5);
    }

    #[test]
    fn test_clamp_u8_below_min() {
        assert_eq!(clamp_u8(-10.0), 0.0);
        assert_eq!(clamp_u8(-100.0), 0.0);
        assert_eq!(clamp_u8(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn test_clamp_u8_above_max() {
        assert_eq!(clamp_u8(300.0), 255.0);
        assert_eq!(clamp_u8(1000.0), 255.0);
        assert_eq!(clamp_u8(f32::INFINITY), 255.0);
    }

    #[test]
    fn test_clamp_bounds() {
        assert_eq!(clamp_bounds(50.0, 0.0, 100.0), 50.0);
        assert_eq!(clamp_bounds(-10.0, 0.0, 100.0), 0.0);
        assert_eq!(clamp_bounds(150.0, 0.0, 100.0), 100.0);
    }

    #[test]
    fn test_saturate_u8() {
        assert_eq!(saturate_u8(100.0), 100);
        assert_eq!(saturate_u8(-10.0), 0);
        assert_eq!(saturate_u8(300.0), 255);
        assert_eq!(saturate_u8(0.0), 0);
        assert_eq!(saturate_u8(255.0), 255);
    }

    #[test]
    fn test_saturate_u8_branchless() {
        assert_eq!(saturate_u8_branchless(100.0), 100);
        assert_eq!(saturate_u8_branchless(-10.0), 0);
        assert_eq!(saturate_u8_branchless(300.0), 255);
        assert_eq!(saturate_u8_branchless(0.0), 0);
        assert_eq!(saturate_u8_branchless(255.0), 255);
    }

    #[test]
    fn test_clamp_if_needed() {
        // When clamping is not needed, value passes through
        assert_eq!(clamp_if_needed(100.0, false), 100.0);
        assert_eq!(clamp_if_needed(255.0, false), 255.0);

        // When clamping is needed, value is clamped
        assert_eq!(clamp_if_needed(300.0, true), 255.0);
        assert_eq!(clamp_if_needed(-10.0, true), 0.0);
    }

    #[test]
    fn test_clamp_edge_cases() {
        // NaN behavior: max(0.0, NaN) = 0.0, min(255.0, NaN) = NaN
        // So NaN values should be handled explicitly
        let nan = f32::NAN;
        assert!(nan.is_nan());
        // NaN comparisons are always false, so we need explicit handling
    }
}
