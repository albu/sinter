// Value range tracking for branch elimination and lazy clamping
//
// Tracks the possible output range of operations to eliminate unnecessary
// clamping operations. This enables:
// 1. Lazy clamping: Only clamp when necessary (before u8 conversion)
// 2. Range propagation: Track value ranges through the pipeline
// 3. Branchless execution: Avoid per-pixel conditionals

/// Represents the possible range of pixel values
///
/// Used for range analysis to determine if clamping is needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValueRange {
    pub min: f32,
    pub max: f32,
}

impl ValueRange {
    /// Create a new range with explicit bounds
    pub const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// The range of u8 pixel values [0, 255]
    pub const U8: Self = Self { min: 0.0, max: 255.0 };

    /// The range of normalized values [0, 1]
    pub const NORMALIZED: Self = Self { min: 0.0, max: 1.0 };

    /// Unknown range (conservative: must clamp)
    pub const UNKNOWN: Self = Self { min: f32::NEG_INFINITY, max: f32::INFINITY };

    /// Check if this range is guaranteed to be within u8 bounds
    pub fn is_u8_safe(&self) -> bool {
        self.min >= 0.0 && self.max <= 255.0
    }

    /// Check if clamping is needed for this range
    pub fn needs_clamp(&self) -> bool {
        !self.is_u8_safe()
    }

    /// Union of two ranges
    pub fn union(&self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Intersection of two ranges
    pub fn intersection(&self, other: Self) -> Self {
        Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    /// Apply an affine transformation y = a*x + b to the range
    pub fn apply_affine(&self, a: f32, b: f32) -> Self {
        // Compute the transformed endpoints
        let (p1, p2) = (a * self.min + b, a * self.max + b);
        // The range is [min(p1,p2), max(p1,p2)] regardless of sign of a
        Self { min: p1.min(p2), max: p1.max(p2) }
    }

    /// Apply a power transformation y = x^gamma to the range
    pub fn apply_power(&self, gamma: f32) -> Self {
        let min = self.min.powf(gamma);
        let max = self.max.powf(gamma);
        Self { min: min.min(max), max: min.max(max) }
    }

    /// Apply LUT transformation (maps u8 to u8, so result is always u8-safe)
    pub fn apply_lut(&self) -> Self {
        Self::U8
    }
}

/// Trait for operations that can compute their output range
pub trait RangedOp {
    /// Get the output range for a given input range
    fn output_range(&self, input: ValueRange) -> ValueRange;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_range_u8_safe() {
        assert!(ValueRange::U8.is_u8_safe());
        assert!(ValueRange::new(0.0, 100.0).is_u8_safe());
        assert!(ValueRange::new(50.0, 255.0).is_u8_safe());
        assert!(!ValueRange::new(-10.0, 255.0).is_u8_safe());
        assert!(!ValueRange::new(0.0, 300.0).is_u8_safe());
        assert!(!ValueRange::UNKNOWN.is_u8_safe());
    }

    #[test]
    fn test_value_range_needs_clamp() {
        assert!(!ValueRange::U8.needs_clamp());
        assert!(!ValueRange::new(0.0, 100.0).needs_clamp());
        assert!(ValueRange::new(-10.0, 255.0).needs_clamp());
        assert!(ValueRange::new(0.0, 300.0).needs_clamp());
        assert!(ValueRange::UNKNOWN.needs_clamp());
    }

    #[test]
    fn test_value_range_union() {
        let r1 = ValueRange::new(0.0, 100.0);
        let r2 = ValueRange::new(50.0, 200.0);
        let union = r1.union(r2);
        assert_eq!(union.min, 0.0);
        assert_eq!(union.max, 200.0);
    }

    #[test]
    fn test_value_range_intersection() {
        let r1 = ValueRange::new(0.0, 100.0);
        let r2 = ValueRange::new(50.0, 200.0);
        let intersection = r1.intersection(r2);
        assert_eq!(intersection.min, 50.0);
        assert_eq!(intersection.max, 100.0);
    }

    #[test]
    fn test_value_range_apply_affine_positive() {
        let range = ValueRange::U8;
        // y = 2*x + 10: [0, 255] -> [10, 520]
        let result = range.apply_affine(2.0, 10.0);
        assert_eq!(result.min, 10.0);
        assert_eq!(result.max, 520.0);
        assert!(result.needs_clamp());
    }

    #[test]
    fn test_value_range_apply_affine_negative() {
        let range = ValueRange::U8;
        // y = -x + 255: [0, 255] -> [0, 255] (inverted)
        let result = range.apply_affine(-1.0, 255.0);
        assert_eq!(result.min, 0.0);
        assert_eq!(result.max, 255.0);
        assert!(!result.needs_clamp());
    }

    #[test]
    fn test_value_range_apply_power() {
        let range = ValueRange::NORMALIZED;
        // Gamma 2.2: [0, 1] -> [0, 1]
        let result = range.apply_power(2.2);
        assert_eq!(result.min, 0.0);
        assert_eq!(result.max, 1.0);
    }

    #[test]
    fn test_value_range_apply_lut() {
        let range = ValueRange::UNKNOWN;
        // LUT always produces u8 output
        let result = range.apply_lut();
        assert!(!result.needs_clamp());
    }
}
