// Random distributions for sampling

use super::traits::Rng;

/// A Bernoulli random variable
///
/// Samples to true with probability p, false otherwise.
/// Used for transforms that may or may not be applied.
#[derive(Debug, Clone, Copy)]
pub struct Bernoulli {
    /// Probability of returning true
    pub p: f32,
}

impl Bernoulli {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p <= 1.0, "Probability must be in [0, 1]");
        Self { p }
    }

    /// Sample this Bernoulli variable
    pub fn sample(&self, rng: &mut dyn Rng) -> bool {
        rng.random_f32() < self.p
    }
}

/// A uniform random variable in [min, max]
#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    pub min: f32,
    pub max: f32,
}

impl Uniform {
    pub fn new(min: f32, max: f32) -> Self {
        assert!(min <= max, "min must be <= max");
        Self { min, max }
    }

    /// Sample this uniform variable
    pub fn sample(&self, rng: &mut dyn Rng) -> f32 {
        self.min + (self.max - self.min) * rng.random_f32()
    }
}

/// An integer uniform random variable in [min, max]
#[derive(Debug, Clone, Copy)]
pub struct UniformInt {
    pub min: i32,
    pub max: i32,
}

impl UniformInt {
    pub fn new(min: i32, max: i32) -> Self {
        assert!(min <= max, "min must be <= max");
        Self { min, max }
    }

    /// Sample this uniform variable
    pub fn sample(&self, rng: &mut dyn Rng) -> i32 {
        let range = self.max - self.min + 1;
        self.min + rng.random_i32(range)
    }
}

// =============================================================================
// Unified Distribution Enum (New API)
// =============================================================================

/// A unified distribution over values
///
/// This enum extends the existing distribution types with:
/// - `Constant` - for deterministic values (no sampling)
/// - `Normal` - Gaussian distribution
///
/// This allows transforms to accept either fixed values or distributions,
/// enabling `Compose` to support both deterministic and random parameters.
///
/// # Example
///
/// ```ignore
/// // Constant value (deterministic)
/// let delta = Dist::Constant(50.0);
///
/// // Uniform sampling
/// let delta = Dist::Uniform { min: -30.0, max: 30.0 };
///
/// // Gaussian sampling
/// let delta = Dist::Normal { mu: 0.0, sigma: 30.0 };
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Dist {
    /// Constant value (no sampling, deterministic)
    Constant(f32),
    /// Uniform distribution over [min, max]
    Uniform { min: f32, max: f32 },
    /// Integer uniform distribution over [min, max]
    UniformInt { min: i32, max: i32 },
    /// Bernoulli distribution (probability of true/success)
    Bernoulli { p: f32 },
    /// Normal (Gaussian) distribution
    Normal { mu: f32, sigma: f32 },
}

impl Dist {
    /// Sample this distribution as an f32
    ///
    /// # Panics
    /// - If called on `UniformInt` (use `sample_i32` instead)
    pub fn sample_f32(&self, rng: &mut dyn Rng) -> f32 {
        match self {
            Dist::Constant(v) => *v,
            Dist::Uniform { min, max } => {
                Uniform::new(*min, *max).sample(rng)
            }
            Dist::Bernoulli { p } => {
                if Bernoulli::new(*p).sample(rng) { 1.0 } else { 0.0 }
            }
            Dist::Normal { mu, sigma } => {
                // Box-Muller transform for Gaussian sampling
                let u1: f32 = rng.random_f32();
                let u2: f32 = rng.random_f32();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                mu + sigma * z
            }
            Dist::UniformInt { .. } => {
                panic!("Cannot sample UniformInt as f32; use sample_i32() instead");
            }
        }
    }

    /// Sample this distribution as an i32
    ///
    /// # Panics
    /// - If called on non-integer distributions
    pub fn sample_i32(&self, rng: &mut dyn Rng) -> i32 {
        match self {
            Dist::Constant(v) => *v as i32,
            Dist::UniformInt { min, max } => {
                UniformInt::new(*min, *max).sample(rng)
            }
            _ => {
                panic!("Cannot sample this distribution as i32");
            }
        }
    }

    /// Sample as a boolean (for Bernoulli distributions)
    ///
    /// # Panics
    /// - If called on non-Bernoulli distributions
    pub fn sample_bool(&self, rng: &mut dyn Rng) -> bool {
        match self {
            Dist::Bernoulli { p } => Bernoulli::new(*p).sample(rng),
            Dist::Constant(v) => *v > 0.0,
            _ => panic!("Cannot sample this distribution as bool"),
        }
    }

    /// Returns true if this is a Constant distribution
    ///
    /// This is an optimization hint - constant distributions don't need
    /// any RNG calls and can be simplified at compile time.
    pub fn is_constant(&self) -> bool {
        matches!(self, Dist::Constant(_))
    }

    // =========================================================================
    // Convenience constructors
    // =========================================================================

    /// Create a Constant distribution
    pub fn constant(value: f32) -> Self {
        Dist::Constant(value)
    }

    /// Create a Uniform distribution
    pub fn uniform(min: f32, max: f32) -> Self {
        Dist::Uniform { min, max }
    }

    /// Create a UniformInt distribution
    pub fn uniform_int(min: i32, max: i32) -> Self {
        Dist::UniformInt { min, max }
    }

    /// Create a Bernoulli distribution
    pub fn bernoulli(p: f32) -> Self {
        Dist::Bernoulli { p }
    }

    /// Create a Normal (Gaussian) distribution
    pub fn normal(mu: f32, sigma: f32) -> Self {
        Dist::Normal { mu, sigma }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Mock RNG that returns predictable values
    struct MockRng {
        f32_value: f32,
        i32_value: i32,
    }

    impl MockRng {
        fn new(f32_value: f32, i32_value: i32) -> Self {
            Self { f32_value, i32_value }
        }
    }

    impl Rng for MockRng {
        fn random_f32(&mut self) -> f32 {
            self.f32_value
        }

        fn random_i32(&mut self, _upper: i32) -> i32 {
            self.i32_value
        }
    }

    #[test]
    fn test_dist_constant() {
        let dist = Dist::constant(42.0);
        let mut rng = MockRng::new(0.5, 0);

        assert_eq!(dist.sample_f32(&mut rng), 42.0);
        assert!(dist.is_constant());
    }

    #[test]
    fn test_dist_uniform() {
        let dist = Dist::uniform(10.0, 20.0);
        let mut rng = MockRng::new(0.5, 0);  // 50% through range

        let result = dist.sample_f32(&mut rng);
        assert_eq!(result, 15.0);  // 10 + (20-10) * 0.5
        assert!(!dist.is_constant());
    }

    #[test]
    fn test_dist_bernoulli_true() {
        let dist = Dist::bernoulli(0.9);
        let mut rng = MockRng::new(0.5, 0);  // 50% < 90%

        assert!(dist.sample_bool(&mut rng));
        assert_eq!(dist.sample_f32(&mut rng), 1.0);
    }

    #[test]
    fn test_dist_bernoulli_false() {
        let dist = Dist::bernoulli(0.1);
        let mut rng = MockRng::new(0.5, 0);  // 50% > 10%

        assert!(!dist.sample_bool(&mut rng));
        assert_eq!(dist.sample_f32(&mut rng), 0.0);
    }

    #[test]
    fn test_dist_uniform_int() {
        let dist = Dist::uniform_int(10, 20);
        let mut rng = MockRng::new(0.0, 5);  // Index 5 in range

        let result = dist.sample_i32(&mut rng);
        assert_eq!(result, 15);  // 10 + 5
    }

    #[test]
    fn test_dist_constant_as_i32() {
        let dist = Dist::constant(42.0);
        let mut rng = MockRng::new(0.5, 0);

        assert_eq!(dist.sample_i32(&mut rng), 42);
    }

    #[test]
    fn test_dist_normal() {
        let dist = Dist::normal(0.0, 1.0);
        let mut rng = MockRng::new(0.5, 0);

        // Just check it doesn't panic and returns something
        let result = dist.sample_f32(&mut rng);
        // With u1=0.5, Box-Muller gives us a valid z value
        // mu + sigma * z = 0 + 1 * z = z
        // Just check it's a finite number
        assert!(result.is_finite());
    }
}
