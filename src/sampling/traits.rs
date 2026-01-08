// Traits for probabilistic transforms

/// Simple RNG trait for sampling
///
/// We use a custom trait instead of rand::Rng to avoid dependencies.
/// Implementations can wrap rand::Rng or provide a simple implementation.
pub trait Rng {
    /// Generate a random f32 in [0, 1)
    fn random_f32(&mut self) -> f32;

    /// Generate a random i32 in [0, upper)
    fn random_i32(&mut self, upper: i32) -> i32;
}

/// Thread-local RNG wrapper
///
/// Wraps rand::ThreadRng for use with our Rng trait.
#[cfg(feature = "python")]
pub struct ThreadRng {
    inner: rand::rngs::ThreadRng,
}

#[cfg(feature = "python")]
impl ThreadRng {
    pub fn new() -> Self {
        Self { inner: rand::thread_rng() }
    }
}

#[cfg(feature = "python")]
impl Rng for ThreadRng {
    fn random_f32(&mut self) -> f32 {
        use rand::Rng;
        self.inner.gen()
    }

    fn random_i32(&mut self, upper: i32) -> i32 {
        use rand::Rng;
        self.inner.gen_range(0..upper)
    }
}
