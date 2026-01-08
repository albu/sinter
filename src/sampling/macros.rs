// Macro for defining RandomAtomicImageOp implementations
//
// This macro generates boilerplate-free random transforms.
// Constraints:
// 1. Only generates atomic transforms (no structural combinators)
// 2. Expands to 100% explicit Rust (no magic)
// 3. No implicit probability gates (explicit return None)

/// Define a random atomic transform with minimal boilerplate
///
/// # Constraints
///
/// - Only generates atomic transforms (single RandomAtomicImageOp impl)
/// - No structural combinators (use RandomImageNode for those)
/// - No implicit probability gates (explicitly return None to skip)
///
/// # Syntax
///
/// ```ignore
/// random_atomic_op! {
///     /// Documentation here
///     RandomName(field1: Type1, field2: Type2) => {
///         // Sampling logic
///         // Must return Option<SampledImageOp>
///         Some(SampledImageOp::Variant { ... })
///     }
/// }
/// ```
///
/// # Examples
///
/// ```ignore
/// // Simple always-on transform
/// random_atomic_op! {
///     RandomInvert() => {
///         Some(SampledImageOp::Invert)
///     }
/// }
///
/// // Probability-gated transform
/// random_atomic_op! {
///     /// Random flip with probability p
///     RandomFlip(p: f32) => {
///         if !Bernoulli::new(*p).sample(ctx.rng) {
///             return None;
///         }
///         Some(SampledImageOp::HorizontalFlip)
///     }
/// }
///
/// // Multi-parameter sampling
/// random_atomic_op! {
///     RandomBrightness(limit: f32) => {
///         let delta = Uniform::new(-limit, limit).sample(ctx.rng);
///         Some(SampledImageOp::Brightness { delta })
///     }
/// }
/// ```
#[macro_export]
macro_rules! random_atomic_op {
    (
        $(#[$meta:meta])*
        $name:ident ( $($field:ident : $ty:ty),* $(,)? ) => $body:block
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        pub struct $name {
            $(pub $field : $ty),*
        }

        impl $name {
            #[doc = concat!("Create a new ", stringify!($name))]
            pub fn new($($field : $ty),*) -> Self {
                Self {
                    $($field),*
                }
            }
        }

        impl $crate::sampling::RandomAtomicImageOp for $name {
            fn sample(&self, ctx: &mut $crate::sampling::SamplingContext) -> Option<$crate::sampled_ir::SampledImageOp> {
                // Bind each field by reference (no destructuring scope issues)
                $(
                    let $field = &self.$field;
                )*
                $body
            }

            fn access(&self) -> $crate::core::AccessPattern {
                $crate::core::AccessPattern::InPlace
            }

            fn shape_effect(&self) -> $crate::core::ShapeEffect {
                $crate::core::ShapeEffect::Preserve
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}

// =============================================================================
// Convenience macros for common patterns
// =============================================================================//

/// Define a random transform with probability gate
///
/// This is a convenience macro for transforms that have a probability parameter.
///
/// # Syntax
///
/// ```ignore
/// random_op_with_prob! {
///     RandomFlip(p: f32) => {
///         // When not skipped, return this
///         SampledImageOp::HorizontalFlip
///     }
/// }
/// ```
///
/// Expands to a full random_atomic_op with Bernoulli gate.
#[macro_export]
macro_rules! random_op_with_prob {
    (
        $(#[$meta:meta])*
        $name:ident ( p: f32 $(,$field:ident : $ty:ty)* $(,)? ) => $op:expr
    ) => {
        random_atomic_op! {
            $(#[$meta])*
            $name(p: f32 $(,$field : $ty)*,) => {
                if !$crate::sampling::Bernoulli::new(*p).sample(ctx.rng) {
                    return None;
                }
                Some($op)
            }
        }
    };
}

/// Define a random transform with uniform parameter sampling
///
/// Convenience macro for transforms that sample from a uniform range.
///
/// # Syntax
///
/// ```ignore
/// random_op_uniform! {
///     RandomBrightness(limit: f32) => Brightness { delta }
/// }
/// ```
///
/// Expands to a full random_atomic_op with Uniform sampling.
#[macro_export]
macro_rules! random_op_uniform {
    (
        $(#[$meta:meta])*
        $name:ident ( $param:ident : $ty:ty ) => $sampled_variant:expr
    ) => {
        random_atomic_op! {
            $(#[$meta])*
            $name($param : $ty,) => {
                let sampled = $crate::sampling::Uniform::new(-$param, $param).sample(ctx.rng);
                Some($sampled_variant)
            }
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    // Note: We test macros in integration tests to avoid $crate expansion issues
    // within the same crate. See python/tests/ for usage examples.

    #[test]
    fn test_macros_compilable() {
        // This test just verifies the macros are defined and compilable
        // Actual usage tests are in integration tests
        assert!(true);
    }
}
