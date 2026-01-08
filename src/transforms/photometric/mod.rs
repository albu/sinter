// Photometric transforms (per-pixel operations that can be fused)
//
// These transforms modify pixel values without changing image geometry,
// making them ideal for fusion into a single-pass execution.

pub mod brightness;
pub mod color_jitter;
pub mod contrast;
pub mod normalize;
pub mod rgb_shift;
pub mod hue_saturation_value;
pub mod gamma;
pub mod gauss_noise;
pub mod multiplicative_noise;
pub mod salt_pepper;
pub mod to_gray;
pub mod to_rgb;
pub mod to_sepia;
pub mod color_temperature;
pub mod channel_mix;
pub mod color_balance;
pub mod channel_shuffle;
pub mod color_tint;
pub mod invert;
pub mod posterize;
pub mod solarize;
pub mod coarse_dropout;
pub mod grid_dropout;
mod histogram;

// Re-export for convenience
pub use brightness::Brightness;
pub use color_jitter::ColorJitter;
pub use contrast::Contrast;
pub use normalize::Normalize;
pub use rgb_shift::RGBShift;
pub use hue_saturation_value::HueSaturationValue;
pub use gamma::Gamma;
pub use gauss_noise::GaussNoise;
pub use multiplicative_noise::{MultiplicativeNoise, NoiseGranularity};
pub use salt_pepper::SaltAndPepper;
pub use to_gray::ToGray;
pub use to_rgb::ToRGB;
pub use to_sepia::ToSepia;
pub use color_temperature::ColorTemperature;
pub use channel_mix::ChannelMix;
pub use color_balance::ColorBalance;
pub use channel_shuffle::{ChannelShuffle, ChannelOrder};
pub use color_tint::ColorTint;
pub use invert::Invert;
pub use posterize::Posterize;
pub use solarize::Solarize;
pub use coarse_dropout::CoarseDropout;
pub use grid_dropout::GridDropout;
pub use histogram::{Equalize, AutoContrast};
