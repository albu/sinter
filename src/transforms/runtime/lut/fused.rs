// FusedLut - a transform representing composed LUT operations
//
// This represents a single LUT that is the composition of multiple LUT transforms.
// It's created by the optimizer when it detects consecutive LUT transforms.

use crate::core::{AccessPattern, BarrierImage, Executable, FusableImage, ShapeEffect, Transform};
use std::any::Any;

use super::{FusedLutExecutor, LutExecutor, LutOp};

/// A fused LUT transform
///
/// This represents a single LUT that is the composition of multiple LUT transforms.
/// It's created by the optimizer when it detects consecutive LUT transforms.
#[derive(Debug, Clone)]
pub struct FusedLut {
    /// The composed LUT table
    pub lut: [u8; 256],
}

impl FusedLut {
    /// Create a new FusedLut from a pre-composed LUT
    pub fn new(lut: [u8; 256]) -> Self {
        Self { lut }
    }

    /// Create a FusedLut from multiple LUT operations
    pub fn from_ops(ops: &[Box<dyn LutOp>]) -> Self {
        Self {
            lut: FusedLutExecutor::compose_luts(ops),
        }
    }

    /// Check if this is effectively the identity transform
    pub fn is_identity(&self) -> bool {
        for i in 0..256 {
            if self.lut[i] != i as u8 {
                return false;
            }
        }
        true
    }
}

impl Transform for FusedLut {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for FusedLut {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        LutExecutor::apply(image, &self.lut);
        None
    }
}

impl LutOp for FusedLut {
    fn build_lut(&self) -> [u8; 256] {
        self.lut
    }

    fn get_lut(&self) -> [u8; 256] {
        self.lut
    }
}
