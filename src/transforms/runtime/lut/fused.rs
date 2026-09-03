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
    /// The composed LUT table (channel 0 or single channel)
    pub lut: [u8; 256],
    /// Optional 3-channel LUT tables [R, G, B] if any transform was channel-specific
    pub luts_3c: Option<[[u8; 256]; 3]>,
}

impl FusedLut {
    /// Create a new FusedLut from a pre-composed LUT
    pub fn new(lut: [u8; 256]) -> Self {
        Self { lut, luts_3c: None }
    }

    /// Create a new 3-channel FusedLut
    pub fn new_3c(luts: [[u8; 256]; 3]) -> Self {
        Self {
            lut: luts[0],
            luts_3c: Some(luts),
        }
    }

    /// Create a FusedLut directly from a slice of SampledImageOp (zero heap allocations!)
    pub fn from_sampled_ops(ops: &[crate::sampled_ir::SampledImageOp]) -> Self {
        let any_3c = ops.iter().any(|op| matches!(op, crate::sampled_ir::SampledImageOp::RGBShift { .. }));
        if any_3c {
            let mut luts = [[0u8; 256]; 3];
            for c in 0..3 {
                for i in 0..256 {
                    luts[c][i] = i as u8;
                }
            }
            for op in ops {
                if let Some(op_3c) = op.build_lut_3c() {
                    for c in 0..3 {
                        for i in 0..256 {
                            luts[c][i] = op_3c[c][luts[c][i] as usize];
                        }
                    }
                } else if let Some(op_1c) = op.build_lut() {
                    for c in 0..3 {
                        for i in 0..256 {
                            luts[c][i] = op_1c[luts[c][i] as usize];
                        }
                    }
                }
            }

            let mut lut_1c = [0u8; 256];
            for i in 0..256 { lut_1c[i] = i as u8; }
            for op in ops {
                if let Some(op_1c) = op.build_lut() {
                    for i in 0..256 {
                        lut_1c[i] = op_1c[lut_1c[i] as usize];
                    }
                }
            }

            if luts[0] == luts[1] && luts[1] == luts[2] {
                Self {
                    lut: luts[0],
                    luts_3c: None,
                }
            } else {
                Self {
                    lut: lut_1c,
                    luts_3c: Some(luts),
                }
            }
        } else {
            let mut lut = [0u8; 256];
            for i in 0..256 { lut[i] = i as u8; }
            for op in ops {
                if let Some(op_1c) = op.build_lut() {
                    for i in 0..256 {
                        lut[i] = op_1c[lut[i] as usize];
                    }
                }
            }
            Self {
                lut,
                luts_3c: None,
            }
        }
    }

    /// Create a FusedLut from multiple LUT operations
    pub fn from_ops(ops: &[Box<dyn LutOp>]) -> Self {
        let any_3c = ops.iter().any(|op| op.is_3c());
        if any_3c {
            let luts = FusedLutExecutor::compose_3c_luts(ops);
            if luts[0] == luts[1] && luts[1] == luts[2] {
                Self {
                    lut: luts[0],
                    luts_3c: None,
                }
            } else {
                Self {
                    lut: FusedLutExecutor::compose_luts(ops),
                    luts_3c: Some(luts),
                }
            }
        } else {
            Self {
                lut: FusedLutExecutor::compose_luts(ops),
                luts_3c: None,
            }
        }
    }

    /// Check if this is effectively the identity transform
    pub fn is_identity(&self) -> bool {
        if let Some(ref luts) = self.luts_3c {
            for c in 0..3 {
                for i in 0..256 {
                    if luts[c][i] != i as u8 {
                        return false;
                    }
                }
            }
            true
        } else {
            for i in 0..256 {
                if self.lut[i] != i as u8 {
                    return false;
                }
            }
            true
        }
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
        if let Some(ref luts) = self.luts_3c {
            if image.channels == 3 {
                LutExecutor::apply_rgb_luts(image, luts);
                return None;
            }
        }
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

    fn build_lut_3c(&self) -> [[u8; 256]; 3] {
        if let Some(ref luts) = self.luts_3c {
            *luts
        } else {
            [self.lut, self.lut, self.lut]
        }
    }

    fn is_3c(&self) -> bool {
        self.luts_3c.is_some()
    }
}

