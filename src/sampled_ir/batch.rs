// SampledBatchOp enum and related types
//
// Deterministic batch-level transforms with all parameters sampled.

use serde::{Deserialize, Serialize};

/// Deterministic batch-level transform (sampled, no randomness)
///
/// Batch transforms operate on (Images, Labels) → (Images, Labels).
/// They are applied AFTER the per-image pipeline.
///
/// All randomness is resolved during sampling:
/// - λ values for MixUp
/// - Permutations and boxes for CutMix
/// - Grid assignments for Mosaic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampledBatchOp {
    /// MixUp: Linear blending of image pairs
    ///
    /// For each sample i:
    /// - image[i] = λ[i] * image[i] + (1 - λ[i]) * image[perm[i]]
    /// - label[i] = λ[i] * label[i] + (1 - λ[i]) * label[perm[i]]
    MixUp {
        /// Mixing coefficient for each sample
        lambda: Vec<f32>,
        /// Permutation: which sample to mix with each index
        perm: Vec<usize>,
    },

    /// CutMix: Replace rectangular regions between images
    ///
    /// For each sample i:
    /// - image[i][box[i]] = image[perm[i]][box[i]]
    /// - label[i] = λ[i] * label[i] + (1 - λ[i]) * label[perm[i]]
    CutMix {
        /// Permutation: which sample to cut from
        perm: Vec<usize>,
        /// Bounding boxes for each sample
        boxes: Vec<Rect>,
        /// Mixing coefficients (based on box area)
        lambda: Vec<f32>,
    },

    /// Mosaic: Combine 4 images into 2x2 grid
    ///
    /// Reduces batch size by 4x, concatenates labels
    Mosaic {
        /// How to arrange each group of 4 images
        layouts: Vec<MosaicLayout>,
    },

    /// Sequence of batch transforms
    Sequence { ops: Vec<SampledBatchOp> },
}

/// Rectangle (for CutMix)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Get the area of this rectangle
    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Calculate the mixing coefficient (box area / total area)
    pub fn lambda(&self, total_width: u32, total_height: u32) -> f32 {
        let box_area = self.area() as f32;
        let total_area = (total_width * total_height) as f32;
        1.0 - (box_area / total_area)
    }
}

/// Mosaic layout (how 4 images are arranged)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MosaicLayout {
    /// Which quadrant each image goes to
    /// (batch_index, quadrant_id) where quadrant_id is 0=TL, 1=TR, 2=BL, 3=BR
    pub positions: [(usize, u8); 4],
    /// How to split the canvas (center_x, center_y)
    pub split: (u32, u32),
}

impl MosaicLayout {
    /// Create a new mosaic layout
    pub fn new(positions: [(usize, u8); 4], split: (u32, u32)) -> Self {
        Self { positions, split }
    }
}

/// A sampled batch-level transform program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledBatchProgram {
    /// IR version
    pub version: u32,
    /// Ordered sequence of batch operations
    pub ops: Vec<SampledBatchOp>,
}

impl SampledBatchProgram {
    pub fn new() -> Self {
        Self {
            version: super::program::IR_VERSION,
            ops: Vec::new(),
        }
    }

    pub fn push(&mut self, op: SampledBatchOp) {
        self.ops.push(op);
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, super::program::SerializationError> {
        bincode::serialize(self)
            .map_err(|e| super::program::SerializationError::BincodeError(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, super::program::SerializationError> {
        let program: SampledBatchProgram = bincode::deserialize(bytes)
            .map_err(|e| super::program::SerializationError::BincodeError(e.to_string()))?;
        if program.version != super::program::IR_VERSION {
            return Err(super::program::SerializationError::InvalidVersion {
                found: program.version,
                expected: super::program::IR_VERSION,
            });
        }
        Ok(program)
    }

    /// Convert to JSON (for inspection)
    pub fn to_json(&self) -> Result<String, super::program::SerializationError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| super::program::SerializationError::JsonError(e.to_string()))
    }
}

impl Default for SampledBatchProgram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_creation() {
        let rect = Rect::new(10, 20, 100, 200);
        assert_eq!(rect.x, 10);
        assert_eq!(rect.y, 20);
        assert_eq!(rect.width, 100);
        assert_eq!(rect.height, 200);
    }

    #[test]
    fn test_rect_area() {
        let rect = Rect::new(0, 0, 100, 200);
        assert_eq!(rect.area(), 20000);
    }

    #[test]
    fn test_rect_lambda() {
        let rect = Rect::new(0, 0, 100, 100);
        let lambda = rect.lambda(200, 200);
        assert!((lambda - 0.75).abs() < 0.01); // 1 - (100*100)/(200*200) = 0.75
    }

    #[test]
    fn test_batch_program_serialization() {
        let mut prog = SampledBatchProgram::new();
        prog.push(SampledBatchOp::MixUp {
            lambda: vec![0.5, 0.5],
            perm: vec![1, 0],
        });

        let bytes = prog.to_bytes().unwrap();
        let loaded = SampledBatchProgram::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn test_mosaic_layout() {
        let layout = MosaicLayout::new([(0, 0), (1, 1), (2, 2), (3, 3)], (128, 128));
        assert_eq!(layout.split.0, 128);
        assert_eq!(layout.split.1, 128);
    }
}
