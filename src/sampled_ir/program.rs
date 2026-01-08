// SampledImageProgram container
//
// A sampled per-image transform program with versioning and serialization.

use super::ops::SampledImageOp;
use serde::{Deserialize, Serialize};

/// Current IR version (for forward compatibility)
pub const IR_VERSION: u32 = 1;

/// Serialization error type
#[derive(Debug, Clone)]
pub enum SerializationError {
    InvalidVersion { found: u32, expected: u32 },
    BincodeError(String),
    JsonError(String),
}

impl std::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationError::InvalidVersion { found, expected } => {
                write!(
                    f,
                    "Invalid IR version: found {}, expected {}",
                    found, expected
                )
            }
            SerializationError::BincodeError(msg) => {
                write!(f, "Bincode serialization error: {}", msg)
            }
            SerializationError::JsonError(msg) => {
                write!(f, "JSON serialization error: {}", msg)
            }
        }
    }
}

impl std::error::Error for SerializationError {}

/// A sampled per-image transform program
///
/// This is the output of the sampling phase and input to the optimizer.
/// All parameters are fixed, all randomness resolved.
///
/// # Properties
/// - Serializable: can save to disk/load from disk
/// - Replayable: same inputs → same outputs
/// - Inspectable: can print/analyze the plan
/// - Versioned: supports forward compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledImageProgram {
    /// IR version (for forward compatibility)
    pub version: u32,
    /// Ordered sequence of sampled operations (flat list, no nesting)
    pub ops: Vec<SampledImageOp>,
}

impl SampledImageProgram {
    /// Create a new empty program
    pub fn new() -> Self {
        Self {
            version: IR_VERSION,
            ops: Vec::new(),
        }
    }

    /// Add an operation to the program
    pub fn push(&mut self, op: SampledImageOp) {
        self.ops.push(op);
    }

    /// Number of operations in the program
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Is the program empty?
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Get iterator over operations
    pub fn iter(&self) -> impl Iterator<Item = &SampledImageOp> {
        self.ops.iter()
    }

    /// Validate the program version
    pub fn validate_version(&self) -> Result<(), SerializationError> {
        if self.version != IR_VERSION {
            return Err(SerializationError::InvalidVersion {
                found: self.version,
                expected: IR_VERSION,
            });
        }
        Ok(())
    }

    /// Serialize to bytes (for saving to disk)
    ///
    /// Uses bincode for compact, fast serialization.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        bincode::serialize(self).map_err(|e| SerializationError::BincodeError(e.to_string()))
    }

    /// Deserialize from bytes
    ///
    /// Validates the version before returning.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        let program: SampledImageProgram = bincode::deserialize(bytes)
            .map_err(|e| SerializationError::BincodeError(e.to_string()))?;
        program.validate_version()?;
        Ok(program)
    }

    /// Convert to JSON (for inspection/debugging)
    ///
    /// Note: JSON format is not guaranteed stable across versions.
    pub fn to_json(&self) -> Result<String, SerializationError> {
        serde_json::to_string_pretty(self).map_err(|e| SerializationError::JsonError(e.to_string()))
    }

    /// Parse from JSON (for testing/debugging)
    pub fn from_json(json: &str) -> Result<Self, SerializationError> {
        let program: SampledImageProgram =
            serde_json::from_str(json).map_err(|e| SerializationError::JsonError(e.to_string()))?;
        program.validate_version()?;
        Ok(program)
    }

    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> Result<(), SerializationError> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| SerializationError::BincodeError(e.to_string()))?;
        Ok(())
    }

    /// Load from file
    pub fn load(path: &std::path::Path) -> Result<Self, SerializationError> {
        let bytes =
            std::fs::read(path).map_err(|e| SerializationError::BincodeError(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    /// Get a summary of the program
    pub fn summary(&self) -> String {
        if self.ops.is_empty() {
            return "Empty program".to_string();
        }

        let mut summary = format!("Program ({} ops):\n", self.ops.len());
        for (i, op) in self.ops.iter().enumerate() {
            summary.push_str(&format!("  [{}] {}\n", i, op.name()));
        }
        summary
    }
}

impl Default for SampledImageProgram {
    fn default() -> Self {
        Self::new()
    }
}

// Extend with builder pattern
impl SampledImageProgram {
    /// Start building a new program
    pub fn builder() -> SampledImageProgramBuilder {
        SampledImageProgramBuilder::new()
    }

    /// Apply the program to a set of bounding boxes
    ///
    /// # Arguments
    /// - `bboxes`: List of [x, y, w, h] bounding boxes
    /// - `image_size`: Initial (width, height) of the image
    ///
    /// # Returns
    /// - List of transformed bounding boxes. Boxes that are clipped/removed are excluded.
    /// Apply the program to a set of bounding boxes
    ///
    /// # Arguments
    /// - `bboxes`: List of [x, y, w, h] bounding boxes
    /// - `image_size`: Initial (width, height) of the image
    ///
    /// # Returns
    /// - (List of transformed bounding boxes, Final image size (width, height))
    pub fn apply_to_bboxes(
        &self,
        bboxes: Vec<[f32; 4]>,
        image_size: (u32, u32),
    ) -> (Vec<[f32; 4]>, (u32, u32)) {
        let mut current_bboxes = bboxes;
        let mut current_w = image_size.0;
        let mut current_h = image_size.1;

        for op in &self.ops {
            // 1. Apply geometric transform if applicable
            if let Some(lbl_transform) = op.to_label_transform() {
                let mut next_bboxes = Vec::with_capacity(current_bboxes.len());
                for bbox in current_bboxes {
                    if let Some(new_bbox) = lbl_transform.map_bbox(bbox, (current_w, current_h)) {
                        next_bboxes.push(new_bbox);
                    }
                }
                current_bboxes = next_bboxes;
            }

            // 2. Update Image Size for next iteration
            // Note: Only dimensions affect bbox tracking - interpolation/mode/value don't change output size
            use crate::sampled_ir::ops::{RotateAngle, SampledImageOp};
            match op {
                SampledImageOp::Resize {
                    width,
                    height,
                    interpolation: _, /* doesn't affect dimensions */
                } => {
                    current_w = *width;
                    current_h = *height;
                }
                SampledImageOp::Crop {
                    x: _,
                    y: _,
                    width,
                    height, /* crop offset doesn't affect size */
                } => {
                    current_w = *width;
                    current_h = *height;
                }
                SampledImageOp::Pad {
                    top,
                    bottom,
                    left,
                    right,
                    mode: _,  /* doesn't affect dimensions */
                    value: _, /* doesn't affect dimensions */
                } => {
                    current_w += left + right;
                    current_h += top + bottom;
                }
                SampledImageOp::Rotate { angle } => {
                    if matches!(angle, RotateAngle::Rotate90 | RotateAngle::Rotate270) {
                        std::mem::swap(&mut current_w, &mut current_h);
                    }
                }
                SampledImageOp::Transpose => {
                    std::mem::swap(&mut current_w, &mut current_h);
                }
                // Affine in SampledImageOp preserves size (no output_size param)
                _ => {}
            }
        }
        (current_bboxes, (current_w, current_h))
    }

    /// Apply the program to a set of keypoints
    ///
    /// # Arguments
    /// - `keypoints`: List of (x, y) coordinates
    /// - `image_size`: Initial (width, height) of the image
    ///
    /// # Returns
    /// - (List of transformed keypoints, Final image size (width, height))
    pub fn apply_to_keypoints(
        &self,
        keypoints: Vec<(f32, f32)>,
        image_size: (u32, u32),
    ) -> (Vec<(f32, f32)>, (u32, u32)) {
        let mut current_points = keypoints;
        let mut current_w = image_size.0;
        let mut current_h = image_size.1;

        for op in &self.ops {
            // 1. Apply geometric transform if applicable
            if let Some(lbl_transform) = op.to_label_transform() {
                let mut next_points = Vec::with_capacity(current_points.len());
                for point in current_points {
                    if let Some(new_point) = lbl_transform.map_point(point, (current_w, current_h))
                    {
                        next_points.push(new_point);
                    }
                }
                current_points = next_points;
            }

            // 2. Update Image Size (same logic as bboxes)
            use crate::sampled_ir::ops::{RotateAngle, SampledImageOp};
            match op {
                SampledImageOp::Resize {
                    width,
                    height,
                    interpolation: _,
                } => {
                    current_w = *width;
                    current_h = *height;
                }
                SampledImageOp::Crop {
                    x: _,
                    y: _,
                    width,
                    height,
                } => {
                    current_w = *width;
                    current_h = *height;
                }
                SampledImageOp::Pad {
                    top,
                    bottom,
                    left,
                    right,
                    mode: _,
                    value: _,
                } => {
                    current_w += left + right;
                    current_h += top + bottom;
                }
                SampledImageOp::Rotate { angle } => {
                    if matches!(angle, RotateAngle::Rotate90 | RotateAngle::Rotate270) {
                        std::mem::swap(&mut current_w, &mut current_h);
                    }
                }
                SampledImageOp::Transpose => {
                    std::mem::swap(&mut current_w, &mut current_h);
                }
                _ => {}
            }
        }
        (current_points, (current_w, current_h))
    }

    /// Apply the program to classification labels
    ///
    /// # Arguments
    /// - `labels`: List of class labels (integers)
    /// - `_image_size`: Initial (width, height) - not used for classification labels
    ///
    /// # Returns
    /// - The same list of labels (pass-through for most transforms)
    ///
    /// # Note
    /// Most geometric transforms (flip, rotate, resize, crop, etc.) do not affect
    /// classification labels. This method returns the labels unchanged.
    /// Future transforms like MixUp or CutMix may need special handling.
    pub fn apply_to_labels(&self, labels: Vec<i32>, _image_size: (u32, u32)) -> Vec<i32> {
        // For now, all transforms pass through classification labels unchanged
        // Future: MixUp/CutMix would need to mix labels based on their blending ratios
        labels
    }
}

/// Builder for constructing SampledImageProgram
pub struct SampledImageProgramBuilder {
    ops: Vec<SampledImageOp>,
}

impl SampledImageProgramBuilder {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn add(mut self, op: SampledImageOp) -> Self {
        self.ops.push(op);
        self
    }

    pub fn build(self) -> SampledImageProgram {
        SampledImageProgram {
            version: IR_VERSION,
            ops: self.ops,
        }
    }
}

impl Default for SampledImageProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SampledImageProgramBuilder {
    fn clone(&self) -> Self {
        Self {
            ops: self.ops.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_program() {
        let prog = SampledImageProgram::new();
        assert!(prog.is_empty());
        assert_eq!(prog.len(), 0);
        assert_eq!(prog.version, IR_VERSION);
    }

    #[test]
    fn test_add_operations() {
        let mut prog = SampledImageProgram::new();
        prog.push(SampledImageOp::Invert);
        prog.push(SampledImageOp::HorizontalFlip);

        assert_eq!(prog.len(), 2);
        assert!(!prog.is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut prog = SampledImageProgram::new();
        prog.push(SampledImageOp::Invert);
        prog.push(SampledImageOp::Brightness { delta: 42.0 });

        let bytes = prog.to_bytes().unwrap();
        let loaded = SampledImageProgram::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.version, IR_VERSION);
    }

    #[test]
    fn test_invalid_version() {
        let mut prog = SampledImageProgram::new();
        prog.push(SampledImageOp::Invert);
        prog.version = 999; // Wrong version

        let bytes = prog.to_bytes().unwrap();
        let result = SampledImageProgram::from_bytes(&bytes);

        assert!(matches!(
            result,
            Err(SerializationError::InvalidVersion { .. })
        ));
    }

    #[test]
    fn test_json_roundtrip() {
        let mut prog = SampledImageProgram::new();
        prog.push(SampledImageOp::Invert);

        let json = prog.to_json().unwrap();
        let loaded = SampledImageProgram::from_json(&json).unwrap();

        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn test_builder_pattern() {
        let prog = SampledImageProgram::builder()
            .add(SampledImageOp::Invert)
            .add(SampledImageOp::HorizontalFlip)
            .build();

        assert_eq!(prog.len(), 2);
    }

    #[test]
    fn test_summary() {
        let mut prog = SampledImageProgram::new();
        prog.push(SampledImageOp::Invert);
        prog.push(SampledImageOp::HorizontalFlip);

        let summary = prog.summary();
        assert!(summary.contains("2 ops"));
        assert!(summary.contains("Invert"));
        assert!(summary.contains("HorizontalFlip"));
    }
}
