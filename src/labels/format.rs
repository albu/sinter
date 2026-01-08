// Format definitions for bounding boxes and keypoints
//
// These enums define the supported input/output formats for spatial labels.

use serde::{Serialize, Deserialize};

/// Bounding box format
///
/// Defines how a bounding box is represented as an array of floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BBoxFormat {
    /// [x_min, y_min, x_max, y_max] - absolute pixels
    Xyxy,

    /// [x_min, y_min, width, height] - absolute pixels
    Xywh,

    /// [center_x, center_y, width, height] - absolute pixels
    Cxcywh,

    /// [x_min, y_min, x_max, y_max] - normalized [0, 1]
    RelXyxy,

    /// [x_min, y_min, width, height] - normalized [0, 1]
    RelXywh,

    /// [center_x, center_y, width, height] - normalized [0, 1]
    RelCxcywh,
}

impl BBoxFormat {
    /// Number of elements in this format
    pub fn len(&self) -> usize {
        match self {
            BBoxFormat::Xyxy | BBoxFormat::Xywh | BBoxFormat::Cxcywh => 4,
            BBoxFormat::RelXyxy | BBoxFormat::RelXywh | BBoxFormat::RelCxcywh => 4,
        }
    }

    /// Whether this format uses normalized coordinates
    pub fn is_normalized(&self) -> bool {
        matches!(self, BBoxFormat::RelXyxy | BBoxFormat::RelXywh | BBoxFormat::RelCxcywh)
    }

    /// Convert from this format to internal [x, y, w, h] absolute format
    ///
    /// # Arguments
    /// - `input`: Input array in this format
    /// - `img_w`: Image width in pixels
    /// - `img_h`: Image height in pixels
    ///
    /// # Returns
    /// [x, y, w, h] in absolute pixels
    #[inline]
    pub fn to_internal(&self, input: [f32; 4], img_w: u32, img_h: u32) -> [f32; 4] {
        match self {
            BBoxFormat::Xyxy => {
                // [x_min, y_min, x_max, y_max] -> [x, y, w, h]
                let w = input[2] - input[0];
                let h = input[3] - input[1];
                [input[0], input[1], w, h]
            }
            BBoxFormat::Xywh => {
                // Already [x, y, w, h]
                input
            }
            BBoxFormat::Cxcywh => {
                // [cx, cy, w, h] -> [x, y, w, h]
                [input[0] - input[2] / 2.0, input[1] - input[3] / 2.0, input[2], input[3]]
            }
            BBoxFormat::RelXyxy => {
                // Normalized [x_min, y_min, x_max, y_max] -> absolute [x, y, w, h]
                let x_min = input[0] * img_w as f32;
                let y_min = input[1] * img_h as f32;
                let x_max = input[2] * img_w as f32;
                let y_max = input[3] * img_h as f32;
                let w = x_max - x_min;
                let h = y_max - y_min;
                [x_min, y_min, w, h]
            }
            BBoxFormat::RelXywh => {
                // Normalized [x, y, w, h] -> absolute [x, y, w, h]
                [input[0] * img_w as f32, input[1] * img_h as f32,
                 input[2] * img_w as f32, input[3] * img_h as f32]
            }
            BBoxFormat::RelCxcywh => {
                // Normalized [cx, cy, w, h] -> absolute [x, y, w, h]
                let cx = input[0] * img_w as f32;
                let cy = input[1] * img_h as f32;
                let w = input[2] * img_w as f32;
                let h = input[3] * img_h as f32;
                [cx - w / 2.0, cy - h / 2.0, w, h]
            }
        }
    }

    /// Convert from internal [x, y, w, h] absolute format to this format
    ///
    /// # Arguments
    /// - `internal`: [x, y, w, h] in absolute pixels
    /// - `img_w`: Image width in pixels
    /// - `img_h`: Image height in pixels
    ///
    /// # Returns
    /// Array in this format
    #[inline]
    pub fn from_internal(&self, internal: [f32; 4], img_w: u32, img_h: u32) -> [f32; 4] {
        let [x, y, w, h] = internal;
        match self {
            BBoxFormat::Xyxy => {
                // [x, y, w, h] -> [x_min, y_min, x_max, y_max]
                [x, y, x + w, y + h]
            }
            BBoxFormat::Xywh => {
                // Already [x, y, w, h]
                internal
            }
            BBoxFormat::Cxcywh => {
                // [x, y, w, h] -> [cx, cy, w, h]
                [x + w / 2.0, y + h / 2.0, w, h]
            }
            BBoxFormat::RelXyxy => {
                // Absolute [x, y, w, h] -> normalized [x_min, y_min, x_max, y_max]
                let x_min = x / img_w as f32;
                let y_min = y / img_h as f32;
                let x_max = (x + w) / img_w as f32;
                let y_max = (y + h) / img_h as f32;
                [x_min, y_min, x_max, y_max]
            }
            BBoxFormat::RelXywh => {
                // Absolute [x, y, w, h] -> normalized [x, y, w, h]
                [x / img_w as f32, y / img_h as f32, w / img_w as f32, h / img_h as f32]
            }
            BBoxFormat::RelCxcywh => {
                // Absolute [x, y, w, h] -> normalized [cx, cy, w, h]
                let cx = (x + w / 2.0) / img_w as f32;
                let cy = (y + h / 2.0) / img_h as f32;
                [cx, cy, w / img_w as f32, h / img_h as f32]
            }
        }
    }
}

/// Keypoint format
///
/// Defines how a keypoint is represented as an array of floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeypointFormat {
    /// [x, y] - absolute pixels
    Xy,

    /// [x, y, visibility] - absolute pixels
    /// visibility: 0=not_visible, 1=occluded, 2=visible
    Xyv,

    /// [x, y] - normalized [0, 1]
    RelXy,

    /// [x, y, visibility] - normalized [0, 1]
    RelXyv,
}

impl KeypointFormat {
    /// Number of elements in this format
    pub fn len(&self) -> usize {
        match self {
            KeypointFormat::Xy | KeypointFormat::RelXy => 2,
            KeypointFormat::Xyv | KeypointFormat::RelXyv => 3,
        }
    }

    /// Whether this format uses normalized coordinates
    pub fn is_normalized(&self) -> bool {
        matches!(self, KeypointFormat::RelXy | KeypointFormat::RelXyv)
    }

    /// Whether this format includes visibility
    pub fn has_visibility(&self) -> bool {
        matches!(self, KeypointFormat::Xyv | KeypointFormat::RelXyv)
    }

    /// Convert from this format to internal (x, y) absolute format
    ///
    /// # Arguments
    /// - `input`: Input array in this format
    /// - `img_w`: Image width in pixels
    /// - `img_h`: Image height in pixels
    ///
    /// # Returns
    /// (x, y, visibility) where visibility is 2 if not present in input
    #[inline]
    pub fn to_internal(&self, input: &[f32], img_w: u32, img_h: u32) -> (f32, f32, u8) {
        let visibility = if self.has_visibility() {
            input[2] as u8
        } else {
            2 // Default to visible
        };

        let (x, y) = match self {
            KeypointFormat::Xy | KeypointFormat::Xyv => {
                (input[0], input[1])
            }
            KeypointFormat::RelXy | KeypointFormat::RelXyv => {
                (input[0] * img_w as f32, input[1] * img_h as f32)
            }
        };

        (x, y, visibility)
    }

    /// Convert from internal (x, y, visibility) absolute format to this format
    ///
    /// # Arguments
    /// - `x`: X coordinate in absolute pixels
    /// - `y`: Y coordinate in absolute pixels
    /// - `visibility`: Visibility flag (0=not_visible, 1=occluded, 2=visible)
    /// - `img_w`: Image width in pixels
    /// - `img_h`: Image height in pixels
    ///
    /// # Returns
    /// Vec in this format
    #[inline]
    pub fn from_internal(&self, x: f32, y: f32, visibility: u8, img_w: u32, img_h: u32) -> Vec<f32> {
        match self {
            KeypointFormat::Xy => vec![x, y],
            KeypointFormat::Xyv => vec![x, y, visibility as f32],
            KeypointFormat::RelXy => vec![x / img_w as f32, y / img_h as f32],
            KeypointFormat::RelXyv => vec![x / img_w as f32, y / img_h as f32, visibility as f32],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_format_xyxy_to_internal() {
        let format = BBoxFormat::Xyxy;
        let input = [10.0, 20.0, 50.0, 60.0]; // [x_min, y_min, x_max, y_max]
        let result = format.to_internal(input, 100, 100);
        assert_eq!(result, [10.0, 20.0, 40.0, 40.0]); // [x, y, w, h]
    }

    #[test]
    fn test_bbox_format_cxcywh_to_internal() {
        let format = BBoxFormat::Cxcywh;
        let input = [50.0, 50.0, 40.0, 40.0]; // [cx, cy, w, h]
        let result = format.to_internal(input, 100, 100);
        assert_eq!(result, [30.0, 30.0, 40.0, 40.0]); // [x, y, w, h]
    }

    #[test]
    fn test_bbox_format_normalized() {
        let format = BBoxFormat::RelXywh;
        let input = [0.1, 0.2, 0.3, 0.4]; // normalized [x, y, w, h]
        let result = format.to_internal(input, 100, 100);

        // Use approximate comparison for floats due to floating point precision
        let expected = [10.0, 20.0, 30.0, 40.0];
        for i in 0..4 {
            assert!((result[i] - expected[i]).abs() < 0.0001, "to_internal failed at index {}: {} vs {}", i, result[i], expected[i]);
        }

        // Convert back - use approximate comparison for floats
        let output = format.from_internal(result, 100, 100);
        for i in 0..4 {
            assert!((output[i] - input[i]).abs() < 0.0001, "Round-trip failed at index {}: {} vs {}", i, output[i], input[i]);
        }
    }

    #[test]
    fn test_keypoint_format_xyv_to_internal() {
        let format = KeypointFormat::Xyv;
        let input = vec![10.0, 20.0, 1.0]; // [x, y, visibility]
        let result = format.to_internal(&input, 100, 100);
        assert_eq!(result, (10.0, 20.0, 1)); // (x, y, visibility)

        // Convert back
        let output = format.from_internal(result.0, result.1, result.2, 100, 100);
        assert_eq!(output, input);
    }

    #[test]
    fn test_keypoint_format_normalized() {
        let format = KeypointFormat::RelXy;
        let input = vec![0.1, 0.2]; // normalized [x, y]
        let result = format.to_internal(&input, 100, 100);
        assert_eq!(result, (10.0, 20.0, 2)); // absolute with default visibility=2

        // Convert back
        let output = format.from_internal(result.0, result.1, result.2, 100, 100);
        assert_eq!(output, vec![0.1, 0.2]);
    }
}
