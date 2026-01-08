// Zero-copy keypoint array wrapper
//
// KeypointArray wraps a numpy array of keypoints without copying data.
// Transformations are applied in-place when needed.

use crate::labels::format::KeypointFormat;

/// Zero-copy wrapper for keypoint arrays
///
/// This type wraps a numpy array (or any contiguous f32 array) without
/// copying the data. Transformations are applied in-place when needed.
///
/// # Layout
///
/// The underlying data is stored as a flat slice: [x1, y1, v1, x2, y2, v2, ...]
/// where visibility is optional depending on format.
///
/// # Visibility Values
///
/// - 0: Not visible (keypoint is outside image or occluded)
/// - 1: Occluded (keypoint is present but not visible)
/// - 2: Visible (keypoint is clearly visible)
///
/// # Format Handling
///
/// - **Input format**: Converted once from specified format to internal (x, y, visibility) absolute
/// - **Storage**: Always stored as (x, y, visibility) absolute pixels
/// - **Output format**: Converted from internal to specified format when extracting
pub struct KeypointArray<'a> {
    /// Raw data slice (borrowed from numpy array)
    data: &'a [f32],
    /// Number of keypoints
    count: usize,
    /// Input format (for conversion on construction)
    input_format: KeypointFormat,
    /// Output format (for conversion on extraction)
    output_format: KeypointFormat,
    /// Image dimensions (for normalized <-> absolute conversion)
    img_w: u32,
    img_h: u32,
}

impl<'a> KeypointArray<'a> {
    /// Create a KeypointArray from a raw slice in the specified format
    ///
    /// # Arguments
    /// - `data`: Flat slice of f32 values
    /// - `format`: Format of the input data
    /// - `img_w`: Image width in pixels
    /// - `img_h`: Image height in pixels
    ///
    /// # Returns
    /// KeypointArray that wraps the data without copying
    pub fn from_slice(
        data: &'a [f32],
        format: KeypointFormat,
        img_w: u32,
        img_h: u32,
    ) -> Self {
        assert!(data.len() % format.len() == 0,
                "Data length {} is not a multiple of format length {}",
                data.len(), format.len());

        let count = data.len() / format.len();

        KeypointArray {
            data,
            count,
            input_format: format,
            output_format: format, // Default: same as input
            img_w,
            img_h,
        }
    }

    /// Set the output format for conversions when extracting
    pub fn with_output_format(mut self, format: KeypointFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Get the number of keypoints
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get image dimensions
    pub fn image_size(&self) -> (u32, u32) {
        (self.img_w, self.img_h)
    }

    /// Iterate over keypoints in internal (x, y, visibility) absolute format
    ///
    /// This performs conversion from input format on the fly.
    pub fn iter_internal(&self) -> impl Iterator<Item = (f32, f32, u8)> + '_ {
        let format = self.input_format;
        let img_w = self.img_w;
        let img_h = self.img_h;
        let stride = format.len();

        self.data.chunks(stride).map(move |chunk| {
            format.to_internal(chunk, img_w, img_h)
        })
    }

    /// Convert to owned Vec of internal format (x, y, visibility)
    ///
    /// This allocates a new Vec and is useful when you need to modify
    /// the keypoints and can't keep the original borrow.
    pub fn to_vec_internal(&self) -> Vec<(f32, f32, u8)> {
        self.iter_internal().collect()
    }

    /// Convert to owned Vec in the output format
    ///
    /// This allocates a new Vec with each keypoint in the output format.
    pub fn to_vec_output(&self) -> Vec<Vec<f32>> {
        let format = self.output_format;
        let img_w = self.img_w;
        let img_h = self.img_h;

        self.iter_internal().map(|(x, y, visibility)| {
            format.from_internal(x, y, visibility, img_w, img_h)
        }).collect()
    }
}

/// Owned version of KeypointArray that stores data after transformation
///
/// This is used when keypoints need to be modified (e.g., filtered, clipped).
pub struct KeypointArrayOwned {
    /// Owned data in internal (x, y, visibility) absolute format
    data: Vec<(f32, f32, u8)>,
    /// Output format (for conversion when extracting)
    output_format: KeypointFormat,
    /// Image dimensions
    img_w: u32,
    img_h: u32,
}

impl KeypointArrayOwned {
    /// Create from a slice of internal-format keypoints
    pub fn from_internal(data: Vec<(f32, f32, u8)>, img_w: u32, img_h: u32) -> Self {
        KeypointArrayOwned {
            data,
            output_format: KeypointFormat::Xy, // Default output format
            img_w,
            img_h,
        }
    }

    /// Set the output format
    pub fn with_output_format(mut self, format: KeypointFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Get number of keypoints
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get image dimensions
    pub fn image_size(&self) -> (u32, u32) {
        (self.img_w, self.img_h)
    }

    /// Get mutable reference to internal data
    pub fn data_mut(&mut self) -> &mut [(f32, f32, u8)] {
        &mut self.data
    }

    /// Get reference to internal data
    pub fn data(&self) -> &[(f32, f32, u8)] {
        &self.data
    }

    /// Convert to output format as Vec<Vec<f32>>
    pub fn to_output(&self) -> Vec<Vec<f32>> {
        let format = self.output_format;
        let img_w = self.img_w;
        let img_h = self.img_h;

        self.data.iter().map(|&(x, y, visibility)| {
            format.from_internal(x, y, visibility, img_w, img_h)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypoint_array_from_slice() {
        let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let kpts = KeypointArray::from_slice(&data, KeypointFormat::Xy, 100, 100);

        assert_eq!(kpts.len(), 2);
        assert!(!kpts.is_empty());
        assert_eq!(kpts.image_size(), (100, 100));
    }

    #[test]
    fn test_keypoint_array_xyv_to_internal() {
        let data: Vec<f32> = vec![10.0, 20.0, 2.0, 30.0, 40.0, 1.0]; // [(x, y, v), (x, y, v)]
        let kpts = KeypointArray::from_slice(&data, KeypointFormat::Xyv, 100, 100);

        let internal: Vec<(f32, f32, u8)> = kpts.iter_internal().collect();
        assert_eq!(internal[0], (10.0, 20.0, 2));
        assert_eq!(internal[1], (30.0, 40.0, 1));
    }

    #[test]
    fn test_keypoint_array_default_visibility() {
        let data: Vec<f32> = vec![10.0, 20.0]; // [x, y] without visibility
        let kpts = KeypointArray::from_slice(&data, KeypointFormat::Xy, 100, 100);

        let internal: Vec<(f32, f32, u8)> = kpts.iter_internal().collect();
        assert_eq!(internal[0], (10.0, 20.0, 2)); // Default visibility=2 (visible)
    }

    #[test]
    fn test_keypoint_array_normalized() {
        let data: Vec<f32> = vec![0.1, 0.2]; // normalized [x, y]
        let kpts = KeypointArray::from_slice(&data, KeypointFormat::RelXy, 100, 100);

        let internal: Vec<(f32, f32, u8)> = kpts.iter_internal().collect();
        assert_eq!(internal[0], (10.0, 20.0, 2)); // Converted to absolute

        // Convert back to normalized
        let output = kpts.with_output_format(KeypointFormat::RelXy).to_vec_output();
        assert_eq!(output[0], vec![0.1, 0.2]);
    }

    #[test]
    fn test_keypoint_array_owned_to_output() {
        let data = vec![(10.0, 20.0, 2), (30.0, 40.0, 1)];
        let kpts = KeypointArrayOwned::from_internal(data, 100, 100)
            .with_output_format(KeypointFormat::Xyv);

        let output = kpts.to_output();
        assert_eq!(output[0], vec![10.0, 20.0, 2.0]);
        assert_eq!(output[1], vec![30.0, 40.0, 1.0]);
    }
}
