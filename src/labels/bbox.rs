// Zero-copy bounding box array wrapper
//
// BBoxArray wraps a numpy array of bounding boxes without copying data.
// Transformations are applied in-place when needed.

use crate::labels::format::BBoxFormat;

/// Zero-copy wrapper for bounding box arrays
///
/// This type wraps a numpy array (or any contiguous f32 array) without
/// copying the data. Transformations are applied in-place when needed.
///
/// # Layout
///
/// The underlying data is stored as a flat slice: [x1, y1, w1, h1, x2, y2, w2, h2, ...]
///
/// # Format Handling
///
/// - **Input format**: Converted once from specified format to internal [x, y, w, h] absolute
/// - **Storage**: Always stored as [x, y, w, h] absolute pixels
/// - **Output format**: Converted from internal to specified format when extracting
pub struct BBoxArray<'a> {
    /// Raw data slice (borrowed from numpy array)
    data: &'a [f32],
    /// Number of bounding boxes
    count: usize,
    /// Input format (for conversion on construction)
    input_format: BBoxFormat,
    /// Output format (for conversion on extraction)
    output_format: BBoxFormat,
    /// Image dimensions (for normalized <-> absolute conversion)
    img_w: u32,
    img_h: u32,
}

impl<'a> BBoxArray<'a> {
    /// Create a BBoxArray from a raw slice in the specified format
    ///
    /// # Arguments
    /// - `data`: Flat slice of f32 values
    /// - `format`: Format of the input data
    /// - `img_w`: Image width in pixels
    /// - `img_h`: Image height in pixels
    ///
    /// # Returns
    /// BBoxArray that wraps the data without copying
    pub fn from_slice(
        data: &'a [f32],
        format: BBoxFormat,
        img_w: u32,
        img_h: u32,
    ) -> Self {
        assert!(data.len() % format.len() == 0,
                "Data length {} is not a multiple of format length {}",
                data.len(), format.len());

        let count = data.len() / format.len();

        BBoxArray {
            data,
            count,
            input_format: format,
            output_format: format, // Default: same as input
            img_w,
            img_h,
        }
    }

    /// Set the output format for conversions when extracting
    pub fn with_output_format(mut self, format: BBoxFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Get the number of bounding boxes
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

    /// Iterate over bounding boxes in internal [x, y, w, h] absolute format
    ///
    /// This performs conversion from input format on the fly.
    pub fn iter_internal(&self) -> impl Iterator<Item = [f32; 4]> + '_ {
        let format = self.input_format;
        let img_w = self.img_w;
        let img_h = self.img_h;
        let stride = format.len();

        self.data.chunks(stride).map(move |chunk| {
            let mut arr = [0.0f32; 4];
            arr.copy_from_slice(&chunk[..4.min(chunk.len())]);
            format.to_internal(arr, img_w, img_h)
        })
    }

    /// Convert to owned Vec of internal format [x, y, w, h]
    ///
    /// This allocates a new Vec and is useful when you need to modify
    /// the bboxes and can't keep the original borrow.
    pub fn to_vec_internal(&self) -> Vec<[f32; 4]> {
        self.iter_internal().collect()
    }

    /// Convert to owned Vec in the output format
    ///
    /// This allocates a new Vec with each bbox in the output format.
    pub fn to_vec_output(&self) -> Vec<Vec<f32>> {
        let format = self.output_format;
        let img_w = self.img_w;
        let img_h = self.img_h;

        self.iter_internal().map(|internal| {
            let output = format.from_internal(internal, img_w, img_h);
            output.to_vec()
        }).collect()
    }
}

/// Owned version of BBoxArray that stores data after transformation
///
/// This is used when bboxes need to be modified (e.g., filtered, clipped).
pub struct BBoxArrayOwned {
    /// Owned data in internal [x, y, w, h] absolute format
    data: Vec<[f32; 4]>,
    /// Output format (for conversion when extracting)
    output_format: BBoxFormat,
    /// Image dimensions
    img_w: u32,
    img_h: u32,
}

impl BBoxArrayOwned {
    /// Create from a slice of internal-format bboxes
    pub fn from_internal(data: Vec<[f32; 4]>, img_w: u32, img_h: u32) -> Self {
        BBoxArrayOwned {
            data,
            output_format: BBoxFormat::Xywh, // Default output format
            img_w,
            img_h,
        }
    }

    /// Set the output format
    pub fn with_output_format(mut self, format: BBoxFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Get number of bboxes
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
    pub fn data_mut(&mut self) -> &mut [[f32; 4]] {
        &mut self.data
    }

    /// Get reference to internal data
    pub fn data(&self) -> &[[f32; 4]] {
        &self.data
    }

    /// Convert to output format as Vec<Vec<f32>>
    pub fn to_output(&self) -> Vec<Vec<f32>> {
        let format = self.output_format;
        let img_w = self.img_w;
        let img_h = self.img_h;

        self.data.iter().map(|&internal| {
            let output = format.from_internal(internal, img_w, img_h);
            output.to_vec()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_array_from_slice() {
        let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let bboxes = BBoxArray::from_slice(&data, BBoxFormat::Xywh, 100, 100);

        assert_eq!(bboxes.len(), 2);
        assert!(!bboxes.is_empty());
        assert_eq!(bboxes.image_size(), (100, 100));
    }

    #[test]
    fn test_bbox_array_format_conversion() {
        let data: Vec<f32> = vec![10.0, 20.0, 40.0, 40.0]; // [x, y, w, h]
        let bboxes = BBoxArray::from_slice(&data, BBoxFormat::Xywh, 100, 100);

        // Should round-trip
        let internal: Vec<[f32; 4]> = bboxes.iter_internal().collect();
        assert_eq!(internal[0], [10.0, 20.0, 40.0, 40.0]);
    }

    #[test]
    fn test_bbox_array_xyxy_to_internal() {
        let data: Vec<f32> = vec![10.0, 20.0, 50.0, 60.0]; // [x_min, y_min, x_max, y_max]
        let bboxes = BBoxArray::from_slice(&data, BBoxFormat::Xyxy, 100, 100);

        let internal: Vec<[f32; 4]> = bboxes.iter_internal().collect();
        assert_eq!(internal[0], [10.0, 20.0, 40.0, 40.0]); // [x, y, w, h]
    }

    #[test]
    fn test_bbox_array_owned_to_output() {
        let data = vec![[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]];
        let bboxes = BBoxArrayOwned::from_internal(data, 100, 100)
            .with_output_format(BBoxFormat::Xyxy);

        let output = bboxes.to_output();
        assert_eq!(output[0], vec![10.0, 20.0, 40.0, 60.0]); // [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        assert_eq!(output[1], vec![50.0, 60.0, 120.0, 140.0]);
    }
}
