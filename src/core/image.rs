// Image types
//
// This module defines the two image representations used by the system:
// - FusableImage: Borrowed view for zero-copy fused operations
// - BarrierImage: Owned data for shape-changing operations

/// Fusable image representation (borrowed view)
///
/// Minimal, boring representation:
/// - HWC layout (Height-Width-Channels)
/// - u8 pixel values (0-255)
/// - Contiguous memory
/// - Explicit lifetime for borrow tracking
///
/// This representation is designed to:
/// - Enable in-place mutation
/// - Allow safe fusion analysis
/// - Avoid hidden allocations
/// - Support zero-copy per-pixel operations
///
/// Used for fused nodes where all transforms are InPlace + Preserve.
pub struct FusableImage<'a> {
    /// Contiguous HWC pixel data
    pub data: &'a mut [u8],
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

/// Barrier image representation (owned data)
///
/// Owns its pixel data. Used for transforms that allocate new buffers
/// (like Resize, Crop, Pad, etc.).
///
/// Features:
/// - Optional stride for padding / kernel safety
/// - Optional alignment for SIMD operations
/// - Flexible layout support
///
/// Used for barrier nodes that change shape or require special memory layout.
#[derive(Clone)]
pub struct BarrierImage {
    /// Owned HWC pixel data
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    /// Stride in bytes between rows (may be larger than width * channels for padding)
    pub stride: usize,
    /// Alignment hint for SIMD operations (0 = no specific alignment)
    pub alignment: usize,
}

impl BarrierImage {
    /// Create a new BarrierImage with the given dimensions
    ///
    /// By default, stride is set to width * channels (contiguous rows)
    /// and alignment is 0 (no specific alignment requirement).
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        let size = width * height * channels;
        let stride = width * channels;
        Self {
            data: vec![0u8; size],
            width,
            height,
            channels,
            stride,
            alignment: 0,
        }
    }

    /// Create a new BarrierImage with custom stride and alignment
    pub fn with_layout(width: usize, height: usize, channels: usize, stride: usize, alignment: usize) -> Self {
        let total_size = stride * height;
        Self {
            data: vec![0u8; total_size],
            width,
            height,
            channels,
            stride,
            alignment,
        }
    }

    /// Create from raw components
    pub fn from_vec(data: Vec<u8>, width: usize, height: usize, channels: usize) -> Self {
        let expected_size = width * height * channels;
        assert_eq!(data.len(), expected_size, "data size mismatch");
        let stride = width * channels;
        Self {
            data,
            width,
            height,
            channels,
            stride,
            alignment: 0,
        }
    }

    /// Total number of pixels
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Is the data contiguous (stride == width * channels)?
    pub fn is_contiguous(&self) -> bool {
        self.stride == self.width * self.channels
    }

    /// Borrow as FusableImage
    ///
    /// This provides a zero-copy view into the BarrierImage's data.
    /// Only valid if the data is contiguous.
    pub fn as_fusable(&mut self) -> FusableImage<'_> {
        assert!(self.is_contiguous(), "Cannot convert non-contiguous BarrierImage to FusableImage");
        FusableImage {
            data: &mut self.data,
            width: self.width,
            height: self.height,
            channels: self.channels,
        }
    }

    /// Convert from FusableImage to BarrierImage
    ///
    /// This allocates a new buffer and copies the data.
    pub fn from_fusable(img: &FusableImage<'_>) -> Self {
        let stride = img.width * img.channels;
        Self {
            data: img.data.to_vec(),
            width: img.width,
            height: img.height,
            channels: img.channels,
            stride,
            alignment: 0,
        }
    }
}

impl<'a> FusableImage<'a> {
    /// Create a new FusableImage from raw components
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `data.len() == width * height * channels`
    /// - data is valid HWC layout
    pub fn new(data: &'a mut [u8], width: usize, height: usize, channels: usize) -> Self {
        assert_eq!(data.len(), width * height * channels, "data size mismatch");
        Self {
            data,
            width,
            height,
            channels,
        }
    }

    /// Create a new FusableImage from a raw pointer (for numpy interop)
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - Pointer is valid for the lifetime 'a
    /// - Memory region is `width * height * channels` bytes
    /// - Memory is valid HWC layout
    /// - No aliasing occurs
    ///
    /// # Arguments
    /// - `ptr`: Raw pointer to image data (must be valid for lifetime 'a)
    /// - `width`: Image width in pixels
    /// - `height`: Image height in pixels
    /// - `channels`: Number of channels (1-4)
    ///
    /// # Safety Contract
    /// The lifetime 'a is trusted - caller must ensure the pointer remains valid
    /// for the entire lifetime. This is used for numpy interop where the numpy
    /// array's lifetime guarantees the validity of the pointer.
    pub unsafe fn new_raw(ptr: *mut u8, width: u32, height: u32, channels: u32) -> Self {
        let len = (width as usize) * (height as usize) * (channels as usize);
        let data = std::slice::from_raw_parts_mut(ptr, len);
        Self {
            data,
            width: width as usize,
            height: height as usize,
            channels: channels as usize,
        }
    }

    /// Total number of pixels
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Stride in bytes between rows (for contiguous HWC, this is width * channels)
    pub fn row_stride(&self) -> usize {
        self.width * self.channels
    }

    /// Convert to BarrierImage (allocates new buffer)
    ///
    /// This is useful when transitioning from a fused block to a barrier node.
    pub fn to_barrier(&self) -> BarrierImage {
        BarrierImage::from_fusable(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusable_image_creation() {
        let mut data = vec![0u8; 100 * 100 * 3];
        let img = FusableImage::new(&mut data, 100, 100, 3);

        assert_eq!(img.width, 100);
        assert_eq!(img.height, 100);
        assert_eq!(img.channels, 3);
        assert_eq!(img.pixel_count(), 10000);
        assert_eq!(img.row_stride(), 300);
    }

    #[test]
    fn test_barrier_image_creation() {
        let img = BarrierImage::new(100, 100, 3);

        assert_eq!(img.width, 100);
        assert_eq!(img.height, 100);
        assert_eq!(img.channels, 3);
        assert_eq!(img.pixel_count(), 10000);
        assert!(img.is_contiguous());
        assert_eq!(img.stride, 300);
    }

    #[test]
    fn test_fusable_to_barrier_conversion() {
        let mut data = vec![42u8; 50 * 50 * 3];
        let fusable = FusableImage::new(&mut data, 50, 50, 3);

        let barrier = fusable.to_barrier();

        assert_eq!(barrier.width, 50);
        assert_eq!(barrier.height, 50);
        assert_eq!(barrier.channels, 3);
        assert_eq!(barrier.data.len(), 50 * 50 * 3);
        assert!(barrier.data.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_barrier_to_fusable_conversion() {
        let mut barrier = BarrierImage::new(64, 64, 1);
        // Fill with pattern
        for (i, px) in barrier.data.iter_mut().enumerate() {
            *px = (i % 256) as u8;
        }

        let fusable = barrier.as_fusable();

        assert_eq!(fusable.width, 64);
        assert_eq!(fusable.height, 64);
        assert_eq!(fusable.channels, 1);
        assert_eq!(fusable.data.len(), 64 * 64);
    }

    #[test]
    #[should_panic(expected = "data size mismatch")]
    fn test_fusable_image_size_validation() {
        let mut data = vec![0u8; 100];
        let _ = FusableImage::new(&mut data, 100, 100, 3);
    }
}
