pub mod jpeg;

use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

/// Read an image from disk directly into RGB8 memory with native SIMD decoding
pub fn imread<P: AsRef<Path>>(path: P) -> Result<ImageBuffer, Box<dyn std::error::Error + Send + Sync>> {
    let bytes = fs::read(path)?;
    imread_bytes(&bytes)
}

/// Read only a specified crop region from disk, skipping MCU dequantization and IDCT outside the crop
pub fn imread_crop<P: AsRef<Path>>(
    path: P,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Result<ImageBuffer, Box<dyn std::error::Error + Send + Sync>> {
    let bytes = fs::read(path)?;
    imread_crop_bytes(&bytes, x, y, w, h)
}

/// Quickly read image dimensions and channel count from file header without decoding pixels
pub fn read_header<P: AsRef<Path>>(path: P) -> Result<(usize, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    let bytes = fs::read(path)?;
    read_header_bytes(&bytes)
}

/// Decode in-memory image bytes directly into an RGB8 ImageBuffer
pub fn imread_bytes(bytes: &[u8]) -> Result<ImageBuffer, Box<dyn std::error::Error + Send + Sync>> {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        let (w, h, c, data) = jpeg::decode_jpeg(bytes)?;
        Ok(ImageBuffer { width: w, height: h, channels: c, data })
    } else {
        Err("Unsupported image format (currently JPEG supported natively)".into())
    }
}

/// Decode an in-memory ROI crop region directly into an RGB8 ImageBuffer
pub fn imread_crop_bytes(
    bytes: &[u8],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Result<ImageBuffer, Box<dyn std::error::Error + Send + Sync>> {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        let (w, h, c, data) = jpeg::decode_jpeg_crop(bytes, x, y, w, h)?;
        Ok(ImageBuffer { width: w, height: h, channels: c, data })
    } else {
        Err("Unsupported image format for crop (currently JPEG supported natively)".into())
    }
}

/// Read dimensions from in-memory byte slice
pub fn read_header_bytes(bytes: &[u8]) -> Result<(usize, usize, usize), Box<dyn std::error::Error + Send + Sync>> {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        let (w, h, c) = jpeg::read_jpeg_header(bytes)?;
        Ok((w, h, c))
    } else {
        Err("Unsupported image format".into())
    }
}
