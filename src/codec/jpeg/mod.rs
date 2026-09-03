pub mod color;
pub mod decoder;
pub mod error;
pub mod huffman;
pub mod idct;
pub mod marker;

pub use decoder::JpegDecoder;
pub use error::JpegError;

/// Decode a full JPEG byte buffer into an interleaved (width, height, 3, RGB8) image buffer.
pub fn decode_jpeg(data: &[u8]) -> Result<(usize, usize, usize, Vec<u8>), JpegError> {
    let mut dec = JpegDecoder::new(data);
    dec.decode()
}

/// Decode only a specified Region of Interest (ROI) crop from a JPEG byte buffer.
/// Skips MCU dequantization, IDCT, and color conversion outside the crop!
pub fn decode_jpeg_crop(
    data: &[u8],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Result<(usize, usize, usize, Vec<u8>), JpegError> {
    let mut dec = JpegDecoder::new(data);
    dec.decode_crop(x, y, w, h)
}

/// Fast header parse returning (width, height, channels) in microseconds without decoding pixels.
pub fn read_jpeg_header(data: &[u8]) -> Result<(usize, usize, usize), JpegError> {
    let mut dec = JpegDecoder::new(data);
    let hdr = dec.parse_header()?;
    Ok((hdr.width, hdr.height, hdr.components.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_jpeg_header_parse() {
        let data = fs::read("assets/logo.jpg").expect("assets/logo.jpg should exist");
        let (w, h, c) = read_jpeg_header(&data).expect("header parse should succeed");
        assert_eq!(w, 2592);
        assert_eq!(h, 1632);
        assert_eq!(c, 3);
    }

    #[test]
    fn test_jpeg_crop_matches_full_decode() {
        let data = fs::read("assets/test_baseline.jpg").expect("assets/test_baseline.jpg should exist");

        // 1. Decode a 128x128 crop at offset (200, 300)
        let (cw, ch, cc, crop_pixels) = decode_jpeg_crop(&data, 200, 300, 128, 128)
            .expect("crop decode should succeed");
        assert_eq!(cw, 128);
        assert_eq!(ch, 128);
        assert_eq!(cc, 3);
        assert_eq!(crop_pixels.len(), 128 * 128 * 3);

        // 2. Full decode
        let (fw, fh, fc, full_pixels) = decode_jpeg(&data)
            .expect("full decode should succeed");
        assert_eq!(fw, 2592);
        assert_eq!(fh, 1632);
        assert_eq!(fc, 3);

        // 3. Verify that pixels in the crop match the full decode pixels exactly
        for y in 0..128 {
            for x in 0..128 {
                let crop_idx = (y * 128 + x) * 3;
                let full_idx = ((300 + y) * 2592 + (200 + x)) * 3;

                assert_eq!(crop_pixels[crop_idx], full_pixels[full_idx], "R mismatch at ({}, {})", x, y);
                assert_eq!(crop_pixels[crop_idx + 1], full_pixels[full_idx + 1], "G mismatch at ({}, {})", x, y);
                assert_eq!(crop_pixels[crop_idx + 2], full_pixels[full_idx + 2], "B mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn test_jpeg_crop_speedup_benchmark() {
        use std::time::Instant;
        let data = fs::read("assets/test_baseline.jpg").expect("assets/test_baseline.jpg should exist");

        // Warmup
        let _ = decode_jpeg_crop(&data, 500, 500, 256, 256).unwrap();
        let _ = decode_jpeg(&data).unwrap();

        // Measure Crop Decode
        let iters = 10;
        let start_crop = Instant::now();
        for _ in 0..iters {
            let _ = decode_jpeg_crop(&data, 500, 500, 256, 256).unwrap();
        }
        let crop_duration = start_crop.elapsed() / iters;

        // Measure Full Decode
        let start_full = Instant::now();
        for _ in 0..iters {
            let _ = decode_jpeg(&data).unwrap();
        }
        let full_duration = start_full.elapsed() / iters;

        let speedup = (full_duration.as_secs_f64() / crop_duration.as_secs_f64());
        println!("\n=== JPEG ROI CROP BENCHMARK (2592x1632 -> 256x256 crop) ===");
        println!("Full Image Decode: {:?}", full_duration);
        println!("MCU ROI Crop Decode: {:?}", crop_duration);
        println!("ROI Crop Speedup: {:.2}x faster!", speedup);
        println!("===========================================================\n");

        assert!(crop_duration < full_duration);
    }
}
