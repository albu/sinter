// Simple median blur helpers
//
// Fast helper functions for median computation, used by tests and edge handling.

/// Edge handling - same as per-pixel median
pub fn median_edge_fast(data: &[u8], width: usize, height: usize, channels: usize, x: usize, y: usize, c: usize) -> u8 {
    median_per_pixel(data, width, height, channels, x, y, c)
}

/// Compute median for a single pixel by collecting and sorting neighbors
fn median_per_pixel(data: &[u8], width: usize, height: usize, channels: usize, x: usize, y: usize, c: usize) -> u8 {
    let mut pixels = [0u8; 9];
    let mut idx = 0;

    let y_start = y.saturating_sub(1);
    let y_end = (y + 2).min(height);
    let x_start = x.saturating_sub(1);
    let x_end = (x + 2).min(width);

    for py in y_start..y_end {
        for px in x_start..x_end {
            pixels[idx] = data[(py * width + px) * channels + c];
            idx += 1;
        }
    }

    pixels[..idx].sort();
    pixels[idx / 2]
}

/// Clipped mean (removes min/max, averages remaining)
pub fn clipped_mean_3x3_scalar(data: &[u8], width: usize, channels: usize, c: usize, y: usize, x: usize) -> u8 {
    let row_stride = width * channels;
    let mut sum: u16 = 0;
    let mut min = u8::MAX;
    let mut max = u8::MIN;

    for dy in -1i32..=1 {
        let py = (y as i32 + dy) as usize;
        for dx in -1i32..=1 {
            let px = (x as i32 + dx) as usize;
            let val = data[py * row_stride + px * channels + c];
            sum += val as u16;
            min = min.min(val);
            max = max.max(val);
        }
    }

    ((sum - min as u16 - max as u16) / 7) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_per_pixel_constant() {
        let data = vec![128u8; 9 * 3]; // 3x3 RGB
        let result = median_per_pixel(&data, 3, 3, 3, 1, 1, 0);
        assert_eq!(result, 128);
    }

    #[test]
    fn test_median_per_pixel_salt_pepper() {
        let mut data = vec![128u8; 9];
        data[4] = 0; // center is salt-pepper
        let result = median_per_pixel(&data, 3, 3, 1, 1, 1, 0);
        // Median of [128,128,128,128,0,128,128,128,128] is 128
        assert_eq!(result, 128);
    }

    #[test]
    fn test_median_per_pixel_edge() {
        let data = vec![100u8; 9];
        let result = median_per_pixel(&data, 3, 3, 1, 0, 0, 0);
        // Corner pixel - only has 4 neighbors (including itself)
        assert_eq!(result, 100);
    }

    #[test]
    fn test_clipped_mean_basic() {
        let data = vec![100u8; 9 * 3]; // 3x3 RGB
        let result = clipped_mean_3x3_scalar(&data, 3, 3, 0, 1, 1);
        assert_eq!(result, 100);
    }
}
