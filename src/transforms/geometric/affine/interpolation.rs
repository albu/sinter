// Interpolation methods for affine transforms

/// Nearest-neighbor interpolation
///
/// Interpolates the value at a non-integer position (x, y) in the image
/// using the nearest pixel (rounding to nearest integer coordinate).
///
/// This is the preferred method for masks and labels to avoid introducing
/// artifacts from interpolation.
#[inline]
pub(crate) fn nearest_interpolate(
    data: &[u8],
    x: f32,
    y: f32,
    width: usize,
    height: usize,
    channels: usize,
    channel: usize,
) -> u8 {
    let xi = (x.round() as i32).clamp(0, width as i32 - 1);
    let yi = (y.round() as i32).clamp(0, height as i32 - 1);

    let idx = (yi as usize * width + xi as usize) * channels + channel;
    data[idx]
}

/// Bilinear interpolation
///
/// Interpolates the value at a non-integer position (x, y) in the image
/// using the 4 nearest neighboring pixels.
///
/// This is preferred for natural images to produce smoother results.
#[inline]
pub(crate) fn bilinear_interpolate(
    data: &[u8],
    x: f32,
    y: f32,
    width: usize,
    height: usize,
    channels: usize,
    channel: usize,
) -> u8 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x.floor();
    let dy = y - y.floor();

    // Sample 4 neighboring pixels with bounds checking
    let sample = |xi: i32, yi: i32| -> f32 {
        if xi < 0 || xi >= width as i32 || yi < 0 || yi >= height as i32 {
            0.0 // Out of bounds = black
        } else {
            let idx = (yi as usize * width + xi as usize) * channels + channel;
            data[idx] as f32
        }
    };

    let v00 = sample(x0, y0);
    let v10 = sample(x1, y0);
    let v01 = sample(x0, y1);
    let v11 = sample(x1, y1);

    // Bilinear interpolation
    let top = v00 * (1.0 - dx) + v10 * dx;
    let bottom = v01 * (1.0 - dx) + v11 * dx;
    let result = top * (1.0 - dy) + bottom * dy;

    result.clamp(0.0, 255.0) as u8
}
