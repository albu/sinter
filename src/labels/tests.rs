// Integration tests for label transformations
//
// Tests that geometric transforms correctly map bounding boxes and keypoints.

use crate::core::LabelTransform;
use crate::transforms::geometric::*;
use crate::transforms::*;

// Helper macro for float comparison with tolerance
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $tol:expr) => {
        let (a, b) = ($a, $b);
        assert!(
            (a - b).abs() < $tol,
            "Values not approximately equal: {} vs {} (tolerance {})",
            a,
            b,
            $tol
        );
    };
}

// =============================================================================
// HorizontalFlip Tests
// =============================================================================

#[test]
fn test_horizontal_flip_bbox() {
    let transform = HorizontalFlip;

    // [x, y, w, h] in a 100x100 image
    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // After flip: x' = img_w - x - w = 100 - 10 - 30 = 60
    assert_approx_eq!(result[0], 60.0, 0.01); // x
    assert_approx_eq!(result[1], 20.0, 0.01); // y (unchanged)
    assert_approx_eq!(result[2], 30.0, 0.01); // w (unchanged)
    assert_approx_eq!(result[3], 40.0, 0.01); // h (unchanged)
}

#[test]
fn test_horizontal_flip_point() {
    let transform = HorizontalFlip;

    let point = (10.0, 20.0);
    let result = transform.map_point(point, (100, 100)).unwrap();

    // After flip: x' = img_w - x = 100 - 10 = 90
    assert_approx_eq!(result.0, 90.0, 0.01); // x
    assert_approx_eq!(result.1, 20.0, 0.01); // y (unchanged)
}

#[test]
fn test_horizontal_flip_bbox_clipped() {
    let transform = HorizontalFlip;

    // Box that will be partially outside after flip
    let bbox = [95.0, 20.0, 10.0, 30.0]; // Starts at x=95, width=10
    let result = transform.map_bbox(bbox, (100, 100));

    // Should be clipped (None) since it would extend outside the image
    assert!(result.is_none());
}

// =============================================================================
// VerticalFlip Tests
// =============================================================================

#[test]
fn test_vertical_flip_bbox() {
    let transform = VerticalFlip;

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // After flip: y' = img_h - y - h = 100 - 20 - 40 = 40
    assert_approx_eq!(result[0], 10.0, 0.01); // x (unchanged)
    assert_approx_eq!(result[1], 40.0, 0.01); // y
    assert_approx_eq!(result[2], 30.0, 0.01); // w (unchanged)
    assert_approx_eq!(result[3], 40.0, 0.01); // h (unchanged)
}

#[test]
fn test_vertical_flip_point() {
    let transform = VerticalFlip;

    let point = (10.0, 20.0);
    let result = transform.map_point(point, (100, 100)).unwrap();

    // After flip: y' = img_h - y = 100 - 20 = 80
    assert_approx_eq!(result.0, 10.0, 0.01); // x (unchanged)
    assert_approx_eq!(result.1, 80.0, 0.01); // y
}

// =============================================================================
// Transpose Tests
// =============================================================================

#[test]
fn test_transpose_bbox() {
    let transform = Transpose;

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // After transpose: swap x/y and w/h
    assert_approx_eq!(result[0], 20.0, 0.01); // x (was y)
    assert_approx_eq!(result[1], 10.0, 0.01); // y (was x)
    assert_approx_eq!(result[2], 40.0, 0.01); // w (was h)
    assert_approx_eq!(result[3], 30.0, 0.01); // h (was w)
}

#[test]
fn test_transpose_point() {
    let transform = Transpose;

    let point = (10.0, 20.0);
    let result = transform.map_point(point, (100, 100)).unwrap();

    // After transpose: swap x and y
    assert_approx_eq!(result.0, 20.0, 0.01); // x (was y)
    assert_approx_eq!(result.1, 10.0, 0.01); // y (was x)
}

// =============================================================================
// Rotate Tests
// =============================================================================

#[test]
fn test_rotate_90_bbox() {
    let transform = Rotate::new(rotate::RotateAngle::Rotate90);

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // Rotate 90° clockwise:
    // x' = img_h - y - h = 100 - 20 - 40 = 40
    // y' = x = 10
    // w' = h = 40
    // h' = w = 30
    assert_approx_eq!(result[0], 40.0, 0.01);
    assert_approx_eq!(result[1], 10.0, 0.01);
    assert_approx_eq!(result[2], 40.0, 0.01);
    assert_approx_eq!(result[3], 30.0, 0.01);
}

#[test]
fn test_rotate_180_bbox() {
    let transform = Rotate::new(rotate::RotateAngle::Rotate180);

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // Rotate 180°:
    // x' = img_w - x - w = 100 - 10 - 30 = 60
    // y' = img_h - y - h = 100 - 20 - 40 = 40
    assert_approx_eq!(result[0], 60.0, 0.01);
    assert_approx_eq!(result[1], 40.0, 0.01);
    assert_approx_eq!(result[2], 30.0, 0.01);
    assert_approx_eq!(result[3], 40.0, 0.01);
}

#[test]
fn test_rotate_270_bbox() {
    let transform = Rotate::new(rotate::RotateAngle::Rotate270);

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // Rotate 270° clockwise (or 90° CCW):
    // x' = y = 20
    // y' = img_w - x - w = 100 - 10 - 30 = 60
    // w' = h = 40
    // h' = w = 30
    assert_approx_eq!(result[0], 20.0, 0.01);
    assert_approx_eq!(result[1], 60.0, 0.01);
    assert_approx_eq!(result[2], 40.0, 0.01);
    assert_approx_eq!(result[3], 30.0, 0.01);
}

#[test]
fn test_rotate_90_point() {
    let transform = Rotate::new(rotate::RotateAngle::Rotate90);

    let point = (10.0, 20.0);
    let result = transform.map_point(point, (100, 100)).unwrap();

    // Rotate 90° clockwise:
    // x' = img_h - y = 100 - 20 = 80
    // y' = x = 10
    assert_approx_eq!(result.0, 80.0, 0.01);
    assert_approx_eq!(result.1, 10.0, 0.01);
}

// =============================================================================
// Crop Tests
// =============================================================================

#[test]
fn test_crop_bbox_inside() {
    let transform = Crop::new(10, 10, 80, 80); // Crop to (10, 10, 80, 80)

    let bbox = [20.0, 30.0, 30.0, 40.0]; // Box fully inside crop
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // After crop: subtract crop offset
    assert_approx_eq!(result[0], 10.0, 0.01); // x = 20 - 10
    assert_approx_eq!(result[1], 20.0, 0.01); // y = 30 - 10
    assert_approx_eq!(result[2], 30.0, 0.01); // w unchanged
    assert_approx_eq!(result[3], 40.0, 0.01); // h unchanged
}

#[test]
fn test_crop_bbox_outside() {
    let transform = Crop::new(10, 10, 80, 80);

    let bbox = [5.0, 5.0, 3.0, 3.0]; // Box fully outside crop (extends to 8,8)
    let result = transform.map_bbox(bbox, (100, 100));

    // Should be clipped (None)
    assert!(result.is_none());
}

#[test]
fn test_crop_bbox_partial_overlap() {
    let transform = Crop::new(10, 10, 80, 80);

    let bbox = [5.0, 10.0, 20.0, 30.0]; // Partially overlaps
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // Should be clipped to crop region
    // Original: x=5, w=20 -> right edge = 25
    // Crop starts at x=10
    // New box: x=0 (relative), w=15 (visible portion)
    assert_approx_eq!(result[0], 0.0, 0.01);
    assert_approx_eq!(result[2], 15.0, 0.01);
}

#[test]
fn test_crop_point_inside() {
    let transform = Crop::new(10, 10, 80, 80);

    let point = (20.0, 30.0);
    let result = transform.map_point(point, (100, 100)).unwrap();

    assert_approx_eq!(result.0, 10.0, 0.01); // x = 20 - 10
    assert_approx_eq!(result.1, 20.0, 0.01); // y = 30 - 10
}

#[test]
fn test_crop_point_outside() {
    let transform = Crop::new(10, 10, 80, 80);

    let point = (5.0, 5.0); // Outside crop
    let result = transform.map_point(point, (100, 100));

    assert!(result.is_none());
}

// =============================================================================
// Pad Tests
// =============================================================================

#[test]
fn test_pad_bbox() {
    let transform = Pad::new(10, 20, 5, 15, pad::PadMode::Constant(0));

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // After padding: add padding offsets
    // Original image was 100x100, padded to 120x135
    // New x = x + left_pad = 10 + 5 = 15
    // New y = y + top_pad = 20 + 10 = 30
    assert_approx_eq!(result[0], 15.0, 0.01);
    assert_approx_eq!(result[1], 30.0, 0.01);
    assert_approx_eq!(result[2], 30.0, 0.01); // w unchanged
    assert_approx_eq!(result[3], 40.0, 0.01); // h unchanged
}

#[test]
fn test_pad_point() {
    let transform = Pad::new(10, 20, 5, 15, pad::PadMode::Constant(0));

    let point = (10.0, 20.0);
    let result = transform.map_point(point, (100, 100)).unwrap();

    assert_approx_eq!(result.0, 15.0, 0.01); // x = 10 + 5
    assert_approx_eq!(result.1, 30.0, 0.01); // y = 20 + 10
}

// =============================================================================
// Resize Tests
// =============================================================================

#[test]
fn test_resize_bbox_upscale() {
    let transform = Resize::new(200, 200); // 2x upscale

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // Scale factor: 2.0
    assert_approx_eq!(result[0], 20.0, 0.01); // x * 2
    assert_approx_eq!(result[1], 40.0, 0.01); // y * 2
    assert_approx_eq!(result[2], 60.0, 0.01); // w * 2
    assert_approx_eq!(result[3], 80.0, 0.01); // h * 2
}

#[test]
fn test_resize_bbox_downscale() {
    let transform = Resize::new(50, 50); // 0.5x downscale

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // Scale factor: 0.5
    assert_approx_eq!(result[0], 5.0, 0.01); // x * 0.5
    assert_approx_eq!(result[1], 10.0, 0.01); // y * 0.5
    assert_approx_eq!(result[2], 15.0, 0.01); // w * 0.5
    assert_approx_eq!(result[3], 20.0, 0.01); // h * 0.5
}

#[test]
fn test_resize_point() {
    let transform = Resize::new(200, 150); // Different scales for x and y

    let point = (50.0, 40.0);
    let result = transform.map_point(point, (100, 100)).unwrap();

    // x scale: 2.0, y scale: 1.5
    assert_approx_eq!(result.0, 100.0, 0.01); // 50 * 2
    assert_approx_eq!(result.1, 60.0, 0.01); // 40 * 1.5
}

// =============================================================================
// Affine Tests
// =============================================================================

#[test]
fn test_affine_bbox_identity() {
    use crate::transforms::geometric::affine::{Affine, AffineParams};

    // Identity transform (no change)
    let params = AffineParams {
        scale: (1.0, 1.0),
        rotate: 0.0,
        translate: (0.0, 0.0),
        shear: (0.0, 0.0),
    };
    let transform = Affine::new(params);

    let bbox = [10.0, 20.0, 30.0, 40.0];
    let result = transform.map_bbox(bbox, (100, 100)).unwrap();

    // Should be approximately the same
    assert_approx_eq!(result[0], 10.0, 1.0);
    assert_approx_eq!(result[1], 20.0, 1.0);
    assert_approx_eq!(result[2], 30.0, 1.0);
    assert_approx_eq!(result[3], 40.0, 1.0);
}

#[test]
fn test_affine_point_translation() {
    use crate::transforms::geometric::affine::{Affine, AffineParams};

    let params = AffineParams {
        scale: (1.0, 1.0),
        rotate: 0.0,
        translate: (10.0, 20.0),
        shear: (0.0, 0.0),
    };
    let transform = Affine::new(params);

    let point = (50.0, 40.0);
    let result = transform.map_point(point, (100, 100)).unwrap();

    assert_approx_eq!(result.0, 60.0, 0.1); // 50 + 10
    assert_approx_eq!(result.1, 60.0, 0.1); // 40 + 20
}

// =============================================================================
// Multi-transform Tests
// =============================================================================

#[test]
fn test_flip_crop_combo() {
    // Test that coordinate transforms compose correctly
    let flip = HorizontalFlip;
    let crop = Crop::new(0, 0, 50, 50); // Crop to left half

    // Start with a box in the right half: x=60, y=20, w=30, h=40
    let bbox = [60.0, 20.0, 30.0, 40.0];

    // After flip: x moves to left side: x' = 100 - 60 - 30 = 10
    let after_flip = flip.map_bbox(bbox, (100, 100)).unwrap();

    // After crop: still in range, just offset by crop
    let after_crop = crop.map_bbox(after_flip, (100, 100));

    // Should not be None (box survives both transforms)
    assert!(after_crop.is_some());
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_size_bbox() {
        let transform = HorizontalFlip;
        let bbox = [50.0, 50.0, 0.0, 0.0]; // Degenerate box

        let _result = transform.map_bbox(bbox, (100, 100));
        // Zero-size boxes might be filtered or kept - depends on implementation
        // Just check it doesn't panic
    }

    #[test]
    fn test_point_on_boundary() {
        let transform = HorizontalFlip;
        let point = (0.0, 50.0); // On left edge

        let result = transform.map_point(point, (100, 100)).unwrap();
        assert_approx_eq!(result.0, 100.0, 0.01); // Should be on right edge
    }

    #[test]
    fn test_bbox_touching_edge() {
        let transform = HorizontalFlip;
        let bbox = [0.0, 20.0, 10.0, 30.0]; // Touches left edge

        let result = transform.map_bbox(bbox, (100, 100)).unwrap();
        assert_approx_eq!(result[0], 90.0, 0.01); // Should touch right edge
    }
}
