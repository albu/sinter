"""
Integration tests for bounding box and keypoint transformations.

Tests that spatial labels are correctly transformed alongside images.
"""

import numpy as np
import pytest

# Try to import sinter, skip if not available
pytest.importorskip("sinter")

from sinter import Compose, HorizontalFlip, VerticalFlip, Rotate, RotateAngle, Resize, Crop


class TestBBoxTransforms:
    """Test bounding box transformations."""

    def test_horizontal_flip_bbox_xywh(self):
        """Test horizontal flip with xywh format."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Box at x=10, y=20, w=30, h=40 in 100x100 image
        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        # After flip: x' = 100 - (10 + 30) = 60
        expected = np.array([[60, 20, 30, 40]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_horizontal_flip_bbox_xyxy(self):
        """Test horizontal flip with xyxy format."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Box in xyxy: [x_min, y_min, x_max, y_max]
        bboxes = np.array([[10, 20, 50, 60]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100), format="xyxy")

        # After flip: x_min' = 100 - 50 = 50, x_max' = 100 - 10 = 90
        expected = np.array([[50, 20, 90, 60]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_horizontal_flip_bbox_normalized(self):
        """Test horizontal flip with normalized coordinates."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Normalized bbox [x, y, w, h] in [0, 1]
        bboxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        result = sampled.apply_to_bboxes(
            bboxes, (100, 100), format="rel_xywh", format_out="rel_xywh"
        )

        # After flip: x' = 1.0 - (0.1 + 0.3) = 0.6
        expected = np.array([[0.6, 0.2, 0.3, 0.4]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_vertical_flip_bbox(self):
        """Test vertical flip."""
        pipeline = Compose([VerticalFlip()])
        sampled = pipeline.sample_with_seed(42)

        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        # After flip: y' = 100 - (20 + 40) = 40
        expected = np.array([[10, 40, 30, 40]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate_90_bbox(self):
        """Test 90-degree rotation."""
        pipeline = Compose([Rotate(angle=RotateAngle.ROTATE_90)])
        sampled = pipeline.sample_with_seed(42)

        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        # Rotate 90° clockwise: x and y swap, w and h swap
        # x' = 100 - 20 - 40 = 40, y' = 10
        # w' = 40, h' = 30
        expected = np.array([[40, 10, 40, 30]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_resize_bbox_upscale(self):
        """Test resize with upscale."""
        pipeline = Compose([Resize(width=200, height=200)])
        sampled = pipeline.sample_with_seed(42)

        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        # 2x upscale
        expected = np.array([[20, 40, 60, 80]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_crop_bbox_inside(self):
        """Test crop with bbox fully inside crop region."""
        pipeline = Compose([Crop(x=10, y=10, width=80, height=80)])
        sampled = pipeline.sample_with_seed(42)

        # Box inside crop region
        bboxes = np.array([[20, 30, 30, 40]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        # After crop: subtract crop offset
        expected = np.array([[10, 20, 30, 40]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_crop_bbox_outside(self):
        """Test crop with bbox fully outside crop region."""
        pipeline = Compose([Crop(x=10, y=10, width=80, height=80)])
        sampled = pipeline.sample_with_seed(42)

        # Box completely outside crop region (ends before crop starts)
        # Box spans (0,0) to (5,5), crop starts at (10,10)
        bboxes = np.array([[0, 0, 5, 5]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        # Should be filtered out (empty array)
        assert len(result) == 0

    def test_multiple_bboxes(self):
        """Test multiple bboxes."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        bboxes = np.array([
            [10, 20, 30, 40],
            [50, 60, 20, 30],
            [0, 0, 10, 10],
        ], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        expected = np.array([
            [60, 20, 30, 40],   # 100 - (10 + 30) = 60
            [30, 60, 20, 30],   # 100 - (50 + 20) = 30
            [90, 0, 10, 10],    # 100 - (0 + 10) = 90
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_format_conversion_xyxy_to_xywh(self):
        """Test format conversion from xyxy to xywh."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Input in xyxy format
        bboxes = np.array([[10, 20, 50, 60]], dtype=np.float32)
        result = sampled.apply_to_bboxes(
            bboxes, (100, 100), format="xyxy", format_out="xywh"
        )

        # Should flip and convert to xywh
        # Flip: [50, 20, 90, 60] in xyxy
        # Convert: [50, 20, 40, 40] in xywh
        expected = np.array([[50, 20, 40, 40]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestKeypointTransforms:
    """Test keypoint transformations."""

    def test_horizontal_flip_keypoint(self):
        """Test horizontal flip with keypoints."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        keypoints = np.array([[10, 20], [50, 60]], dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100))

        # After flip: x' = 100 - x
        expected = np.array([[90, 20], [50, 60]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_horizontal_flip_keypoint_with_visibility(self):
        """Test horizontal flip with visibility."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Format: [x, y, visibility]
        keypoints = np.array([[10, 20, 2], [50, 60, 1]], dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100), format="xyv")

        # After flip: x coordinates change, visibility preserved
        expected = np.array([[90, 20, 2], [50, 60, 1]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_vertical_flip_keypoint(self):
        """Test vertical flip."""
        pipeline = Compose([VerticalFlip()])
        sampled = pipeline.sample_with_seed(42)

        keypoints = np.array([[10, 20]], dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100))

        # After flip: y' = 100 - y
        expected = np.array([[10, 80]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate_90_keypoint(self):
        """Test 90-degree rotation."""
        pipeline = Compose([Rotate(angle=RotateAngle.ROTATE_90)])
        sampled = pipeline.sample_with_seed(42)

        keypoints = np.array([[10, 20]], dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100))

        # Rotate 90°: x' = 100 - y = 80, y' = x = 10
        expected = np.array([[80, 10]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_resize_keypoint(self):
        """Test resize."""
        pipeline = Compose([Resize(width=200, height=200)])
        sampled = pipeline.sample_with_seed(42)

        keypoints = np.array([[50, 40]], dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100))

        # 2x upscale
        expected = np.array([[100, 80]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_crop_keypoint_inside(self):
        """Test crop with keypoint inside."""
        pipeline = Compose([Crop(x=10, y=10, width=80, height=80)])
        sampled = pipeline.sample_with_seed(42)

        keypoints = np.array([[20, 30]], dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100))

        # After crop: subtract offset
        expected = np.array([[10, 20]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_crop_keypoint_outside(self):
        """Test crop with keypoint outside."""
        pipeline = Compose([Crop(x=10, y=10, width=80, height=80)])
        sampled = pipeline.sample_with_seed(42)

        keypoints = np.array([[5, 5]], dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100))

        # Should be filtered out
        assert len(result) == 0

    def test_normalized_keypoint(self):
        """Test normalized coordinates."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Normalized [x, y]
        keypoints = np.array([[0.1, 0.2]], dtype=np.float32)
        result = sampled.apply_to_keypoints(
            keypoints, (100, 100), format="rel_xy", format_out="rel_xy"
        )

        # After flip: x' = 1.0 - 0.1 = 0.9
        expected = np.array([[0.9, 0.2]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_keypoint_default_visibility(self):
        """Test that xy format gets default visibility=2 when converting to xyv."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Input without visibility
        keypoints = np.array([[10, 20]], dtype=np.float32)
        result = sampled.apply_to_keypoints(
            keypoints, (100, 100), format="xy", format_out="xyv"
        )

        # Should have visibility=2 (visible)
        expected = np.array([[90, 20, 2]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestMultiTransform:
    """Test multiple transforms in sequence."""

    def test_flip_then_resize(self):
        """Test flip followed by resize."""
        pipeline = Compose([
            HorizontalFlip(),
            Resize(width=200, height=200),
        ])
        sampled = pipeline.sample_with_seed(42)

        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        # Flip: x' = 100 - 40 = 60
        # Resize: x'' = 60 * 2 = 120
        #        y'' = 20 * 2 = 40
        #        w'' = 30 * 2 = 60
        #        h'' = 40 * 2 = 80
        expected = np.array([[120, 40, 60, 80]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_combined_image_and_labels(self):
        """Test that image and labels transform consistently."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Create test image with a marker at x=10
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :10] = 255  # Left strip white

        # Apply to image
        result_img = sampled.apply(img.copy())

        # White strip should now be on the right
        assert np.all(result_img[:, -10:] == 255)
        assert np.all(result_img[:, :-10] == 0)

        # Bbox should also flip
        bboxes = np.array([[0, 0, 10, 100]], dtype=np.float32)
        result_bboxes = sampled.apply_to_bboxes(bboxes, (100, 100))

        # x' = 100 - 10 = 90
        expected = np.array([[90, 0, 10, 100]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result_bboxes, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_bboxes(self):
        """Test with empty bbox array."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        bboxes = np.zeros((0, 4), dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        assert len(result) == 0

    def test_empty_keypoints(self):
        """Test with empty keypoint array."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        keypoints = np.zeros((0, 2), dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100))

        assert len(result) == 0

    def test_invalid_format_string(self):
        """Test invalid format string raises error."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)

        with pytest.raises(ValueError, match="Unknown bbox format"):
            sampled.apply_to_bboxes(bboxes, (100, 100), format="invalid")

    def test_bbox_on_boundary(self):
        """Test bbox touching image boundary."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Box touching left edge
        bboxes = np.array([[0, 20, 10, 30]], dtype=np.float32)
        result = sampled.apply_to_bboxes(bboxes, (100, 100))

        # Should end up touching right edge
        expected = np.array([[90, 20, 10, 30]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_point_on_boundary(self):
        """Test keypoint on image boundary."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Point on left edge
        keypoints = np.array([[0, 50]], dtype=np.float32)
        result = sampled.apply_to_keypoints(keypoints, (100, 100))

        # Should end up on right edge
        expected = np.array([[100, 50]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestLabelTransforms:
    """Test classification label transformations."""

    def test_labels_pass_through_flip(self):
        """Test that labels pass through flip unchanged."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        labels = np.array([0, 1, 2], dtype=np.int32)
        result = sampled.apply_to_labels(labels, (100, 100))

        np.testing.assert_array_equal(result, labels)

    def test_labels_pass_through_rotate(self):
        """Test that labels pass through rotate unchanged."""
        pipeline = Compose([Rotate(angle=RotateAngle.ROTATE_90)])
        sampled = pipeline.sample_with_seed(42)

        labels = np.array([5, 10, 15], dtype=np.int32)
        result = sampled.apply_to_labels(labels, (100, 100))

        np.testing.assert_array_equal(result, labels)

    def test_labels_pass_through_resize(self):
        """Test that labels pass through resize unchanged."""
        pipeline = Compose([Resize(width=200, height=200)])
        sampled = pipeline.sample_with_seed(42)

        labels = np.array([1, 2, 3], dtype=np.int32)
        result = sampled.apply_to_labels(labels, (100, 100))

        np.testing.assert_array_equal(result, labels)

    def test_labels_multiple_transforms(self):
        """Test that labels pass through multiple transforms unchanged."""
        pipeline = Compose([
            HorizontalFlip(),
            VerticalFlip(),
            Resize(width=150, height=150),
        ])
        sampled = pipeline.sample_with_seed(42)

        labels = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        result = sampled.apply_to_labels(labels, (100, 100))

        np.testing.assert_array_equal(result, labels)

    def test_empty_labels(self):
        """Test with empty label array."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        labels = np.zeros(0, dtype=np.int32)
        result = sampled.apply_to_labels(labels, (100, 100))

        assert len(result) == 0


class TestMaskTransforms:
    """Test segmentation mask transformations."""

    def test_mask_horizontal_flip(self):
        """Test horizontal flip with 2D mask."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Create a mask with a white square on the left
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, :10] = 255

        result = sampled.apply_to_masks(mask, (100, 100))

        # White strip should now be on the right
        assert np.all(result[:, -10:] == 255)
        assert np.all(result[:, :-10] == 0)

    def test_mask_vertical_flip(self):
        """Test vertical flip with 2D mask."""
        pipeline = Compose([VerticalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Create a mask with a white square on top
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:10, :] = 255

        result = sampled.apply_to_masks(mask, (100, 100))

        # White strip should now be on the bottom
        assert np.all(result[-10:, :] == 255)
        assert np.all(result[:-10, :] == 0)

    def test_mask_resize_upscale(self):
        """Test resize upscale with 2D mask."""
        pipeline = Compose([Resize(width=200, height=200)])
        sampled = pipeline.sample_with_seed(42)

        # Create a mask with a small square
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:50] = 128

        result = sampled.apply_to_masks(mask, (100, 100))

        # Result should be 200x200
        assert result.shape == (200, 200)
        # The square should be roughly doubled in size
        # (check center region)
        assert np.any(result[40:80, 60:100] > 0)

    def test_mask_crop(self):
        """Test crop with 2D mask."""
        pipeline = Compose([Crop(x=10, y=10, width=80, height=80)])
        sampled = pipeline.sample_with_seed(42)

        # Create a mask with a value inside the crop region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30, 40] = 100

        result = sampled.apply_to_masks(mask, (100, 100))

        # Result should be 80x80
        assert result.shape == (80, 80)
        # The point should be at (20, 30) after crop
        assert result[20, 30] == 100

    def test_mask_3d_horizontal_flip(self):
        """Test horizontal flip with 3D mask (H, W, 1)."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Create a 3D mask with a white square on the left
        mask = np.zeros((100, 100, 1), dtype=np.uint8)
        mask[:, :10] = 255

        result = sampled.apply_to_masks(mask, (100, 100))

        # White strip should now be on the right
        assert result.shape == (100, 100, 1)
        assert np.all(result[:, -10:] == 255)
        assert np.all(result[:, :-10] == 0)

    def test_mask_multiple_transforms(self):
        """Test multiple transforms with mask."""
        pipeline = Compose([
            HorizontalFlip(),
            Resize(width=150, height=150),
        ])
        sampled = pipeline.sample_with_seed(42)

        # Create a mask with a marker on the left
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, :5] = 200

        result = sampled.apply_to_masks(mask, (100, 100))

        # Result should be 150x150 with marker on the right
        assert result.shape == (150, 150)
        assert np.any(result[:, -10:] > 0)

    def test_mask_binary_labels(self):
        """Test mask with binary segmentation labels."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Create a binary mask (0 = background, 1 = foreground)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 10:50] = 1

        result = sampled.apply_to_masks(mask, (100, 100))

        # The foreground region should be flipped horizontally
        assert np.any(result[20:60, 50:90] == 1)

    def test_mask_multiclass_labels(self):
        """Test mask with multiple class labels."""
        pipeline = Compose([HorizontalFlip()])
        sampled = pipeline.sample_with_seed(42)

        # Create a multi-class mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # Class 1
        mask[40:60, 10:30] = 2  # Class 2
        mask[70:90, 10:30] = 3  # Class 3

        result = sampled.apply_to_masks(mask, (100, 100))

        # Classes should be preserved but flipped horizontally
        assert np.any(result[10:30, 70:90] == 1)
        assert np.any(result[40:60, 70:90] == 2)
        assert np.any(result[70:90, 70:90] == 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
