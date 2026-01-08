
import numpy as np
import pytest

# Try to import sinter, skip if not available
pytest.importorskip("sinter")

from sinter import Compose, HorizontalFlip, Resize

class TestComposeUnified:
    """Test the Unified Compose.__call__ API."""

    def test_call_image_only(self):
        """Test calling compose with just an image."""
        pipeline = Compose([HorizontalFlip()])
        
        # 100x100 RGB image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :10] = 255  # Left strip white

        # Call __call__ implicitly
        result = pipeline(img)

        # Should return a dictionary
        assert isinstance(result, dict)
        assert "image" in result
        assert "bboxes" not in result
        assert "keypoints" not in result

        # Check image content
        result_img = result["image"]
        assert result_img.shape == (100, 100, 3)
        assert np.all(result_img[:, -10:] == 255)  # Strip moved to right

    def test_call_image_and_bboxes(self):
        """Test calling compose with image and bboxes."""
        pipeline = Compose([HorizontalFlip()])
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # xywh format: [x, y, w, h]
        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)

        # Call with bboxes
        result = pipeline(img, bboxes=bboxes)

        assert "image" in result
        assert "bboxes" in result
        assert "keypoints" not in result

        # Check bbox result
        # x' = 100 - (10 + 30) = 60
        expected_bbox = np.array([[60, 20, 30, 40]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result["bboxes"], expected_bbox)

    def test_call_image_and_keypoints(self):
        """Test calling compose with image and keypoints."""
        pipeline = Compose([HorizontalFlip()])
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        keypoints = np.array([[10, 20]], dtype=np.float32)

        # Call with keypoints
        result = pipeline(img, keypoints=keypoints)

        assert "image" in result
        assert "bboxes" not in result
        assert "keypoints" in result

        # Check keypoint result
        # x' = 100 - 10 = 90
        expected_kpt = np.array([[90, 20]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result["keypoints"], expected_kpt)

    def test_call_all_targets(self):
        """Test calling compose with everything."""
        pipeline = Compose([HorizontalFlip()])
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        keypoints = np.array([[10, 20]], dtype=np.float32)

        result = pipeline(
            img, 
            bboxes=bboxes, 
            keypoints=keypoints,
            bbox_format="xywh",
            keypoint_format="xy"
        )

        assert "image" in result
        assert "bboxes" in result
        assert "keypoints" in result

        # Verify all transformations happened correctly (same random state used)
        
        # Image flipped
        # Bbox flipped to x=60
        expected_bbox = np.array([[60, 20, 30, 40]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result["bboxes"], expected_bbox)
        
        # Keypoint flipped to x=90
        expected_kpt = np.array([[90, 20]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result["keypoints"], expected_kpt)

    def test_custom_formats(self):
        """Test custom formats in __call__."""
        pipeline = Compose([Resize(width=200, height=200)])
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Normalized bbox: [0.1, 0.2, 0.3, 0.4] -> 2x upscale -> same normalized coords
        bboxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        
        result = pipeline(
            img, 
            bboxes=bboxes, 
            bbox_format="rel_xywh"
        )
        
        # Normalized coords should be preserved (logic relies on image size,
        # but 2x upscale on normalized coords means they stay same relative to new size)
        # Wait, Sinter internally converts to absolute, transforms, then back.
        # So 0.1 * 100 = 10px. Resize 2x -> 20px. 20px / 200 = 0.1. Correct.
        np.testing.assert_array_almost_equal(result["bboxes"], bboxes)
        assert result["image"].shape == (200, 200, 3)

    def test_call_with_masks_2d(self):
        """Test calling compose with 2D masks."""
        pipeline = Compose([HorizontalFlip()])

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # 2D mask (H, W)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 10:50] = 1  # Left-side rectangle

        result = pipeline(img, masks=mask)

        assert "image" in result
        assert "masks" in result

        # Check mask was flipped horizontally
        result_mask = result["masks"]
        assert result_mask.shape == (100, 100)
        # After flip: rectangle should be on the right side
        # Original: x=10:50, width=40. After flip: x'=100-(10+40)=50:90
        assert np.all(result_mask[20:60, 50:90] == 1)

    def test_call_with_all_targets_including_masks(self):
        """Test calling compose with image, bboxes, keypoints, and masks."""
        pipeline = Compose([HorizontalFlip()])

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bboxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        keypoints = np.array([[10, 20]], dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 10:50] = 1

        result = pipeline(
            img,
            bboxes=bboxes,
            keypoints=keypoints,
            masks=mask,
            seed=42  # Deterministic
        )

        assert "image" in result
        assert "bboxes" in result
        assert "keypoints" in result
        assert "masks" in result

        # All should be transformed with the same random state
        expected_bbox = np.array([[60, 20, 30, 40]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result["bboxes"], expected_bbox)

        expected_kpt = np.array([[90, 20]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result["keypoints"], expected_kpt)

        # Mask flipped
        assert np.all(result["masks"][20:60, 50:90] == 1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
