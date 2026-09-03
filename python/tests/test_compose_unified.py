
import numpy as np
import pytest

# Try to import sinter, skip if not available
pytest.importorskip("sinter")

from sinter import (
    Compose,
    HorizontalFlip,
    Resize,
    Brightness,
    Contrast,
    Gamma,
    Solarize,
    Posterize,
    ToSepia,
    ColorTemperature,
    HueSaturationValue,
    ColorTint,
    VerticalFlip,
)

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

    def test_long_pipeline_deterministic_and_bounded(self):
        """A long mixed pipeline (LUT + matrix + geometric) must be
        deterministic for a fixed seed, keep shape/dtype, and stay within a
        documented bound of sequential per-op application."""
        pipeline = Compose(
            [
                Brightness(delta=20),
                Contrast(factor=1.2),
                Gamma(gamma=0.9),
                Solarize(threshold=128),
                Posterize(bits=4),
                ToSepia(),
                ColorTemperature(temperature=50),
                HueSaturationValue(hue_shift=0, saturation_scale=1.3, value_scale=1.0),
                ColorTint(tint=(255, 200, 100, 0.5)),
                HorizontalFlip(),
                VerticalFlip(),
            ]
        )
        rng = np.random.default_rng(1234)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        a = pipeline.apply(img.copy())
        b = pipeline.apply(img.copy())
        # Deterministic for the same seed is not guaranteed via apply (random
        # seed each call), so use sample_with_seed for reproducibility.
        s1 = pipeline.sample_with_seed(7).apply(img.copy())
        s2 = pipeline.sample_with_seed(7).apply(img.copy())
        np.testing.assert_array_equal(s1, s2)

        assert s1.shape == img.shape and s1.dtype == np.uint8

        # Bounded divergence vs sequential application (matrix fusion clamps
        # once instead of per-op; LUT/geometric fusion are exact).
        seq = img.copy()
        ops_seq = [
            Brightness(delta=20),
            Contrast(factor=1.2),
            Gamma(gamma=0.9),
            Solarize(threshold=128),
            Posterize(bits=4),
            ToSepia(),
            ColorTemperature(temperature=50),
            HueSaturationValue(hue_shift=0, saturation_scale=1.3, value_scale=1.0),
            ColorTint(tint=(255, 200, 100, 0.5)),
            HorizontalFlip(),
            VerticalFlip(),
        ]
        for op in ops_seq:
            seq = Compose([op]).apply(seq)
        max_diff = int(np.abs(s1.astype(int) - seq.astype(int)).max())
        assert max_diff <= 32, f"long pipeline divergence exceeded bound: {max_diff}"

    def test_crop_hoisting_pointwise_equivalence(self):
        """Test that compiler hoists Crop before pointwise photometric ops bit-exactly."""
        from sinter import Crop

        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)

        # Pipeline with photometric ops followed by Crop
        pipe_color_then_crop = Compose([
            Brightness(delta=25),
            Contrast(factor=1.3),
            Crop(x=20, y=30, width=80, height=80),
        ])

        # Pipeline with Crop explicitly first
        pipe_crop_then_color = Compose([
            Crop(x=20, y=30, width=80, height=80),
            Brightness(delta=25),
            Contrast(factor=1.3),
        ])

        out_a = pipe_color_then_crop.apply(img)
        out_b = pipe_crop_then_color.apply(img)

        # Bit-exact equivalence
        np.testing.assert_array_equal(out_a, out_b)

        # Check execution plan: Crop is hoisted to Node 1!
        plan_str = pipe_color_then_crop.explain()
        assert "2 execution nodes" in plan_str
        assert "Node 1: Barrier" in plan_str
        assert "Node 2: Fused(2 ops)" in plan_str

    def test_random_crop_hoisting_multimodal(self):
        """Test that RandomCrop hoisting preserves multimodal labels (bboxes, mask)."""
        from sinter import RandomCrop

        rng = np.random.default_rng(99)
        img = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
        mask = rng.integers(0, 5, (300, 300), dtype=np.uint8)
        bboxes = np.array([[50.0, 50.0, 60.0, 60.0]], dtype=np.float32)

        pipe = Compose([
            Brightness(delta=20),
            Contrast(factor=1.1),
            RandomCrop(width=150, height=150),
        ])

        res = pipe(image=img, mask=mask, bboxes=bboxes, seed=42)
        assert res["image"].shape == (150, 150, 3)
        assert res["mask"].shape == (150, 150)
        assert len(res["bboxes"]) <= 1  # BBox may be cropped or kept

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
