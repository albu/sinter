import numpy as np
import pytest

pytest.importorskip("sinter")

from sinter import (
    Compose,
    SampledImageProgram,
    Brightness,
    Contrast,
    Gamma,
    Solarize,
    Posterize,
    ToSepia,
    ColorTemperature,
    HueSaturationValue,
    RGBShift,
    GaussNoise,
    Crop,
    ColorTint,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    Resize,
    Pad,
    Affine,
    Normalize,
    GaussianBlur,
    MedianBlur,
    ChannelShuffle,
    Emboss,
    EdgeDetection,
    CoarseDropout,
    GridDropout,
    Uniform,
    UniformInt,
    Bernoulli,
    Constant,
    Normal,
)


class TestSafeMemoryDefaults:
    """Test that safe copy-by-default is respected everywhere unless inplace=True is requested."""

    def test_compose_apply_safe_default(self):
        pipeline = Compose([Brightness(delta=50)])
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        original = img.copy()

        result = pipeline.apply(img)

        # Original array MUST NOT be mutated!
        np.testing.assert_array_equal(img, original)
        assert result is not img
        assert result[0, 0, 0] == 150

    def test_compose_apply_inplace_opt_in(self):
        pipeline = Compose([Brightness(delta=50)])
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100

        result = pipeline.apply(img, inplace=True)

        # Original array WAS mutated in place!
        assert img[0, 0, 0] == 150
        assert result is img

    def test_compose_call_safe_default(self):
        pipeline = Compose([Brightness(delta=50)])
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        original = img.copy()

        res_dict = pipeline(img)

        np.testing.assert_array_equal(img, original)
        assert res_dict["image"] is not img
        assert res_dict["image"][0, 0, 0] == 150

    def test_compose_call_inplace_opt_in(self):
        pipeline = Compose([Brightness(delta=50)])
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100

        res_dict = pipeline(img, inplace=True)

        assert img[0, 0, 0] == 150
        assert res_dict["image"] is img

    def test_sampled_program_safe_default(self):
        pipeline = Compose([Brightness(delta=50)])
        sampled = pipeline.sample_with_seed(42)

        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        original = img.copy()

        result = sampled.apply(img)
        np.testing.assert_array_equal(img, original)
        assert result is not img
        assert result[0, 0, 0] == 150

        # Now in-place
        res_inplace = sampled.apply(img, inplace=True)
        assert img[0, 0, 0] == 150
        assert res_inplace is img

    def test_individual_transform_safe_default(self):
        t = Brightness(delta=50)
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        original = img.copy()

        # Calling individual transform on image returns transformed array directly
        res1 = t(img)
        np.testing.assert_array_equal(img, original)
        assert res1[0, 0, 0] == 150

        res2 = t.apply(img)
        np.testing.assert_array_equal(img, original)
        assert res2[0, 0, 0] == 150

        res3 = t.apply(img, inplace=True)
        assert img[0, 0, 0] == 150
        assert res3 is img


class TestCallableTransformsAndDefaults:
    """Test that individual transforms can be constructed with sensible defaults and called directly."""

    def test_default_constructors(self):
        # All transforms should instantiate cleanly without arguments
        transforms = [
            HorizontalFlip(),
            VerticalFlip(),
            Brightness(),
            Contrast(),
            Gamma(),
            Solarize(),
            Posterize(),
            ToSepia(),
            ColorTemperature(),
            HueSaturationValue(),
            ColorTint(),
            Rotate(),
            Pad(),
            Affine(),
            Normalize(),
            GaussianBlur(),
            MedianBlur(),
            ChannelShuffle(),
            Emboss(),
            EdgeDetection(),
            CoarseDropout(),
            GridDropout(),
        ]
        assert len(transforms) == 22
        for t in transforms:
            assert "<dist>" not in repr(t)

    def test_direct_transform_calling(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :10] = 255

        # Call individual transform with () -> returns array directly
        res_arr = HorizontalFlip()(img)
        assert isinstance(res_arr, np.ndarray)
        assert np.all(res_arr[:, -10:] == 255)

        # Call with targets -> returns dict
        res_dict = HorizontalFlip()(img, bboxes=np.array([[10, 20, 30, 40]], dtype=np.float32))
        assert isinstance(res_dict, dict)
        assert "image" in res_dict
        assert "bboxes" in res_dict

        # Call with .apply()
        res_img = HorizontalFlip().apply(img)
        assert isinstance(res_img, np.ndarray)
        assert np.all(res_img[:, -10:] == 255)


class TestFlexibleEnumAndDistributionParameters:
    """Test string/int literals for enums and tuple sugar for distributions."""

    def test_rotate_angle_literals(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        for angle in [90, 180, 270, "90", "180", "270"]:
            t = Rotate(angle=angle)
            res = t.apply(img)
            if str(angle) in ["90", "270"]:
                assert res.shape == (200, 100, 3)
            else:
                assert res.shape == (100, 200, 3)

    def test_resize_interpolation_literals(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for interp in ["nearest", "bilinear", "bicubic", "lanczos4"]:
            res = Resize(50, 50, interpolation=interp).apply(img)
            assert res.shape == (50, 50, 3)

    def test_pad_mode_literals(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 42
        for mode in ["reflect", "replicate", "wrap", "constant"]:
            res = Pad(10, 10, 10, 10, mode=mode).apply(img)
            assert res.shape == (70, 70, 3)

    def test_emboss_direction_literals(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        for direction in ["top_left", "top", "top_right", "right", "bottom_right", "bottom", "bottom_left", "left"]:
            res = Emboss(direction=direction).apply(img)
            assert res.shape == (50, 50, 3)

    def test_edge_detection_method_literals(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        for method in ["sobel", "prewitt", "laplacian", "canny"]:
            res = EdgeDetection(method=method).apply(img)
            assert res.shape == (50, 50, 3)

    def test_channel_shuffle_literals(self):
        img = np.ones((50, 50, 3), dtype=np.uint8)
        img[:, :, 0] = 10
        img[:, :, 1] = 20
        img[:, :, 2] = 30

        res = ChannelShuffle(order="BGR").apply(img)
        assert res[0, 0, 0] == 30
        assert res[0, 0, 1] == 20
        assert res[0, 0, 2] == 10

    def test_tuple_distribution_sugar(self):
        # (min, max) tuple should implicitly create Uniform(min, max)
        t = Brightness(delta=(-30, 30))
        assert "Uniform(-30, 30)" in repr(t)

        pipeline = Compose([
            Brightness(delta=(-20, 20)),
            Contrast(factor=(0.8, 1.2)),
        ])
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        res = pipeline.apply(img)
        assert res.shape == img.shape


class TestMaskGeometricIntegrity:
    """Test that masks only undergo geometric transformations (with nearest-neighbor)
    and ignore photometric/noise ops completely."""

    def test_masks_ignore_photometric_ops(self):
        pipeline = Compose([
            Brightness(delta=100),
            Contrast(factor=2.0),
            Gamma(gamma=0.5),
            Normalize(mean=128, std=50),
        ])

        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        mask = np.repeat(np.repeat(mask, 25, axis=0), 25, axis=1)  # 50x50

        original_mask = mask.copy()

        res = pipeline(img, masks=mask)

        # Image was transformed by photometric ops
        assert res["image"][0, 0, 0] != 100

        # Mask MUST remain 100% identical!
        np.testing.assert_array_equal(res["masks"], original_mask)

    def test_masks_follow_geometric_ops(self):
        pipeline = Compose([
            Brightness(delta=100),  # photometric op (ignored by mask)
            HorizontalFlip(),       # geometric op (applied to mask)
        ])

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[:, :10] = 7  # class label 7 in left strip

        res = pipeline(img, masks=mask)

        # Mask was horizontally flipped
        assert np.all(res["masks"][:, -10:] == 7)
        assert np.all(res["masks"][:, :10] == 0)


class TestExtraColumnBoundingBoxes:
    """Test that bounding boxes with extra metadata columns (such as class_id, confidence)
    are preserved without shape desync."""

    def test_5_column_bboxes_passthrough(self):
        pipeline = Compose([HorizontalFlip()])
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # [x, y, w, h, class_id]
        bboxes = np.array([
            [10, 20, 30, 40, 5],
            [60, 10, 20, 30, 9],
        ], dtype=np.float32)

        res = pipeline(img, bboxes=bboxes)
        out_boxes = res["bboxes"]

        assert out_boxes.shape == (2, 5)
        # Check coordinates flipped: 100 - 10 - 30 = 60
        assert out_boxes[0, 0] == 60
        assert out_boxes[0, 1] == 20
        assert out_boxes[0, 2] == 30
        assert out_boxes[0, 3] == 40
        assert out_boxes[0, 4] == 5  # class_id preserved!

        # Second box: 100 - 60 - 20 = 20
        assert out_boxes[1, 0] == 20
        assert out_boxes[1, 1] == 10
        assert out_boxes[1, 2] == 20
        assert out_boxes[1, 3] == 30
        assert out_boxes[1, 4] == 9  # class_id preserved!

    def test_6_column_bboxes_passthrough(self):
        pipeline = Compose([HorizontalFlip()])
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # [x, y, w, h, class_id, confidence]
        bboxes = np.array([
            [10, 20, 30, 40, 3, 0.95],
        ], dtype=np.float32)

        res = pipeline(img, bboxes=bboxes)
        out_boxes = res["bboxes"]

        assert out_boxes.shape == (1, 6)
        assert out_boxes[0, 0] == 60
        assert out_boxes[0, 4] == 3
        assert np.isclose(out_boxes[0, 5], 0.95)


class TestContainerProtocol:
    """Test Compose and SampledImageProgram container operations and human ergonomics."""

    def test_len_and_repr(self):
        pipeline = Compose([
            Brightness(),
            Contrast(),
            HorizontalFlip(),
        ])
        assert len(pipeline) == 3
        r = repr(pipeline)
        assert "Brightness" in r
        assert "Contrast" in r
        assert "HorizontalFlip" in r

    def test_indexing_and_iteration(self):
        t1 = Brightness(delta=10)
        t2 = Contrast(factor=1.2)
        t3 = HorizontalFlip()
        pipeline = Compose([t1, t2, t3])

        assert len(pipeline) == 3
        assert "Brightness" in repr(pipeline[0])
        assert "Contrast" in repr(pipeline[1])
        assert "HorizontalFlip" in repr(pipeline[-1])

        collected = [repr(t) for t in pipeline]
        assert len(collected) == 3

    def test_direct_explain_and_summary(self):
        pipeline = Compose([
            Brightness(delta=10),
            Contrast(factor=1.2),
            HorizontalFlip(),
        ])
        explanation = pipeline.explain()
        assert "Execution Plan:" in explanation
        assert len(pipeline.summary()) > 0

    def test_mask_singular_alias(self):
        pipeline = Compose([HorizontalFlip()])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1

        # Calling with mask (singular) returns dict with 'mask'
        res = pipeline(image=img, mask=mask)
        assert "mask" in res
        assert "masks" not in res
        assert res["mask"].shape == (100, 100)

        # Calling with masks (plural) returns dict with 'masks'
        res_plural = pipeline(image=img, masks=mask)
        assert "masks" in res_plural
        assert "mask" not in res_plural

    def test_bbox_aliases(self):
        pipeline = Compose([HorizontalFlip()])
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Absolute formats
        for fmt in ["xywh", "coco", "xyxy", "pascal_voc"]:
            res = pipeline(img, bboxes=[[10.0, 20.0, 30.0, 40.0]], bbox_format=fmt)
            assert "bboxes" in res
            assert len(res["bboxes"]) == 1

        # Center format
        res = pipeline(img, bboxes=[[50.0, 50.0, 30.0, 40.0]], bbox_format="cxcywh")
        assert "bboxes" in res
        assert len(res["bboxes"]) == 1

        # Relative formats in [0, 1]
        for fmt in ["rel_xyxy", "albumentations", "rel_cxcywh", "yolo", "rel_xywh"]:
            res = pipeline(img, bboxes=[[0.5, 0.5, 0.2, 0.2]], bbox_format=fmt)
            assert "bboxes" in res
            assert len(res["bboxes"]) == 1

    def test_python_list_and_empty_bboxes_keypoints(self):
        pipeline = Compose([HorizontalFlip()])
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Python list bboxes
        res = pipeline(img, bboxes=[[10, 20, 30, 40]], bbox_format="xywh")
        assert isinstance(res["bboxes"], list)
        assert res["bboxes"][0][0] == 60.0

        # Empty python list bboxes
        res_empty = pipeline(img, bboxes=[], keypoints=[])
        assert isinstance(res_empty["bboxes"], list)
        assert len(res_empty["bboxes"]) == 0
        assert isinstance(res_empty["keypoints"], list)
        assert len(res_empty["keypoints"]) == 0

    def test_pipeline_slicing_and_concatenation(self):
        t1 = Brightness(delta=10)
        t2 = Contrast(factor=1.2)
        t3 = HorizontalFlip()
        t4 = VerticalFlip()
        pipeline = Compose([t1, t2, t3, t4])

        # Slicing
        sub = pipeline[1:3]
        assert isinstance(sub, Compose)
        assert len(sub) == 2
        assert "Contrast" in repr(sub[0])
        assert "HorizontalFlip" in repr(sub[1])

        # Addition with list
        p_added = sub + [Brightness(delta=20)]
        assert isinstance(p_added, Compose)
        assert len(p_added) == 3

        # Addition with Compose
        p_merged = sub + Compose([t4])
        assert isinstance(p_merged, Compose)
        assert len(p_merged) == 3

        # Right addition with list
        p_radded = [t1] + sub
        assert isinstance(p_radded, Compose)
        assert len(p_radded) == 3

        # Direct sample() without explicit seed
        sampled = p_merged.sample()
        assert sampled.len() == 3

    def test_flexible_transform_constructors(self):
        # HueSaturationValue with various alias formats
        hsv = HueSaturationValue(hue_shift=(-20, 20), sat_shift=(-30, 30), val_shift=(-20, 20))
        assert "HueSaturationValue" in repr(hsv)

        # RGBShift with limits
        rgb = RGBShift(r_shift_limit=20, g_shift_limit=(-10, 10))
        assert "RGBShift" in repr(rgb)

        # GaussNoise with var_limit
        gn = GaussNoise(var_limit=(10, 50))
        assert "GaussNoise" in repr(gn)

        # Crop with x_min/y_min/x_max/y_max
        crop = Crop(x_min=10, y_min=10, x_max=90, y_max=90)
        assert "Crop" in repr(crop)
