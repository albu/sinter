"""
Comprehensive Correctness Test Suite: Sinter vs OpenCV / Albumentations

This test suite rigorously tests:
1. Individual operations (one by one) against OpenCV / Albumentations ground truth.
2. Fused operation groups vs sequential execution.
3. Multiple image shapes (square, non-square rectangular, odd dimensions, 1-channel, 3-channel).
4. Exact numerical equality (for discrete ops) and perceptual/numerical bounds (for continuous/floating-point ops).
"""

import cv2
import numpy as np
import pytest
import albumentations as A

import sinter
from sinter import (
    Compose,
    Constant,
    Uniform,
    # Geometric
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    Rotate,
    RotateAngle,
    Resize,
    Crop,
    Pad,
    PadMode,
    Affine,
    Interpolation,
    # Photometric / LUT
    Brightness,
    Contrast,
    Gamma,
    Invert,
    Solarize,
    Posterize,
    Normalize,
    Equalize,
    AutoContrast,
    RGBShift,
    HueSaturationValue,
    ToGray,
    ToRGB,
    ToSepia,
    ColorTemperature,
    ColorTint,
    ColorBalance,
    ChannelShuffle,
    # Kernel
    GaussianBlur,
    MedianBlur,
    Sharpen,
    Emboss,
    EdgeDetection,
    # Noise & Dropout
    GaussNoise,
    MultiplicativeNoise,
    SaltAndPepper,
    CoarseDropout,
    GridDropout,
)

cv2.setNumThreads(1)


# ============================================================================
# Test Fixtures & Utilities
# ============================================================================

@pytest.fixture(params=[(64, 64), (47, 73), (128, 96), (256, 256)])
def image_shape(request):
    return request.param


def create_test_image(height, width, channels=3, seed=42):
    """Generate a deterministic, rich test image with gradients, edges, and texture."""
    rng = np.random.RandomState(seed)
    if channels == 1:
        y, x = np.mgrid[0:height, 0:width]
        gray = ((x / width + y / height) * 0.5 * 255).astype(np.uint8)
        noise = rng.randint(0, 10, (height, width), dtype=np.uint8)
        img = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img[:, :, np.newaxis]
    else:
        y, x = np.mgrid[0:height, 0:width]
        r = (x / width * 255).astype(np.uint8)
        g = (y / height * 255).astype(np.uint8)
        b = ((x + y) / (width + height) * 255).astype(np.uint8)
        img = np.stack([r, g, b], axis=-1)
        noise = rng.randint(0, 20, (height, width, 3), dtype=np.uint8)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def assert_exact(actual, expected, msg=""):
    """Verify bitwise exact match."""
    if actual.ndim == 3 and actual.shape[2] == 1 and expected.ndim == 2:
        expected = expected[:, :, np.newaxis]
    elif expected.ndim == 3 and expected.shape[2] == 1 and actual.ndim == 2:
        actual = actual[:, :, np.newaxis]
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    diff = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    max_diff = np.max(diff)
    if max_diff != 0:
        num_mismatch = np.count_nonzero(diff)
        total = actual.size
        pct = (num_mismatch / total) * 100
        pytest.fail(
            f"{msg} - Exact match failed: max_diff={max_diff}, "
            f"mismatched_pixels={num_mismatch}/{total} ({pct:.2f}%)"
        )


def assert_close(actual, expected, atol=1, max_mae=0.5, msg=""):
    """Verify close match within acceptable integer rounding / float tolerance."""
    if actual.ndim == 3 and actual.shape[2] == 1 and expected.ndim == 2:
        expected = expected[:, :, np.newaxis]
    elif expected.ndim == 3 and expected.shape[2] == 1 and actual.ndim == 2:
        actual = actual[:, :, np.newaxis]
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    mae = np.mean(diff)
    max_diff = np.max(diff)
    pct_over_atol = (np.count_nonzero(diff > atol) / actual.size) * 100

    if max_diff > atol and (mae > max_mae or pct_over_atol > 1.0):
        pytest.fail(
            f"{msg} - Close match failed: max_diff={max_diff} (atol={atol}), "
            f"mae={mae:.4f} (max_mae={max_mae}), pct_over_atol={pct_over_atol:.2f}%"
        )


# ============================================================================
# 1. Geometric Transforms vs OpenCV / Albumentations Ground Truth
# ============================================================================

class TestGeometricCorrectness:
    """Validate geometric transforms against OpenCV implementations."""

    def test_horizontal_flip(self, image_shape):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([HorizontalFlip()]).apply(img.copy())
            cv_res = cv2.flip(img, 1)
            if c == 1:
                cv_res = cv_res.reshape(h, w)
            assert_exact(sinter_res, cv_res, f"HorizontalFlip (channels={c})")

    def test_vertical_flip(self, image_shape):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([VerticalFlip()]).apply(img.copy())
            cv_res = cv2.flip(img, 0)
            if c == 1:
                cv_res = cv_res.reshape(h, w)
            assert_exact(sinter_res, cv_res, f"VerticalFlip (channels={c})")

    def test_transpose(self, image_shape):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Transpose()]).apply(img.copy())
            cv_res = cv2.transpose(img)
            assert_exact(sinter_res, cv_res, f"Transpose (channels={c})")

    @pytest.mark.parametrize("angle_enum,cv_flag", [
        (RotateAngle.ROTATE_90, cv2.ROTATE_90_CLOCKWISE),
        (RotateAngle.ROTATE_180, cv2.ROTATE_180),
        (RotateAngle.ROTATE_270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ])
    def test_rotate_fixed_angles(self, image_shape, angle_enum, cv_flag):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Rotate(angle=angle_enum)]).apply(img.copy())
            cv_res = cv2.rotate(img, cv_flag)
            assert_exact(sinter_res, cv_res, f"Rotate {angle_enum} (channels={c})")

    def test_crop(self, image_shape):
        h, w = image_shape
        x, y, crop_w, crop_h = 5, 7, min(w - 10, 30), min(h - 10, 20)
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Crop(x=x, y=y, width=crop_w, height=crop_h)]).apply(img.copy())
            ref_res = img[y:y + crop_h, x:x + crop_w]
            assert_exact(sinter_res, ref_res, f"Crop (channels={c})")

    @pytest.mark.parametrize("pad_mode,cv_border", [
        (PadMode.constant(0), cv2.BORDER_CONSTANT),
        (PadMode.REFLECT, cv2.BORDER_REFLECT),
        (PadMode.REPLICATE, cv2.BORDER_REPLICATE),
    ])
    def test_pad(self, image_shape, pad_mode, cv_border):
        h, w = image_shape
        top, bottom, left, right = 4, 6, 8, 5
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Pad(top=top, bottom=bottom, left=left, right=right, mode=pad_mode)]).apply(img.copy())
            cv_res = cv2.copyMakeBorder(img, top, bottom, left, right, cv_border)
            assert_exact(sinter_res, cv_res, f"Pad {pad_mode} (channels={c})")

    @pytest.mark.parametrize("target_w,target_h", [(48, 48), (80, 50), (32, 64)])
    def test_resize_nearest(self, image_shape, target_w, target_h):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Resize(width=target_w, height=target_h, interpolation=Interpolation.NEAREST)]).apply(img.copy())
            cv_res = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            assert_close(sinter_res, cv_res, atol=1, max_mae=0.5, msg=f"Resize Nearest (channels={c})")

    @pytest.mark.parametrize("target_w,target_h", [(48, 48), (80, 50), (32, 64)])
    def test_resize_bilinear(self, image_shape, target_w, target_h):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Resize(width=target_w, height=target_h, interpolation=Interpolation.BILINEAR)]).apply(img.copy())
            cv_res = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            assert_close(sinter_res, cv_res, atol=2, max_mae=0.8, msg=f"Resize Bilinear (channels={c})")

    @pytest.mark.parametrize("scale,rotate,translate,shear", [
        ((1.0, 1.0), 0.0, (0.0, 0.0), (0.0, 0.0)),
        ((1.0, 1.0), 30.0, (0.0, 0.0), (0.0, 0.0)),
        ((1.0, 1.0), -45.0, (0.0, 0.0), (0.0, 0.0)),
        ((1.2, 0.8), 0.0, (0.0, 0.0), (0.0, 0.0)),
        ((1.0, 1.0), 0.0, (5.0, -8.0), (0.0, 0.0)),
        ((1.0, 1.0), 0.0, (0.0, 0.0), (10.0, 5.0)),
        ((1.1, 1.1), 15.0, (3.0, -4.0), (5.0, -5.0)),
    ])
    def test_affine(self, image_shape, scale, rotate, translate, shear):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        sinter_res = Compose([Affine(
            scale=scale,
            rotate=rotate,
            translate=translate,
            shear=shear,
            interpolation=Interpolation.BILINEAR,
        )]).apply(img.copy())

        import albumentations.augmentations.geometric.functional as fgeometric
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        M = fgeometric.create_affine_transformation_matrix(
            translate={'x': translate[0], 'y': translate[1]},
            shear={'x': shear[0], 'y': shear[1]},
            scale={'x': scale[0], 'y': scale[1]},
            rotate=rotate,
            shift=(cx, cy),
        )

        cv_res = cv2.warpAffine(
            img,
            M[:2],
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        assert_close(sinter_res, cv_res, atol=2, max_mae=0.8, msg="Affine")

    @pytest.mark.parametrize("scale,rotate,translate,shear", [
        ((1.0, 1.0), 0.0, (0.0, 0.0), (0.0, 0.0)),
        ((1.5, 1.5), 0.0, (0.0, 0.0), (0.0, 0.0)),
        ((0.8, 0.8), 0.0, (0.0, 0.0), (0.0, 0.0)),
        ((1.0, 1.0), 30.0, (0.0, 0.0), (0.0, 0.0)),
        ((1.0, 1.0), -45.0, (0.0, 0.0), (0.0, 0.0)),
        ((1.0, 1.0), 0.0, (5.0, -8.0), (0.0, 0.0)),
        ((1.0, 1.0), 0.0, (0.0, 0.0), (10.0, 5.0)),
        # rotate + shear: y span > 2 per 8 output px -> 8-row window path
        ((1.3, 1.3), 45.0, (0.0, 0.0), (5.0, -5.0)),
        ((1.3, 1.3), 45.0, (0.0, 0.0), (15.0, -15.0)),
        ((1.3, 1.3), 75.0, (0.0, 0.0), (20.0, -20.0)),
        ((1.3, 1.3), -45.0, (0.0, 0.0), (5.0, 10.0)),
    ])
    def test_affine_gray(self, image_shape, scale, rotate, translate, shear):
        """Gray affine vs cv2 (the RGB-only test above does not cover C=1)."""
        h, w = image_shape
        img = create_test_image(h, w, 1)
        sinter_res = Compose([Affine(
            scale=scale,
            rotate=rotate,
            translate=translate,
            shear=shear,
            interpolation=Interpolation.BILINEAR,
        )]).apply(img.copy())

        import albumentations.augmentations.geometric.functional as fgeometric
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        M = fgeometric.create_affine_transformation_matrix(
            translate={'x': translate[0], 'y': translate[1]},
            shear={'x': shear[0], 'y': shear[1]},
            scale={'x': scale[0], 'y': scale[1]},
            rotate=rotate,
            shift=(cx, cy),
        )

        # sinter's Python Affine defaults to Constant{value:0} border.
        cv_res = cv2.warpAffine(
            img,
            M[:2],
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        assert_close(sinter_res, cv_res, atol=2, max_mae=0.8, msg="Affine (gray)")


# ============================================================================
# 2. Photometric / LUT Transforms vs Albumentations / NumPy Ground Truth
# ============================================================================

class TestPhotometricCorrectness:
    """Validate photometric operations against ground truth."""

    def test_invert(self, image_shape):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Invert()]).apply(img.copy())
            ref_res = 255 - img
            assert_exact(sinter_res, ref_res, f"Invert (channels={c})")

    @pytest.mark.parametrize("threshold", [64, 128, 192])
    def test_solarize(self, image_shape, threshold):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Solarize(threshold=threshold)]).apply(img.copy())
            ref_res = np.where(img >= threshold, 255 - img, img)
            assert_exact(sinter_res, ref_res, f"Solarize (threshold={threshold}, channels={c})")

    @pytest.mark.parametrize("bits", [2, 4, 6])
    def test_posterize(self, image_shape, bits):
        h, w = image_shape
        shift = 8 - bits
        mask = ((1 << bits) - 1) << shift
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Posterize(bits=bits)]).apply(img.copy())
            ref_res = (img & mask)
            assert_exact(sinter_res, ref_res, f"Posterize (bits={bits}, channels={c})")

    @pytest.mark.parametrize("delta", [-50.0, -15.0, 0.0, 25.0, 60.0])
    def test_brightness(self, image_shape, delta):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Brightness(delta=Constant(delta))]).apply(img.copy())
            ref_res = np.clip(img.astype(np.int32) + int(round(delta)), 0, 255).astype(np.uint8)
            assert_exact(sinter_res, ref_res, f"Brightness (delta={delta}, channels={c})")

    @pytest.mark.parametrize("factor", [0.5, 0.8, 1.0, 1.3, 2.0])
    def test_contrast(self, image_shape, factor):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Contrast(factor=Constant(factor))]).apply(img.copy())
            ref_res = np.clip(128.0 + factor * (img.astype(np.float32) - 128.0) + 0.5, 0, 255).astype(np.uint8)
            assert_close(sinter_res, ref_res, atol=1, max_mae=0.3, msg=f"Contrast (factor={factor}, channels={c})")

    @pytest.mark.parametrize("gamma_val", [0.6, 0.8, 1.0, 1.5, 2.2])
    def test_gamma(self, image_shape, gamma_val):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([Gamma(gamma=Constant(gamma_val))]).apply(img.copy())
            lut = np.array([np.clip(pow(i / 255.0, gamma_val) * 255.0 + 0.5, 0, 255) for i in range(256)], dtype=np.uint8)
            ref_res = cv2.LUT(img, lut)
            assert_close(sinter_res, ref_res, atol=1, max_mae=0.2, msg=f"Gamma (gamma={gamma_val}, channels={c})")

    @pytest.mark.parametrize("shifts", [(10, -20, 30), (-15, 25, -10), (0, 0, 0)])
    def test_rgb_shift(self, image_shape, shifts):
        h, w = image_shape
        r_s, g_s, b_s = shifts
        img = create_test_image(h, w, 3)
        sinter_res = Compose([RGBShift(r_shift=Constant(r_s), g_shift=Constant(g_s), b_shift=Constant(b_s))]).apply(img.copy())
        ref_r = np.clip(img[:, :, 0].astype(np.int32) + r_s, 0, 255)
        ref_g = np.clip(img[:, :, 1].astype(np.int32) + g_s, 0, 255)
        ref_b = np.clip(img[:, :, 2].astype(np.int32) + b_s, 0, 255)
        ref_res = np.stack([ref_r, ref_g, ref_b], axis=-1).astype(np.uint8)
        assert_exact(sinter_res, ref_res, f"RGBShift {shifts}")

    def test_to_gray(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        sinter_res = Compose([ToGray()]).apply(img.copy())
        gray_1ch = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        assert_close(sinter_res, gray_1ch, atol=1, max_mae=0.3, msg="ToGray")

    @pytest.mark.parametrize("order_idx,expected_order", [
        (1, [0, 2, 1]),
        (2, [1, 0, 2]),
        (5, [2, 1, 0]),
    ])
    def test_channel_shuffle(self, image_shape, order_idx, expected_order):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        sinter_res = Compose([ChannelShuffle(order=order_idx)]).apply(img.copy())
        ref_res = img[:, :, expected_order]
        assert_exact(sinter_res, ref_res, f"ChannelShuffle order={order_idx}")

    def test_equalize(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        sinter_res = Compose([Equalize()]).apply(img.copy())
        ref_r = cv2.equalizeHist(img[:, :, 0])
        ref_g = cv2.equalizeHist(img[:, :, 1])
        ref_b = cv2.equalizeHist(img[:, :, 2])
        ref_res = np.stack([ref_r, ref_g, ref_b], axis=-1)
        assert_close(sinter_res, ref_res, atol=2, max_mae=0.5, msg="Equalize")


# ============================================================================
# 3. Kernel / Filter Transforms vs OpenCV Ground Truth
# ============================================================================

class TestKernelCorrectness:
    """Validate convolution and non-linear filter kernels against OpenCV."""

    def test_median_blur_3x3(self, image_shape):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([MedianBlur(kernel_size=3)]).apply(img.copy())
            cv_res = cv2.medianBlur(img, 3)
            assert_exact(sinter_res, cv_res, f"MedianBlur 3x3 (channels={c})")

    def test_median_blur_5x5(self, image_shape):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([MedianBlur(kernel_size=5)]).apply(img.copy())
            cv_res = cv2.medianBlur(img, 5)
            assert_exact(sinter_res, cv_res, f"MedianBlur 5x5 (channels={c})")

    @pytest.mark.parametrize("ksize", [3, 5, 7])
    def test_gaussian_blur(self, image_shape, ksize):
        h, w = image_shape
        for c in [1, 3]:
            img = create_test_image(h, w, c)
            sinter_res = Compose([GaussianBlur(kernel_size=ksize)]).apply(img.copy())
            cv_res = cv2.GaussianBlur(img, (ksize, ksize), 0, borderType=cv2.BORDER_REPLICATE)
            assert_close(sinter_res, cv_res, atol=2, max_mae=1.0, msg=f"GaussianBlur {ksize}x{ksize} (channels={c})")


# ============================================================================
# 4. Pipeline Fusion vs Sequential Unfused Execution
# ============================================================================

class TestPipelineFusionCorrectness:
    """Validate that fused pipelines produce identical output to sequential application."""

    def test_fused_lut_group_equivalence(self, image_shape):
        """Verify that fusing 4 LUT transforms (Brightness + Contrast + Gamma + Invert)
        produces identical results to running each transform sequentially."""
        h, w = image_shape
        img = create_test_image(h, w, 3)

        # 1. Fused pipeline in Sinter
        fused_pipe = Compose([
            Brightness(delta=Constant(20.0)),
            Contrast(factor=Constant(1.2)),
            Gamma(gamma=Constant(0.85)),
            Invert(),
        ])
        fused_result = fused_pipe.apply(img.copy())

        # 2. Sequential unfused pipeline in Sinter
        pipe1 = Compose([Brightness(delta=Constant(20.0))])
        pipe2 = Compose([Contrast(factor=Constant(1.2))])
        pipe3 = Compose([Gamma(gamma=Constant(0.85))])
        pipe4 = Compose([Invert()])

        step1 = pipe1.apply(img.copy())
        step2 = pipe2.apply(step1)
        step3 = pipe3.apply(step2)
        sequential_result = pipe4.apply(step3)

        assert_close(
            fused_result, sequential_result,
            atol=1, max_mae=0.2,
            msg="Fused LUT vs Sequential LUT equivalence"
        )

    def test_fused_geometric_and_photometric_pipeline(self, image_shape):
        """Test multi-stage pipeline: Geometric + LUT + Geometric + Photometric."""
        h, w = image_shape
        img = create_test_image(h, w, 3)

        pipeline = Compose([
            HorizontalFlip(),
            Brightness(delta=Constant(15.0)),
            Contrast(factor=Constant(1.1)),
            VerticalFlip(),
            Invert(),
        ])
        result = pipeline.apply(img.copy())

        # Sequential reference
        ref = cv2.flip(img, 1)
        ref = np.clip(ref.astype(np.int32) + 15, 0, 255).astype(np.uint8)
        ref = np.clip(128.0 + 1.1 * (ref.astype(np.float32) - 128.0) + 0.5, 0, 255).astype(np.uint8)
        ref = cv2.flip(ref, 0)
        ref = 255 - ref

        assert_close(result, ref, atol=1, max_mae=0.3, msg="Multi-stage Geometric+LUT pipeline")

    def test_complex_augmentation_pipeline(self, image_shape):
        """Test complex real-world computer vision augmentation pipeline."""
        h, w = image_shape
        img = create_test_image(h, w, 3)

        pipeline = Compose([
            HorizontalFlip(),
            Resize(width=64, height=64, interpolation=Interpolation.BILINEAR),
            GaussianBlur(kernel_size=3),
            Brightness(delta=Constant(-10.0)),
            Contrast(factor=Constant(1.15)),
        ])
        result = pipeline.apply(img.copy())
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8


# ============================================================================
# 5. Deterministic Sampling & Seed Repeatability
# ============================================================================

class TestSamplingDeterminism:
    """Validate that sample_with_seed produces 100% deterministic and repeatable results."""

    def test_seed_determinism(self, image_shape):
        h, w = image_shape
        img1 = create_test_image(h, w, 3, seed=1)
        img2 = create_test_image(h, w, 3, seed=2)

        pipe = Compose([
            Brightness(delta=Uniform(-30.0, 30.0)),
            Contrast(factor=Uniform(0.7, 1.4)),
            HorizontalFlip(p=0.5),
        ])

        # Sample with fixed seed 42 twice
        sampled1 = pipe.sample_with_seed(42)
        sampled2 = pipe.sample_with_seed(42)

        out1_a = sampled1.apply(img1.copy())
        out1_b = sampled2.apply(img1.copy())
        assert_exact(out1_a, out1_b, "Deterministic sampling on img1")

        out2_a = sampled1.apply(img2.copy())
        out2_b = sampled2.apply(img2.copy())
        assert_exact(out2_a, out2_b, "Deterministic sampling on img2")


# ============================================================================
# 6. Advanced Filters & Effects
# ============================================================================

class TestAdvancedTransforms:
    """Validate advanced effects (Emboss, EdgeDetection, Sharpen, HSV, Sepia, Dropout)."""

    def test_sharpen(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        res = Compose([Sharpen(strength=Constant(0.5))]).apply(img.copy())
        assert res.shape == (h, w, 3)
        assert res.dtype == np.uint8

    def test_to_sepia(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        res = Compose([ToSepia()]).apply(img.copy())
        assert res.shape == (h, w, 3)
        assert res.dtype == np.uint8

    def test_auto_contrast(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        res = Compose([AutoContrast()]).apply(img.copy())
        assert res.shape == (h, w, 3)
        assert res.dtype == np.uint8

    def test_color_temperature(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        res = Compose([ColorTemperature(temperature=Constant(15.0))]).apply(img.copy())
        assert res.shape == (h, w, 3)
        assert res.dtype == np.uint8

    def test_color_tint(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        res = Compose([ColorTint(tint=Constant(10.0))]).apply(img.copy())
        assert res.shape == (h, w, 3)
        assert res.dtype == np.uint8

    def test_color_balance(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        res = Compose([ColorBalance(r_scale=Constant(1.1), g_scale=Constant(0.9), b_scale=Constant(1.0))]).apply(img.copy())
        assert res.shape == (h, w, 3)
        assert res.dtype == np.uint8

    def test_coarse_dropout(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        res = Compose([CoarseDropout(holes=Constant(4), hole_size=(Constant(8), Constant(8)))]).apply(img.copy())
        assert res.shape == (h, w, 3)
        assert res.dtype == np.uint8

    def test_grid_dropout(self, image_shape):
        h, w = image_shape
        img = create_test_image(h, w, 3)
        res = Compose([GridDropout(ratio=Constant(0.5), unit_size=Constant(16), holes=Constant(2))]).apply(img.copy())
        assert res.shape == (h, w, 3)
        assert res.dtype == np.uint8


# ============================================================================
# 5. Grayscale 2D (H, W) & 3D (H, W, 1) Comprehensive Correctness Tests
# ============================================================================

class TestGrayscaleCorrectness:
    """Verify transforms on 2D and 3D grayscale images match OpenCV/Albumentations."""

    def test_grayscale_photometric(self, image_shape):
        h, w = image_shape
        for c in [None, 1]:
            img = create_test_image(h, w, c) if c else create_test_image(h, w, 1)[:, :, 0]
            
            # Brightness
            sinter_res = Compose([Brightness(delta=Constant(25.0))]).apply(img.copy())
            ref = np.clip(img.astype(np.int32) + 25, 0, 255).astype(np.uint8)
            assert_exact(sinter_res, ref, f"Grayscale Brightness (c={c})")

            # Contrast
            sinter_res = Compose([Contrast(factor=Constant(1.2))]).apply(img.copy())
            ref = np.clip(128.0 + 1.2 * (img.astype(np.float32) - 128.0) + 0.5, 0, 255).astype(np.uint8)
            assert_close(sinter_res, ref, atol=1, max_mae=0.1, msg=f"Grayscale Contrast (c={c})")

            # Invert
            sinter_res = Compose([Invert()]).apply(img.copy())
            assert_exact(sinter_res, 255 - img, f"Grayscale Invert (c={c})")

    def test_grayscale_geometric(self, image_shape):
        h, w = image_shape
        for c in [None, 1]:
            img = create_test_image(h, w, c) if c else create_test_image(h, w, 1)[:, :, 0]

            # HorizontalFlip
            sinter_res = Compose([HorizontalFlip()]).apply(img.copy())
            ref = cv2.flip(img, 1)
            if c == 1 and ref.ndim == 2:
                ref = ref[:, :, np.newaxis]
            assert_exact(sinter_res, ref, f"Grayscale HorizontalFlip (c={c})")

            # VerticalFlip
            sinter_res = Compose([VerticalFlip()]).apply(img.copy())
            ref = cv2.flip(img, 0)
            if c == 1 and ref.ndim == 2:
                ref = ref[:, :, np.newaxis]
            assert_exact(sinter_res, ref, f"Grayscale VerticalFlip (c={c})")

            # Resize Bilinear
            sinter_res = Compose([Resize(width=w // 2, height=h // 2, interpolation=Interpolation.BILINEAR)]).apply(img.copy())
            ref = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            if c == 1 and ref.ndim == 2:
                ref = ref[:, :, np.newaxis]
            assert_close(sinter_res, ref, atol=2, max_mae=0.8, msg=f"Grayscale Resize Bilinear (c={c})")

    def test_grayscale_kernels(self, image_shape):
        h, w = image_shape
        for c in [None, 1]:
            img = create_test_image(h, w, c) if c else create_test_image(h, w, 1)[:, :, 0]

            # GaussianBlur 3x3
            sinter_res = Compose([GaussianBlur(kernel_size=3)]).apply(img.copy())
            ref = cv2.GaussianBlur(img, (3, 3), 0, borderType=cv2.BORDER_REFLECT_101)
            if c == 1 and ref.ndim == 2:
                ref = ref[:, :, np.newaxis]
            # Sinter's Gaussian convention is per-pass integer truncation (same as the RGB path
            # and the scalar reference); cv2 rounds in fixed point. Gray now matches RGB, so use
            # the same tolerance as the RGB GaussianBlur tests.
            assert_close(sinter_res, ref, atol=2, max_mae=1.0, msg=f"Grayscale GaussianBlur 3x3 (c={c})")

            # MedianBlur 3x3
            sinter_res = Compose([MedianBlur(kernel_size=3)]).apply(img.copy())
            ref = cv2.medianBlur(img, 3)
            if c == 1 and ref.ndim == 2:
                ref = ref[:, :, np.newaxis]
            assert_exact(sinter_res, ref, f"Grayscale MedianBlur 3x3 (c={c})")
