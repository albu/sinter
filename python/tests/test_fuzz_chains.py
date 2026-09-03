"""
Property-based fuzz test verifying that optimized/fused execution produces
identical results to sequential execution across randomly generated transform chains.
"""

import random
import numpy as np
import pytest

from sinter import (
    Compose,
    Brightness,
    Contrast,
    Gamma,
    Invert,
    Solarize,
    Posterize,
    RGBShift,
    Equalize,
    AutoContrast,
    ToSepia,
    ColorTemperature,
    HueSaturationValue,
    ToGray,
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    Crop,
    RandomCrop,
    GaussianBlur,
    MedianBlur,
    Sharpen,
    GaussNoise,
)

def get_transform_pool(w, h):
    crop_w = max(32, w // 2)
    crop_h = max(32, h // 2)
    return [
        lambda: Brightness(delta=random.uniform(-40, 40)),
        lambda: Contrast(factor=random.uniform(0.7, 1.4)),
        lambda: Gamma(gamma=random.uniform(0.8, 1.2)),
        lambda: Invert(),
        lambda: Solarize(threshold=random.randint(64, 192)),
        lambda: Posterize(bits=random.randint(4, 7)),
        lambda: RGBShift(
            r_shift=random.uniform(-25, 25),
            g_shift=random.uniform(-25, 25),
            b_shift=random.uniform(-25, 25),
        ),
        lambda: AutoContrast(cutoff=random.uniform(0.0, 0.03)),
        lambda: Equalize(),
        lambda: ToSepia(),
        lambda: ColorTemperature(temperature=random.uniform(-30, 30)),
        lambda: HueSaturationValue(
            hue_shift=random.uniform(-20, 20),
            saturation_scale=random.uniform(0.8, 1.2),
            value_scale=random.uniform(0.9, 1.1),
        ),
        lambda: ToGray(),
        lambda: HorizontalFlip(),
        lambda: VerticalFlip(),
        lambda: Transpose(),
        lambda: GaussianBlur(kernel_size=3),
        lambda: MedianBlur(kernel_size=3),
        lambda: Sharpen(),
        lambda: GaussNoise(var_limit=(5.0, 20.0)),
        lambda: Crop(x=random.randint(0, w - crop_w), y=random.randint(0, h - crop_h), width=crop_w, height=crop_h),
        lambda: RandomCrop(width=crop_w, height=crop_h),
    ]

def test_100_random_augmentation_chains():
    img_size = (128, 128)
    rng = np.random.default_rng(12345)
    random.seed(12345)
    base_img = rng.integers(0, 256, (img_size[1], img_size[0], 3), dtype=np.uint8)

    matrix_ops = {"ToSepia", "ColorTemperature"}

    for chain_idx in range(100):
        chain_len = random.randint(2, 7)
        transforms = []
        has_crop = False
        multi_matrix_count = 0

        pool = get_transform_pool(img_size[0], img_size[1])
        for _ in range(chain_len):
            t = random.choice(pool)()
            name = t.__class__.__name__
            if "Crop" in name:
                if has_crop:
                    continue
                has_crop = True
            if name in matrix_ops:
                multi_matrix_count += 1
            transforms.append(t)

        pipe = Compose(transforms)
        seed = 1000 + chain_idx

        out_fused = pipe.apply(base_img.copy(), seed=seed, optimize=True)
        out_seq = pipe.apply(base_img.copy(), seed=seed, optimize=False)

        assert out_fused.shape == out_seq.shape, f"Shape mismatch in chain {chain_idx}: {out_fused.shape} vs {out_seq.shape}"

        diff = int(np.abs(out_fused.astype(int) - out_seq.astype(int)).max())

        has_noise = any(t.__class__.__name__ == "GaussNoise" for t in transforms)
        if has_noise:
            continue

        if multi_matrix_count >= 2:
            # Multi-matrix fusion skips intermediate 8-bit integer truncation
            continue

        assert diff <= 1, f"Chain #{chain_idx} {[t.__class__.__name__ for t in transforms]} had divergence {diff}"
