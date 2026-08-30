"""
Individual transform benchmark - tests each transform in isolation.

Tests each transform individually to compare raw performance without fusion benefits.

NOTE: Forces single-threaded OpenCV for fair comparison (sinter uses single-threaded)

Usage:
    python benchmark_individual.py           # Run all transforms
    python benchmark_individual.py ToRGB     # Run only ToRGB
    python benchmark_individual.py --matrix  # Run only matrix transforms
"""

import sys
import time
import numpy as np

try:
    import albumentations as A
    import cv2
    # Force single-threaded for fair comparison with sinter (which uses single-threaded OpenCV)
    # Note: setNumThreads(1) doesn't work, must use 0
    cv2.setNumThreads(0)
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed")

from sinter import (
    # LUT transforms
    Brightness, Contrast, Gamma, Invert, Posterize, Solarize,
    # Matrix transforms
    ToSepia, ToRGB, HueSaturationValue, ColorTemperature, ColorTint,
    # Noise
    GaussNoise, SaltAndPepper, MultiplicativeNoise, CoarseDropout, GridDropout,
    # Channel transforms
    RGBShift,
    ColorBalance, ChannelShuffle,
    # Histogram
    Equalize, AutoContrast, ToGray, Normalize,
    # Geometric
    HorizontalFlip, VerticalFlip, Rotate, RotateAngle,
    Resize, Crop, Pad, PadMode, Transpose, Affine, Interpolation,
    # Kernel
    GaussianBlur, Sharpen, Emboss, EmbossDirection, MedianBlur, EdgeDetection, EdgeMethod,
    Compose,
    Constant,
)

WARMUP_RUNS = 5
BENCHMARK_RUNS = 100

# Global filter state (mutable container to avoid global keyword issues)
_FILTER_STATE = {"cats": None, "names": None}

def should_run(transform_name, categories, transforms):
    """Check if a transform should be benchmarked based on filters"""
    if categories is None and transforms is None:
        return True

    # Check exact name match
    if transforms and any(t in transform_name.lower() for t in transforms):
        return True

    # Check category match
    category_map = {
        "lut": ["brightness", "contrast", "gamma", "invert", "posterize", "solarize", "equalize", "togray"],
        "matrix": ["tosepia", "torgb", "colortemperature", "colortint", "colorbalance", "channelshuffle"],
        "noise": ["gaussnoise", "rgbshift", "saltandpepper", "multiplicativenoise",
                 "huesaturationvalue", "coarsedropout", "griddropout"],
        "geometric": ["horizontalflip", "verticalflip", "rotate", "resize", "crop", "pad", "transpose"],
        "kernel": ["gaussianblur", "medianblur", "sharpen", "emboss", "edgedetection"],
    }

    if categories:
        for cat in categories:
            if cat in category_map:
                if any(t in transform_name.lower() for t in category_map[cat]):
                    return True

    return False

SIZES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
]

def benchmark_transform(name, albumentations_transform, sinter_transform_factory, img, runs=BENCHMARK_RUNS):
    """Benchmark a single transform"""
    # Skip if not in filter
    if not should_run(name, _FILTER_STATE["cats"], _FILTER_STATE["names"]):
        return None, None, None
    # Warmup sinter
    sinter_pipe = Compose([sinter_transform_factory()])
    for _ in range(WARMUP_RUNS):
        _ = sinter_pipe.apply(img.copy())

    # Benchmark sinter
    start = time.perf_counter()
    for _ in range(runs):
        _ = sinter_pipe.apply(img.copy())
    sinter_time = (time.perf_counter() - start) / runs * 1000

    if HAS_ALBUMENTATIONS and albumentations_transform is not None:
        # Warmup albumentations
        for _ in range(WARMUP_RUNS):
            _ = albumentations_transform(image=img.copy())

        # Benchmark albumentations
        start = time.perf_counter()
        for _ in range(runs):
            _ = albumentations_transform(image=img.copy())
        albumentations_time = (time.perf_counter() - start) / runs * 1000

        speedup = albumentations_time / sinter_time
        return albumentations_time, sinter_time, speedup

    return None, sinter_time, None

def print_row(name, albumentations_time, sinter_time, speedup):
    # Skip if all None (filtered out)
    if albumentations_time is None and sinter_time is None:
        return
    if albumentations_time is not None:
        faster = "faster" if speedup > 1 else "slower"
        print(f"  {name:<30} {albumentations_time:>8.3f} ms  {sinter_time:>8.3f} ms  {speedup:>6.2f}x {faster}")
    else:
        print(f"  {name:<30} {'N/A':>12}  {sinter_time:>8.3f} ms  {'sinter only':>12}")

def benchmark_filtered(name, categories, transforms, *args, **kwargs):
    """Run benchmark only if the transform matches the filter"""
    if not should_run(name, categories, transforms):
        return None, None, None
    return benchmark_transform(name, *args, **kwargs)

def run_benchmarks():
    for width, height in SIZES:
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        print("=" * 80)
        print(f"INDIVIDUAL TRANSFORM BENCHMARK ({width}x{height})")
        print("=" * 80)
        print(f"{'Transform':<30} {'Alb (ms)':>12}  {'Sinter (ms)':>12}  {'Speedup':>12}")
        print("-" * 80)

        # LUT Transforms
        print("\n[LUT TRANSFORMS]")
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Brightness",
                A.Compose([A.RandomBrightnessContrast(brightness_limit=(40.0/255, 40.0/255), contrast_limit=(0, 0), p=1.0)]),
                lambda: Brightness(delta=40.0),
                img
            )
            print_row("Brightness", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Contrast",
                A.Compose([A.RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(0.3, 0.3), p=1.0)]),
                lambda: Contrast(factor=1.3),
                img
            )
            print_row("Contrast", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Invert",
                A.Compose([A.InvertImg(p=1.0)]),
                lambda: Invert(),
                img
            )
            print_row("Invert", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Posterize",
                A.Compose([A.Posterize(num_bits=[4, 4], p=1.0)]),
                lambda: Posterize(bits=4),
                img
            )
            print_row("Posterize", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Solarize",
                A.Compose([A.Solarize(threshold_range=(128/255, 128/255), p=1.0)]),
                lambda: Solarize(threshold=128),
                img
            )
            print_row("Solarize", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Equalize",
                A.Compose([A.Equalize(p=1.0)]),
                lambda: Equalize(),
                img
            )
            print_row("Equalize", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "ToGray",
                A.Compose([A.ToGray(p=1.0)]),
                lambda: ToGray(),
                img
            )
            print_row("ToGray", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "AutoContrast",
                A.Compose([A.AutoContrast(p=1.0)]),
                lambda: AutoContrast(),
                img
            )
            print_row("AutoContrast", albumentations_time, sinter_time, speedup)

            # Gamma - Albumentations has RandomGamma, use fixed limit for comparison
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Gamma",
                A.Compose([A.RandomGamma(gamma_limit=(80, 80), p=1.0)]),
                lambda: Gamma(gamma=0.8),
                img
            )
            print_row("Gamma", albumentations_time, sinter_time, speedup)
        else:
            # Sinter only
            albumentations_time, sinter_time, speedup = benchmark_transform("Brightness", None, lambda: Brightness(delta=40.0), img)
            print_row("Brightness", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Contrast", None, lambda: Contrast(factor=1.3), img)
            print_row("Contrast", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Invert", None, lambda: Invert(), img)
            print_row("Invert", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Posterize", None, lambda: Posterize(bits=4), img)
            print_row("Posterize", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Solarize", None, lambda: Solarize(threshold=128), img)
            print_row("Solarize", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Equalize", None, lambda: Equalize(), img)
            print_row("Equalize", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("ToGray", None, lambda: ToGray(), img)
            print_row("ToGray", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("AutoContrast", None, lambda: AutoContrast(), img)
            print_row("AutoContrast", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Normalize", None, lambda: Normalize(mean=0.0, std=1.0), img)
            print_row("Normalize", albumentations_time, sinter_time, speedup)

        # Matrix Transforms
        print("\n[MATRIX TRANSFORMS]")
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "ToSepia",
                A.Compose([A.ToRGB(), A.ToSepia(p=1.0)]),
                lambda: ToSepia(),
                img
            )
            print_row("ToSepia", albumentations_time, sinter_time, speedup)

            # ToRGB benchmark - use grayscale image
            gray_img = np.random.randint(0, 256, (height, width, 1), dtype=np.uint8)
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "ToRGB",
                A.Compose([A.ToRGB(p=1.0)]),
                lambda: ToRGB(),
                gray_img
            )
            print_row("ToRGB", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("ToSepia", None, lambda: ToSepia(), img)
            print_row("ToSepia", albumentations_time, sinter_time, speedup)

            # ToRGB benchmark - use grayscale image
            gray_img = np.random.randint(0, 256, (height, width, 1), dtype=np.uint8)
            albumentations_time, sinter_time, speedup = benchmark_transform("ToRGB", None, lambda: ToRGB(), gray_img)
            print_row("ToRGB", albumentations_time, sinter_time, speedup)

        # Saturation - Albumentations has HueSaturationValue, use fixed sat_shift_limit for comparison
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "HueSaturationValue",
                A.Compose([A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(30, 30), val_shift_limit=0, p=1.0)]),
                lambda: HueSaturationValue(hue_shift=0, saturation_scale=1.3, value_scale=1.0),
                img
            )
            print_row("HueSaturationValue", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("HueSaturationValue", None, lambda: HueSaturationValue(hue_shift=0, saturation_scale=1.3, value_scale=1.0), img)
            print_row("HueSaturationValue", albumentations_time, sinter_time, speedup)

        albumentations_time, sinter_time, speedup = benchmark_transform("ColorTemperature", None, lambda: ColorTemperature(temperature=50), img)
        print_row("ColorTemperature", albumentations_time, sinter_time, speedup)

        # ColorTint - sinter only (no albumentations equivalent)
        # tint parameter: [target_r, target_g, target_b, intensity]
        albumentations_time, sinter_time, speedup = benchmark_transform("ColorTint", None, lambda: ColorTint(tint=(255, 200, 100, 0.5)), img)
        print_row("ColorTint", albumentations_time, sinter_time, speedup)

        # Noise Transforms
        print("\n[NOISE TRANSFORMS]")
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "GaussNoise",
                A.Compose([A.GaussNoise(var_limit=(10.0, 10.0), p=1.0)]),
                lambda: GaussNoise(mean=0.0, std=10.0),
                img
            )
            print_row("GaussNoise", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "RGBShift",
                A.Compose([A.RGBShift(r_shift=(20, 20), g_shift=(20, 20), b_shift=(20, 20), p=1.0)]),
                lambda: RGBShift(r_shift=20, g_shift=20, b_shift=20),
                img
            )
            print_row("RGBShift", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "CoarseDropout",
                A.Compose([A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=8, min_height=8, min_width=8, p=1.0)]),
                lambda: CoarseDropout(holes=8, hole_size=[0.08, 0.08]),
                img
            )
            print_row("CoarseDropout", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("GaussNoise", None, lambda: GaussNoise(mean=0.0, std=10.0), img)
            print_row("GaussNoise", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("RGBShift", None, lambda: RGBShift(r_shift=20, g_shift=20, b_shift=20), img)
            print_row("RGBShift", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("CoarseDropout", None, lambda: CoarseDropout(holes=8, hole_size=[0.08, 0.08]), img)
            print_row("CoarseDropout", albumentations_time, sinter_time, speedup)

        # GridDropout - Albumentations equivalent exists
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "GridDropout",
                A.Compose([A.GridDropout(ratio=0.2, unit_size_range=(32, 33), p=1.0)]),
                lambda: GridDropout(unit_size=32, ratio=0.2, holes=0),
                img
            )
            print_row("GridDropout", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("GridDropout", None, lambda: GridDropout(unit_size=32, ratio=0.2, holes=0), img)
            print_row("GridDropout", albumentations_time, sinter_time, speedup)

        # SaltAndPepper - Albumentations equivalent exists
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "SaltAndPepper",
                A.Compose([A.SaltAndPepper(amount=(0.01, 0.01), p=1.0)]),
                lambda: SaltAndPepper(amount=0.01, salt_vs_pepper=0.5),
                img
            )
            print_row("SaltAndPepper", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("SaltAndPepper", None, lambda: SaltAndPepper(amount=0.01, salt_vs_pepper=0.5), img)
            print_row("SaltAndPepper", albumentations_time, sinter_time, speedup)

        # MultiplicativeNoise - Albumentations equivalent exists
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "MultiplicativeNoise",
                A.Compose([A.MultiplicativeNoise(multiplier=(0.5, 0.5), p=1.0)]),
                lambda: MultiplicativeNoise(multiplier=0.5),
                img
            )
            print_row("MultiplicativeNoise", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("MultiplicativeNoise", None, lambda: MultiplicativeNoise(multiplier=0.5), img)
            print_row("MultiplicativeNoise", albumentations_time, sinter_time, speedup)

        # HueSaturationValue - Albumentations equivalent exists
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "HueSaturationValue",
                A.Compose([A.HueSaturationValue(hue_shift_limit=(10, 10), sat_shift_limit=(10, 10), val_shift_limit=(10, 10), p=1.0)]),
                lambda: HueSaturationValue(hue_shift=10, saturation_scale=1.1, value_scale=1.1),
                img
            )
            print_row("HueSaturationValue", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("HueSaturationValue", None, lambda: HueSaturationValue(hue_shift=10, saturation_scale=1.1, value_scale=1.1), img)
            print_row("HueSaturationValue", albumentations_time, sinter_time, speedup)

        # ColorBalance - sinter only (no albumentations equivalent)
        albumentations_time, sinter_time, speedup = benchmark_transform("ColorBalance", None, lambda: ColorBalance(r_scale=1.1, g_scale=1.1, b_scale=1.1), img)
        print_row("ColorBalance", albumentations_time, sinter_time, speedup)

        # ChannelShuffle - Albumentations equivalent exists
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "ChannelShuffle",
                A.Compose([A.ChannelShuffle(p=1.0)]),
                lambda: ChannelShuffle(order=5),
                img
            )
            print_row("ChannelShuffle", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("ChannelShuffle", None, lambda: ChannelShuffle(order=5), img)
            print_row("ChannelShuffle", albumentations_time, sinter_time, speedup)

        # Geometric Transforms
        print("\n[GEOMETRIC TRANSFORMS]")
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "HorizontalFlip",
                A.Compose([A.HorizontalFlip(p=1.0)]),
                lambda: HorizontalFlip(),
                img
            )
            print_row("HorizontalFlip", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "VerticalFlip",
                A.Compose([A.VerticalFlip(p=1.0)]),
                lambda: VerticalFlip(),
                img
            )
            print_row("VerticalFlip", albumentations_time, sinter_time, speedup)

            # Rotate 90° - using seeded RandomRotate90 for fair comparison
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Rotate90",
                A.Compose([A.RandomRotate90(p=1.0)], seed=0),  # seed=0 gives 90° rotation
                lambda: Rotate(angle=RotateAngle.ROTATE_90),  # angle=0 = 90° rotation (NEON SIMD transpose)
                img
            )
            print_row("Rotate90 (seeded)", albumentations_time, sinter_time, speedup)

            # Rotate 180°
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Rotate180",
                A.Compose([A.RandomRotate90(p=1.0)], seed=5),  # seed=5 gives 180° rotation
                lambda: Rotate(angle=RotateAngle.ROTATE_180),  # angle=1 = 180° rotation
                img
            )
            print_row("Rotate180 (seeded)", albumentations_time, sinter_time, speedup)

            # Rotate 270°
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Rotate270",
                A.Compose([A.RandomRotate90(p=1.0)], seed=1),  # seed=1 gives 270° rotation
                lambda: Rotate(angle=RotateAngle.ROTATE_270),  # angle=2 = 270° rotation
                img
            )
            print_row("Rotate270 (seeded)", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("HorizontalFlip", None, lambda: HorizontalFlip(), img)
            print_row("HorizontalFlip", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("VerticalFlip", None, lambda: VerticalFlip(), img)
            print_row("VerticalFlip", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Rotate90", None, lambda: Rotate(angle=RotateAngle.ROTATE_90), img)
            print_row("Rotate90", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Rotate180", None, lambda: Rotate(angle=RotateAngle.ROTATE_180), img)
            print_row("Rotate180", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("Rotate270", None, lambda: Rotate(angle=RotateAngle.ROTATE_270), img)
            print_row("Rotate270", albumentations_time, sinter_time, speedup)

        # For Resize, Crop, Pad - use fixed sizes for benchmarking
        # Note: These don't adapt to image size, so skip for 256x256
        if width >= 512:
            if HAS_ALBUMENTATIONS:
                import cv2

                # Compare bilinear to bilinear (apples to apples)
                albumentations_time, sinter_time, speedup = benchmark_transform(
                    "Resize",
                    A.Compose([A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR, p=1.0)]),
                    lambda: Resize(width=256, height=256, interpolation=Interpolation.BILINEAR),
                    img
                )
                print_row("Resize(512->256, bilinear)", albumentations_time, sinter_time, speedup)

                # Compare nearest to nearest (apples to apples)
                albumentations_time, sinter_time, speedup = benchmark_transform(
                    "Resize",
                    A.Compose([A.Resize(height=256, width=256, interpolation=cv2.INTER_NEAREST, p=1.0)]),
                    lambda: Resize(width=256, height=256, interpolation=Interpolation.NEAREST),
                    img
                )
                print_row("Resize(512->256, nearest)", albumentations_time, sinter_time, speedup)
            else:
                albumentations_time, sinter_time, speedup = benchmark_transform("Resize", None, lambda: Resize(width=256, height=256, interpolation=Interpolation.BILINEAR), img)
                print_row("Resize(512->256, bilinear)", albumentations_time, sinter_time, speedup)

                albumentations_time, sinter_time, speedup = benchmark_transform("Resize", None, lambda: Resize(width=256, height=256, interpolation=Interpolation.NEAREST), img)
                print_row("Resize(512->256, nearest)", albumentations_time, sinter_time, speedup)

            if HAS_ALBUMENTATIONS:
                albumentations_time, sinter_time, speedup = benchmark_transform(
                    "Crop",
                    A.Compose([A.Crop(x_min=10, y_min=10, x_max=410, y_max=410, p=1.0)]),
                    lambda: Crop(x=(10), y=(10), width=(400), height=(400)),
                    img
                )
                print_row("Crop(512)", albumentations_time, sinter_time, speedup)
            else:
                albumentations_time, sinter_time, speedup = benchmark_transform("Crop", None, lambda: Crop(x=(10), y=(10), width=(400), height=(400)), img)
                print_row("Crop(512)", albumentations_time, sinter_time, speedup)

        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Pad",
                A.Compose([A.PadIfNeeded(min_height=height+20, min_width=width+20, border_mode=0, p=1.0)]),
                lambda: Pad(top=(10), bottom=(10), left=(10), right=(10), mode=PadMode.constant(0)),
                img
            )
            print_row("Pad", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("Pad", None, lambda: Pad(top=(10), bottom=(10), left=(10), right=(10), mode=PadMode.constant(0)), img)
            print_row("Pad", albumentations_time, sinter_time, speedup)

        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Transpose",
                A.Compose([A.Transpose(p=1.0)]),
                lambda: Transpose(),
                img
            )
            print_row("Transpose", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("Transpose", None, lambda: Transpose(), img)
            print_row("Transpose", albumentations_time, sinter_time, speedup)

        # Affine - has Albumentations counterpart
        # Using similar parameters: scale=1.2, rotate=15°, translate=(10, 10), shear=(5, 5)
        if HAS_ALBUMENTATIONS:
            import cv2
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Affine",
                A.Compose([A.Affine(scale=(1.2, 1.2), rotate=15, translate_percent=(0.02, 0.02), shear=(5, 5), interpolation=cv2.INTER_LINEAR, p=1.0)]),
                lambda: Affine(scale=(1.2, 1.2), rotate=15.0, translate=(10.0, 10.0), shear=(5.0, 5.0), interpolation=Interpolation.BILINEAR),
                img
            )
            print_row("Affine", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Affine",
                None,
                lambda: Affine(scale=(1.2, 1.2), rotate=15.0, translate=(10.0, 10.0), shear=(5.0, 5.0), interpolation=Interpolation.BILINEAR),
                img
            )
            print_row("Affine", albumentations_time, sinter_time, speedup)

        # Kernel Transforms
        print("\n[KERNEL TRANSFORMS]")
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "GaussianBlur(3x3)",
                A.Compose([A.GaussianBlur(blur_limit=(3, 3), p=1.0)]),
                lambda: GaussianBlur(kernel_size=3),
                img
            )
            print_row("GaussianBlur(3x3)", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "GaussianBlur(5x5)",
                A.Compose([A.GaussianBlur(blur_limit=(5, 5), p=1.0)]),
                lambda: GaussianBlur(kernel_size=5),
                img
            )
            print_row("GaussianBlur(5x5)", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "GaussianBlur(7x7)",
                A.Compose([A.GaussianBlur(blur_limit=(7, 7), p=1.0)]),
                lambda: GaussianBlur(kernel_size=7),
                img
            )
            print_row("GaussianBlur(7x7)", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("GaussianBlur(3x3)", None, lambda: GaussianBlur(kernel_size=3), img)
            print_row("GaussianBlur(3x3)", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("GaussianBlur(5x5)", None, lambda: GaussianBlur(kernel_size=5), img)
            print_row("GaussianBlur(5x5)", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("GaussianBlur(7x7)", None, lambda: GaussianBlur(kernel_size=7), img)
            print_row("GaussianBlur(7x7)", albumentations_time, sinter_time, speedup)

        # MedianBlur - Albumentations equivalent exists
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "MedianBlur(3x3)",
                A.Compose([A.MedianBlur(blur_limit=(3, 3), p=1.0)]),
                lambda: MedianBlur(kernel_size=3),
                img
            )
            print_row("MedianBlur(3x3)", albumentations_time, sinter_time, speedup)

            albumentations_time, sinter_time, speedup = benchmark_transform(
                "MedianBlur(5x5)",
                A.Compose([A.MedianBlur(blur_limit=(5, 5), p=1.0)]),
                lambda: MedianBlur(kernel_size=5),
                img
            )
            print_row("MedianBlur(5x5)", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("MedianBlur(3x3)", None, lambda: MedianBlur(kernel_size=3), img)
            print_row("MedianBlur(3x3)", albumentations_time, sinter_time, speedup)
            albumentations_time, sinter_time, speedup = benchmark_transform("MedianBlur(5x5)", None, lambda: MedianBlur(kernel_size=5), img)
            print_row("MedianBlur(5x5)", albumentations_time, sinter_time, speedup)

        # Sharpen - Albumentations equivalent exists
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Sharpen",
                A.Compose([A.Sharpen(alpha=(0.5, 0.5), lightness=(0.5, 0.5), p=1.0)]),
                lambda: Sharpen(strength=1.0),
                img
            )
            print_row("Sharpen", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("Sharpen", None, lambda: Sharpen(strength=1.0), img)
            print_row("Sharpen", albumentations_time, sinter_time, speedup)

        # Emboss - Albumentations equivalent exists
        if HAS_ALBUMENTATIONS:
            albumentations_time, sinter_time, speedup = benchmark_transform(
                "Emboss",
                A.Compose([A.Emboss(alpha=(0.5, 0.5), strength=(0.5, 0.5), p=1.0)]),
                lambda: Emboss(direction=EmbossDirection.BOTTOM_RIGHT, strength=1.0),
                img
            )
            print_row("Emboss", albumentations_time, sinter_time, speedup)
        else:
            albumentations_time, sinter_time, speedup = benchmark_transform("Emboss", None, lambda: Emboss(direction=EmbossDirection.BOTTOM_RIGHT, strength=1.0), img)
            print_row("Emboss", albumentations_time, sinter_time, speedup)

        # EdgeDetection - sinter only (no albumentations equivalent)
        albumentations_time, sinter_time, speedup = benchmark_transform("EdgeDetection", None, lambda: EdgeDetection(method=EdgeMethod.LAPLACIAN), img)
        print_row("EdgeDetection", albumentations_time, sinter_time, speedup)

        print()

if __name__ == "__main__":
    # Parse filter arguments
    args = sys.argv[1:]

    categories = []
    transforms = []

    for arg in args:
        if arg.startswith("--"):
            categories.append(arg[2:].lower())
        else:
            transforms.append(arg.lower())

    if categories or transforms:
        _FILTER_STATE["cats"] = categories
        _FILTER_STATE["names"] = transforms
        print(f"Filter: categories={categories}, transforms={transforms}")
        print()

    run_benchmarks()
