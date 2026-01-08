"""
Fusion benchmark comparing Albumentations vs Sinter.
Tests pipeline fusion benefits - multiple transforms composed together.
"""
import time
import numpy as np

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed")

from sinter import (
    Brightness, Contrast, Solarize, Invert, Posterize, Gamma, Equalize, AutoContrast,
    HorizontalFlip, VerticalFlip, Compose, Transpose,
    ToSepia, HueSaturationValue, ColorTemperature, ColorTint, ToGray,
    GaussNoise, RGBShift, SaltAndPepper, CoarseDropout,
    GaussianBlur, MedianBlur, Sharpen,
    Pad, PadMode, Resize, Crop,
)

def benchmark(image_size=(512, 512, 3), iterations=100):
    image = np.random.randint(0, 256, image_size, dtype=np.uint8)
    # Pre-allocate copies for sinter benchmarks (albumentations creates copies internally)
    image_copies = [image.copy() for _ in range(iterations + 10)]

    print("=" * 70)
    print(f"FUSION BENCHMARK - {image_size[0]}x{image_size[1]}x{image_size[2]}")
    print("=" * 70)

    # Test 1: LUT-only pipeline (4 transforms)
    print("\n1. LUT-only (4 transforms: Brightness, Contrast, Solarize, Invert)")
    print("-" * 70)

    if HAS_ALBUMENTATIONS:
        # Albumentations uses RandomBrightnessContrast as a combined transform
        # For fair comparison, we apply it twice with different params
        albumentations_pipe = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(40.0/255, 40.0/255), contrast_limit=(0.3, 0.3), p=1.0),
            A.Solarize(threshold_range=(0.5, 0.5), p=1.0),
            A.InvertImg(p=1.0),
        ])
        for _ in range(5):
            _ = albumentations_pipe(image=image)["image"]
        start = time.perf_counter()
        for _ in range(iterations):
            _ = albumentations_pipe(image=image)["image"]
        t_albumentations = (time.perf_counter() - start) / iterations * 1000
        print(f"  Albumentations: {t_albumentations:.3f} ms")

    sinter_pipe = Compose([
        Brightness(delta=40.0),
        Contrast(factor=1.3),
        Solarize(threshold=128),
        Invert(),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    if HAS_ALBUMENTATIONS:
        print(f"  → Sinter is {t_albumentations/t_sinter:.2f}x {'faster' if t_albumentations > t_sinter else 'slower'}")

    # Test 2: Heavy LUT (8 transforms)
    print("\n2. Heavy LUT (8 transforms)")
    print("-" * 70)

    if HAS_ALBUMENTATIONS:
        albumentations_pipe = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(30.0/255, 30.0/255), contrast_limit=(0.2, 0.2), p=1.0),
            A.Solarize(threshold_range=(0.4, 0.4), p=1.0),
            A.Posterize(num_bits=6, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(20.0/255, 20.0/255), contrast_limit=(-0.2, -0.2), p=1.0),
            A.Solarize(threshold_range=(0.6, 0.6), p=1.0),
            A.InvertImg(p=1.0),
        ])
        for _ in range(5):
            _ = albumentations_pipe(image=image)["image"]
        start = time.perf_counter()
        for _ in range(iterations):
            _ = albumentations_pipe(image=image)["image"]
        t_albumentations = (time.perf_counter() - start) / iterations * 1000
        print(f"  Albumentations: {t_albumentations:.3f} ms")

    sinter_pipe = Compose([
        Brightness(delta=30.0),
        Contrast(factor=1.2),
        Solarize(threshold=100),
        Posterize(bits=6),
        Brightness(delta=20.0),
        Contrast(factor=0.8),
        Solarize(threshold=150),
        Invert(),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    if HAS_ALBUMENTATIONS:
        print(f"  → Sinter is {t_albumentations/t_sinter:.2f}x {'faster' if t_albumentations > t_sinter else 'slower'}")

    # Test 3: Geometric + LUT mix
    print("\n3. Mixed: Geometric + LUT (Flip, Brightness, Contrast)")
    print("-" * 70)

    if HAS_ALBUMENTATIONS:
        albumentations_pipe = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(40.0/255, 40.0/255), contrast_limit=(0.3, 0.3), p=1.0),
        ])
        for _ in range(5):
            _ = albumentations_pipe(image=image)["image"]
        start = time.perf_counter()
        for _ in range(iterations):
            _ = albumentations_pipe(image=image)["image"]
        t_albumentations = (time.perf_counter() - start) / iterations * 1000
        print(f"  Albumentations: {t_albumentations:.3f} ms")

    sinter_pipe = Compose([
        HorizontalFlip(),
        Brightness(delta=40.0),
        Contrast(factor=1.3),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    if HAS_ALBUMENTATIONS:
        print(f"  → Sinter is {t_albumentations/t_sinter:.2f}x {'faster' if t_albumentations > t_sinter else 'slower'}")

    # Test 5: Full ImageNet-style pipeline
    print("\n5. Full ImageNet-style (Flip, Brightness, Contrast)")
    print("-" * 70)

    if HAS_ALBUMENTATIONS:
        albumentations_pipe = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(40.0/255, 40.0/255), contrast_limit=(0.3, 0.3), p=1.0),
        ])
        for _ in range(5):
            _ = albumentations_pipe(image=image)["image"]
        start = time.perf_counter()
        for _ in range(iterations):
            _ = albumentations_pipe(image=image)["image"]
        t_albumentations = (time.perf_counter() - start) / iterations * 1000
        print(f"  Albumentations: {t_albumentations:.3f} ms")

    sinter_pipe = Compose([
        HorizontalFlip(),
        Brightness(delta=40.0),
        Contrast(factor=1.3),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    if HAS_ALBUMENTATIONS:
        print(f"  → Sinter is {t_albumentations/t_sinter:.2f}x {'faster' if t_albumentations > t_sinter else 'slower'}")

    print("\n" + "=" * 70)

    # Test 6: Matrix fusion (ToSepia only - single transform baseline)
    print("\n6. Matrix Transform - ToSepia (single transform)")
    print("-" * 70)

    if HAS_ALBUMENTATIONS:
        albumentations_pipe = A.Compose([
            A.ToSepia(p=1.0),
        ])
        for _ in range(5):
            _ = albumentations_pipe(image=image)["image"]
        start = time.perf_counter()
        for _ in range(iterations):
            _ = albumentations_pipe(image=image)["image"]
        t_albumentations = (time.perf_counter() - start) / iterations * 1000
        print(f"  Albumentations: {t_albumentations:.3f} ms")

    sinter_pipe = Compose([ToSepia()])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    if HAS_ALBUMENTATIONS:
        print(f"  → Sinter is {t_albumentations/t_sinter:.2f}x {'faster' if t_albumentations > t_sinter else 'slower'}")

    # Test 6b: Matrix fusion (ToSepia + Saturation - 2 matrix ops)
    print("\n6b. Matrix Fusion (ToSepia + Saturation - 2 matrix ops)")
    print("-" * 70)

    if HAS_ALBUMENTATIONS:
        albumentations_pipe = A.Compose([
            A.ToSepia(p=1.0),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(30, 30), val_shift_limit=0, p=1.0),
        ])
        for _ in range(5):
            _ = albumentations_pipe(image=image)["image"]
        start = time.perf_counter()
        for _ in range(iterations):
            _ = albumentations_pipe(image=image)["image"]
        t_albumentations = (time.perf_counter() - start) / iterations * 1000
        print(f"  Albumentations: {t_albumentations:.3f} ms")

    sinter_pipe = Compose([
        ToSepia(),
        HueSaturationValue(hue_shift=0, saturation_scale=1.3, value_scale=1.0),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    if HAS_ALBUMENTATIONS:
        print(f"  → Sinter is {t_albumentations/t_sinter:.2f}x {'faster' if t_albumentations > t_sinter else 'slower'}")

    # Test 6c: Heavy matrix fusion (4 matrix ops)
    print("\n6c. Heavy Matrix Fusion (ToSepia + Saturation + ColorTemp + ColorTint)")
    print("-" * 70)

    sinter_pipe = Compose([
        ToSepia(),
        HueSaturationValue(hue_shift=0, saturation_scale=1.3, value_scale=1.0),
        ColorTemperature(temperature=50),
        ColorTint(tint=[200, 180, 150, 0.5]),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")

    # Test 7: Mixed Matrix + LUT fusion
    print("\n7. Mixed Fusion (Matrix + LUT: ToSepia, Brightness, Solarize)")
    print("-" * 70)

    if HAS_ALBUMENTATIONS:
        albumentations_pipe = A.Compose([
            A.ToSepia(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(40.0/255, 40.0/255), contrast_limit=(0, 0), p=1.0),
            A.Solarize(threshold_range=(0.5, 0.5), p=1.0),
        ])
        for _ in range(5):
            _ = albumentations_pipe(image=image)["image"]
        start = time.perf_counter()
        for _ in range(iterations):
            _ = albumentations_pipe(image=image)["image"]
        t_albumentations = (time.perf_counter() - start) / iterations * 1000
        print(f"  Albumentations: {t_albumentations:.3f} ms")

    sinter_pipe = Compose([
        ToSepia(),
        Brightness(delta=40.0),
        Solarize(threshold=128),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    if HAS_ALBUMENTATIONS:
        print(f"  → Sinter is {t_albumentations/t_sinter:.2f}x {'faster' if t_albumentations > t_sinter else 'slower'}")

    print("\n" + "=" * 70)

    # Test 10: Structural composition (FlipH + FlipV → Rot180)
    print("\n10. Structural Composition (FlipH + FlipV should compose to Rot180)")
    print("-" * 70)

    sinter_pipe = Compose([
        HorizontalFlip(),
        VerticalFlip(),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    print(f"  Pipeline: {sinter_pipe}")

    # Test 11: Heavy LUT Fusion (10 LUT transforms)
    print("\n11. Heavy LUT Fusion (10 LUT transforms)")
    print("-" * 70)

    sinter_pipe = Compose([
        Brightness(delta=10),
        Contrast(factor=1.1),
        Solarize(threshold=100),
        Invert(),
        Posterize(bits=7),
        Brightness(delta=-10),
        Contrast(factor=0.9),
        Solarize(threshold=200),
        Invert(),
        Posterize(bits=6),
    ])
    for i in range(5):
        _ = sinter_pipe.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = sinter_pipe.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter: {t_sinter:.3f} ms")
    print(f"  Pipeline: {sinter_pipe}")

    print("\n" + "=" * 70)

    # Test 12: Heavy pipeline with EQUIVALENT transforms (fair comparison)
    print("\n12. HEAVY PIPELINE (equivalent transforms - fair comparison)")
    print("-" * 70)

    # Both libraries use the SAME transforms for fair comparison
    if HAS_ALBUMENTATIONS:
        albumentations_pipe = A.Compose([
            # Geometric
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Transpose(p=1.0),
            # LUT transforms
            A.RandomBrightnessContrast(brightness_limit=(30.0/255, 30.0/255), contrast_limit=(0.2, 0.2), p=1.0),
            A.Solarize(threshold_range=(0.5, 0.5), p=1.0),
            A.InvertImg(p=1.0),
            A.Posterize(num_bits=6, p=1.0),
            A.Equalize(p=1.0),
            A.AutoContrast(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(20.0/255, 20.0/255), contrast_limit=(-0.2, -0.2), p=1.0),
            # Color
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=10, p=1.0),
            # Noise & Blur
            A.GaussNoise(std_range=(15.0/255, 15.0/255), p=1.0),
            A.GaussianBlur(blur_limit=(5, 5), p=1.0),
            A.MedianBlur(blur_limit=(3, 3), p=1.0),
            A.ToGray(p=1.0),
        ])
        for _ in range(5):
            _ = albumentations_pipe(image=image)["image"]
        start = time.perf_counter()
        for _ in range(iterations):
            _ = albumentations_pipe(image=image)["image"]
        t_albumentations = (time.perf_counter() - start) / iterations * 1000
        print(f"  Albumentations: {t_albumentations:.3f} ms (14 transforms)")

    v2_heavy = Compose([
        # Geometric
        HorizontalFlip(),
        VerticalFlip(),
        Transpose(),
        # LUT transforms
        Brightness(delta=30),
        Contrast(factor=1.2),
        Solarize(threshold=128),
        Invert(),
        Posterize(bits=6),
        Equalize(),
        AutoContrast(),
        Brightness(delta=-20),
        Contrast(factor=0.8),
        # Color
        HueSaturationValue(hue_shift=10, saturation_scale=1.3, value_scale=1.1),
        # Noise & Blur
        GaussNoise(mean=0.0, std=15.0),
        GaussianBlur(kernel_size=5),
        MedianBlur(kernel_size=3),
        ToGray(),
    ])

    print(f"  Pipeline: {v2_heavy}")

    for i in range(5):
        _ = v2_heavy.apply(image_copies[i])
    start = time.perf_counter()
    for i in range(iterations):
        _ = v2_heavy.apply(image_copies[i + 5])
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter:         {t_sinter:.3f} ms (16 transforms)")

    if HAS_ALBUMENTATIONS:
        print(f"  → Sinter is {t_albumentations/t_sinter:.2f}x {'faster' if t_albumentations > t_sinter else 'slower'}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    # Small image
    benchmark((256, 256, 3), 100)

    # Large image
    benchmark((512, 512, 3), 100)

    # XL image
    benchmark((1024, 1024, 3), 50)
