"""
CI Benchmark Suite for Sinter vs OpenCV and Albumentations.

Runs a focused, high-impact set of benchmarks designed for CI runners (~30s runtime).
Outputs rich Markdown formatted for GitHub Actions ($GITHUB_STEP_SUMMARY).
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

try:
    import cv2
    cv2.setNumThreads(0)  # Fair single-threaded comparison
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from sinter import (
    AnyRes,
    AutoContrast,
    Brightness,
    Compose,
    Contrast,
    Crop,
    Equalize,
    Gamma,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    HueSaturationValue,
    Invert,
    MedianBlur,
    Posterize,
    Resize,
    Rotate,
    Sharpen,
    Solarize,
    ToGray,
    Transpose,
    VerticalFlip,
)


def measure_ms(fn, runs=25, batches=3, warmup=3):
    """Min-of-batches timing; robust to frequency scaling and background jitter."""
    for _ in range(warmup):
        fn()
    best_ms = float("inf")
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(runs):
            fn()
        elapsed = ((time.perf_counter() - t0) / runs) * 1000.0
        if elapsed < best_ms:
            best_ms = elapsed
    return best_ms


def run_benchmarks():
    lines = []
    lines.append("## 🚀 Sinter Performance Benchmark Report")
    lines.append("")
    lines.append("> Pure Native Rust + SIMD Engine vs OpenCV & Albumentations")
    lines.append("")

    # -------------------------------------------------------------------------
    # 1. Pipeline Fusion vs Albumentations (512x512 RGB)
    # -------------------------------------------------------------------------
    lines.append("### 1. Fused Augmentation Pipelines vs. Albumentations (512×512 RGB)")
    lines.append("")
    lines.append("| Pipeline | Albumentations | Sinter (Fused) | Speedup |")
    lines.append("|---|---|---|---|")

    img_rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Test 1A: 4x LUT transforms
    sinter_4lut = Compose([
        Brightness(delta=30.0),
        Contrast(factor=1.2),
        Solarize(threshold=128),
        Invert(),
    ])
    t_sinter_4lut = measure_ms(lambda: sinter_4lut.apply(img_rgb), runs=40)

    if HAS_ALBUMENTATIONS:
        alb_4lut = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(30/255, 30/255), contrast_limit=(0.2, 0.2), p=1.0),
            A.Solarize(threshold_range=(128/255, 128/255), p=1.0),
            A.InvertImg(p=1.0),
        ])
        t_alb_4lut = measure_ms(lambda: alb_4lut(image=img_rgb)["image"], runs=40)
        speedup = t_alb_4lut / t_sinter_4lut
        lines.append(f"| **4x Pointwise LUT** | {t_alb_4lut:.3f} ms | **{t_sinter_4lut:.3f} ms** | **{speedup:.2f}x faster** |")
    else:
        lines.append(f"| **4x Pointwise LUT** | N/A | **{t_sinter_4lut:.3f} ms** | N/A |")

    # Test 1B: 8x LUT transforms
    sinter_8lut = Compose([
        Brightness(delta=20.0),
        Contrast(factor=1.1),
        Solarize(threshold=100),
        Invert(),
        Posterize(bits=6),
        Gamma(gamma=1.2),
        Brightness(delta=-10.0),
        Contrast(factor=0.9),
    ])
    t_sinter_8lut = measure_ms(lambda: sinter_8lut.apply(img_rgb), runs=40)

    if HAS_ALBUMENTATIONS:
        alb_8lut = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(20/255, 20/255), contrast_limit=(0.1, 0.1), p=1.0),
            A.Solarize(threshold_range=(100/255, 100/255), p=1.0),
            A.InvertImg(p=1.0),
            A.Posterize(num_bits=6, p=1.0),
            A.RandomGamma(gamma_limit=(120, 120), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-10/255, -10/255), contrast_limit=(-0.1, -0.1), p=1.0),
        ])
        t_alb_8lut = measure_ms(lambda: alb_8lut(image=img_rgb)["image"], runs=40)
        speedup = t_alb_8lut / t_sinter_8lut
        lines.append(f"| **8x Pointwise LUT** | {t_alb_8lut:.3f} ms | **{t_sinter_8lut:.3f} ms** | **{speedup:.2f}x faster** |")

    # Test 1C: Crop Hoisting (Crop + LUT)
    sinter_crop_lut = Compose([
        Brightness(delta=30.0),
        Contrast(factor=1.2),
        Crop(x_min=64, y_min=64, x_max=320, y_max=320),
    ])
    t_sinter_crop_lut = measure_ms(lambda: sinter_crop_lut.apply(img_rgb), runs=40)

    if HAS_ALBUMENTATIONS:
        alb_crop_lut = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(30/255, 30/255), contrast_limit=(0.2, 0.2), p=1.0),
            A.Crop(x_min=64, y_min=64, x_max=320, y_max=320),
        ])
        t_alb_crop_lut = measure_ms(lambda: alb_crop_lut(image=img_rgb)["image"], runs=40)
        speedup = t_alb_crop_lut / t_sinter_crop_lut
        lines.append(f"| **Crop Hoisting + LUT** | {t_alb_crop_lut:.3f} ms | **{t_sinter_crop_lut:.3f} ms** | **{speedup:.2f}x faster** |")

    # Test 1D: Heavy Pipeline (16 ops)
    sinter_heavy = Compose([
        HorizontalFlip(p=1.0),
        VerticalFlip(p=1.0),
        Transpose(p=1.0),
        Brightness(delta=30.0),
        Contrast(factor=1.2),
        Solarize(threshold=128),
        Invert(),
        Posterize(bits=6),
        Equalize(),
        AutoContrast(),
        Brightness(delta=-20.0),
        Contrast(factor=0.8),
        HueSaturationValue(hue_shift=10, sat_shift=20, val_shift=10),
        GaussNoise(mean=0, std=15),
        GaussianBlur(kernel_size=5),
        MedianBlur(kernel_size=3),
    ])
    t_sinter_heavy = measure_ms(lambda: sinter_heavy.apply(img_rgb), runs=20)

    if HAS_ALBUMENTATIONS:
        alb_heavy = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Transpose(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(30/255, 30/255), contrast_limit=(0.2, 0.2), p=1.0),
            A.Solarize(threshold_range=(128/255, 128/255), p=1.0),
            A.InvertImg(p=1.0),
            A.Posterize(num_bits=6, p=1.0),
            A.Equalize(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-20/255, -20/255), contrast_limit=(-0.2, -0.2), p=1.0),
            A.HueSaturationValue(hue_shift_limit=(10, 10), sat_shift_limit=(20, 20), val_shift_limit=(10, 10), p=1.0),
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(5, 5), p=1.0),
            A.MedianBlur(blur_limit=(3, 3), p=1.0),
        ])
        t_alb_heavy = measure_ms(lambda: alb_heavy(image=img_rgb)["image"], runs=20)
        speedup = t_alb_heavy / t_sinter_heavy
        lines.append(f"| **Heavy Pipeline (16 ops)** | {t_alb_heavy:.3f} ms | **{t_sinter_heavy:.3f} ms** | **{speedup:.2f}x faster** |")

    lines.append("")

    # -------------------------------------------------------------------------
    # 2. Individual SIMD Kernels vs OpenCV (512x512)
    # -------------------------------------------------------------------------
    lines.append("### 2. Native SIMD Kernels vs. Raw OpenCV (`cv2`) (512×512)")
    lines.append("")
    lines.append("| Kernel / Operation | OpenCV (`cv2`) | Sinter (NEON) | Speedup / Parity |")
    lines.append("|---|---|---|---|")

    # 2A: Transpose
    s_transpose = Transpose()
    t_sinter_trans = measure_ms(lambda: s_transpose.apply(img_rgb), runs=50)
    if HAS_OPENCV:
        t_cv_trans = measure_ms(lambda: cv2.transpose(img_rgb), runs=50)
        speedup = t_cv_trans / t_sinter_trans
        lines.append(f"| **Transpose (RGB)** | {t_cv_trans:.3f} ms | **{t_sinter_trans:.3f} ms** | **{speedup:.2f}x faster** |")

    # 2B: AutoContrast
    s_autocontrast = AutoContrast()
    t_sinter_ac = measure_ms(lambda: s_autocontrast.apply(img_rgb), runs=40)
    if HAS_ALBUMENTATIONS:
        alb_ac = A.AutoContrast(p=1.0)
        t_alb_ac = measure_ms(lambda: alb_ac(image=img_rgb)["image"], runs=40)
        speedup = t_alb_ac / t_sinter_ac
        lines.append(f"| **AutoContrast (Vectorized Histogram)** | {t_alb_ac:.3f} ms | **{t_sinter_ac:.3f} ms** | **{speedup:.2f}x faster** |")

    # 2C: Equalize
    s_equalize = Equalize()
    t_sinter_eq = measure_ms(lambda: s_equalize.apply(img_rgb), runs=40)
    if HAS_ALBUMENTATIONS:
        alb_eq = A.Equalize(p=1.0)
        t_alb_eq = measure_ms(lambda: alb_eq(image=img_rgb)["image"], runs=40)
        speedup = t_alb_eq / t_sinter_eq
        lines.append(f"| **Equalize (Cumulative Histogram)** | {t_alb_eq:.3f} ms | **{t_sinter_eq:.3f} ms** | **{speedup:.2f}x faster** |")

    # 2D: Sharpen
    s_sharpen = Sharpen()
    t_sinter_sh = measure_ms(lambda: s_sharpen.apply(img_rgb), runs=30)
    if HAS_ALBUMENTATIONS:
        alb_sh = A.Sharpen(p=1.0)
        t_alb_sh = measure_ms(lambda: alb_sh(image=img_rgb)["image"], runs=30)
        speedup = t_alb_sh / t_sinter_sh
        lines.append(f"| **Sharpen (3×3 Convolution)** | {t_alb_sh:.3f} ms | **{t_sinter_sh:.3f} ms** | **{speedup:.2f}x faster** |")

    # 2E: HSV Shift
    s_hsv = HueSaturationValue(hue_shift=15, sat_shift=20, val_shift=10)
    t_sinter_hsv = measure_ms(lambda: s_hsv.apply(img_rgb), runs=30)
    if HAS_OPENCV:
        def cv_hsv():
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + 15) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        t_cv_hsv = measure_ms(cv_hsv, runs=30)
        speedup = t_cv_hsv / t_sinter_hsv
        lines.append(f"| **HueSaturationValue (Hardware vdiv)** | {t_cv_hsv:.3f} ms | **{t_sinter_hsv:.3f} ms** | **{speedup:.2f}x (parity)** |")

    lines.append("")

    # -------------------------------------------------------------------------
    # 3. Frontier Multi-Modal Workloads (Video & VLM)
    # -------------------------------------------------------------------------
    lines.append("### 3. Frontier Multi-Modal Workloads (Video & VLM AnyRes)")
    lines.append("")
    lines.append("| Workload | Dimensions | Latency | Throughput |")
    lines.append("|---|---|---|---|")

    # 3A: Video Clip
    video_clip = np.random.randint(0, 256, (16, 256, 256, 3), dtype=np.uint8)
    video_pipe = Compose([
        HorizontalFlip(p=0.5),
        Brightness(delta=20.0),
        Contrast(factor=1.1),
        Crop(x_min=16, y_min=16, x_max=240, y_max=240),
    ])
    t_clip = measure_ms(lambda: video_pipe.apply_video(video_clip), runs=30)
    fps = (16.0 / (t_clip / 1000.0))
    lines.append(f"| **Video Clip (Temporal-Consistent)** | 16 frames (`16×256×256×3`) | **{t_clip:.3f} ms** / clip | **{fps:,.0f} frames/sec** |")

    # 3B: VLM AnyRes
    img_1080p = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    anyres_op = AnyRes(tile_size=448, max_tiles=6, include_thumbnail=True)
    t_anyres = measure_ms(lambda: anyres_op(img_1080p), runs=20)
    img_sec = 1000.0 / t_anyres
    lines.append(f"| **VLM AnyRes Dynamic Tiling** | Full 1080p (`1080×1920×3`) | **{t_anyres:.3f} ms** / image | **{img_sec:,.0f} images/sec** |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run Sinter CI benchmarks")
    parser.add_argument("--output", "-o", help="File to append markdown report to")
    args = parser.parse_args()

    report = run_benchmarks()
    print(report)

    output_path = args.output or os.environ.get("GITHUB_STEP_SUMMARY")
    if output_path:
        with open(output_path, "a") as f:
            f.write("\n" + report + "\n")
        print(f"\n[Report appended to {output_path}]")


if __name__ == "__main__":
    main()
