"""
Grayscale (single-channel) benchmark.

The main benchmark scripts only exercise RGB; this covers the C=1 paths
(NEON fallbacks, scalar tails) against OpenCV single-threaded.

Usage:
    python benchmark_grayscale.py [--sizes 1024,512,128]
"""

import sys
import time
import numpy as np
import cv2

cv2.setNumThreads(0)

from sinter import (
    Compose,
    GaussianBlur,
    MedianBlur,
    Invert,
    Transpose,
    HorizontalFlip,
    Rotate,
    RotateAngle,
)


def timeit_min(fn, img, runs, batches=5, warmup=3):
    """Min-of-batches timing. Both sides receive the SAME input so harness
    overhead (a per-call copy for in-place ops) cancels in ratios."""
    for _ in range(warmup):
        fn(img.copy())
    best = float("inf")
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(runs):
            fn(img.copy())
        batch = (time.perf_counter() - start) / runs * 1000
        best = min(best, batch)
    return best


def main():
    sizes = [1024, 512, 128]
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--sizes"):
            if "=" in arg:
                val = arg.split("=")[1]
            else:
                i += 1
                val = args[i]
            sizes = [int(x) for x in val.split(",")]
        i += 1

    print(f"{'op':24s} {'size':>6s} {'sinter(ms)':>11s} {'opencv(ms)':>11s} {'ratio':>7s}")
    for n in sizes:
        img = np.random.randint(0, 256, (n, n), dtype=np.uint8)
        runs = 50 if n <= 512 else 20

        cases = [
            ("GaussianBlur3x3", Compose([GaussianBlur(kernel_size=3)]),
             lambda x: cv2.GaussianBlur(x, (3, 3), 0)),
            ("GaussianBlur5x5", Compose([GaussianBlur(kernel_size=5)]),
             lambda x: cv2.GaussianBlur(x, (5, 5), 0)),
            ("GaussianBlur7x7", Compose([GaussianBlur(kernel_size=7)]),
             lambda x: cv2.GaussianBlur(x, (7, 7), 0)),
            ("MedianBlur3x3", Compose([MedianBlur(kernel_size=3)]),
             lambda x: cv2.medianBlur(x, 3)),
            ("MedianBlur5x5", Compose([MedianBlur(kernel_size=5)]),
             lambda x: cv2.medianBlur(x, 5)),
            ("Invert", Compose([Invert()]), lambda x: cv2.bitwise_not(x)),
            ("Transpose", Compose([Transpose()]), lambda x: cv2.transpose(x)),
            ("HorizontalFlip", Compose([HorizontalFlip()]), lambda x: cv2.flip(x, 1)),
            ("Rotate90", Compose([Rotate(angle=RotateAngle.ROTATE_90)]),
             lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)),
        ]

        for name, sinter_pipe, cv_fn in cases:
            def s(x):
                # x[:, :, None] is already C-contiguous; no extra copy needed.
                return sinter_pipe.apply(x[:, :, None])[:, :, 0]
            st = timeit_min(s, img, runs)
            ct = timeit_min(cv_fn, img, runs)
            print(f"{name:24s} {n:6d} {st:11.3f} {ct:11.3f} {st / ct:6.2f}x")


if __name__ == "__main__":
    main()
