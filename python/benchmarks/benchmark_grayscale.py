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


def timeit(fn, img, runs):
    for _ in range(3):
        fn(img.copy())
    t = time.perf_counter()
    for _ in range(runs):
        fn(img.copy())
    return (time.perf_counter() - t) / runs * 1000


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
            ("Invert", Compose([Invert()]), lambda x: 255 - x),
            ("Transpose", Compose([Transpose()]), lambda x: x.T.copy()),
            ("HorizontalFlip", Compose([HorizontalFlip()]), lambda x: x[:, ::-1].copy()),
            ("Rotate90", Compose([Rotate(angle=RotateAngle.ROTATE_90)]),
             lambda x: np.rot90(x, 1).copy()),
        ]

        for name, sinter_pipe, cv_fn in cases:
            def s(x):
                return sinter_pipe.apply(x[:, :, None].copy())[:, :, 0]
            st = timeit(s, img, runs)
            ct = timeit(cv_fn, img, runs)
            print(f"{name:24s} {n:6d} {st:11.3f} {ct:11.3f} {st / ct:6.2f}x")


if __name__ == "__main__":
    main()
