#!/usr/bin/env python3
"""Detect whether the installed sinter extension matches the current source.

The .venv extension has repeatedly been replaced by a stale build
(pre-338d777, restored from a uv wheel cache), which fails the Python suite
with ~64 errors. This probe checks *behavior*, so it works regardless of
build hash. It verifies two things the stale build gets wrong:

  1. GaussianBlur(7) is bit-exact vs the integer two-pass reference
     (the stale build uses a different 7x7 kernel).
  2. Stochastic ops seeded differently produce different output
     (the stale build predates per-pipeline seeding).

Usage: python scripts/verify_sinter_install.py
Exit code 0  = installed build matches current source behavior.
Exit code 1  = stale/broken build (rebuild with scripts/rebuild_sinter.sh).
"""

import sys

import numpy as np

from sinter import Compose, GaussianBlur, SaltAndPepper


def exact_two_pass(a, kernel, scale):
    k = np.array(kernel, dtype=np.int64)
    r = len(k) // 2
    H, W, C = a.shape
    tmp = np.zeros_like(a, dtype=np.int64)
    res = np.zeros_like(a, dtype=np.int64)
    for x in range(W):
        for kk, kv in enumerate(k):
            xx = min(max(x + kk - r, 0), W - 1)
            tmp[:, x, :] += a[:, xx, :].astype(np.int64) * kv
    tmp = np.clip(tmp // scale, 0, 255)
    for y in range(H):
        for kk, kv in enumerate(k):
            yy = min(max(y + kk - r, 0), H - 1)
            res[y, :, :] += tmp[yy, :, :] * kv
    return np.clip(res // scale, 0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(2026)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    problems = []

    # 1) GaussianBlur(7) kernel check.
    got = Compose([GaussianBlur(kernel_size=7)]).apply(img.copy())
    ref = exact_two_pass(img, [2, 7, 14, 18, 14, 7, 2], 64)
    d = np.abs(got.astype(int) - ref.astype(int))
    if d.max() != 0:
        problems.append(
            f"GaussianBlur(7) differs from the exact reference (max diff {d.max()})"
        )

    # 2) Stochastic seeding check.
    p = Compose([SaltAndPepper(amount=0.05, salt_vs_pepper=0.5)])
    a1 = p.sample_with_seed(1).apply(img.copy())
    a2 = p.sample_with_seed(1).apply(img.copy())
    b = p.sample_with_seed(2).apply(img.copy())
    if not np.array_equal(a1, a2):
        problems.append("SaltAndPepper is not reproducible for a fixed seed")
    elif np.array_equal(a1, b):
        problems.append("SaltAndPepper produces identical output for different seeds")

    if problems:
        print("STALE/BROKEN sinter install:")
        for p in problems:
            print(f"  - {p}")
        print("Rebuild with: scripts/rebuild_sinter.sh")
        return 1

    print("OK: installed sinter matches current source behavior")
    return 0


if __name__ == "__main__":
    sys.exit(main())
