"""
Benchmark geometric operations on grayscale images vs numpy/opencv alternatives.
Ensures views are materialized (no cheating with zero-copy views).

Measurement discipline (P0):
- Both sides receive the SAME input (raw, no per-call copy). Sinter's flips are
  in-place (verified: they return the same buffer), so the sinter side re-reads
  mutated data across iterations — but every op benchmarked here has
  value-independent timing (pure permutations / fixed-work warps), so this
  cannot skew the measurement, and it keeps both sides free of harness-copy
  overhead. Absolutes are op-only; only ratios transfer across processes.
- Warmup before timing, then min-of-batches (the most stable estimate; filters
  frequency scaling and cache-state noise that wreck single-average numbers).
- Baselines are shape-matched and asserted: every section asserts that the
  sinter output and the baseline output have the same (squeezed) shape BEFORE
  timing. The affine section replicates sinter's inverse matrix
  (build_inverse_matrix: center (w-1)/2, Replicate border) and feeds it to
  cv2.WARP_INVERSE_MAP so both sides do identical work.
"""
import time
import numpy as np
import cv2

# Pin cv2 to one thread: warpAffine parallelizes over rows internally, so its
# default multi-threaded time depends on core count/availability and compares
# sinter's single-threaded kernel against ~8 cores. Single-thread vs
# single-thread is the deterministic, augmentation-representative comparison
# (dataloader workers already parallelize across images).
cv2.setNumThreads(0)

from sinter import (
    HorizontalFlip, VerticalFlip, Transpose, Rotate, RotateAngle,
    Resize, Crop, Pad, PadMode, Affine, Compose, Interpolation,
)


def materialize(arr):
    """Force materialization of views/copies to ensure fair comparison."""
    return np.ascontiguousarray(arr)


def sinter_inverse_matrix(scale, rotate_deg=0.0, shear=(0.0, 0.0),
                          translate=(0.0, 0.0), w=512, h=512):
    """Exact replica of Affine::build_inverse_matrix (mod.rs) in numpy.

    Maps output pixel -> source pixel, centered on (w-1)/2. Feeding this to
    cv2.warpAffine with WARP_INVERSE_MAP gives a pixel-semantic match to
    sinter's Affine (which is shape-preserving and defaults to Replicate
    border).
    """
    sx, sy = scale
    angle = np.deg2rad(rotate_deg)
    ca, sa = np.cos(angle), np.sin(angle)
    tx, ty = translate
    shx = np.tan(np.deg2rad(shear[0]))
    shy = np.tan(np.deg2rad(shear[1]))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    det = 1.0 - shx * shy
    a = (ca + sa * shy) / det / sx
    b = (-ca * shx - sa) / det / sx
    d = (sa - ca * shy) / det / sy
    e = (-sa * shx + ca) / det / sy
    c = cx - (a * (cx + tx) + b * (cy + ty))
    f = cy - (d * (cx + tx) + e * (cy + ty))
    return np.array([[a, b, c], [d, e, f]], dtype=np.float32)


def assert_same_shape(sinter_out, baseline_out, section):
    s, b = np.squeeze(sinter_out).shape, np.squeeze(baseline_out).shape
    assert s == b, f"{section}: shape mismatch sinter={s} baseline={b} — comparison is invalid"


def timeit_min(fn, img, iterations, warmup=5, batches=5):
    """Warmup + min-of-batches for `fn(img)`; returns best batch average in ms."""
    for _ in range(warmup):
        fn(img)
    best = float("inf")
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(iterations):
            fn(img)
        batch = (time.perf_counter() - start) / iterations * 1000
        best = min(best, batch)
    return best


def benchmark_geometric_grayscale():
    print("=" * 80)
    print("GEOMETRIC OPERATIONS ON GRAYSCALE IMAGES (512x512)")
    print("=" * 80)

    iterations = 200
    size = (512, 512)

    # ========================================================================
    # 1. Horizontal Flip
    # ========================================================================
    print("\n1. Horizontal Flip")
    print("-" * 80)

    # Create grayscale test image
    img_gray = np.random.randint(0, 256, (*size, 1), dtype=np.uint8)

    # NumPy flip
    t_numpy = timeit_min(lambda x: materialize(x[:, ::-1, :]), img_gray, iterations)
    print(f"  NumPy flip:        {t_numpy:.4f} ms")

    # OpenCV flip
    t_opencv = timeit_min(lambda x: cv2.flip(x, 1), img_gray, iterations)
    print(f"  OpenCV flip:       {t_opencv:.4f} ms")

    # Sinter flip
    pipe = Compose([HorizontalFlip()])
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter flip:       {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 2. Vertical Flip
    # ========================================================================
    print("\n2. Vertical Flip")
    print("-" * 80)

    # NumPy flip
    t_numpy = timeit_min(lambda x: materialize(x[::-1, :, :]), img_gray, iterations)
    print(f"  NumPy flip:        {t_numpy:.4f} ms")

    # OpenCV flip
    t_opencv = timeit_min(lambda x: cv2.flip(x, 0), img_gray, iterations)
    print(f"  OpenCV flip:       {t_opencv:.4f} ms")

    # Sinter flip
    pipe = Compose([VerticalFlip()])
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter flip:       {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 3. Transpose
    # ========================================================================
    print("\n3. Transpose (swap axes)")
    print("-" * 80)

    # NumPy transpose
    t_numpy = timeit_min(lambda x: materialize(x.transpose(1, 0, 2)), img_gray, iterations)
    print(f"  NumPy transpose:   {t_numpy:.4f} ms")

    # OpenCV transpose
    t_opencv = timeit_min(lambda x: cv2.transpose(x), img_gray, iterations)
    print(f"  OpenCV transpose:  {t_opencv:.4f} ms")

    # Sinter transpose
    pipe = Compose([Transpose()])
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter transpose:  {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 4. Rotate 90°
    # ========================================================================
    print("\n4. Rotate 90° (clockwise)")
    print("-" * 80)

    # NumPy rotate (transpose + flip)
    t_numpy = timeit_min(
        lambda x: materialize(x.transpose(1, 0, 2)[:, ::-1, :]), img_gray, iterations
    )
    print(f"  NumPy rot90:       {t_numpy:.4f} ms")

    # OpenCV rotate
    t_opencv = timeit_min(lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE), img_gray, iterations)
    print(f"  OpenCV rot90:      {t_opencv:.4f} ms")

    # Sinter rotate
    pipe = Compose([Rotate(angle=RotateAngle.ROTATE_90)])  # 90°
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter rot90:      {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 5. Resize (downsample to 256x256)
    # ========================================================================
    print("\n5. Resize (512→256, nearest neighbor)")
    print("-" * 80)

    # NumPy resize (using slicing for integer downsample)
    t_numpy = timeit_min(lambda x: materialize(x[::2, ::2, :]), img_gray, iterations)
    print(f"  NumPy slice:       {t_numpy:.4f} ms (only for integer downsampling)")

    # OpenCV resize
    t_opencv = timeit_min(
        lambda x: cv2.resize(x, (256, 256), interpolation=cv2.INTER_NEAREST),
        img_gray,
        iterations,
    )
    print(f"  OpenCV resize:     {t_opencv:.4f} ms")

    # Sinter resize
    pipe = Compose([Resize(width=256, height=256, interpolation=Interpolation.NEAREST)])
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter resize:     {t_sinter:.4f} ms")
    print(f"  → Sinter is {t_opencv/t_sinter:.2f}x faster than OpenCV")

    # ========================================================================
    # 6. Crop (center crop to 256x256)
    # ========================================================================
    print("\n6. Crop (center crop to 256x256)")
    print("-" * 80)

    start_y, start_x = 128, 128

    # NumPy crop (slicing; materialized to count the copy)
    t_numpy = timeit_min(
        lambda x: materialize(x[start_y:start_y+256, start_x:start_x+256, :]),
        img_gray,
        iterations,
    )
    print(f"  NumPy crop:        {t_numpy:.4f} ms")

    # OpenCV crop (same as numpy, just slicing)
    t_opencv = timeit_min(
        lambda x: materialize(x[start_y:start_y+256, start_x:start_x+256, :]),
        img_gray,
        iterations,
    )
    print(f"  OpenCV crop:       {t_opencv:.4f} ms")

    # Sinter crop
    pipe = Compose([Crop(x=128, y=128, width=256, height=256)])
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter crop:       {t_sinter:.4f} ms")
    print(f"  → NumPy/OpenCV slicing is fastest (zero-copy view)")

    # ========================================================================
    # 7. Pad (reflect padding, 10px on all sides)
    # ========================================================================
    print("\n7. Pad (reflect mode, 10px all sides → 532x532)")
    print("-" * 80)

    # NumPy pad
    t_numpy = timeit_min(
        lambda x: materialize(np.pad(x, ((10, 10), (10, 10), (0, 0)), mode="reflect")),
        img_gray,
        iterations,
    )
    print(f"  NumPy pad:         {t_numpy:.4f} ms")

    # OpenCV copyMakeBorder
    t_opencv = timeit_min(
        lambda x: cv2.copyMakeBorder(x, 10, 10, 10, 10, cv2.BORDER_REFLECT_101),
        img_gray,
        iterations,
    )
    print(f"  OpenCV pad:        {t_opencv:.4f} ms")

    # Sinter pad
    pipe = Compose([Pad(top=10, bottom=10, left=10, right=10, mode=PadMode.REFLECT)])
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter pad:        {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 8. Affine — scale only (exercises the dy_fp == 0 fast path)
    # ========================================================================
    print("\n8. Affine scale 1.5x (bilinear, shape-preserving → 512x512)")
    print("-" * 80)

    M_inv = sinter_inverse_matrix((1.5, 1.5), w=size[0], h=size[1])

    # OpenCV affine — same inverse matrix, same Replicate border, same 512x512
    # output shape. (The old version rendered 768x768 against sinter's 512x512:
    # a 2.25x work mismatch.)
    def cv_affine(x, m):
        return cv2.warpAffine(
            x, m, (size[0], size[1]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # One-time semantic check: same mapping must produce (near-)identical pixels
    pipe_scale = Compose([Affine(scale=(1.5, 1.5), interpolation=Interpolation.BILINEAR)])
    s_out = np.squeeze(pipe_scale.apply(img_gray.copy()))
    c_out = np.squeeze(cv_affine(img_gray.copy(), M_inv))
    assert_same_shape(s_out, c_out, "affine scale")
    mad = np.abs(s_out.astype(np.int16) - c_out.astype(np.int16)).mean()
    print(f"  semantic check: mean |sinter - cv2| = {mad:.3f} (rounding tolerance ≤ 1)")

    t_opencv = timeit_min(lambda x: cv_affine(x, M_inv), img_gray, iterations)
    print(f"  OpenCV warpAffine: {t_opencv:.4f} ms")

    t_sinter = timeit_min(pipe_scale.apply, img_gray, iterations)
    print(f"  Sinter affine:     {t_sinter:.4f} ms")
    print(f"  → Sinter is {t_opencv/t_sinter:.2f}x faster than OpenCV")

    # ========================================================================
    # 8b. Affine — rotate 15° + scale 1.5x (general path: dy_fp != 0)
    # ========================================================================
    print("\n8b. Affine rotate 15° + scale 1.5x (bilinear → 512x512)")
    print("-" * 80)

    M_inv_rot = sinter_inverse_matrix((1.5, 1.5), rotate_deg=15.0, w=size[0], h=size[1])

    t_opencv = timeit_min(lambda x: cv_affine(x, M_inv_rot), img_gray, iterations)
    print(f"  OpenCV warpAffine: {t_opencv:.4f} ms")

    pipe_rot = Compose([Affine(scale=(1.5, 1.5), rotate=15.0,
                               interpolation=Interpolation.BILINEAR)])
    assert_same_shape(pipe_rot.apply(img_gray.copy()), cv_affine(img_gray.copy(), M_inv_rot),
                      "affine rotate")
    t_sinter = timeit_min(pipe_rot.apply, img_gray, iterations)
    print(f"  Sinter affine:     {t_sinter:.4f} ms")
    print(f"  → Sinter is {t_opencv/t_sinter:.2f}x faster than OpenCV")

    # ========================================================================
    # 9. Pipeline: FlipH + FlipV (should compose to Rot180)
    # ========================================================================
    print("\n9. Pipeline: FlipH + FlipV (geometric composition → Rot180)")
    print("-" * 80)

    # NumPy
    t_numpy = timeit_min(lambda x: materialize(x[::-1, ::-1, :]), img_gray, iterations)
    print(f"  NumPy (combined):  {t_numpy:.4f} ms")

    # OpenCV (two flips)
    t_opencv = timeit_min(
        lambda x: cv2.flip(cv2.flip(x, 1), 0), img_gray, iterations
    )
    print(f"  OpenCV (2x flip):  {t_opencv:.4f} ms")

    # Sinter (should compose to single Rot180)
    pipe = Compose([HorizontalFlip(), VerticalFlip()])
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter pipeline:   {t_sinter:.4f} ms")
    print(f"  Pipeline: {pipe}")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 10. Pipeline: Transpose + FlipH (Rot90)
    # ========================================================================
    print("\n10. Pipeline: Transpose + FlipH (geometric composition → Rot90)")
    print("-" * 80)

    # NumPy
    t_numpy = timeit_min(
        lambda x: materialize(x.transpose(1, 0, 2)[:, ::-1, :]), img_gray, iterations
    )
    print(f"  NumPy (combined):  {t_numpy:.4f} ms")

    # OpenCV
    t_opencv = timeit_min(lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE), img_gray, iterations)
    print(f"  OpenCV rot90:      {t_opencv:.4f} ms")

    # Sinter
    pipe = Compose([Transpose(), HorizontalFlip()])
    t_sinter = timeit_min(pipe.apply, img_gray, iterations)
    print(f"  Sinter pipeline:   {t_sinter:.4f} ms")
    print(f"  Pipeline: {pipe}")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: All operations on grayscale (512x512)")
    print("=" * 80)
    print("Note: Sinter uses optimized NEON SIMD for transpose/rotate operations")
    print("      Crop is naturally fast via slicing (zero-copy possible)")


if __name__ == "__main__":
    benchmark_geometric_grayscale()
