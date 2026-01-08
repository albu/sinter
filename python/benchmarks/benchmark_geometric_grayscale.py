"""
Benchmark geometric operations on grayscale images vs numpy/opencv alternatives.
Ensures views are materialized (no cheating with zero-copy views).
"""
import time
import numpy as np
import cv2

from sinter import (
    HorizontalFlip, VerticalFlip, Transpose, Rotate, RotateAngle,
    Resize, Crop, Pad, PadMode, Affine, Compose, Interpolation,
)

def materialize(arr):
    """Force materialization of views/copies to ensure fair comparison."""
    return np.ascontiguousarray(arr)

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
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray[:, ::-1, :])
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy flip:        {t_numpy:.4f} ms")

    # OpenCV flip
    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.flip(img_gray, 1)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV flip:       {t_opencv:.4f} ms")

    # Sinter flip
    pipe = Compose([HorizontalFlip()])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter flip:       {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 2. Vertical Flip
    # ========================================================================
    print("\n2. Vertical Flip")
    print("-" * 80)

    # NumPy flip
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray[::-1, :, :])
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy flip:        {t_numpy:.4f} ms")

    # OpenCV flip
    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.flip(img_gray, 0)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV flip:       {t_opencv:.4f} ms")

    # Sinter flip
    pipe = Compose([VerticalFlip()])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter flip:       {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 3. Transpose
    # ========================================================================
    print("\n3. Transpose (swap axes)")
    print("-" * 80)

    # NumPy transpose
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray.transpose(1, 0, 2))
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy transpose:   {t_numpy:.4f} ms")

    # OpenCV transpose
    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.transpose(img_gray)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV transpose:  {t_opencv:.4f} ms")

    # Sinter transpose
    pipe = Compose([Transpose()])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter transpose:  {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 4. Rotate 90°
    # ========================================================================
    print("\n4. Rotate 90° (clockwise)")
    print("-" * 80)

    # NumPy rotate (transpose + flip)
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray.transpose(1, 0, 2)[:, ::-1, :])
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy rot90:       {t_numpy:.4f} ms")

    # OpenCV rotate
    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV rot90:      {t_opencv:.4f} ms")

    # Sinter rotate
    pipe = Compose([Rotate(angle=RotateAngle.ROTATE_90)])  # 90°
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter rot90:      {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 5. Resize (downsample to 256x256)
    # ========================================================================
    print("\n5. Resize (512→256, nearest neighbor)")
    print("-" * 80)

    # NumPy resize (using slicing for integer downsample)
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray[::2, ::2, :])
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy slice:       {t_numpy:.4f} ms (only for integer downsampling)")

    # OpenCV resize
    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV resize:     {t_opencv:.4f} ms")

    # Sinter resize
    pipe = Compose([Resize(width=256, height=256, interpolation=Interpolation.NEAREST)])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter resize:     {t_sinter:.4f} ms")
    print(f"  → Sinter is {t_opencv/t_sinter:.2f}x faster than OpenCV")

    # ========================================================================
    # 6. Crop (center crop to 256x256)
    # ========================================================================
    print("\n6. Crop (center crop to 256x256)")
    print("-" * 80)

    start_y, start_x = 128, 128

    # NumPy crop
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray[start_y:start_y+256, start_x:start_x+256, :])
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy crop:        {t_numpy:.4f} ms")

    # OpenCV crop (same as numpy, just slicing)
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray[start_y:start_y+256, start_x:start_x+256, :])
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV crop:       {t_opencv:.4f} ms")

    # Sinter crop
    pipe = Compose([Crop(x=128, y=128, width=256, height=256)])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter crop:       {t_sinter:.4f} ms")
    print(f"  → NumPy/OpenCV slicing is fastest (zero-copy view)")

    # ========================================================================
    # 7. Pad (reflect padding, 10px on all sides)
    # ========================================================================
    print("\n7. Pad (reflect mode, 10px all sides → 532x532)")
    print("-" * 80)

    # NumPy pad
    start = time.perf_counter()
    for _ in range(iterations):
        result = np.pad(img_gray, ((10, 10), (10, 10), (0, 0)), mode='reflect')
        result = materialize(result)
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy pad:         {t_numpy:.4f} ms")

    # OpenCV copyMakeBorder
    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.copyMakeBorder(img_gray, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV pad:        {t_opencv:.4f} ms")

    # Sinter pad
    pipe = Compose([Pad(top=10, bottom=10, left=10, right=10, mode=PadMode.REFLECT)])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter pad:        {t_sinter:.4f} ms")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 8. Affine (simple scale 1.5x, center)
    # ========================================================================
    print("\n8. Affine (scale 1.5x, bilinear → 768x768)")
    print("-" * 80)

    # OpenCV affine (warp)
    M = np.array([[1.5, 0, 128], [0, 1.5, 128]], dtype=np.float32)
    dsize = (768, 768)

    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.warpAffine(img_gray, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV warpAffine: {t_opencv:.4f} ms")

    # Sinter affine
    pipe = Compose([Affine(scale=(1.5, 1.5), interpolation=Interpolation.BILINEAR)])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter affine:     {t_sinter:.4f} ms")
    print(f"  → Sinter is {t_opencv/t_sinter:.2f}x faster than OpenCV")

    # ========================================================================
    # 9. Pipeline: FlipH + FlipV (should compose to Rot180)
    # ========================================================================
    print("\n9. Pipeline: FlipH + FlipV (geometric composition → Rot180)")
    print("-" * 80)

    # NumPy
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray[::-1, ::-1, :])
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy (combined):  {t_numpy:.4f} ms")

    # OpenCV (two flips)
    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.flip(cv2.flip(img_gray, 1), 0)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV (2x flip):  {t_opencv:.4f} ms")

    # Sinter (should compose to single Rot180)
    pipe = Compose([HorizontalFlip(), VerticalFlip()])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
    print(f"  Sinter pipeline:   {t_sinter:.4f} ms")
    print(f"  Pipeline: {pipe}")
    print(f"  → Sinter is {min(t_numpy, t_opencv)/t_sinter:.2f}x faster than best alternative")

    # ========================================================================
    # 10. Pipeline: Transpose + FlipH (Rot90)
    # ========================================================================
    print("\n10. Pipeline: Transpose + FlipH (geometric composition → Rot90)")
    print("-" * 80)

    # NumPy
    start = time.perf_counter()
    for _ in range(iterations):
        result = materialize(img_gray.transpose(1, 0, 2)[:, ::-1, :])
    t_numpy = (time.perf_counter() - start) / iterations * 1000
    print(f"  NumPy (combined):  {t_numpy:.4f} ms")

    # OpenCV
    start = time.perf_counter()
    for _ in range(iterations):
        result = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
    t_opencv = (time.perf_counter() - start) / iterations * 1000
    print(f"  OpenCV rot90:      {t_opencv:.4f} ms")

    # Sinter
    pipe = Compose([Transpose(), HorizontalFlip()])
    for _ in range(5):
        _ = pipe.apply(img_gray.copy())
    start = time.perf_counter()
    for _ in range(iterations):
        result = pipe.apply(img_gray.copy())
    t_sinter = (time.perf_counter() - start) / iterations * 1000
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
