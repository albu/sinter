# Sinter

<p align="center">
  <img src="assets/logo.jpg" width="600" alt="Sinter Logo">
</p>

<p align="center">
  <i>A compiler-accelerated image augmentation engine in pure Rust + SIMD</i>
</p>

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)

---

## The Core Idea: Compilation, Not Composition

Traditional augmentation libraries compose transforms at runtime:

```python
# Typical library: 4 separate passes through the image
Compose([Brightness(), Contrast(), Gamma(), HorizontalFlip()])
# → pass1 → pass2 → pass3 → pass4
```

Sinter compiles transforms into an optimized execution plan:

```python
# Sinter: analyze pipeline → fuse compatible ops → fewer passes
Compose([Brightness(), Contrast(), Gamma(), HorizontalFlip()])
# → compile → [Fused LUT, HorizontalFlip] → 2 passes

# All photometric? True single pass:
Compose([Brightness(), Contrast(), Gamma()])
# → compile → Fused LUT → 1 pass
```

This is inspired by database query optimizers and JIT compilers: analyze the operations, find optimization opportunities, and execute a fused plan.

---

## Architecture Overview

**Two-phase execution:**

1. **Planning**: Analyze your pipeline, find compatible transforms, fuse them together
2. **Execution**: Run the optimized plan with fewer passes through the image

**Example:**
```python
Compose([Brightness(), Contrast(), Gamma()])
# → Plan: "All 3 are LUT transforms, can fuse"
# → Execute: Single pass with one 256-entry lookup table

Compose([Brightness(), HorizontalFlip(), Contrast()])
# → Plan: "Flip breaks fusion, handle separately"
# → Execute: [Fused LUT, HorizontalFlip] → 2 passes
```

**What gets fused:**

| Transform Type | Fuses With |
|----------------|------------|
| Photometric (LUT) | Other photometric LUT transforms → Single lookup table |
| Photometric (Matrix) | Other matrix transforms → Single 3×3 matrix multiply |
| Geometric (flips/rotates) | Other geometric transforms → Composed transform |
| Mixed | Geometric transforms break photometric fusion (barriers) |

---

---

## Modern Ergonomics & API Highlights

Sinter is designed for developer delight, eliminating the ceremony and ambiguity common in computer vision augmentation pipelines:

### 1. Direct Return on Single Images (Zero Dictionary Tax)
When augmenting a single image, you get the transformed image or tensor directly. Target dictionaries are only returned when you actually pass multimodal targets:

```python
# Returns the transformed ndarray or PyTorch tensor directly
out = transform(img)
out = pipeline.apply(img)

# Multimodal calls return a structured dictionary
res = pipeline(image=img, mask=mask, bboxes=boxes)
```

### 2. Explicit Distributions Over Arbitrary Limits
Every transform parameter accepts explicit distributions, clean tuples, or scalars. No guessing what a parameter named "limit" represents or how it scales:

```python
from sinter import Brightness, Contrast, HueSaturationValue, Normal, UniformInt

pipeline = Compose([
    Brightness(delta=Normal(0.0, 15.0)),               # Gaussian distribution centered at 0
    Contrast(factor=(0.8, 1.2)),                        # Continuous uniform range [0.8, 1.2]
    HueSaturationValue(hue_shift=UniformInt(-15, 15)),  # Discrete integer uniform range
])
```

### 3. Clean Stochastic Branching with `Choice`
Select exactly one candidate transform using explicit candidate weights and an overall activation probability—with zero nested probability ambiguity:

```python
from sinter import Choice, GaussianBlur, MedianBlur, Identity

# 70% Gaussian, 30% Median, triggered with 90% probability
aug = Choice([GaussianBlur(3), MedianBlur(3)], weights=[0.7, 0.3], p=0.9)

# Identity is a first-class no-op transform
aug = Choice([GaussianBlur(3), Identity()], weights=[0.8, 0.2])
```

### 4. Configure Target Formats Once
Specify bounding box and keypoint conventions once on the pipeline instead of repeating them on every single call:

```python
pipeline = Compose(
    [HorizontalFlip(p=0.5), Resize(256, 256)],
    bbox_format="pascal_voc",
    keypoint_format="xy",
)

# Format is automatically inherited across all dataset calls
res = pipeline(image=img, bboxes=boxes, keypoints=kpts)
```

### 5. Safe Bounding Box Labels (Zero Desynchronization)
Category labels ride directly in column 5+ of the bounding box array (`[x, y, w, h, class_id]`). When boxes are cropped or filtered out, labels stay bound to their coordinates with zero risk of parallel list desynchronization.

### 6. Seamless Metadata Pass-Through
Dataset metadata, file paths, sample IDs, and image-level classification labels pass straight through untouched:

```python
res = pipeline(
    image=img,
    mask=mask,
    sample_id=42,
    filepath="train/001.jpg",
    labels=[1, 0, 0],  # Image-level labels pass through cleanly
)
assert res["sample_id"] == 42
```

### 7. Zero-Friction 4D Batch Execution
Pass a 4D NumPy array `(B, H, W, C)` or PyTorch tensor `(B, C, H, W)` directly to execute across CPU cores via Rayon multi-threading with Python GIL release:

```python
# Automatically runs parallel across CPU cores with independent per-image seeds
batch_out = pipeline(images_4d)["image"]
batch_out = pipeline.apply(images_4d)
```

---

## Benchmarks

Benchmark results on **Apple M4** (ARM64), single-threaded, generated with the benchmark suite in `python/benchmarks/` (comparing against Albumentations 2.0.8 on identical inputs with zero harness asymmetry).

### Pipeline Speedups (RGB)

| Pipeline | 512×512 Image | 1024×1024 Image | Fusion Strategy |
|----------|---------------|-----------------|-----------------|
| **4 LUT transforms** | **1.9x faster** (0.09 ms vs 0.17 ms) | **1.7x faster** (0.37 ms vs 0.62 ms) | 4 ops → 1 lookup table pass |
| **8 LUT transforms** | **4.9x faster** (0.08 ms vs 0.39 ms) | **4.3x faster** (0.34 ms vs 1.43 ms) | 8 ops → 1 lookup table pass |
| **Heavy pipeline** (16 sinter / 14 alb transforms) | **4.7x faster** (1.46 ms vs 6.90 ms) | **4.7x faster** (5.96 ms vs 28.24 ms) | Multi-phase fusion + SIMD |

### Individual Transform Speedups (ARM64 NEON vs Albumentations)

Sinter includes hand-written NEON intrinsics for ARM64 that provide significant speedups even for individual transforms:

| Transform | 512×512 Speedup | 1024×1024 Speedup | SIMD Technique |
|-----------|-----------------|-------------------|----------------|
| **Transpose** | **12.2x** | **7.0x** | 8×8 block tiling with `vtrn1/vtrn2` |
| **AutoContrast** | **5.9x** | **5.8x** | Vectorized histogram + LUT executor |
| **Equalize** | **2.9x** | **2.9x** | Cumulative histogram + NEON LUT |
| **GaussianBlur (3×3)** | **4.2x** | **4.2x** | Fused rolling separable `[1,2,1]` |
| **GaussianBlur (5×5)** | **3.5x** | **3.8x** | Fused rolling separable `[1,4,6,4,1]` |
| **GaussianBlur (7×7)** | **2.7x** | **3.1x** | Fused rolling separable `[1,6,15,20,15,6,1]` |
| **Sharpen** | **3.3x** | **3.2x** | 3×3 convolution kernel |
| **ToGray** | **1.4x** | **1.4x** | SIMD RGB luminance weighting |

**Notes**:
- Primary target is ARM64 (Apple Silicon, AWS Graviton) with native NEON intrinsics alongside portable scalar fallbacks.
- Speedup scales with transform count: the more operations in the pipeline, the more passes Sinter fuses away.
- Run the benchmark suites yourself via `python python/benchmarks/benchmark_fusion.py` and `python python/benchmarks/benchmark_individual.py`.

---

## Sampling & Distributions

**Transforms can use probability distributions.**

Most transforms accept distributions for their parameters:

```python
from sinter import Compose, Brightness, Uniform, Constant, Bernoulli
import numpy as np

# Define a pipeline with distributions
pipeline = Compose([
    Brightness(delta=Uniform(-30.0, 30.0)),  # Random brightness in range
    Contrast(factor=Constant(1.2)),            # Fixed contrast
])
```

**Transforms with probability parameter:**

Some transforms have a `p` parameter that accepts a distribution (default: always applied):

```python
from sinter import CoarseDropout, GaussianBlur

CoarseDropout(holes=8, hole_size=[0.08, 0.08], p=0.7)  # 70% chance
GaussianBlur(kernel_size=5, p=0.5)  # 50% chance
```

**Sample once, reuse many times:**

```python
# Sample the pipeline to get deterministic transforms
sampled = pipeline.sample_with_seed(42)

# Apply to multiple images with the SAME transforms
img1_result = sampled.apply(img1.copy())
img2_result = sampled.apply(img2.copy())
img3_result = sampled.apply(img3.copy())
# All images get the same brightness values

# Or sample fresh for each image
for img in batch:
    result = pipeline.apply(img.copy())  # Different random values each time
```

**Why this matters:**
- **Reproducibility**: Same seed → same transforms across all images
- **Distributed training**: Sample once, send to workers
- **Efficiency**: Avoid re-sampling for every image

**Available distributions:**
- `Uniform(a, b)` - Random value in range [a, b]
- `UniformInt(a, b)` - Random integer in range [a, b]
- `Constant(value)` - Fixed value (deterministic)
- `Bernoulli(p)` - Binary choice with probability p
- `Normal(mean, std)` - Gaussian distribution

---

## Why Pure Rust + SIMD?
 
Sinter is built in 100% pure native Rust to deliver:
 
1. **Zero-cost abstractions**: Compile transform pipelines into single-pass execution plans without runtime overhead.
2. **Compiler-proven optimization**: Provable operator fusion for photometric, geometric, and matrix pipelines.
3. **Native SIMD acceleration**: Hand-optimized NEON (ARM64) and SIMD intrinsics with zero C++ or OpenCV dependencies.
 
Performance matters: the ~1.6–17× speedup vs traditional libraries comes from both compiler fusion optimizations and hand-tuned native SIMD kernels.

---

## Quick Experiment

```python
import numpy as np
import torch
from sinter import (
    Compose, HorizontalFlip, Affine, Resize,
    Brightness, Contrast, HueSaturationValue, RGBShift, GaussNoise,
    GaussianBlur, MedianBlur, Choice, Uniform
)

# 1. Pipeline creation with distributions and configured target format
geom = Compose([
    HorizontalFlip(p=0.5),
    Affine(scale=(0.9, 1.1), rotate=Uniform(-10, 10), border_mode="reflect", p=0.8),
    Resize(width=256, height=256),
], bbox_format="coco")

photo = Compose([
    Brightness(delta=(-20, 20)),
    Contrast(factor=(0.8, 1.2)),
    HueSaturationValue(hue_shift=(-15, 15), sat_shift=(-20, 20)),
    RGBShift(r_shift=(-15, 15), g_shift=(-15, 15), b_shift=(-15, 15)),
    GaussNoise(std=(10, 30)),
])

# 2. Composition & Slicing
pipeline = geom + photo + [Choice([GaussianBlur(5), MedianBlur(3)], weights=[0.7, 0.3], p=0.4)]
sub_pipeline = pipeline[1:4]  # Slicing returns a sub-Compose

# 3. Direct Introspection
print(pipeline.explain())     # Shows fused execution nodes
print(pipeline.to_mermaid())  # Renders Mermaid diagram

# 4. Multi-Target Call (NumPy arrays, PyTorch CHW tensors, Python lists)
img_tensor = torch.randint(0, 255, (3, 300, 300), dtype=torch.uint8)
seg_mask = np.zeros((300, 300), dtype=np.uint8)
coco_boxes = [[20, 30, 100, 120, 1]]

res = pipeline(image=img_tensor, mask=seg_mask, bboxes=coco_boxes)
out_img = res["image"]    # torch.Tensor (CHW preserved)
out_mask = res["mask"]    # np.ndarray
out_boxes = res["bboxes"] # Python list

# 5. Multi-Core Rayon Batching (Releases GIL)
batch = torch.randint(0, 255, (16, 3, 256, 256), dtype=torch.uint8)
out_batch = pipeline.apply_batch(batch, num_threads=4)

# 6. Sampled Program for Deterministic Multi-Frame Reuse
sampled = pipeline.sample(seed=42)
frame1 = sampled(image=img_tensor)
frame2 = sampled(image=img_tensor)
```

**Memory Semantics**: By default, Sinter uses safe memory semantics (`inplace=False`), so your original image arrays are never modified. Out-of-place pipelines (Resize, Crop, Pad, Affine) execute with zero-copy overhead. For maximum in-place performance on disposable buffers, pass `inplace=True`.

---

## Installation

```bash
# Build and install from source
git clone https://github.com/albu/sinter.git
cd sinter
pip install maturin
maturin develop --release
```

Sinter is **100% pure Rust + SIMD** with zero OpenCV or C++ dependencies. The only Python dependency is `numpy>=1.20` (and optionally `torch` for tensor workflows).

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Deep dive into IR design, fusion rules, and optimization
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - How transforms are implemented and extended
- **[OPERATORS.md](OPERATORS.md)** - Complete reference of all supported transforms and fusion rules

---

## Project Status

This is an ongoing research project. Key milestones completed:

- [x] Compilation & JIT-style operator fusion (LUT, matrix, geometric D4)
- [x] Pure native SIMD architecture (zero C++ dependencies)
- [x] Zero-copy PyTorch tensor & multi-target transformation
- [x] High-throughput parallel batch execution (Rayon + GIL release)
- [x] Visualization of compiled plans (`explain`, `to_mermaid`, `visualize`)
- [x] Broad photometric, geometric, kernel, and dropout transform coverage
- [x] Clean ergonomics: `Choice`, distribution engine, pass-through metadata, format inheritance

---

## Background

From a co-creator of [Albumentations](https://github.com/albumentations-team/albumentations), Sinter is a next-generation exploration into compiler-accelerated computer vision pipelines, rethinking image augmentation from the ground up through IR compilation, operator fusion, and zero-copy native SIMD execution.

---

## License

Dual-licensed under [MIT](LICENSE) or [Apache-2.0](LICENSE-APACHE), at your option.
