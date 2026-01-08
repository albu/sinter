# Sinter

<p align="center">
  <img src="assets/logo.jpg" width="600" alt="Sinter Logo">
</p>

<p align="center">
  <i>A research prototype exploring compiler-based optimization for image augmentation</i>
</p>

[![License: CC-BY-NC-SA 4.0](https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)

---

 *"Surprise! You've built a ~44,000 line research project!"*

**Research prototype. No warranty provided. Not intended for production use. API stability not guaranteed.**

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

## Why Rust + NEON?

This prototype is written in Rust to explore:

1. **Zero-cost abstractions**: Can we express transform composition without runtime overhead?
2. **Type-level optimization**: Can the compiler prove fusion safety?
3. **SIMD integration**: Hand-written NEON intrinsics for ARM64 (Apple Silicon, AWS Graviton)

Performance matters: the 2-5x speedup vs traditional libraries comes from both fusion optimizations and hand-tuned NEON code.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Deep dive into IR design, fusion rules, and optimization
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - How transforms are implemented and extended

## Benchmarks

Benchmark results on Apple M1 Pro (ARM64), single-threaded, 512×512 RGB images.

**Run the benchmarks yourself**:
```bash
# First build with --release (critical!)
maturin develop --release --features "python,opencv"
pip install albumentations pytest
python python/benchmarks/benchmark_fusion.py
```

### Fair Comparison (Equivalent Transforms)

**512×512 images**:
| Pipeline | Albumentations | Sinter | Speedup |
|----------|----------------|--------|---------|
| **8 LUT transforms** | 0.53 ms | 0.09 ms | **5.9x** |
| **4 LUT transforms** | 0.22 ms | 0.10 ms | **2.2x** |
| **Full pipeline** (16 transforms) | 12.38 ms | 3.92 ms | **3.2x** |

**1024×1024 images** (speedup scales with image size):
| Pipeline | Albumentations | Sinter | Speedup |
|----------|----------------|--------|---------|
| **8 LUT transforms** | 1.85 ms | 0.34 ms | **5.5x** |
| **4 LUT transforms** | 0.82 ms | 0.37 ms | **2.2x** |
| **Full pipeline** (16 transforms) | 51.88 ms | 15.82 ms | **3.3x** |

### Notable Architectural Wins

These benchmarks demonstrate specific fusion strategies:

| Strategy | Example | Speedup |
|----------|---------|---------|
| **LUT Fusion** | 8 photometric transforms → single lookup table | **5-6x** |
| **Geometric Composition** | FlipH + FlipV → Rot180 via D4 group | **1.2-1.4x** |
| **Heavy Pipeline** | 16 transforms with mixed fusion | **3.2-3.3x** |

**Why the speedup?**

1. **LUT Fusion**: 8 photometric transforms fuse into a single 256-entry lookup table (one pass)
2. **Geometric Composition**: Transform sequences compose via group theory algebra
3. **Matrix Fusion**: Multiple color matrix ops → single 3×3 matrix multiply
4. **Fewer allocations**: Fused ops avoid intermediate buffers between transforms
5. **NEON SIMD**: Hand-optimized ARM64 intrinsics

**Caveats**:
- ARM64 (Apple Silicon, AWS Graviton) is the primary target with hand-tuned NEON code
- Speedup scales with transform count - more transforms = more fusion opportunities
- Single-threaded comparison; both libraries can use threading for batches

See `python/benchmarks/benchmark_fusion.py` for the fusion benchmark suite.

### Individual Transform Speedups (NEON Optimizations)

Sinter includes hand-written NEON intrinsics for ARM64 that provide significant speedups even for single transforms:

| Transform | Speedup (256x256) | Speedup (512x512) | Speedup (1024x1024) | Technique |
|-----------|-------------------|-------------------|---------------------|-----------|
| **Transpose** | **17.4x** | **16.0x** | **8.6x** | 8x8 block tiling with `vtrn1/vtrn2` |
| **AutoContrast** | **7.1x** | **7.1x** | **5.3x** | LUT executor with `vqtbl4q_u8` |
| **Equalize** | **2.9x** | **2.8x** | **2.7x** | LUT executor with `vqtbl4q_u8` |
| **GaussianBlur(3x3)** | **2.5x** | **1.7x** | **1.8x** | Symmetric folding + `vld3q_u8` |
| **GaussianBlur(7x7)** | **2.5x** | **2.3x** | **2.1x** | Separable convolution |
| **ToGray** | **2.2x** | 1.1x | **1.8x** | RGB luminance formula |
| **Sharpen** | **1.7x** | **1.7x** | **1.7x** | 3x3 convolution kernel |
| **Solarize** | **1.8x** | 1.2x | 1.4x | LUT executor |
| **HueSaturationValue** | 1.0x | 1.3x | **1.5x** | SIMD FP HSV conversion |

**Notable NEON implementations**:

- **Transpose** (`src/transforms/geometric/transpose/neon.rs`): Implements true 8x8 block transpose using a 3-stage algorithm (8-bit → 16-bit → 32-bit) with `vtrn1/vtrn2` instructions, plus RGB channel deinterleaving via `vld3_u8`/`vst3_u8`. Matrix transpose is notoriously difficult in SIMD - this solution is 8-17x faster than the scalar approach.

- **LUT Executor** (`src/transforms/runtime/lut/executor/neon.rs`): Uses ARM's `vqtbl4q_u8` instruction to perform **4 parallel 64-byte table lookups in a single instruction**. The 256-byte LUT is split into 4×64-byte chunks with bitwise merging via `vbslq_u8`. A specialized 4-way interleaved version processes 64 pixels per iteration to saturate M1 Pro's 4 NEON execution units.

- **GaussianBlur 3x3** (`src/transforms/kernel/convolve_simd/kernel_3x3.rs`): Applies symmetric folding to reduce `(row[-1]×1 + row[0]×2 + row[1]×1) >> 2` to `(row[-1] + row[1]) + (row[0] << 1) >> 2`, replacing multiplies with adds/shifts. Uses `vld3q_u8`/`vst3q_u8` for efficient RGB channel handling.

- **HSV Conversion** (`src/transforms/photometric/hue_saturation_value/neon.rs`): Performs full RGB ↔ HSV conversion in SIMD floating-point with conditional logic via `vbslq_f32`. Complex type conversion chain: `u8 → u16 → u32 → f32 → processing → f32 → u32 → u16 → u8`. HSV involves complex mathematical operations that are typically not SIMD-friendly.

Run individual benchmarks:
```bash
python python/benchmarks/benchmark_individual.py --lut Brightness AutoContrast
python python/benchmarks/benchmark_individual.py Transpose HorizontalFlip
python python/benchmarks/benchmark_individual.py --kernel GaussianBlur Sharpen
```

## Installation

```bash
# Requires Rust toolchain and maturin
pip install maturin
maturin develop --release --features "python,opencv"
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for build details and OpenCV integration.

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

## Quick Experiment

```python
from sinter import Compose, Brightness, Contrast, Gamma
import numpy as np

# Define a pipeline
pipeline = Compose([
    Brightness(delta=50.0),
    Contrast(factor=0.2),
    Gamma(power=1.5),
])

# Apply to numpy arrays
img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
result = pipeline.apply(img.copy())
```

**Note**: Most transforms mutate in-place for performance (fewer allocations). Copy your input if you need to preserve the original.

## Project Status

This is an ongoing research project. Topics I'm exploring:

- [ ] More photometric transform types
- [ ] Visualization of compiled plans

## Background

I co-created [Albumentations](https://github.com/albumentations-team/albumentations) ~8 years ago. It became successful, which eventually became exhausting. Sinter is my exploration of whether there's a fundamentally different approach to the problem - not as a replacement, but as a way to explore ideas that didn't fit into the Albumentations architecture.

## License

Creative Commons BY-NC-SA 4.0 - See [LICENSE](LICENSE)

**Non-commercial, share-alike**. This is research, not a product.
