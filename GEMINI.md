# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Documentation

This project uses a **minimal documentation system** with just 3 files:

| File | Purpose | When to Load |
|------|---------|--------------|
| **GEMINI.md** (this file) | Commands + quick reference | Always - quick reference |
| **ARCHITECTURE.md** | System architecture + fusion | Understanding how it works |
| **DEVELOPMENT.md** | Adding transforms + extending | Making code changes |

**Usage**:
- Start here (GEMINI.md) for commands and quick reference
- Read `ARCHITECTURE.md` to understand the architecture, fusion, and optimization
- Read `DEVELOPMENT.md` when adding transforms or extending the system

## Working Efficiently

**Prompt patterns for minimal token usage**:

```bash
# Quick task (only GEMINI.md loads)
"Add brightness transform"

# Architecture task
"Read ARCHITECTURE.md, then debug the fusion logic"

# Development task
"Read DEVELOPMENT.md, then add a new transform"
```

**Key principles**:
- Load docs **on demand** based on your task
- Don't ask to "read all docs" - waste of tokens
- Clear context and reload only what you need when switching tasks

## Commands

```bash
# Build (quiet)
cargo build -q 2>&1 | grep -E "(error|warning:.*generated|Finished)"

# Test (quiet - saves tokens!)
cargo test -q 2>&1 | grep -E "(^test |^running |FAILED|passed|failed|error:)"

# Test single test
cargo test test_name -q 2>&1 | grep -E "(^test |FAILED|passed|failed|error:)"

# Test single module
cargo test module_name:: -q 2>&1 | grep -E "(^test |FAILED|passed|failed|error:)"

# Run
cargo run
```

**IMPORTANT**: Always use `-q` flag with `cargo build` and `cargo test` to save tokens. The grep filter shows only errors/warnings and the final result.

## Python Bindings

**CRITICAL**: For Python builds, use `maturin` NOT `cargo build`:
```bash
pip install maturin

# Development build (debug mode - slow!)
maturin develop --features python -q 2>&1 | grep -E "(error|warning|Finished|Compiling)"

# Release build (optimized - FAST!)
maturin develop --features python --release -q 2>&1 | grep -E "(error|warning|Finished|Compiling)"
```

**Do NOT use `cargo build --features python`** - it will fail during linking. Always use `maturin` for Python extension builds.

**CRITICAL**: Always use `--release` flag when running benchmarks! Debug mode is **15-40x slower**.

### Pure Native Architecture (No OpenCV Dependency)

Sinter is **100% pure Rust + SIMD**. All operations (including MedianBlur, GaussianBlur, Sharpen, Emboss, EdgeDetection, HSV, and Affine) are natively implemented and vectorized with zero C++ dependencies.

Run Python tests (quiet):
```bash
pip install numpy pytest
pytest python/tests/ -q 2>&1 | grep -E "(PASSED|FAILED|ERROR|test_|===)"
```

Run Python benchmarks:
```bash
# Rebuild release wheel
maturin develop --release --features "python"

# Run benchmarks
python python/benchmarks/benchmark_individual.py
python python/benchmarks/benchmark_fusion.py
```

Python usage:
```python
from sinter import Compose, Brightness, Contrast, Uniform, Constant, HorizontalFlip, Resize
import numpy as np
import torch

# Create a pipeline with distribution support
pipeline = Compose([
    HorizontalFlip(p=0.5),
    Brightness(delta=Uniform(-30.0, 30.0)),
    Contrast(factor=Constant(1.2)),
    Resize(width=256, height=256),
])

# Multi-target call (supports numpy arrays, PyTorch tensors, Python lists for bboxes/keypoints)
img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
mask = np.zeros((300, 300), dtype=np.uint8)
boxes = [[10, 20, 50, 60, 1]]

res = pipeline(image=img, mask=mask, bboxes=boxes, bbox_format="coco")
out_img = res["image"]
out_mask = res["mask"]
out_boxes = res["bboxes"]

# Multi-core batch processing (Rayon + GIL release)
batch = torch.randint(0, 255, (16, 3, 256, 256), dtype=torch.uint8)
out_batch = pipeline.apply_batch(batch, num_threads=4)

# Sample once for deterministic reuse across multiple frames / cameras
sampled = pipeline.sample_with_seed(42)  # or pipeline.sample()
frame1_res = sampled(image=img1)
frame2_res = sampled(image=img2)

# Direct introspection
print(pipeline.explain())
print(pipeline.to_mermaid())
```

### Key Distributions

- `Constant(v)`: Fixed value
- `Uniform(min, max)`: Random float in range
- `UniformInt(min, max)`: Random integer in range
- `Bernoulli(p)`: Probability of success (returns bool or 0.0/1.0)
- `Normal(mu, sigma)`: Gaussian distribution

## Architecture Overview

Sinter is a **compiled image augmentation engine**. Transforms are compiled into an optimized execution plan before running.

### Two-Phase Execution

```
Planning:   Plan -> Optimizer -> ExecPlan
Execution:  ExecPlan (fused ops + barriers)
```

### Quick Reference

| Concept | Location |
|---------|----------|
| Architecture Overview | `ARCHITECTURE.md` |
| Transform Semantics | `ARCHITECTURE.md` - Core Traits |
| Fusion Rules | `ARCHITECTURE.md` - Fusion & Optimization |
| Adding Transforms | `DEVELOPMENT.md` |

### Module Structure

```
src/
├── core/       # Transform, Executable, FusableImage, BarrierImage
├── ir/         # Transform IR (Plan)
├── sampled_ir/ # Sampled IR (pure enums, serializable)
├── exec_ir/    # Execution IR + Optimizer
└── transforms/ # Individual transforms (photometric, geometric, kernel, lut)
```

### Adding a New Transform

See `DEVELOPMENT.md` for the complete step-by-step guide.

Quick summary:
1. Implement `Transform` trait (declare `access()`, `shape_effect()`, `reorder_rule()`)
2. Implement `Executable` trait
3. For photometric ops: implement `LutOp` to enable fusion
4. Register in `src/exec_ir/execution.rs` (dispatch lists)

## Common Pitfalls

### Memory Semantics: Copy-By-Default, `inplace=True` To Opt Out

`apply(...)` and `__call__(...)` default to `inplace=False`: the input array is **never** modified (the engine copies it defensively when needed, or executes zero-copy when out-of-place buffer allocation already occurs). Pass `inplace=True` for zero-copy in-place execution when the input buffer is disposable.

```python
# Default (safe): returns a new array, img_array is untouched
result = pipeline.apply(img_array)

# Fast path: mutates img_array in place, zero allocations
result = pipeline.apply(img_array, inplace=True)  # img_array is now modified!
```

**Return-value convention**:
- `apply(...)` and a bare `transform(image)` call return the transformed **array**
- Passing label targets (`bboxes=`/`keypoints=`/`masks=`/`mask=`) or calling `Compose(image, ...)` returns a **dict** of targets.
