# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation

This project uses a **minimal documentation system** with just 3 files:

| File | Purpose | When to Load |
|------|---------|--------------|
| **CLAUDE.md** (this file) | Commands + quick reference | Always - quick reference |
| **ARCHITECTURE.md** | System architecture + fusion | Understanding how it works |
| **DEVELOPMENT.md** | Adding transforms + extending | Making code changes |

**Usage**:
- Start here (CLAUDE.md) for commands and quick reference
- Read `ARCHITECTURE.md` to understand the architecture, fusion, and optimization
- Read `DEVELOPMENT.md` when adding transforms or extending the system

## Working Efficiently with Claude

**Prompt patterns for minimal token usage**:

```bash
# Quick task (only CLAUDE.md loads)
"Add brightness transform"

# Architecture task
"Read ARCHITECTURE.md, then debug the fusion logic"

# Development task
"Read DEVELOPMENT.md, then add a new transform"
```

**Key principles**:
- Let Claude load docs **on demand** based on your task
- Don't ask Claude to "read all docs" - waste of tokens
- Use `/clear` if context feels cluttered, then reload only what you need

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
from sinter import Compose, Brightness, Contrast, Uniform, Constant
import numpy as np

# Create a pipeline with distribution support
pipeline = Compose([
    Brightness(delta=Uniform(-30.0, 30.0)),
    Contrast(factor=Constant(1.2)),
])

# Apply directly to numpy arrays (copy-by-default: img_array is NOT modified)
img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
result = pipeline.apply(img_array)

# Zero-copy opt-in when the input buffer is disposable
fast = pipeline.apply(img_array, inplace=True)

# Sample once for deterministic reuse
sampled = pipeline.sample_with_seed(42)
result1 = sampled.apply(img1)
result2 = sampled.apply(img2)
```

## Architecture Overview

Sinter is a **compiled image augmentation engine**. Transforms are compiled into an optimized execution plan before running.

### Two-Phase Execution

```
Planning:   Plan → Optimizer → ExecPlan
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
├── exec/       # Execution IR + Optimizer
└── transforms/ # Individual transforms (photometric, geometric, kernel, lut)
```

### Adding a New Transform

See `DEVELOPMENT.md` for the complete step-by-step guide.

Quick summary:
1. Implement `Transform` trait (declare `access()`, `shape_effect()`, `reorder_rule()`)
2. Implement `Executable` trait
3. For photometric ops: implement `LutOp` to enable fusion
4. Register in `src/exec/exec_ir/execution.rs` (2 locations in dispatch lists)

## Common Pitfalls

### Memory Semantics: Copy-By-Default, `inplace=True` To Opt Out

`apply(...)` and `__call__(...)` default to `inplace=False`: the input array is **never** modified (the engine copies it first). Pass `inplace=True` for zero-copy in-place execution when the input buffer is disposable.

```python
# Default (safe): returns a new array, img_array is untouched
result = pipeline.apply(img_array)

# Fast path: mutates img_array in place, zero allocations
result = pipeline.apply(img_array, inplace=True)  # img_array is now modified!
```

**Why**: transforms execute `InPlace` on the working buffer; the safe default pays one array copy (~0.06 ms at 1024x1024 RGB), `inplace=True` skips it.

**Return-value convention**:
- `apply(...)` and a bare `transform(image)` call return the transformed **array**
- Passing label targets (`bboxes=`/`keypoints=`/`masks=`) or calling `Compose(image, ...)` returns a **dict** of targets

**Symptoms of ignoring this**: passing `inplace=True` on a buffer you still need, then reading stale/mutated values from the "original".
