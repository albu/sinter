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

### Building with OpenCV Feature

**CRITICAL - ALWAYS build with opencv feature!**

NEVER build without `--features opencv`. The OpenCV backend provides:
- **Affine transforms**: 1.5x faster than OpenCV Python (with opencv feature) vs 4x slower (without)
- **GaussianBlur, MedianBlur, Sharpen**: Hand-optimized C++ performance
- **Hue/Saturation/Value**: OpenCV's highly optimized HSV implementation

Without OpenCV, Sinter falls back to Rust implementations that are **significantly slower** for these operations.

**IMPORTANT - OpenCV Threading**: ALL benchmarks set `cv2.setNumThreads(0)` to ensure fair single-threaded comparison. The sinter opencv-rust wrapper also runs single-threaded. Any performance differences are NOT due to threading.

**ALWAYS use this build command** (conda environment):
```bash
export DYLD_LIBRARY_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib:/Users/aleksandrbuslaev/miniconda3/envs/py311/lib:$DYLD_LIBRARY_PATH
maturin develop --release --features "python,opencv"
```

**NOTE**: DYLD_LIBRARY_PATH needs both:
- Xcode toolchain (for libclang during OpenCV crate build)
- Conda env (for runtime OpenCV libraries)

**NOTE**: On macOS, `DYLD_LIBRARY_PATH` must be passed inline (cannot be set persistently due to SIP).

### Static OpenCV Linking

For self-contained binaries with zero runtime dependencies:

```bash
# 1. Build minimal OpenCV (4.12, core+imgproc, ~5-10 min)
./scripts/build_opencv_static.sh

# 2. Build Sinter statically
bash -c 'export DYLD_LIBRARY_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib:$DYLD_LIBRARY_PATH && source scripts/setup_static_opencv.sh && maturin develop --release --features "python,opencv-static"'

# 3. Verify no OpenCV runtime dependencies
otool -L /path/to/sinter.so | grep opencv  # Should return nothing
```

**Trade-offs**: ✅ No runtime deps | ✅ Portable | ❌ 5-10 min build | ❌ ~10MB larger

Run Python tests (quiet):
```bash
pip install numpy pytest
pytest python/tests/ -q 2>&1 | grep -E "(PASSED|FAILED|ERROR|test_|===)"
```

Run Python benchmarks (ALWAYS use --release build WITH opencv):
```bash
# First rebuild with --release AND opencv
maturin develop --release --features "python,opencv"

# Then run benchmarks
python python/tests/benchmark_fair_v1_v2.py
```

**WARNING**: Running benchmarks without the opencv feature will show **misleading results** - Affine will appear 4x slower when it's actually just using the Rust fallback!

Python usage:
```python
from sinter import Compose, Brightness, Contrast, Uniform, Constant
import numpy as np

# Create a pipeline with distribution support
pipeline = Compose([
    Brightness(delta=Uniform(-30.0, 30.0)),
    Contrast(factor=Constant(1.2)),
])

# Apply directly to numpy arrays
img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
result = pipeline.apply(img_array.copy())

# Sample once for deterministic reuse
sampled = pipeline.sample_with_seed(42)
result1 = sampled.apply(img1.copy())
result2 = sampled.apply(img2.copy())
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

### ALWAYS Use `.copy()` When Applying Transforms

**NEVER forget `.copy()` - transforms modify arrays in-place!**

```python
# ❌ WRONG - modifies the original array!
result = pipeline.apply(img_array)
print(img_array.mean())  # This has been modified!

# ✅ CORRECT - preserves the original
result = pipeline.apply(img_array.copy())
print(img_array.mean())  # Original is unchanged
```

**Why this happens**: Most transforms are `InPlace` - they modify the input array directly without allocating a new buffer. This is a performance optimization.

**Symptoms of this bug**:
- Unexpected values in your "original" array
- Multiple transforms affecting each other's inputs
- Data corruption when reusing arrays

**Lesson**: Always use `.copy()` unless you explicitly want to modify the original!
