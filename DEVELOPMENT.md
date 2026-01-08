# Development Guide

This guide covers adding new transforms and extending Sinter.

## Adding a New Transform

### Overview

Adding a transform requires changes in several places:

```
src/transforms/
├── photometric/          # Transform implementation
├── mod.rs                # Re-export
src/python/transforms/
├── photometric.rs        # Python wrapper
├── mod.rs                # Python re-export
├── compose.rs            # Add to Compose
src/python/
├── mod.rs                # Module exports
src/exec/exec_ir/
└── execution.rs          # Dispatch list (2 locations!)
```

---

## Step-by-Step Example: Equalize

We'll walk through adding the **Equalize** transform (histogram equalization).

### Step 1: Implement the Rust Transform

Create `src/transforms/photometric/histogram.rs`:

```rust
use crate::core::{AccessPattern, Executable, FusableImage, BarrierImage, ShapeEffect, Transform};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equalize;

impl Equalize {
    pub fn new() -> Self {
        Self
    }

    fn build_lut(&self, image: &FusableImage) -> [u8; 256] {
        // Build histogram
        let mut histogram = [0u32; 256];
        let total_pixels = (image.width * image.height * image.channels) as f32;

        for &pixel in image.data.iter() {
            histogram[pixel as usize] += 1;
        }

        // Build equalization LUT using CDF
        let mut cdf = 0u32;
        let mut lut = [0u8; 256];

        for i in 0..256 {
            cdf += histogram[i];
            lut[i] = ((cdf as f32 / total_pixels) * 255.0).clamp(0.0, 255.0) as u8;
        }

        lut
    }
}

impl Transform for Equalize {
    fn access(&self) -> AccessPattern {
        AccessPattern::InPlace
    }

    fn shape_effect(&self) -> ShapeEffect {
        ShapeEffect::Preserve
    }

    fn reorder_rule(&self) -> ReorderRule {
        ReorderRule::Barrier  // Depends on image content
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_executable(&self) -> Option<&dyn Executable> {
        Some(self)
    }
}

impl Executable for Equalize {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage> {
        let lut = self.build_lut(image);

        for pixel in image.data.iter_mut() {
            *pixel = lut[*pixel as usize];
        }

        None  // In-place, no new buffer
    }
}
```

**Key traits:**
- `Transform`: Declares access pattern, shape effect, and reordering behavior
- `Executable`: Performs the actual transformation

**ReorderRule selection:**
- `CommutesWithGeometry`: Per-pixel ops (Brightness, Contrast, Gamma)
- `Geometry`: Coordinate transforms (Flip, Rotate)
- `Barrier`: Content-dependent or shape-changing (default)

---

## Copy vs Clone: Choosing the Right Pattern

When implementing a transform, you need to choose between `Copy` and `Clone` traits. This choice affects whether the transform can cache its LUT (lookup table).

### Stateless Transforms (Use `Copy`)

For simple transforms without internal state:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brightness {
    pub delta: f32,
    // No LUT caching - Copy trait prevents interior mutability
}
```

**When to use Copy:**
- Transform has only simple fields (numbers, simple enums)
- LUT is inexpensive to rebuild
- Transform is used many times with same parameters (fusion caches at planning time)

**Examples:** `Brightness`, `Invert`, geometric transforms (`Flip`, `Rotate`)

### Stateful Transforms with LUT Caching (Use `Clone`)

For transforms that benefit from LUT caching:

```rust
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub struct Contrast {
    pub factor: f32,
    /// Cached LUT - built once on first access
    lut: OnceLock<[u8; 256]>,
}

impl LutOp for Contrast {
    fn build_lut(&self) -> [u8; 256] {
        // Compute LUT
        let mut lut = [0u8; 256];
        // ... lut computation ...
        lut
    }

    fn get_lut(&self) -> [u8; 256] {
        *self.lut.get_or_init(|| self.build_lut())
    }
}
```

**When to use Clone + OnceLock:**
- LUT computation is moderately expensive
- Transform may be executed multiple times without fusion
- You want to cache the LUT across executions

**Examples:** `Contrast`, `Gamma`, `Solarize`, `Posterize`, `Normalize`

### Why Not Both?

`Copy` trait requires that all fields implement `Copy`. `OnceLock` does not implement `Copy` because it manages interior mutability (the cached value).

### Performance Considerations

- **With Copy + fusion**: LUT built once during planning
- **With Clone + OnceLock**: LUT built once on first use, cached thereafter
- **Without caching**: LUT rebuilt on every execution

For single-transform use, `Clone + OnceLock` may be faster. For fused pipelines, `Copy` is sufficient since the optimizer builds the fused LUT once during planning.

### Step 2: Export from Rust Modules

Edit `src/transforms/photometric/mod.rs`:

```rust
mod histogram;  // Add this

pub use histogram::{Equalize, AutoContrast};  // Add this
```

Edit `src/transforms/mod.rs`:

```rust
pub use photometric::{..., Equalize, AutoContrast};  // Add to existing list
```

### Step 3: Create Python Wrapper

Edit `src/python/transforms/photometric.rs`:

First, add to imports:
```rust
use crate::transforms::{..., Equalize, AutoContrast};
```

Then add the wrapper:
```rust
#[cfg(feature = "python")]
impl PyTransformExtract for PyEqualize {
    fn as_transform(&self) -> Box<dyn Transform> {
        Box::new(self.inner)
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Equalize")]
pub struct PyEqualize {
    inner: Equalize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEqualize {
    #[new]
    fn new() -> Self {
        Self { inner: Equalize::new() }
    }

    fn __call__(&self, image: &PyImage) -> PyResult<PyImage> {
        let mut barrier = image.inner.clone();
        let mut fusable = barrier.as_fusable();
        if let Some(new_barrier) = self.inner.execute(&mut fusable) {
            Ok(PyImage { inner: new_barrier })
        } else {
            Ok(PyImage { inner: barrier })
        }
    }

    fn __repr__(&self) -> String {
        "Equalize()".to_string()
    }
}
```

**For transforms with parameters:**
```rust
#[new]
#[pyo3(signature = (cutoff=0.0))]  // Default parameter
fn new(cutoff: f32) -> PyResult<Self> {
    if !(0.0..=0.5).contains(&cutoff) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("cutoff must be in [0.0, 0.5], got {}", cutoff)
        ));
    }
    Ok(Self { inner: AutoContrast::new(cutoff) })
}
```

### Step 4: Export Python Bindings

Edit `src/python/transforms/mod.rs`:
```rust
pub use photometric::{..., PyEqualize, PyAutoContrast};  // Add to list
```

Edit `src/python/mod.rs`:
```rust
use transforms::{..., PyEqualize, PyAutoContrast, ...};  // Add to imports

// In pymodule function:
m.add_class::<PyEqualize>()?;
m.add_class::<PyAutoContrast>()?;
```

### Step 5: Add to Compose

Edit `src/python/transforms/compose.rs`:

Add to imports:
```rust
use super::{
    PyTransformExtract,
    ..., PyEqualize, PyAutoContrast,  // Add here
    ...
};
```

Add to extract chain in `new()` method:
```rust
} else if let Ok(equalize) = item.extract::<PyRef<PyEqualize>>(py) {
    equalize.as_transform()
} else if let Ok(autocontrast) = item.extract::<PyRef<PyAutoContrast>>(py) {
    autocontrast.as_transform()
} else {
    return Err(...);
};
```

### Step 6: Add to Execution Dispatch

**CRITICAL**: Your transform must be added to **TWO locations** in `src/exec/exec_ir/execution.rs`:

1. **Imports** (top of file):
```rust
use crate::transforms::{
    ..., Equalize, AutoContrast,  // Add here
    ...
};
```

2. **First dispatch list** (search for similar transforms in the Fused branch):
```rust
try_execute_transform!(transform, image, matched, result, {
    // ... existing transforms ...
    Equalize,       // Add
    AutoContrast,   // Add
});
```

3. **Second dispatch list** (search for similar transforms in the Barrier branch):
```rust
try_execute_transform!(transform, image, matched, result, {
    // ... existing transforms ...
    Equalize,       // Add
    AutoContrast,   // Add
});
```

**Note**: The dispatch uses the `try_execute_transform!` macro.

### Step 7: Build and Test

```bash
maturin develop --release --features "python,opencv"

# Test in Python
python -c "
from sinter import Equalize, Compose
import numpy as np

arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Individual
result = Equalize()(arr)

# In Compose
pipe = Compose([Equalize()])
result = pipe.apply(arr)

print('Success!')
"
```

---

## Transform Types

### LUT Transforms (Fusable)

Transforms that apply the same operation to each pixel independently:

**Examples:** Brightness, Contrast, Solarize, Posterize, Invert, Gamma

```rust
impl LutOp for MyTransform {
    fn build_lut(&self, channels: u8) -> [u8; 256] {
        // Pre-compute output for each input value [0, 255]
        let mut lut = [0u8; 256];
        for i in 0..256 {
            lut[i] = /* compute output for input i */;
        }
        lut
    }
}
```

These automatically fuse with other LUT transforms!

### Matrix Transforms (Fusable)

Transforms that apply 3x3 RGB matrix multiplication:

**Examples:** ToSepia, Saturation, ColorTemperature, ChannelMix

```rust
impl MatrixOp for MyTransform {
    fn matrix(&self) -> [[f32; 3]; 3] {
        // Return 3x3 transformation matrix
        [
            [r_r, r_g, r_b],
            [g_r, g_g, g_b],
            [b_r, b_g, b_b],
        ]
    }
}
```

These compose by matrix multiplication for single-pass execution.

### Histogram-Dependent Transforms

Transforms that need to analyze the image first:

**Examples:** Equalize, AutoContrast

These cannot be pre-fused because the LUT depends on image content, but they still work in Compose as individual passes.

### Geometric Transforms

Transforms that change pixel positions:

**Examples:** HorizontalFlip, VerticalFlip, Rotate, Transpose, Resize, Crop, Pad

- **InPlace geometric** (Flip, Rotate90/180/270, Transpose): Can fuse via D4 group composition
- **OutOfPlace geometric** (Resize, Crop, Pad): Return `Some(BarrierImage)` from `execute()`

### Noise Transforms

Transforms that generate random noise:

**Examples:** GaussNoise, SaltAndPepper, MultiplicativeNoise

These do not fuse because they are not pure functions (random).

---

## Common Patterns

### Parameter Validation

```rust
pub fn new(delta: f32) -> Self {
    assert!(
        (-255.0..=255.0).contains(&delta),
        "delta must be in [-255, 255], got {}",
        delta
    );
    Self { delta }
}
```

Or in Python wrapper for user-friendly errors:
```rust
#[new]
fn new(delta: f32) -> PyResult<Self> {
    if !(-255.0..=255.0).contains(&delta) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("delta must be in [-255, 255], got {}", delta)
        ));
    }
    Ok(Self { inner: Brightness::new(delta) })
}
```

### Copy vs Clone

For transforms with no data (unit struct):
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MyTransform;
```

For transforms with data:
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct MyTransform {
    pub value: f32,
}
```

Use `Copy` only for types that are trivially copyable (no heap data).

---

## Debugging Tips

### "Transform not valid" Error

Means the transform is missing from:
- `src/python/transforms/compose.rs` extract chain, OR
- `src/exec/exec_ir/execution.rs` import list, OR
- One of the dispatch lists in `execution.rs`

### "Does not implement Executable" Error

Means the transform is missing from:
- Both dispatch lists in `src/exec/exec_ir/execution.rs`

Check ALL dispatch locations (Fused branch and Barrier branch)!

### Import Error

Check all `mod.rs` files in the import chain:
1. `src/transforms/photometric/mod.rs`
2. `src/transforms/mod.rs`
3. `src/python/transforms/mod.rs`
4. `src/python/mod.rs`

---

## Performance Tips

1. **Use LUT fusion** for per-pixel transforms - implement `LutOp` trait
2. **Use Matrix fusion** for RGB color transforms - implement `MatrixOp` trait
3. **Avoid allocations** in `execute()` - return `None` for in-place ops
4. **Pre-compute** what you can in `new()` rather than `execute()`
5. **Profile** with `--release` builds - debug mode is 15-40x slower

```bash
# Always benchmark in release mode
maturin develop --features python --release
python benchmark.py
```

---

## Future Design: Multi-Label Support

This is a proposed design for supporting bounding boxes, keypoints, and masks.

### Python API Design

```python
transform = sinter.Compose([
    sinter.HorizontalFlip(p=0.5),
    sinter.Rotate(angle=90, p=0.5),
    sinter.Brightness(delta=0.2, p=1.0), # Photometric (ignored for bboxes)
], bbox_params=sinter.BboxParams(format="coco"))

result = transform(image=img, bboxes=[[10, 10, 50, 50, 1]]) # x, y, w, h, label
print(result['bboxes'])
```

### Proposed Rust Trait

```rust
pub trait LabelTransform {
    /// Transform a single 2D point (x, y) -> (x', y')
    fn map_point(&self, point: (f32, f32), image_size: (u32, u32)) -> Option<(f32, f32)>;

    /// Transform a bounding box (x, y, w, h) -> (x', y', w', h')
    fn map_bbox(&self, bbox: [f32; 4], image_size: (u32, u32)) -> Option<[f32; 4]>;
}
```

### Implementation Approach

Geometric transforms would implement `LabelTransform` to apply coordinate transformations:
- `HorizontalFlip`: Swap x
- `VerticalFlip`: Swap y
- `Rotate`: Rotate coordinates (0, 90, 180, 270)
- `Resize`: Scale coordinates
- `Crop`: Translate coordinates, filter if outside
- `Pad`: Translate coordinates
- `Affine`: Apply affine matrix multiplication

Photometric transforms would skip label processing.

### Status

This is a **future design**. See `docs/design/label_support.md` for full details.

---

## Checklist

When adding a new transform:

- [ ] Implement `Transform` trait with correct `ReorderRule`
- [ ] Implement `Executable` trait
- [ ] Add unit tests in the transform file
- [ ] Export from `src/transforms/photometric/mod.rs`
- [ ] Export from `src/transforms/mod.rs`
- [ ] Create Python wrapper
- [ ] Export from `src/python/transforms/mod.rs`
- [ ] Add to `src/python/mod.rs` module
- [ ] Add to Compose
- [ ] Add to execution dispatch (BOTH lists!)
- [ ] Build successfully
- [ ] Test individual usage
- [ ] Test in Compose
- [ ] Benchmark against Albumentations (if applicable)
