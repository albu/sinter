# Sinter Architecture

Compiled image augmentation engine with zero-cost fusion and algebraic optimization.

## Overview

Sinter uses a **two-phase execution model**:

```
Planning:   Sampled IR (Plan) → Optimizer → ExecPlan
Execution:  ExecPlan (fused ops + barriers)
```

**Key design principles:**
- **Zero-cost abstraction**: Optimized compilation at planning time
- **Explicit ownership**: `FusableImage<'a>` (borrowed) vs `BarrierImage` (owned)
- **No RTTI**: Enum dispatch for performance
- **Algebraic optimization**: Geometric hoisting via commutativity

---

## System Overview

### Sampled IR (Type: Core)

**What**: Pure data enums with all randomness resolved. Serializable via serde. Contains the `Plan` type that serves as input to the optimizer.

**Where**: `src/sampled_ir/`

**Key Types**: `SampledImageOp`, `Plan`, `Transform`

**Key Properties**:
- Pure enums (no trait objects)
- Flat structure (no nesting)
- 25+ transform types
- Supports replay via `sample_with_seed()`
- `Plan` - wrapper around `Vec<SampledImageOp>`, input to optimizer
- `bridge` - converts `SampledImageProgram` → `Plan`
- `transform` - `Transform` trait implementation

**Dependencies**: `core` traits

---

### Execution IR (Type: Core)

**What**: Optimized execution plan with fused blocks and barrier nodes.

**Where**: `src/exec_ir/`

**Key Types**: `ExecPlan`, `ExecNode`, `FusedBlock`, `Barrier`

**Dependencies**: Sampled IR, Optimizer

---

### Optimizer (Type: Compiler)

**What**: Converts Sampled IR (Plan) to Execution IR via 4-phase pipeline.

**Where**: `src/exec_ir/optimizer/`

**Pipeline**:
1. Block splitting at barriers
2. Canonicalization (geometric hoisting)
3. Extractive fusion
4. Fast-path kernel selection

---

### Core Traits (Type: Foundation)

**What**: Foundational traits declaring transform semantics for optimizer reasoning.

**Where**: `src/core/traits.rs`, `src/core/mod.rs`

**Key Traits**: `Transform`, `Executable`, `AccessPattern`, `ShapeEffect`, `ReorderRule`

**Details**: See [Core Traits](#core-traits) below.

---

### Image Types (Type: Core)

**What**: Two image types for zero-copy fusion and owned data.

**Where**: `src/core/image.rs`

**Key Types**:
- `FusableImage<'a>`: Borrowed view for zero-copy operations
- `BarrierImage`: Owned data with flexible layout

**Purpose**: Explicit ownership semantics for safe in-place mutation

---

### Python Bindings (Type: Interface)

**What**: PyO3 bindings exposing Rust transforms to Python.

**Where**: `src/python/`

**Structure**:
- `batch/` - Batch transforms (MixUp, CutMix, Mosaic)
- `sampled/` - Sampled IR bindings
- `transforms/` - Individual transform wrappers
- `tensor/` - PyTorch integration

---

### Transforms (Type: Library)

**What**: Individual transform implementations.

**Where**: `src/transforms/`

**Subdirs**:
- `photometric/` - Per-pixel ops (brightness, contrast, etc.)
- `geometric/` - Spatial transforms (flips, resize, crop, pad)
- `kernel/` - Convolution-based ops (blur, sharpen)
- `lut/` - Lookup table based ops
- `matrix/` - 3x3 RGB matrix operations

---

## Core Traits

### Transform Trait

Declares **what** a transform does, not how to execute it:

```rust
pub trait Transform: std::any::Any + Send + Sync {
    fn access(&self) -> AccessPattern;
    fn shape_effect(&self) -> ShapeEffect;
    fn reorder_rule(&self) -> ReorderRule { ReorderRule::Barrier }
    fn as_executable(&self) -> Option<&dyn Executable>;
}
```

### AccessPattern

Memory access pattern:

| Variant | Meaning | Fusion |
|---------|---------|--------|
| `InPlace` | Mutates input buffer | ✅ Fusable |
| `ReadOnly` | Reads input, produces output | ❌ Barrier |
| `OutOfPlace` | Requires new buffer | ❌ Barrier |

### ShapeEffect

Shape impact:

| Variant | Meaning | Fusion |
|----------|---------|--------|
| `Preserve` | H/W/C unchanged | ✅ Fusable |
| `Resize` | Changes H/W | ❌ Barrier |
| `Crop` | Reduces H/W | ❌ Barrier |
| `Pad` | Increases H/W | ❌ Barrier |

### ReorderRule

**KEY INNOVATION**: Declares how transforms can be reordered during canonicalization:

```rust
pub enum ReorderRule {
    CommutesWithGeometry,  // Per-pixel photometric ops
    Geometry,              // Coordinate remapping
    Barrier,               // Cannot reorder (default)
}
```

| Rule | Meaning | Examples |
|------|---------|----------|
| `CommutesWithGeometry` | Per-pixel photometric ops that can be hoisted across geometry | Brightness, Contrast, Gamma, LUT ops, Matrix ops |
| `Geometry` | Coordinate remapping (can compose via D4 group) | HorizontalFlip, VerticalFlip, Rotate |
| `Barrier` | Cannot reorder | Resize, Crop, ToGray, Blur, Noise, Histogram ops |

**The Algebraic Rule**:

Per-pixel photometric transforms commute with geometric coordinate transforms:

```
P(f(x)) = f(P(x))
```

Where:
- `P` = per-pixel photometric operation
- `f` = bijective coordinate remapping

**CRITICAL**: Photometric ops do **NOT** commute with each other:
```
Brightness ∘ Contrast ≠ Contrast ∘ Brightness
```

They only commute with geometry, not with each other.

### Executable Trait

Runtime execution:

```rust
pub trait Executable {
    fn execute(&self, image: &mut FusableImage) -> Option<BarrierImage>;
}
```

**Returns**:
- `None`: Modified in-place (no new buffer)
- `Some(BarrierImage)`: Allocated new buffer (shape change)

---

## Fusion & Optimization

### 4-Phase Pipeline

The optimizer uses a compiler-like pipeline:

```
Phase 1: Block Splitting       → Split at hard barriers
Phase 2: Canonicalization      → Geometric hoisting (P(f(x)) = f(P(x)))
Phase 3: Extractive Fusion     → Fuse contiguous homogeneous groups
Phase 4: Fast-Path Selection   → Optimized kernels for common patterns
```

### Phase 2: Canonicalization (Geometric Hoisting)

Before fusion, photometric transforms are hoisted across geometric transforms:

```text
Input:  [Solarize, Contrast, VerticalFlip, Brightness, Gamma]
Output: [VerticalFlip, Solarize, Contrast, Brightness, Gamma]
```

This enables larger fusion groups by grouping contiguous photometric ops together.

### Phase 3: Extractive Fusion

**Extract what you CAN fuse, leave the rest as individual nodes.**

Algorithm:

```rust
while start < fused.len() {
    // Try geometric group (2+)
    if has_geometric_group(start, 2+) {
        compose_via_D4_group();
        continue;
    }

    // Try LUT group (2+)
    if has_lut_group(start, 2+) {
        compose_into_fused_lut();
        continue;
    }

    // Try Matrix group (2+)
    if has_matrix_group(start, 2+) {
        compose_into_fused_matrix();
        continue;
    }

    // Single transform
    create_individual_node();
}
```

### What Gets Fused

| Group Type | Condition | Result |
|------------|-----------|--------|
| **Geometric** | 2+ contiguous geometric transforms | Composed via D4 group |
| **LUT** | 2+ contiguous LUT transforms | FusedLut (256-entry lookup table) |
| **Matrix** | 2+ contiguous Matrix transforms | FusedMatrix (3x3 matrix mult) |
| **Single** | Anything else | Individual node |

### Examples

#### Example 1: Heterogeneous Block

```rust
// Input: [Solarize, Contrast, VerticalFlip, Brightness, Gamma, Posterize]
//
// Phase 2 (Canonicalization): Geometric hoisting
// → [VerticalFlip, Solarize, Contrast, Brightness, Gamma, Posterize]
//
// Phase 3 (Extractive Fusion):
//   - Geometric: [VerticalFlip] → Individual (only 1)
//   - LUT: [Solarize, Contrast] → FusedLut
//   - LUT: [Brightness] → Individual (Gamma breaks contiguity)
//   - Other: [Gamma] → Individual
//   - LUT: [Posterize] → Individual
//
// Output: 4 nodes (1 fused, 3 individual)
```

#### Example 2: All Geometric

```rust
// Input: [Rotate90, Rotate90, HorizontalFlip]
//
// All geometric → D4 group composition
//
// Output: [Composed(Orientation)]
```

#### Example 3: All LUT

```rust
// Input: [Brightness, Contrast, Invert, Posterize]
//
// All LUT → Composed lookup table
//
// Output: [FusedLut(composed_lut)]
```

#### Example 4: All Matrix

```rust
// Input: [ToSepia, Saturation(0.7)]
//
// All Matrix → Composed matrix
//
// Output: [FusedMatrix(M_sat × M_sepia)]
```

#### Example 5: Barrier Split

```rust
// Input: [Brightness, Resize, Contrast]
//
// Resize is a barrier → split before Phase 2
//
// Output: [Fused([Brightness]), Barrier(Resize), Fused([Contrast])]
```

### Implementation

The fusion module is structured as:

```
src/exec_ir/fusion/
├── mod.rs         # Entry point: `fuse_transform_block()`
├── extractive.rs  # Extractive fusion (main algorithm)
├── geometric.rs   # Geometric D4 group composition
└── utils.rs       # Helper functions (try_as_lut_op, try_as_matrix_op)
```

### Key Difference from Old Design

**Old**: 7 all-or-nothing strategies (Geometric-only, Structural, Matrix, LUT, Mixed, General, Smart)

**New**: 1 extractive algorithm that fuses what it CAN

### Benefits

1. **Predictable**: Transforms declare their semantics via `ReorderRule`
2. **Principled**: Based on algebraic properties, not heuristics
3. **Extensible**: New transforms just declare their `ReorderRule`
4. **Extractive**: Fuses contiguous groups instead of all-or-nothing

---

## Distribution API

All transforms support a **unified distribution API**:

```python
from sinter import Compose, Brightness, Uniform, Constant, Bernoulli

# Implicit constant
Compose([Brightness(delta=50)])

# Explicit distributions
Compose([
    HorizontalFlip(p=Bernoulli(0.5)),
    Brightness(delta=Uniform(-30, 30)),
    Contrast(factor=Constant(1.2)),
])

# Sample once for deterministic reuse
sampled = pipeline.sample_with_seed(42)
result1 = sampled.apply(img1)
result2 = sampled.apply(img2)
```

**Available distributions**: `Constant`, `Uniform`, `UniformInt`, `Bernoulli`, `Normal`

---

## Module Structure

```
src/
├── core/          # Transform, Executable, FusableImage, BarrierImage
├── sampled_ir/    # Sampled IR (pure enums, serializable)
│   ├── ops.rs     # SampledImageOp enum definitions
│   ├── plan.rs    # Plan - wrapper around Vec<SampledImageOp>
│   ├── bridge.rs  # SampledImageProgram → Plan conversion
│   └── transform.rs # Transform trait implementation
├── sampling/      # Distribution sampling infrastructure
├── exec_ir/       # Execution IR + Optimizer + Fusion
│   ├── optimizer/  # 4-phase optimization pipeline
│   └── fusion/     # Fusion strategies
└── transforms/    # Individual transforms (photometric, geometric, kernel, lut, matrix)
```
