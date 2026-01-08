# Supported Operators

Complete reference of all supported transforms, organized by category with fusion status.

**Legend**: [icon] meaning

| Icon | Meaning |
|------|---------|
| ✅ | Fusable - can be fused into single-pass execution |
| ⚡ | Fast - uses optimized kernel (pre-bound) |
| 🔄 | OutOfPlace - allocates new buffer (fusion barrier) |
| ❌ | Not Fusable - independent execution required |

---

## Photometric Transforms

*Per-pixel color operations that modify image values without changing geometry.*

| Transform | Fusable | Traits | Description |
|-----------|---------|--------|-------------|
| **Brightness** | ✅ ⚡ | LutOp | Adjust image brightness |
| **Contrast** | ✅ ⚡ | LutOp | Adjust image contrast |
| **Gamma** | ✅ ⚡ | LutOp | Apply gamma correction |
| **Invert** | ✅ ⚡ | LutOp | Invert image colors |
| **Solarize** | ✅ ⚡ | LutOp | Solarize image (invert above threshold) |
| **Posterize** | ✅ ⚡ | LutOp | Reduce number of color levels |
| **Normalize** | ✅ ⚡ | LutOp | Normalize to mean/std |
| **ToSepia** | ✅ ⚡ | MatrixOp | Convert to sepia tone |
| **ColorTemperature** | ✅ ⚡ | MatrixOp | Adjust color temperature |
| **ColorTint** | ✅ ⚡ | MatrixOp | Apply customizable color tint |
| **ColorBalance** | ✅ ⚡ | MatrixOp | Adjust color balance |
| **ChannelShuffle** | ✅ | MatrixOp | Shuffle RGB channels |
| **ToGray** | 🔄 | - | Convert to grayscale |
| **ToRGB** | 🔄 | - | Convert to RGB (expand channels) |
| **RGBShift** | ❌ | - | Shift RGB channels |
| **HueSaturationValue** | ❌ | - | Adjust HSV (hue, saturation, value) |
| **GaussNoise** | ❌ | - | Add Gaussian noise |
| **SaltAndPepper** | ❌ | - | Add salt-and-pepper noise |
| **MultiplicativeNoise** | ❌ | - | Add multiplicative noise |
| **Equalize** | ✅ | - | Histogram equalization |
| **AutoContrast** | ✅ | - | Automatic contrast adjustment |
| **CoarseDropout** | ❌ | - | Random coarse dropout (rectangular masks) |
| **GridDropout** | ❌ | - | Grid dropout mask |

**Photometric Fusion Strategies:**
- **LutOp** transforms → LUT fusion (single composed 256-entry lookup table)
- **MatrixOp** transforms → Matrix fusion (3x3 matrix composition)

---

## Geometric Transforms

*Spatial transformations that change pixel positions.*

| Transform | Fusable | AccessPattern | Description |
|-----------|---------|---------------|-------------|
| **HorizontalFlip** | ✅ ⚡ | InPlace | Flip horizontally |
| **VerticalFlip** | ✅ ⚡ | InPlace | Flip vertically |
| **Rotate90** | ✅ ⚡ | InPlace | Rotate 90 degrees |
| **Rotate180** | ✅ ⚡ | InPlace | Rotate 180 degrees |
| **Rotate270** | ✅ ⚡ | InPlace | Rotate 270 degrees |
| **Transpose** | ✅ ⚡ | InPlace | Transpose image |
| **Affine** | 🔄 | OutOfPlace | Affine transformation |
| **Resize** | 🔄 | OutOfPlace | Resize image |
| **Crop** | 🔄 | OutOfPlace | Crop image |
| **Pad** | 🔄 | OutOfPlace | Pad image |

**Geometric Fusion Strategies:**
- **Pure geometric** → Geometric fusion (D4 group composition)

**D4 Group Composition Examples:**
```
Rotate90 + Rotate90 → Rotate180
HorizontalFlip + VerticalFlip → Rotate180
Rotate90 + HorizontalFlip → Transpose
```

---

## Kernel Transforms

*Convolution-based operations using spatial kernels.*

| Transform | Fusable | Description |
|-----------|---------|-------------|
| **GaussianBlur** | ❌ | Gaussian blur (optionally OpenCV) |
| **MedianBlur** | ❌ | Median blur |
| **Sharpen** | ❌ | Sharpen kernel |
| **Emboss** | ❌ | Emboss effect |
| **EdgeDetection** | ❌ | Edge detection |

**Note**: Kernel transforms are not fusable due to spatial dependencies.

---

## Fusion Rules

### Can Fuse (✅)
- `AccessPattern::InPlace` **AND** `ShapeEffect::Preserve`
- Geometric transforms (D4 group) compose via group theory
- LutOp transforms compose via LUT composition
- MatrixOp transforms compose via matrix multiplication

### Cannot Fuse (❌)
- `AccessPattern::OutOfPlace` - requires new buffer
- `ShapeEffect::Resize`, `ShapeEffect::Crop`, `ShapeEffect::Pad`
- Channel-dependent operations (RGBShift, HueSaturationValue)
- Random operations (GaussNoise, SaltAndPepper)

### Barriers (🔄)
- Shape-changing transforms break fusion chains
- Intermediate barriers are inserted automatically
- Each barrier starts a new fusion block

---

## Performance Characteristics

| Category | Passes | Allocations | Speedup |
|----------|--------|-------------|---------|
| N LUT fused | 1 | 0 | Nx |
| N Matrix fused | 1 | 0 | Nx |
| N Geometric fused | 1 | 0 | Nx |
| N General fused | 1 | 0 | ~2x |
| N separate | N | N-1 | 1x |

---

## Implementation Notes

### Pre-bound Kernels (⚡)

Transforms with ⚡ icon use pre-bound function pointers for 8-10x speedup:
- Avoids dynamic dispatch overhead
- Set during optimization phase
- Falls back to macro dispatch if not available

### Zero-Copy Execution

- **FusableImage**: Borrowed view into existing data
- **BarrierImage**: Owned buffer (allocated only when needed)
- Fused operations modify in-place with no allocations

---

## Operator Count Summary

| Category | Total | Fusable | Fast (⚡) | Barriers (🔄) |
|----------|-------|---------|----------|---------------|
| Photometric | 23 | 14 | 11 | 2 |
| Geometric | 10 | 6 | 6 | 4 |
| Kernel | 5 | 0 | 0 | 0 |
| **Total** | **38** | **20** | **17** | **6** |

---

## See Also

- [README.md](README.md) - Project overview and quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and fusion details
- [DEVELOPMENT.md](DEVELOPMENT.md) - Adding transforms and extending the system
