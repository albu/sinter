# AGENTS.md

Guidance for AI coding assistants (Claude, Gemini, Antigravity, Cursor, Copilot).

## Documentation Index

Load docs **on demand** based on task to conserve context tokens:

| File | Purpose | When to Load |
|---|---|---|
| **AGENTS.md** | Commands, architecture rules, memory semantics | Always (quick reference) |
| **ARCHITECTURE.md** | Two-phase IR, compiler optimizer, fusion rules | Architecture & compiler tasks |
| **DEVELOPMENT.md** | Step-by-step adding & registering transforms | Adding new transforms |
| **OPERATORS.md** | Supported transforms & fusion matrix | Checking operator capabilities |

---

## Commands

Always use `-q` to keep output token-efficient:

```bash
# Rust build & test
cargo build -q 2>&1 | grep -E "(error|warning:.*generated|Finished)"
cargo test -q 2>&1 | grep -E "(^test |^running |FAILED|passed|failed|error:)"
cargo test test_name -q 2>&1 | grep -E "(^test |FAILED|passed|failed|error:)"

# Python extension build (NEVER use cargo build --features python!)
maturin develop --features python --release -q  # or ./scripts/rebuild_sinter.sh

# Python tests & benchmarks (always --release for benchmarks: debug is 15-40x slower)
pytest python/tests/ -q 2>&1 | grep -E "(PASSED|FAILED|ERROR|test_|===)"
python python/benchmarks/benchmark_fusion.py
```

---

## Critical Rules & Truths

1. **100% Pure Native Rust + SIMD**: Zero OpenCV / C++ dependencies. All kernels (Gaussian, Median, HSV, Affine, etc.) are native SIMD.
2. **Memory Semantics**:
   - Default is **copy-by-default / safe** (`inplace=False`). Original arrays are never modified.
   - Out-of-place pipelines (`Resize`, `Pad`, `Crop`, `Affine`) have **zero copy tax** (allocates destination buffer directly).
   - In-place pipelines (`Brightness`, `Contrast`, `LUT`) allocate a defensive copy unless `inplace=True` is explicitly passed.
3. **Return Conventions**:
   - `apply(img)` and bare `transform(img)` return the transformed **array / tensor**.
   - `transform(image=img, mask=..., bboxes=...)` or `Compose(...)` returns a **dict** of targets.
4. **Multimodal Target Handling**:
   - `mask` (singular) returns `{"mask": ...}`; `masks` (plural) returns `{"masks": ...}`.
   - Accepts NumPy ndarrays, PyTorch CHW tensors, and Python lists for coords (preserves container types across round-trips).
   - BBox format aliases: `"coco"` (`"xywh"`), `"pascal_voc"` (`"xyxy"`), `"yolo"` (`"rel_cxcywh"`), `"albumentations"` (`"rel_xyxy"`).
5. **Container & Pipeline Ergonomics**:
   - `len(p)`, `p[i]`, `p[1:4]` (slicing returns sub-`Compose`), `p1 + p2`, `for t in p:`.
   - Direct introspection: `p.explain()`, `p.summary()`, `p.to_mermaid()`, `p.visualize()`, `p.sample()`.
   - Parallel batching: `p.apply_batch(batch, num_threads=4)` (Rayon multi-threading + Python GIL release).
6. **Distributions**:
   - `Constant(v)`, `Uniform(min, max)`, `UniformInt(min, max)`, `Bernoulli(p)`, `Normal(mu, sigma)`. Transforms also accept plain tuples `(-20, 20)` and scalars.

---

## Architecture at a Glance

```
Planning:   Sampled IR (Plan) → Optimizer (4-phase) → ExecPlan
Execution:  ExecPlan (Fused LUT / Matrix / D4 Geometric blocks + Barriers)
```

Adding a transform:
1. Implement `Transform` + `Executable` traits in `src/transforms/`.
2. For photometric ops: implement `LutOp` or `MatrixOp` to enable compiler fusion.
3. Expose Python wrapper in `src/python/transforms/` and register in `src/exec_ir/execution.rs`.
