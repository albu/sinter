#!/bin/bash
set -e

# Runs the benchmark suite with the local Python environment (no Docker).
# Requires the `sinter` release module built into the active interpreter:
#   maturin develop --release --features python

PYTHON="${PYTHON:-python3}"

echo "🚀 Running individual benchmarks..."
"$PYTHON" python/benchmarks/benchmark_individual.py

echo "📊 Running fusion benchmarks..."
"$PYTHON" python/benchmarks/benchmark_fusion.py
