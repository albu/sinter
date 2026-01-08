#!/bin/bash
set -e

# Quick correctness check for x86_64 cross-compilation
# This runs locally on ARM64 to catch compilation/logic errors before Hetzner

echo "🔍 Step 1: Checking if cross is installed..."
if ! command -v cross &> /dev/null; then
    echo "❌ cross not found. Installing..."
    cargo install cross
else
    echo "✅ cross is installed: $(cross --version | head -1)"
fi

echo ""
echo "🏗️  Step 2: Checking x86_64 compilation..."
cargo check --target x86_64-unknown-linux-gnu

echo ""
echo "🧪 Step 3: Running cross tests (correctness only)..."
# Run tests - note: SIMD code will be emulated, but logic is tested
cross test --target x86_64-unknown-linux-gnu

echo ""
echo "✅ All checks passed!"
echo "📤 Ready for Hetzner performance testing..."
echo ""
echo "Next steps:"
echo "  1. SSH into Hetzner: hcloud server ssh simd-test"
echo "  2. Clone repo and run: ./run_benchmarks.sh"
echo "  3. Or: docker run -it --rm sinter-bench"
