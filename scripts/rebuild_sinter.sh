#!/usr/bin/env bash
# Deterministic rebuild + install of the sinter extension.
#
# Background: the .venv extension has repeatedly been replaced by a stale
# wheel that uv/maturin resolved from its cache (an Aug 30 build), causing
# ~64 Python test failures that are NOT present in the source. This script:
#
#   1. builds a fresh wheel from the CURRENT tree,
#   2. force-installs it into .venv with --no-deps (bypassing stale cache),
#   3. verifies the installed build behaves like the current source.
#
# Usage: ./scripts/rebuild_sinter.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
OUT=dist
mkdir -p "$OUT"

echo "==> Building wheel (release)"
"$PY" -m maturin build --release --out "$OUT"

WHEEL="$(ls -t "$OUT"/sinter-*.whl | head -1)"
echo "==> Installing $WHEEL"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
uv pip install --no-deps --force-reinstall --python .venv/bin/python "$WHEEL"

echo "==> Verifying install behavior"
"$PY" scripts/verify_sinter_install.py
echo "==> Done"
