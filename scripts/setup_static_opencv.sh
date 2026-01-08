#!/bin/bash
# scripts/setup_static_opencv.sh
# Source this script to set environment variables for static OpenCV linking
# Usage: source scripts/setup_static_opencv.sh

# Get absolute path to repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATIC_DIR="${REPO_ROOT}/opencv-static/build"

if [ ! -d "$STATIC_DIR" ]; then
    echo "❌ Error: OpenCV static build not found at $STATIC_DIR"
    echo "Please run ./scripts/build_opencv_static.sh first."
    return 1 2>/dev/null || exit 1
fi

LIB_PATH="$STATIC_DIR/lib"
INCLUDE_PATH="$STATIC_DIR/include/opencv4"

# Validate paths
if [ ! -d "$LIB_PATH" ]; then
    echo "❌ Error: Library path not found: $LIB_PATH"
    return 1 2>/dev/null || exit 1
fi

# Export variables for the 'opencv' crate build script
# Add 3rdparty libs path (where tegra_hal/carotene lives)
THIRDPARTY_PATH="$LIB_PATH/opencv4/3rdparty"
export OPENCV_LINK_PATHS="$LIB_PATH,$THIRDPARTY_PATH"
export OPENCV_INCLUDE_PATHS="$INCLUDE_PATH"
export OPENCV_LINK_LIBS="static=opencv_imgproc,static=opencv_core,static=tegra_hal,framework=Accelerate,framework=OpenCL" 
export OPENCV_LINK_STATIC=1
export OPENCV_DISABLE_PROBES=1

echo "✅ OpenCV Static Environment Configured"
echo "   Libs: $OPENCV_LINK_LIBS"
echo "   Path: $OPENCV_LINK_PATHS"
