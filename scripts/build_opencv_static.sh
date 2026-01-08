#!/usr/bin/env bash
# Build minimal static OpenCV (core + imgproc only)
#
# Usage:
#   ./scripts/build_opencv_static.sh [/path/to/install]
#
# Defaults to ./opencv-static/build if no path specified

set -e

OPENCV_VERSION=${OPENCV_VERSION:-4.12.0}
INSTALL_DIR=${1:-$(pwd)/opencv-static/build}
OPENCV_DIR=$(pwd)/opencv-static/src

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "OpenCV Static Build Script"
echo "========================================"
echo "Version: $OPENCV_VERSION"
echo "Install dir: $INSTALL_DIR"
echo ""

# Detect number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(nproc 2>/dev/null || echo 4)
fi
echo "Using $CORES cores for build"
echo ""

# Check for required tools
for cmd in git cmake make; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd not found. Please install it first.${NC}"
        exit 1
    fi
done

# Clone OpenCV if not present
if [ ! -d "$OPENCV_DIR" ]; then
    echo -e "${YELLOW}Cloning OpenCV ${OPENCV_VERSION}...${NC}"
    mkdir -p "$OPENCV_DIR"
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git "$OPENCV_DIR"
else
    echo -e "${GREEN}OpenCV source already exists at $OPENCV_DIR${NC}"
fi

# Create build directory
BUILD_DIR="$OPENCV_DIR/build_static"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo -e "${YELLOW}Configuring OpenCV (minimal build)...${NC}"
echo "  Modules: core, imgproc"
echo "  Disabled: TBB, image codecs, FFmpeg, GTK, GStreamer"
echo ""

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_DEFAULT_CMP0069=NEW \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_LIST=core,imgproc \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_GTK=OFF \
    -DWITH_VTK=OFF \
    -DWITH_JPEG=OFF \
    -DWITH_PNG=OFF \
    -DWITH_TIFF=OFF \
    -DWITH_1394=OFF \
    -DWITH_WEBP=OFF \
    -DWITH_TBB=OFF \
    -DWITH_OPENMP=OFF \
    -DWITH_IPP=OFF \
    -DWITH_OPENCL=ON \
    -DBUILD_WITH_DYNAMIC_IPP=OFF \
    -DWITH_EIGEN=OFF \
    -DWITH_FREETYPE=OFF \
    -DWITH_GTHREAD=OFF \
    -DWITH_CAROTENE=ON \
    -DWITH_LAPACK=ON \
    -DBUILD_ITT=OFF \
    -DWITH_ITT=OFF \
    -DCV_TRACE=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_calib3d=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DBUILD_opencv_features2d=OFF \
    -DBUILD_opencv_flann=OFF \
    -DBUILD_opencv_gapi=OFF \
    -DBUILD_opencv_highgui=OFF \
    -DBUILD_opencv_imgcodecs=OFF \
    -DBUILD_opencv_ml=OFF \
    -DBUILD_opencv_objdetect=OFF \
    -DBUILD_opencv_photo=OFF \
    -DBUILD_opencv_stitching=OFF \
    -DBUILD_opencv_video=OFF \
    -DBUILD_opencv_videoio=OFF

echo ""
echo -e "${YELLOW}Building OpenCV...${NC}"
make -j${CORES}

echo ""
echo -e "${YELLOW}Installing to $INSTALL_DIR...${NC}"
make install

echo ""
echo -e "${GREEN}========================================"
echo "Build complete!"
echo "========================================${NC}"
echo ""
echo "Static libraries installed to: $INSTALL_DIR"
echo ""
echo "To use with Sinter:"
echo "  export OPENCV_STATIC_DIR=$INSTALL_DIR"
echo "  maturin develop --release --features 'python,opencv-static'"
echo ""
echo "Expected library files:"
ls -lh "$INSTALL_DIR/lib"/libopencv_*.{a,dylib} 2>/dev/null || ls -lh "$INSTALL_DIR/lib"/libopencv_*.{a,so} 2>/dev/null || true
