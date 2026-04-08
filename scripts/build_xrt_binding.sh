#!/bin/bash
# build_xrt_binding.sh — Build and install vten_xrt Python extension
#
# Prerequisites:
#   - XRT SDK installed (/opt/xilinx/xrt or XILINX_XRT env var)
#   - pybind11: pip install pybind11
#   - cmake >= 3.15
#
# Usage: bash scripts/build_xrt_binding.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
XRT_BINDING_DIR="$PROJECT_DIR/vten/backend/_xrt_binding"

# Check XRT SDK
if [ -n "$XILINX_XRT" ]; then
    XRT_DIR="$XILINX_XRT"
elif [ -d /opt/xilinx/xrt ]; then
    XRT_DIR="/opt/xilinx/xrt"
    echo "Sourcing XRT setup..."
    source /opt/xilinx/xrt/setup.sh
else
    echo "ERROR: XRT SDK not found."
    echo "  Install XRT: https://github.com/Xilinx/XRT"
    echo "  Or set XILINX_XRT environment variable."
    exit 1
fi
echo "XRT SDK: $XRT_DIR"

# Check pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "Installing pybind11..."
    pip install pybind11
fi

# Check cmake
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install cmake >= 3.15."
    exit 1
fi

# Build
echo "=== Building vten_xrt ==="
BUILD_DIR="$XRT_BINDING_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install to site-packages
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
SO_FILE=$(find "$BUILD_DIR" -name "vten_xrt*.so" -type f | head -1)

if [ -z "$SO_FILE" ]; then
    echo "ERROR: vten_xrt.so not built"
    exit 1
fi

echo "=== Installing to $SITE_PACKAGES ==="
cp "$SO_FILE" "$SITE_PACKAGES/"

echo ""
echo "=== Done ==="
echo "  vten_xrt installed: $SITE_PACKAGES/$(basename $SO_FILE)"
echo "  Verify: python3 -c 'import vten_xrt; print(dir(vten_xrt))'"
