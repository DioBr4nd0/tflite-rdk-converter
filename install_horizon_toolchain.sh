#!/bin/bash
# Script to download and setup Horizon X5 Toolchain

echo "Attempting to download Horizon X5 Toolchain..."

# Try to download from D-Robotics developer site
# Note: This typically requires authentication/registration

TOOLCHAIN_URL="https://archive.d-robotics.cc/downloads/ai_toolchain/"

echo ""
echo "The Horizon X5 toolchain requires registration at:"
echo "https://developer.d-robotics.cc/"
echo ""
echo "Manual download steps:"
echo "1. Register at: https://developer.d-robotics.cc/"
echo "2. Navigate to: Downloads > AI Toolchain"
echo "3. Download: horizon_x5_toolchain*.whl or docker image"
echo "4. Place in this directory"
echo ""
echo "Once downloaded, install with:"
echo "  pip install horizon_x5_toolchain*.whl"
echo ""

# Check if already installed
if command -v hb_mapper &> /dev/null; then
    echo "✓ hb_mapper is already installed!"
    hb_mapper --version
else
    echo "✗ hb_mapper not found. Please install Horizon toolchain."
fi

