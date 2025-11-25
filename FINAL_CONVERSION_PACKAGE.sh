#!/bin/bash
# ========================================================================
# FINAL STEP: Convert ONNX to .bin for RDK X5
# Run this script on: RDK X5 device OR any x86 Linux machine with Docker
# ========================================================================

set -e

echo "========================================================================="
echo "  Horizon X5 BPU - Final ONNX to .bin Conversion"
echo "========================================================================="
echo ""

# Detect architecture
ARCH=$(uname -m)
echo "System Architecture: $ARCH"
echo ""

if [[ "$ARCH" == "aarch64" ]]; then
    echo "✓ Running on ARM64 (likely RDK X5)"
    echo "  Using local hb_mapper toolchain..."
    echo ""
    
    # Check if hb_mapper exists
    if command -v hb_mapper &> /dev/null; then
        echo "✓ hb_mapper found!"
        
        # Run conversion directly
        echo ""
        echo "[1/2] Converting ONNX to BPU .bin format..."
        hb_mapper makertbin \
          --model face_landmark.onnx \
          --model-type onnx \
          --march bayes-e \
          --input-name input_1 \
          --input-shape "1,192,192,3" \
          --output face_landmark_bpu.bin
        
        echo ""
        echo "========================================================================="
        echo "✓ CONVERSION COMPLETE!"
        echo "========================================================================="
        echo ""
        echo "Output: face_landmark_bpu.bin"
        ls -lh face_landmark_bpu.bin
        echo ""
        echo "Ready for inference on RDK X5!"
        echo "========================================================================="
        
    else
        echo "✗ hb_mapper not found!"
        echo "  Please install Horizon toolchain first."
        exit 1
    fi
    
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "✓ Running on x86_64"
    echo "  Using Docker container with Horizon toolchain..."
    echo ""
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "✗ Docker not installed!"
        echo "  Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    echo "[1/3] Pulling Horizon toolchain Docker image..."
    docker pull openexplorer/ai_toolchain_ubuntu_20_j5_gpu:v1.1.74
    
    echo ""
    echo "[2/3] Creating conversion config..."
    cat > horizon_config.yaml << 'YAML_EOF'
model_parameters:
  onnx_model: '/workspace/face_landmark.onnx'
  march: 'bayes-e'
  working_dir: '/workspace/horizon_output'
  output_model_file_prefix: 'face_landmark_bpu'

input_parameters:
  input_name: 'input_1'
  input_shape: '1,192,192,3'
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NHWC'
  input_layout_rt: 'NHWC'
  norm_type: 'data_scale'
  mean_value: 127.5
  scale_value: 127.5

calibration_parameters:
  cal_data_type: 'float32'
  calibration_type: 'default'

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
YAML_EOF
    
    mkdir -p horizon_output
    
    echo ""
    echo "[3/3] Running conversion in Docker..."
    docker run --rm \
      -v $(pwd):/workspace \
      -w /workspace \
      openexplorer/ai_toolchain_ubuntu_20_j5_gpu:v1.1.74 \
      hb_mapper makertbin --config horizon_config.yaml --model-type onnx
    
    echo ""
    echo "========================================================================="
    echo "✓ CONVERSION COMPLETE!"
    echo "========================================================================="
    echo ""
    echo "Output directory: horizon_output/"
    ls -lh horizon_output/
    echo ""
    echo "BPU model: horizon_output/face_landmark_bpu.bin"
    echo "========================================================================="
    
else
    echo "✗ Unsupported architecture: $ARCH"
    echo "  This script works on: x86_64 (with Docker) or aarch64 (RDK X5)"
    exit 1
fi
