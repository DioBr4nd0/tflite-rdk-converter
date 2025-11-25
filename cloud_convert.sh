#!/bin/bash
# Cloud Conversion Script - Run this on any x86 Linux machine or cloud instance
# This script converts ONNX to .bin using Horizon toolchain Docker image

set -e

echo "========================================================================"
echo "Horizon X5 ONNX to .bin Converter (Cloud/x86 Linux)"
echo "========================================================================"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found. Please install Docker first."
    exit 1
fi

echo ""
echo "[1/4] Pulling Horizon AI Toolchain Docker image..."
docker pull openexplorer/ai_toolchain_ubuntu_20_j5_gpu:v1.1.74

echo ""
echo "[2/4] Preparing workspace..."
mkdir -p output horizon_output

# Check if ONNX file exists
if [ ! -f "output/face_landmark.onnx" ]; then
    echo "Error: output/face_landmark.onnx not found!"
    echo "Please upload face_landmark.onnx to the output/ directory"
    exit 1
fi

echo "Found ONNX model: output/face_landmark.onnx"

echo ""
echo "[3/4] Creating conversion configuration..."
cat > horizon_convert_config.yaml << 'EOF'
model_parameters:
  onnx_model: '/workspace/output/face_landmark.onnx'
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
  max_percentile: 0.99999

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O3'
  debug: false
EOF

echo "Configuration created: horizon_convert_config.yaml"

echo ""
echo "[4/4] Running Horizon toolchain conversion..."
echo "This may take 5-10 minutes..."

docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  openexplorer/ai_toolchain_ubuntu_20_j5_gpu:v1.1.74 \
  bash -c "
    cd /workspace && \
    hb_mapper makertbin \
      --config horizon_convert_config.yaml \
      --model-type onnx
  "

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ CONVERSION SUCCESSFUL!"
    echo "========================================================================"
    echo ""
    echo "Output files:"
    ls -lh horizon_output/
    echo ""
    echo "BPU model ready for RDK X5:"
    echo "  → horizon_output/face_landmark_bpu.bin"
    echo ""
    echo "Transfer to RDK X5:"
    echo "  scp horizon_output/face_landmark_bpu.bin root@<RDK_X5_IP>:/userdata/models/"
    echo ""
    echo "========================================================================"
else
    echo ""
    echo "✗ Conversion failed. Check the error messages above."
    exit 1
fi

