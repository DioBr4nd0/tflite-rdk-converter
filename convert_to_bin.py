#!/usr/bin/env python3
"""
ONNX to Horizon X5 BPU .bin Converter
This script converts the ONNX model to BPU .bin format using Horizon toolchain
"""

import os
import sys
import subprocess
import yaml

print("=" * 70)
print("ONNX to Horizon X5 BPU .bin Converter")
print("=" * 70)

# Check if hb_mapper is available
def check_horizon_toolchain():
    """Check if Horizon toolchain is installed"""
    try:
        result = subprocess.run(['hb_mapper', '--version'], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            print("\n✓ Horizon toolchain found!")
            print(f"  Version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("\n✗ Horizon toolchain NOT found!")
    print("\n" + "=" * 70)
    print("INSTALLATION REQUIRED")
    print("=" * 70)
    print("\nOption 1: Manual Installation")
    print("  1. Register at: https://developer.d-robotics.cc/")
    print("  2. Download: AI Toolchain for X5")
    print("  3. Install: pip install horizon_*.whl")
    print("\nOption 2: Use RDK X5 Device")
    print("  The toolchain is pre-installed on RDK X5 devices")
    print("  Transfer ONNX model and convert there")
    print("\nOption 3: Docker (if available)")
    print("  Contact D-Robotics for official Docker image")
    print("=" * 70)
    return False

# Create Horizon configuration
def create_horizon_config():
    """Create configuration file for hb_mapper"""
    
    config = {
        'model_parameters': {
            'onnx_model': 'face_landmark.onnx',
            'march': 'bayes-e',  # X5 architecture
            'working_dir': 'horizon_output',
            'output_model_file_prefix': 'face_landmark_bpu'
        },
        'input_parameters': {
            'input_name': 'input_1',
            'input_shape': '1,192,192,3',
            'input_type_rt': 'nv12',
            'input_type_train': 'rgb',
            'input_layout_train': 'NHWC',
            'input_layout_rt': 'NHWC',
            'norm_type': 'data_scale',
            'mean_value': 127.5,
            'scale_value': 127.5
        },
        'calibration_parameters': {
            'cal_data_type': 'float32',
            'calibration_type': 'default',
            'max_percentile': 0.99999
        },
        'compiler_parameters': {
            'compile_mode': 'latency',
            'optimize_level': 'O3',
            'debug': False
        }
    }
    
    config_path = 'output/horizon_conversion_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✓ Configuration created: {config_path}")
    return config_path

# Run conversion
def convert_to_bin(config_path):
    """Convert ONNX to .bin using hb_mapper"""
    
    onnx_path = 'output/face_landmark.onnx'
    
    if not os.path.exists(onnx_path):
        print(f"\n✗ Error: ONNX model not found: {onnx_path}")
        return False
    
    print(f"\n[1/2] Converting ONNX to BPU .bin format...")
    print(f"  Input: {onnx_path}")
    print(f"  Config: {config_path}")
    
    # Create output directory
    os.makedirs('output/horizon_output', exist_ok=True)
    
    # Run hb_mapper
    cmd = [
        'hb_mapper',
        'makertbin',
        '--config', config_path,
        '--model-type', 'onnx'
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("This may take several minutes...")
    
    try:
        result = subprocess.run(cmd, 
                              capture_output=True,
                              text=True,
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n✓ Conversion successful!")
            print("\n" + result.stdout)
            return True
        else:
            print("\n✗ Conversion failed!")
            print("\nError output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n✗ Error running hb_mapper: {e}")
        return False

# Main execution
def main():
    # Check toolchain
    if not check_horizon_toolchain():
        print("\n" + "=" * 70)
        print("WORKAROUND: Convert on RDK X5 Device")
        print("=" * 70)
        print("\n1. Transfer ONNX to RDK X5:")
        print("   scp output/face_landmark.onnx root@<RDK_X5_IP>:/userdata/models/")
        print("\n2. SSH to RDK X5:")
        print("   ssh root@<RDK_X5_IP>")
        print("\n3. Run conversion on device:")
        print("   cd /userdata/models")
        print("   hb_mapper makertbin \\")
        print("     --model face_landmark.onnx \\")
        print("     --model-type onnx \\")
        print("     --march bayes-e \\")
        print("     --input-name input_1 \\")
        print("     --input-shape \"1,192,192,3\" \\")
        print("     --output face_landmark_bpu.bin")
        print("\n" + "=" * 70)
        sys.exit(1)
    
    # Create configuration
    config_path = create_horizon_config()
    
    # Convert to .bin
    success = convert_to_bin(config_path)
    
    if success:
        print("\n" + "=" * 70)
        print("✓ CONVERSION TO .BIN COMPLETE!")
        print("=" * 70)
        print("\nOutput files in: output/horizon_output/")
        print("  - face_landmark_bpu.bin")
        print("  - face_landmark_bpu.hbm")
        print("\nDeploy to RDK X5:")
        print("  scp output/horizon_output/face_landmark_bpu.bin root@<RDK_X5_IP>:/userdata/models/")
        print("=" * 70)
    else:
        print("\n✗ Conversion failed. Check errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()


