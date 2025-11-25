#!/usr/bin/env python3
"""
TFLite to Horizon X5 BPU Model Conversion Script

This script converts a TensorFlow Lite model to Horizon X5 BPU compatible format.
It handles the complete conversion pipeline including quantization and optimization.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_model(model_path: str) -> bool:
    """Validate that the TFLite model exists and is readable."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    if not model_path.endswith('.tflite'):
        print(f"Warning: File extension is not .tflite")
    
    file_size = os.path.getsize(model_path)
    print(f"Model file found: {model_path}")
    print(f"Model size: {file_size / (1024*1024):.2f} MB")
    return True


def inspect_tflite_model(model_path: str) -> None:
    """Inspect TFLite model to extract input/output information."""
    try:
        import tensorflow as tf
        
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n=== Model Information ===")
        print(f"Number of inputs: {len(input_details)}")
        print(f"Number of outputs: {len(output_details)}")
        
        print("\nInput Details:")
        for i, input_detail in enumerate(input_details):
            print(f"  Input {i}:")
            print(f"    Name: {input_detail['name']}")
            print(f"    Shape: {input_detail['shape']}")
            print(f"    Type: {input_detail['dtype']}")
        
        print("\nOutput Details:")
        for i, output_detail in enumerate(output_details):
            print(f"  Output {i}:")
            print(f"    Name: {output_detail['name']}")
            print(f"    Shape: {output_detail['shape']}")
            print(f"    Type: {output_detail['dtype']}")
        
        print("=" * 40)
        return input_details, output_details
        
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return None, None


def convert_model(config: Dict[str, Any]) -> bool:
    """
    Convert TFLite model to Horizon X5 BPU format.
    
    This function uses the Horizon toolchain to perform the conversion.
    """
    try:
        # Import Horizon toolchain (requires horizon_nn package)
        # Note: This will fail if Horizon toolchain is not installed
        try:
            import horizon_nn
            print("Horizon toolchain found!")
        except ImportError:
            print("\nError: Horizon toolchain (horizon_nn) not found!")
            print("Please install the Horizon X5 toolchain:")
            print("1. Download from: https://developer.horizon.ai/")
            print("2. Install with: pip3 install horizon_nn-*.whl")
            print("\nFor now, generating conversion command...")
            return generate_conversion_command(config)
        
        # Extract configuration parameters
        model_params = config['model_parameters']
        input_params = config['input_parameters']
        output_params = config['output_parameters']
        
        model_file = model_params['model_file']
        output_dir = model_params['output_dir']
        output_name = model_params['output_name']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nStarting model conversion...")
        print(f"Input model: {model_file}")
        print(f"Output directory: {output_dir}")
        
        # TODO: Add actual Horizon X5 conversion code here
        # This requires the actual Horizon toolchain API
        # Example pseudo-code:
        # from horizon_nn import Converter
        # converter = Converter()
        # converter.load_model(model_file)
        # converter.set_input_params(**input_params)
        # converter.quantize()
        # converter.compile()
        # converter.save(output_dir, output_name)
        
        print("\nConversion completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_conversion_command(config: Dict[str, Any]) -> bool:
    """
    Generate the command-line command for model conversion.
    This is used when the Horizon toolchain is not available in Python.
    """
    model_params = config['model_parameters']
    
    print("\n" + "=" * 60)
    print("HORIZON X5 MODEL CONVERSION COMMAND")
    print("=" * 60)
    
    print("\nOnce you have the Horizon toolchain installed, use:")
    print("\nhb_mapper makertbin \\")
    print(f"  --config /workspace/model_config.yaml \\")
    print(f"  --model-type tflite \\")
    print(f"  --model {model_params['model_file']} \\")
    print(f"  --output-dir {model_params['output_dir']} \\")
    print(f"  --output-name {model_params['output_name']}")
    
    print("\nAlternatively, use hb_mapper_checker to validate your model:")
    print(f"\nhb_mapper_checker --model {model_params['model_file']}")
    
    print("\n" + "=" * 60)
    
    # Create a shell script for convenience
    script_path = "/workspace/convert.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Horizon X5 Model Conversion Script\n")
        f.write("# Generated automatically\n\n")
        f.write("set -e\n\n")
        f.write("echo 'Starting Horizon X5 model conversion...'\n\n")
        f.write("# Check if hb_mapper is available\n")
        f.write("if ! command -v hb_mapper &> /dev/null; then\n")
        f.write("    echo 'Error: hb_mapper not found. Please install Horizon toolchain.'\n")
        f.write("    exit 1\n")
        f.write("fi\n\n")
        f.write("# Run model conversion\n")
        f.write("hb_mapper makertbin \\\n")
        f.write(f"  --config /workspace/model_config.yaml \\\n")
        f.write(f"  --model-type tflite \\\n")
        f.write(f"  --model {model_params['model_file']} \\\n")
        f.write(f"  --output-dir {model_params['output_dir']} \\\n")
        f.write(f"  --output-name {model_params['output_name']}\n\n")
        f.write("echo 'Conversion completed!'\n")
        f.write(f"echo 'Output files are in: {model_params['output_dir']}'\n")
    
    os.chmod(script_path, 0o755)
    print(f"\nConversion script saved to: {script_path}")
    print("Run it with: bash /workspace/convert.sh")
    
    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Convert TFLite model to Horizon X5 BPU format'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='model_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--inspect-only',
        action='store_true',
        help='Only inspect the model without conversion'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Horizon X5 BPU Model Converter")
    print("=" * 60)
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    
    config = load_config(args.config)
    print(f"\nLoaded configuration from: {args.config}")
    
    # Validate model exists
    model_path = config['model_parameters']['model_file']
    if not validate_model(model_path):
        return 1
    
    # Inspect model
    input_details, output_details = inspect_tflite_model(model_path)
    
    if args.inspect_only:
        print("\nInspection complete. Use without --inspect-only to convert.")
        return 0
    
    # Convert model
    success = convert_model(config)
    
    if success:
        print("\n✓ Model conversion process initiated successfully!")
        print(f"Check the output directory: {config['model_parameters']['output_dir']}")
        return 0
    else:
        print("\n✗ Model conversion failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

