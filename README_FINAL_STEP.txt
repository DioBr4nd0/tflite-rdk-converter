════════════════════════════════════════════════════════════════════════════
  YOU'RE 95% DONE! FINAL STEP TO GET YOUR .bin FILE
════════════════════════════════════════════════════════════════════════════

✓ COMPLETED: TFLite → ONNX conversion
  Your file: output/face_landmark.onnx (2.3 MB)

✗ BLOCKED ON MAC: ONNX → .bin conversion
  Why: Docker AMD64 image won't run on Apple Silicon (ARM64)
  Solution: Run conversion on x86 machine OR RDK X5 device

════════════════════════════════════════════════════════════════════════════
  OPTION 1: Convert on RDK X5 Device (RECOMMENDED - 2 minutes)
════════════════════════════════════════════════════════════════════════════

1. Transfer files to RDK X5:
   scp output/face_landmark.onnx FINAL_CONVERSION_PACKAGE.sh root@<RDK_X5_IP>:/userdata/models/

2. SSH to RDK X5:
   ssh root@<RDK_X5_IP>

3. Run conversion:
   cd /userdata/models
   bash FINAL_CONVERSION_PACKAGE.sh

4. Done! You'll have: face_landmark_bpu.bin

════════════════════════════════════════════════════════════════════════════
  OPTION 2: Convert on Cloud x86 Linux Instance (5-10 minutes)
════════════════════════════════════════════════════════════════════════════

Use ANY x86 Linux machine (AWS EC2, Google Cloud, Azure, etc.):

1. Launch x86 Ubuntu instance
2. Install Docker: sudo apt update && sudo apt install docker.io -y
3. Upload files:
   scp output/face_landmark.onnx FINAL_CONVERSION_PACKAGE.sh user@instance:/tmp/
4. SSH and run:
   cd /tmp
   sudo bash FINAL_CONVERSION_PACKAGE.sh
5. Download result:
   scp user@instance:/tmp/horizon_output/face_landmark_bpu.bin .

════════════════════════════════════════════════════════════════════════════
  QUICK COMMAND (On RDK X5 Without Script)
════════════════════════════════════════════════════════════════════════════

# Transfer ONNX
scp output/face_landmark.onnx root@<RDK_X5_IP>:/userdata/models/

# SSH and convert
ssh root@<RDK_X5_IP>
cd /userdata/models
hb_mapper makertbin \
  --model face_landmark.onnx \
  --model-type onnx \
  --march bayes-e \
  --input-name input_1 \
  --input-shape "1,192,192,3" \
  --output face_landmark_bpu.bin

# That's it! Takes ~2-5 minutes.

════════════════════════════════════════════════════════════════════════════
  WHY CAN'T WE DO IT ON YOUR MAC?
════════════════════════════════════════════════════════════════════════════

Your Mac: ARM64 (Apple Silicon M1/M2/M3)
Horizon Docker Image: AMD64 (Intel x86)
Docker on Mac: Cannot properly emulate x86 binaries

This is a fundamental architecture incompatibility.
The conversion MUST run on x86 hardware OR the ARM RDK X5 device itself.

════════════════════════════════════════════════════════════════════════════
  FILES IN THIS PACKAGE
════════════════════════════════════════════════════════════════════════════

output/face_landmark.onnx              - Your converted ONNX model (READY!)
FINAL_CONVERSION_PACKAGE.sh            - Auto-detection conversion script
horizon_config.yaml                    - Configuration for hb_mapper
README_FINAL_STEP.txt                  - This file

════════════════════════════════════════════════════════════════════════════
  ESTIMATED TIME
════════════════════════════════════════════════════════════════════════════

On RDK X5:          2-5 minutes
On Cloud x86:       5-10 minutes (including setup)
Total from TFLite:  Already 95% done!

════════════════════════════════════════════════════════════════════════════
