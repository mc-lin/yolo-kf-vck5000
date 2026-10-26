#!/usr/bin/env bash
# Downloads YOLOX-nano (Vitis-AI 3.0 Model Zoo):
#   1) float + quantized PyTorch package -> golden FP32 reference
#   2) prebuilt xmodel for VCK5000 DPUCVDX8H-4pe -> DPU inference
#      (pair with: source setup.sh DPUCVDX8H_4pe_miscdwc)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p downloads && cd downloads

BASE="https://www.xilinx.com/bin/public/openDownload?filename="
FLOAT=pt_yolox-nano_coco_416_416_1G_3.0.zip
XMODEL=yolox_nano_pt-vck5000-DPUCVDX8H-4pe-r3.0.0.tar.gz

fetch() {
  local name=$1 sum=$2
  [ -f "$name" ] || wget -O "$name" "${BASE}${name}"
  echo "$sum  $name" | md5sum -c -
}

fetch "$FLOAT"  9c0280cdea713a335efa213a0c8dda63
fetch "$XMODEL" 3269e19a095c601facac9ac947ea6f58

unzip -q -o "$FLOAT"
tar -xzf "$XMODEL"
echo "Done. Contents of model/downloads:"
ls
