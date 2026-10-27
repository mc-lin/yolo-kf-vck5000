#!/usr/bin/env bash
# Run once per container session (pip --user installs are lost when the container exits).
# Usage:  conda activate vitis-ai-pytorch && bash setup/install_float_deps.sh
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)/model/downloads/pt_yolox-nano_coco_416_416_1G_3.0"
pip install --user -r "$PKG/requirements.txt"
cd "$PKG/code" && pip install --user -v -e .
