#!/usr/bin/env bash
# Performance: CPU FP32 vs DPU INT8 on the 40-frame MOT17-09 subset.
# Run INSIDE the container after setup.sh, conda activate vitis-ai-pytorch,
# and setup/install_float_deps.sh. Timing scope (both): pre + inference + post,
# image already in memory.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
ls "$REPO"/data/mot17_subset/*.jpg > /tmp/mot17_subset_list.txt

echo "== CPU FP32 (detect_float.py) =="
python src/detect_float.py data/mot17_subset/*.jpg --out /tmp/float_perf.json | tail -2

echo "== DPU INT8 (test_performance_yolovx), 1 and 4 threads, 30 s each =="
mkdir -p results/dpu_vis
[ -e results/dpu_vis/yolox_nano_pt ] || ln -s "$REPO/model/downloads/yolox_nano_pt" results/dpu_vis/yolox_nano_pt
cd results/dpu_vis
PERF=/workspace/examples/vai_library/samples/yolovx/test_performance_yolovx
for T in 1 4; do
  echo "-- threads=$T"
  $PERF yolox_nano_pt /tmp/mot17_subset_list.txt -t "$T" -s 30 2>&1 | grep -E "FPS|MEAN"
done
