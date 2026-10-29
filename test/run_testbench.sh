#!/usr/bin/env bash
# One-command testbench for Project Update 1.
# Run INSIDE the Vitis-AI container after README section 6.3 (setup.sh sourced,
# conda env vitis-ai-pytorch active, setup/install_float_deps.sh run).
#
#   bash test/run_testbench.sh                 full: fresh DPU runs -> compare vs committed golden -> tracking
#   bash test/run_testbench.sh --offline       no DPU: validate committed DPU outputs + run tracking
#   bash test/run_testbench.sh --regen-golden  also regenerate the FP32 golden outputs on CPU first
set -uo pipefail
cd "$(dirname "$0")/.."

MODE=full; REGEN=0
for arg in "$@"; do
  case "$arg" in
    --offline) MODE=offline ;;
    --regen-golden) REGEN=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

fail=0
step()  { echo; echo "================ $* ================"; }
check() { if "$@"; then echo "  -> stage OK"; else echo "  -> stage FAILED"; fail=1; fi; }

# ---- prerequisites ----
if [ "$MODE" = full ]; then
  if [ -z "${XLNX_VART_FIRMWARE:-}" ]; then
    echo "DPU not set up. Run first:"
    echo "  source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_4pe_miscdwc"
    exit 2
  fi
  [ -d model/downloads/yolox_nano_pt ] || ./model/download_model.sh
  [ -d data/mot17_subset ] || ./data/download_mot17.sh
fi

# ---- optional: regenerate golden FP32 outputs (CPU) ----
if [ "$REGEN" = 1 ]; then
  step "Regenerating FP32 golden outputs (CPU)"
  check python src/detect_float.py data/test_images/dog.jpg --out golden/float_dog.json
  check python src/detect_float.py data/mot17_subset/*.jpg  --out golden/float_mot17_09_subset.json
fi

# ---- fresh DPU runs ----
if [ "$MODE" = full ]; then
  step "Layer 1: DPU INT8 on dog.jpg"
  check python src/detect_dpu.py data/test_images/dog.jpg --out results/dpu_dog.json
  step "Layer 1: DPU INT8 on 40 MOT17-09 frames"
  check python src/detect_dpu.py data/mot17_subset/*.jpg --out results/dpu_mot17_09_subset.json
  step "Layer 2 input: DPU INT8 on all 525 MOT17-09 frames"
  check python src/detect_dpu.py data/MOT17/train/MOT17-09-FRCNN/img1/*.jpg \
        --out results/dpu_mot17_09_full.json
fi

# ---- testbench: DPU vs committed golden ----
step "Testbench 1: dog.jpg, DPU vs golden (expected IoU 0.952, PASS)"
check python test/compare.py --golden golden/float_dog.json --test results/dpu_dog.json \
      --report results/compare_report_dog.json

step "Testbench 2: 40 MOT17-09 frames, person, DPU vs golden (expected mAP@0.5 0.733, PASS)"
check python test/compare.py --quiet --classes person \
      --golden golden/float_mot17_09_subset.json --test results/dpu_mot17_09_subset.json \
      --report results/compare_report_person.json

# ---- Layer 2: tracking ----
step "Layer 2: KF tracking on DPU detections (525 frames)"
check python src/run_tracker.py --vis-every 25

echo
if [ "$fail" = 0 ]; then echo "ALL STAGES PASSED"; else echo "SOME STAGES FAILED (see above)"; fi
exit "$fail"
