# YOLOX-nano Detection + Kalman Filter Tracking on AMD VCK5000


YOLOX-nano (INT8) runs on the VCK5000's DPU to detect people in each video frame. 
A Kalman Filter tracker on the host CPU follows each person across frames. The DPU output is validated against the FP32 model on the CPU, tracking is evaluated against MOT17 ground truth, and CPU vs
DPU performance is measured.

This README explains **how to run the code**. Results, metrics, and discussion are in the
report: [`doc/report_update2.md`](doc/report_update2.md).


## 1. Tools and settings

| Item | Value |
|---|---|
| Testbed | Open Cloud Testbed (CloudLab), profile `oct-vck5000` |
| Target device | AMD VCK5000 (Versal), shell `xilinx_vck5000_gen4x8_qdma_base_2` |
| DPU | `DPUCVDX8H`, configuration 4PE (`DPUCVDX8H_4pe_miscdwc`), 350 MHz |
| Vitis-AI | repo and Model Zoo tag `v3.0`; Docker image `xilinx/vitis-ai-pytorch-cpu:latest` |
| Model | YOLOX-nano, COCO, 416×416 input; FP32 (`yolox_nano.pth`) on CPU, INT8 (`yolox_nano_pt.xmodel`, 4PE) on DPU |
| Detection thresholds | confidence 0.3, NMS 0.65 (from `yolox_nano_pt.prototxt`), both pipelines |
| Tracker defaults | greedy IoU matching, `iou_thr 0.3`, `max_misses 15`, `min_hits 3` |
| Input data | `data/test_images/dog.jpg`; MOT17-09 (525 frames, 30 FPS |


## 2. Repository contents

| Path | Purpose |
|---|---|
| `src/detect_float.py` | FP32 YOLOX-nano on the CPU → detections JSON (golden reference) |
| `src/detect_dpu.py` | INT8 YOLOX-nano on the DPU → detections JSON (same format) |
| `src/tracker/kalman.py`, `src/tracker/tracker.py` | Constant-velocity Kalman Filter; multi-object tracker (greedy or Hungarian matching) |
| `src/run_tracker.py` | Runs the tracker over a sequence of DPU detections |
| `test/compare.py` | Testbench: DPU vs FP32 golden (IoU, recall, precision, mAP, PASS/FAIL) |
| `test/eval_tracking.py` | Tracking vs MOT17 ground truth (MOTA, IDF1, ID switches, …) |
| `test/run_testbench.sh` | **One command** that runs detection, validation, tracking, and tracking evaluation |
| `test/run_perf.sh` | CPU FP32 vs DPU INT8 timing |
| `golden/float_dog.json`, `golden/float_mot17_09_subset.json` | **Golden outputs** (FP32 detections) |
| `data/test_images/dog.jpg` | Smoke-test input |
| `data/download_mot17.sh` | Downloads MOT17, keeps MOT17-09, builds the 40-frame validation subset |
| `model/download_model.sh` | Downloads the FP32 and VCK5000 4PE models from the Model Zoo (MD5-checked) |
| `setup/docker_run_mount.patch` | Lets the Vitis-AI container see the project drive |
| `setup/install_float_deps.sh` | Installs AMD's YOLOX code and `motmetrics` (each container session) |
| `doc/report_update2.md` | Project Update 2 report |



## 3. One-time setup (host, outside the container)

```bash
# Docker access. If `groups` does not list docker, run this, then log out and back in.
sudo usermod -aG docker $USER

cd /proj/octfpga-PG0/<user>/
git clone -b project-update2 https://github.com/mc-lin/yolo-kf-vck5000.git
git clone --depth 1 --branch v3.0 https://github.com/Xilinx/Vitis-AI.git

# Make the project drive visible inside the container
cd Vitis-AI && git apply ../yolo-kf-vck5000/setup/docker_run_mount.patch
```


```bash
cd ../yolo-kf-vck5000
./model/download_model.sh      # FP32 package + 4PE xmodel, checksums verified
./data/download_mot17.sh       # MOT17-09, 40-frame subset for validation
```

## 4. Every session (start the container)

```bash
# host
source /opt/xilinx/xrt/setup.sh && xbutil examine    
cd /proj/octfpga-PG0/<user>/Vitis-AI
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest    # must be started from inside Vitis-AI/
```

Inside the container (prompt `vitis-ai-user@...`):

```bash
source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_4pe_miscdwc  
xdputil query                                   
conda activate vitis-ai-pytorch
cd /proj/octfpga-PG0/<user>/yolo-kf-vck5000
bash setup/install_float_deps.sh              
```


## 5. Running the code (container)

### 5.1 Full testbench: one command

```bash
bash test/run_testbench.sh
```
run_testbench.sh runs the whole pipeline: DPU detection, FP32 golden detection, comparison, tracking, and tracking evaluation. It prints a summary of each stage and the final result.

| Stage | Expected |
|---|---|
| Testbench 1: `dog.jpg`, DPU vs golden | 1 match, IoU 0.952, **PASS** |
| Testbench 2: 40 MOT17-09 frames, `person`, DPU vs golden | mean IoU 0.838, recall 0.756, precision 0.809, mAP@0.5 0.733, **PASS** |
| Tracking on all 525 frames | 133 confirmed tracks, 538 gap recoveries |
| Tracking vs MOT17-09 ground truth | MOTA 40.2%, IDF1 38.7%, 265 ID switches |
| Final line | `ALL STAGES PASSED` |

The tracking-evaluation stage reports metrics; it fails only if something breaks (missing file
or package), not on a score threshold.

Options:

```bash
bash test/run_testbench.sh --offline        # skip DPU runs; re-check outputs of an earlier full run in results/
bash test/run_testbench.sh --regen-golden   # also regenerate the FP32 golden outputs on the CPU first
```

`--offline` needs `results/` from a previous full run on the same machine (generated files are
not in the repository).

### 5.2 Performance: CPU FP32 vs DPU INT8

```bash
bash test/run_perf.sh
```

Timing scope on both sides: preprocessing + inference + postprocessing, with the image already
in memory and the model loaded once. The DPU is timed with AMD's
`test_performance_yolovx` for 30 s each with 1 and 4 threads.

Expected (node pc176; values vary slightly between runs and nodes):

| Platform | Output |
|---|---|
| CPU FP32 (40 threads) | about 34 ms/image |
| DPU INT8, 1 thread | `FPS≈587`, `E2E_MEAN≈6810` (µs per batch of 4 images) |
| DPU INT8, 4 threads | `FPS≈1371` |


## 6. Code not written by me

| Code | Source | Used by |
|---|---|---|
| YOLOX preprocessing, inference, postprocessing (`Predictor` in `tools/demo.py`) | AMD Vitis-AI Model Zoo package `pt_yolox-nano_coco_416_416_1G_3.0` (based on YOLOX, Megvii) | `src/detect_float.py` |
| `test_jpeg_yolovx`, `test_performance_yolovx` | AMD Vitis-AI Library samples (`examples/vai_library/samples/yolovx`) | `src/detect_dpu.py`, `test/run_perf.sh` |
| `docker_run.sh`, board setup scripts | AMD Vitis-AI repository | setup |
| `motmetrics` | py-motmetrics (open-source library) | `test/eval_tracking.py` |
| Tracker design | follows SORT (Bewley et al., 2016); no code copied | `src/tracker/` |

My scripts in `src/`, `test/`, `model/`, `data/`, and `setup/` were written with help from
Claude.