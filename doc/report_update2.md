# YOLO-nano Detection + Kalman Filter Tracking on AMD VCK5000

## Overview

This project accelerates real-time object detection and tracking. A small YOLO-family CNN, YLOLX-nano, detects objects in each video frame, and a Kalman Filter (KF) tracker follows each person across frames and estimates their position and velocity. 
YOLOX-nano comes from the Vitis AI 3.0 Model Zoo. It was trained in FP32, then quantized to INT8 and compiled for the DPU. 

The three layers of this project:

Layer 1: detection on the DPU, validated against the FP32 model on the CPU.
Layer 2: KF tracking on the host CPU, fed by the DPU detections.
Layer 3: move the KF tracker onto the VCK5000's ARM processor as a PS kernel.

## Results from project update 1

In this first project deadline, I aim to implement the YOLO-nano object detection model combined with a Kalman filter for tracking on the AMD VCK5000 platform. The goal is to have a working pipeline for object detection and tracking. Ensuring each stage functions correctly with Vitis AI. 
YOLO-nano model quantized from FP32 to INT8 and compiled with Vitis-A with each frame produces the bounding boxes, class labels, and confidence scores. 
Layer 1 of this project is the detection stage, and layer 2 is the tracking stage using the Kalman filter. DPU detections are fed into the Kalman filter on the host CPU, which estimates each person's position and velocity over time.

First, I tested the model with a simple dog image to verify the detection and compare them on FP32 on CPU and INT8 on DPU. I used the FP32 version on CPU as the golden reference, and the INT8 version on DPU for performance evaluation.

After ensuring both detections are working on both models (are they using the same yolo model but just run on different hardware?), I proceeded to integrate the Kalman filter for tracking, which goes into layer 2 of the pipeline. 
From the detection stage, the testing dog image has results in the table below.
|    | Class | Confidence | Box (x1, y1, x2, y2) |
|----|-------|------------|---------------------|
| FP32, CPU  | Dog   | 0.891       | (110.61, 85, 437.08, 449.24) |
| INT8, DPU  | Dog   | 0.528       | (108.04, 78.73, 440.04, 454.93) |

IoU between boxes = 0.952. Quantization kept the box location almost unchanged but lowered the confidence on this image.

Validation on 40 frames of MOT17, person class
Every 13th frame of the MOT17 dataset.
| Metric(DPU INT8 vs FP32 golden) | Value |
|----|-------|
| Mean IoU of matched boxes | 0.838 |
| Recall/Precision | 0.756/0.809 |
| Mean confidence change | -0.045 |
| mAP@0.5 vs float | 0.733 |
| Result | Pass (mean IoU >= 0.75 and recall >= 0.75) |

MOT17 only annotated pedestrians on street, aAP@0.5 dropped to 0.105 with the object like skateboard, backpack, handbag in the scene each have score AP = 0, but person makes up 352 of 363 golden boxes.

## Project update 2: what I did for this update

For this update, I measured the CPU PF32 vs DPU INT8 latency and throughput with the same timing scope on both sides. 
I evaluated the tracker against the MOT17-09 human-labeled ground truth (MOTA, IDF1, IDSW)
I changed one tracker setting at a time and added Hungarian matching as an option to check if the tracking could be improved.
I ran both models on my own photo to see how DPU behaves on new input.

## Input and output

Inputs: data/test_images/dog.jpg for detection validation. MOT17-09 from the MOT17 benchmark (525 frames, 1920×1080, 30 FPS, stationary camera, 26 ground-truth pedestrians). 

Outputs and the interpretation of the results:

| Output | What it contains | How to interpret it |
|---|---|---|
| Detection JSON (`golden/float_*.json`, `results/dpu_*.json`) | class, confidence score, box `[x1, y1, x2, y2]` in pixels | The FP32 file is the golden reference. The DPU file is what is being checked |
| `test/compare.py` summary | Matched boxes, IoU, recall, precision, mAP, PASS/FAIL | How closely the DPU reproduces the FP32 model |
| Tracks (`results/tracks_mot17_09.json` / `.txt`) | track ID, box, velocity (vx, vy), detected or predicted | The same ID across frames means the tracker thinks it is the same person |
| `test/eval_tracking.py` summary | MOTA, IDF1, ID switches, FP, FN, and more | How well the tracks match the real people in `gt.txt` |
| `test/run_perf.sh` output | ms per image and FPS for CPU and DPU | Speed of each pipeline |

## Metrics
Detection: IoU, recall, precision, mAP@0.5.
Speed: latency and throughput.
Tracking: FN, FP, IDSW, MOTA, IDF1, MT/PT/ML, MOTP, and FM.
IoU is the intersection over union shows how much 2 boxes overlap and DPU box counts as matching a golden box if the have same class and IoU >=0.5.
Recall is the share of golden boxes that DPU found. 
Precision is the share of DPU boxes that match a golden box.
mAP@0.5 is the mean average precision at IoU threshold 0.5. 
Latency is the time from image going in to its results.
Throughput is the number of images finished per second.
FN indicates no box reported when a real person is there.
FP is a box reported but no real person. 
IDSW is when a person is tracked with one ID in one frame and a different ID in another frame.
MOTA = 1 - (FN + FP + IDSW) / GT boxes. 
MT/PT/ML are the number of ground-truth tracks that were mostly tracked, partially tracked, or mostly lost.
MOTP is the mean of matched boxes
FM is the number of times a track was interrupted and resumed. 

## Results

These are the performances on DPU end-to-end with test_performance_yolovx, and CPU FP32 timing on the same 40 frames.
With 4 threads: 1371 FPS. Four streams overlap the CPU-side work with the DPU work, so the DPU sits idle less.

| Metric | CPU FP32 | DPU INT8 | Speedup |
| --- | --- | --- | --- |
| Latency per image | 34.03 ms | 6.81 ms | about 5.0× |
| Throughput, 1 thread | 29.4 FPS | 587 FPS | about 20× |
| Throughput, 4 threads | 29.4 FPS | 1371 FPS | about 47× |
Latency improved about 5×, much less than the raw inference speedup, since each image waits for its batch and for the CPU-side steps. Throughput improved 20–47×, since batching and parallel streams keep the DPU busy. 
CPU was measured one image at a time, without batching. 

## Tracking vs MOT17-09 ground truth (baseline)

The baseline tracker settings are greedy matching, iou_thr 0.3, max_misses 15, min_hits 3. The evaluation with 'test/eval_tracking.py' with motmetrics 1.4.0 and ground truth boxes with flag 1 and class 1: the match needs IoU >=0.5. 

| MOTA | IDF1 | IDSW | FP | FN | Prcn | Rcll | MT/PT/ML | FM | MOTP (1−IoU) |
|---|---|---|---|---|---|---|---|---|---|
| 40.2% | 38.7% | 265 | 770 | 2150 | 80.5% | 59.6% | 8/14/4 | 185 | 0.234 |

MOTA = 1 − (2150 + 770 + 265) / 5325 = 40.2%. FNs are the biggest contributor to the error, about 40% of all ground-truth boxes, and they mostly come from the detector. IDF1 is lower than MOTA, and 265 IDSW for 26 people is about 10 per person, which shows that the tracks are fragmented.
Recall is 59.6%, which is lower than Layer 1's recall. Layer 1 compared the DPU with the FP32 model, but it compares it with real people. FP32 model also misses some people, and the DPU also misses those, plus its own misses. 

## Tracker experiments
I changed one tracker setting at a time; the DPU detections were the same, only the tracker rules changed. 

| Run | MOTA | IDF1 | IDSW | FP | FN | Rcll | Tracks created |
|---|---|---|---|---|---|---|---|
| Baseline (greedy, 15 / 0.3 / 3) | 40.2% | 38.7% | 265 | 770 | 2150 | 59.6% | 203 |
| `--max-misses 30` | 40.2% | 37.4% | 265 | 791 | 2127 | 60.1% | 172 |
| `--iou-thr 0.2` | 39.5% | 39.5% | 272 | 819 | 2132 | 60.0% | 163 |
| `--min-hits 1` | 35.9% | 37.7% | 291 | 1055 | 2066 | 61.2% | 203 |
| Hungarian matching | 40.0% | 38.6% | 265 | 791 | 2139 | 59.8% | 183 |
| Hungarian + `--iou-thr 0.2` | 39.9% | 35.7% | 278 | 802 | 2119 | 60.2% | 164 |

For max-misses 15 vs 30, the tracks lived longer, but IDSW stayed at 265, and IDF1 went down. During a long gap, the FK keeps moving the track at its last velocity, so after many frames it can be far from the real person. 
For iou_thr 0.3 vs 0.2, the lower threshold allowed more matches; it has the best IDF1, but a few more wrong matches. 
min_hits 3 vs 1: the tracks are created sooner; that has 84 fewer FN, but 285 more false positives, and MOTA dropped 4.3 points. It shows min_hits 3 is useful for filtering one-frame false boxes. 
Hungarian matching created fewer tracks (183 vs 203), so it did avoid some unnecessary new IDs. But ID switches stayed at exactly 265, and MOTA and IDF1 moved by only 0.1–0.2 points. That shows that situations where greedy and Hungarian disagree are rare in this video. 

Conclusion: Tracker-side changes changed the MOTA by at most 0.7 points. The limits are detection recall and matching that only uses box positions; when 2 people are close together, the box overlap cannot tell them apart. Improving the tracking further needs better detection. 

## Pass criterion for detection validation
compare.py uses mean IoU>= 0.75 and recall>=0.75 as the pass criterion (person class, DOU versus FP32), and the 40-frame validation passes with 0.838 and 0.756. 
The thresholds were chosen before the validation run because IoU>=0.75 so when DPU finds a person, the box is in nearly the same place as FP32; recall >= 0.75, so DPU finds most of the people FP32 found. The thresholds are not too tight, since the DPU is a quantized model and the FP32 model is not perfect either.

## Own image test
I ran both models on my own photo of a burger. They both agreed on the main object (a sandwich, IoU 0.975), but disagreed on the objects near the 0.3 confidence threshold. The FP32 model found a person (0.349) that DPU missed, and DPU found 2 bottles and 2 cups that the FP32 model did not report. So recall was 0.5 and precision 0.2 on the image. Quantization moves the confidence scores both up and down, so objects close to the threshold can flip in or out. The image can be found in 'data/test_images/tasty.jpeg'

## Plan for the rest of the project

1. FPGA vs AI Engine: the node has four DPU bitstreams. The 6PE versions run some operations
   on the AI Engines (`6PE_aieDWC` runs depthwise convolution on the AIE, `6PE_aieMISC` runs other
   operations on the AIE), while the 4PE version I use runs them in the FPGA fabric. I will compile
   YOLOX-nano for the other DPU configurations from the quantized model with `vai_c_xir` (the
   Model Zoo only has a 4PE version) and compare speed and accuracy.

2. Data types (INT4): The DPU only runs INT8. I can measure INT4 accuracy by
   simulating it in the Vitis-AI quantizer.

3. GPU baseline: run the FP32 model on a V100 on pc178 or pc179, timed the same way.

4. Layer 3: follow the PS kernel tutorial to move the KF tracker to the VCK5000's ARM
   processor, if the OCT shell supports it, and compare it with the host version.

5. If time allows: Replace the MOT17 with self-recorded video with a stationary camera and do the analysis on it.

## Code I did not write

| Code | Source |
|---|---|
| YOLOX preprocessing, inference, and postprocessing (`Predictor` in `tools/demo.py`), used by `src/detect_float.py` | AMD Vitis-AI Model Zoo package `pt_yolox-nano_coco_416_416_1G_3.0`, based on YOLOX (Megvii) |
| `test_jpeg_yolovx` and `test_performance_yolovx`, called by `src/detect_dpu.py` and `test/run_perf.sh` | AMD Vitis-AI Library samples |
| `docker_run.sh` and the VCK5000 board setup scripts | AMD Vitis-AI repository (I added one mount line, `setup/docker_run_mount.patch`) |
| MOT metrics in `test/eval_tracking.py` | py-motmetrics library |
| Tracker design | Follows SORT (Bewley et al.); no code copied |

The rest of the scripts in `src/`, `test/`, `model/`, `data/`, and `setup/` were written by me with
help from Claude.

## AI disclaimer

I used Claude to help set up the CloudLab and Vitis-AI environment(e.g., Docker and version management). I used Claude to assist with writing the scripts detect_float.py and detect_dpu.py, compare.py, and the KF tracker. For eval_tracking.py I have Claude to analyze the results and debug the issues from the experiments, like discussing the Hungarian matching option, testbench and performance scripts and download scripts. Claude was also help for debugging for example, finding the DPU demo only processes one 4-image batch per cell, or low all-class mAP came from rare classes. Claude was used for brainstorming and understanding the KF, CNN, quantization, Vitis-AI stack, co-simulation, and MOT metrics concepts. Claude was used to help revise the README and the report update to verify if I meet the requirements.

## References

1. AMD/Xilinx, Vitis AI Model Zoo v3.0, https://github.com/Xilinx/Vitis-AI/tree/v3.0/model_zoo
2. AMD/Xilinx, Vitis AI Library (yolovx samples), https://github.com/Xilinx/Vitis-AI
3. Z. Ge, S. Liu, F. Wang, Z. Li, J. Sun, "YOLOX: Exceeding YOLO Series in 2021," arXiv:2107.08430
4. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified, Real-Time Object
   Detection," arXiv:1506.02640
5. MOT Challenge, MOT17 dataset, https://motchallenge.net/data/MOT17/
6. A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler, "MOT16: A Benchmark for Multi-Object
   Tracking," arXiv:1603.00831
7. L. Leal-Taixé, A. Milan, I. Reid, S. Roth, K. Schindler, "MOTChallenge 2015: Towards a
   Benchmark for Multi-Target Tracking," arXiv:1504.01942
8. A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft, "Simple Online and Realtime Tracking," ICIP 2016
9. N. Wojke, A. Bewley, D. Paulus, "Simple Online and Real-Time Tracking with a Deep Association
   Metric" (DeepSORT), ICIP 2017
10. K. Bernardin, R. Stiefelhagen, "Evaluating Multiple Object Tracking Performance: The CLEAR MOT
    Metrics," EURASIP Journal on Image and Video Processing, 2008
11. E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi, "Performance Measures and a Data Set for
    Multi-Target, Multi-Camera Tracking" (IDF1), ECCV Workshops 2016
12. py-motmetrics, https://github.com/cheind/py-motmetrics
13. R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," Journal of Basic
    Engineering, 1960
14. H. W. Kuhn, "The Hungarian Method for the Assignment Problem," Naval Research Logistics
    Quarterly, 1955
15. AMD, "Using PS Kernel to Control AIE Graph on VCK5000" (Vitis 2022.2 tutorial)

