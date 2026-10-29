# YOLO-nano Detection + Kalman Filter Tracking on AMD VCK5000

# Overview

In this first project deadline, I aim to implement the yolo-nano object detection model combined with a Kalman filter for tracking on the AMD VCK5000 platform. The goal is to have working pipeline for object detection and tracking. Ensuring each stage functions correctly with Vitis AI. 
YOLO-nano model quantized from FP32 to INT8 and compiled with Vitis-A with each frame produces the bounding boxes, class labels, and confidence scores. 
Layer 1 of this project is the detection stage, and layer 2 is the tracking stage using the Kalman filter. DPU detections are fed into the Kalman filter on host CPU, which estimates each person's position and velocity over time.

First I tested the model with simple dog image to verify the detection, and compare them on FP32 on CPU and INT8 on DPU. I used FP32 version on CPU as the golden reference, and INT8 version on DPU for performance evaluation.

After ensuring both detections are working on both model(are they using the same yolo model but just ran on different hardware?), I proceeded to integrate the Kalman filter for tracking which goes into the layer 2 of the pipeline. 

## Requirements

- AMD VCK5000 platform
- YOLO-nano model
- Kalman filter implementation
- Python 3.x
- OpenCV
- NumPy
- PyTorch (for YOLO-nano model)

## How to Run
Environment needs the CLoudLab profile of oct-vck5000, Docker image , and DPU config `DPUCVDX8H_4pe_miscdwc`.

Host setup
```bash
sudo usermod -aG docker $USER        # then log out and back in
cd /proj/octfpga-PG0/<user>/
git clone -b project-update-1 https://github.com/mc-lin/yolo-kf-vck5000.git
git clone --depth 1 --branch v3.0 https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI && git apply ../yolo-kf-vck5000/setup/docker_run_mount.patch   # edit path inside if needed
cd ../yolo-kf-vck5000 && ./model/download_model.sh && ./data/download_mot17.sh
```

Starting the container
```bash
cd /proj/octfpga-PG0/<user>/Vitis-AI
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
# inside the container:
source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_4pe_miscdwc
conda activate vitis-ai-pytorch
cd /proj/octfpga-PG0/<user>/yolo-kf-vck5000
bash setup/install_float_deps.sh
```

Running the testbench
```bash
bash test/run_testbench.sh            # full: fresh DPU runs -> compare vs golden -> tracking
bash test/run_testbench.sh --offline  # no DPU needed: committed DPU outputs -> compare -> tracking
```

## Results

From the detection stage the testing dog image have result in table below.
|    | Class | Confidence | Box (x1, y1, x2, y2) |
|----|-------|------------|---------------------|
| FP32, CPU  | Dog   | 0.891       | (110.61, 85, 437.08, 449.24) |
| INT8, DPU  | Dog   | 0.528       | (108.04, 78.73, 440.04, 454.93) |

IoU between boxes = 0.952. Quantization kept the box location almost unchanged but lower the confidence on this image.

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

## Plan for the rest of the project
Layer 2: Evaluate the tracking performance versus the MOT17 ground truth. Reduce the KF noise to improve accuracy.
I will try use self-recorded video as another input to compare the results from MOT17 dataset. The self-recorded video will be stationary camera that is similar to MOT17 datasets.
Layer 3: Investigate moving the KF tracker to a PS kernel running on VCK5000’s ARM
processor. Compare the detection accuracy and latency between host-based and VCK5000.

## AI disclaimer

I used CLaude for helping to set up the CloudLab and Vitis-AI environment(e.g. Docker and version management). I have Claude to assist with writing the script on detect_float.py and detect_dpu.py and compare.py and the KF tracker. I have Claude to anaylize the results and debug the issues from the experiments. Claude was used for brainstorming and understanding the KF, CNN, quantization concepts.