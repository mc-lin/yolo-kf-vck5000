#!/usr/bin/env python3
"""
DPU INT8 inference: YOLOX-nano on VCK5000 (DPUCVDX8H 4PE).

Runs AMD's Vitis-AI Library yolovx sample (test_jpeg_yolovx), which applies AMD's
own preprocessing and postprocessing (thresholds from yolox_nano_pt.prototxt:
conf 0.3, NMS 0.65), and parses its RESULT lines into the SAME JSON format as
src/detect_float.py, so test/compare.py can compare them directly.

Prereq (inside container, every session):
  source /workspace/board_setup/vck5000/setup.sh DPUCVDX8H_4pe_miscdwc
Usage:
  python src/detect_dpu.py data/test_images/*.jpg --out results/dpu_detections.json
"""
import argparse, json, os, re, subprocess, sys

REPO      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(REPO, "model", "downloads", "yolox_nano_pt")
DEMO      = "/workspace/examples/vai_library/samples/yolovx/test_jpeg_yolovx"

try:  # class names are optional; compare.py only uses the class id
    sys.path.insert(0, os.path.join(REPO, "model", "downloads",
                                    "pt_yolox-nano_coco_416_416_1G_3.0", "code"))
    from yolox.data.datasets import COCO_CLASSES
except Exception:
    COCO_CLASSES = None

IMG_RE = re.compile(r"image:\s*(\S+)")
RES_RE = re.compile(r"RESULT:\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "dpu_detections.json"))
    a = ap.parse_args()

    images = [os.path.abspath(p) for p in a.images]
    names = [os.path.basename(p) for p in images]
    if len(set(names)) != len(names):
        sys.exit("Image file names must be unique (results are keyed by file name).")

    # The demo looks up the model by name in its working directory, and writes
    # annotated images there, so run it in results/dpu_vis with a link to the model.
    workdir = os.path.join(REPO, "results", "dpu_vis")
    os.makedirs(workdir, exist_ok=True)
    link = os.path.join(workdir, "yolox_nano_pt")
    if not os.path.exists(link):
        os.symlink(MODEL_DIR, link)

    proc = subprocess.run([DEMO, "yolox_nano_pt", *images], cwd=workdir,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = proc.stdout
    with open(os.path.join(REPO, "results", "dpu_raw_log.txt"), "w") as f:
        f.write(log)
    if proc.returncode != 0:
        print(log[-3000:])
        sys.exit(f"test_jpeg_yolovx failed (exit {proc.returncode}). Did you source setup.sh?")

    # The DPU processes batches of 4; with fewer images, the demo repeats images to
    # fill the batch. Keep only the FIRST block of results for each image name.
    results, current, collecting = {}, None, False
    for line in log.splitlines():
        m = IMG_RE.search(line)
        if m:
            current = m.group(1)
            collecting = current not in results
            if collecting:
                results[current] = []
            continue
        m = RES_RE.search(line)
        if m and current and collecting:
            cls = int(m.group(1))
            x1, y1, x2, y2, score = map(float, m.groups()[1:])
            results[current].append({
                "class": cls,
                "name": COCO_CLASSES[cls] if COCO_CLASSES else str(cls),
                "score": score,
                "box": [x1, y1, x2, y2],
            })

    missing = [n for n in names if n not in results]
    if missing:
        print(f"WARNING: no output found for {missing}")
    results = {n: results.get(n, []) for n in names}

    for n, dets in results.items():
        for d in dets:
            print(f"{n}  {d['class']:3d} {d['name']:<12} {d['score']:.3f}  "
                  + " ".join(f"{v:7.2f}" for v in d["box"]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {a.out}  (raw log: results/dpu_raw_log.txt, images: results/dpu_vis/)")


if __name__ == "__main__":
    main()
