#!/usr/bin/env python3
"""
Golden FP32 reference: AMD Model Zoo YOLOX-nano (float) on CPU.

Reuses AMD's tools/demo.py Predictor so preprocessing (BGR, letterbox to 416,
pad 114) and postprocessing (decode + NMS) are identical to AMD's float pipeline.
Thresholds default to the DPU model's prototxt (conf 0.3, NMS 0.65).

Usage (container, conda env vitis-ai-pytorch):
  python src/detect_float.py data/test_images/*.jpg --out golden/float_detections.json
"""
import argparse, json, os, sys, time
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG  = os.path.join(REPO, "model", "downloads", "pt_yolox-nano_coco_416_416_1G_3.0")
CODE = os.path.join(PKG, "code")
sys.path.insert(0, CODE)

from yolox.exp import get_exp
from yolox.data.datasets import COCO_CLASSES
from tools.demo import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default=os.path.join(REPO, "golden", "float_detections.json"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()

    exp = get_exp(os.path.join(CODE, "exps/example/custom/yolox_nano_deploy_relu.py"), None)
    exp.test_conf, exp.nmsthre, exp.test_size = a.conf, a.nms, (416, 416)

    model = exp.get_model().eval()
    ckpt = torch.load(os.path.join(PKG, "float", "yolox_nano.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    pred = Predictor(model, exp, COCO_CLASSES, None, None, "cpu", False, False)

    # Warm-up so one-time costs don't distort the timing
    for _ in range(a.warmup):
        pred.inference(a.images[0])

    results, times = {}, []
    for path in a.images:
        t0 = time.perf_counter()
        outputs, info = pred.inference(path)
        times.append(time.perf_counter() - t0)

        dets, r = [], info["ratio"]
        if outputs[0] is not None:
            for x1, y1, x2, y2, obj, cls_conf, cls in outputs[0].cpu().numpy():
                dets.append({
                    "class": int(cls),
                    "name": COCO_CLASSES[int(cls)],
                    "score": float(obj * cls_conf),
                    "box": [float(x1 / r), float(y1 / r), float(x2 / r), float(y2 / r)],
                })
        results[os.path.basename(path)] = dets
        for d in dets:
            print(f"{os.path.basename(path)}  {d['class']:3d} {d['name']:<12} "
                  f"{d['score']:.3f}  " + " ".join(f"{v:7.2f}" for v in d["box"]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)

    avg_ms = 1000 * sum(times) / len(times)
    print(f"\nSaved {a.out}")
    print(f"CPU FP32: {len(times)} images, avg {avg_ms:.2f} ms/image "
          f"(end-to-end incl. pre/post, after {a.warmup} warm-up runs)")


if __name__ == "__main__":
    main()
