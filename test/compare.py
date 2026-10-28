#!/usr/bin/env python3
"""
Testbench: compare DPU (INT8) detections against the golden FP32 float detections.

Both inputs use the JSON format written by src/detect_float.py and src/detect_dpu.py:
  { "image.jpg": [ {"class": int, "name": str, "score": float, "box": [x1,y1,x2,y2]}, ... ] }

Reports:
  - per-image box matching (same class, IoU >= --iou-thr): IoU, score change,
    golden boxes the DPU missed, and extra DPU boxes
  - mean IoU of matched pairs, precision/recall of DPU vs golden
  - mAP of the DPU using the float detections as reference (pseudo ground truth):
    mAP@0.5 and mAP@[0.5:0.95] (COCO-style average over IoU thresholds)
  - PASS/FAIL against --min-iou and --min-recall

Usage:
  python test/compare.py [--golden golden/float_detections.json]
                         [--test results/dpu_detections.json]
"""
import argparse, json, os
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(golden, test, thr):
    """Greedy matching: highest-confidence test box first, to the best unused
    golden box of the same class with IoU >= thr."""
    used, pairs, extra = set(), [], []
    for ti in sorted(range(len(test)), key=lambda i: -test[i]["score"]):
        t, best, best_iou = test[ti], None, thr
        for gi, g in enumerate(golden):
            if gi in used or g["class"] != t["class"]:
                continue
            v = iou(g["box"], t["box"])
            if v >= best_iou:
                best, best_iou = gi, v
        if best is None:
            extra.append(t)
        else:
            used.add(best)
            pairs.append((golden[best], t, best_iou))
    missed = [g for gi, g in enumerate(golden) if gi not in used]
    return pairs, missed, extra


def average_precision(golden_all, test_all, cls, thr):
    """VOC-style all-point-interpolated AP for one class at one IoU threshold."""
    gts = {img: [g for g in dets if g["class"] == cls] for img, dets in golden_all.items()}
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return None
    preds = [(img, t) for img, dets in test_all.items() for t in dets if t["class"] == cls]
    preds.sort(key=lambda p: -p[1]["score"])
    used = {img: [False] * len(v) for img, v in gts.items()}
    tp, fp = [], []
    for img, t in preds:
        best, best_iou = None, thr
        for gi, g in enumerate(gts.get(img, [])):
            v = iou(g["box"], t["box"])
            if v >= best_iou and not used[img][gi]:
                best, best_iou = gi, v
        if best is None:
            tp.append(0); fp.append(1)
        else:
            used[img][best] = True
            tp.append(1); fp.append(0)
    rec, prec, ctp, cfp = [], [], 0, 0
    for a, b in zip(tp, fp):
        ctp += a; cfp += b
        rec.append(ctp / npos); prec.append(ctp / (ctp + cfp))
    mrec, mpre = [0.0] + rec + [1.0], [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))


def mean_ap(golden_all, test_all, thr):
    classes = {g["class"] for dets in golden_all.values() for g in dets}
    aps = [ap for c in classes if (ap := average_precision(golden_all, test_all, c, thr)) is not None]
    return sum(aps) / len(aps) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(REPO, "golden", "float_detections.json"))
    ap.add_argument("--test", default=os.path.join(REPO, "results", "dpu_detections.json"))
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU needed to count as a match")
    ap.add_argument("--min-iou", type=float, default=0.75, help="PASS needs mean matched IoU >= this")
    ap.add_argument("--min-recall", type=float, default=0.75, help="PASS needs recall >= this")
    ap.add_argument("--report", default=os.path.join(REPO, "results", "compare_report.json"))
    ap.add_argument("--quiet", action="store_true", help="hide per-image details")
    a = ap.parse_args()

    golden = json.load(open(a.golden))
    test = json.load(open(a.test))
    images = sorted(set(golden) & set(test))
    skipped = sorted(set(golden) ^ set(test))
    if skipped:
        print(f"WARNING: images in only one file (skipped): {skipped}")
    golden = {k: golden[k] for k in images}
    test = {k: test[k] for k in images}

    all_ious, all_dscore, n_gold, n_test, n_match = [], [], 0, 0, 0
    per_image = {}
    for img in images:
        pairs, missed, extra = match_image(golden[img], test[img], a.iou_thr)
        n_gold += len(golden[img]); n_test += len(test[img]); n_match += len(pairs)
        all_ious += [v for _, _, v in pairs]
        all_dscore += [t["score"] - g["score"] for g, t, _ in pairs]
        per_image[img] = {"matched": len(pairs), "missed": len(missed), "extra": len(extra),
                          "ious": [round(v, 4) for _, _, v in pairs]}
        if not a.quiet:
            print(f"\n== {img}: golden {len(golden[img])}, DPU {len(test[img])}, matched {len(pairs)}")
            for g, t, v in pairs:
                print(f"   match  {g['name']:<12} IoU {v:.3f}  score float {g['score']:.3f} "
                      f"-> DPU {t['score']:.3f}  ({t['score'] - g['score']:+.3f})")
            for g in missed:
                print(f"   MISSED {g['name']:<12} float score {g['score']:.3f}  box {[round(x,1) for x in g['box']]}")
            for t in extra:
                print(f"   EXTRA  {t['name']:<12} DPU score {t['score']:.3f}  box {[round(x,1) for x in t['box']]}")

    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    recall = n_match / n_gold if n_gold else 0.0
    precision = n_match / n_test if n_test else 0.0
    mean_dscore = sum(all_dscore) / len(all_dscore) if all_dscore else 0.0
    map50 = mean_ap(golden, test, 0.5)
    map5095 = sum(mean_ap(golden, test, t / 100) for t in range(50, 100, 5)) / 10
    passed = mean_iou >= a.min_iou and recall >= a.min_recall

    print("\n================ SUMMARY (DPU INT8 vs float FP32 golden) ================")
    print(f"images                 : {len(images)}")
    print(f"boxes golden / DPU     : {n_gold} / {n_test}   matched: {n_match} (IoU >= {a.iou_thr})")
    print(f"mean IoU (matched)     : {mean_iou:.4f}")
    print(f"recall / precision     : {recall:.4f} / {precision:.4f}")
    print(f"mean score change      : {mean_dscore:+.4f}  (DPU - float, matched boxes)")
    print(f"mAP@0.5   vs float     : {map50:.4f}")
    print(f"mAP@.5:.95 vs float    : {map5095:.4f}")
    print(f"RESULT                 : {'PASS' if passed else 'FAIL'} "
          f"(needs mean IoU >= {a.min_iou} and recall >= {a.min_recall})")

    report = {"images": len(images), "golden_boxes": n_gold, "dpu_boxes": n_test,
              "matched": n_match, "mean_iou": mean_iou, "recall": recall,
              "precision": precision, "mean_score_change": mean_dscore,
              "map50_vs_float": map50, "map50_95_vs_float": map5095,
              "pass": passed, "thresholds": {"iou_thr": a.iou_thr, "min_iou": a.min_iou,
                                             "min_recall": a.min_recall},
              "per_image": per_image}
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(report, open(a.report, "w"), indent=2)
    print(f"report saved           : {a.report}")


if __name__ == "__main__":
    main()
