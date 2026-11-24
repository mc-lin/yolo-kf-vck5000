#!/usr/bin/env python3
"""
Evaluate tracker output against MOT17 ground truth (CLEAR MOT + identity metrics).

Ground truth: only pedestrians that count for evaluation (flag == 1, class == 1).
Matching: a track box matches a GT box when IoU >= --iou (default 0.5).

Note: simplified evaluation. The official MOTChallenge tool also ignores tracker boxes
that match GT "distractor" classes (e.g. static persons, reflections); this script
does not, so FP may be slightly higher than the official numbers.

Usage (container):
  python test/eval_tracking.py [--gt data/MOT17/train/MOT17-09-FRCNN/gt/gt.txt]
                               [--pred results/tracks_mot17_09.txt]
"""
import argparse, collections, os
import motmetrics as mm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(path, is_gt):
    """Return {frame: [(id, [x, y, w, h]), ...]} from a MOT-format text file."""
    per_frame = collections.defaultdict(list)
    for line in open(path):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        frame, tid = int(float(p[0])), int(float(p[1]))
        if is_gt and (int(float(p[6])) != 1 or int(float(p[7])) != 1):
            continue                        # keep only evaluated pedestrians
        per_frame[frame].append((tid, [float(v) for v in p[2:6]]))
    return per_frame


def main():
    R = lambda *p: os.path.join(REPO, *p)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=R("data", "MOT17", "train", "MOT17-09-FRCNN", "gt", "gt.txt"))
    ap.add_argument("--pred", default=R("results", "tracks_mot17_09.txt"))
    ap.add_argument("--iou", type=float, default=0.5, help="min IoU for a match")
    ap.add_argument("--out", default=R("results", "track_eval.txt"))
    a = ap.parse_args()

    gt, pred = load(a.gt, True), load(a.pred, False)
    acc = mm.MOTAccumulator(auto_id=False)
    last = max(list(gt) + list(pred))
    for f in range(1, last + 1):
        g, h = gt.get(f, []), pred.get(f, [])
        # distance = 1 - IoU; pairs with IoU below the threshold cannot match
        d = mm.distances.iou_matrix([b for _, b in g], [b for _, b in h], max_iou=1 - a.iou)
        acc.update([i for i, _ in g], [i for i, _ in h], d, frameid=f)

    mh = mm.metrics.create()
    metrics = ["num_frames", "num_unique_objects", "num_objects", "mota", "idf1",
               "num_switches", "num_false_positives", "num_misses", "precision", "recall",
               "mostly_tracked", "partially_tracked", "mostly_lost", "num_fragmentations", "motp"]
    summary = mh.compute(acc, metrics=metrics, name="MOT17-09")
    text = mm.io.render_summary(summary, formatters=mh.formatters,
                                namemap=mm.io.motchallenge_metric_names)
    print(text)
    print("(MOTP here = mean (1 - IoU) of matched boxes: lower is better)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(text + "\n")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
