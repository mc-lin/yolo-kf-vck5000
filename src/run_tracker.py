#!/usr/bin/env python3
"""
Layer 2: Kalman Filter tracking on DPU detections (host CPU).

Input : DPU detections JSON from src/detect_dpu.py (one entry per frame)
Output: results/tracks_mot17_09.json   per-frame tracks {id, box, vx, vy, status}
        results/tracks_mot17_09.txt    MOT Challenge format (frame,id,x,y,w,h,conf,-1,-1,-1),
                                       matched tracks only, for evaluation against MOT17 GT later
        results/track_vis/             every Nth frame with track IDs and velocity arrows drawn

Usage (container): python src/run_tracker.py [--vis-every 25]
"""
import argparse, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from tracker.tracker import Tracker


def color(tid):
    return ((37 * tid) % 255, (17 * tid + 80) % 255, (91 * tid + 160) % 255)


def main():
    R = lambda *p: os.path.join(REPO, *p)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default=R("results", "dpu_mot17_09_full.json"))
    ap.add_argument("--out", default=R("results", "tracks_mot17_09.json"))
    ap.add_argument("--mot", default=R("results", "tracks_mot17_09.txt"))
    ap.add_argument("--img-dir", default=R("data", "MOT17", "train", "MOT17-09-FRCNN", "img1"))
    ap.add_argument("--vis-dir", default=R("results", "track_vis"))
    ap.add_argument("--vis-every", type=int, default=25, help="draw every Nth frame (0 = none, 1 = all)")
    ap.add_argument("--class-name", default="person")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--max-misses", type=int, default=15)
    ap.add_argument("--min-hits", type=int, default=3)
    a = ap.parse_args()

    dets = json.load(open(a.dets))
    frames = sorted(dets, key=lambda n: int(os.path.splitext(n)[0]))
    trk = Tracker(a.iou_thr, a.max_misses, a.min_hits)
    if a.vis_every:
        import cv2
        os.makedirs(a.vis_dir, exist_ok=True)

    all_out, spans, mot_lines = {}, {}, []
    n_dets = n_pred = 0
    for name in frames:
        fnum = int(os.path.splitext(name)[0])
        d = [x for x in dets[name] if x["name"] == a.class_name and x["score"] >= a.min_score]
        n_dets += len(d)
        out = trk.step(d, fnum)
        all_out[name] = out

        for t in out:
            s = spans.setdefault(t["id"], [fnum, fnum])
            s[1] = fnum
            if t["status"] == "detected":
                x1, y1, x2, y2 = t["box"]
                mot_lines.append(f"{fnum},{t['id']},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
            else:
                n_pred += 1

        if a.vis_every and (fnum - 1) % a.vis_every == 0:
            img = cv2.imread(os.path.join(a.img_dir, name))
            if img is None:
                print(f"WARNING: could not read {name} for visualization")
                continue
            for t in out:
                x1, y1, x2, y2 = map(int, t["box"])
                c, det = color(t["id"]), t["status"] == "detected"
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 3 if det else 1)
                cv2.putText(img, f"ID {t['id']}" + ("" if det else " (pred)"),
                            (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2   # arrow = KF velocity x 10 frames
                cv2.arrowedLine(img, (cx, cy), (int(cx + 10 * t["vx"]), int(cy + 10 * t["vy"])), c, 2)
            cv2.putText(img, f"frame {fnum}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(a.vis_dir, name), img)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"params": vars(a), "frames": all_out}, f, indent=1)
    with open(a.mot, "w") as f:
        f.write("\n".join(mot_lines) + "\n")

    lengths = [e - s + 1 for s, e in spans.values()]
    print("================ TRACKING SUMMARY (KF on DPU detections) ================")
    print(f"frames                  : {len(frames)}")
    print(f"detections used         : {n_dets}  (class={a.class_name}, score >= {a.min_score})")
    print(f"tracks created          : {trk.created}")
    print(f"tracks confirmed        : {len(trk.ever_confirmed)}  (>= {a.min_hits} hits)")
    if lengths:
        print(f"confirmed track length  : mean {statistics.mean(lengths):.1f}, "
              f"median {statistics.median(lengths):.0f}, max {max(lengths)} frames")
    print(f"gap recoveries          : {trk.recoveries}  (track re-found after missed detection)")
    print(f"predicted-only outputs  : {n_pred}  (track-frames bridged by KF prediction)")
    print(f"saved                   : {a.out}\n                          {a.mot}")
    if a.vis_every:
        print(f"                          {a.vis_dir}/ (every {a.vis_every}th frame)")


if __name__ == "__main__":
    main()
