"""
SORT-style multi-object tracker: one KalmanBoxFilter per track,
greedy IoU association between predicted track boxes and new detections.
"""
from .kalman import KalmanBoxFilter


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box, score, frame):
        self.id = tid
        self.kf = KalmanBoxFilter(box)
        self.hits = 1           # frames with a matched detection
        self.misses = 0         # consecutive frames without one
        self.start = frame
        self.last_score = score
        self.confirmed = False


class Tracker:
    def __init__(self, iou_thr=0.3, max_misses=15, min_hits=3):
        self.iou_thr = iou_thr          # min IoU(prediction, detection) to match
        self.max_misses = max_misses    # delete a track after this many missed frames
        self.min_hits = min_hits        # report a track only after this many hits (filters one-off false boxes)
        self.tracks, self.next_id = [], 1
        self.created, self.recoveries = 0, 0
        self.ever_confirmed = set()

    def step(self, dets, frame):
        # 1. PREDICT every track forward one frame
        preds = [t.kf.predict() for t in self.tracks]

        # 2. MATCH: greedy, highest IoU pair first, each track/detection used at most once
        cand = [(iou(pb, d["box"]), ti, di)
                for ti, pb in enumerate(preds) for di, d in enumerate(dets)]
        cand = sorted((c for c in cand if c[0] >= self.iou_thr), reverse=True)
        mt, md = set(), set()
        for _, ti, di in cand:
            if ti in mt or di in md:
                continue
            mt.add(ti)
            md.add(di)
            t = self.tracks[ti]
            if t.misses > 0:
                self.recoveries += 1        # track survived a gap and was re-found
            t.kf.update(dets[di]["box"])    # 3a. UPDATE: correct prediction with detection
            t.hits += 1
            t.misses = 0
            t.last_score = dets[di]["score"]

        # 3b. unmatched tracks coast on their prediction (missed detection / occlusion)
        for ti, t in enumerate(self.tracks):
            if ti not in mt:
                t.misses += 1

        # 3c. unmatched detections start new tracks
        for di, d in enumerate(dets):
            if di not in md:
                self.tracks.append(Track(self.next_id, d["box"], d["score"], frame))
                self.next_id += 1
                self.created += 1

        # 4. DELETE tracks lost for too long
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # Output confirmed tracks: "detected" (matched this frame) or "predicted" (coasting)
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True
                self.ever_confirmed.add(t.id)
            if t.confirmed:
                vx, vy = t.kf.velocity()
                out.append({
                    "id": t.id,
                    "box": [round(v, 2) for v in t.kf.box()],
                    "vx": round(vx, 3), "vy": round(vy, 3),
                    "status": "detected" if t.misses == 0 else "predicted",
                    "misses": t.misses,
                })
        return out
