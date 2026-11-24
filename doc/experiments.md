## 2026-11-19 — Own photo (tasty.jpeg), DPU vs FP32

**Commands**
- `python src/detect_float.py data/test_images/tasty.jpeg --out golden/float_tasty.json`
- `python src/detect_dpu.py data/test_images/tasty.jpeg --out results/dpu_tasty.json`
- `python test/compare.py --golden golden/float_tasty.json --test results/dpu_tasty.json --report results/compare_report_tasty.json`

**Result:** recall 0.50, precision 0.20, FAIL. Sandwich matched with IoU 0.975 (0.614 → 0.590).
FP32 found a person (0.349) that the DPU missed; the DPU found 2 bottles + 2 cups that FP32 did not report.

**Interpretation:** quantization shifts scores up *and* down; objects near the 0.3 threshold
flip in or out. Localization of the agreed object is near-identical.

**Open questions / next checks**
- [ ] Rerun FP32 with `--conf 0.1`: does FP32 see the bottles/cups just below 0.3?
- [ ] Check the photo: which objects are really there? (correctness vs precision)
- [ ] Are the overlapping cup/bottle pairs real duplicates that NMS (0.65) let through?

Performance: DPU end-to-end with test_performance_yolovx, and CPU FP32 timing on the same 40 frames. Write the numbers into docs/experiments.md
With 4 threads: 1371 FPS. Four streams overlap the CPU-side work with the DPU work, so the DPU sits idle less. That run printed only FPS, which is fine, since throughput is its purpose.
| Metric | CPU FP32 | DPU INT8 | Speedup |
| --- | --- | --- | --- |
| Latency per image | 34.03 ms | 6.81 ms | ≈ 5.0× |
| Throughput, 1 thread | 29.4 FPS | 587 FPS | ≈ 20× |
| Throughput, 4 threads | 29.4 FPS | 1371 FPS | ≈ 47× |
Latency improved about 5×, much less than the raw inference speedup, because each image waits for its batch and for the CPU-side steps. Throughput improved 20–47×, because batching and parallel streams keep the DPU busy. Both numbers are real; they measure different things, and the report should give both.
the CPU was measured one image at a time, without batching. A batched CPU run might improve its throughput somewhat. That's worth a sentence, or a quick extra experiment if you have time.

## Tracking vs MOT17-09 ground truth (baseline, Update 1 tracker settings)

Settings: iou_thr 0.3, max_misses 15, min_hits 3, greedy matching; output = matched boxes only.
Eval: test/eval_tracking.py (motmetrics 1.4.0), GT flag==1 & class==1, IoU ≥ 0.5.

| MOTA | IDF1 | IDSW | FP | FN | Prcn | Rcll | MT/PT/ML | FM | MOTP (1−IoU) |
|---|---|---|---|---|---|---|---|---|---|
| 40.2% | 38.7% | 265 | 770 | 2150 | 80.5% | 59.6% | 8/14/4 | 185 | 0.234 |

FN dominates MOTA (40% of GT boxes) → mostly detector misses. IDF1 < MOTA and ~10 IDSW per person
→ identity fragmentation in the tracker.