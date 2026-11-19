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