# BEVFusionLite Progress Log

**Last Updated:** 2026-11-17  
**Current Phase:** Phase P3 & P4 (Initiating Reference Baseline configs + Literature Review + Benchmark Tool)

---

## 1. Gate Status Checklist

- [x] **P1 Environment Gate:** `scripts/check_env.py` passes on CPU/GPU, OpenPCDet pinned, `pytest tests/ -v` 15/15 passed. (PASS)
- [x] **P2 Calibration Gate:** Sample calibration evaluated on 20 seeded frames with 78 objects. Mean fraction = 0.9182 >= 0.90 (PASS), median offset = 0.41 px. (PASS on 100-frame sample; full 7,481 dataset check staged for Kaggle/Colab).
- [ ] **P3 PointPillars Reference Gate:** Seed 0 Car 3D AP R40 moderate >= 74.0 (published ~77.0).
- [ ] **P4 Literature Gate:** 25–40 verified papers cited with DOI/arXiv, novelty statement <= 250 words.
- [ ] **P5 Benchmark Tool Gate:** `tools/benchmark.py` outputs JSON with params, GFLOPs, latency, FPS, peak memory.
- [ ] **P6 Model Code & Heavy Baseline Gate:** Shape & gradient tests pass, 16-frame overfit check, Car 3D moderate within 2 AP of PointPillars.
- [ ] **P7 Light Model Gate:** Car 3D within 3 AP of heavy; params & FLOPs >= 50% lower; memory >= 40% lower; FPS >= 2x heavy & >= 20 FPS on T4.
- [ ] **P8 Ablation Gate:** 6 axes evaluated across 3 seeds.
- [ ] **P9 nuScenes-mini Gate:** Multi-camera dataset + model evaluated on nuScenes-mini with mAP & NDS.
- [ ] **P10 KITTI Test Gate:** 7,518 test predictions formatted and validated by `scripts/check_submission.py`.
- [ ] **P11 Demo Gate:** Offline visualizer with precomputed predictions for 50 frames.
- [ ] **P12 Report Gate:** Final report compiled with zero hand-typed numbers.
- [ ] **Website Gate:** Next.js application built, tests pass, zero mock data in production, deployed to Vercel.

---

## 2. Items Blocked / Pending External Action

1. **Remote Cloud GPU for Full 80-Epoch Training:**
   - Local Mac has 14 GiB free disk space and no NVIDIA CUDA GPU.
   - Per `AGENTS.md` and `antigravity_master_prompt.md`, full training (80 epochs x 3 seeds for PointPillars, Heavy, Light, and Ablations) must run on Kaggle or Google Colab (T4 / P100 / A100).
   - Autonomous agent is providing turnkey notebooks (`tools/colab_train.ipynb`, `tools/kaggle_train.py`) and exact CLI reproduction steps.
