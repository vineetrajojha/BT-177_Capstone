# BEVFusionLite Status Audit

**Audit Timestamp:** 2026-11-17  
**Auditor:** Antigravity Autonomous Agent  
**Environment:** macOS (Darwin arm64), Python 3.14.0, PyTorch 2.15.0, Torchvision 0.29.2  
**OpenPCDet Commit:** `233f849829b6ac19afb8af8837a0246890908755`  
**Available Local Disk:** 14 GiB  

---

## 1. Prompt Status Matrix (P0 – P12 + Website)

| Phase | Objective | Status | Evidence & Test Output | Deviations & Action Items |
|---|---|---|---|---|
| **P0** | **Rules File & Guidelines** | **DONE** | [`AGENTS.md`](file:///Users/vineetcres/Desktop/Github/BT-177_Capstone/AGENTS.md) | Agent constraints pinned at root. Golden path adherence. |
| **P1** | **Environment & Dependencies** | **DONE** | `pytest tests/test_env.py` (6/6 passed); `python scripts/check_env.py` passed with `ALL ENVIRONMENT AND MODEL CHECKS PASSED`. | OpenPCDet patched for CPU/CUDA agility in `anchor_generator.py` and `loss_utils.py`. |
| **P2** | **KITTI Pipeline & Calibration** | **PARTIAL** | `pytest tests/test_kitti_pipeline.py` (9/9 passed); `scripts/calibration_check.py` evaluated 20 seeded frames (78 objects, mean fraction = 0.9182 >= 0.90, median offset = 0.41 px); 20 overlay & boxcorner PNGs in `results/calibration/`. | **Data subset:** 100 real KITTI frames downloaded (`000000`–`000099`). Full 7,481 dataset check requires remote Kaggle/Colab run due to 14 GiB local disk limit. |
| **P3** | **PointPillars Reference Baseline** | **NOT STARTED** | Needs `configs/lidar_only_pointpillars.yaml`, remote Kaggle/Colab training scripts for seeds 0, 1, 2, and `results/pointpillars_kitti_val.csv`. | Target: Car 3D moderate AP R40 >= 74.0 (published ~77.0). Must run on Kaggle/Colab. |
| **P4** | **Literature Review & Novelty** | **NOT STARTED** | Needs `reports/literature_review.md`, `reports/novelty_statement.md`, `reports/references.bib`. | 25–40 verified citations with DOI/arXiv IDs. |
| **P5** | **Benchmark Tool** | **NOT STARTED** | Needs `tools/benchmark.py` measuring latency, FPS, GFLOPs, peak memory, and parameter counts. | Strict T4 benchmarking protocol (50 warmup, 300 timed runs, synchronized). |
| **P6** | **Model Code & Heavy Baseline** | **PARTIAL** | Modules exist in `bevfusionlite/models/`, but require updating to spec. | **Deviation to fix:** Current `depth_net.py` / `lss_transform.py` uses unguided LSS (88 bins, RGB-only). Spec requires **LiDAR-depth-guided image-to-BEV transform** (sparse depth map min-pooled to feature stride, depth head with depth+mask channels, D=48 bins over 1 to 69m, aux depth loss weight 0.1). Also needs ResNet-50 / ResNet-18 unified model class, ablation modes, and tests. |
| **P7** | **Light Model** | **NOT STARTED** | Needs `docs/ARCHITECTURE_LIGHT.md`, configs, benchmarks, and `results/main_comparison.csv`. | Target: >= 50% fewer params/FLOPs, >= 40% memory reduction, >= 2x FPS, Car 3D within 3 AP of heavy. |
| **P8** | **Ablations (3 Seeds Each)** | **NOT STARTED** | Needs `results/ablation_table.csv`, LaTeX table, AP-vs-latency plots. | 6 ablation axes: image backbone, voxel size, fusion type, depth guidance, depth bins, precision. |
| **P9** | **nuScenes-mini Multi-Camera** | **NOT STARTED** | Needs `bevfusionlite/datasets/nuscenes_fusion.py` (6 cameras, 10 sweeps), configs, and `reports/nuscenes_notes.md`. | Secondary benchmark evaluation. |
| **P10** | **KITTI Test Server Package** | **NOT STARTED** | Needs `scripts/check_submission.py`, `results/kitti_test_submission.zip`, instructions. | Median-seed submission formatting for 7,518 test samples. |
| **P11** | **Streamlit / Rerun Demo** | **NOT STARTED** | Needs `demo/`, `scripts/precompute_demo.py` caching predictions for 50 fixed frames. | Offline / internal fallback interactive visualizer. |
| **P12** | **Final Report & Audit** | **NOT STARTED** | Needs `reports/final_report.md`, `scripts/make_report_tables.py`, `docs/REPRODUCE.md`, `scripts/audit.py`. | Fully automated report generation with zero hardcoded values. |
| **Website** | **Public Hosted Website** | **NOT STARTED** | Needs Next.js + TypeScript + Tailwind in `web/`, Playwright smoke tests, `scripts/export_web_data.py`, Vercel deploy. | 12 interactive pages/views, dark research-grade UI, zero hand-typed numbers. |

---

## 2. Dataset Audit (`data/kitti/`)

- **Current local data:**
  - `data/kitti/training/calib`: 100 files (`000000.txt` to `000099.txt`)
  - `data/kitti/training/image_2`: 100 images
  - `data/kitti/training/velodyne`: 100 `.bin` point clouds
  - `data/kitti/training/label_2`: 100 label files
  - `data/kitti/testing/calib`: 2 files
- **Required official split:**
  - 7,481 training/validation frames (3,712 train, 3,769 val)
  - 7,518 testing frames
- **Storage constraint:**
  - Local disk has only **14 GiB** available.
  - The full KITTI dataset requires ~35 GiB archive download and ~80 GiB uncompressed space.
  - **Verdict:** Per rule 4 of `antigravity_master_prompt.md` and rule 4 of `AGENTS.md`, full dataset preparation and 80-epoch 3-seed training must be offloaded to Kaggle / Google Colab cloud runners. Ready-to-run automation scripts and notebooks will be provided.

---

## 3. Specification Deviations & Architecture Gap Analysis

1. **Depth Guidance in LSS:**
   - *Current Implementation:* Vanilla LSS predicts categorical depth directly from 2D features without LiDAR depth guidance.
   - *Master Prompt Specification:* Must project LiDAR points to camera image plane to form a sparse depth map, min-pool it to stride 8 feature resolution, concatenate depth + mask into the depth predictor head, use $D=48$ bins over $[1.0, 69.0]\,\text{m}$, and compute an auxiliary depth cross-entropy loss with weight $0.1$.
2. **Unified Model Class:**
   - *Current Implementation:* `CameraBackbone` only supports ResNet-18.
   - *Master Prompt Specification:* Must support both ResNet-50 (Heavy baseline: 128 image channels, 64 LiDAR channels) and ResNet-18 (Light model: 64 image channels, 32 LiDAR channels), plus MobileNetV3-Large for ablations, configurable purely via YAML.
3. **Fusion Ablation Modes:**
   - *Current Implementation:* Simple `ConvFuser` only.
   - *Master Prompt Specification:* Fuser must support `concat_2conv`, `concat_1x1`, `channel_attention`, as well as `camera_only` and `lidar_only` ablation modes.
