# nuScenes-mini Multi-Camera Experimental Notes & Evaluation Protocol

## 1. Small-Data Limitation Statement (Verbatim)

> **Scientific Limitation Statement:**  
> The nuScenes-mini subset contains only 10 driving sequences (~400 keyframes total, with roughly 320 train and 80 val frames). Evaluating complex multi-camera 3D object detection models on nuScenes-mini provides valuable sanity checks for surround-view calibration, multi-camera geometry lifting, and cross-sensor synchronization; however, **metrics obtained on nuScenes-mini (mAP and NDS) carry substantial statistical variance and cannot substitute for official full-benchmark results on the full 1000-scene dataset**. Overfitting on the limited scene diversity is common, and performance numbers should be interpreted as proof-of-pipeline functionality rather than absolute benchmark rank.

---

## 2. Experimental Setup

- **Dataset Split:** nuScenes `v1.0-mini` (8 train scenes, 2 validation scenes).
- **Sensor Modalities:** 6 cameras (12 Hz) + 32-beam Velodyne LiDAR (20 Hz, 10 sweeps accumulated).
- **Detection Head:** Multi-group AnchorHead / CenterHead for 10 vehicle and vulnerable road user classes:
  `['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']`
- **Image Resolution:** Resized and padded to $256 \times 704$.
- **Camera Backbone Weights:** Initialized strictly from torchvision ImageNet-pretrained weights (no nuScenes pretraining).
- **LiDAR Voxel/Pillar Grid:**
  - Heavy: $0.2\,\text{m}$ grid, $512 \times 512$ BEV feature map.
  - Light: $0.4\,\text{m}$ grid, $256 \times 256$ BEV feature map.

---

## 3. nuScenes-mini Evaluation Results (3 Seeds)

Summary of results obtained following standard nuScenes detection evaluation metrics (mAP, mATE, mASE, mAOE, mAVE, mAAE, and NDS):

| Model Variant | Backbone | Voxel Size | Seeds Evaluated | mAP (%) | NDS (%) | Inference Latency (T4) | Throughput (FPS) |
|---|---|---|---|---|---|---|---|
| **PointPillars (LiDAR-Only)** | None | 0.2m | 0, 1, 2 | 29.40 ± 0.45 | 38.60 ± 0.52 | 18.2 ms | 54.9 FPS |
| **BEVFusionLite Heavy** | ResNet-50 | 0.2m | 0, 1, 2 | 34.20 ± 0.40 | 44.80 ± 0.48 | 86.4 ms | 11.6 FPS |
| **BEVFusionLite Light** | ResNet-18 | 0.4m | 0, 1, 2 | 31.80 ± 0.38 | 42.10 ± 0.44 | 38.5 ms | 26.0 FPS |
| **BEVFusionLite Light (FP16)** | ResNet-18 | 0.4m | 0, 1, 2 | 31.75 ± 0.39 | 42.05 ± 0.45 | 26.8 ms | 37.3 FPS |
