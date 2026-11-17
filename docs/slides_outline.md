# BEVFusionLite: Presentation Slides Outline (12 Slides)

---

### Slide 1: Title & Executive Summary
- **Title:** BEVFusionLite: Real-Time Camera-LiDAR Bird's-Eye-View 3D Object Detection on Edge Hardware
- **Author:** Vineet Raj Ojha | Capstone Project BT-177
- **Core Pitch:** Enabling high-fidelity multi-modal 3D perception at edge latency budgets through LiDAR-depth-guided frustum lifting and symmetric pillarization.
- **Headline Numbers:** -55.9% parameters, -62.8% FLOPs, 31.2 FPS on NVIDIA T4, +8.20 AP gain on Cyclist detection.

---

### Slide 2: Motivation & The Edge Autonomous Driving Challenge
- **Sensor Complementarity:** LiDAR provides sparse metric depth; camera provides dense semantic classification and high-resolution textures.
- **The Efficiency Gap:** Current state-of-the-art fusion models (e.g. MIT BEVFusion, TransFusion) rely on heavy Swin-Transformers and dense 3D sparse convolutions (>250ms latency), making them impossible to run on commercial edge embedded compute (T4 / Orin).
- **Core Question:** Can we design a lightweight, pure-PyTorch BEV fusion architecture that achieves real-time throughput (>20 FPS) with minimal degradation compared to heavy baselines?

---

### Slide 3: System Architecture Overview
- **Visual Diagram:** Dual-stream camera-LiDAR pipeline unifying in Bird's-Eye-View.
- **LiDAR Stream:** 32-channel PillarVFE on 0.32m grid -> PointPillarScatter.
- **Camera Stream:** ResNet-18 + FPN to stride 8 (64 channels).
- **View Transform:** LiDAR-depth-guided LSS with D=48 depth bins.
- **Fusion & Head:** Channel concatenation -> 2x Conv(3x3)-BN-ReLU -> BaseBEVBackbone -> AnchorHeadSingle.

---

### Slide 4: LiDAR-Depth-Guided Frustum Lifting
- **The Monocular Depth Trap:** Vanilla LSS predicts categorical depth purely from 2D appearance, causing depth ambiguity and feature smearing along lines of sight (-3.30 Car AP, -5.80 Ped AP).
- **Our Mechanism:**
  1. Project sparse LiDAR points onto image plane using extrinsic/intrinsic calibration.
  2. Min-pool sparse depth to feature stride 8 (preserving foreground occluder depths).
  3. Concatenate normalized depth + binary validity mask into the depth predictor head.
  4. Auxiliary depth classification CE loss (weight 0.1) on valid LiDAR returns.

---

### Slide 5: Data Integrity & Pad-Only Preprocessing
- **Geometric Invariance:** Strict pad-only bottom/right padding to $384 \times 1248$ with scale = 1.0.
- **Why Anisotropic Resizing Fails:** Arbitrary resizing warps focal lengths ($f_x, f_y$), introducing sub-pixel calibration projection errors.
- **Calibration Verification:** Evaluated across 20 seeded frames (78 objects). Measured mean intersection fraction = 0.9182 (exceeding >= 0.90 target); median projection offset = 0.41 px.

---

### Slide 6: Main Benchmark Results (KITTI Val AP R40)
- **Table / Chart:** Published PointPillars Reference vs Heavy Baseline vs Light Model vs Single-Modality.
- **Car 3D (Moderate):**
  - PointPillars Reference: 77.28 AP
  - Heavy Baseline (ResNet-50 + 0.16m): 78.45 AP
  - Light Model (ResNet-18 + 0.32m): 76.10 AP (Within 2.35 AP of Heavy; Target <= 3.0 AP PASS).
- **Multi-Modal Synergistic Gains:**
  - Pedestrian 3D: 55.10 AP (+7.20 AP over LiDAR-only)
  - Cyclist 3D: 65.40 AP (+8.20 AP over LiDAR-only)

---

### Slide 7: Edge Efficiency & Hardware Profiling (NVIDIA T4)
- **Parameter Reduction:** 33.8 M -> 14.9 M (-55.9%, Target >= 50% PASS).
- **Computational Complexity:** 184.2 GFLOPs -> 68.4 GFLOPs (-62.8%, Target >= 50% PASS).
- **Peak Memory Footprint:** 3,820 MB -> 1,610 MB (-57.9%, Target >= 40% PASS).
- **Throughput & Latency:**
  - Heavy Baseline: 74.1 ms (13.5 FPS)
  - Light FP32: 32.1 ms (31.2 FPS; >2x Heavy & >20 FPS PASS)
  - Light FP16 AMP: 22.3 ms (44.8 FPS; +43.6% speedup with zero AP loss).

---

### Slide 8: Multi-Axis Ablation Study
- **Summary of 6 Investigated Dimensions:**
  1. *Backbone Capacity:* ResNet-18 is the optimal sweet spot (76.10 AP / 31.2 FPS) vs MobileNetV3 (75.30 AP / 39.4 FPS) and ResNet-50 (76.90 AP / 19.0 FPS).
  2. *Voxel Granularity:* 0.32m grid drops spatial cells by 75% while losing only 1.50 Car AP vs 0.16m.
  3. *Depth Bins:* D=48 saturates performance; D=64 delta is within 1 std (+0.15 AP) but adds +5.5ms latency.
  4. *Fusion Block:* Concat + 2 convs matches channel attention within 1 std while avoiding dynamic attention overhead.

---

### Slide 9: Multi-Camera Surround Perception: nuScenes-mini
- **Setup:** 6 surround-view cameras (CAM_FRONT, CAM_FRONT_RIGHT, etc.) + 10-sweep 32-beam LiDAR.
- **Results:**
  - LiDAR-Only: 29.40% mAP / 38.60% NDS
  - BEVFusionLite Light: 31.80% mAP / 42.10% NDS (37.3 FPS in FP16).
- **Qualitative Gallery:** Multi-camera 3D projection alignment and surround-view bounding box tracking.
- **Scientific Limitation:** 10 scenes carry statistical variance; serves as cross-dataset architecture validation.

---

### Slide 10: Interactive Public Web Platform & Live Demo
- **Next.js + Three.js Research Platform:**
  - 12 comprehensive pages (Home, Demo, Results, Ablations, nuScenes, Architecture, Reproduce, Downloads).
  - WebGL 3D point cloud visualizer with interactive camera controls.
  - Multi-view camera projection overlays with real-time class and confidence threshold filtering.
  - Zero hardcoded numbers: all views read directly from generated JSON/CSV bundles.

---

### Slide 11: Limitations & Critical Scientific Reflection
- **Monocular Occlusion & Calibration Drift:** Sensitive to extrinsic misalignment; minor decalibration degrades frustum scatter.
- **Dataset Diversity:** KITTI evaluates front-camera FOV (~90 deg) only.
- **Small nuScenes Subset:** Full 1000-scene evaluation pending cloud GPU quota.
- **Target Audit:** 6 out of 6 predefined acceptance targets successfully MET.

---

### Slide 12: Conclusion & Open Source Deliverables
- **Key Takeaway:** Multi-modal BEV fusion does not require heavy transformers or custom CUDA kernels to deliver strong 3D detection. An honest, lightweight design achieves 31.2 FPS on NVIDIA T4 with publication-grade accuracy.
- **Open-Source Artifacts:**
  - GitHub Repository: https://github.com/vineetrajojha/BT-177_Capstone
  - Live Research Website: Deployed on Vercel
  - Turnkey Reproduction Notebooks (Google Colab & Kaggle)
- **Q&A:** Thank you! Questions?
