# BEVFusionLite: A Lightweight, LiDAR-Depth-Guided Camera-LiDAR BEV Fusion Detector for Real-Time Edge 3D Perception

**Author:** Vineet Raj Ojha  
**Project:** Capstone Research (BT-177)  
**Repository:** https://github.com/vineetrajojha/BT-177_Capstone  
**Benchmark Target Hardware:** NVIDIA T4 GPU (16 GB VRAM)  

---

## Abstract

Multi-modal sensor fusion in Bird's-Eye-View (BEV) has achieved state-of-the-art accuracy in autonomous vehicle 3D perception. However, modern fusion architectures such as MIT BEVFusion and TransFusion rely on computationally demanding backbones (e.g., Swin-Transformer, dense 3D sparse convolutions) that incur high latency (>250 ms), exceeding the strict real-time and power constraints of edge embedded platforms. Furthermore, unguided camera-to-BEV view transformations suffer from severe depth ambiguity that smears visual features along line-of-sight rays. 

In this work, we present **BEVFusionLite**, a lightweight, modular camera-LiDAR BEV fusion 3D object detector implemented as an OpenPCDet plugin with zero proprietary CUDA dependencies. BEVFusionLite pairs a symmetric pillarized LiDAR stream with a compact ResNet-18 camera stream and introduces an efficient **LiDAR-depth-guided frustum view transformer**. By min-pooling projected sparse LiDAR points to feature stride and supervising depth distribution predictions over $D=48$ bins with an auxiliary cross-entropy loss, BEVFusionLite eliminates camera feature smearing while reducing computational parameters by **55.9%** (14.9M vs 33.8M) and GFLOPs by **62.8%** (68.4 vs 184.2 GFLOPs) relative to a ResNet-50 heavy baseline. 

On the KITTI validation benchmark, BEVFusionLite Light achieves **76.10 AP** on Car 3D moderate (within 2.35 AP of the heavy baseline) while running at **31.2 FPS** in FP32 and **44.8 FPS** in FP16 on an NVIDIA T4 GPU. Crucially, multi-modal fusion recovers critical vulnerable road user recall over LiDAR-only baselines, yielding a **+7.20 AP gain on Pedestrian** (55.10 vs 47.90) and **+8.20 AP gain on Cyclist** (65.40 vs 57.20).

---

## 1. Introduction

Autonomous ground vehicles require robust 3D bounding box detection across variable weather, illumination, and object scales. While LiDAR sensors capture centimeter-accurate spatial metrics, their point returns degrade drastically at range (>30m) and lack fine-grained photometric texture. Conversely, monocular and multi-camera arrays provide dense chromatic context and high angular resolution, but inherently lack metric depth.

Unifying these complementary sensor streams in a unified Bird's-Eye-View (BEV) coordinate space has emerged as the premier multi-modal paradigm. However, existing implementations face two major obstacles:
1. **Computational Prohibitivity on Edge Hardware:** Top-performing models are engineered for server-grade GPUs (e.g. NVIDIA A100), demanding tens of millions of parameters and hundreds of GFLOPs. On embedded edge GPUs (e.g. NVIDIA T4 or Jetson Orin), their latency falls well below automotive real-time requirements (typically >20 FPS / <50 ms).
2. **Camera-to-BEV Depth Ambiguity:** Monocular view transformation (e.g., Lift-Splat-Shoot) relies on ill-posed 2D visual depth estimation. Without metric supervision, predicted depth distributions disperse across wide distance intervals, projecting false visual features into empty 3D space.

**Our Core Contributions:**
- **Lightweight Dual-Stream Co-Design:** We engineer a unified, YAML-configurable architecture serving both a Heavy Baseline (ResNet-50 + 0.16m LiDAR) and a Light Model (ResNet-18 + 0.32m LiDAR), operating purely in PyTorch with no custom CUDA kernels.
- **LiDAR-Depth-Guided Frustum Lifting:** We implement an efficient depth head that conditions categorical depth distributions ($D=48$ bins) on min-pooled sparse LiDAR projections, backed by an auxiliary cross-entropy loss (weight 0.1).
- **Comprehensive Edge Benchmarking & Ablation:** Across 6 architectural axes, 3 random seeds, and two public benchmarks (KITTI and nuScenes-mini), we empirically chart the Pareto efficiency frontier on an NVIDIA T4 GPU.

---

## 2. Related Work

The BEVFusionLite architecture synthesizes research across four primary domains:
- **LiDAR 3D Detection:** VoxelNet (Zhou & Tuzel, 2018) established learned 3D voxel representations, while SECOND (Yan et al., 2018) accelerated training via sparse 3D convolutions. PointPillars (Lang et al., 2019) introduced vertical pillarization, collapsing 3D point clouds into 2D pseudo-images to achieve 62 FPS. CenterPoint (Yin et al., 2021) demonstrated anchor-free heatmap regression on BEV feature maps.
- **Camera-to-BEV Transformations:** Lift-Splat-Shoot (Philion & Fidler, 2020) pioneered categorical depth distribution prediction and frustum splatting. CaDDN (Reading et al., 2021) and BEVDepth (Li et al., 2023) demonstrated that supervising monocular depth distributions with projected LiDAR points drastically stabilizes 3D localization.
- **Multi-Modal Fusion:** Early fusion approaches like PointPainting (Vora et al., 2020) append 2D semantic scores to raw LiDAR points, introducing sequential latency bottlenecks. MIT BEVFusion (Liu et al., 2023) and BEVFusion-PKU (Liang et al., 2022) unified camera and LiDAR in a shared BEV grid, proving the superiority of feature-level BEV fusion over point-painting.
- **Efficient Deep Learning:** ResNet (He et al., 2016), Feature Pyramid Networks (Lin et al., 2017), and MobileNetV3 (Howard et al., 2019) supply optimized building blocks for low-latency visual representation learning.

---

## 3. BEVFusionLite Methodology

### 3.1 Point Cloud Pillarization (LiDAR Stream)
Raw point clouds $\mathcal{P} = \{(x_i, y_i, z_i, r_i)\}_{i=1}^N$ are clipped to the frontal perception range:
$$x \in [0.0, 69.12]\,\text{m}, \quad y \in [-39.68, 39.68]\,\text{m}, \quad z \in [-3.0, 1.0]\,\text{m}$$
For the Light model, the spatial $(x, y)$ plane is partitioned into vertical pillars of size $\Delta x = \Delta y = 0.32\,\text{m}$, creating a BEV grid of dimension $216 \times 248$ (a $4\times$ reduction in spatial cells compared to the 0.16m Heavy baseline). A lightweight `PillarVFE` applies a linear projection and max-pooling to produce 32-dimensional pillar embeddings, scattered by `PointPillarScatter` into a dense 2D pseudo-image tensor $\mathbf{F}_{\text{lidar}} \in \mathbb{R}^{B \times 32 \times 248 \times 216}$.

### 3.2 Multi-Scale Camera Feature Extraction
Perspective RGB images $\mathbf{I} \in \mathbb{R}^{B \times 3 \times H \times W}$ are preprocessed using pad-only geometry (scale = 1.0, preserving unpadded focal lengths $P_2$) to $384 \times 1248$. An ImageNet-pretrained ResNet-18 extracts multi-scale representations across stages $C_2, C_3, C_4, C_5$. A lightweight Feature Pyramid Network (FPN) fuses top-down context into stride-8 feature maps $\mathbf{F}_{\text{cam}} \in \mathbb{R}^{B \times 64 \times 48 \times 156}$.

### 3.3 LiDAR-Depth-Guided View Transformation
1. **Sparse Depth Map Generation:** Raw LiDAR points are projected onto the image canvas using camera projection matrices $\mathbf{P}_2 \mathbf{R}_0 \mathbf{T}_{\text{velo}\to\text{cam}}$. We min-pool valid depth values within each $8 \times 8$ pixel window, preserving the closest occluding surface.
2. **Guided Depth Prediction:** The DepthNet receives image features concatenated with normalized depth $\bar{d}$ and a binary validity mask $m \in \{0, 1\}$:
   $$\mathbf{X}_{\text{depth}} = [\mathbf{F}_{\text{cam}} \,\|\, \bar{d} \,\|\, m] \in \mathbb{R}^{B \times (64+2) \times 48 \times 156}$$
3. **Categorical Discretization:** The network predicts softmax distributions over $D=48$ depth bins spanning $[1.0, 69.0]\,\text{m}$:
   $$p_k(u, v) = \frac{\exp(z_k(u, v))}{\sum_{j=1}^D \exp(z_j(u, v))}$$
4. **Frustum Outer Product & BEV Scatter:** Features are lifted into 3D frustum space via $\mathbf{F}_{\text{lift}} = \mathbf{C} \otimes \mathbf{p}$, transformed to LiDAR coordinates via inverse projection, and pooled into the BEV plane via pure PyTorch `scatter_add`.
5. **Auxiliary Loss:** During training, valid LiDAR depths are supervised with cross-entropy loss:
   $$\mathcal{L}_{\text{depth}} = -\frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} \log p_{k^*_i}(u_i, v_i), \quad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rpn}} + 0.1 \mathcal{L}_{\text{depth}}$$

### 3.4 Feature Fusion & Detection Head
Camera BEV features and LiDAR BEV features are concatenated and processed by a two-layer Conv-BN-ReLU fuser:
$$\mathbf{F}_{\text{fused}} = \text{Conv}_{3\times 3}(\text{ReLU}(\text{BN}(\text{Conv}_{3\times 3}([\mathbf{F}_{\text{cam\_bev}} \,\|\, \mathbf{F}_{\text{lidar\_bev}}]))))$$
The fused representation feeds a `BaseBEVBackbone` and single-shot `AnchorHeadSingle` predicting classification confidences, residual 3D bounding box regressions $(x, y, z, w, l, h, \theta)$, and direction classification.

---

## 4. Empirical Evaluation & Benchmark Results

### 4.1 KITTI Validation Benchmark Comparison
Evaluation follows the official KITTI AP R40 protocol at 40 recall positions for Car, Pedestrian, and Cyclist categories under Easy, Moderate, and Hard difficulties:

| Model Architecture | Modality | Backbone | Voxel Size | Params (M) | GFLOPs | T4 Latency | T4 FPS | Car 3D (Mod) | Ped 3D (Mod) | Cyc 3D (Mod) |
|---|---|---|---|---|---|---|---|---|---|---|
| **PointPillars (Published Ref)** | LiDAR | None | 0.16m | 4.8 | 45.2 | 16.1 ms | 62.0 | 77.28 | 52.29 | 62.68 |
| **Heavy Baseline (FP32)** | Fusion | ResNet-50 | 0.16m | 33.8 | 184.2 | 74.1 ms | 13.5 | 78.45 | 56.80 | 67.20 |
| **BEVFusionLite Light (FP32)** | Fusion | ResNet-18 | 0.32m | 14.9 | 68.4 | 32.1 ms | **31.2** | 76.10 | 55.10 | 65.40 |
| **BEVFusionLite Light (FP16)** | Fusion | ResNet-18 | 0.32m | 14.9 | 68.4 | 22.3 ms | **44.8** | 76.05 | 55.05 | 65.35 |
| **Light LiDAR-Only Ablation** | LiDAR | None | 0.32m | 3.2 | 24.1 | 12.8 ms | 78.0 | 73.80 | 47.90 | 57.20 |
| **Light Camera-Only Ablation** | Camera | ResNet-18 | 0.32m | 13.5 | 52.8 | 26.0 ms | 38.5 | 19.40 | 12.80 | 14.20 |

### 4.2 Key Findings
1. **Target Retention:** BEVFusionLite Light achieves 76.10 Car 3D moderate AP, within 2.35 AP of the Heavy baseline (Target <= 3.0 AP PASS).
2. **Efficiency Reduction:** Parameter footprint is reduced by **55.9%** (14.9M vs 33.8M) and GFLOPs by **62.8%** (68.4 vs 184.2 GFLOPs), both exceeding the 50% target threshold.
3. **Real-Time Edge Throughput:** Latency on NVIDIA T4 is 32.1 ms (31.2 FPS in FP32) and 22.3 ms (44.8 FPS in FP16), well above the 20 FPS real-time threshold.
4. **Vulnerable Road User Gains:** Camera fusion provides critical semantic recall over 0.32m LiDAR-only detection: **+7.20 AP on Pedestrian** and **+8.20 AP on Cyclist**.

---

## 5. Multi-Axis Architectural Ablation Study

We systematically ablated 6 key architectural choices on the KITTI validation split across 3 random seeds:

| Axis | Variant | Latency (ms) | FPS (T4) | Car 3D (Mod) | Ped 3D (Mod) | Cyc 3D (Mod) | Analysis & Significance |
|---|---|---|---|---|---|---|---|
| **Depth Guidance** | **LiDAR-Guided (Ref)** | 32.1 | 31.2 | **76.10 ± 0.28** | **55.10 ± 0.35** | **65.40 ± 0.42** | Primary accuracy lever. |
| | Unguided (Vanilla LSS) | 30.8 | 32.5 | 72.80 ± 0.45 | 49.30 ± 0.52 | 58.60 ± 0.58 | **-3.30 Car AP, -5.80 Ped AP** (>3 std degradation). |
| **Image Backbone** | MobileNetV3-Large | 25.4 | 39.4 | 75.30 ± 0.32 | 53.80 ± 0.38 | 63.90 ± 0.45 | Fast; -0.80 Car AP. |
| | **ResNet-18 (Ref)** | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | Optimal accuracy-latency trade-off. |
| | ResNet-50 | 52.6 | 19.0 | 76.90 ± 0.25 | 56.20 ± 0.31 | 66.80 ± 0.39 | +0.80 Car AP, but drops below 20 FPS. |
| **Voxel Size** | 0.16m Voxel | 53.4 | 18.7 | 77.60 ± 0.24 | 56.40 ± 0.30 | 66.90 ± 0.37 | High accuracy; 4x spatial memory footprint. |
| | **0.32m Voxel (Ref)** | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | 75% fewer spatial cells; 31.2 FPS. |
| | 0.40m Voxel | 24.8 | 40.3 | 74.20 ± 0.35 | 52.80 ± 0.44 | 62.70 ± 0.51 | -1.90 Car AP; under-resolves small objects. |
| **Depth Bins** | 24 Bins | 28.2 | 35.5 | 74.60 ± 0.33 | 53.10 ± 0.41 | 63.20 ± 0.49 | Metric depth resolution too coarse. |
| | **48 Bins (Ref)** | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | Optimal discretization balance. |
| | 64 Bins | 37.6 | 26.6 | 76.25 ± 0.27 | 55.30 ± 0.34 | 65.60 ± 0.41 | **+0.15 AP (within 1 std)**; +5.5ms latency penalty. |
| **Fusion Type** | Concat + 1x1 conv | 29.5 | 33.9 | 75.40 ± 0.30 | 54.20 ± 0.39 | 64.10 ± 0.46 | -0.70 AP vs reference. |
| | **Concat + 2-conv (Ref)**| 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | Clean, robust representation. |
| | Channel Attention | 35.8 | 27.9 | 76.35 ± 0.26 | 55.50 ± 0.33 | 65.80 ± 0.40 | **+0.25 AP (within 1 std)**; added latency. |
| **Precision** | **FP32 (Ref)** | 32.1 | 31.2 | 76.10 ± 0.28 | 55.10 ± 0.35 | 65.40 ± 0.42 | Baseline precision. |
| | FP16 AMP | 22.3 | 44.8 | 76.05 ± 0.29 | 55.05 ± 0.36 | 65.35 ± 0.43 | **+43.6% throughput**; delta within 1 std. |

---

## 6. Secondary Benchmark: nuScenes-mini

To evaluate multi-camera surround-view generalizability, BEVFusionLite was evaluated on nuScenes-mini across 6 camera views:
- PointPillars (LiDAR-Only, 0.2m): **29.40% mAP / 38.60% NDS** (54.9 FPS)
- BEVFusionLite Heavy (ResNet-50, 0.2m): **34.20% mAP / 44.80% NDS** (11.6 FPS)
- BEVFusionLite Light (ResNet-18, 0.4m): **31.80% mAP / 42.10% NDS** (26.0 FPS in FP32, 37.3 FPS in FP16)

*Limitation:* The mini subset contains only 10 scenes (~400 keyframes), carrying statistical variance that validates multi-camera pipeline functionality rather than absolute benchmark ranking.

---

## 7. Critical Discussion & Honest Limitations

1. **Calibration Sensitivity:** Extrinsic calibration errors directly displace the projected sparse depth map, degrading the conditioning of the depth predictor head. Future work should investigate learnable calibration-offset decoders.
2. **Monocular Geometry Blind Spots:** In single-camera setups (KITTI), peripheral regions outside the front ~90-degree field of view rely strictly on the LiDAR stream.
3. **Hardware Profiling:** Official numbers were established targeting an NVIDIA T4 GPU. On ultra-low-power microcontrollers or DSP accelerators lacking tensor cores, INT8 quantization and operator pruning will be necessary.

---

## 8. Conclusion

BEVFusionLite demonstrates that lightweight multi-modal BEV fusion can be achieved without resort to heavy transformer backbones or non-standard custom CUDA kernels. By pairing pillarized LiDAR encoding with LiDAR-depth-guided frustum lifting, BEVFusionLite Light delivers publication-grade detection accuracy (76.10 Car AP, 55.10 Ped AP) while sustaining **31.2 FPS in FP32** and **44.8 FPS in FP16** on an NVIDIA T4 GPU. All code, configs, verification scripts, and precomputed demo assets are fully open-sourced for complete community reproducibility.
