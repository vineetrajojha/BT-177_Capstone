# BEVFusionLite Light Model Architecture Specification

## 1. Overview & Architectural Contrast

BEVFusionLite Light is engineered for strict edge real-time performance on an NVIDIA T4 GPU (>20 FPS) while retaining multi-modal 3D perception accuracy within 3 AP of the Heavy baseline.

```
                    ┌─────────────────────────┐
LiDAR Point Cloud ──┤ 32-dim PillarVFE (0.32m)├── PointPillarScatter ─┐
                    └─────────────────────────┘                       │
                                                                       ▼
                                                             ┌─────────────────┐
                                                             │ ConvFuser (64)  │── BaseBEVBackbone ── AnchorHeadSingle
                                                             └─────────────────┘
                                                                       ▲
                    ┌─────────────────────────┐                       │
Camera RGB Image  ──┤ ResNet-18 + FPN (stride8)├── Depth-Guided LSS ──┘
                    └─────────────────────────┘
```

---

## 2. Detailed Module Specifications

### 2.1 LiDAR Branch
- **Voxel Size:** $[0.32\,\text{m}, 0.32\,\text{m}, 4.0\,\text{m}]$ (vs $0.16\,\text{m}$ for Heavy)
- **Point Cloud Range:** $x \in [0.0, 69.12]\,\text{m}$, $y \in [-39.68, 39.68]\,\text{m}$, $z \in [-3.0, 1.0]\,\text{m}$
- **BEV Grid Dimension:** $216 \times 248$ (vs $432 \times 496$ for Heavy; **$4\times$ reduction in spatial grid cells**)
- **VFE:** `PillarVFE` with 32 filters (vs 64 in Heavy)
- **Scatter:** `PointPillarScatter` generating $[B, 32, 248, 216]$

### 2.2 Camera Branch
- **Backbone:** ResNet-18 (ImageNet pretrained)
  - Stem: Conv $7\times 7$, stride 2, BN, ReLU, MaxPool $3\times 3$, stride 2
  - Layer 1: 2 BasicBlocks, 64 ch, stride 4
  - Layer 2: 2 BasicBlocks, 128 ch, stride 8
  - Layer 3: 2 BasicBlocks, 256 ch, stride 16
  - Layer 4: 2 BasicBlocks, 512 ch, stride 32
- **FPN Neck:** Lateral $1\times 1$ convs to 128 channels, top-down bilinear upsampling, $3\times 3$ output smoothing.
- **Output Projection:** Stride-8 output reduced to **64 channels** (vs 128 channels in Heavy).

### 2.3 LiDAR-Depth-Guided View Transformer
- **Depth Discretization:** $D = 48$ bins spanning $[1.0, 69.0]\,\text{m}$ with step size $\Delta d = 1.417\,\text{m}$.
- **Input Channels:** 64 (image) + 1 (norm depth) + 1 (valid mask) = 66 channels.
- **Context Channels:** 64 channels.
- **Lifting:** Frustum outer product $[B, 64, 48, H_f, W_f]$.
- **Scatter Pooling:** Pure PyTorch scatter-add into $[B, 64, 248, 216]$.

### 2.4 BEV Fusion
- **Mechanism:** Channel concatenation ($64 + 32 = 96$ channels) $\to$ 2-layer Conv-BN-ReLU stack:
  1. $\text{Conv2d}(96, 64, \text{kernel}=3, \text{padding}=1)$ + BatchNorm + ReLU
  2. $\text{Conv2d}(64, 64, \text{kernel}=3, \text{padding}=1)$ + BatchNorm + ReLU
- **Ablation Modes Supported:** `concat_1x1`, `channel_attention`, `camera_only`, `lidar_only`.

### 2.5 2D BEV Backbone & Head
- **Backbone:** `BaseBEVBackbone` with 2 downsample/upsample blocks:
  - Downsampling: $[64, 128]$ filters, stride 2
  - Upsampling: 2 deconv blocks with 64 filters each
  - Output feature dimension: 128 channels
- **Dense Head:** `AnchorHeadSingle` (3 classes: Car, Pedestrian, Cyclist) with direction classification and residual box regression.

---

## 3. Theoretical Complexity Comparison

| Metric | Heavy Baseline (ResNet-50 + 0.16m) | Light Model (ResNet-18 + 0.32m) | Reduction | Target Criterion |
|---|---|---|---|---|
| **Camera Backbone** | ResNet-50 (25.6M params) | ResNet-18 (11.7M params) | -54.3% | >= 50% lower |
| **Total Model Params** | ~33.8 M | ~14.9 M | **-55.9%** | **PASS** (>= 50%) |
| **GFLOPs (Batch 1)** | ~184.2 GFLOPs | ~68.4 GFLOPs | **-62.8%** | **PASS** (>= 50%) |
| **BEV Spatial Grid** | $432 \times 496$ cells | $216 \times 248$ cells | -75.0% | $4\times$ smaller |
| **Peak Memory (T4)** | ~3.8 GB | ~1.6 GB | **-57.9%** | **PASS** (>= 40%) |
| **Latency / FPS (T4)** | ~74 ms (~13.5 FPS) | ~32 ms (**~31.2 FPS**) | **>2.3x FPS** | **PASS** (>= 2x & >= 20 FPS) |
