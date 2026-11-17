# Third-Party Assets, Datasets, and Pretrained Weights Register

All third-party code, pretrained checkpoints, and external assets used in this research are catalogued below with exact versions, URLs, licenses, and usage roles.

---

## 1. Frameworks & Codebases

| Asset | Source / URL | Version / Commit | License | Usage Role |
|---|---|---|---|---|
| **OpenPCDet** | https://github.com/open-mmlab/OpenPCDet | `233f849829b6ac19afb8af8837a0246890908755` | Apache 2.0 | Base 3D perception library; LiDAR VFE, PointPillarScatter, 2D BEV Backbone, AnchorHeadSingle, and KITTI evaluation protocol. |
| **Lift-Splat-Shoot** | https://github.com/nv-tlabs/lift-splat-shoot | Paper: ECCV 2020 | Apache 2.0 / MIT | Conceptual reference for frustum construction and camera-to-BEV pooling. |
| **BEVFusion (MIT)** | https://github.com/mit-han-lab/bevfusion | Paper: ICRA 2023 | MIT | Architectural reference for dual-stream concatenation and LiDAR-depth guidance concepts. |

---

## 2. Pretrained Weights

| Model | Source | Checkpoint Identifier | License | Usage Role |
|---|---|---|---|---|
| **ResNet-18** | PyTorch / Torchvision Hub | `ResNet18_Weights.IMAGENET1K_V1` | BSD-3-Clause | 2D camera backbone initialization for Light model and ablation studies. |
| **ResNet-50** | PyTorch / Torchvision Hub | `ResNet50_Weights.IMAGENET1K_V2` | BSD-3-Clause | 2D camera backbone initialization for Heavy baseline. |
| **MobileNetV3-Large** | PyTorch / Torchvision Hub | `MobileNet_V3_Large_Weights.IMAGENET1K_V1` | BSD-3-Clause | 2D camera backbone initialization for lightweight ablation row. |
| **PointPillars KITTI Zoo Checkpoint** | OpenPCDet Model Zoo | `pointpillar_7728.pth` | Apache 2.0 | Published reference comparison (strictly labelled as external published reference; never claimed as our training run). |

---

## 3. Datasets

| Dataset | Provider / Source | Split / Size | License / Terms | Usage |
|---|---|---|---|---|
| **KITTI 3D Object Detection Benchmark** | Karlsruhe Institute of Technology & Toyota Technological Institute at Chicago | 7,481 train/val samples, 7,518 test samples | CC BY-NC-SA 3.0 (Non-Commercial) | Primary benchmark for 3D object detection across Car, Pedestrian, and Cyclist. |
| **nuScenes** | Motional | `v1.0-mini` (10 scenes, 6 cameras, 32-beam LiDAR) | CC BY-NC-SA 4.0 (Non-Commercial) | Secondary benchmark for multi-camera surround-view BEV fusion. |
