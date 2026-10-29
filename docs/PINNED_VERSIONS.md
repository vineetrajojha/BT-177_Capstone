# Pinned Dependency and Component Versions

This document records the exact pinned commits, releases, and dependencies for BEVFusionLite to ensure 100% reproducibility across environments.

## Third-Party Submodules & Libraries

| Component | Pinned Version / Commit Hash | License | Purpose |
|---|---|---|---|
| **OpenPCDet** | `233f849829b6ac19afb8af8837a0246890908755` | Apache-2.0 | Base 3D perception framework, point cloud backbones, datasets |
| **Python** | `3.10` | PSF | Core runtime |
| **PyTorch** | `>=2.1.0` (with CUDA 12.1 / 11.8) | BSD-3-Clause | Deep learning compute engine |
| **Torchvision** | `>=0.16.0` | BSD-3-Clause | Camera backbone / image utilities |

## Key Python Dependencies

| Package | Purpose |
|---|---|
| `torch` | Tensor operations and neural network primitives |
| `torchvision` | ResNet-18 pretrained weights and transformations |
| `numpy` | Linear algebra and coordinate transformations |
| `scipy` | Spatial metrics and rotation matrix utilities |
| `opencv-python` | Image loading, resizing, and color conversions |
| `pillow` | Image handling |
| `pyyaml` | Configuration parsing |
| `easydict` | Hierarchical configuration attribute access |
| `tensorboard` | Training diagnostics and metric tracking |
| `tqdm` | Progress reporting |
| `pytest` | Unit testing and gradient flow verification |
| `fvcore` | Flops counting and deep learning utilities |
| `scikit-image` | Image and coordinate utilities |
| `numba` | High-performance JIT for pillarization & voxelization |
| `matplotlib` | Visualization and PR curve generation |
| `pandas` | Results aggregation and tabular reporting |
| `open3d` | 3D point cloud visualization |
| `kornia` | Differentiable geometry and image transformations |

## Hardware Target

- **Inference Hardware Target:** 1x NVIDIA T4 (or compatible Turing/Ampere NVIDIA GPU, e.g. RTX 3050 Laptop GPU for local dev)
- **Batch Size:** 1 (for all latency, memory, and FPS benchmarking)
- **CUDA Runtime:** 12.1 / 11.8
