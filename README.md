# BEVFusionLite

**Lightweight Camera-LiDAR BEV Fusion 3D Object Detector**

A from-scratch implementation of a lightweight BEV fusion architecture built as an [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) plugin. Evaluated on KITTI (primary) and nuScenes-mini (secondary).

## Architecture

| Component | Implementation |
|---|---|
| LiDAR encoder | PillarVFE → PointPillarScatter → BaseBEVBackbone |
| Camera encoder | ResNet-18 (pretrained) → FPN → LSS View Transform |
| BEV fusion | Channel concatenation → 2-layer Conv |
| Detection head | AnchorHeadSingle (3-class: Car, Pedestrian, Cyclist) |

## Quick Start

```bash
# 1. Setup environment
bash scripts/setup_env.sh

# 2. Clone and patch OpenPCDet
bash scripts/apply_patches.sh

# 3. Prepare KITTI data
bash scripts/prepare_kitti.sh

# 4. Train
bash scripts/train_fusion.sh

# 5. Evaluate
bash scripts/eval_fusion.sh
```

## Evaluation Protocol

- **Dataset:** KITTI val split (3769 samples)
- **Metric:** AP R40 (3D and BEV)
- **Classes:** Car (IoU 0.7), Pedestrian (IoU 0.5), Cyclist (IoU 0.5)
- **Difficulties:** Easy, Moderate, Hard
- **Seeds:** 0, 1, 2 — report mean ± std

## Project Structure

```
bevfusionlite_project/
├── configs/          # YAML configs for all experiments
├── bevfusionlite/    # Custom code (OpenPCDet plugin)
├── scripts/          # Shell scripts for setup, training, eval
├── tests/            # pytest unit tests
├── tools/            # Utility scripts (benchmarking, result collection)
├── results/          # Training outputs, eval results (gitignored heavies)
├── reports/          # Final reports and tables
├── demo/             # Inference demos
├── docs/             # Architecture and quickstart docs
├── third_party/      # OpenPCDet (cloned, never edited directly)
├── patches/          # Documented patches for OpenPCDet
└── data/             # Dataset symlinks (gitignored)
```

## License

Apache 2.0
