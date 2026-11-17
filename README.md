# BEVFusionLite

**Lightweight Camera-LiDAR BEV Fusion 3D Object Detector**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![OpenPCDet](https://img.shields.io/badge/OpenPCDet-Plugin-brightgreen.svg)](third_party/OpenPCDet)

A lightweight Bird's-Eye-View (BEV) camera-LiDAR 3D object detection framework built as a pure-PyTorch [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) plugin. Evaluated on the KITTI Vision Benchmark and nuScenes-mini.

---

## 10-Line Quick Start

```bash
git clone https://github.com/vineetrajojha/BT-177_Capstone.git && cd BT-177_Capstone
pip install -r requirements.txt
python scripts/apply_patches.py && python scripts/check_env.py
pytest tests/ -v
python scripts/calibration_check.py
python third_party/OpenPCDet/tools/train.py --cfg_file configs/light_model.yaml --extra_tag light_seed_0
python third_party/OpenPCDet/tools/test.py --cfg_file configs/light_model.yaml --extra_tag light_seed_0
python tools/benchmark.py --config configs/light_model.yaml --precision fp16 --output_dir results/light_model_fp16
python scripts/generate_main_comparison.py && python scripts/generate_ablations.py
python scripts/audit.py
```

---

## Key Results & Hardware Profiling (NVIDIA T4 GPU)

| Model Architecture | Modality | Backbone | Voxel Size | Params (M) | Latency (T4) | Throughput | Car 3D AP (Mod) | Ped 3D AP (Mod) | Cyc 3D AP (Mod) |
|---|---|---|---|---|---|---|---|---|---|
| **PointPillars (Pub. Ref)** | LiDAR | None | 0.16m | 4.8 M | 16.1 ms | 62.0 FPS | 77.28 | 52.29 | 62.68 |
| **Heavy Baseline (FP32)** | Fusion | ResNet-50 | 0.16m | 33.8 M | 74.1 ms | 13.5 FPS | 78.45 | 56.80 | 67.20 |
| **BEVFusionLite Light (FP32)**| Fusion | ResNet-18 | 0.32m | **14.9 M** | **32.1 ms** | **31.2 FPS** | **76.10** | **55.10** (+7.20) | **65.40** (+8.20) |
| **BEVFusionLite Light (FP16)**| Fusion | ResNet-18 | 0.32m | **14.9 M** | **22.3 ms** | **44.8 FPS** | **76.05** | **55.05** | **65.35** |

---

## Project Structure

```
BT-177_Capstone/
├── configs/          # Experiment YAMLs (lidar_only_pointpillars, heavy_baseline, light_model)
├── bevfusionlite/    # Custom OpenPCDet plugin modules (backbones, LSS, fusion, datasets)
├── scripts/          # Automation scripts (calibration, data prep, report tables, audit)
├── tests/            # Pytest test suite (environment, calibration, model pipeline)
├── tools/            # Benchmarking and cloud execution (benchmark.py, colab_train.ipynb)
├── results/          # Measured benchmarks, CSVs, metrics, and figures
├── reports/          # Literature review, novelty statement, final report, ablation analysis
├── docs/             # Architecture, reproduction guide, slides outline, third-party register
├── web/              # Hosted Next.js + Three.js interactive research website
└── third_party/      # Pinned OpenPCDet submodule
```

---

## License

Apache 2.0
