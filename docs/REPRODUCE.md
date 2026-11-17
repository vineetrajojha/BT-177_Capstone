# BEVFusionLite Reproduction Guide

This document details the exact environment, dependencies, commands, and seeds required to reproduce 100% of the experimental findings in this project.

---

## 1. System Requirements & Hardware Target

- **Operating System:** Linux (Ubuntu 20.04/22.04 recommended) or macOS (Darwin arm64 for local dev).
- **Python:** 3.10 (or 3.10–3.14 compatible).
- **Target Evaluation GPU:** 1x NVIDIA T4 (16 GB VRAM) for official latency and memory numbers.
- **CUDA Version:** CUDA 12.1 or 11.8.

---

## 2. Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/vineetrajojha/BT-177_Capstone.git
cd BT-177_Capstone

# 2. Set up Python environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml easydict numpy scipy numba tqdm matplotlib pandas scikit-image fvcore opencv-python pillow pytest

# 4. Verify environment
python scripts/check_env.py
pytest tests/ -v
```

---

## 3. Dataset Preparation

### KITTI 3D Object Detection Benchmark
1. Download official archives from Karlsruhe Institute of Technology:
   - `data_object_image_2.zip` (12 GB)
   - `data_object_velodyne.zip` (26 GB)
   - `data_object_calib.zip` (16 MB)
   - `data_object_label_2.zip` (5 MB)
2. Unpack into `data/kitti/`:
   ```bash
   data/kitti/
   ├── ImageSets/
   ├── training/
   │   ├── calib/
   │   ├── image_2/
   │   ├── label_2/
   │   └── velodyne/
   └── testing/
       ├── calib/
       ├── image_2/
       └── velodyne/
   ```
3. Run verification and info generation:
   ```bash
   python scripts/verify_full_kitti.py --kitti_root data/kitti
   python scripts/prepare_kitti.py --kitti_root data/kitti
   python scripts/calibration_check.py
   ```

---

## 4. Training & Benchmarking

### Step 1: PointPillars Reference Baseline (Prompt 3)
```bash
# Train seeds 0, 1, 2
python third_party/OpenPCDet/tools/train.py --cfg_file configs/lidar_only_pointpillars.yaml --extra_tag seed_0
python third_party/OpenPCDet/tools/train.py --cfg_file configs/lidar_only_pointpillars.yaml --extra_tag seed_1
python third_party/OpenPCDet/tools/train.py --cfg_file configs/lidar_only_pointpillars.yaml --extra_tag seed_2

# Aggregate outputs
python results/aggregate.py
```

### Step 2: BEVFusionLite Heavy Baseline (Prompt 6)
```bash
python third_party/OpenPCDet/tools/train.py --cfg_file configs/heavy_baseline.yaml --extra_tag heavy_seed_0
```

### Step 3: BEVFusionLite Light Model (Prompt 7)
```bash
# Train Light Model seeds 0, 1, 2
python third_party/OpenPCDet/tools/train.py --cfg_file configs/light_model.yaml --extra_tag light_seed_0
python third_party/OpenPCDet/tools/train.py --cfg_file configs/light_model.yaml --extra_tag light_seed_1
python third_party/OpenPCDet/tools/train.py --cfg_file configs/light_model.yaml --extra_tag light_seed_2

# Generate main comparison and targets check
python scripts/generate_main_comparison.py
```

### Step 4: Hardware Benchmarking (Prompt 5)
```bash
# FP32 Benchmark
python tools/benchmark.py --config configs/light_model.yaml --precision fp32 --output_dir results/light_model_fp32

# FP16 Benchmark
python tools/benchmark.py --config configs/light_model.yaml --precision fp16 --output_dir results/light_model_fp16
```

### Step 5: Ablation Studies (Prompt 8)
```bash
python scripts/generate_ablations.py
```

---

## 5. Cloud Execution (Google Colab & Kaggle)

Because the full KITTI dataset (~80 GB uncompressed) and multi-seed training exceed local workstation resources:
- **Google Colab Notebook:** Run `tools/colab_train.ipynb`. Mounts Google Drive, pulls latest code, configures environment, and trains with per-epoch checkpointing.
- **Kaggle Kernel Script:** Run `tools/kaggle_train.py` with the official KITTI dataset attached.
