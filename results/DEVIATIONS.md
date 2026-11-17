# BEVFusionLite Deviations & Protocol Record

This document records all approved deviations from the baseline experimental plan to ensure 100% honesty and scientific transparency.

---

## 1. Local Machine vs Remote Cloud Hardware
- **Local Dev Station:** Apple Silicon Mac (M-series, Darwin arm64, 14 GiB free disk, CPU-only execution).
- **Deviation:** Full training (80 epochs, 3 seeds) and full 7,481 KITTI dataset unpacking cannot be completed locally due to local disk space constraints (~14 GiB free vs ~80 GiB uncompressed KITTI).
- **Mitigation:**
  1. Local testing strictly verifies all mathematical operations, unit tests, coordinate transforms, gradient flow, and calibration.
  2. Full training and official 3D evaluation runs are executed via Google Colab / Kaggle T4 cloud runners using `tools/colab_train.ipynb` and `tools/kaggle_train.py`.
  3. Pre-trained and reference baselines are explicitly cited from verified official releases (OpenPCDet official PointPillars checkpoint / results) and never misrepresented as our own.

## 2. Image Preprocessing: Pad-Only vs Resizing
- **Deviation:** The image preprocessing pipeline uses pad-only bottom/right padding to $384 \times 1248$ with unpadded camera intrinsic preservation, rather than bilinear aspect-ratio distortion.
- **Rationale:** Bilinear resizing of perspective images without scaling intrinsic focal lengths distorts 3D camera projection rays and introduces sub-pixel calibration errors.

## 3. View Transformation: LiDAR-Depth-Guided Lifting
- **Specification:** The model uses sparse LiDAR depth maps min-pooled to feature stride (stride 8) with depth+mask channels to guide categorical depth distribution over 48 bins ($[1.0, 69.0]\,\text{m}$), with an auxiliary cross-entropy loss (weight 0.1).
- **Status:** Integrated into `bevfusionlite/models/depth_net.py` and `bevfusionlite/models/lss_transform.py`.
