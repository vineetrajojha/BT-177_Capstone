# KITTI 3D Object Detection Benchmark — Submission Instructions

This guide provides exact steps for uploading test predictions from the median-seed BEVFusionLite Light model to the official KITTI Vision Benchmark server.

---

## 1. Median-Seed Model Selection Policy

Per protocol guidelines, the test submission is produced by the **median-performing seed** evaluated on the KITTI validation split:
- Seed 0: 76.10 Car 3D AP (Median seed selected for official test submission)
- Seed 1: 76.38 Car 3D AP
- Seed 2: 75.82 Car 3D AP

---

## 2. Coordinate System Verification

KITTI evaluation requires that 3D bounding boxes in submission files are reported in the **rectified camera coordinate frame** ($x$: right, $y$: down, $z$: forward):
- Dimensions: $[h, w, l]$ in meters.
- Location: $[x, y, z]$ representing the bottom-center of the 3D bounding box in camera coordinates.
- Rotation $r_y$: Observation angle in radians around camera vertical $y$-axis ($[-\pi, \pi]$).
- All predictions have been validated with `python scripts/check_submission.py`.

---

## 3. Step-by-Step Submission Procedure

1. **Login to KITTI Benchmark:**
   - Navigate to https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
   - Sign in using your registered academic/personal account.

2. **Locate Submission Package:**
   - The verified submission zip is located at:
     `results/kitti_test_submission.zip`
   - Archive layout:
     ```
     kitti_test_submission.zip
     └── data/
         ├── 000000.txt
         ├── 000001.txt
         └── ... (exactly 7,518 files up to 007517.txt)
     ```

3. **Upload Archive:**
   - In the upload form, enter:
     - **Method Name:** `BEVFusionLite`
     - **Description:** Lightweight Camera-LiDAR BEV Fusion 3D Object Detector (OpenPCDet Plugin, ResNet-18 + 0.32m Pillar with LiDAR Depth Guidance)
     - **Sensor Modality:** Multi-Modal (Camera + LiDAR)
     - Select `results/kitti_test_submission.zip` and submit.

4. **Record Official Results:**
   - Once processed by KITTI evaluation servers (~15–30 minutes), copy the returned table of AP R40 numbers (Car, Pedestrian, Cyclist) into `results/kitti_test_results.csv`.
