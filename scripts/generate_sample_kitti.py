#!/usr/bin/env python3
"""
Generate realistic sample KITTI data for calibration verification, pipeline testing,
and development without requiring 40+ GB of dataset download.

Creates 20 samples (000000 to 000019) in data/kitti/ with:
- training/calib/ (authentic KITTI calibration files)
- training/image_2/ (375x1242 RGB images)
- training/velodyne/ (float32 point clouds simulating Velodyne HDL-64E)
- training/label_2/ (KITTI 3D labels with Car, Pedestrian, Cyclist)
- ImageSets/ (train.txt, val.txt, trainval.txt, test.txt)
"""

import os
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "kitti"

# Authentic KITTI calibration constants
CALIB_TEMPLATES = [
    {
        "P0": "7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00",
        "P1": "7.215377e+02 0.000000e+00 6.095593e+02 -3.798145e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00",
        "P2": "7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03",
        "P3": "7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03",
        "R0_rect": "9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01",
        "Tr_velo_to_cam": "7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01",
        "Tr_imu_to_velo": "9.999976e-01 7.553071e-04 -2.035820e-03 -8.086759e-01 -7.854027e-04 9.998889e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"
    },
    {
        "P0": "7.070912e+02 0.000000e+00 6.018873e+02 0.000000e+00 0.000000e+00 7.070912e+02 1.815987e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00",
        "P1": "7.070912e+02 0.000000e+00 6.018873e+02 -3.797544e+02 0.000000e+00 7.070912e+02 1.815987e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00",
        "P2": "7.070912e+02 0.000000e+00 6.018873e+02 4.575831e+01 0.000000e+00 7.070912e+02 1.815987e+02 -3.454157e-01 0.000000e+00 0.000000e+00 1.000000e+00 4.981016e-03",
        "P3": "7.070912e+02 0.000000e+00 6.018873e+02 -3.341081e+02 0.000000e+00 7.070912e+02 1.815987e+02 2.330660e+00 0.000000e+00 0.000000e+00 1.000000e+00 3.201153e-03",
        "R0_rect": "9.999128e-01 1.009263e-02 -8.511932e-03 -1.012729e-02 9.999406e-01 -4.037675e-03 8.470501e-03 4.123522e-03 9.999556e-01",
        "Tr_velo_to_cam": "6.927964e-03 -9.999722e-01 -2.757829e-03 -2.457729e-02 -1.162982e-03 2.749836e-03 -9.999955e-01 -6.127237e-02 9.999753e-01 6.931141e-03 -1.143897e-03 -3.321029e-01",
        "Tr_imu_to_velo": "9.999976e-01 7.553071e-04 -2.035820e-03 -8.086759e-01 -7.854027e-04 9.998889e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"
    }
]

def make_sample(idx_str: str, seed: int):
    rng = np.random.RandomState(seed)
    calib_dict = CALIB_TEMPLATES[seed % len(CALIB_TEMPLATES)]

    # 1. Parse calibration matrices
    P2 = np.array([float(x) for x in calib_dict["P2"].split()]).reshape(3, 4).astype(np.float32)
    R0 = np.array([float(x) for x in calib_dict["R0_rect"].split()]).reshape(3, 3).astype(np.float32)
    V2C = np.array([float(x) for x in calib_dict["Tr_velo_to_cam"].split()]).reshape(3, 4).astype(np.float32)

    R0_4x4 = np.eye(4, dtype=np.float32)
    R0_4x4[:3, :3] = R0
    V2C_4x4 = np.eye(4, dtype=np.float32)
    V2C_4x4[:3, :4] = V2C

    # cam2lidar = inv(R0_4x4 @ V2C_4x4)
    cam2lidar = np.linalg.inv(R0_4x4 @ V2C_4x4)
    lidar2cam = R0_4x4 @ V2C_4x4

    # 2. Generate Ground Truth Objects in 3D (camera and LiDAR frames)
    # Define 2 to 4 realistic objects per frame
    num_objs = rng.randint(2, 5)
    labels = []
    gt_boxes_lidar = []

    for o_i in range(num_objs):
        cls_name = rng.choice(["Car", "Car", "Pedestrian", "Cyclist"])
        if cls_name == "Car":
            h, w, l = float(rng.uniform(1.4, 1.7)), float(rng.uniform(1.6, 2.0)), float(rng.uniform(3.8, 4.8))
            x_lidar = float(rng.uniform(8.0, 35.0))
            y_lidar = float(rng.uniform(-6.0, 6.0))
            z_lidar = float(-1.6 + h / 2.0)
            heading = float(rng.uniform(-0.3, 0.3))
        elif cls_name == "Pedestrian":
            h, w, l = float(rng.uniform(1.6, 1.85)), float(rng.uniform(0.5, 0.7)), float(rng.uniform(0.6, 0.9))
            x_lidar = float(rng.uniform(5.0, 20.0))
            y_lidar = float(rng.uniform(-4.0, 4.0))
            z_lidar = float(-1.6 + h / 2.0)
            heading = float(rng.uniform(-np.pi, np.pi))
        else:
            h, w, l = float(rng.uniform(1.4, 1.7)), float(rng.uniform(0.5, 0.8)), float(rng.uniform(1.5, 1.9))
            x_lidar = float(rng.uniform(7.0, 25.0))
            y_lidar = float(rng.uniform(-5.0, 5.0))
            z_lidar = float(-1.6 + h / 2.0)
            heading = float(rng.uniform(-0.5, 0.5))

        gt_boxes_lidar.append([x_lidar, y_lidar, z_lidar, l, w, h, heading])

        # Convert box center to camera rectified frame
        pos_lidar_hom = np.array([x_lidar, y_lidar, z_lidar - h / 2.0, 1.0], dtype=np.float32)
        pos_cam = (lidar2cam @ pos_lidar_hom)[:3]
        x_cam, y_cam, z_cam = pos_cam[0], pos_cam[1], pos_cam[2]
        ry = float(-heading - np.pi / 2.0)

        # 2D bounding box projection
        # Project 3D box corners to image
        corners_lidar = np.array([
            [-l/2, -w/2, -h/2],
            [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2],
            [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2],
            [l/2, -w/2, h/2],
            [l/2, w/2, h/2],
            [-l/2, w/2, h/2],
        ])
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        R_rot = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
        corners_lidar = (R_rot @ corners_lidar.T).T + np.array([x_lidar, y_lidar, z_lidar])
        corners_hom = np.hstack([corners_lidar, np.ones((8, 1))])
        corners_cam = (lidar2cam @ corners_hom.T).T[:, :3]
        corners_img_hom = (P2 @ np.hstack([corners_cam, np.ones((8, 1))]).T).T
        corners_2d = corners_img_hom[:, :2] / corners_img_hom[:, 2:3]
        x1 = float(np.clip(corners_2d[:, 0].min(), 0, 1241))
        y1 = float(np.clip(corners_2d[:, 1].min(), 0, 374))
        x2 = float(np.clip(corners_2d[:, 0].max(), 0, 1241))
        y2 = float(np.clip(corners_2d[:, 1].max(), 0, 374))

        # KITTI label line format:
        # type truncated occluded alpha bbox_x1 bbox_y1 bbox_x2 bbox_y2 h w l x y z ry
        alpha = float(ry - np.arctan2(x_cam, z_cam))
        label_line = (
            f"{cls_name} 0.00 0 {alpha:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} {x_cam:.2f} {y_cam:.2f} {z_cam:.2f} {ry:.2f}"
        )
        labels.append((label_line, (x1, y1, x2, y2), cls_name))

    # 3. Generate Realistic LiDAR Points (Velodyne HDL-64E)
    # Ground plane at z ~ -1.6m
    pts_list = []
    # Ground grid
    num_ground = 12000
    xg = rng.uniform(2.0, 70.0, num_ground)
    yg = rng.uniform(-35.0, 35.0, num_ground)
    zg = -1.65 + rng.normal(0, 0.04, num_ground)
    ig = rng.uniform(0.1, 0.4, num_ground)
    ground_pts = np.stack([xg, yg, zg, ig], axis=1)
    pts_list.append(ground_pts)

    # Roadside buildings/trees/poles
    num_side = 5000
    xs = rng.uniform(5.0, 65.0, num_side)
    ys = rng.choice([-1, 1], num_side) * rng.uniform(8.0, 35.0, num_side)
    zs = rng.uniform(-1.6, 5.0, num_side)
    is_ = rng.uniform(0.2, 0.9, num_side)
    pts_list.append(np.stack([xs, ys, zs, is_], axis=1))

    # Points on the objects
    for box in gt_boxes_lidar:
        bx, by, bz, bl, bw, bh, bhead = box
        num_obj_pts = rng.randint(250, 800)
        ox = rng.uniform(-bl/2, bl/2, num_obj_pts)
        oy = rng.uniform(-bw/2, bw/2, num_obj_pts)
        oz = rng.uniform(-bh/2, bh/2, num_obj_pts)
        cos_b, sin_b = np.cos(bhead), np.sin(bhead)
        R_b = np.array([[cos_b, -sin_b, 0], [sin_b, cos_b, 0], [0, 0, 1]])
        local_pts = np.stack([ox, oy, oz], axis=1)
        rot_pts = (R_b @ local_pts.T).T + np.array([bx, by, bz])
        oi = rng.uniform(0.5, 0.95, num_obj_pts)
        pts_list.append(np.column_stack([rot_pts, oi]))

    all_points = np.vstack(pts_list).astype(np.float32)

    # 4. Generate Realistic Image (375x1242)
    H, W = 375, 1242
    img = np.zeros((H, W, 3), dtype=np.uint8)
    horizon = int(P2[1, 2])  # ~172
    # Sky gradient
    for r in range(horizon):
        alpha = r / horizon
        color = np.array([215 * (1 - 0.3 * alpha), 180 * (1 - 0.2 * alpha), 130 * (1 - 0.1 * alpha)])
        img[r, :] = color.astype(np.uint8)
    # Ground / road gradient
    for r in range(horizon, H):
        alpha = (r - horizon) / (H - horizon)
        road_c = np.array([55 + 25 * alpha, 55 + 25 * alpha, 55 + 25 * alpha])
        img[r, :] = road_c.astype(np.uint8)
    # Draw lane lines
    pts_lane_l = np.array([[W * 0.46, horizon], [W * 0.15, H], [W * 0.18, H], [W * 0.47, horizon]], dtype=np.int32)
    pts_lane_r = np.array([[W * 0.54, horizon], [W * 0.85, H], [W * 0.82, H], [W * 0.53, horizon]], dtype=np.int32)
    cv2.fillPoly(img, [pts_lane_l], (200, 200, 200))
    cv2.fillPoly(img, [pts_lane_r], (200, 200, 200))

    # Draw vehicles / pedestrians in the image
    for (line, (x1, y1, x2, y2), cls_name) in labels:
        if x2 > x1 + 4 and y2 > y1 + 4:
            x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
            if cls_name == "Car":
                obj_col = (rng.randint(30, 180), rng.randint(40, 190), rng.randint(50, 210))
            elif cls_name == "Pedestrian":
                obj_col = (180, 100, 50)
            else:
                obj_col = (60, 160, 60)
            cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), obj_col, -1)
            cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), (20, 20, 20), 1)

    return calib_dict, img, all_points, [l[0] for l in labels]


def main():
    print(f"Creating sample KITTI dataset at {DATA_ROOT}...")

    split_train = DATA_ROOT / "training"
    (split_train / "calib").mkdir(parents=True, exist_ok=True)
    (split_train / "image_2").mkdir(parents=True, exist_ok=True)
    (split_train / "velodyne").mkdir(parents=True, exist_ok=True)
    (split_train / "label_2").mkdir(parents=True, exist_ok=True)

    imagesets_dir = DATA_ROOT / "ImageSets"
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [f"{i:06d}" for i in range(20)]

    for i, idx_str in enumerate(sample_ids):
        calib_dict, img, points, labels = make_sample(idx_str, seed=42 + i)

        # Write calib
        calib_file = split_train / "calib" / f"{idx_str}.txt"
        with open(calib_file, "w") as f:
            for k in ["P0", "P1", "P2", "P3", "R0_rect", "Tr_velo_to_cam", "Tr_imu_to_velo"]:
                f.write(f"{k}: {calib_dict[k]}\n")

        # Write image
        img_file = split_train / "image_2" / f"{idx_str}.png"
        cv2.imwrite(str(img_file), img)

        # Write velodyne bin
        velo_file = split_train / "velodyne" / f"{idx_str}.bin"
        points.astype(np.float32).tofile(str(velo_file))

        # Write label_2
        label_file = split_train / "label_2" / f"{idx_str}.txt"
        with open(label_file, "w") as f:
            f.write("\n".join(labels) + "\n")

        print(f"  [Created] sample {idx_str}: {len(points)} pts, {img.shape}, {len(labels)} objects")

    # Write ImageSets
    train_ids = sample_ids[:12]
    val_ids = sample_ids[12:]

    with open(imagesets_dir / "train.txt", "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(imagesets_dir / "val.txt", "w") as f:
        f.write("\n".join(val_ids) + "\n")
    with open(imagesets_dir / "trainval.txt", "w") as f:
        f.write("\n".join(sample_ids) + "\n")
    test_calib_dir = DATA_ROOT / "testing" / "calib"
    if test_calib_dir.exists():
        test_ids = sorted([f.stem for f in test_calib_dir.glob("*.txt")])
    else:
        test_ids = ["000000", "000001"]

    with open(imagesets_dir / "test.txt", "w") as f:
        f.write("\n".join(test_ids) + "\n")

    print(f"Created ImageSets (train: {len(train_ids)}, val: {len(val_ids)})")
    print(f"Sample KITTI data generated successfully!")


if __name__ == "__main__":
    main()
