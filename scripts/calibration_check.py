#!/usr/bin/env python3
"""
Calibration Check Script for KITTI Camera-LiDAR Fusion.

For each of 20 fixed frames (seeded, IDs spread across the sample):
1. Load LiDAR points, calibration, and label files.
2. For each GT object (Car, Pedestrian, Cyclist — excluding DontCare):
   - Find LiDAR points inside the 3D GT bounding box.
   - Project them with lidar2img (P2 @ R0 @ V2C).
   - Compute the fraction that fall inside the object's 2D GT bbox (expanded by 2 px).
3. Report per-frame and overall mean to results/calibration/calibration_check.csv.
   Target >= 0.90 mean fraction.
4. Report the median pixel offset between the projected 3D box corners' 2D extent
   and the GT 2D bbox from the label file.
5. Save 20 overlay PNGs (LiDAR points colored by depth) to results/calibration/.
6. Save 20 box-corner reprojection PNGs to results/calibration/.
"""

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bevfusionlite.datasets.kitti_fusion import KITTICalibration

KITTI_ROOT = os.environ.get("KITTI_ROOT", str(PROJECT_ROOT / "data" / "kitti"))
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", str(PROJECT_ROOT / "results" / "calibration"))


def parse_label_file(label_path):
    """Parse a KITTI label file, returning list of dicts for each GT object."""
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls_type = parts[0]
            if cls_type == "DontCare":
                continue
            obj = {
                "type": cls_type,
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox2d": np.array([float(parts[4]), float(parts[5]),
                                    float(parts[6]), float(parts[7])]),
                "h": float(parts[8]),
                "w": float(parts[9]),
                "l": float(parts[10]),
                "x_cam": float(parts[11]),
                "y_cam": float(parts[12]),
                "z_cam": float(parts[13]),
                "ry": float(parts[14]),
            }
            objects.append(obj)
    return objects


def get_3d_box_corners_camera(obj):
    """Get 8 corners of a 3D box in rectified camera coordinates.

    KITTI convention: (x,y,z) is camera center, y points down,
    the box bottom is at y, the box extends from y-h to y.
    """
    h, w, l = obj["h"], obj["w"], obj["l"]
    x, y, z, ry = obj["x_cam"], obj["y_cam"], obj["z_cam"], obj["ry"]

    # 8 corners in object frame (camera convention: x=right, y=down, z=forward)
    # Bottom face at y=0, top face at y=-h (upward in camera = negative y)
    corners = np.array([
        [ l/2,  0,  w/2],
        [ l/2,  0, -w/2],
        [-l/2,  0, -w/2],
        [-l/2,  0,  w/2],
        [ l/2, -h,  w/2],
        [ l/2, -h, -w/2],
        [-l/2, -h, -w/2],
        [-l/2, -h,  w/2],
    ], dtype=np.float64)

    # Rotate around Y axis by ry
    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)
    R_y = np.array([
        [ cos_ry, 0, sin_ry],
        [      0, 1,      0],
        [-sin_ry, 0, cos_ry],
    ], dtype=np.float64)

    corners = (R_y @ corners.T).T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z

    return corners  # (8, 3) in rectified camera frame


def get_3d_box_corners_lidar(obj, calib):
    """Get 8 corners of a 3D box in LiDAR frame by converting from camera."""
    corners_cam = get_3d_box_corners_camera(obj)  # (8, 3)
    corners_hom = np.hstack([corners_cam, np.ones((8, 1))])
    corners_lidar = (calib.cam2lidar @ corners_hom.T).T[:, :3]
    return corners_lidar


def points_in_3d_box_lidar(points_xyz, box_corners_lidar):
    """Check which points are inside a 3D oriented bounding box.

    Uses the convex hull approach: project points onto each face normal
    and check they are between min/max of corners along each axis.

    For simplicity, we compute the box center, axes, and half-lengths
    from the 8 corners.
    """
    # Compute local axes from corners
    # corners[0], corners[1], corners[3], corners[4] define the box
    c = box_corners_lidar
    center = c.mean(axis=0)
    # Three axes defined by edges from corner 0
    ax1 = c[1] - c[0]  # along l
    ax2 = c[3] - c[0]  # along w
    ax3 = c[4] - c[0]  # along h

    # Normalize
    l1 = np.linalg.norm(ax1)
    l2 = np.linalg.norm(ax2)
    l3 = np.linalg.norm(ax3)

    if l1 < 1e-6 or l2 < 1e-6 or l3 < 1e-6:
        return np.zeros(len(points_xyz), dtype=bool)

    u1 = ax1 / l1
    u2 = ax2 / l2
    u3 = ax3 / l3

    # Project points relative to corner[0]
    diff = points_xyz - c[0]
    proj1 = diff @ u1
    proj2 = diff @ u2
    proj3 = diff @ u3

    inside = (
        (proj1 >= -0.05) & (proj1 <= l1 + 0.05) &
        (proj2 >= -0.05) & (proj2 <= l2 + 0.05) &
        (proj3 >= -0.05) & (proj3 <= l3 + 0.05)
    )
    return inside


def project_to_image(pts_3d_lidar, calib):
    """Project LiDAR points to image coordinates.

    Returns (N, 2) pixel coords and (N,) depths.
    """
    pts_rect = calib.lidar_to_rect(pts_3d_lidar[:, :3].astype(np.float32))
    pts_img, depths = calib.rect_to_img(pts_rect)
    return pts_img, depths


def draw_overlay_png(img, points, calib, save_path):
    """Draw LiDAR points colored by depth on the image."""
    pts_img, depths = project_to_image(points, calib)

    # Filter to valid image coordinates
    H, W = img.shape[:2]
    valid = (depths > 0) & (pts_img[:, 0] >= 0) & (pts_img[:, 0] < W) & \
            (pts_img[:, 1] >= 0) & (pts_img[:, 1] < H)
    pts_img = pts_img[valid]
    depths = depths[valid]

    vis = img.copy()

    if len(depths) > 0:
        # Normalize depth for color
        d_min, d_max = depths.min(), depths.max()
        if d_max - d_min < 1e-6:
            d_norm = np.zeros_like(depths)
        else:
            d_norm = (depths - d_min) / (d_max - d_min)

        # Color map: near=red, far=blue
        for i in range(len(pts_img)):
            u, v = int(pts_img[i, 0]), int(pts_img[i, 1])
            t = d_norm[i]
            r = int(255 * (1 - t))
            b = int(255 * t)
            g = int(100 * (1 - abs(2 * t - 1)))
            cv2.circle(vis, (u, v), 2, (b, g, r), -1)  # BGR

    cv2.imwrite(str(save_path), vis)


def draw_box_corners_png(img, objects, calib, save_path):
    """Draw projected 3D box corners and GT 2D bbox on the image."""
    vis = img.copy()
    colors_3d = {"Car": (0, 255, 0), "Pedestrian": (0, 255, 255), "Cyclist": (255, 0, 0)}

    for obj in objects:
        cls = obj["type"]
        if cls not in ("Car", "Pedestrian", "Cyclist"):
            continue

        # Draw GT 2D bbox in white
        x1, y1, x2, y2 = obj["bbox2d"].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Project 3D corners to image
        corners_cam = get_3d_box_corners_camera(obj)
        corners_hom = np.hstack([corners_cam, np.ones((8, 1))]).astype(np.float32)
        corners_2d = (calib.P2 @ corners_hom.T).T
        corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]

        color = colors_3d.get(cls, (128, 128, 128))

        # Draw edges of 3D box
        # Bottom face: 0-1-2-3-0, Top face: 4-5-6-7-4, Verticals: 0-4, 1-5, 2-6, 3-7
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i_e, j_e in edges:
            p1 = tuple(corners_2d[i_e].astype(int))
            p2 = tuple(corners_2d[j_e].astype(int))
            cv2.line(vis, p1, p2, color, 1)

        # Draw projected extent as dashed rectangle
        px1 = int(corners_2d[:, 0].min())
        py1 = int(corners_2d[:, 1].min())
        px2 = int(corners_2d[:, 0].max())
        py2 = int(corners_2d[:, 1].max())
        # Dashed rect in color
        for seg_start in range(px1, px2, 8):
            seg_end = min(seg_start + 4, px2)
            cv2.line(vis, (seg_start, py1), (seg_end, py1), color, 1)
            cv2.line(vis, (seg_start, py2), (seg_end, py2), color, 1)
        for seg_start in range(py1, py2, 8):
            seg_end = min(seg_start + 4, py2)
            cv2.line(vis, (px1, seg_start), (px1, seg_end), color, 1)
            cv2.line(vis, (px2, seg_start), (px2, seg_end), color, 1)

    cv2.imwrite(str(save_path), vis)


def run_calibration_check():
    """Main calibration check routine."""
    kitti_root = Path(KITTI_ROOT)
    output_dir = Path(OUTPUT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all available sample IDs from trainval
    trainval_file = kitti_root / "ImageSets" / "trainval.txt"
    if trainval_file.exists():
        all_ids = [l.strip() for l in open(trainval_file).readlines() if l.strip()]
    else:
        # Fallback: scan training directory
        calib_dir = kitti_root / "training" / "calib"
        all_ids = sorted([f.stem for f in calib_dir.glob("*.txt")])

    # Select 20 fixed frames, seeded, spread across the dataset
    rng = np.random.RandomState(seed=2024)
    if len(all_ids) <= 20:
        selected_ids = all_ids
    else:
        indices = np.linspace(0, len(all_ids) - 1, 20, dtype=int)
        # Add some jitter
        selected_ids = [all_ids[i] for i in indices]

    print(f"Selected {len(selected_ids)} frames for calibration check")
    print(f"Frame IDs: {selected_ids}")

    # CSV results
    csv_rows = []
    all_fractions = []
    all_pixel_offsets = []

    for frame_idx, frame_id in enumerate(selected_ids):
        print(f"\n--- Frame {frame_id} ({frame_idx+1}/{len(selected_ids)}) ---")

        # Load data
        calib_file = kitti_root / "training" / "calib" / f"{frame_id}.txt"
        label_file = kitti_root / "training" / "label_2" / f"{frame_id}.txt"
        velo_file = kitti_root / "training" / "velodyne" / f"{frame_id}.bin"
        img_file = kitti_root / "training" / "image_2" / f"{frame_id}.png"

        if not all(f.exists() for f in [calib_file, label_file, velo_file, img_file]):
            print(f"  SKIP: Missing files for {frame_id}")
            continue

        calib = KITTICalibration(calib_file)
        objects = parse_label_file(label_file)
        points = np.fromfile(str(velo_file), dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(str(img_file))

        # Filter objects to Car, Pedestrian, Cyclist
        objects = [o for o in objects if o["type"] in ("Car", "Pedestrian", "Cyclist")]

        frame_fractions = []
        frame_offsets = []

        for obj_idx, obj in enumerate(objects):
            cls = obj["type"]
            bbox2d = obj["bbox2d"]  # [x1, y1, x2, y2]

            # Expand 2D bbox by 2 pixels
            bbox2d_expanded = bbox2d.copy()
            bbox2d_expanded[0] -= 2  # x1
            bbox2d_expanded[1] -= 2  # y1
            bbox2d_expanded[2] += 2  # x2
            bbox2d_expanded[3] += 2  # y2

            # Get 3D box corners in LiDAR frame
            corners_lidar = get_3d_box_corners_lidar(obj, calib)

            # Find LiDAR points inside the 3D box
            in_box = points_in_3d_box_lidar(points[:, :3], corners_lidar)
            pts_in_box = points[in_box]

            if len(pts_in_box) == 0:
                print(f"  {cls} #{obj_idx}: 0 points in 3D box (skipped)")
                continue

            # Project to image
            pts_img, depths = project_to_image(pts_in_box, calib)
            valid_depth = depths > 0
            pts_img_valid = pts_img[valid_depth]

            if len(pts_img_valid) == 0:
                print(f"  {cls} #{obj_idx}: 0 points with positive depth (skipped)")
                continue

            # Check fraction inside expanded 2D bbox
            inside_2d = (
                (pts_img_valid[:, 0] >= bbox2d_expanded[0]) &
                (pts_img_valid[:, 0] <= bbox2d_expanded[2]) &
                (pts_img_valid[:, 1] >= bbox2d_expanded[1]) &
                (pts_img_valid[:, 1] <= bbox2d_expanded[3])
            )
            fraction = inside_2d.sum() / len(pts_img_valid)
            frame_fractions.append(fraction)

            # Compute 3D box corner reprojection extent
            corners_cam = get_3d_box_corners_camera(obj)
            corners_hom = np.hstack([corners_cam, np.ones((8, 1))]).astype(np.float32)
            corners_2d = (calib.P2 @ corners_hom.T).T
            corners_2d_px = corners_2d[:, :2] / corners_2d[:, 2:3]

            proj_x1 = corners_2d_px[:, 0].min()
            proj_y1 = corners_2d_px[:, 1].min()
            proj_x2 = corners_2d_px[:, 0].max()
            proj_y2 = corners_2d_px[:, 1].max()

            offset_x1 = abs(proj_x1 - bbox2d[0])
            offset_y1 = abs(proj_y1 - bbox2d[1])
            offset_x2 = abs(proj_x2 - bbox2d[2])
            offset_y2 = abs(proj_y2 - bbox2d[3])
            median_offset = np.median([offset_x1, offset_y1, offset_x2, offset_y2])
            frame_offsets.append(median_offset)

            print(f"  {cls} #{obj_idx}: {len(pts_in_box)} pts in box, "
                  f"fraction_in_2d={fraction:.4f}, median_px_offset={median_offset:.2f}")

        if frame_fractions:
            frame_mean = np.mean(frame_fractions)
            frame_median_offset = np.median(frame_offsets)
            all_fractions.extend(frame_fractions)
            all_pixel_offsets.extend(frame_offsets)
        else:
            frame_mean = float("nan")
            frame_median_offset = float("nan")

        csv_rows.append({
            "frame_id": frame_id,
            "num_objects": len(objects),
            "num_evaluated": len(frame_fractions),
            "mean_fraction": f"{frame_mean:.4f}",
            "median_pixel_offset": f"{frame_median_offset:.2f}",
        })

        print(f"  Frame mean fraction: {frame_mean:.4f}, "
              f"median pixel offset: {frame_median_offset:.2f}")

        # Save overlay PNG (LiDAR points colored by depth)
        overlay_path = output_dir / f"overlay_{frame_id}.png"
        draw_overlay_png(img, points, calib, overlay_path)

        # Save box-corner reprojection PNG
        boxcorner_path = output_dir / f"boxcorners_{frame_id}.png"
        draw_box_corners_png(img, objects, calib, boxcorner_path)

    # Overall statistics
    if all_fractions:
        overall_mean = np.mean(all_fractions)
        overall_median_offset = np.median(all_pixel_offsets)
    else:
        overall_mean = float("nan")
        overall_median_offset = float("nan")

    csv_rows.append({
        "frame_id": "OVERALL",
        "num_objects": "",
        "num_evaluated": len(all_fractions),
        "mean_fraction": f"{overall_mean:.4f}",
        "median_pixel_offset": f"{overall_median_offset:.2f}",
    })

    # Write CSV
    csv_path = output_dir / "calibration_check.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "num_objects", "num_evaluated",
            "mean_fraction", "median_pixel_offset"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print(f"CALIBRATION CHECK RESULTS")
    print(f"{'='*60}")
    print(f"Frames evaluated:        {len(selected_ids)}")
    print(f"Total objects evaluated:  {len(all_fractions)}")
    print(f"Overall mean fraction:   {overall_mean:.4f}  (target >= 0.90)")
    print(f"Overall median px offset:{overall_median_offset:.2f}")
    print(f"CSV saved to:            {csv_path}")
    print(f"Overlay PNGs saved to:   {output_dir}")
    print(f"Box-corner PNGs saved to:{output_dir}")

    if overall_mean >= 0.90:
        print(f"\n[PASS] Overall mean fraction {overall_mean:.4f} >= 0.90")
    else:
        print(f"\n[FAIL] Overall mean fraction {overall_mean:.4f} < 0.90 -- DEBUG NEEDED")

    return overall_mean >= 0.90


if __name__ == "__main__":
    success = run_calibration_check()
    sys.exit(0 if success else 1)
