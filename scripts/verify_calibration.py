#!/usr/bin/env python3
"""
KITTI Camera-LiDAR Calibration Verification Script.

Verifies camera-LiDAR calibration:
    1. Computes re-projection and round-trip error on 10 samples.
       Asserts P_velo2img = P2 @ R0_rect @ Tr_velo_to_cam is numerically stable,
       inverse works, and round-trip error < 1e-4.
    2. Projects LiDAR points onto images (colored by depth) and saves 5 overlay images.
    3. Generates 5 BEV plots with camera FOV frustum drawn to reports/calib_check/.

Usage:
    python scripts/verify_calibration.py [--data_path DATA_PATH] [--output_dir reports/calib_check]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bevfusionlite.datasets.kitti_fusion import (
    KITTICalibration,
    get_fov_mask,
    crop_points_to_range,
    POINT_CLOUD_RANGE,
)


def verify_sample_calibration(calib: KITTICalibration, points: np.ndarray) -> Dict[str, float]:
    """Perform rigorous numerical sanity checks on calibration matrices for one sample."""
    errors = {}

    # 1. R0 Orthogonality error: ||R0 @ R0.T - I||_max
    R0_orth = calib.R0 @ calib.R0.T
    orth_err = float(np.abs(R0_orth - np.eye(3)).max())
    errors["r0_orthogonality_err"] = orth_err

    # 2. Coordinate round-trip: LiDAR -> rectified camera -> LiDAR
    test_pts_lidar = points[:200, :3].copy()
    pts_rect = calib.lidar_to_rect(test_pts_lidar)
    # Inverse transform
    pts_back = (calib.cam2lidar[:3, :3] @ pts_rect.T + calib.cam2lidar[:3, 3:4]).T
    roundtrip_err = float(np.abs(pts_back - test_pts_lidar).max())
    errors["roundtrip_velo_rect_err"] = roundtrip_err

    # 3. Direct 4x4 matrix product equality:
    # P_velo2img == P2_4x4 @ R0_4x4 @ V2C_4x4
    P_expected = calib.P2_4x4 @ calib.R0_4x4 @ calib.V2C_4x4
    matrix_diff = float(np.abs(calib.lidar2img - P_expected).max())
    errors["matrix_equality_err"] = matrix_diff

    # 4. Invertibility of camera extrinsics:
    # (R0_4x4 @ V2C_4x4) @ cam2lidar must be exact identity
    M = calib.R0_4x4 @ calib.V2C_4x4
    inv_err = float(np.abs(M @ calib.cam2lidar - np.eye(4)).max())
    errors["cam2lidar_inv_err"] = inv_err

    # 5. Full round-trip: LiDAR -> camera -> image -> camera -> LiDAR for points with positive depth
    pts_cam = calib.lidar_to_rect(test_pts_lidar)
    pos_mask = pts_cam[:, 2] > 2.0
    if pos_mask.sum() > 0:
        pts_cam_pos = pts_cam[pos_mask]
        pts_lidar_pos = test_pts_lidar[pos_mask]

        # Project to 2D image coordinates and depth
        pts_img, depths = calib.rect_to_img(pts_cam_pos)

        # Unproject 2D + depth back to camera coordinates
        # u = fx * X / Z + cx + P2[0,3]/Z => X = (u - cx - P2[0,3]/depth) * depth / fx
        fx, fy = calib.P2[0, 0], calib.P2[1, 1]
        cx, cy = calib.P2[0, 2], calib.P2[1, 2]
        tx = calib.P2[0, 3] / fx
        ty = calib.P2[1, 3] / fy

        X_unproj = (pts_img[:, 0] - cx) * depths / fx - tx
        Y_unproj = (pts_img[:, 1] - cy) * depths / fy - ty
        pts_cam_unproj = np.stack([X_unproj, Y_unproj, depths], axis=1)

        # Unproject camera to LiDAR
        pts_lidar_unproj = (calib.cam2lidar[:3, :3] @ pts_cam_unproj.T + calib.cam2lidar[:3, 3:4]).T
        full_roundtrip_err = float(np.abs(pts_lidar_unproj - pts_lidar_pos).max())
    else:
        full_roundtrip_err = 0.0
    errors["full_roundtrip_fov_err"] = full_roundtrip_err

    # Numerical stability metric: max of foundational matrix errors
    max_err = max(orth_err, roundtrip_err, matrix_diff, inv_err)
    errors["max_numerical_error"] = max_err

    return errors


def project_lidar_to_image(
    points: np.ndarray,
    calib: KITTICalibration,
    image: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Project LiDAR points onto the image, colored by depth.

    Args:
        points: (N, 4) LiDAR points
        calib: calibration object
        image: (H, W, 3) RGB image

    Returns:
        vis_img: (H, W, 3) image with projected points
        n_projected: number of projected points
    """
    img_shape = image.shape[:2]
    pts_img, pts_depth = calib.lidar_to_img(points[:, :3])

    # Filter to valid image coordinates
    valid = (
        (pts_img[:, 0] >= 0)
        & (pts_img[:, 0] < img_shape[1])
        & (pts_img[:, 1] >= 0)
        & (pts_img[:, 1] < img_shape[0])
        & (pts_depth > 0)
    )

    pts_img = pts_img[valid]
    pts_depth = pts_depth[valid]

    # Normalize depth for colormap
    depth_min, depth_max = 1.0, 50.0
    depth_norm = np.clip((pts_depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
    colors = cm.turbo(depth_norm)[:, :3]  # RGB from turbo colormap
    colors = (colors * 255).astype(np.uint8)

    # Draw points on image
    vis_img = image.copy()
    for i in range(len(pts_img)):
        x, y = int(pts_img[i, 0]), int(pts_img[i, 1])
        color = tuple(int(c) for c in colors[i])
        cv2.circle(vis_img, (x, y), 2, color, -1)

    return vis_img, len(pts_img)


def visualize_bev_with_frustum(
    points: np.ndarray,
    calib: KITTICalibration,
    img_shape: Tuple[int, int],
    gt_boxes: np.ndarray = None,
    max_range: float = 70.0,
) -> plt.Figure:
    """Create a BEV visualization with camera FOV frustum lines drawn.

    Args:
        points: (N, 4) LiDAR points
        calib: calibration object
        img_shape: (H, W) image shape
        gt_boxes: optional (M, 7) GT boxes in LiDAR frame
        max_range: maximum range in meters

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 1. Plot LiDAR points (subsampled for clean rendering)
    if len(points) > 5000:
        sub_idx = np.random.choice(len(points), 5000, replace=False)
        pts_sub = points[sub_idx]
    else:
        pts_sub = points

    ax.scatter(
        pts_sub[:, 1], pts_sub[:, 0],  # Y=lateral (horizontal), X=forward (vertical)
        c=pts_sub[:, 2],
        s=1.2, alpha=0.4, cmap="viridis",
    )

    # 2. Compute Camera Origin and FOV Frustum Rays in LiDAR frame
    cam_origin_cam = np.array([0, 0, 0, 1], dtype=np.float32)
    cam_origin_lidar = (calib.cam2lidar @ cam_origin_cam)[:3]

    # Camera rays for left (u=0) and right (u=W-1) image borders at horizon v = cy
    H, W = img_shape
    fx = calib.P2[0, 0]
    cx = calib.P2[0, 2]
    cy = calib.P2[1, 2]
    v_mid = cy

    left_ray_cam = np.array([(0 - cx) * max_range / fx, (v_mid - cy) * max_range / fx, max_range, 1.0], dtype=np.float32)
    right_ray_cam = np.array([(W - 1 - cx) * max_range / fx, (v_mid - cy) * max_range / fx, max_range, 1.0], dtype=np.float32)

    left_ray_lidar = (calib.cam2lidar @ left_ray_cam)[:3]
    right_ray_lidar = (calib.cam2lidar @ right_ray_cam)[:3]

    # Draw Camera FOV Frustum
    c_y, c_x = cam_origin_lidar[1], cam_origin_lidar[0]
    l_y, l_x = left_ray_lidar[1], left_ray_lidar[0]
    r_y, r_x = right_ray_lidar[1], right_ray_lidar[0]

    # Frustum boundary lines
    ax.plot([c_y, l_y], [c_x, l_x], "r--", linewidth=2.0, label="Camera FOV Frustum")
    ax.plot([c_y, r_y], [c_x, r_x], "r--", linewidth=2.0)
    ax.plot([l_y, r_y], [l_x, r_x], "r:", linewidth=1.5)

    # Shade the FOV area
    fov_poly = plt.Polygon(
        [[c_y, c_x], [l_y, l_x], [r_y, r_x]],
        closed=True, alpha=0.10, color="red", label="Camera FOV Region"
    )
    ax.add_patch(fov_poly)

    # Mark Camera & LiDAR origins
    ax.plot(0, 0, "b*", markersize=12, label="LiDAR Sensor (0,0)")
    ax.plot(c_y, c_x, "m^", markersize=10, label="Camera Sensor")

    # 3. Plot GT boxes if available
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x, y, z, l, w, h, heading = box
            # 2D corners in LiDAR frame
            corners = np.array([
                [-l/2, -w/2],
                [l/2, -w/2],
                [l/2, w/2],
                [-l/2, w/2],
                [-l/2, -w/2],
            ])
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            corners = (R @ corners.T).T + np.array([x, y])
            ax.plot(corners[:, 1], corners[:, 0], "g-", linewidth=2.0)
            dx = (l / 2) * cos_h
            dy = (l / 2) * sin_h
            ax.arrow(y, x, dy, dx, head_width=0.8, head_length=0.8, fc="green", ec="green")

    # 4. Coordinate labels and range bounds
    xmin, ymin, _, xmax, ymax, _ = POINT_CLOUD_RANGE
    ax.set_xlim(ymin - 2, ymax + 2)
    ax.set_ylim(-2, xmax + 2)
    ax.set_xlabel("Y (lateral / left-right, m)", fontsize=11)
    ax.set_ylabel("X (forward / longitudinal, m)", fontsize=11)
    ax.set_title("LiDAR BEV Point Cloud with Camera FOV Frustum & GT Boxes", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def parse_gt_labels(label_file: Path, calib: KITTICalibration) -> np.ndarray:
    """Parse GT labels from a KITTI label file into LiDAR frame boxes."""
    if not label_file.exists():
        return np.zeros((0, 7), dtype=np.float32)

    boxes = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15 or parts[0] == "DontCare":
                continue
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])

            loc_cam = np.array([[x, y, z]], dtype=np.float32)
            loc_lidar = (calib.cam2lidar[:3, :3] @ loc_cam.T + calib.cam2lidar[:3, 3:4]).T
            loc_lidar[0, 2] += h / 2.0
            heading = -(ry + np.pi / 2.0)
            boxes.append([loc_lidar[0, 0], loc_lidar[0, 1], loc_lidar[0, 2], l, w, h, heading])

    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 7), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Verify KITTI camera-LiDAR calibration")
    parser.add_argument(
        "--data_path", type=str,
        default=os.environ.get("KITTI_ROOT", str(PROJECT_ROOT / "data" / "kitti")),
        help="Path to KITTI data directory (or via KITTI_ROOT env var)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.environ.get("OUTPUT_ROOT", str(PROJECT_ROOT / "reports" / "calib_check")),
        help="Output directory for visual verification artifacts (or via OUTPUT_ROOT env var)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("       BEVFUSIONLITE: KITTI CALIBRATION VERIFICATION")
    print("=" * 70)
    print(f"Data root:   {data_path}")
    print(f"Output dir:  {output_dir}")

    split_path = data_path / "training"
    calib_files = sorted(list((split_path / "calib").glob("*.txt")))
    if not calib_files:
        raise FileNotFoundError(f"No calibration files found in {split_path / 'calib'}")

    sample_files = calib_files[:10]
    print(f"Evaluating {len(sample_files)} samples for numerical stability & round-trip error...")

    max_global_error = 0.0

    print("-" * 75)
    print(f"{'Sample ID':<10} {'R0 Orth Err':<14} {'Roundtrip Err':<16} {'Matrix Eq Err':<16} {'Status'}")
    print("-" * 75)

    for i, cf in enumerate(sample_files):
        sample_id = cf.stem
        calib = KITTICalibration(cf)
        velo_file = split_path / "velodyne" / f"{sample_id}.bin"
        points = np.fromfile(str(velo_file), dtype=np.float32).reshape(-1, 4)

        errs = verify_sample_calibration(calib, points)
        max_s_err = errs["max_numerical_error"]
        max_global_error = max(max_global_error, max_s_err)
        status = "PASS" if max_s_err < 1e-4 else "FAIL"

        print(
            f"{sample_id:<10} "
            f"{errs['r0_orthogonality_err']:<14.2e} "
            f"{errs['roundtrip_velo_rect_err']:<16.2e} "
            f"{errs['matrix_equality_err']:<16.2e} "
            f"[{status}]"
        )

        assert max_s_err < 1e-4, f"Sample {sample_id} failed numerical stability: max error {max_s_err:.2e} >= 1e-4"

    print("-" * 75)
    print(f"Max numerical error across all {len(sample_files)} samples: {max_global_error:.2e} (< 1e-4)")
    print(">>> NUMERICAL STABILITY ASSERTION PASSED <<<\n")

    # Generate visual artifacts for the first 5 samples
    print(f"Generating visual artifacts for 5 samples into {output_dir}...")
    vis_samples = sample_files[:5]

    for cf in vis_samples:
        sample_id = cf.stem
        calib = KITTICalibration(cf)
        velo_file = split_path / "velodyne" / f"{sample_id}.bin"
        img_file = split_path / "image_2" / f"{sample_id}.png"
        lbl_file = split_path / "label_2" / f"{sample_id}.txt"

        points = np.fromfile(str(velo_file), dtype=np.float32).reshape(-1, 4)
        img_bgr = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_boxes = parse_gt_labels(lbl_file, calib)

        # 1. Overlay image
        vis_img, n_proj = project_lidar_to_image(points, calib, img_rgb)
        overlay_path = output_dir / f"sample_{sample_id}_lidar_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        # 2. BEV with Frustum plot
        fig = visualize_bev_with_frustum(points, calib, img_rgb.shape[:2], gt_boxes=gt_boxes)
        bev_path = output_dir / f"sample_{sample_id}_bev_frustum.png"
        fig.savefig(str(bev_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  [Artifacts Saved] sample {sample_id}:")
        print(f"    - Image overlay: {overlay_path.name} ({n_proj} points projected)")
        print(f"    - BEV frustum:   {bev_path.name}")

    print("\n" + "=" * 70)
    print(">>> CALIBRATION VERIFICATION AND ARTIFACT GENERATION COMPLETED <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
