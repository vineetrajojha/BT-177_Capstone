"""
Unit tests for KITTI Multimodal Fusion Data Pipeline and Calibration.

Tests:
1. Calibration matrix properties and numerical stability (orthogonality, round-trip).
2. LiDAR -> Camera -> Image -> Camera -> LiDAR round-trip coordinate transformations.
3. Dataset loading, output tensor shapes, types, and value ranges.
4. Custom collate_fn batching and padding.
5. Multimodal augmentation consistency (horizontal flip, Z-rotation, scaling).
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bevfusionlite.datasets.kitti_fusion import (
    KITTICalibration,
    KITTIFusionDataset,
    get_fov_mask,
    crop_points_to_range,
    apply_image_transform_to_calib,
    flip_points_y,
    flip_boxes_y,
    flip_image_horizontal,
    flip_calib_y,
    rotate_points_z,
    rotate_boxes_z,
    rotate_calib_z,
    scale_points,
    scale_boxes,
    scale_calib,
    POINT_CLOUD_RANGE,
    DEFAULT_IMAGE_SIZE,
)
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("KITTI_ROOT", str(PROJECT_ROOT / "data" / "kitti")))
SAMPLE_CALIB = DATA_ROOT / "training" / "calib" / "000000.txt"


@pytest.fixture
def calib():
    """Load calibration object for sample 000000."""
    assert SAMPLE_CALIB.exists(), f"Calibration file not found at {SAMPLE_CALIB}"
    return KITTICalibration(SAMPLE_CALIB)


# ======================================================================
# Test 1: Calibration Matrices & Numerical Sanity
# ======================================================================
def test_kitti_calibration_matrices(calib):
    """Verify matrix dimensions, R0 orthogonality, and round-trip consistency."""
    # Shapes
    assert calib.P2.shape == (3, 4), f"P2 shape {calib.P2.shape} != (3, 4)"
    assert calib.R0.shape == (3, 3), f"R0 shape {calib.R0.shape} != (3, 3)"
    assert calib.V2C.shape == (3, 4), f"V2C shape {calib.V2C.shape} != (3, 4)"
    assert calib.R0_4x4.shape == (4, 4)
    assert calib.V2C_4x4.shape == (4, 4)
    assert calib.lidar2img.shape == (4, 4)
    assert calib.cam2lidar.shape == (4, 4)

    # R0 Orthogonality: ||R0 @ R0.T - I|| < 1e-5
    r0_orth_err = np.abs(calib.R0 @ calib.R0.T - np.eye(3)).max()
    assert r0_orth_err < 1e-5, f"R0 orthogonality error {r0_orth_err:.2e} >= 1e-5"

    # Forward-inverse consistency: (R0 @ V2C) @ cam2lidar == I
    M = calib.R0_4x4 @ calib.V2C_4x4
    inv_err = np.abs(M @ calib.cam2lidar - np.eye(4)).max()
    assert inv_err < 1e-5, f"cam2lidar inverse error {inv_err:.2e} >= 1e-5"

    # Direct 4x4 matrix equality: lidar2img == P2_4x4 @ R0_4x4 @ V2C_4x4
    expected_lidar2img = calib.P2_4x4 @ calib.R0_4x4 @ calib.V2C_4x4
    matrix_diff = np.abs(calib.lidar2img - expected_lidar2img).max()
    assert matrix_diff < 1e-6, f"lidar2img matrix diff {matrix_diff:.2e} >= 1e-6"


# ======================================================================
# Test 2: Coordinate Transform Round-Trip within FOV (< 1e-3 error)
# ======================================================================
def test_coordinate_transform_roundtrip_fov(calib):
    """Test LiDAR -> Camera -> Image -> Camera -> LiDAR round-trip error < 1e-3."""
    # Synthetic LiDAR points located inside the camera FOV (forward positive x, reasonable depth)
    pts_lidar = np.array([
        [12.5, 1.2, -0.4],
        [25.0, -3.5, 0.2],
        [40.0, 4.0, -1.0],
        [18.0, 0.0, -0.8],
        [32.0, -2.1, 0.5],
    ], dtype=np.float32)

    # 1. LiDAR -> rectified Camera
    pts_cam = calib.lidar_to_rect(pts_lidar)
    assert (pts_cam[:, 2] > 0).all(), "All test points should have positive depth in camera frame"

    # 2. Camera -> Image plane (2D pixel coordinates and depth)
    pts_img, depths = calib.rect_to_img(pts_cam)

    # 3. Image plane + depth -> rectified Camera using exact inverse of intrinsic projection
    K3 = calib.P2[:3, :3]
    inv_K3 = np.linalg.inv(K3)
    rhs = np.stack([pts_img[:, 0] * depths, pts_img[:, 1] * depths, depths], axis=1) - calib.P2[:3, 3]
    pts_cam_unproj = (inv_K3 @ rhs.T).T

    # Assert camera-frame unprojection error < 1e-4
    cam_unproj_err = np.abs(pts_cam_unproj - pts_cam).max()
    assert cam_unproj_err < 1e-4, f"Camera unprojection error {cam_unproj_err:.2e} >= 1e-4"

    # 4. Rectified Camera -> LiDAR
    pts_lidar_unproj = (calib.cam2lidar[:3, :3] @ pts_cam_unproj.T + calib.cam2lidar[:3, 3:4]).T

    # Overall round-trip error must be < 1e-3 (requirement)
    total_roundtrip_err = np.abs(pts_lidar_unproj - pts_lidar).max()
    assert total_roundtrip_err < 1e-3, f"Round-trip error {total_roundtrip_err:.2e} >= 1e-3"


# ======================================================================
# Test 3: Dataset Loading, Shapes, Types, and Filtering
# ======================================================================
def test_dataset_loading_shapes():
    """Verify KITTIFusionDataset output shapes, types, and cropping."""
    dataset = KITTIFusionDataset(
        data_root=DATA_ROOT,
        split="train",
        image_size=DEFAULT_IMAGE_SIZE,
        augment=False,
    )
    assert len(dataset) > 0, "Dataset should contain samples"

    sample = dataset[0]

    # Verify keys
    required_keys = [
        "points", "images", "P2", "R0_rect", "Tr_velo_to_cam",
        "lidar2img", "gt_boxes_lidar", "gt_names", "frame_id", "image_shape"
    ]
    for k in required_keys:
        assert k in sample, f"Missing key '{k}' in sample dict"

    # Check images shape and normalization: (3, 384, 1248)
    img = sample["images"]
    assert isinstance(img, np.ndarray)
    assert img.shape == (3, 384, 1248), f"Image shape {img.shape} != (3, 384, 1248)"
    assert img.dtype == np.float32

    # Check points: (N, 4) float32
    pts = sample["points"]
    assert isinstance(pts, np.ndarray)
    assert pts.ndim == 2 and pts.shape[1] == 4, f"Points shape {pts.shape} invalid"
    assert pts.dtype == np.float32
    assert len(pts) > 0, "Expected non-empty point cloud"

    # Verify points lie strictly within point_cloud_range
    xmin, ymin, zmin, xmax, ymax, zmax = POINT_CLOUD_RANGE
    assert (pts[:, 0] >= xmin).all() and (pts[:, 0] <= xmax).all()
    assert (pts[:, 1] >= ymin).all() and (pts[:, 1] <= ymax).all()
    assert (pts[:, 2] >= zmin).all() and (pts[:, 2] <= zmax).all()

    # Check calib tensors shapes
    assert sample["P2"].shape == (3, 4)
    assert sample["R0_rect"].shape == (4, 4)
    assert sample["Tr_velo_to_cam"].shape == (4, 4)
    assert sample["lidar2img"].shape == (4, 4)

    # Check GT boxes
    gt_boxes = sample["gt_boxes_lidar"]
    assert isinstance(gt_boxes, np.ndarray)
    assert gt_boxes.ndim == 2 and gt_boxes.shape[1] == 7


# ======================================================================
# Test 4: Custom Collate Function
# ======================================================================
def test_dataset_collate_fn():
    """Verify custom collate_fn pads points correctly and returns PyTorch tensors."""
    dataset = KITTIFusionDataset(
        data_root=DATA_ROOT,
        split="train",
        image_size=DEFAULT_IMAGE_SIZE,
        augment=False,
    )
    samples = [dataset[0], dataset[1]]
    batch = KITTIFusionDataset.collate_fn(samples)

    assert batch["batch_size"] == 2
    assert isinstance(batch["images"], torch.Tensor)
    assert batch["images"].shape == (2, 3, 384, 1248)

    assert isinstance(batch["points"], torch.Tensor)
    assert batch["points"].ndim == 3 and batch["points"].shape[0] == 2 and batch["points"].shape[2] == 4
    assert isinstance(batch["points_mask"], torch.Tensor)
    assert batch["points_mask"].shape == batch["points"].shape[:2]

    assert isinstance(batch["P2"], torch.Tensor) and batch["P2"].shape == (2, 3, 4)
    assert isinstance(batch["lidar2img"], torch.Tensor) and batch["lidar2img"].shape == (2, 4, 4)
    assert isinstance(batch["gt_boxes_lidar"], torch.Tensor) and batch["gt_boxes_lidar"].shape[0] == 2


# ======================================================================
# Test 5: Multimodal Augmentation Consistency
# ======================================================================
def test_horizontal_flip_consistency(calib):
    """Verify horizontal flip consistency between LiDAR, boxes, image, and calibration."""
    img_width = 1242
    img_height = 375

    # Test point
    p_orig = np.array([[15.0, 3.2, -0.5, 0.8]], dtype=np.float32)
    p_flipped = flip_points_y(p_orig)

    # 1. LiDAR Y-flip
    assert np.isclose(p_flipped[0, 1], -p_orig[0, 1])
    assert np.isclose(p_flipped[0, 0], p_orig[0, 0])
    assert np.isclose(p_flipped[0, 2], p_orig[0, 2])

    # 2. GT box heading negation
    box_orig = np.array([[15.0, 3.2, -0.5, 4.2, 1.8, 1.5, 0.45]], dtype=np.float32)
    box_flipped = flip_boxes_y(box_orig)
    assert np.isclose(box_flipped[0, 1], -box_orig[0, 1])
    assert np.isclose(box_flipped[0, 6], -box_orig[0, 6])

    # 3. Image flip + Calib projection consistency:
    # Point projected through original calib to (u, v)
    # Point flipped projected through flipped calib must land at (W - 1 - u, v)
    pts_img_orig, depth_orig = calib.lidar_to_img(p_orig[:, :3])
    calib_flipped = flip_calib_y(calib, img_width)
    pts_img_flipped, depth_flipped = calib_flipped.lidar_to_img(p_flipped[:, :3])

    u_orig, v_orig = pts_img_orig[0, 0], pts_img_orig[0, 1]
    u_flip, v_flip = pts_img_flipped[0, 0], pts_img_flipped[0, 1]

    # Verify horizontal coordinate is mirrored: u_flip == (W - 1) - u_orig
    mirror_err = abs(u_flip - (img_width - 1 - u_orig))
    v_err = abs(v_flip - v_orig)
    depth_err = abs(depth_flipped[0] - depth_orig[0])

    assert mirror_err < 1e-3, f"Image projection mirror error {mirror_err:.2e} >= 1e-3"
    assert v_err < 1e-3, f"Vertical coordinate changed during flip: {v_err:.2e} >= 1e-3"
    assert depth_err < 1e-4, f"Depth changed during flip: {depth_err:.2e} >= 1e-4"


def test_rotation_calib_consistency(calib):
    """Verify that updating calibration under Z-rotation preserves exact 2D projection."""
    pts_orig = np.array([[20.0, 2.5, -0.6], [35.0, -1.8, 0.1]], dtype=np.float32)
    angle = np.deg2rad(18.5)

    pts_rot = rotate_points_z(pts_orig, angle)
    calib_rot = rotate_calib_z(calib, angle)

    pts_img_orig, d_orig = calib.lidar_to_img(pts_orig)
    pts_img_rot, d_rot = calib_rot.lidar_to_img(pts_rot)

    reproj_diff = np.abs(pts_img_rot - pts_img_orig).max()
    depth_diff = np.abs(d_rot - d_orig).max()

    assert reproj_diff < 1e-4, f"Scaling reprojection error {reproj_diff:.2e} >= 1e-4"
    assert depth_diff < 1e-4, f"Scaling depth error {depth_diff:.2e} >= 1e-4"


# ======================================================================
# Test 6: Image Pad-Only Handling (Scale=1.0) Across Variable Image Sizes
# ======================================================================
def test_resize_padding_lidar2img(calib):
    """Verify pad-only handling (scale=1.0, pad bottom/right) across variable KITTI image sizes.

    KITTI images have variable native resolutions (e.g. 1224x370, 1238x374, 1242x375).
    Requirements:
    1. Scale is exactly 1.0 (no distortion/scaling applied).
    2. Image is placed at top-left [:orig_h, :orig_w] with bottom/right padding to (384, 1248).
    3. lidar2img remains completely unchanged under bottom/right padding.
    4. A known 3D LiDAR point reprojects to the exact same pixel in padded image (u_pad == u_orig, v_pad == v_orig).
    5. Horizontal flip uses the true unpadded image width W_orig for that specific frame:
       u_flip = (W_orig - 1) - u_orig.
    """
    test_shapes = [
        (370, 1224),  # Frame 000000 size
        (374, 1238),  # Intermediate size
        (375, 1242),  # Common standard size
    ]
    target_size = (384, 1248)

    # Known 3D LiDAR test points inside camera FOV
    pts_lidar = np.array([
        [15.0, 2.0, -0.5],
        [22.0, -1.5, 0.2],
        [30.0, 3.5, -0.8],
    ], dtype=np.float32)

    # Unpadded projection
    pts_img_orig, d_orig = calib.lidar_to_img(pts_lidar)

    for orig_h, orig_w in test_shapes:
        orig_shape = (orig_h, orig_w)

        # 1. Apply calibration adjustment
        calib_padded, scale, pad_offsets = apply_image_transform_to_calib(
            calib, orig_shape, target_size
        )
        pad_top, pad_left, pad_bottom, pad_right = pad_offsets

        # Verify scale is exactly 1.0
        assert scale == 1.0, f"Expected scale 1.0, got {scale}"
        assert pad_top == 0 and pad_left == 0
        assert pad_bottom == target_size[0] - orig_h
        assert pad_right == target_size[1] - orig_w

        # Under bottom/right padding with pad_top=0, pad_left=0, lidar2img must be unchanged
        matrix_diff = np.abs(calib_padded.lidar2img - calib.lidar2img).max()
        assert matrix_diff < 1e-6, f"lidar2img changed under bottom-right pad: {matrix_diff}"

        # 2. Project with padded calib
        pts_img_padded, d_padded = calib_padded.lidar_to_img(pts_lidar)

        # Depths must be identical
        assert np.allclose(d_orig, d_padded, atol=1e-5), "Depth should not change with padding"

        # Reprojected pixel coords in padded image must be identical to unpadded coords
        diff_u = np.abs(pts_img_padded[:, 0] - pts_img_orig[:, 0]).max()
        diff_v = np.abs(pts_img_padded[:, 1] - pts_img_orig[:, 1]).max()

        assert diff_u < 1e-4, f"Padded u reprojection error {diff_u:.2e} for shape {orig_shape}"
        assert diff_v < 1e-4, f"Padded v reprojection error {diff_v:.2e} for shape {orig_shape}"

        # 3. Horizontal flip MUST use the true unpadded width orig_w for this frame
        pts_flipped_lidar = flip_points_y(pts_lidar)
        calib_flipped = flip_calib_y(calib_padded, img_width=orig_w)
        pts_img_flipped, _ = calib_flipped.lidar_to_img(pts_flipped_lidar)

        # Flip formula: u_flipped = (orig_w - 1) - u_orig
        expected_u_flipped = (orig_w - 1.0) - pts_img_orig[:, 0]
        flip_u_err = np.abs(pts_img_flipped[:, 0] - expected_u_flipped).max()
        flip_v_err = np.abs(pts_img_flipped[:, 1] - pts_img_orig[:, 1]).max()

        assert flip_u_err < 1e-3, f"Flip u error {flip_u_err:.2e} for shape {orig_shape}"
        assert flip_v_err < 1e-3, f"Flip v error {flip_v_err:.2e} for shape {orig_shape}"


# ======================================================================
# Test 7: GT Sampling Config Flag (Default OFF for Multimodal)
# ======================================================================
def test_gt_sampling_config_flag():
    """Verify GT sampling is a config flag defaulting to False for multimodal models."""
    # Default initialization
    dataset_default = KITTIFusionDataset(
        data_root=DATA_ROOT,
        split="train",
        augment=False,
    )
    assert dataset_default.enable_gt_sampling is False, (
        "enable_gt_sampling must default to False for multimodal models"
    )

    # Explicit False
    dataset_off = KITTIFusionDataset(
        data_root=DATA_ROOT,
        split="train",
        augment=False,
        enable_gt_sampling=False,
    )
    assert dataset_off.enable_gt_sampling is False

    # Explicit True (for LiDAR-only ablation)
    dataset_on = KITTIFusionDataset(
        data_root=DATA_ROOT,
        split="train",
        augment=False,
        enable_gt_sampling=True,
    )
    assert dataset_on.enable_gt_sampling is True


def test_scaling_calib_consistency(calib):
    """Verify that updating calibration under scaling preserves exact 2D projection."""
    pts_orig = np.array([[20.0, 2.5, -0.6], [35.0, -1.8, 0.1]], dtype=np.float32)
    scale = 1.04

    pts_scaled = scale_points(pts_orig, scale)
    calib_scaled = scale_calib(calib, scale)

    pts_img_orig, d_orig = calib.lidar_to_img(pts_orig)
    pts_img_scaled, d_scaled = calib_scaled.lidar_to_img(pts_scaled)

    reproj_diff = np.abs(pts_img_scaled - pts_img_orig).max()
    depth_diff = np.abs(d_scaled - d_orig).max()

    assert reproj_diff < 1e-4, f"Scaling reprojection error {reproj_diff:.2e} >= 1e-4"
    assert depth_diff < 1e-4, f"Scaling depth error {depth_diff:.2e} >= 1e-4"
