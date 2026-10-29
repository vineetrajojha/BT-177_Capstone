"""
KITTI Fusion Dataset for BEVFusionLite.

A Dataset subclass that returns per-sample:
    - points (N, 4): LiDAR points cropped to camera FOV AND range
      x=[0, 69.12], y=[-39.68, 39.68], z=[-3, 1]
    - image (3, 384, 1248): resized/padded, ImageNet-normalized
    - calib matrices: P2 (3x4), R0_rect (4x4), Tr_velo_to_cam (4x4) as tensors
    - lidar2img (4x4): LiDAR-to-image projection matrix
    - gt_boxes: ground truth 3D boxes in LiDAR frame

Augmentation is applied consistently to both modalities:
    - Random horizontal flip (flip points Y, mirror image, adjust calib)
    - Random global rotation [-pi/4, pi/4] (rotate points and boxes, not image)
    - Random global scaling [0.95, 1.05]
    - GT sampling (LiDAR only, from OpenPCDet)

Image augmentations (applied only to images):
    - Random photometric distortion (brightness, contrast, hue, saturation)
"""

import copy
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Point cloud range: x=[0, 69.12], y=[-39.68, 39.68], z=[-3, 1]
POINT_CLOUD_RANGE = np.array([0, -39.68, -3, 69.12, 39.68, 1], dtype=np.float32)

# Default image size
DEFAULT_IMAGE_SIZE = (384, 1248)  # (H, W)

# ImageNet normalization constants
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class KITTICalibration:
    """KITTI calibration data parser.

    Reads a KITTI calibration file and provides convenience methods
    for coordinate transformations between camera and LiDAR frames.
    """

    def __init__(self, calib_file: Union[str, Path]):
        calib = self._read_calib_file(calib_file)

        # 3x4 projection matrix for left color camera
        self.P2 = calib["P2"].reshape(3, 4).astype(np.float32)

        # 3x3 rectification rotation matrix
        self.R0 = calib["R0_rect"].reshape(3, 3).astype(np.float32)

        # 3x4 Velodyne-to-camera transform
        self.V2C = calib["Tr_velo_to_cam"].reshape(3, 4).astype(np.float32)

        # Precompute useful matrices
        self._compute_derived()

    @staticmethod
    def _read_calib_file(filepath: Union[str, Path]) -> dict:
        """Read a KITTI calibration file."""
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                data[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()]
                )
        return data

    def _compute_derived(self):
        """Compute derived transformation matrices."""
        # R0 as 4x4
        self.R0_4x4 = np.eye(4, dtype=np.float32)
        self.R0_4x4[:3, :3] = self.R0

        # V2C as 4x4
        self.V2C_4x4 = np.eye(4, dtype=np.float32)
        self.V2C_4x4[:3, :4] = self.V2C

        # P2 as 4x4
        self.P2_4x4 = np.eye(4, dtype=np.float32)
        self.P2_4x4[:3, :4] = self.P2

        # LiDAR to image projection: P2 @ R0 @ V2C
        self.lidar2img = self.P2_4x4 @ self.R0_4x4 @ self.V2C_4x4

        # Camera to LiDAR: inv(R0 @ V2C)
        self.cam2lidar = np.linalg.inv(self.R0_4x4 @ self.V2C_4x4)

        # Camera intrinsics (upper-left 3x3 of P2)
        self.K = self.P2[:3, :3].copy()

    def lidar_to_rect(self, pts_lidar: np.ndarray) -> np.ndarray:
        """Transform points from LiDAR to rectified camera coordinates.

        Args:
            pts_lidar: (N, 3) points in LiDAR frame

        Returns:
            (N, 3) points in rectified camera frame
        """
        pts_hom = np.hstack([pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)])
        pts_rect = (self.R0_4x4 @ self.V2C_4x4 @ pts_hom.T).T
        return pts_rect[:, :3]

    def rect_to_img(self, pts_rect: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project rectified camera points to image plane.

        Args:
            pts_rect: (N, 3) points in rectified camera frame

        Returns:
            pts_img: (N, 2) pixel coordinates
            pts_depth: (N,) depth values
        """
        pts_hom = np.hstack([pts_rect, np.ones((pts_rect.shape[0], 1), dtype=np.float32)])
        pts_2d = (self.P2 @ pts_hom.T).T
        pts_depth = pts_2d[:, 2]
        pts_img = pts_2d[:, :2] / (pts_depth[:, None] + 1e-6)
        return pts_img, pts_depth

    def lidar_to_img(self, pts_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project LiDAR points directly to image plane.

        Args:
            pts_lidar: (N, 3) or (N, 4) points in LiDAR frame

        Returns:
            pts_img: (N, 2) pixel coordinates
            pts_depth: (N,) depth values
        """
        pts_rect = self.lidar_to_rect(pts_lidar[:, :3])
        return self.rect_to_img(pts_rect)


def get_fov_mask(
    points: np.ndarray,
    calib: KITTICalibration,
    img_shape: Tuple[int, int],
) -> np.ndarray:
    """Get a boolean mask for points within the camera FOV.

    Args:
        points: (N, 4) LiDAR points [x, y, z, intensity]
        calib: Calibration object
        img_shape: (H, W) of the image

    Returns:
        (N,) boolean mask
    """
    pts_rect = calib.lidar_to_rect(points[:, :3])
    pts_img, pts_depth = calib.rect_to_img(pts_rect)

    in_x = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < img_shape[1])
    in_y = (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img_shape[0])
    in_front = pts_depth >= 0

    return in_x & in_y & in_front


def crop_points_to_range(
    points: np.ndarray,
    point_range: np.ndarray = POINT_CLOUD_RANGE,
) -> np.ndarray:
    """Crop points to the specified range.

    Args:
        points: (N, 4) LiDAR points [x, y, z, intensity]
        point_range: [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
        (M, 4) cropped points
    """
    mask = (
        (points[:, 0] >= point_range[0])
        & (points[:, 0] <= point_range[3])
        & (points[:, 1] >= point_range[1])
        & (points[:, 1] <= point_range[4])
        & (points[:, 2] >= point_range[2])
        & (points[:, 2] <= point_range[5])
    )
    return points[mask]


def load_and_preprocess_image(
    img_path: Union[str, Path],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Load, resize/pad, and normalize an image.

    Args:
        img_path: Path to the image file
        target_size: (H, W) target size

    Returns:
        image: (3, H, W) float32 normalized array
        scale: (scale_h, scale_w) scaling factors applied
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    target_h, target_w = target_size

    # Pad-only (no resize): scale = 1.0, preserving exact native pixel coordinates
    # Bottom/right padding: unpadded image placed at top-left [:orig_h, :orig_w]
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h_clip = min(orig_h, target_h)
    w_clip = min(orig_w, target_w)
    padded[:h_clip, :w_clip, :] = img[:h_clip, :w_clip, :]

    # Normalize
    img_float = padded.astype(np.float32) / 255.0
    img_norm = (img_float - IMG_MEAN) / IMG_STD

    # HWC -> CHW
    img_chw = img_norm.transpose(2, 0, 1)

    return img_chw, (1.0, 1.0)


def apply_image_transform_to_calib(
    calib: KITTICalibration,
    orig_shape: Tuple[int, int],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    pad_top: int = 0,
    pad_left: int = 0,
) -> Tuple[KITTICalibration, float, Tuple[int, int, int, int]]:
    """Adjust calibration matrices for pad-only handling (scale = 1.0).

    KITTI images have variable native resolutions (e.g. 1224x370, 1238x374, 1242x375).
    Under pad-only handling:
      - scale = 1.0 (exact native pixel geometry preserved)
      - pad_top = 0, pad_left = 0 (bottom-right padding only)
      - pad_bottom = target_h - orig_h
      - pad_right = target_w - orig_w

    Because scale = 1.0 and pad_top = 0, pad_left = 0:
      u_padded = 1.0 * u_orig + pad_left = u_orig
      v_padded = 1.0 * v_orig + pad_top = v_orig
    Hence lidar2img remains completely identical to native calib.lidar2img.
    If top/left padding were ever non-zero, P2[0, :] += pad_left * P2[2, :].

    Returns:
        calib: Calibration object (unchanged under bottom-right pad)
        scale: 1.0
        pad_offsets: (pad_top, pad_left, pad_bottom, pad_right)
    """
    calib = copy.deepcopy(calib)
    orig_h, orig_w = orig_shape
    target_h, target_w = target_size

    scale = 1.0
    pad_bottom = target_h - orig_h
    pad_right = target_w - orig_w

    if pad_left != 0:
        calib.P2[0, :] = calib.P2[0, :] + pad_left * calib.P2[2, :]
    if pad_top != 0:
        calib.P2[1, :] = calib.P2[1, :] + pad_top * calib.P2[2, :]

    calib._compute_derived()
    return calib, scale, (pad_top, pad_left, pad_bottom, pad_right)


def flip_image_horizontal(image: np.ndarray) -> np.ndarray:
    """Flip image horizontally. Input: (3, H, W)."""
    return image[:, :, ::-1].copy()


def flip_points_y(points: np.ndarray) -> np.ndarray:
    """Flip points along the Y axis."""
    points = points.copy()
    points[:, 1] = -points[:, 1]
    return points


def flip_boxes_y(boxes: np.ndarray) -> np.ndarray:
    """Flip 3D boxes along Y axis.

    Args:
        boxes: (N, 7) [x, y, z, l, w, h, heading]

    Returns:
        (N, 7) flipped boxes
    """
    boxes = boxes.copy()
    boxes[:, 1] = -boxes[:, 1]
    boxes[:, 6] = -boxes[:, 6]  # negate heading
    return boxes


def flip_calib_y(calib: KITTICalibration, img_width: int) -> KITTICalibration:
    """Adjust calibration for horizontal flip.

    Under horizontal flip:
    1. Image is mirrored: u' = (W - 1) - u
    2. LiDAR points are flipped along Y: y' = -y
    To maintain exact 2D-3D correspondence:
    - P2 row 0 is mirrored: P2'[0, :] = -P2[0, :] + (img_width - 1.0) * P2[2, :]
    - V2C column 1 is negated to match y' = -y
    This produces exact correspondence: diff == 0.0.
    """
    calib = copy.deepcopy(calib)
    calib.P2[0, :] = -calib.P2[0, :] + (img_width - 1.0) * calib.P2[2, :]
    calib.V2C[:, 1] = -calib.V2C[:, 1]
    calib._compute_derived()
    return calib


def rotate_calib_z(calib: KITTICalibration, angle: float) -> KITTICalibration:
    """Adjust calibration when LiDAR points are rotated around Z.

    Since p' = R_z(angle) @ p, to maintain exact image projection:
    V2C' = V2C @ inv(R_z(angle)) = V2C @ R_z(-angle).
    """
    calib = copy.deepcopy(calib)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R_z_inv = np.eye(4, dtype=np.float32)
    R_z_inv[0, 0] = cos_a
    R_z_inv[0, 1] = sin_a
    R_z_inv[1, 0] = -sin_a
    R_z_inv[1, 1] = cos_a

    calib.V2C_4x4 = calib.V2C_4x4 @ R_z_inv
    calib.V2C = calib.V2C_4x4[:3, :4]
    calib._compute_derived()
    return calib


def scale_calib(calib: KITTICalibration, scale: float) -> KITTICalibration:
    """Adjust calibration when LiDAR points are scaled.

    Since p' = scale * p, to maintain exact image projection:
    V2C' = V2C @ diag(1/scale, 1/scale, 1/scale, 1).
    """
    calib = copy.deepcopy(calib)
    S_inv = np.eye(4, dtype=np.float32)
    S_inv[0, 0] = 1.0 / scale
    S_inv[1, 1] = 1.0 / scale
    S_inv[2, 2] = 1.0 / scale

    calib.V2C_4x4 = calib.V2C_4x4 @ S_inv
    calib.V2C = calib.V2C_4x4[:3, :4]
    calib._compute_derived()
    return calib


def rotate_points_z(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points around the Z axis.

    Args:
        points: (N, 4+) points
        angle: rotation angle in radians

    Returns:
        Rotated points
    """
    points = points.copy()
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rot.T
    return points


def rotate_boxes_z(boxes: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 3D boxes around the Z axis.

    Args:
        boxes: (N, 7) [x, y, z, l, w, h, heading]
        angle: rotation angle in radians
    """
    boxes = boxes.copy()
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # Rotate center
    new_x = boxes[:, 0] * cos_a - boxes[:, 1] * sin_a
    new_y = boxes[:, 0] * sin_a + boxes[:, 1] * cos_a
    boxes[:, 0] = new_x
    boxes[:, 1] = new_y
    boxes[:, 6] += angle
    return boxes


def scale_points(points: np.ndarray, scale: float) -> np.ndarray:
    """Scale point coordinates."""
    points = points.copy()
    points[:, :3] *= scale
    return points


def scale_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    """Scale 3D box dimensions and positions."""
    boxes = boxes.copy()
    boxes[:, :6] *= scale  # x, y, z, l, w, h
    return boxes


def photometric_distortion(image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Apply random photometric distortion to a normalized image.

    Args:
        image: (3, H, W) ImageNet-normalized float32
        rng: random state for reproducibility

    Returns:
        (3, H, W) distorted image
    """
    # De-normalize for manipulation
    img = image.copy().transpose(1, 2, 0)  # CHW -> HWC
    img = img * IMG_STD + IMG_MEAN
    img = np.clip(img, 0, 1)

    # Brightness
    if rng.random() > 0.5:
        delta = rng.uniform(-0.1, 0.1)
        img = np.clip(img + delta, 0, 1)

    # Contrast
    if rng.random() > 0.5:
        factor = rng.uniform(0.8, 1.2)
        mean = img.mean()
        img = np.clip((img - mean) * factor + mean, 0, 1)

    # Saturation (convert to HSV)
    if rng.random() > 0.5:
        img_uint8 = (img * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # Re-normalize
    img = (img - IMG_MEAN) / IMG_STD
    return img.transpose(2, 0, 1)  # HWC -> CHW


class KITTIFusionDataset(Dataset):
    """KITTI multimodal dataset for camera-LiDAR BEV fusion.

    Returns per-sample:
        - points: (N, 4) float32 — FOV-cropped and range-cropped LiDAR points
        - images: (3, 384, 1248) float32 — resized/padded, ImageNet-normalized
        - P2: (3, 4) float32 — camera projection matrix
        - R0_rect: (4, 4) float32 — rectification matrix
        - Tr_velo_to_cam: (4, 4) float32 — Velodyne-to-camera transform
        - lidar2img: (4, 4) float32 — LiDAR-to-image projection matrix
        - gt_boxes_lidar: (M, 7) float32 — [x, y, z, l, w, h, heading] in LiDAR frame
        - gt_names: list of str — class names
        - frame_id: str — sample index
        - image_shape: (2,) — original image shape [H, W]

    Args:
        data_root: Path to KITTI data directory (e.g., data/kitti)
        split: 'train' or 'val'
        class_names: List of class names to keep
        image_size: (H, W) target image size
        point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        augment: Whether to apply data augmentation
        seed: Random seed for augmentation reproducibility
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        split: str = "train",
        class_names: List[str] = None,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        point_cloud_range: Optional[np.ndarray] = None,
        augment: bool = True,
        seed: int = 0,
        enable_gt_sampling: bool = False,
    ):
        if data_root is None:
            data_root = os.environ.get("KITTI_ROOT", "data/kitti")
        self.data_root = Path(data_root)
        self.split = split
        self.class_names = class_names or ["Car", "Pedestrian", "Cyclist"]
        self.image_size = image_size
        self.point_cloud_range = (
            point_cloud_range if point_cloud_range is not None else POINT_CLOUD_RANGE
        )
        self.augment = augment and (split == "train")
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.enable_gt_sampling = enable_gt_sampling

        # Determine split path
        self.split_path = self.data_root / ("training" if split != "test" else "testing")

        # Load sample IDs
        split_file = self.data_root / "ImageSets" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        self.sample_ids = [l.strip() for l in open(split_file).readlines() if l.strip()]
        logger.info(f"KITTIFusionDataset: {split} split, {len(self.sample_ids)} samples")

        # Load info files if available
        info_file = self.data_root / f"kitti_infos_{split}.pkl"
        self.infos = None
        if info_file.exists():
            with open(info_file, "rb") as f:
                self.infos = pickle.load(f)
            logger.info(f"Loaded {len(self.infos)} infos from {info_file}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _get_lidar(self, idx: str) -> np.ndarray:
        """Load LiDAR point cloud."""
        lidar_file = self.split_path / "velodyne" / f"{idx}.bin"
        assert lidar_file.exists(), f"LiDAR file not found: {lidar_file}"
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def _get_image_path(self, idx: str) -> Path:
        """Get image file path."""
        return self.split_path / "image_2" / f"{idx}.png"

    def _get_calib(self, idx: str) -> KITTICalibration:
        """Load calibration data."""
        calib_file = self.split_path / "calib" / f"{idx}.txt"
        assert calib_file.exists(), f"Calib file not found: {calib_file}"
        return KITTICalibration(calib_file)

    def _get_label(self, idx: str) -> Tuple[np.ndarray, List[str]]:
        """Load ground truth labels.

        Returns:
            gt_boxes: (N, 7) [x, y, z, l, w, h, heading] in LiDAR frame
            gt_names: list of class names
        """
        label_file = self.split_path / "label_2" / f"{idx}.txt"
        if not label_file.exists():
            return np.zeros((0, 7), dtype=np.float32), []

        calib = self._get_calib(idx)

        gt_boxes = []
        gt_names = []

        with open(label_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                cls_type = parts[0]
                if cls_type == "DontCare":
                    continue

                # Camera frame: [h, w, l, x, y, z, ry]
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])
                ry = float(parts[14])

                # Convert from camera to LiDAR coordinates
                # Camera: x=right, y=down, z=forward
                # LiDAR: x=forward, y=left, z=up
                loc_cam = np.array([[x, y, z]], dtype=np.float32)
                loc_lidar = (calib.cam2lidar[:3, :3] @ loc_cam.T + calib.cam2lidar[:3, 3:4]).T

                # Adjust z: KITTI labels have the box bottom at y in camera frame
                # After transform, adjust z so box center is at the right height
                loc_lidar[0, 2] += h / 2.0

                # heading: camera ry to LiDAR heading
                heading = -(ry + np.pi / 2)

                gt_boxes.append([
                    loc_lidar[0, 0], loc_lidar[0, 1], loc_lidar[0, 2],
                    l, w, h, heading,
                ])
                gt_names.append(cls_type)

        if len(gt_boxes) == 0:
            return np.zeros((0, 7), dtype=np.float32), []

        return np.array(gt_boxes, dtype=np.float32), gt_names

    def _filter_by_class(
        self, gt_boxes: np.ndarray, gt_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Filter ground truth by allowed class names."""
        if len(gt_names) == 0:
            return gt_boxes, gt_names

        mask = [name in self.class_names for name in gt_names]
        gt_boxes = gt_boxes[mask]
        gt_names = [n for n, m in zip(gt_names, mask) if m]
        return gt_boxes, gt_names

    def __getitem__(self, index: int) -> Dict:
        idx = self.sample_ids[index]

        # Load data
        points = self._get_lidar(idx)
        calib = self._get_calib(idx)
        img_path = self._get_image_path(idx)

        # Get original image shape before preprocessing
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_shape = np.array(orig_img.shape[:2], dtype=np.int32)  # [H, W]

        # Load and preprocess image
        image, img_scale = load_and_preprocess_image(img_path, self.image_size)

        # Load GT if available (not for test split)
        if self.split != "test":
            gt_boxes, gt_names = self._get_label(idx)
            gt_boxes, gt_names = self._filter_by_class(gt_boxes, gt_names)
        else:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_names = []

        # Crop points to camera FOV (using raw calib and original image shape)
        fov_mask = get_fov_mask(points, calib, img_shape)
        points = points[fov_mask]

        # Crop points to range
        points = crop_points_to_range(points, self.point_cloud_range)

        # Adapt calibration to resized/padded image
        calib_orig = copy.deepcopy(calib)
        calib_resized, scale, pad_offsets = apply_image_transform_to_calib(
            calib, img_shape, self.image_size
        )
        calib = calib_resized

        # Apply augmentation (using true unpadded image dimensions of THAT frame)
        if self.augment:
            points, gt_boxes, image, calib = self._augment(
                points, gt_boxes, image, calib, img_shape
            )

        # Build output dict
        data_dict = {
            "points": points.astype(np.float32),
            "images": image.astype(np.float32),
            "P2": calib.P2.astype(np.float32),
            "R0_rect": calib.R0_4x4.astype(np.float32),
            "Tr_velo_to_cam": calib.V2C_4x4.astype(np.float32),
            "lidar2img": calib.lidar2img.astype(np.float32),
            "lidar2img_orig": calib_orig.lidar2img.astype(np.float32),
            "scale_factor": float(scale),
            "pad_offsets": np.array(pad_offsets, dtype=np.int32),
            "gt_boxes_lidar": gt_boxes.astype(np.float32),
            "gt_names": gt_names,
            "frame_id": idx,
            "image_shape": img_shape,
        }

        return data_dict

    def _augment(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        image: np.ndarray,
        calib: KITTICalibration,
        img_shape: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, KITTICalibration]:
        """Apply consistent augmentation to points, boxes, image, and calib.

        Augmentations:
            1. Random horizontal flip (50% prob) using true unpadded image width
            2. Random global rotation [-pi/4, pi/4]
            3. Random global scaling [0.95, 1.05]
            4. Random photometric distortion (image only)
        """
        orig_h, orig_w = img_shape[:2]

        # 1. Random horizontal flip
        if self.rng.random() > 0.5:
            points = flip_points_y(points)
            if len(gt_boxes) > 0:
                gt_boxes = flip_boxes_y(gt_boxes)
            # Flip unpadded active region of image; padded zeros remain at bottom/right
            image[:, :orig_h, :orig_w] = image[:, :orig_h, :orig_w, ::-1].copy()
            # Flip calib using true unpadded width of THAT frame
            calib = flip_calib_y(calib, orig_w)

        # 2. Random global rotation (LiDAR points and boxes only)
        rot_angle = self.rng.uniform(-np.pi / 4, np.pi / 4)
        points = rotate_points_z(points, rot_angle)
        if len(gt_boxes) > 0:
            gt_boxes = rotate_boxes_z(gt_boxes, rot_angle)
        calib = rotate_calib_z(calib, rot_angle)

        # 3. Random global scaling
        scale_factor = self.rng.uniform(0.95, 1.05)
        points = scale_points(points, scale_factor)
        if len(gt_boxes) > 0:
            gt_boxes = scale_boxes(gt_boxes, scale_factor)
        calib = scale_calib(calib, scale_factor)

        # 4. Random photometric distortion (image only)
        image = photometric_distortion(image, self.rng)

        return points, gt_boxes, image, calib

    @staticmethod
    def collate_fn(batch_list: List[Dict]) -> Dict:
        """Custom collate function for the dataloader.

        Handles variable-length points with padding and stacks fixed-size tensors.
        """
        batch_size = len(batch_list)

        # Find max number of points
        max_points = max(d["points"].shape[0] for d in batch_list)

        # Pad points and create mask
        points_padded = np.zeros((batch_size, max_points, 4), dtype=np.float32)
        points_mask = np.zeros((batch_size, max_points), dtype=bool)

        for i, d in enumerate(batch_list):
            n = d["points"].shape[0]
            points_padded[i, :n, :] = d["points"]
            points_mask[i, :n] = True

        # Stack fixed-size arrays
        images = np.stack([d["images"] for d in batch_list], axis=0)
        P2 = np.stack([d["P2"] for d in batch_list], axis=0)
        R0_rect = np.stack([d["R0_rect"] for d in batch_list], axis=0)
        Tr_velo = np.stack([d["Tr_velo_to_cam"] for d in batch_list], axis=0)
        lidar2img = np.stack([d["lidar2img"] for d in batch_list], axis=0)
        image_shapes = np.stack([d["image_shape"] for d in batch_list], axis=0)

        # Handle variable GT boxes
        max_gt = max(d["gt_boxes_lidar"].shape[0] for d in batch_list) if batch_list else 0
        max_gt = max(max_gt, 1)  # at least 1

        gt_boxes = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
        gt_mask = np.zeros((batch_size, max_gt), dtype=bool)

        for i, d in enumerate(batch_list):
            n = d["gt_boxes_lidar"].shape[0]
            if n > 0:
                gt_boxes[i, :n, :] = d["gt_boxes_lidar"]
                gt_mask[i, :n] = True

        result = {
            "points": torch.from_numpy(points_padded),
            "points_mask": torch.from_numpy(points_mask),
            "images": torch.from_numpy(images),
            "P2": torch.from_numpy(P2),
            "R0_rect": torch.from_numpy(R0_rect),
            "Tr_velo_to_cam": torch.from_numpy(Tr_velo),
            "lidar2img": torch.from_numpy(lidar2img),
            "gt_boxes_lidar": torch.from_numpy(gt_boxes),
            "gt_mask": torch.from_numpy(gt_mask),
            "image_shape": torch.from_numpy(image_shapes),
            "frame_id": [d["frame_id"] for d in batch_list],
            "gt_names": [d["gt_names"] for d in batch_list],
            "batch_size": batch_size,
        }

        if "lidar2img_orig" in batch_list[0]:
            result["lidar2img_orig"] = torch.from_numpy(
                np.stack([d["lidar2img_orig"] for d in batch_list], axis=0)
            )

        return result
