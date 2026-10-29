"""
Multimodal KITTI Dataset — loads point clouds, images, and calibration.

Extends OpenPCDet's KittiDataset to additionally provide:
    - RGB images (resized and normalized)
    - Camera intrinsic matrices
    - Camera-to-LiDAR extrinsic transforms

This enables the BEVFusionLite detector to use both modalities.
"""

import numpy as np
import torch
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    _HAS_PCDET = True
except ImportError:
    KittiDataset = object
    _HAS_PCDET = False
    logger.warning("OpenPCDet not found; KittiMultimodalDataset will not work")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV not found; image loading will fail")


class KittiMultimodalDataset(KittiDataset if _HAS_PCDET else object):
    """KITTI dataset with image and calibration loading for BEVFusionLite.

    Extends OpenPCDet's KittiDataset by adding image loading, resizing,
    normalization, and calibration matrix extraction to each sample.

    Args:
        dataset_cfg: Dataset config from YAML (EasyDict).
        class_names: List of class names.
        training: Whether this is training or evaluation.
        root_path: Root path to the KITTI data directory.
        logger: Logger instance.
        image_size: Target (H, W) for resized images.
    """

    # ImageNet normalization
    IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        image_size: Tuple[int, int] = (384, 1280),
    ):
        if _HAS_PCDET:
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=training,
                root_path=root_path,
                logger=logger,
            )
        self.image_size = image_size  # (H, W)

    def _load_image(self, idx: int) -> Optional[np.ndarray]:
        """Load and preprocess an RGB image.

        Args:
            idx: Sample index (frame id).

        Returns:
            Preprocessed image as float32 array [3, H, W], or None on failure.
        """
        if not _HAS_CV2:
            return None

        img_file = self.root_split_path / "image_2" / f"{idx:06d}.png"
        if not img_file.exists():
            logger.warning(f"Image not found: {img_file}")
            return None

        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

        # Normalize: float32, /255, ImageNet mean/std
        img = img.astype(np.float32) / 255.0
        img = (img - self.IMG_MEAN) / self.IMG_STD

        # HWC → CHW
        img = img.transpose(2, 0, 1)

        return img

    def _get_calib_matrices(self, calib) -> Dict[str, np.ndarray]:
        """Extract camera intrinsics and camera-to-LiDAR transform from KITTI calib.

        Args:
            calib: OpenPCDet Calibration object.

        Returns:
            Dict with 'camera_intrinsics' [3, 3] and 'camera2lidar' [4, 4].
        """
        # Camera intrinsics from P2 (left color camera)
        # P2 is [3, 4]: P2 = K @ [R|t]
        # For KITTI, the intrinsic matrix K can be extracted from P2
        P2 = calib.P2  # [3, 4]

        # Intrinsic matrix (upper-left 3×3 of P2, assuming rectified)
        K = P2[:3, :3].copy()

        # Camera-to-LiDAR transform
        # In KITTI: V2C = Velodyne-to-Camera, R0 = rectification rotation
        # camera_point = R0 @ V2C @ lidar_point
        # So: lidar_point = V2C_inv @ R0_inv @ camera_point
        # camera2lidar = V2C_inv @ R0_inv

        R0 = np.eye(4, dtype=np.float32)
        R0[:3, :3] = calib.R0

        V2C = np.eye(4, dtype=np.float32)
        V2C[:3, :4] = calib.V2C

        # camera2lidar = inv(V2C @ R0) = inv(R0) @ inv(V2C)
        cam2lidar = np.linalg.inv(V2C @ R0)

        return {
            "camera_intrinsics": K.astype(np.float32),
            "camera2lidar": cam2lidar.astype(np.float32),
        }

    def __getitem__(self, index):
        """Get a single sample with point cloud, image, and calibration.

        Returns:
            data_dict: Standard OpenPCDet dict augmented with:
                - 'images': [3, H, W] float32 tensor
                - 'camera_intrinsics': [3, 3] float32 tensor
                - 'camera2lidar': [4, 4] float32 tensor
        """
        if not _HAS_PCDET:
            raise RuntimeError("OpenPCDet is required for KittiMultimodalDataset")

        data_dict = super().__getitem__(index)

        # Get frame index
        frame_id = data_dict.get("frame_id", None)
        if frame_id is not None:
            try:
                idx = int(frame_id)
            except (ValueError, TypeError):
                idx = index
        else:
            idx = index

        # Load image
        img = self._load_image(idx)
        if img is not None:
            data_dict["images"] = img

        # Get calibration matrices
        if hasattr(self, "get_calib") and frame_id is not None:
            calib = self.get_calib(idx)
            calib_dict = self._get_calib_matrices(calib)
            data_dict["camera_intrinsics"] = calib_dict["camera_intrinsics"]
            data_dict["camera2lidar"] = calib_dict["camera2lidar"]

        return data_dict

    @staticmethod
    def collate_batch(batch_list):
        """Custom collate that handles image and calibration tensors.

        Extends OpenPCDet's collate to stack images and calibration matrices.
        """
        if _HAS_PCDET:
            data_dict = KittiDataset.collate_batch(batch_list)
        else:
            data_dict = {}

        # Stack images if present
        if "images" in batch_list[0]:
            images = np.stack([d["images"] for d in batch_list], axis=0)
            data_dict["images"] = torch.from_numpy(images)

        # Stack calibration matrices
        for key in ["camera_intrinsics", "camera2lidar"]:
            if key in batch_list[0]:
                mats = np.stack([d[key] for d in batch_list], axis=0)
                data_dict[key] = torch.from_numpy(mats)

        return data_dict
