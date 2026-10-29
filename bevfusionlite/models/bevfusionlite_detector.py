"""
BEVFusionLite Detector — Main detector class.

This is the top-level detector that integrates:
    - LiDAR stream: PillarVFE → PointPillarScatter → BaseBEVBackbone (from OpenPCDet)
    - Camera stream: ResNet-18 + FPN → LSS View Transform (custom)
    - Fusion: ConvFuser (custom)
    - Detection head: AnchorHeadSingle (from OpenPCDet)

Designed as an OpenPCDet plugin — inherits Detector3DTemplate and is
registered in bevfusionlite.__init__.get_registered_detectors().
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# These imports will work once OpenPCDet is on sys.path
# (handled by bevfusionlite.__init__._add_openpcdet_to_path)
try:
    from pcdet.models.detectors.detector3d_template import Detector3DTemplate
except ImportError:
    # Allow module to be imported for testing without OpenPCDet
    Detector3DTemplate = nn.Module
    logger.warning("OpenPCDet not found; using nn.Module as base class (testing mode)")

from .camera_backbone import CameraBackbone
from .lss_transform import LSSTransform
from .bev_fusion import ConvFuser


class BEVFusionLiteDetector(Detector3DTemplate):
    """Lightweight Camera-LiDAR BEV Fusion 3D Object Detector.

    Processes point cloud through OpenPCDet's pillar-based LiDAR pipeline
    and RGB images through a custom camera encoder + LSS view transform,
    fuses both in BEV space, and predicts 3D bounding boxes.

    Config structure (YAML):
        MODEL:
            NAME: BEVFusionLiteDetector
            CAMERA:
                ENABLED: True
                BACKBONE:
                    PRETRAINED: True
                    FPN_OUT_CHANNELS: 256
                    OUT_CHANNELS: 256
                    FROZEN_STAGES: 1
                IMAGE_SIZE: [384, 1280]
                FEATURE_SIZE: [48, 160]
                LSS:
                    OUT_CHANNELS: 80
                    XBOUND: [0.0, 69.12, 0.16]
                    YBOUND: [-39.68, 39.68, 0.16]
                    ZBOUND: [-3.0, 1.0, 4.0]
                    DBOUND: [1.0, 45.0, 0.5]
            FUSION:
                IN_CHANNELS_CAMERA: 80
                IN_CHANNELS_LIDAR: 128
                OUT_CHANNELS: 256
            VFE / BACKBONE_3D / MAP_TO_BEV / BACKBONE_2D / DENSE_HEAD:
                (standard OpenPCDet config)
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        # --- Camera stream (custom) ---
        camera_cfg = model_cfg.get("CAMERA", None)
        self.camera_enabled = camera_cfg is not None and camera_cfg.get("ENABLED", True)

        if self.camera_enabled:
            bb_cfg = camera_cfg.get("BACKBONE", {})
            lss_cfg = camera_cfg.get("LSS", {})

            image_size = camera_cfg.get("IMAGE_SIZE", [384, 1280])
            feature_size = camera_cfg.get("FEATURE_SIZE", [48, 160])
            cam_out_ch = bb_cfg.get("OUT_CHANNELS", 256)
            lss_out_ch = lss_cfg.get("OUT_CHANNELS", 80)

            self.camera_backbone = CameraBackbone(
                pretrained=bb_cfg.get("PRETRAINED", True),
                fpn_out_channels=bb_cfg.get("FPN_OUT_CHANNELS", 256),
                out_channels=cam_out_ch,
                frozen_stages=bb_cfg.get("FROZEN_STAGES", 1),
            )

            self.lss_transform = LSSTransform(
                in_channels=cam_out_ch,
                out_channels=lss_out_ch,
                image_size=tuple(image_size),
                feature_size=tuple(feature_size),
                xbound=lss_cfg.get("XBOUND", [0.0, 69.12, 0.16]),
                ybound=lss_cfg.get("YBOUND", [-39.68, 39.68, 0.16]),
                zbound=lss_cfg.get("ZBOUND", [-3.0, 1.0, 4.0]),
                dbound=lss_cfg.get("DBOUND", [1.0, 45.0, 0.5]),
            )

            # --- Fusion module ---
            fusion_cfg = model_cfg.get("FUSION", {})
            self.bev_fuser = ConvFuser(
                in_channels_camera=fusion_cfg.get("IN_CHANNELS_CAMERA", lss_out_ch),
                in_channels_lidar=fusion_cfg.get("IN_CHANNELS_LIDAR", 128),
                out_channels=fusion_cfg.get("OUT_CHANNELS", 256),
            )
        else:
            self.camera_backbone = None
            self.lss_transform = None
            self.bev_fuser = None

        # --- LiDAR stream (OpenPCDet modules) ---
        # build_networks() builds VFE, backbone_3d, map_to_bev, backbone_2d,
        # dense_head, point_head, roi_head based on model_cfg
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        """Forward pass.

        In training mode, returns (ret_dict with 'loss', tb_dict, disp_dict).
        In eval mode, returns (pred_dicts, recall_dicts).

        Args:
            batch_dict: Dictionary with keys like 'points', 'voxels',
                'images', 'camera_intrinsics', 'camera2lidar', etc.

        Returns:
            Training: (loss_dict, tensorboard_dict, display_dict)
            Inference: (prediction_dicts, recall_dicts)
        """
        # --- LiDAR stream via OpenPCDet module_list ---
        # The module_list includes: vfe, backbone_3d/map_to_bev, backbone_2d
        # We run them sequentially but intercept before the detection head
        # to inject camera features.
        for i, cur_module in enumerate(self.module_list):
            # Check if this is the dense_head — inject fusion before it
            module_name = cur_module.__class__.__name__
            if "Head" in module_name and self.camera_enabled:
                # At this point batch_dict['spatial_features_2d'] has LiDAR BEV
                batch_dict = self._fuse_camera(batch_dict)

            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {"loss": loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def _fuse_camera(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run camera encoder + LSS + fusion and update batch_dict.

        Args:
            batch_dict: Must contain:
                - 'images': [B, 3, H, W] RGB images
                - 'camera_intrinsics': [B, 3, 3] or [B, 4, 4]
                - 'camera2lidar': [B, 4, 4]
                - 'spatial_features_2d': [B, C_lidar, H_bev, W_bev]

        Returns:
            Updated batch_dict with fused 'spatial_features_2d'.
        """
        images = batch_dict.get("images", None)
        if images is None:
            logger.warning("No images in batch_dict; skipping camera fusion")
            return batch_dict

        intrinsics = batch_dict["camera_intrinsics"]
        cam2lidar = batch_dict["camera2lidar"]

        # Camera backbone: [B, 3, H, W] → [B, C, H/8, W/8]
        cam_features = self.camera_backbone(images)

        # LSS view transform: [B, C, H/8, W/8] → [B, C_out, nx, ny]
        camera_bev = self.lss_transform(cam_features, intrinsics, cam2lidar)

        # Fuse with LiDAR BEV
        lidar_bev = batch_dict["spatial_features_2d"]
        fused_bev = self.bev_fuser(camera_bev, lidar_bev)

        # Replace the BEV features with fused features
        batch_dict["spatial_features_2d"] = fused_bev

        return batch_dict

    def get_training_loss(self):
        """Aggregate losses from all heads."""
        disp_dict = {}
        loss = 0
        tb_dict = {}

        if hasattr(self, "dense_head") and self.dense_head is not None:
            loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
            loss += loss_rpn
            tb_dict.update(tb_dict_rpn)

        if hasattr(self, "point_head") and self.point_head is not None:
            loss_point, tb_dict_point = self.point_head.get_loss()
            loss += loss_point
            tb_dict.update(tb_dict_point)

        if hasattr(self, "roi_head") and self.roi_head is not None:
            loss_rcnn, tb_dict_rcnn = self.roi_head.get_loss()
            loss += loss_rcnn
            tb_dict.update(tb_dict_rcnn)

        return loss, tb_dict, disp_dict
