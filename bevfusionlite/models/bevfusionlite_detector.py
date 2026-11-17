"""
BEVFusionLite Detector — Unified Architecture for Heavy Baseline & Light Models.

Configurable via YAML for:
  - Heavy Baseline: ResNet-50 camera backbone (128 image ch), 0.16m LiDAR (64 LiDAR ch)
  - Light Model: ResNet-18 camera backbone (64 image ch), 0.32m LiDAR (32 LiDAR ch)
  - Ablation Modes: Camera-only, LiDAR-only, MobileNetV3-Large, alternate fusion blocks.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from pcdet.models.detectors.detector3d_template import Detector3DTemplate
except ImportError:
    Detector3DTemplate = nn.Module
    logger.warning("OpenPCDet not found; using nn.Module as base class")

from .camera_backbone import CameraBackbone
from .lss_transform import LSSTransform
from .bev_fusion import ConvFuser


class BEVFusionLiteDetector(Detector3DTemplate):
    """Unified Camera-LiDAR BEV Fusion 3D Object Detector.

    Integrates:
      1. PointPillars LiDAR branch (VFE + PointPillarScatter)
      2. Multi-backbone camera branch (ResNet-50 / ResNet-18 / MobileNetV3 + FPN)
      3. LiDAR-depth-guided LSS View Transformer with auxiliary depth supervision
      4. Configurable ConvFuser
      5. 2D BEV Backbone + AnchorHeadSingle detection head
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        # 1. Camera Stream Configuration
        camera_cfg = model_cfg.get("CAMERA", None)
        self.camera_enabled = camera_cfg is not None and camera_cfg.get("ENABLED", True)

        if self.camera_enabled:
            bb_cfg = camera_cfg.get("BACKBONE", {})
            lss_cfg = camera_cfg.get("LSS", {})

            backbone_name = bb_cfg.get("NAME", "resnet18")
            fpn_out_ch = bb_cfg.get("FPN_OUT_CHANNELS", 128)
            cam_out_ch = bb_cfg.get("OUT_CHANNELS", 64)
            lss_out_ch = lss_cfg.get("OUT_CHANNELS", cam_out_ch)

            image_size = camera_cfg.get("IMAGE_SIZE", [384, 1248])
            feature_size = camera_cfg.get("FEATURE_SIZE", [48, 156])

            self.camera_backbone = CameraBackbone(
                model_name=backbone_name,
                pretrained=bb_cfg.get("PRETRAINED", True),
                fpn_out_channels=fpn_out_ch,
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
                dbound=lss_cfg.get("DBOUND", [1.0, 69.0, 48]),
                downsample=lss_cfg.get("DOWNSAMPLE", 8),
                use_depth_guidance=lss_cfg.get("USE_DEPTH_GUIDANCE", True),
            )
        else:
            self.camera_backbone = None
            self.lss_transform = None
            lss_out_ch = 0

        # 2. LiDAR & BEV Feature Channels
        lidar_out_ch = model_cfg.get("VFE", {}).get("NUM_FILTERS", [64])[-1]
        fusion_cfg = model_cfg.get("FUSION", {})
        fused_out_ch = fusion_cfg.get("OUT_CHANNELS", 128)
        fusion_type = fusion_cfg.get("TYPE", "concat_2conv")
        fusion_mode = fusion_cfg.get("MODE", "fusion")

        # 3. Fusion Module
        self.fuser = ConvFuser(
            in_channels_camera=lss_out_ch,
            in_channels_lidar=lidar_out_ch,
            out_channels=fused_out_ch,
            fusion_type=fusion_type,
            fusion_mode=fusion_mode,
        )

        # 4. Standard OpenPCDet LiDAR Pipeline & Heads
        self.module_list = self.build_networks()

    def build_networks(self):
        # OpenPCDet detector template building (VFE, MAP_TO_BEV, BACKBONE_2D, DENSE_HEAD)
        model_info_dict = {
            "module_list": [],
            "num_rawpoint_features": self.dataset.point_feature_encoder.num_point_features,
            "num_point_features": 0,
            "grid_size": self.dataset.grid_size,
            "voxel_size": self.dataset.voxel_size,
            "point_cloud_range": self.dataset.point_cloud_range,
        }

        # Build VFE
        if self.model_cfg.get("VFE", None):
            vfe_module, model_info_dict = self.build_vfe(model_info_dict=model_info_dict)
            model_info_dict["module_list"].append(vfe_module)

        # Build Map to BEV (Scatter)
        if self.model_cfg.get("MAP_TO_BEV", None):
            map_to_bev_module, model_info_dict = self.build_map_to_bev(model_info_dict=model_info_dict)
            model_info_dict["module_list"].append(map_to_bev_module)

        # Build 2D BEV Backbone
        if self.model_cfg.get("BACKBONE_2D", None):
            backbone_2d_module, model_info_dict = self.build_backbone_2d(model_info_dict=model_info_dict)
            model_info_dict["module_list"].append(backbone_2d_module)

        # Build Dense Head
        if self.model_cfg.get("DENSE_HEAD", None):
            dense_head_module, model_info_dict = self.build_dense_head(model_info_dict=model_info_dict)
            model_info_dict["module_list"].append(dense_head_module)

        return nn.ModuleList(model_info_dict["module_list"])

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass across camera, LiDAR, and fusion streams."""
        # 1. LiDAR stream (VFE + Scatter)
        # Module 0: VFE, Module 1: Scatter
        for cur_module in self.module_list[:2]:
            batch_dict = cur_module(batch_dict)

        lidar_bev = batch_dict.get("spatial_features", None)

        # 2. Camera stream
        camera_bev = None
        aux_depth_loss = None
        if self.camera_enabled and "images" in batch_dict:
            images = batch_dict["images"]
            lidar2img = batch_dict.get("lidar2img", None)
            raw_points = batch_dict.get("points", None)

            # Camera backbone feature extraction
            cam_feats = self.camera_backbone(images)

            # LiDAR-depth-guided LSS view transform
            if lidar2img is not None:
                camera_bev, aux_depth_loss = self.lss_transform(
                    image_features=cam_feats,
                    lidar2img=lidar2img,
                    raw_points=raw_points,
                )

        # 3. BEV Fusion
        fused_bev = self.fuser(camera_bev=camera_bev, lidar_bev=lidar_bev)
        batch_dict["spatial_features"] = fused_bev

        # 4. Downstream 2D BEV Backbone & Dense Head
        for cur_module in self.module_list[2:]:
            batch_dict = cur_module(batch_dict)

        # Add auxiliary depth loss to batch_dict during training
        if self.training and aux_depth_loss is not None:
            batch_dict["aux_depth_loss"] = aux_depth_loss

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            if aux_depth_loss is not None:
                loss = loss + aux_depth_loss
                tb_dict["loss_aux_depth"] = aux_depth_loss.item()

            ret_dict = {"loss": loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            "loss_rpn": loss_rpn.item(),
            **tb_dict
        }
        loss = loss_rpn
        return loss, tb_dict, disp_dict
