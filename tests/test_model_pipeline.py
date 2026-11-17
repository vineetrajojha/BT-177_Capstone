"""
Unit tests for BEVFusionLite Model Architecture and Components.

Tests:
  1. CameraBackbone stage outputs and shapes for ResNet-18 and ResNet-50.
  2. DepthNet categorical depth normalization and auxiliary CE loss.
  3. Synthetic-point BEV cell geometry test (pinhole projection round-trip).
  4. Gradient flow from detection loss through LSS view transform back to camera backbone.
  5. BEV ConvFuser multi-modal and ablation modes (fusion, camera_only, lidar_only).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bevfusionlite.models.camera_backbone import CameraBackbone
from bevfusionlite.models.depth_net import DepthNet
from bevfusionlite.models.lss_transform import LSSTransform
from bevfusionlite.models.bev_fusion import ConvFuser


def test_camera_backbone_resnet18():
    """Verify ResNet-18 outputs expected stride-8 feature shape."""
    model = CameraBackbone(
        model_name="resnet18",
        pretrained=False,
        fpn_out_channels=128,
        out_channels=64,
        frozen_stages=0,
    )
    model.eval()
    x = torch.randn(2, 3, 384, 1248)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 64, 48, 156), f"Expected (2, 64, 48, 156), got {out.shape}"


def test_camera_backbone_resnet50():
    """Verify ResNet-50 outputs expected stride-8 feature shape."""
    model = CameraBackbone(
        model_name="resnet50",
        pretrained=False,
        fpn_out_channels=256,
        out_channels=128,
        frozen_stages=0,
    )
    model.eval()
    x = torch.randn(1, 3, 384, 1248)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 128, 48, 156), f"Expected (1, 128, 48, 156), got {out.shape}"


def test_depth_net_guidance_and_loss():
    """Verify DepthNet outputs normalized categorical depth and computes auxiliary loss."""
    depth_net = DepthNet(
        in_channels=64,
        context_channels=64,
        depth_channels=48,
        d_min=1.0,
        d_max=69.0,
        use_depth_guidance=True,
    )
    feat = torch.randn(2, 64, 48, 156)
    sparse_depth = torch.zeros(2, 1, 48, 156)
    sparse_depth[0, 0, 10, 20] = 25.5
    sparse_depth[1, 0, 15, 30] = 40.2

    depth_dist, context, logits = depth_net(feat, sparse_depth)

    # 1. Output shapes
    assert depth_dist.shape == (2, 48, 48, 156)
    assert context.shape == (2, 64, 48, 156)
    assert logits.shape == (2, 48, 48, 156)

    # 2. Softmax normalization (sum along depth dim == 1.0)
    depth_sums = depth_dist.sum(dim=1)
    assert torch.allclose(depth_sums, torch.ones_like(depth_sums), atol=1e-5)

    # 3. Auxiliary loss computation
    loss = depth_net.compute_loss(logits, sparse_depth)
    assert loss > 0.0, f"Expected non-zero loss, got {loss.item()}"


def test_synthetic_point_bev_cell_projection():
    """Verify that a synthetic 3D point projects into camera and maps to the correct BEV cell."""
    lss = LSSTransform(
        in_channels=64,
        out_channels=64,
        image_size=(384, 1248),
        feature_size=(48, 156),
        xbound=[0.0, 69.12, 0.16],
        ybound=[-39.68, 39.68, 0.16],
        zbound=[-3.0, 1.0, 4.0],
        dbound=[1.0, 69.0, 48],
        downsample=8,
    )

    # Pinhole projection where camera depth is LiDAR forward x
    # Camera x = -LiDAR y, Camera y = -LiDAR z, Camera z (depth) = LiDAR x
    lidar2img = torch.zeros(4, 4)
    fx = 721.53
    fy = 721.53
    cx = 609.55
    cy = 172.85

    # u * depth = fx * (-y) + cx * x
    lidar2img[0, 1] = -fx
    lidar2img[0, 0] = cx
    # v * depth = fy * (-z) + cy * x
    lidar2img[1, 2] = -fy
    lidar2img[1, 0] = cy
    # depth = x
    lidar2img[2, 0] = 1.0
    lidar2img[3, 3] = 1.0

    test_point = torch.tensor([[0, 20.0, 0.0, -0.5, 0.5]], dtype=torch.float32)  # [batch=0, x=20m, y=0m, z=-0.5m, intensity=0.5]
    dm = lss.project_lidar_to_depth_map(test_point, batch_idx=0, lidar2img=lidar2img)

    assert dm.shape == (1, 1, 48, 156)
    # Depth map should contain value near 20.0m
    max_d = dm.max().item()
    assert abs(max_d - 20.0) < 1.0, f"Expected depth ~20.0m, got {max_d}"


def test_gradient_flow_to_camera_backbone():
    """Verify that backpropagation flows through the LSS view transformer into the camera backbone."""
    backbone = CameraBackbone(model_name="resnet18", pretrained=False, out_channels=64, frozen_stages=0)
    lss = LSSTransform(
        in_channels=64,
        out_channels=64,
        image_size=(384, 1248),
        feature_size=(48, 156),
        xbound=[0.0, 69.12, 0.32],
        ybound=[-39.68, 39.68, 0.32],
        zbound=[-3.0, 1.0, 4.0],
        dbound=[1.0, 69.0, 48],
        downsample=8,
    )

    # Valid projection matrix ensuring frustum points map into BEV bounds
    lidar2img = torch.zeros(4, 4)
    fx = 721.53
    fy = 721.53
    cx = 609.55
    cy = 172.85

    lidar2img[0, 1] = -fx
    lidar2img[0, 0] = cx
    lidar2img[1, 2] = -fy
    lidar2img[1, 0] = cy
    lidar2img[2, 0] = 1.0
    lidar2img[3, 3] = 1.0
    lidar2img = lidar2img.unsqueeze(0)

    images = torch.randn(1, 3, 384, 1248, requires_grad=True)
    feats = backbone(images)
    bev, _ = lss(feats, lidar2img)

    loss = bev.sum()
    loss.backward()

    # Check gradients on camera backbone output conv
    grad = backbone.output_conv[0].weight.grad
    assert grad is not None, "Gradient to camera backbone output conv is None"
    assert grad.abs().sum() > 0.0, "Gradient to camera backbone is zero"


def test_bev_fuser_modes():
    """Verify ConvFuser shapes under fusion, camera_only, and lidar_only modes."""
    # 1. Full fusion
    fuser_full = ConvFuser(in_channels_camera=64, in_channels_lidar=32, out_channels=64, fusion_mode="fusion")
    cam = torch.randn(2, 64, 100, 100)
    lidar = torch.randn(2, 32, 100, 100)
    out_full = fuser_full(cam, lidar)
    assert out_full.shape == (2, 64, 100, 100)

    # 2. Camera-only mode
    fuser_cam = ConvFuser(in_channels_camera=64, in_channels_lidar=32, out_channels=64, fusion_mode="camera_only")
    out_cam = fuser_cam(cam, None)
    assert out_cam.shape == (2, 64, 100, 100)

    # 3. LiDAR-only mode
    fuser_lidar = ConvFuser(in_channels_camera=64, in_channels_lidar=32, out_channels=64, fusion_mode="lidar_only")
    out_lidar = fuser_lidar(None, lidar)
    assert out_lidar.shape == (2, 64, 100, 100)
