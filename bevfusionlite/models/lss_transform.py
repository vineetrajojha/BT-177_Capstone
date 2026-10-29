"""
Lift-Splat-Shoot (LSS) View Transformer.

Transforms perspective camera features into Bird's-Eye-View (BEV) features
by predicting depth distributions and splatting features into a voxel grid.

Pipeline:
    1. DepthNet predicts per-pixel depth distribution + context features.
    2. Outer product creates a frustum of features in 3D camera space.
    3. Frustum points are projected into the ego/LiDAR coordinate frame
       using camera intrinsics and extrinsics.
    4. Features are accumulated (splatted) into a BEV grid via summation.

Reference: Philion & Fidler, "Lift, Splat, Shoot", ECCV 2020.

Adapted for KITTI single-camera setup with frontal FOV BEV grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .depth_net import DepthNet


class LSSTransform(nn.Module):
    """Lift-Splat-Shoot view transformer for camera → BEV projection.

    Converts perspective image features into a BEV feature map by:
    1. Predicting depth distributions at each spatial location
    2. Creating a point cloud of features (frustum) in 3D
    3. Projecting and accumulating them into a BEV grid

    Args:
        in_channels: Input feature channels from camera backbone.
        out_channels: Output BEV feature channels (= DepthNet context channels).
        image_size: (H, W) of the input images (before backbone downsampling).
        feature_size: (H_feat, W_feat) of the backbone output features.
        xbound: [xmin, xmax, xstep] for BEV X axis (forward in KITTI).
        ybound: [ymin, ymax, ystep] for BEV Y axis (left-right in KITTI).
        zbound: [zmin, zmax, zstep] for BEV Z axis (up in KITTI).
        dbound: [dmin, dmax, dstep] for depth discretization.
        downsample: Downsampling factor from image to feature map.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 80,
        image_size: Tuple[int, int] = (384, 1280),
        feature_size: Tuple[int, int] = (48, 160),
        xbound: List[float] = None,
        ybound: List[float] = None,
        zbound: List[float] = None,
        dbound: List[float] = None,
        downsample: int = 8,
    ):
        super().__init__()

        # Default KITTI-appropriate bounds
        if xbound is None:
            xbound = [0.0, 69.12, 0.16]
        if ybound is None:
            ybound = [-39.68, 39.68, 0.16]
        if zbound is None:
            zbound = [-3.0, 1.0, 4.0]
        if dbound is None:
            dbound = [1.0, 45.0, 0.5]

        self.image_size = image_size
        self.feature_size = feature_size
        self.downsample = downsample
        self.out_channels = out_channels

        # Grid configuration
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        # Compute grid dimensions
        self.nx = int((xbound[1] - xbound[0]) / xbound[2])  # BEV X cells
        self.ny = int((ybound[1] - ybound[0]) / ybound[2])  # BEV Y cells
        self.nz = int((zbound[1] - zbound[0]) / zbound[2])  # BEV Z cells (usually 1)

        # Depth bins
        self.depth_channels = int((dbound[1] - dbound[0]) / dbound[2])

        # Depth + context prediction network
        self.depth_net = DepthNet(
            in_channels=in_channels,
            mid_channels=in_channels,
            depth_channels=self.depth_channels,
            context_channels=out_channels,
        )

        # Pre-compute the frustum grid (pixel coords × depth bins)
        self.frustum = self._create_frustum()

    def _create_frustum(self) -> torch.Tensor:
        """Create a frustum of 3D points in image coordinates.

        Each point is (u_img, v_img, depth) where u,v are in original
        image pixel coordinates (not feature map coordinates).

        Returns:
            Frustum tensor of shape [D, H_feat, W_feat, 3].
        """
        H_feat, W_feat = self.feature_size
        ds = torch.arange(
            self.dbound[0], self.dbound[1], self.dbound[2], dtype=torch.float32
        )  # [D]
        D = ds.shape[0]

        # Create pixel grid at feature resolution, then scale to image resolution
        xs = torch.linspace(0, self.image_size[1] - 1, W_feat, dtype=torch.float32)
        ys = torch.linspace(0, self.image_size[0] - 1, H_feat, dtype=torch.float32)

        # meshgrid: ys along H, xs along W
        ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing="ij")  # [H_feat, W_feat]

        # Expand for depth dimension
        xs_grid = xs_grid.unsqueeze(0).expand(D, -1, -1)   # [D, H, W]
        ys_grid = ys_grid.unsqueeze(0).expand(D, -1, -1)   # [D, H, W]
        ds_grid = ds.view(-1, 1, 1).expand(-1, H_feat, W_feat)  # [D, H, W]

        frustum = torch.stack([xs_grid, ys_grid, ds_grid], dim=-1)  # [D, H, W, 3]
        return nn.Parameter(frustum, requires_grad=False)

    def _get_geometry(
        self,
        camera_intrinsics: torch.Tensor,
        camera2lidar: torch.Tensor,
    ) -> torch.Tensor:
        """Project frustum points from image coordinates to LiDAR/ego coordinates.

        Args:
            camera_intrinsics: Camera intrinsic matrix [B, 3, 3] or [B, 4, 4].
            camera2lidar: Camera-to-LiDAR extrinsic transform [B, 4, 4].

        Returns:
            Points in LiDAR frame of shape [B, D, H_feat, W_feat, 3].
        """
        B = camera_intrinsics.shape[0]
        D, H, W, _ = self.frustum.shape

        # Expand frustum for batch: [B, D, H, W, 3]
        points = self.frustum.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        # Unpack pixel coords and depth
        u = points[..., 0:1]  # [B, D, H, W, 1]
        v = points[..., 1:2]
        d = points[..., 2:3]

        # Back-project to 3D camera coordinates:
        #   X_cam = (u - cx) * d / fx
        #   Y_cam = (v - cy) * d / fy
        #   Z_cam = d
        # Extract intrinsic params
        K = camera_intrinsics[:, :3, :3]  # [B, 3, 3]
        fx = K[:, 0, 0].view(B, 1, 1, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1, 1, 1)

        x_cam = (u - cx) * d / (fx + 1e-8)
        y_cam = (v - cy) * d / (fy + 1e-8)
        z_cam = d

        # Points in camera frame: [B, D, H, W, 3]
        points_cam = torch.cat([x_cam, y_cam, z_cam], dim=-1)

        # Transform to LiDAR frame: p_lidar = R @ p_cam + t
        R = camera2lidar[:, :3, :3]  # [B, 3, 3]
        t = camera2lidar[:, :3, 3]   # [B, 3]

        # Reshape for matmul: [B, D*H*W, 3]
        points_flat = points_cam.view(B, -1, 3)
        points_lidar = torch.bmm(points_flat, R.transpose(1, 2)) + t.unsqueeze(1)

        # Reshape back: [B, D, H, W, 3]
        points_lidar = points_lidar.view(B, D, H, W, 3)

        return points_lidar

    def _voxel_pooling(
        self,
        geom: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Splat features into BEV grid by summing features in each voxel.

        Args:
            geom: 3D points in LiDAR frame [B, D, H_feat, W_feat, 3].
            features: Frustum features [B, C, D, H_feat, W_feat].

        Returns:
            BEV feature map [B, C, nx, ny].
        """
        B, C, D, H, W = features.shape

        # Compute voxel indices for each point
        # BEV grid: X maps to first spatial dim, Y maps to second
        voxel_x = ((geom[..., 0] - self.xbound[0]) / self.xbound[2]).long()
        voxel_y = ((geom[..., 1] - self.ybound[0]) / self.ybound[2]).long()

        # Validity mask: points inside the BEV grid
        valid = (
            (voxel_x >= 0) & (voxel_x < self.nx) &
            (voxel_y >= 0) & (voxel_y < self.ny)
        )  # [B, D, H, W]

        # Flatten spatial dimensions
        # features: [B, C, D, H, W] → [B, C, D*H*W]
        features_flat = features.view(B, C, -1)

        # Compute linear indices for scatter: batch_idx * nx * ny + voxel_x * ny + voxel_y
        voxel_x_flat = voxel_x.view(B, -1)  # [B, D*H*W]
        voxel_y_flat = voxel_y.view(B, -1)
        valid_flat = valid.view(B, -1)

        # Initialize output BEV grid
        bev = features.new_zeros(B, C, self.nx, self.ny)

        for b in range(B):
            mask = valid_flat[b]  # [D*H*W]
            if mask.sum() == 0:
                continue

            ix = voxel_x_flat[b][mask]  # [N_valid]
            iy = voxel_y_flat[b][mask]
            feat = features_flat[b, :, mask]  # [C, N_valid]

            # Linear index for scatter_add
            linear_idx = ix * self.ny + iy  # [N_valid]
            linear_idx = linear_idx.unsqueeze(0).expand(C, -1)  # [C, N_valid]

            bev_flat = bev[b].view(C, -1)  # [C, nx*ny]
            bev_flat.scatter_add_(1, linear_idx, feat)

        return bev

    def forward(
        self,
        image_features: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera2lidar: torch.Tensor,
    ) -> torch.Tensor:
        """Transform image features to BEV features.

        Args:
            image_features: Features from camera backbone [B, C_in, H_feat, W_feat].
            camera_intrinsics: Camera intrinsic matrix [B, 3, 3] or [B, 4, 4].
            camera2lidar: Camera-to-LiDAR extrinsic [B, 4, 4].

        Returns:
            BEV feature map [B, C_out, nx, ny].
        """
        B = image_features.shape[0]

        # 1. Predict depth distribution and context features
        depth, context = self.depth_net(image_features)
        # depth: [B, D, H, W], context: [B, C_out, H, W]

        # 2. Create frustum features via outer product
        # depth: [B, D, H, W] → [B, 1, D, H, W]
        # context: [B, C, H, W] → [B, C, 1, H, W]
        # frustum_features: [B, C, D, H, W]
        frustum_features = depth.unsqueeze(1) * context.unsqueeze(2)

        # 3. Get 3D geometry (project frustum to LiDAR frame)
        geom = self._get_geometry(camera_intrinsics, camera2lidar)

        # 4. Splat into BEV grid
        bev = self._voxel_pooling(geom, frustum_features)

        return bev
