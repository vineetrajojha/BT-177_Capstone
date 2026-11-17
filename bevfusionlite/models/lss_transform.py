"""
LiDAR-Depth-Guided Lift-Splat-Shoot (LSS) View Transformer.

Converts perspective camera features into Bird's-Eye-View (BEV) features:
  1. Projects raw LiDAR points onto camera image plane to form a sparse depth map.
  2. Min-pools sparse depth to feature stride (stride 8) preserving foreground depth.
  3. Predicts categorical depth distribution over D=48 bins guided by LiDAR depth.
  4. Lifts 2D context features into 3D frustum using outer product.
  5. Splats 3D frustum points into BEV space via pure PyTorch scatter-add.
  6. Supports N cameras (N=1 for KITTI, N=6 for nuScenes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .depth_net import DepthNet


class LSSTransform(nn.Module):
    """LiDAR-Depth-Guided View Transformer.

    Args:
        in_channels: Input channels from camera backbone (e.g. 64 or 128).
        out_channels: Output BEV feature channels (e.g. 64 or 128).
        image_size: (H, W) input image dimensions (e.g. 384, 1248).
        feature_size: (H_feat, W_feat) stride-8 feature dimensions (e.g. 48, 156).
        xbound: [xmin, xmax, xstep] for BEV X axis (forward).
        ybound: [ymin, ymax, ystep] for BEV Y axis (cross-range).
        zbound: [zmin, zmax, zstep] for BEV Z axis (vertical height).
        dbound: [dmin, dmax, dstep] for depth discretization (default: [1.0, 69.0, 48 bins]).
        downsample: Spatial downsample factor from image to feature map (default: 8).
        use_depth_guidance: Whether to use sparse LiDAR depth maps in DepthNet.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        image_size: Tuple[int, int] = (384, 1248),
        feature_size: Tuple[int, int] = (48, 156),
        xbound: List[float] = None,
        ybound: List[float] = None,
        zbound: List[float] = None,
        dbound: List[float] = None,
        downsample: int = 8,
        use_depth_guidance: bool = True,
    ):
        super().__init__()
        # Default bounds: KITTI range x[0, 69.12], y[-39.68, 39.68], z[-3.0, 1.0]
        self.xbound = xbound or [0.0, 69.12, 0.16]
        self.ybound = ybound or [-39.68, 39.68, 0.16]
        self.zbound = zbound or [-3.0, 1.0, 4.0]
        self.dbound = dbound or [1.0, 69.0, 48]  # [d_min, d_max, num_bins]

        self.image_size = image_size
        self.feature_size = feature_size
        self.downsample = downsample
        self.out_channels = out_channels
        self.use_depth_guidance = use_depth_guidance

        # Grid dimensions in BEV
        self.nx = int(round((self.xbound[1] - self.xbound[0]) / self.xbound[2]))
        self.ny = int(round((self.ybound[1] - self.ybound[0]) / self.ybound[2]))
        self.nz = int(round((self.zbound[1] - self.zbound[0]) / self.zbound[2]))

        # Depth discretization
        self.d_min = float(self.dbound[0])
        self.d_max = float(self.dbound[1])
        self.num_bins = int(self.dbound[2])

        # DepthNet
        self.depth_net = DepthNet(
            in_channels=in_channels,
            context_channels=out_channels,
            depth_channels=self.num_bins,
            d_min=self.d_min,
            d_max=self.d_max,
            use_depth_guidance=use_depth_guidance,
        )

        # Precompute camera frustum template
        frustum = self._create_frustum()
        self.register_buffer("frustum", frustum, persistent=False)

    def _create_frustum(self) -> torch.Tensor:
        """Creates [D, H_feat, W_feat, 3] grid of camera frustum coordinates (u, v, d)."""
        H_feat, W_feat = self.feature_size
        ds = self.downsample

        # Pixel sample centers in original image coordinates
        xs = torch.linspace(ds / 2.0, self.image_size[1] - ds / 2.0, W_feat, dtype=torch.float32)
        ys = torch.linspace(ds / 2.0, self.image_size[0] - ds / 2.0, H_feat, dtype=torch.float32)
        ds_vals = torch.linspace(self.d_min, self.d_max, self.num_bins, dtype=torch.float32)

        # Meshgrid: [D, H_feat, W_feat]
        D_grid, Y_grid, X_grid = torch.meshgrid(ds_vals, ys, xs, indexing="ij")
        # Stack to [D, H_feat, W_feat, 3] where last dim is (u, v, d)
        frustum = torch.stack([X_grid, Y_grid, D_grid], dim=-1)
        return frustum

    def project_lidar_to_depth_map(
        self,
        points: torch.Tensor,
        batch_idx: int,
        lidar2img: torch.Tensor,
        cam_idx: int = 0,
    ) -> torch.Tensor:
        """Projects LiDAR points to sparse depth map at feature resolution with min-pooling.

        Args:
            points: [P, 4+] LiDAR points (x, y, z, ...), first col may be batch_idx or raw x.
            batch_idx: Batch index to filter points.
            lidar2img: [4, 4] LiDAR-to-image projection matrix.
            cam_idx: Camera index for multi-camera setups.

        Returns:
            sparse_depth_feat: [1, 1, H_feat, W_feat] min-pooled sparse depth map.
        """
        device = lidar2img.device
        H_feat, W_feat = self.feature_size
        ds = self.downsample
        H_orig, W_orig = self.image_size

        # Select points for this batch
        if points.shape[1] >= 5:
            pts_mask = (points[:, 0] == batch_idx)
            pts_xyz = points[pts_mask, 1:4]
        else:
            pts_xyz = points[:, :3]

        if pts_xyz.shape[0] == 0:
            return torch.zeros((1, 1, H_feat, W_feat), device=device, dtype=torch.float32)

        # Transform to homogeneous coords
        ones = torch.ones((pts_xyz.shape[0], 1), device=device, dtype=pts_xyz.dtype)
        pts_hom = torch.cat([pts_xyz, ones], dim=1)  # [P, 4]

        # Project: pts_img_hom = pts_hom @ lidar2img.T
        pts_img_hom = pts_hom @ lidar2img.T.to(pts_hom.dtype)  # [P, 4]
        depth = pts_img_hom[:, 2]

        valid = (depth > self.d_min) & (depth < self.d_max)
        pts_img_hom = pts_img_hom[valid]
        depth = depth[valid]

        if pts_img_hom.shape[0] == 0:
            return torch.zeros((1, 1, H_feat, W_feat), device=device, dtype=torch.float32)

        u = pts_img_hom[:, 0] / depth
        v = pts_img_hom[:, 1] / depth

        in_bounds = (u >= 0) & (u < W_orig) & (v >= 0) & (v < H_orig)
        u = u[in_bounds]
        v = v[in_bounds]
        depth = depth[in_bounds]

        if depth.shape[0] == 0:
            return torch.zeros((1, 1, H_feat, W_feat), device=device, dtype=torch.float32)

        # Quantize to feature map cells
        uf = (u / ds).long().clamp(0, W_feat - 1)
        vf = (v / ds).long().clamp(0, H_feat - 1)

        # Min-pooling: Keep minimum depth per cell
        depth_map = torch.zeros((H_feat, W_feat), device=device, dtype=torch.float32)
        # Use a high sentinel for min-pooling
        min_depth = torch.full((H_feat, W_feat), 999.0, device=device, dtype=torch.float32)

        flat_idx = vf * W_feat + uf
        # Sort by depth descending so when we scatter, smaller depths overwrite larger ones
        sorted_indices = torch.argsort(depth, descending=True)
        flat_idx_sorted = flat_idx[sorted_indices]
        depth_sorted = depth[sorted_indices]

        flat_min = min_depth.view(-1)
        flat_min.scatter_(0, flat_idx_sorted, depth_sorted)
        depth_map = torch.where(min_depth < 900.0, min_depth, torch.zeros_like(min_depth))

        return depth_map.unsqueeze(0).unsqueeze(0)

    def get_geometry(
        self,
        lidar2img: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Unprojects frustum coordinates back into the LiDAR coordinate frame.

        Args:
            lidar2img: [B, N, 4, 4] or [4, 4] transformation matrix.

        Returns:
            points_lidar: [B, N, D, H_f, W_f, 3] 3D coordinates in LiDAR frame.
        """
        D, H_feat, W_feat, _ = self.frustum.shape
        u = self.frustum[..., 0]  # [D, H_f, W_f]
        v = self.frustum[..., 1]
        d = self.frustum[..., 2]

        # Homogeneous image ray points: [u*d, v*d, d, 1]
        x_img = u * d
        y_img = v * d
        z_img = d
        ones = torch.ones_like(d)
        points_hom = torch.stack([x_img, y_img, z_img, ones], dim=-1)  # [D, H_f, W_f, 4]

        # Reshape for matrix multiplication
        points_hom = points_hom.view(-1, 4).unsqueeze(0)  # [1, D*H*W, 4]

        # Invert lidar2img matrix
        if lidar2img.dim() == 2:
            img2lidar = torch.inverse(lidar2img.float()).to(device)  # [4, 4]
            pts_lidar = (points_hom @ img2lidar.T)[0, :, :3]  # [D*H*W, 3]
            return pts_lidar.view(D, H_feat, W_feat, 3)
        else:
            # Batched: [B, N, 4, 4]
            B, N = lidar2img.shape[:2]
            img2lidar = torch.inverse(lidar2img.float()).to(device)  # [B, N, 4, 4]
            points_hom_batch = points_hom.repeat(B * N, 1, 1)        # [B*N, D*H*W, 4]
            pts_lidar = points_hom_batch @ img2lidar.view(B * N, 4, 4).permute(0, 2, 1)
            pts_lidar = pts_lidar[:, :, :3].view(B, N, D, H_feat, W_feat, 3)
            return pts_lidar

    def bev_pool_scatter(
        self,
        geom_feats: torch.Tensor,
        x_features: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates 3D lifted features into BEV grid using pure PyTorch scatter-add.

        Args:
            geom_feats: [B, N, D, H_f, W_f, 3] 3D spatial points in LiDAR frame.
            x_features: [B, N, C, D, H_f, W_f] lifted feature tensor.

        Returns:
            bev_feature: [B, C, ny, nx] Bird's-Eye-View feature map.
        """
        B, N, C, D, H_feat, W_feat = x_features.shape
        device = x_features.device

        # Reshape features to [B, C, N*D*H*W]
        feats = x_features.permute(0, 2, 1, 3, 4, 5).reshape(B, C, -1)

        # Coordinates to BEV grid indices
        gx = geom_feats[..., 0].reshape(B, -1)  # [B, N*D*H*W]
        gy = geom_feats[..., 1].reshape(B, -1)
        gz = geom_feats[..., 2].reshape(B, -1)

        # In-bounds check
        x_min, x_max, x_res = self.xbound
        y_min, y_max, y_res = self.ybound
        z_min, z_max, _ = self.zbound

        valid = (
            (gx >= x_min) & (gx < x_max) &
            (gy >= y_min) & (gy < y_max) &
            (gz >= z_min) & (gz < z_max)
        )

        ix = ((gx - x_min) / x_res).long().clamp(0, self.nx - 1)
        iy = ((gy - y_min) / y_res).long().clamp(0, self.ny - 1)

        bev_list = []
        bev_idx = iy * self.nx + ix

        for b in range(B):
            b_valid = valid[b]
            if b_valid.sum() == 0:
                bev_list.append(torch.zeros((1, C, self.ny, self.nx), device=device, dtype=x_features.dtype))
                continue

            b_idx = bev_idx[b, b_valid]
            b_feats = feats[b, :, b_valid]  # [C, N_valid]

            flat_bev = torch.zeros((C, self.ny * self.nx), device=device, dtype=x_features.dtype)
            flat_bev = flat_bev.scatter_add(1, b_idx.unsqueeze(0).expand(C, -1), b_feats)
            bev_list.append(flat_bev.view(1, C, self.ny, self.nx))

        bev_features = torch.cat(bev_list, dim=0)
        return bev_features

    def forward(
        self,
        image_features: torch.Tensor,
        lidar2img: torch.Tensor,
        raw_points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """LSS Forward Transform with LiDAR-Depth Guidance.

        Args:
            image_features: [B, N, C_in, H_f, W_f] or [B, C_in, H_f, W_f].
            lidar2img: [B, N, 4, 4] or [B, 4, 4] calibration projection matrix.
            raw_points: [P, 4+] LiDAR point cloud for sparse depth extraction.

        Returns:
            bev_map: [B, C_out, ny, nx] BEV feature map.
            depth_loss: Auxiliary depth cross-entropy loss if raw_points provided.
        """
        device = image_features.device
        if image_features.dim() == 4:
            image_features = image_features.unsqueeze(1)
        if lidar2img.dim() == 3:
            lidar2img = lidar2img.unsqueeze(1)

        B, N, C_in, H_feat, W_feat = image_features.shape

        # 1. Compute sparse depth maps from LiDAR if available
        sparse_depth_maps = []
        if self.use_depth_guidance and raw_points is not None:
            for b in range(B):
                for n in range(N):
                    dm = self.project_lidar_to_depth_map(
                        points=raw_points,
                        batch_idx=b,
                        lidar2img=lidar2img[b, n],
                        cam_idx=n,
                    )
                    sparse_depth_maps.append(dm)
            sparse_depth_tensor = torch.cat(sparse_depth_maps, dim=0)  # [B*N, 1, H_f, W_f]
        else:
            sparse_depth_tensor = None

        # 2. Predict depth distributions and context features
        img_feats_flat = image_features.view(B * N, C_in, H_feat, W_feat)
        depth_dist, context, depth_logits = self.depth_net(
            features=img_feats_flat,
            depth_map=sparse_depth_tensor,
        )

        # 3. Outer product: Lift 2D features into 3D frustum
        # context: [B*N, C_out, 1, H_f, W_f]
        # depth_dist: [B*N, 1, D, H_f, W_f]
        # lifted: [B, N, C_out, D, H_f, W_f]
        context_lift = context.unsqueeze(2)
        depth_dist_lift = depth_dist.unsqueeze(1)
        lifted = (context_lift * depth_dist_lift).view(B, N, self.out_channels, self.num_bins, H_feat, W_feat)

        # 4. Geometry unprojection to LiDAR coordinates
        geom_points = self.get_geometry(lidar2img, device=device)  # [B, N, D, H_f, W_f, 3]

        # 5. Scatter-add splatting into BEV grid
        bev_feature = self.bev_pool_scatter(geom_points, lifted)  # [B, C_out, ny, nx]

        # 6. Auxiliary depth loss
        depth_loss = None
        if self.training and sparse_depth_tensor is not None:
            depth_loss = self.depth_net.compute_loss(depth_logits, sparse_depth_tensor)

        return bev_feature, depth_loss
