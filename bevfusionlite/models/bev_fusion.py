"""
BEV Feature Fusion Module.

Fuses camera BEV features and LiDAR BEV features via channel concatenation
followed by convolutional layers. This is the simplest and most robust
fusion strategy, following the BEVFusion (MIT) design.

Camera BEV [B, C_cam, H, W]  ──┐
                                ├── Cat → Conv → BN → ReLU → Conv → BN → ReLU
LiDAR BEV  [B, C_lidar, H, W] ─┘
                                     → Fused BEV [B, C_out, H, W]
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvFuser(nn.Module):
    """Channel concatenation + convolutional fusion for BEV features.

    Concatenates camera and LiDAR BEV features along the channel dimension,
    then applies two conv layers to produce a fused representation.

    Args:
        in_channels_camera: Number of camera BEV feature channels.
        in_channels_lidar: Number of LiDAR BEV feature channels.
        out_channels: Number of output fused feature channels.
        mid_channels: Number of intermediate channels (default: out_channels).
    """

    def __init__(
        self,
        in_channels_camera: int = 80,
        in_channels_lidar: int = 128,
        out_channels: int = 256,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        total_in = in_channels_camera + in_channels_lidar

        self.fusion = nn.Sequential(
            nn.Conv2d(total_in, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.in_channels_camera = in_channels_camera
        self.in_channels_lidar = in_channels_lidar
        self.out_channels = out_channels

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        camera_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse camera and LiDAR BEV features.

        Both inputs must have the same spatial dimensions (H, W).
        If they differ, the camera BEV is bilinearly interpolated to
        match the LiDAR BEV spatial size.

        Args:
            camera_bev: Camera BEV features [B, C_cam, H_cam, W_cam].
            lidar_bev: LiDAR BEV features [B, C_lidar, H_lidar, W_lidar].

        Returns:
            Fused BEV features [B, C_out, H, W].
        """
        # Align spatial dimensions if needed
        if camera_bev.shape[2:] != lidar_bev.shape[2:]:
            camera_bev = nn.functional.interpolate(
                camera_bev,
                size=lidar_bev.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Concatenate along channel dimension
        fused = torch.cat([camera_bev, lidar_bev], dim=1)

        # Apply fusion convolutions
        fused = self.fusion(fused)

        return fused
