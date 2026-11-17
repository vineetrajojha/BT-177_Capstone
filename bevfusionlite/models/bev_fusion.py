"""
BEV Feature Fusion Module supporting Multi-Modal and Ablation Configurations.

Fusion Types:
  - 'concat_2conv': Channel concatenation + 2x Conv(3x3)-BN-ReLU (Reference)
  - 'concat_1x1': Channel concatenation + 1x Conv(1x1)-BN-ReLU (Lightweight ablation)
  - 'channel_attention': Squeeze-and-Excitation / Channel-Attention over concatenated BEV

Fusion Modes:
  - 'fusion': Full camera + LiDAR fusion
  - 'camera_only': Zeroes out or bypasses LiDAR features (Camera-only ablation)
  - 'lidar_only': Zeroes out or bypasses Camera features (LiDAR-only ablation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid_ch = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.fc(x)
        return x * attn


class ConvFuser(nn.Module):
    """Configurable BEV Fusion Module.

    Args:
        in_channels_camera: Number of camera BEV feature channels (e.g. 64 or 128).
        in_channels_lidar: Number of LiDAR BEV feature channels (e.g. 32 or 64).
        out_channels: Output fused BEV feature channels (e.g. 128 or 256).
        fusion_type: 'concat_2conv', 'concat_1x1', or 'channel_attention'.
        fusion_mode: 'fusion', 'camera_only', or 'lidar_only'.
    """

    def __init__(
        self,
        in_channels_camera: int = 64,
        in_channels_lidar: int = 64,
        out_channels: int = 128,
        fusion_type: str = "concat_2conv",
        fusion_mode: str = "fusion",
    ):
        super().__init__()
        self.in_channels_camera = in_channels_camera
        self.in_channels_lidar = in_channels_lidar
        self.out_channels = out_channels
        self.fusion_type = fusion_type.lower()
        self.fusion_mode = fusion_mode.lower()

        total_in = in_channels_camera + in_channels_lidar

        if self.fusion_type == "concat_2conv":
            self.fusion_block = nn.Sequential(
                nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif self.fusion_type == "concat_1x1":
            self.fusion_block = nn.Sequential(
                nn.Conv2d(total_in, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif self.fusion_type == "channel_attention":
            self.fusion_block = nn.Sequential(
                nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                ChannelAttention(out_channels, reduction=4),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        # Fallback projection for single-modality modes
        self.cam_proj = nn.Sequential(
            nn.Conv2d(in_channels_camera, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) if in_channels_camera != out_channels else nn.Identity()

        self.lidar_proj = nn.Sequential(
            nn.Conv2d(in_channels_lidar, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) if in_channels_lidar != out_channels else nn.Identity()

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
        camera_bev: Optional[torch.Tensor],
        lidar_bev: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fuses camera and LiDAR BEV representations.

        Args:
            camera_bev: [B, C_cam, H, W] camera feature map (or None).
            lidar_bev: [B, C_lidar, H, W] LiDAR feature map (or None).

        Returns:
            fused_bev: [B, C_out, H, W] fused feature map.
        """
        # Handle single-modality ablation modes
        if self.fusion_mode == "camera_only" or lidar_bev is None:
            assert camera_bev is not None, "camera_bev must be provided in camera_only mode"
            return self.cam_proj(camera_bev)

        if self.fusion_mode == "lidar_only" or camera_bev is None:
            assert lidar_bev is not None, "lidar_bev must be provided in lidar_only mode"
            return self.lidar_proj(lidar_bev)

        # Align spatial dimensions if necessary
        if camera_bev.shape[2:] != lidar_bev.shape[2:]:
            camera_bev = F.interpolate(
                camera_bev,
                size=lidar_bev.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Concatenate along channel dimension
        fused = torch.cat([camera_bev, lidar_bev], dim=1)
        return self.fusion_block(fused)
