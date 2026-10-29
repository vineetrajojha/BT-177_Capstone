"""
Depth distribution prediction network for LSS view transform.

Predicts a categorical depth distribution for each spatial location in
the image feature map. This distribution is used by the LSS view transformer
to "lift" 2D image features into 3D space.

Reference: Philion & Fidler, "Lift, Splat, Shoot", ECCV 2020.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DepthNet(nn.Module):
    """Predicts per-pixel depth distributions and context features.

    For each spatial location in the input feature map, this module predicts:
    1. A categorical depth distribution over D bins (softmax-normalized).
    2. A context feature vector of dimension C_out.

    The outer product of depth distribution × context features yields a
    frustum of features that the LSS view transformer splatters into BEV.

    Args:
        in_channels: Number of input feature channels from the camera backbone.
        mid_channels: Number of intermediate channels in the prediction head.
        depth_channels: Number of depth bins (D).
        context_channels: Number of output context feature channels (C_out).
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 256,
        depth_channels: int = 88,
        context_channels: int = 80,
    ):
        super().__init__()
        self.depth_channels = depth_channels
        self.context_channels = context_channels

        # Shared feature head
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Depth prediction branch
        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, bias=True),
        )

        # Context feature branch
        self.context_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, context_channels, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict depth distributions and context features.

        Args:
            features: Image features of shape [B, C_in, H_feat, W_feat].

        Returns:
            depth: Depth distribution of shape [B, D, H_feat, W_feat].
                   Softmax-normalized along the depth dimension.
            context: Context features of shape [B, C_out, H_feat, W_feat].
        """
        x = self.reduce_conv(features)

        depth_logits = self.depth_conv(x)  # [B, D, H, W]
        depth = depth_logits.softmax(dim=1)

        context = self.context_conv(x)  # [B, C_out, H, W]

        return depth, context
