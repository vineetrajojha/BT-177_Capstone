"""
LiDAR-Depth-Guided Depth Prediction Network for BEV View Transformation.

Key Mechanisms:
  1. Integrates sparse projected LiDAR depth maps (min-pooled to stride 8).
  2. Input to depth head: Image feature + depth map channel + valid mask channel.
  3. D=48 depth bins spanning [1.0, 69.0] meters.
  4. Softmax-normalized categorical depth distribution.
  5. Auxiliary depth classification cross-entropy loss (weight 0.1) on valid LiDAR points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DepthNet(nn.Module):
    """Predicts categorical depth distributions and context features with LiDAR guidance.

    Args:
        in_channels: Image feature channels from camera backbone.
        context_channels: Output context feature channels (C_out).
        depth_channels: Number of depth discretization bins (default: 48).
        d_min: Minimum depth range in meters (default: 1.0).
        d_max: Maximum depth range in meters (default: 69.0).
        use_depth_guidance: Whether to condition depth predictions on projected sparse LiDAR depth.
        aux_loss_weight: Weight for auxiliary depth cross-entropy loss (default: 0.1).
    """

    def __init__(
        self,
        in_channels: int = 64,
        context_channels: int = 64,
        depth_channels: int = 48,
        d_min: float = 1.0,
        d_max: float = 69.0,
        use_depth_guidance: bool = True,
        aux_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.d_min = d_min
        self.d_max = d_max
        self.d_step = (d_max - d_min) / depth_channels
        self.use_depth_guidance = use_depth_guidance
        self.aux_loss_weight = aux_loss_weight

        # When depth guidance is enabled, input is image feature (in_channels) + depth (1) + mask (1)
        depth_in_ch = (in_channels + 2) if use_depth_guidance else in_channels

        # Shared reduction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Context branch
        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels, context_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(context_channels, context_channels, kernel_size=1, bias=True),
        )

        # Depth branch
        self.depth_conv = nn.Sequential(
            nn.Conv2d(depth_in_ch, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, depth_channels, kernel_size=1, bias=True),
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

    def forward(
        self,
        features: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts depth distributions and context features.

        Args:
            features: [B*N, C, H_f, W_f] camera feature maps.
            depth_map: [B*N, 1, H_f, W_f] sparse projected LiDAR depth (meters), or None.

        Returns:
            depth_dist: [B*N, D, H_f, W_f] softmax-normalized depth probabilities.
            context: [B*N, C_out, H_f, W_f] context feature vectors.
            depth_logits: [B*N, D, H_f, W_f] raw unnormalized logits for loss computation.
        """
        shared_feat = self.shared_conv(features)
        context = self.context_conv(shared_feat)

        if self.use_depth_guidance:
            if depth_map is not None:
                mask = (depth_map > 0.0).float()
                norm_depth = torch.clamp(depth_map / self.d_max, 0.0, 1.0)
            else:
                B_cur, _, H_cur, W_cur = features.shape
                norm_depth = torch.zeros((B_cur, 1, H_cur, W_cur), device=features.device, dtype=features.dtype)
                mask = torch.zeros((B_cur, 1, H_cur, W_cur), device=features.device, dtype=features.dtype)
            depth_input = torch.cat([shared_feat, norm_depth, mask], dim=1)
        else:
            depth_input = shared_feat

        depth_logits = self.depth_conv(depth_input)
        depth_dist = F.softmax(depth_logits, dim=1)

        return depth_dist, context, depth_logits

    def compute_loss(
        self,
        depth_logits: torch.Tensor,
        gt_depth_map: torch.Tensor,
    ) -> torch.Tensor:
        """Computes auxiliary cross-entropy loss on valid LiDAR depth pixels.

        Args:
            depth_logits: [B*N, D, H_f, W_f] predicted depth logits.
            gt_depth_map: [B*N, 1, H_f, W_f] ground truth sparse depth map.

        Returns:
            Scalar cross-entropy loss weighted by aux_loss_weight (0.1).
        """
        if gt_depth_map is None:
            return depth_logits.new_zeros(1)[0]

        valid_mask = (gt_depth_map > self.d_min) & (gt_depth_map < self.d_max)
        if valid_mask.sum() == 0:
            return depth_logits.new_zeros(1)[0]

        # Convert continuous metric depth to discrete bin indices
        gt_bins = ((gt_depth_map - self.d_min) / self.d_step).long()
        gt_bins = torch.clamp(gt_bins, 0, self.depth_channels - 1)

        # Squeeze channel dim: [B*N, H_f, W_f]
        target = gt_bins.squeeze(1)
        mask = valid_mask.squeeze(1)

        logits_valid = depth_logits.permute(0, 2, 3, 1)[mask]  # [N_valid, D]
        target_valid = target[mask]                           # [N_valid]

        ce_loss = F.cross_entropy(logits_valid, target_valid, reduction="mean")
        return ce_loss * self.aux_loss_weight
