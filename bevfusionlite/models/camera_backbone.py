"""
Camera Backbone: ResNet-18 + FPN for image feature extraction.

Extracts multi-scale features from RGB images using an ImageNet-pretrained
ResNet-18 backbone and a lightweight Feature Pyramid Network (FPN) neck.

Output: Single-scale feature map at 1/8 resolution for the LSS view transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional


class FPN(nn.Module):
    """Lightweight Feature Pyramid Network.

    Takes multi-scale features from a backbone (e.g., C2, C3, C4, C5 from ResNet)
    and produces a single unified feature map via lateral connections and top-down
    pathway fusion.

    Args:
        in_channels_list: List of channel dimensions from backbone stages.
            E.g., [64, 128, 256, 512] for ResNet-18.
        out_channels: Number of output channels for all FPN levels.
        extra_blocks: If True, adds an extra conv block on top of the last FPN level.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        extra_blocks: bool = False,
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            )
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            )

        self.extra_blocks = extra_blocks
        if extra_blocks:
            self.extra_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through FPN.

        Args:
            features: List of feature tensors from backbone stages,
                ordered from finest (highest resolution) to coarsest.

        Returns:
            List of FPN output feature tensors, same order as input.
        """
        assert len(features) == len(self.lateral_convs), (
            f"Expected {len(self.lateral_convs)} feature maps, got {len(features)}"
        )

        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Output convolutions (anti-aliasing after upsampling)
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        if self.extra_blocks:
            outputs.append(self.extra_conv(outputs[-1]))

        return outputs


class CameraBackbone(nn.Module):
    """ResNet-18 + FPN camera feature extractor.

    Extracts a dense feature map from an RGB image suitable for the
    LSS view transformer. Uses ImageNet-pretrained ResNet-18 as the
    backbone with a lightweight FPN neck.

    Args:
        pretrained: Whether to use ImageNet-pretrained weights.
        fpn_out_channels: Number of output channels from the FPN.
        out_channels: Final output channels (after reduction conv).
        frozen_stages: Number of backbone stages to freeze (0-4).
            0 means no freezing, 1 means freeze stem+layer1, etc.
    """

    # ResNet-18 stage output channels
    STAGE_CHANNELS = [64, 128, 256, 512]

    def __init__(
        self,
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        out_channels: int = 256,
        frozen_stages: int = 1,
    ):
        super().__init__()

        # Load ResNet-18 backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)

        # Extract backbone stages
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # stride 4, 64 channels
        self.layer2 = resnet.layer2  # stride 8, 128 channels
        self.layer3 = resnet.layer3  # stride 16, 256 channels
        self.layer4 = resnet.layer4  # stride 32, 512 channels

        # FPN neck
        self.fpn = FPN(
            in_channels_list=self.STAGE_CHANNELS,
            out_channels=fpn_out_channels,
        )

        # Reduce to single output scale at stride 8
        # We take FPN level corresponding to layer2 (stride 8)
        self.output_conv = nn.Sequential(
            nn.Conv2d(fpn_out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze backbone stages up to frozen_stages."""
        if self.frozen_stages >= 1:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False

        stages = [self.layer2, self.layer3, self.layer4]
        for i, stage in enumerate(stages, start=2):
            if i <= self.frozen_stages:
                for param in stage.parameters():
                    param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from RGB images.

        Args:
            images: Input tensor of shape [B, 3, H, W], normalized RGB images.

        Returns:
            Feature tensor of shape [B, out_channels, H/8, W/8].
        """
        # Backbone forward
        x = self.stem(images)
        c1 = self.layer1(x)   # [B, 64,  H/4,  W/4]
        c2 = self.layer2(c1)  # [B, 128, H/8,  W/8]
        c3 = self.layer3(c2)  # [B, 256, H/16, W/16]
        c4 = self.layer4(c3)  # [B, 512, H/32, W/32]

        # FPN
        fpn_outs = self.fpn([c1, c2, c3, c4])

        # Take stride-8 level (index 1, corresponding to c2 resolution)
        feat = fpn_outs[1]  # [B, fpn_out_channels, H/8, W/8]

        # Output projection
        feat = self.output_conv(feat)  # [B, out_channels, H/8, W/8]

        return feat
