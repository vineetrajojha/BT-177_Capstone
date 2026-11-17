"""
Camera Backbone: Unified Multi-Backbone Feature Extractor with FPN.

Supports:
  - ResNet-50 (Heavy baseline: 128 output channels)
  - ResNet-18 (Light model: 64 output channels)
  - MobileNetV3-Large (Ablation model: 64 output channels)

Extracts multi-scale features and produces a single unified stride-8 feature map
for the LiDAR-depth-guided BEV view transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional, Tuple


class FPN(nn.Module):
    """Lightweight Feature Pyramid Network (FPN).

    Takes multi-scale features from backbone stages (C2, C3, C4, C5)
    and produces lateral/top-down fused feature representations.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 128,
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
        assert len(features) == len(self.lateral_convs), (
            f"Expected {len(self.lateral_convs)} feature maps, got {len(features)}"
        )

        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        if self.extra_blocks:
            outputs.append(self.extra_conv(outputs[-1]))

        return outputs


class CameraBackbone(nn.Module):
    """Configurable Camera Backbone supporting ResNet-18, ResNet-50, and MobileNetV3.

    Args:
        model_name: 'resnet18', 'resnet50', or 'mobilenet_v3_large'.
        pretrained: Whether to load ImageNet pretrained weights.
        fpn_out_channels: Intermediate FPN channels.
        out_channels: Final projected channels at stride 8.
        frozen_stages: Number of stages to freeze during training.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        fpn_out_channels: int = 128,
        out_channels: int = 64,
        frozen_stages: int = 1,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.out_channels = out_channels
        self.frozen_stages = frozen_stages

        if self.model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            stage_channels = [64, 128, 256, 512]
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1  # stride 4
            self.layer2 = resnet.layer2  # stride 8
            self.layer3 = resnet.layer3  # stride 16
            self.layer4 = resnet.layer4  # stride 32

        elif self.model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
            stage_channels = [256, 512, 1024, 2048]
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1  # stride 4
            self.layer2 = resnet.layer2  # stride 8
            self.layer3 = resnet.layer3  # stride 16
            self.layer4 = resnet.layer4  # stride 32

        elif self.model_name in ["mobilenet_v3_large", "mobilenetv3"]:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            mobilenet = models.mobilenet_v3_large(weights=weights)
            features = mobilenet.features
            # Stride points in MobileNetV3-Large:
            # stage 1 (stride 4): indices 0..3 -> 24 ch
            # stage 2 (stride 8): indices 4..6 -> 40 ch
            # stage 3 (stride 16): indices 7..12 -> 112 ch
            # stage 4 (stride 32): indices 13..16 -> 960 ch
            self.stem = nn.Sequential(features[0], features[1], features[2], features[3])
            self.layer1 = nn.Identity()
            self.layer2 = nn.Sequential(features[4], features[5], features[6])
            self.layer3 = nn.Sequential(features[7], features[8], features[9], features[10], features[11], features[12])
            self.layer4 = nn.Sequential(features[13], features[14], features[15], features[16])
            stage_channels = [24, 40, 112, 960]

        else:
            raise ValueError(f"Unsupported camera backbone: {self.model_name}")

        self.fpn = FPN(
            in_channels_list=stage_channels,
            out_channels=fpn_out_channels,
        )

        # Stride 8 output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(fpn_out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 1:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts stride-8 image features.

        Args:
            images: [B, C, H, W] or [B, N, C, H, W] tensor.

        Returns:
            features: [B, out_channels, H/8, W/8] or [B, N, out_channels, H/8, W/8].
        """
        orig_shape = images.shape
        if images.dim() == 5:
            # Multi-camera: [B, N, C, H, W] -> flatten B*N
            B, N, C, H, W = orig_shape
            images = images.view(B * N, C, H, W)
        else:
            B, N = orig_shape[0], 1

        x = self.stem(images)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        fpn_outs = self.fpn([c1, c2, c3, c4])
        feat = fpn_outs[1]  # Stride 8 (corresponding to c2)
        feat = self.output_conv(feat)

        if len(orig_shape) == 5:
            feat = feat.view(B, N, self.out_channels, feat.shape[2], feat.shape[3])

        return feat
