"""
BEVFusionLite — Lightweight Camera-LiDAR BEV Fusion 3D Detector.

This package is designed as an OpenPCDet plugin. All custom modules
(detectors, datasets, model components) are registered here and
made available to OpenPCDet's build system via the plugin hooks below.

Architecture:
    Camera: ResNet-18 + FPN → LSS View Transform → Camera BEV
    LiDAR:  PillarVFE → PointPillarScatter → BaseBEVBackbone → LiDAR BEV
    Fusion: Channel Concat + Conv → Fused BEV → AnchorHead → 3D Boxes
"""

__version__ = "0.1.0"

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _add_openpcdet_to_path():
    """Add OpenPCDet to sys.path so we can import pcdet modules."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    openpcdet_path = os.path.join(project_root, "third_party", "OpenPCDet")
    if openpcdet_path not in sys.path:
        sys.path.insert(0, openpcdet_path)
        logger.debug(f"Added OpenPCDet to sys.path: {openpcdet_path}")


# Ensure OpenPCDet is importable when bevfusionlite is imported
_add_openpcdet_to_path()


def get_registered_detectors():
    """Return a dict of {name: class} for all BEVFusionLite custom detectors.

    This is the plugin hook that OpenPCDet's patched build_network() calls
    to discover our custom detector classes.
    """
    from bevfusionlite.models.bevfusionlite_detector import BEVFusionLiteDetector

    return {
        "BEVFusionLiteDetector": BEVFusionLiteDetector,
    }


def get_registered_datasets():
    """Return a dict of {name: class} for all BEVFusionLite custom datasets.

    Plugin hook for OpenPCDet's dataset builder.
    """
    from bevfusionlite.datasets.kitti_multimodal import KittiMultimodalDataset

    return {
        "KittiMultimodalDataset": KittiMultimodalDataset,
    }
