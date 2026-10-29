"""
BEVFusionLite Plugin Registry.

Provides a decorator-based registry so custom modules can be discovered
by OpenPCDet's config-driven build system without modifying OpenPCDet source.

Usage in bevfusionlite modules:
    from bevfusionlite.registry import DETECTORS, DATASETS

    @DETECTORS.register
    class MyDetector(Detector3DTemplate):
        ...

Usage in OpenPCDet (via patch):
    from bevfusionlite.registry import get_all_registered
    plugin_modules = get_all_registered()
"""

import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)


class Registry:
    """Simple string→class registry with decorator support.

    Example:
        MODELS = Registry("models")

        @MODELS.register
        class MyModel(nn.Module):
            pass

        cls = MODELS.get("MyModel")
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, cls: Type = None, *, name: str = None):
        """Register a class. Can be used as a decorator or called directly.

        @REGISTRY.register
        class Foo: ...

        @REGISTRY.register(name="CustomName")
        class Bar: ...

        REGISTRY.register(Baz)
        """
        def _do_register(cls_to_reg):
            reg_name = name if name is not None else cls_to_reg.__name__
            if reg_name in self._registry:
                logger.warning(
                    f"Registry '{self.name}': overwriting '{reg_name}' "
                    f"({self._registry[reg_name]} → {cls_to_reg})"
                )
            self._registry[reg_name] = cls_to_reg
            logger.debug(f"Registry '{self.name}': registered '{reg_name}'")
            return cls_to_reg

        if cls is not None:
            # Used as @REGISTRY.register (no parentheses)
            return _do_register(cls)
        else:
            # Used as @REGISTRY.register(name="...")
            return _do_register

    def get(self, name: str) -> Type:
        """Retrieve a class by name."""
        if name not in self._registry:
            raise KeyError(
                f"Registry '{self.name}' has no entry '{name}'. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def get_all(self) -> Dict[str, Type]:
        """Return all registered classes."""
        return dict(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, entries={list(self._registry.keys())})"


# ── Global registries ────────────────────────────────────────────────
DETECTORS = Registry("detectors")
DATASETS = Registry("datasets")
BACKBONES = Registry("backbones")
HEADS = Registry("heads")
VFES = Registry("vfes")
FUSION_MODULES = Registry("fusion_modules")


def get_all_registered() -> Dict[str, Dict[str, Type]]:
    """Return all registered plugin modules, grouped by category.

    Returns:
        Dict like:
            {
                "detectors": {"BEVFusionLiteDetector": <class>},
                "datasets": {"KittiMultimodalDataset": <class>},
                ...
            }
    """
    return {
        "detectors": DETECTORS.get_all(),
        "datasets": DATASETS.get_all(),
        "backbones": BACKBONES.get_all(),
        "heads": HEADS.get_all(),
        "vfes": VFES.get_all(),
        "fusion_modules": FUSION_MODULES.get_all(),
    }


# ── Auto-register BEVFusionLite modules on import ───────────────────
def _auto_register():
    """Import and register all BEVFusionLite modules."""
    try:
        from bevfusionlite.models.bevfusionlite_detector import BEVFusionLiteDetector
        DETECTORS.register(BEVFusionLiteDetector)
    except ImportError as e:
        logger.debug(f"Could not auto-register BEVFusionLiteDetector: {e}")

    try:
        from bevfusionlite.datasets.kitti_multimodal import KittiMultimodalDataset
        DATASETS.register(KittiMultimodalDataset)
    except ImportError as e:
        logger.debug(f"Could not auto-register KittiMultimodalDataset: {e}")

    try:
        from bevfusionlite.models.camera_backbone import CameraBackbone
        BACKBONES.register(CameraBackbone)
    except ImportError as e:
        logger.debug(f"Could not auto-register CameraBackbone: {e}")

    try:
        from bevfusionlite.models.bev_fusion import ConvFuser
        FUSION_MODULES.register(ConvFuser)
    except ImportError as e:
        logger.debug(f"Could not auto-register ConvFuser: {e}")


_auto_register()
