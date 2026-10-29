"""
Unit tests for environment, framework dependencies, and plugin registration.
"""

import sys
import os
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))


def test_python_version():
    """Ensure Python version is 3.10 or higher."""
    assert sys.version_info >= (3, 10), f"Expected Python >= 3.10, got {sys.version}"


def test_torch_import():
    """Ensure PyTorch is installed and >= 2.1.0."""
    import torch
    major, minor = [int(x) for x in torch.__version__.split("+")[0].split(".")[:2]]
    assert (major, minor) >= (2, 1), f"Expected PyTorch >= 2.1, got {torch.__version__}"


def test_bevfusionlite_import():
    """Ensure bevfusionlite can be imported without error."""
    import bevfusionlite
    assert hasattr(bevfusionlite, "__version__")
    assert bevfusionlite.__version__ == "0.1.0"


def test_bevfusionlite_registry():
    """Ensure bevfusionlite plugin registry contains core components."""
    from bevfusionlite.registry import DETECTORS, DATASETS, get_all_registered

    registered = get_all_registered()
    assert "detectors" in registered
    assert "datasets" in registered
    assert "BEVFusionLiteDetector" in registered["detectors"]
    assert "KittiMultimodalDataset" in registered["datasets"]


def test_openpcdet_installed():
    """Ensure OpenPCDet is cloned and pcdet package is discoverable."""
    assert OPENPCDET_ROOT.exists(), f"OpenPCDet directory not found at {OPENPCDET_ROOT}"
    import pcdet
    assert hasattr(pcdet, "__version__") or hasattr(pcdet, "models")


def test_cuda_or_cpu_tensor():
    """Ensure basic tensor allocation and math works on available device."""
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.ones((3, 3), device=device)
    y = x * 2.0
    assert torch.allclose(y, torch.full((3, 3), 2.0, device=device))
