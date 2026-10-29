"""
Reproducibility utilities: seed management, environment capture, git hash logging.

Every training run uses these utilities to ensure full reproducibility:
    - Fixed random seeds across Python, NumPy, PyTorch, CUDA
    - Git commit hash capture
    - pip freeze / conda env export
    - Config snapshot saved alongside checkpoints
"""

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 0) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default is 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode — trades some speed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"All random seeds set to {seed}")


def get_git_hash(repo_path: Optional[str] = None) -> str:
    """Get the current git commit hash.

    Args:
        repo_path: Path to the git repository. If None, uses project root.

    Returns:
        Git commit hash string, or 'unknown' if not in a git repo.
    """
    if repo_path is None:
        repo_path = str(Path(__file__).parent.parent.parent)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def get_git_diff(repo_path: Optional[str] = None) -> str:
    """Get the current git diff (uncommitted changes).

    Args:
        repo_path: Path to the git repository. If None, uses project root.

    Returns:
        Git diff string, or empty string if clean or not in a git repo.
    """
    if repo_path is None:
        repo_path = str(Path(__file__).parent.parent.parent)
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def capture_pip_freeze(output_path: str) -> None:
    """Capture pip freeze output to a file.

    Args:
        output_path: Path to write the pip freeze output.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(result.stdout)
            logger.info(f"pip freeze saved to {output_path}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Failed to capture pip freeze: {e}")


def save_run_metadata(
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save complete run metadata for reproducibility.

    Creates a metadata.json file in the output directory containing:
    - Git commit hash and diff status
    - Seed value
    - Timestamp
    - PyTorch and CUDA versions
    - Config snapshot
    - Any extra info provided

    Args:
        output_dir: Directory to save metadata files.
        config: Configuration dictionary to snapshot.
        seed: Random seed used for this run.
        extra_info: Additional metadata to include.

    Returns:
        The metadata dictionary that was saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "git_hash": get_git_hash(),
        "git_dirty": bool(get_git_diff()),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
    }

    if config is not None:
        metadata["config"] = config

    if extra_info is not None:
        metadata.update(extra_info)

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Run metadata saved to {metadata_path}")

    # Also save pip freeze
    capture_pip_freeze(os.path.join(output_dir, "env.txt"))

    return metadata


def verify_determinism(fn, *args, n_trials: int = 3, seed: int = 0, **kwargs) -> bool:
    """Verify that a function produces deterministic results with a fixed seed.

    Args:
        fn: Function to test for determinism.
        *args: Positional arguments to pass to fn.
        n_trials: Number of repeated trials.
        seed: Seed to use for each trial.
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        True if all trials produce identical results.
    """
    results = []
    for _ in range(n_trials):
        set_seed(seed)
        result = fn(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            results.append(result.detach().cpu().numpy().tobytes())
        elif isinstance(result, dict):
            # Hash the dict values that are tensors
            h = hashlib.md5()
            for k in sorted(result.keys()):
                v = result[k]
                if isinstance(v, torch.Tensor):
                    h.update(v.detach().cpu().numpy().tobytes())
            results.append(h.hexdigest())
        else:
            results.append(str(result))

    is_deterministic = all(r == results[0] for r in results)
    if not is_deterministic:
        logger.warning("Non-deterministic behavior detected!")
    return is_deterministic
