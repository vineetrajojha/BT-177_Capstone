#!/usr/bin/env python3
"""
Environment and Model Sanity Check Script for BEVFusionLite.

Checks and prints:
  - PyTorch version
  - CUDA availability, CUDA version, and GPU device name
  - OpenPCDet commit hash
  - OpenPCDet PointPillars forward pass on synthetic data
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root and OpenPCDet to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

import torch
import numpy as np


def get_git_commit(repo_path: Path) -> str:
    """Retrieve git commit hash of a repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception as e:
        return f"Unknown ({e})"
    return "Unknown"


def print_env_info():
    """Print system, PyTorch, CUDA, and OpenPCDet details."""
    print("=" * 70)
    print(" " * 20 + "BEVFUSIONLITE ENVIRONMENT CHECK")
    print("=" * 70)
    print(f"Python Version    : {sys.version.split()[0]} ({sys.executable})")
    print(f"PyTorch Version   : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available    : {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Version      : {torch.version.cuda}")
        print(f"GPU Device Count  : {torch.cuda.device_count()}")
        print(f"Primary GPU Name  : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total GPU VRAM    : {mem_gb:.2f} GB")
    else:
        print("Primary GPU Name  : None (CPU mode)")

    pcdet_commit = get_git_commit(OPENPCDET_ROOT)
    print(f"OpenPCDet Commit  : {pcdet_commit}")
    print("=" * 70)


def test_pointpillars_forward():
    """Run a synthetic forward pass through OpenPCDet PointPillars components."""
    print("\n[+] Running OpenPCDet PointPillars forward pass on random data...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"    Target Device: {device}")

    try:
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network
        from easydict import EasyDict

        cfg_file = OPENPCDET_ROOT / "tools" / "cfgs" / "kitti_models" / "pointpillar.yaml"
        if cfg_file.exists():
            orig_cwd = os.getcwd()
            try:
                os.chdir(str(OPENPCDET_ROOT / "tools"))
                cfg_from_yaml_file(str(cfg_file), cfg)
            finally:
                os.chdir(orig_cwd)
            model_cfg = cfg.MODEL
            class_names = cfg.CLASS_NAMES
        else:
            # Fallback lightweight config if file not found
            model_cfg = EasyDict({
                "NAME": "PointPillar",
                "VFE": {
                    "NAME": "PillarVFE",
                    "WITH_DISTANCE": False,
                    "USE_ABSLOTE_XYZ": True,
                    "USE_NORM": True,
                    "NUM_FILTERS": [64],
                },
                "MAP_TO_BEV": {
                    "NAME": "PointPillarScatter",
                    "NUM_BEV_FEATURES": 64,
                },
                "BACKBONE_2D": {
                    "NAME": "BaseBEVBackbone",
                    "LAYER_NUMS": [3, 5, 5],
                    "LAYER_STRIDES": [2, 2, 2],
                    "NUM_FILTERS": [64, 128, 256],
                    "UPSAMPLE_STRIDES": [1, 2, 4],
                    "NUM_UPSAMPLE_FILTERS": [128, 128, 128],
                },
                "DENSE_HEAD": {
                    "NAME": "AnchorHeadSingle",
                    "CLASS_AGNOSTIC": False,
                    "USE_DIRECTION_CLASSIFIER": True,
                    "DIR_OFFSET": 0.78539,
                    "DIR_LIMIT_OFFSET": 0.0,
                    "NUM_DIR_BINS": 2,
                    "ANCHOR_GENERATOR_CONFIG": [
                        {
                            "class_name": "Car",
                            "anchor_sizes": [[3.9, 1.6, 1.56]],
                            "anchor_rotations": [0, 1.57],
                            "anchor_bottom_heights": [-1.78],
                            "align_center": False,
                            "feature_map_stride": 2,
                            "matched_threshold": 0.6,
                            "unmatched_threshold": 0.45,
                        }
                    ],
                    "TARGET_ASSIGNER_CONFIG": {
                        "NAME": "AxisAlignedTargetAssigner",
                        "BOX_CODER": "ResidualCoder",
                    },
                    "LOSS_CONFIG": {
                        "LOSS_WEIGHTS": {
                            "cls_weight": 1.0,
                            "loc_weight": 2.0,
                            "dir_weight": 0.2,
                            "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        }
                    },
                },
                "POST_PROCESSING": {
                    "RECALL_THRESH_LIST": [0.3, 0.5, 0.7],
                    "SCORE_THRESH": 0.1,
                    "OUTPUT_RAW_SCORE": False,
                    "EVAL_METRIC": "kitti",
                    "NMS_CONFIG": {
                        "MULTI_CLASSES_NMS": False,
                        "NMS_TYPE": "nms_gpu" if torch.cuda.is_available() else "nms_cpu",
                        "NMS_THRESH": 0.01,
                        "NMS_PRE_MAXSIZE": 4096,
                        "NMS_POST_MAXSIZE": 500,
                    },
                },
            })
            class_names = ["Car"]

        # Instantiate dataset stub for template requirements
        class DatasetStub:
            def __init__(self, class_names):
                self.class_names = class_names
                self.point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
                self.grid_size = np.array([432, 496, 1])
                self.voxel_size = [0.16, 0.16, 4]
                self.depth_downsample_factor = None

                class PointFeatureEncoderStub:
                    num_point_features = 4

                self.point_feature_encoder = PointFeatureEncoderStub()

        dataset = DatasetStub(class_names)
        num_class = len(class_names)

        model = build_network(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        model.to(device)
        model.eval()

        # Create synthetic input batch
        # Synthetic voxels/pillars for PointPillars:
        # voxels: [M, max_points_per_voxel, point_features]
        # voxel_coords: [M, 4] (batch_idx, z, y, x)
        # voxel_num_points: [M]
        num_voxels = 1000
        max_points = 32
        num_features = 4  # x, y, z, intensity

        voxels = torch.randn((num_voxels, max_points, num_features), dtype=torch.float32, device=device)
        voxel_coords = torch.zeros((num_voxels, 4), dtype=torch.int32, device=device)
        voxel_coords[:, 0] = 0  # batch 0
        voxel_coords[:, 1] = 0  # z
        voxel_coords[:, 2] = torch.randint(0, 496, (num_voxels,), device=device)  # y
        voxel_coords[:, 3] = torch.randint(0, 432, (num_voxels,), device=device)  # x
        voxel_num_points = torch.randint(1, max_points + 1, (num_voxels,), dtype=torch.int32, device=device)

        batch_dict = {
            "voxels": voxels,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "batch_size": 1,
        }

        with torch.no_grad():
            batch_dict = model.vfe(batch_dict)
            batch_dict = model.map_to_bev_module(batch_dict)
            batch_dict = model.backbone_2d(batch_dict)
            spatial_feats = batch_dict["spatial_features_2d"]

        print(f"    [PASS] VFE + PointPillarScatter + BaseBEVBackbone output shape: {tuple(spatial_feats.shape)}")
        print(f"    Output Tensor Device: {spatial_feats.device}")
        print("    [PASS] PointPillars forward pass executed successfully!")
        return True

    except Exception as e:
        print(f"    [FAIL] PointPillars forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print_env_info()
    success = test_pointpillars_forward()
    if success:
        print("\n>>> ALL ENVIRONMENT AND MODEL CHECKS PASSED <<<\n")
        sys.exit(0)
    else:
        print("\n>>> CHECK FAILED <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
