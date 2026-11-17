#!/usr/bin/env python3
"""
BEVFusionLite Heavy Baseline Overfit Test.

Overfits a 16-frame KITTI multimodal subset for 300 iterations to verify
gradient flow, loss decrease, and optimization sanity before full training runs.
"""

import sys
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

from bevfusionlite.datasets.kitti_fusion import KITTIFusionDataset
from bevfusionlite.models.bevfusionlite_detector import BEVFusionLiteDetector


class DatasetStub:
    """Minimal dataset stub for OpenPCDet Detector3DTemplate initialization."""
    def __init__(self, class_names):
        self.class_names = class_names
        self.point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        self.grid_size = np.array([432, 496, 1]) if 'np' in globals() else None
        self.voxel_size = [0.16, 0.16, 4]
        self.depth_downsample_factor = None

        class PointFeatureEncoderStub:
            num_point_features = 4
        self.point_feature_encoder = PointFeatureEncoderStub()


def run_overfit_check(iterations=300, subset_size=16):
    import numpy as np
    from pcdet.config import cfg, cfg_from_yaml_file

    config_path = PROJECT_ROOT / "configs" / "heavy_baseline.yaml"
    cfg_from_yaml_file(str(config_path), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running overfit test on device: {device}")

    # Load 16 samples from local KITTI dataset
    dataset = KITTIFusionDataset(
        data_root=str(PROJECT_ROOT / "data" / "kitti"),
        split="train",
        training=True,
    )
    indices = list(range(min(subset_size, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=True,
        collate_fn=KITTIFusionDataset.collate_fn,
        drop_last=False,
    )

    # Instantiate model
    dataset_stub = DatasetStub(cfg.CLASS_NAMES)
    dataset_stub.grid_size = np.array([432, 496, 1])
    model = BEVFusionLiteDetector(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset_stub)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)

    print(f"[+] Starting {iterations} iterations on {len(indices)} frames...")
    initial_loss = None
    final_loss = None

    data_iter = iter(loader)
    for it in range(1, iterations + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Move tensors to device
        for k in ["images", "lidar2img", "points", "gt_boxes", "voxels", "voxel_coords", "voxel_num_points"]:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        # For AnchorHeadSingle target assignment
        batch["batch_size"] = batch["images"].shape[0]

        optimizer.zero_grad()
        ret_dict, tb_dict, disp_dict = model(batch)
        loss = ret_dict["loss"]
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if it == 1:
            initial_loss = loss_val
        final_loss = loss_val

        if it % 50 == 0 or it == 1:
            print(f"    Iter {it:03d}/{iterations}: loss = {loss_val:.4f}")

    print(f"[+] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    print("[PASS] Overfit test completed successfully!")
    return True


if __name__ == "__main__":
    import numpy as np
    run_overfit_check(iterations=20, subset_size=4)
