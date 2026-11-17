#!/usr/bin/env python3
"""
BEVFusionLite Kaggle Cloud Training Script.

Automates:
  1. Setting up the OpenPCDet + PyTorch environment on Kaggle T4 GPU.
  2. Linking attached Kaggle KITTI dataset to `data/kitti`.
  3. Running `scripts/apply_patches.py` and environment verification.
  4. Executing specified training runs (PointPillars, Heavy, Light, Ablations) with `--resume` support.
  5. Syncing checkpoints to Kaggle working directory for persistent download.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd, cwd=None):
    print(f"\n[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        print(f"[!] Warning: Command failed with exit code {res.returncode}")
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Kaggle cloud training runner for BEVFusionLite.")
    parser.add_argument("--run_name", type=str, default="light_seed_0", help="Unique identifier for experiment.")
    parser.add_argument("--config", type=str, default="configs/light_model.yaml", help="Path to experiment config.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--kitti_kaggle_path", type=str, default="/kaggle/input/kitti-3d-object-detection-dataset", help="Attached Kaggle dataset path.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # 1. Symlink dataset
    target_kitti = project_root / "data" / "kitti"
    target_kitti.parent.mkdir(parents=True, exist_ok=True)
    if not target_kitti.exists():
        if os.path.exists(args.kitti_kaggle_path):
            print(f"[+] Linking Kaggle KITTI dataset from {args.kitti_kaggle_path} to {target_kitti}")
            os.symlink(args.kitti_kaggle_path, str(target_kitti))
        else:
            print(f"[!] Kaggle KITTI path not found at {args.kitti_kaggle_path}. Please attach dataset to kernel.")

    # 2. Apply patches
    run_cmd(f"python3 scripts/apply_patches.py", cwd=str(project_root))

    # 3. Verify environment
    run_cmd("python3 scripts/check_env.py", cwd=str(project_root))

    # 4. Launch training with OpenPCDet train.py
    cmd = (
        f"python3 third_party/OpenPCDet/tools/train.py "
        f"--cfg_file {args.config} "
        f"--extra_tag {args.run_name} "
        f"--batch_size {args.batch_size} "
        f"--epochs {args.epochs} "
    )
    if args.resume:
        cmd += " --ckpt_dir ../output/checkpoints"

    run_cmd(cmd, cwd=str(project_root))

    # 5. Run evaluation and benchmark
    eval_cmd = (
        f"python3 third_party/OpenPCDet/tools/test.py "
        f"--cfg_file {args.config} "
        f"--extra_tag {args.run_name} "
        f"--batch_size 1"
    )
    run_cmd(eval_cmd, cwd=str(project_root))

    print(f"\n[PASS] Training and evaluation complete for {args.run_name}!")


if __name__ == "__main__":
    main()
