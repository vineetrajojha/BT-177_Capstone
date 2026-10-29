#!/usr/bin/env python3
"""
KITTI Data Preparation Script for BEVFusionLite.

Generates:
    - kitti_infos_train.pkl, kitti_infos_val.pkl, kitti_infos_trainval.pkl, kitti_infos_test.pkl
    - gt_database/ with per-object point cloud crops
    - kitti_dbinfos_train.pkl

All output goes under data/kitti/. Requires KITTI 3D object detection data
to be already downloaded and organized in the standard layout:
    data/kitti/
        ImageSets/
            train.txt   (3712 indices)
            val.txt     (3769 indices)
            test.txt    (7518 indices)
        training/
            calib/      (7481 files)
            image_2/    (7481 files)
            label_2/    (7481 files)
            velodyne/   (7481 files)
        testing/
            calib/      (7518 files)
            image_2/    (7518 files)
            velodyne/   (7518 files)

Usage:
    python scripts/prepare_kitti.py [--data_path DATA_PATH] [--workers N]
"""

import argparse
import os
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))


def validate_kitti_layout(data_path: Path) -> bool:
    """Validate that the KITTI data directory has the expected layout."""
    print(f"\n[1/4] Validating KITTI data layout at: {data_path}")

    required_dirs = [
        "ImageSets",
        "training/calib",
        "training/image_2",
        "training/label_2",
        "training/velodyne",
        "testing/calib",
        "testing/image_2",
        "testing/velodyne",
    ]

    all_ok = True
    for d in required_dirs:
        path = data_path / d
        if not path.exists():
            print(f"  [MISSING] {d}")
            all_ok = False
        else:
            count = len(list(path.iterdir())) if path.is_dir() else 0
            print(f"  [OK] {d} ({count} files)")

    # Check ImageSets
    expected_splits = {
        "train.txt": 3712,
        "val.txt": 3769,
        "test.txt": 7518,
    }
    for fname, expected in expected_splits.items():
        split_file = data_path / "ImageSets" / fname
        if split_file.exists():
            count = len([l for l in open(split_file).readlines() if l.strip()])
            status = "OK" if count == expected else f"WARN (expected {expected})"
            print(f"  [{status}] ImageSets/{fname}: {count} samples")
        else:
            print(f"  [MISSING] ImageSets/{fname}")
            all_ok = False

    return all_ok


def create_imagesets_if_missing(data_path: Path):
    """Create standard KITTI ImageSets if they don't exist."""
    imagesets_dir = data_path / "ImageSets"
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    # Check if training data exists to count frames
    train_calib = data_path / "training" / "calib"
    test_calib = data_path / "testing" / "calib"

    if not (data_path / "ImageSets" / "train.txt").exists():
        if train_calib.exists():
            all_ids = sorted([f.stem for f in train_calib.glob("*.txt")])
            print(f"  Found {len(all_ids)} training frames")

            # Standard KITTI split: first 3712 = train, last 3769 = val
            # Use Chen et al. split if available
            train_ids = all_ids[:3712]
            val_ids = all_ids[3712:]

            with open(imagesets_dir / "train.txt", "w") as f:
                f.write("\n".join(train_ids) + "\n")
            with open(imagesets_dir / "val.txt", "w") as f:
                f.write("\n".join(val_ids) + "\n")
            with open(imagesets_dir / "trainval.txt", "w") as f:
                f.write("\n".join(all_ids) + "\n")

            print(f"  Created train.txt ({len(train_ids)}), val.txt ({len(val_ids)})")

    if not (data_path / "ImageSets" / "test.txt").exists():
        if test_calib.exists():
            test_ids = sorted([f.stem for f in test_calib.glob("*.txt")])
            with open(imagesets_dir / "test.txt", "w") as f:
                f.write("\n".join(test_ids) + "\n")
            print(f"  Created test.txt ({len(test_ids)})")


def generate_infos(data_path: Path, workers: int = 4):
    """Generate KITTI info pickle files using OpenPCDet's pipeline."""
    print(f"\n[2/4] Generating KITTI info files...")

    import yaml
    from easydict import EasyDict

    # Load the default KITTI dataset config
    cfg_file = OPENPCDET_ROOT / "tools" / "cfgs" / "dataset_configs" / "kitti_dataset.yaml"
    with open(cfg_file, "r") as f:
        dataset_cfg = EasyDict(yaml.safe_load(f))

    # Override data path
    dataset_cfg.DATA_PATH = str(data_path)

    from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos

    create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=["Car", "Pedestrian", "Cyclist"],
        data_path=data_path,
        save_path=data_path,
        workers=workers,
    )


def verify_outputs(data_path: Path) -> bool:
    """Verify that all expected output files were created."""
    print(f"\n[3/4] Verifying generated outputs...")

    expected_files = [
        "kitti_infos_train.pkl",
        "kitti_infos_val.pkl",
        "kitti_infos_trainval.pkl",
        "kitti_infos_test.pkl",
        "kitti_dbinfos_train.pkl",
    ]

    all_ok = True
    for fname in expected_files:
        fpath = data_path / fname
        if fpath.exists():
            import pickle
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                print(f"  [OK] {fname}: {len(data)} entries")
            elif isinstance(data, dict):
                total = sum(len(v) for v in data.values())
                print(f"  [OK] {fname}: {len(data)} classes, {total} total entries")
            else:
                print(f"  [OK] {fname}: exists")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    gt_db = data_path / "gt_database"
    if gt_db.exists():
        count = len(list(gt_db.glob("*.bin")))
        print(f"  [OK] gt_database/: {count} object crops")
    else:
        print(f"  [MISSING] gt_database/")
        all_ok = False

    # Verify info files contain image path and calib
    train_info_path = data_path / "kitti_infos_train.pkl"
    if train_info_path.exists():
        import pickle
        with open(train_info_path, "rb") as f:
            infos = pickle.load(f)
        if len(infos) > 0:
            sample = infos[0]
            has_image = "image" in sample
            has_calib = "calib" in sample
            has_pc = "point_cloud" in sample
            print(f"  [{'OK' if has_image else 'FAIL'}] Info contains 'image' key")
            print(f"  [{'OK' if has_calib else 'FAIL'}] Info contains 'calib' key (P2, R0_rect, Tr_velo_to_cam)")
            print(f"  [{'OK' if has_pc else 'FAIL'}] Info contains 'point_cloud' key")

            if has_calib:
                calib_keys = list(sample["calib"].keys())
                print(f"    Calib keys: {calib_keys}")

    return all_ok


def print_summary(data_path: Path):
    """Print a summary of the prepared data."""
    print(f"\n[4/4] KITTI Data Preparation Summary")
    print("=" * 60)
    print(f"  Data root: {data_path}")

    for split in ["train", "val", "trainval", "test"]:
        split_file = data_path / "ImageSets" / f"{split}.txt"
        if split_file.exists():
            count = len([l for l in open(split_file).readlines() if l.strip()])
            print(f"  {split:>10s}: {count} samples")

    gt_db = data_path / "gt_database"
    if gt_db.exists():
        count = len(list(gt_db.glob("*.bin")))
        print(f"  {'gt_database':>10s}: {count} object crops")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare KITTI data for BEVFusionLite")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("KITTI_ROOT", str(PROJECT_ROOT / "data" / "kitti")),
        help="Path to KITTI data directory (or via KITTI_ROOT env var)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for info generation",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip initial data layout validation",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    print(f"KITTI Data Preparation for BEVFusionLite")
    print(f"{'=' * 60}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path:    {data_path}")

    # Step 0: Create ImageSets if missing
    create_imagesets_if_missing(data_path)

    # Step 1: Validate
    if not args.skip_validation:
        if not validate_kitti_layout(data_path):
            print("\n[ERROR] KITTI data layout validation failed.")
            print("Please download KITTI 3D Object Detection data from:")
            print("  https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d")
            print("\nExpected files:")
            print("  - Left color images (image_2)")
            print("  - Velodyne point clouds")
            print("  - Camera calibration files")
            print("  - Training labels")
            sys.exit(1)

    # Step 2: Generate infos
    generate_infos(data_path, workers=args.workers)

    # Step 3: Verify
    all_ok = verify_outputs(data_path)

    # Step 4: Summary
    print_summary(data_path)

    if all_ok:
        print("\n>>> KITTI DATA PREPARATION COMPLETE <<<\n")
    else:
        print("\n>>> KITTI DATA PREPARATION INCOMPLETE (some outputs missing) <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
