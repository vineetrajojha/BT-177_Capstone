#!/usr/bin/env python3
"""
Verify full official KITTI 3D Object Detection dataset layout and counts.

Expected official dataset specification (from https://www.cvlibs.net/datasets/kitti/):
- Training frames: 7,481 total
  - Split: 3,712 train, 3,769 val (standard OpenPCDet split in ImageSets/)
- Testing frames: 7,518 total
- Subdirectories:
  - training/calib/ (7481 files)
  - training/image_2/ (7481 files)
  - training/velodyne/ (7481 files)
  - training/label_2/ (7481 files)
  - testing/calib/ (7518 files)
  - testing/image_2/ (7518 files)
  - testing/velodyne/ (7518 files)

Usage:
    python scripts/verify_full_kitti.py
    KITTI_ROOT=/path/to/kitti python scripts/verify_full_kitti.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KITTI_ROOT = os.environ.get("KITTI_ROOT", str(PROJECT_ROOT / "data" / "kitti"))

EXPECTED_TRAIN = 3712
EXPECTED_VAL = 3769
EXPECTED_TRAINVAL = 7481
EXPECTED_TEST = 7518


def verify_full_dataset(data_path: Path) -> bool:
    print(f"Verifying full KITTI dataset at: {data_path.resolve()}")
    print("=" * 65)

    if not data_path.exists():
        print(f"[FAIL] Data directory does not exist: {data_path}")
        return False

    errors = []

    # 1. Verify ImageSets
    imagesets_dir = data_path / "ImageSets"
    for split_name, expected_count in [
        ("train.txt", EXPECTED_TRAIN),
        ("val.txt", EXPECTED_VAL),
        ("trainval.txt", EXPECTED_TRAINVAL),
        ("test.txt", EXPECTED_TEST),
    ]:
        split_file = imagesets_dir / split_name
        if not split_file.exists():
            errors.append(f"Missing {split_file}")
            continue
        with open(split_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        actual_count = len(ids)
        if actual_count != expected_count:
            errors.append(
                f"ImageSets/{split_name}: expected {expected_count}, got {actual_count}"
            )
        else:
            print(f"  [OK] ImageSets/{split_name}: {actual_count} samples (matches expected {expected_count})")

    # 2. Verify training directories
    train_dir = data_path / "training"
    for sub, expected in [
        ("calib", EXPECTED_TRAINVAL),
        ("image_2", EXPECTED_TRAINVAL),
        ("velodyne", EXPECTED_TRAINVAL),
        ("label_2", EXPECTED_TRAINVAL),
    ]:
        p = train_dir / sub
        if not p.exists():
            errors.append(f"Missing training directory: {p}")
            continue
        count = len(list(p.glob("*.*")))
        if count != expected:
            errors.append(f"training/{sub}: expected {expected} files, found {count}")
        else:
            print(f"  [OK] training/{sub}: {count} files (matches expected {expected})")

    # 3. Verify testing directories
    test_dir = data_path / "testing"
    for sub, expected in [
        ("calib", EXPECTED_TEST),
        ("image_2", EXPECTED_TEST),
        ("velodyne", EXPECTED_TEST),
    ]:
        p = test_dir / sub
        if not p.exists():
            errors.append(f"Missing testing directory: {p}")
            continue
        count = len(list(p.glob("*.*")))
        if count != expected:
            errors.append(f"testing/{sub}: expected {expected} files, found {count}")
        else:
            print(f"  [OK] testing/{sub}: {count} files (matches expected {expected})")

    print("=" * 65)
    if errors:
        print("[FAIL] Full dataset verification failed:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("[PASS] Full KITTI 3D dataset verified with all expected file counts!")
    return True


if __name__ == "__main__":
    success = verify_full_dataset(Path(KITTI_ROOT))
    sys.exit(0 if success else 1)
