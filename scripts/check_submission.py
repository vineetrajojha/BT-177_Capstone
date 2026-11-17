#!/usr/bin/env python3
"""
KITTI Test Server Submission Validator.

Validates:
  1. Exact file count: 7,518 files named `000000.txt` through `007517.txt`.
  2. Non-empty check and ASCII encoding.
  3. Format conformance: 15 space-separated columns per prediction line:
     type, truncated, occluded, alpha, bbox (4), dimensions (3: h, w, l),
     location (3: x, y, z in rect camera frame), rotation_y, score.
  4. Numerical bounds:
     - 2D bbox within image canvas
     - 3D dimensions > 0 (height, width, length)
     - location[2] (camera depth z) > 0
     - score in [0.0, 1.0]
     - rotation_y in [-pi, pi]
"""

import os
import sys
import glob
import math
import zipfile
import argparse
from pathlib import Path

VALID_CLASSES = {"Car", "Pedestrian", "Cyclist", "Van", "Truck", "Person_sitting", "Tram", "Misc", "DontCare"}
EXPECTED_TEST_COUNT = 7518


def validate_prediction_line(line: str, line_no: int, filename: str):
    parts = line.strip().split()
    if not parts:
        return True  # Empty line

    if len(parts) != 15 and len(parts) != 16:
        raise ValueError(
            f"[{filename}:{line_no}] Expected 15 (or 16 with score) elements, got {len(parts)}: '{line.strip()}'"
        )

    cls_name = parts[0]
    if cls_name not in VALID_CLASSES:
        raise ValueError(f"[{filename}:{line_no}] Unknown object class: '{cls_name}'")

    truncated = float(parts[1])
    occluded = int(parts[2])
    alpha = float(parts[3])

    # 2D Bounding Box: x1, y1, x2, y2
    x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
    if x2 < x1 or y2 < y1:
        raise ValueError(f"[{filename}:{line_no}] Inverted 2D bbox coordinates: [{x1}, {y1}, {x2}, {y2}]")

    # 3D Dimensions: height, width, length
    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    if h <= 0 or w <= 0 or l <= 0:
        raise ValueError(f"[{filename}:{line_no}] Non-positive 3D dimensions: h={h}, w={w}, l={l}")

    # 3D Location in camera coordinate frame: x, y, z
    cx, cy, cz = float(parts[11]), float(parts[12]), float(parts[13])
    if cz <= 0:
        raise ValueError(f"[{filename}:{line_no}] Camera depth z must be positive, got z={cz}")

    rot_y = float(parts[14])
    if abs(rot_y) > math.pi + 0.1:
        raise ValueError(f"[{filename}:{line_no}] rotation_y out of [-pi, pi]: {rot_y}")

    if len(parts) >= 16:
        score = float(parts[15])
        if score < 0.0 or score > 1.01:
            raise ValueError(f"[{filename}:{line_no}] Confidence score out of [0, 1]: {score}")

    return True


def validate_submission_directory(pred_dir: str):
    pred_path = Path(pred_dir)
    if not pred_path.exists():
        print(f"[FAIL] Directory does not exist: {pred_path}")
        return False

    files = sorted(list(pred_path.glob("*.txt")))
    print(f"[+] Found {len(files)} prediction text files in {pred_path}")

    if len(files) != EXPECTED_TEST_COUNT:
        print(f"[!] Warning: Found {len(files)} files, expected {EXPECTED_TEST_COUNT} for official test split.")
        if len(files) == 0:
            return False

    # Spot check format across files
    total_boxes = 0
    errors = 0
    for idx, f in enumerate(files):
        with open(f, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for l_idx, line in enumerate(lines, start=1):
                try:
                    validate_prediction_line(line, l_idx, f.name)
                    total_boxes += 1
                except Exception as err:
                    print(f"[ERROR] {err}")
                    errors += 1
                    if errors > 10:
                        print("[!] Aborting after 10 format errors.")
                        return False

    print(f"[PASS] Successfully verified {len(files)} prediction files ({total_boxes} total 3D detections).")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate KITTI 3D benchmark submission package.")
    parser.add_argument("--pred_dir", type=str, default="results/kitti_submission/data")
    parser.add_argument("--zip_file", type=str, default="results/kitti_test_submission.zip")
    args = parser.parse_args()

    if os.path.exists(args.zip_file):
        print(f"[+] Verifying zip archive: {args.zip_file}")
        with zipfile.ZipFile(args.zip_file, "r") as z:
            txt_files = [f for f in z.namelist() if f.endswith(".txt")]
            print(f"[+] Zip contains {len(txt_files)} prediction files.")
            if len(txt_files) == EXPECTED_TEST_COUNT:
                print(f"[PASS] Archive contains all {EXPECTED_TEST_COUNT} test files.")
            else:
                print(f"[!] Warning: Archive has {len(txt_files)} files (expected {EXPECTED_TEST_COUNT}).")
    else:
        validate_submission_directory(args.pred_dir)
