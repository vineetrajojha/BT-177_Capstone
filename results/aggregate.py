#!/usr/bin/env python3
"""
BEVFusionLite Results Aggregator.

Parses evaluation result logs or pickle files across multiple seeds,
computes mean +/- standard deviation, and outputs summary CSVs.
"""

import os
import re
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Published reference values from OpenPCDet PointPillars KITTI val split (AP R40)
PUBLISHED_POINTPILLARS_REF = {
    "Car_3d_easy": 86.62,
    "Car_3d_mod": 77.28,
    "Car_3d_hard": 74.29,
    "Car_bev_easy": 92.44,
    "Car_bev_mod": 88.61,
    "Car_bev_hard": 86.11,
    "Pedestrian_3d_easy": 57.75,
    "Pedestrian_3d_mod": 52.29,
    "Pedestrian_3d_hard": 47.90,
    "Pedestrian_bev_easy": 63.84,
    "Pedestrian_bev_mod": 58.74,
    "Pedestrian_bev_hard": 54.12,
    "Cyclist_3d_easy": 78.24,
    "Cyclist_3d_mod": 62.68,
    "Cyclist_3d_hard": 58.62,
    "Cyclist_bev_easy": 81.33,
    "Cyclist_bev_mod": 65.50,
    "Cyclist_bev_hard": 61.27,
}


def parse_kitti_eval_log(log_path):
    """Parses standard KITTI AP_R40 results from an OpenPCDet evaluation log file."""
    metrics = {}
    if not os.path.exists(log_path):
        return metrics

    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Match AP_R40 patterns in OpenPCDet eval output
    # Example format:
    # Car AP_R40@0.70, 0.70, 0.70:
    # bbox AP: 95.12, 90.15, 88.20
    # bev  AP: 92.44, 88.61, 86.11
    # 3d   AP: 86.62, 77.28, 74.29
    for cls in ["Car", "Pedestrian", "Cyclist"]:
        pat_3d = rf"{cls} AP_R40.*?\n(?:.*?AP:.*?\n)*?.*?3d\s+AP:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
        m3 = re.search(pat_3d, text, re.IGNORECASE)
        if m3:
            metrics[f"{cls}_3d_easy"] = float(m3.group(1))
            metrics[f"{cls}_3d_mod"] = float(m3.group(2))
            metrics[f"{cls}_3d_hard"] = float(m3.group(3))

        pat_bev = rf"{cls} AP_R40.*?\n(?:.*?AP:.*?\n)*?.*?bev\s+AP:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
        mb = re.search(pat_bev, text, re.IGNORECASE)
        if mb:
            metrics[f"{cls}_bev_easy"] = float(mb.group(1))
            metrics[f"{cls}_bev_mod"] = float(mb.group(2))
            metrics[f"{cls}_bev_hard"] = float(mb.group(3))

    return metrics


def aggregate_pointpillars(results_dir, output_csv):
    """Aggregates PointPillars seeds 0, 1, 2 and compares side-by-side with published reference."""
    results_dir = Path(results_dir)
    seed_dirs = sorted(glob.glob(str(results_dir / "seed_*")))

    records = []
    seed_metrics = []

    for s_dir in seed_dirs:
        seed_name = Path(s_dir).name
        log_files = glob.glob(str(Path(s_dir) / "eval*.log")) or glob.glob(str(Path(s_dir) / "log_eval*.txt"))
        if log_files:
            m = parse_kitti_eval_log(log_files[0])
            m["source"] = seed_name
            records.append(m)
            seed_metrics.append(m)

    # Compute mean and std if seeds exist
    if seed_metrics:
        keys = [k for k in PUBLISHED_POINTPILLARS_REF.keys() if k in seed_metrics[0]]
        mean_row = {"source": "Our_Reproduction_Mean"}
        std_row = {"source": "Our_Reproduction_Std"}
        for k in keys:
            vals = [sm[k] for sm in seed_metrics if k in sm]
            mean_row[k] = float(np.mean(vals)) if vals else np.nan
            std_row[k] = float(np.std(vals)) if vals else np.nan
        records.append(mean_row)
        records.append(std_row)

    # Add published reference
    pub_row = {"source": "OpenPCDet_Published_Reference"}
    pub_row.update(PUBLISHED_POINTPILLARS_REF)
    records.append(pub_row)

    df = pd.DataFrame(records)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[i] PointPillars summary saved to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/pointpillars_baseline")
    parser.add_argument("--output_csv", type=str, default="results/pointpillars_kitti_val.csv")
    args = parser.parse_args()

    aggregate_pointpillars(args.results_dir, args.output_csv)
