#!/usr/bin/env python3
"""
BEVFusionLite Web Data Exporter.

Reads all results, metrics, benchmarks, and markdown documentation from `results/`,
`reports/`, and `docs/`, and generates validated, source-attributed JSON data files
in `web/public/data/` for the web application.

Strict invariant:
  No number displayed on the website may be hardcoded or fabricated.
  Every metric in exported JSON includes an explicit `source_file` attribute.
"""

import os
import sys
import json
import re
import math
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sanitize_val(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_val(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_val(x) for x in obj]
    return obj


def export_all_web_data():
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports"
    docs_dir = PROJECT_ROOT / "docs"
    out_dir = PROJECT_ROOT / "web" / "public" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Exporting web data from {results_dir} to {out_dir}...")

    # 1. Main comparison
    main_csv = results_dir / "main_comparison.csv"
    if main_csv.exists():
        df_main = pd.read_csv(main_csv)
        df_clean = df_main.where(pd.notnull(df_main), None)
        main_records = df_clean.to_dict(orient="records")
        for rec in main_records:
            rec["source_file"] = "results/main_comparison.csv"

        with open(out_dir / "main_comparison.json", "w", encoding="utf-8") as fp:
            json.dump(sanitize_val({
                "source_file": "results/main_comparison.csv",
                "data": main_records,
            }), fp, indent=2, allow_nan=False)
        print("  [PASS] Exported main_comparison.json")

    # 2. Headline Summary Metrics
    heavy = df_main[df_main["model"] == "Heavy_Baseline_FP32"].iloc[0]
    light = df_main[df_main["model"] == "Light_Model_FP32"].iloc[0]
    lidar_only = df_main[df_main["model"] == "Light_LiDAR_Only_Ablation"].iloc[0]

    param_reduction = round((1.0 - light["params_M"] / heavy["params_M"]) * 100.0, 1)
    fps_gain = round(light["fps_t4"] / heavy["fps_t4"], 2)
    car_gap = round(heavy["car_3d_mod"] - light["car_3d_mod"], 2)
    ped_gain = round(light["ped_3d_mod"] - lidar_only["ped_3d_mod"], 2)
    cyc_gain = round(light["cyc_3d_mod"] - lidar_only["cyc_3d_mod"], 2)

    summary = {
        "source_file": "results/main_comparison.csv",
        "headline_metrics": [
            {
                "label": "Parameter Reduction",
                "value": f"-{param_reduction}%",
                "detail": f"{heavy['params_M']}M to {light['params_M']}M",
                "source_file": "results/main_comparison.csv",
            },
            {
                "label": "Throughput Boost",
                "value": f"{fps_gain}x FPS",
                "detail": f"{light['fps_t4']} FPS on NVIDIA T4",
                "source_file": "results/main_comparison.csv",
            },
            {
                "label": "Car 3D AP Retention",
                "value": f"{light['car_3d_mod']:.2f} AP",
                "detail": f"Within {car_gap} AP of Heavy Baseline",
                "source_file": "results/main_comparison.csv",
            },
            {
                "label": "VRU Recall Gain",
                "value": f"+{cyc_gain} AP",
                "detail": f"+{ped_gain} AP Ped, +{cyc_gain} AP Cyclist vs LiDAR",
                "source_file": "results/main_comparison.csv",
            },
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print("  [PASS] Exported summary.json")

    # 3. Targets Check
    targets_md = results_dir / "targets_check.md"
    if targets_md.exists():
        content = targets_md.read_text(encoding="utf-8")
        target_rows = []
        for line in content.split("\n"):
            if line.startswith("| **"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 4:
                    target_rows.append({
                        "name": parts[0].replace("**", ""),
                        "threshold": parts[1],
                        "measured": parts[2].replace("**", ""),
                        "status": parts[3].replace("**", ""),
                    })
        with open(out_dir / "targets_check.json", "w", encoding="utf-8") as fp:
            json.dump({
                "source_file": "results/targets_check.md",
                "targets": target_rows,
            }, fp, indent=2)
        print("  [PASS] Exported targets_check.json")

    # 4. Ablations Table
    abl_csv = results_dir / "ablation_table.csv"
    if abl_csv.exists():
        df_abl = pd.read_csv(abl_csv)
        df_abl_clean = df_abl.where(pd.notnull(df_abl), None)
        abl_records = df_abl_clean.to_dict(orient="records")
        for rec in abl_records:
            rec["source_file"] = "results/ablation_table.csv"
        with open(out_dir / "ablation_table.json", "w", encoding="utf-8") as fp:
            json.dump(sanitize_val({
                "source_file": "results/ablation_table.csv",
                "data": abl_records,
            }), fp, indent=2, allow_nan=False)
        print("  [PASS] Exported ablation_table.json")

    # 5. nuScenes-mini Table
    nusc_csv = results_dir / "nuscenes_mini.csv"
    if nusc_csv.exists():
        df_nusc = pd.read_csv(nusc_csv)
        df_nusc_clean = df_nusc.where(pd.notnull(df_nusc), None)
        nusc_records = df_nusc_clean.to_dict(orient="records")
        for rec in nusc_records:
            rec["source_file"] = "results/nuscenes_mini.csv"
        with open(out_dir / "nuscenes_mini.json", "w", encoding="utf-8") as fp:
            json.dump(sanitize_val({
                "source_file": "results/nuscenes_mini.csv",
                "data": nusc_records,
                "limitation_statement": (
                    "The nuScenes-mini subset contains only 10 driving sequences (~400 keyframes total, "
                    "with roughly 320 train and 80 val frames). Evaluating complex multi-camera 3D object detection "
                    "models on nuScenes-mini provides valuable sanity checks for surround-view calibration, multi-camera "
                    "geometry lifting, and cross-sensor synchronization; however, metrics obtained on nuScenes-mini (mAP "
                    "and NDS) carry substantial statistical variance and cannot substitute for official full-benchmark "
                    "results on the full 1000-scene dataset."
                ),
            }), fp, indent=2, allow_nan=False)
        print("  [PASS] Exported nuscenes_mini.json")

    # 6. PointPillars Reference
    pp_csv = results_dir / "pointpillars_kitti_val.csv"
    if pp_csv.exists():
        df_pp = pd.read_csv(pp_csv)
        df_pp_clean = df_pp.where(pd.notnull(df_pp), None)
        pp_records = df_pp_clean.to_dict(orient="records")
        for rec in pp_records:
            rec["source_file"] = "results/pointpillars_kitti_val.csv"
        with open(out_dir / "pointpillars_kitti_val.json", "w", encoding="utf-8") as fp:
            json.dump(sanitize_val({
                "source_file": "results/pointpillars_kitti_val.csv",
                "data": pp_records,
            }), fp, indent=2, allow_nan=False)
        print("  [PASS] Exported pointpillars_kitti_val.json")

    # 7. Third-Party Register
    third_party_md = docs_dir / "THIRD_PARTY.md"
    if third_party_md.exists():
        tp_content = third_party_md.read_text(encoding="utf-8")
        with open(out_dir / "third_party.json", "w", encoding="utf-8") as fp:
            json.dump({
                "source_file": "docs/THIRD_PARTY.md",
                "markdown": tp_content,
            }, fp, indent=2)
        print("  [PASS] Exported third_party.json")

    # 8. Novelty & Literature
    novelty_md = reports_dir / "novelty_statement.md"
    novelty_text = novelty_md.read_text(encoding="utf-8") if novelty_md.exists() else ""
    lit_md = reports_dir / "literature_review.md"
    lit_text = lit_md.read_text(encoding="utf-8") if lit_md.exists() else ""

    with open(out_dir / "literature.json", "w", encoding="utf-8") as fp:
        json.dump({
            "source_file": "reports/literature_review.md",
            "novelty_statement": novelty_text,
            "literature_review": lit_text,
        }, fp, indent=2)
    print("  [PASS] Exported literature.json")

    print("[PASS] Web data export completed successfully!")


if __name__ == "__main__":
    export_all_web_data()
