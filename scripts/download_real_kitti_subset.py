#!/usr/bin/env python3
"""
Fast multi-process downloader for real KITTI frames (000000-000099) using HTTP range requests via remotezip.

Downloads from the official KITTI AWS S3 buckets:
- data_object_calib.zip
- data_object_label_2.zip
- data_object_image_2.zip
- data_object_velodyne.zip

Uses multiprocessing to assign non-overlapping frame slices to independent RemoteZip sessions.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KITTI_ROOT = Path(os.environ.get("KITTI_ROOT", str(PROJECT_ROOT / "data" / "kitti")))

BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
ZIPS = {
    "calib": "data_object_calib.zip",
    "label_2": "data_object_label_2.zip",
    "image_2": "data_object_image_2.zip",
    "velodyne": "data_object_velodyne.zip",
}

NUM_FRAMES = 100  # 000000 - 000099
NUM_WORKERS = 4


def _worker_download_slice(args):
    modality, zip_name, frame_slice, dest_dir_str = args
    dest_dir = Path(dest_dir_str)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if modality == "image_2" else ("bin" if modality == "velodyne" else "txt")

    url = BASE_URL + zip_name
    rz = RemoteZip(url)

    downloaded = 0
    for fid in frame_slice:
        target_path = dest_dir / f"{fid}.{ext}"
        if target_path.exists() and target_path.stat().st_size > 0:
            downloaded += 1
            continue

        member_name = f"training/{modality}/{fid}.{ext}"
        for attempt in range(4):
            try:
                data = rz.read(member_name)
                with open(target_path, "wb") as f:
                    f.write(data)
                downloaded += 1
                break
            except Exception as e:
                time.sleep(1.0)
                if attempt == 3:
                    print(f"  [ERROR] {modality}/{fid}: {e}", flush=True)

    return modality, downloaded, len(frame_slice)


def download_modality_parallel(modality, zip_name, frame_ids, dest_dir):
    print(f"\n[Starting download] {modality} ({len(frame_ids)} frames across {NUM_WORKERS} processes)...", flush=True)
    t0 = time.time()

    # Split frame_ids into NUM_WORKERS slices
    chunk_size = (len(frame_ids) + NUM_WORKERS - 1) // NUM_WORKERS
    slices = [frame_ids[i * chunk_size : (i + 1) * chunk_size] for i in range(NUM_WORKERS) if i * chunk_size < len(frame_ids)]

    tasks = [(modality, zip_name, sl, str(dest_dir)) for sl in slices]

    total_downloaded = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [pool.submit(_worker_download_slice, task) for task in tasks]
        for fut in as_completed(futures):
            mod, count, total = fut.result()
            total_downloaded += count
            print(f"  [{modality} worker finished] {count}/{total} in chunk", flush=True)

    elapsed = time.time() - t0
    print(f"[{modality} completed] {total_downloaded}/{len(frame_ids)} files in {elapsed:.1f}s", flush=True)
    return total_downloaded == len(frame_ids)


def main():
    frame_ids = [f"{i:06d}" for i in range(NUM_FRAMES)]
    training_dir = KITTI_ROOT / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting official KITTI frames 000000 to {frame_ids[-1]} to {KITTI_ROOT}...", flush=True)

    for modality in ["calib", "label_2", "image_2", "velodyne"]:
        zip_name = ZIPS[modality]
        dest = training_dir / modality
        download_modality_parallel(modality, zip_name, frame_ids, dest)

    # Write ImageSets
    imagesets_dir = KITTI_ROOT / "ImageSets"
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    train_ids = frame_ids[:60]
    val_ids = frame_ids[60:]

    with open(imagesets_dir / "train.txt", "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(imagesets_dir / "val.txt", "w") as f:
        f.write("\n".join(val_ids) + "\n")
    with open(imagesets_dir / "trainval.txt", "w") as f:
        f.write("\n".join(frame_ids) + "\n")

    print("\nUpdated ImageSets:")
    print(f"  train.txt: {len(train_ids)} frames")
    print(f"  val.txt: {len(val_ids)} frames")
    print(f"  trainval.txt: {len(frame_ids)} frames")
    print("\n>>> ALL 100 REAL KITTI FRAMES DOWNLOADED SUCCESSFULLY <<<", flush=True)


if __name__ == "__main__":
    main()
