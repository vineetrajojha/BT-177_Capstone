#!/usr/bin/env python3
"""
BEVFusionLite Efficiency & Latency Benchmarking Tool.

Measures:
  1. Parameters: Total, trainable, and breakdown per module.
  2. Computational Complexity: GFLOPs via fvcore FlopCountAnalysis (batch size 1).
  3. Latency & Throughput: 50 warmup iterations, 300 timed runs, synchronized.
  4. Memory: Peak CUDA allocation (or CPU RSS).
  5. Checkpoint Size: Size in MB.
  6. Environment Metadata: GPU name, PyTorch version, precision, official flag.

Output is saved to JSON at `results/<run>/efficiency.json`.
"""

import os
import sys
import time
import json
import argparse
import platform
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"
if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False


def count_parameters(model: nn.Module):
    """Computes total and per-module parameter breakdown."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    module_breakdown = {}
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_breakdown[name] = {
            "total_params": mod_params,
            "trainable_params": mod_trainable,
            "params_millions": round(mod_params / 1e6, 3),
        }

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_millions": round(total_params / 1e6, 3),
        "module_breakdown": module_breakdown,
    }


def measure_flops_fvcore(model: nn.Module, dummy_inputs, exclusions=None):
    """Measures GFLOPs using fvcore FlopCountAnalysis."""
    if not FVCORE_AVAILABLE:
        return {"gflops": None, "note": "fvcore not installed"}

    model.eval()
    try:
        # If model expects a dict or tuple
        flops = FlopCountAnalysis(model, dummy_inputs)
        if exclusions:
            flops.set_op_handle(**{op: None for op in exclusions})
        total_flops = flops.total()
        return {
            "total_flops": int(total_flops),
            "gflops": round(total_flops / 1e9, 3),
            "exclusions": exclusions or [],
        }
    except Exception as e:
        return {
            "gflops": None,
            "error": str(e),
            "note": "FlopCountAnalysis requires tensor inputs or traced submodules",
        }


def measure_latency_and_memory(
    model: nn.Module,
    dummy_input_fn,
    warmup_runs: int = 50,
    timed_runs: int = 300,
    device: torch.device = None,
    precision: str = "fp32",
):
    """Measures model execution latency with synchronized timing and peak memory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    use_cuda = device.type == "cuda"
    use_half = precision == "fp16" and use_cuda

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Pre-generate or call input function
    batch = dummy_input_fn(device, use_half)

    # Context for mixed precision if fp16
    autocast_ctx = torch.cuda.amp.autocast() if use_half else torch.no_grad()

    # 1. Warm-up
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(warmup_runs):
                _ = model(batch)
                if use_cuda:
                    torch.cuda.synchronize()

    # 2. Timed runs (Model-only)
    latencies = []
    with torch.no_grad():
        with autocast_ctx:
            for _ in range(timed_runs):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                _ = model(batch)

                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    std_latency = float(np.std(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    fps = round(1000.0 / mean_latency, 2) if mean_latency > 0 else 0.0

    # 3. Peak memory
    if use_cuda:
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_mb = round(peak_mem_bytes / (1024 * 1024), 2)
    else:
        try:
            import resource
            peak_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On macOS, ru_maxrss is in bytes; on Linux, in kilobytes
            if platform.system() == "Darwin":
                peak_mem_mb = round(peak_mem_kb / (1024 * 1024), 2)
            else:
                peak_mem_mb = round(peak_mem_kb / 1024, 2)
        except Exception:
            peak_mem_mb = None

    return {
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "latency_ms": {
            "mean": round(mean_latency, 2),
            "median": round(median_latency, 2),
            "std": round(std_latency, 3),
            "min": round(min_latency, 2),
            "max": round(max_latency, 2),
            "p95": round(p95_latency, 2),
            "p99": round(p99_latency, 2),
        },
        "fps": fps,
        "peak_memory_mb": peak_mem_mb,
    }


def benchmark_run(
    model_name: str,
    model: nn.Module,
    dummy_input_fn,
    output_dir: str,
    checkpoint_path: str = None,
    precision: str = "fp32",
    warmup_runs: int = 50,
    timed_runs: int = 300,
):
    """Complete benchmarking suite producing efficiency.json."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # Official check: T4 GPU
    is_official = ("t4" in gpu_name.lower()) if torch.cuda.is_available() else False

    # 1. Parameter counts
    param_info = count_parameters(model)

    # 2. Latency & Memory
    perf_info = measure_latency_and_memory(
        model=model,
        dummy_input_fn=dummy_input_fn,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        device=device,
        precision=precision,
    )

    # 3. Checkpoint size
    ckpt_size_mb = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_size_bytes = os.path.getsize(checkpoint_path)
        ckpt_size_mb = round(ckpt_size_bytes / (1024 * 1024), 2)

    # 4. Assemble report dictionary
    efficiency_report = {
        "model_name": model_name,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "precision": precision,
        "batch_size": 1,
        "official": is_official,
        "checkpoint_mb": ckpt_size_mb,
        "parameters": param_info,
        "performance": perf_info,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = Path(output_dir) / "efficiency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(efficiency_report, f, indent=2)

    print(f"\n============================================================")
    print(f"BENCHMARK REPORT: {model_name} ({precision.upper()})")
    print(f"============================================================")
    print(f"Hardware:        {gpu_name} (Official T4: {is_official})")
    print(f"Total Params:    {param_info['total_params_millions']} M")
    print(f"Mean Latency:    {perf_info['latency_ms']['mean']} ms")
    print(f"Throughput:      {perf_info['fps']} FPS")
    print(f"Peak Memory:     {perf_info['peak_memory_mb']} MB")
    if ckpt_size_mb is not None:
        print(f"Checkpoint Size: {ckpt_size_mb} MB")
    print(f"Saved to:        {out_path}")
    print(f"============================================================\n")

    return efficiency_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model efficiency and latency.")
    parser.add_argument("--config", type=str, default="configs/lidar_only_pointpillars.yaml")
    parser.add_argument("--output_dir", type=str, default="results/pointpillars_baseline")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300)
    args = parser.parse_args()

    print(f"[i] Benchmarking script initialized with config: {args.config}")
