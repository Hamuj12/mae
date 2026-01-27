#!/usr/bin/env python3
"""
Benchmark YOLOv8 PyTorch (.pt) vs TensorRT (.engine).

Metrics:
- warmup time
- mean / p50 / p90 / p99 latency
- FPS
- GPU memory usage (allocated / reserved)
- parameter count (PyTorch only)
- model file size
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean
import numpy as np
import torch

from ultralytics import YOLO


# -------------------------
# Utilities
# -------------------------
def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def count_params(model: YOLO) -> int:
    # Only valid for PyTorch models
    try:
        return sum(p.numel() for p in model.model.parameters())
    except Exception:
        return -1


def percentile(vals, p):
    return float(np.percentile(vals, p))


# -------------------------
# Benchmark core
# -------------------------
def benchmark_model(
    model_path: Path,
    images: list[Path],
    imgsz: int,
    device: str,
    warmup_iters: int,
    iters: int,
):
    print(f"\n=== Benchmarking: {model_path.name} ===")

    model = YOLO(str(model_path))

    # Reset CUDA stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # -------------------------
    # Warmup
    # -------------------------
    start = time.time()
    for i in range(warmup_iters):
        model.predict(
            source=str(images[i % len(images)]),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
    torch.cuda.synchronize()
    warmup_time = time.time() - start

    # -------------------------
    # Timed inference
    # -------------------------
    latencies_ms = []

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for i in range(iters):
        img = images[i % len(images)]
        starter.record()
        model.predict(
            source=str(img),
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        ender.record()
        torch.cuda.synchronize()
        latencies_ms.append(starter.elapsed_time(ender))

    # -------------------------
    # Memory
    # -------------------------
    mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    # -------------------------
    # Params + size
    # -------------------------
    params = count_params(model)
    file_size = get_file_size_mb(model_path)

    # -------------------------
    # Summary
    # -------------------------
    results = {
        "model": model_path.name,
        "backend": "TensorRT" if model_path.suffix == ".engine" else "PyTorch",
        "warmup_time_s": warmup_time,
        "mean_latency_ms": mean(latencies_ms),
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p90_latency_ms": percentile(latencies_ms, 90),
        "p99_latency_ms": percentile(latencies_ms, 99),
        "fps": 1000.0 / mean(latencies_ms),
        "gpu_mem_allocated_mb": mem_alloc,
        "gpu_mem_reserved_mb": mem_reserved,
        "params_million": params / 1e6 if params > 0 else None,
        "model_size_mb": file_size,
    }

    return results


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", type=str, required=True, help="YOLOv8 PyTorch .pt model")
    p.add_argument("--engine", type=str, required=True, help="YOLOv8 TensorRT .engine model")
    p.add_argument("--imgdir", type=str, required=True, help="Folder with images")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — this benchmark is GPU-only.")

    imgdir = Path(args.imgdir)
    images = sorted(
        p for p in imgdir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise RuntimeError("No images found for benchmarking.")

    pt_results = benchmark_model(
        Path(args.pt),
        images,
        args.imgsz,
        device,
        args.warmup,
        args.iters,
    )

    trt_results = benchmark_model(
        Path(args.engine),
        images,
        args.imgsz,
        device,
        args.warmup,
        args.iters,
    )

    # -------------------------
    # Pretty print
    # -------------------------
    print("\n=== Benchmark Summary ===")
    for r in (pt_results, trt_results):
        print(f"\nModel: {r['model']} ({r['backend']})")
        print(f"  Warmup time:        {r['warmup_time_s']:.2f} s")
        print(f"  Mean latency:       {r['mean_latency_ms']:.2f} ms")
        print(f"  p50 / p90 / p99:    {r['p50_latency_ms']:.2f} / "
              f"{r['p90_latency_ms']:.2f} / {r['p99_latency_ms']:.2f} ms")
        print(f"  FPS:                {r['fps']:.1f}")
        print(f"  GPU mem allocated:  {r['gpu_mem_allocated_mb']:.1f} MB")
        print(f"  GPU mem reserved:   {r['gpu_mem_reserved_mb']:.1f} MB")
        print(f"  Params:             "
              f"{r['params_million']:.2f} M" if r["params_million"] else "  Params: N/A")
        print(f"  Model size on disk: {r['model_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
    
# PYTHONPATH=. python3 benchmark_inference.py \
#   --pt /home/hm25936/mae/runs/yolov8_fda/baseline/weights/best.pt \
#   --engine yolov8_fda_fp16.engine \
#   --imgdir /home/hm25936/datasets_for_yolo/midLighting_rmag_5m_to_100m/images/test \
#   --imgsz 1024 \
#   --warmup 20 \
#   --iters 200 \
#   --device cuda:0