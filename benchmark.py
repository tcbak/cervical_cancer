import os
import json
import time
import argparse
from datetime import datetime

import torch

from model import (
    get_model,
    get_input_size,
    get_model_info
)

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_model_size_mb(model, tmp_path="._tmp_size_check.pth"):
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


def measure_flops(model, input_tensor):
    if not HAS_THOP:
        return None, None
    model_copy = model
    macs, params = thop_profile(model_copy, inputs=(input_tensor,), verbose=False)
    # thop reports MACs; report both MACs and the conventional "FLOPs" (~2x MACs)
    return macs, macs * 2


def measure_latency(model, input_tensor, device, n_warmup=10, n_runs=50):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    avg_latency_ms = (end - start) / n_runs * 1000
    throughput_fps = 1000.0 / avg_latency_ms
    return avg_latency_ms, throughput_fps

# ==========================================================
# Helper Functions
# ==========================================================

def save_json(data, filename):
    """
    Save dictionary as JSON.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def create_benchmark_folder(output_root, model_name):
    """
    Creates

    benchmark_results/
        model_name/
            benchmark/
    """

    benchmark_folder = os.path.join(
        output_root,
        model_name,
        "benchmark"
    )

    os.makedirs(
        benchmark_folder,
        exist_ok=True
    )

    return benchmark_folder


def run_benchmark(args):

    print("=" * 80)
    print("Model Benchmark")
    print("=" * 80)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device : {device}")

    model_info = get_model_info(args.model_name)

    print(f"Model  : {model_info['display_name']}")
    print(f"Type   : {model_info['category']}")

    benchmark_folder = create_benchmark_folder(
        args.output_dir,
        args.model_name
    )

    h, w = get_input_size(args.model_name)

    print(f"Input Size : ({h}, {w})")

    batch1 = torch.randn(
        1,
        3,
        h,
        w
    )

    benchmark_start = time.time()

    print("\nBuilding Model...")

    model = get_model(
        num_classes=args.num_classes,
        model_name=args.model_name,
        pretrained=True,
        freeze_backbone=False
    )

    total_params, trainable_params = count_params(model)
    size_mb = measure_model_size_mb(model)
    macs, flops = measure_flops(model, batch1)

    # --- CPU latency (always measured -- relevant to low-resource deployment) ---
    cpu_latency_ms, cpu_fps = measure_latency(model, batch1, torch.device("cpu"))

    # --- GPU latency + peak memory (only if CUDA available) ---
    gpu_latency_ms, gpu_fps, gpu_peak_mem_mb = None, None, None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_latency_ms, gpu_fps = measure_latency(model, batch1, torch.device("cuda"))
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    benchmark_time = time.time() - benchmark_start

    result = {

        "model_name": args.model_name,

        "display_name": model_info["display_name"],

        "category": model_info["category"],

        "date": str(datetime.now()),

        "input_size": f"{h}x{w}",

        "benchmark_time_seconds": round(benchmark_time, 2),

        "total_parameters": total_params,

        "trainable_parameters": trainable_params,

        "total_parameters_million": round(total_params / 1e6, 3),

        "model_size_mb": round(size_mb, 3),

        "macs": macs,

        "macs_million": round(macs / 1e6, 2) if macs else None,

        "flops_million": round(flops / 1e6, 2) if flops else None,

        "cpu_latency_ms": round(cpu_latency_ms, 3),

        "cpu_throughput_fps": round(cpu_fps, 2),

        "gpu_latency_ms": round(gpu_latency_ms, 3) if gpu_latency_ms else None,

        "gpu_throughput_fps": round(gpu_fps, 2) if gpu_fps else None,

        "gpu_peak_memory_mb": round(gpu_peak_mem_mb, 2) if gpu_peak_mem_mb else None
    }
    print(json.dumps(result, indent=2))

    if not HAS_THOP:
        print("\n[Note] `thop` is not installed, so MACs/FLOPs are unavailable.")
        print("Install it with:  pip install thop --break-system-packages")

    benchmark_file = os.path.join(

        benchmark_folder,

        "benchmark.json"

    )

    save_json(

        result,

        benchmark_file

    )

    print("\nBenchmark Results")

    print(json.dumps(result, indent=4))

    print("\nBenchmark saved to:")

    print(benchmark_file)

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark a model's efficiency / lightweightness"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="efficientnet_b0, resnet18, resnet50, densenet121, mobilenet_v2, shufflenet_v2_x1_0, inception_v3, vgg16"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=4
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results"
    )

    args = parser.parse_args()

    run_benchmark(args)