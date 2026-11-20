"""M5 computational-cost measurements for a loaded model."""
import time
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

@torch.inference_mode()
def benchmark_forward(model, sample, widths, device=None, warmup=3, repeats=20):
    device = torch.device(device or next(model.parameters()).device)
    model = model.to(device).eval()
    sample, widths = sample.to(device), widths.to("cpu")
    for _ in range(warmup):
        model(sample, widths)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(sample, widths)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    result = {
        "device": str(device),
        "repeats": repeats,
        "batch_size": int(sample.shape[0]),
        "inference_ms_per_image": elapsed * 1000 / (repeats * sample.shape[0]),
        "parameter_count": count_parameters(model),
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1048576
    return result

def checkpoint_size_mb(path):
    from pathlib import Path
    return Path(path).stat().st_size / 1048576
