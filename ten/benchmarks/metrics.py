"""
Metrics Utilities
=================

Functions for measuring throughput, memory, and latency.
"""

import time
from typing import Callable, Dict, Any, Optional
import gc

import torch
import torch.nn as nn


def measure_throughput(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_amp: bool = True,
) -> float:
    """
    Measure model throughput in tokens per second.
    
    Args:
        model: Model to measure
        input_ids: Input tensor
        num_iterations: Number of iterations for measurement
        warmup_iterations: Number of warmup iterations
        use_amp: Use automatic mixed precision
    
    Returns:
        Throughput in tokens per second
    """
    device = next(model.parameters()).device
    model.eval()
    
    batch_size, seq_len = input_ids.shape
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(input_ids)
            else:
                _ = model(input_ids)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(input_ids)
            else:
                _ = model(input_ids)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_tokens = batch_size * seq_len * num_iterations
    
    return total_tokens / total_time


def measure_memory(
    model: nn.Module,
    input_ids: torch.Tensor,
    include_backward: bool = False,
) -> Dict[str, float]:
    """
    Measure peak GPU memory usage.
    
    Args:
        model: Model to measure
        input_ids: Input tensor
        include_backward: Include backward pass memory
    
    Returns:
        Dict with memory statistics in MB
    """
    device = next(model.parameters()).device
    
    if device.type != "cuda":
        return {"peak_mb": 0, "allocated_mb": 0}
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    if include_backward:
        model.train()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        loss.backward()
    else:
        model.eval()
        with torch.no_grad():
            _ = model(input_ids)
    
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "peak_mb": peak_memory,
        "allocated_mb": allocated_memory,
    }


def measure_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_amp: bool = True,
    percentiles: tuple = (50, 90, 99),
) -> Dict[str, float]:
    """
    Measure model latency with percentile statistics.
    
    Args:
        model: Model to measure
        input_ids: Input tensor
        num_iterations: Number of iterations
        warmup_iterations: Number of warmup iterations
        use_amp: Use automatic mixed precision
        percentiles: Percentiles to compute
    
    Returns:
        Dict with latency statistics in milliseconds
    """
    device = next(model.parameters()).device
    model.eval()
    
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(input_ids)
            else:
                _ = model(input_ids)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure individual iterations
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(input_ids)
            else:
                _ = model(input_ids)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    import numpy as np
    latencies = np.array(latencies)
    
    result = {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
    }
    
    for p in percentiles:
        result[f"p{p}_ms"] = np.percentile(latencies, p)
    
    return result


def count_flops(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cuda",
) -> int:
    """
    Estimate FLOPs for model forward pass.
    
    This is an approximation based on model structure.
    
    Returns:
        Estimated FLOPs
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        
        input_ids = torch.randint(0, 50000, input_shape, device=device)
        
        model = model.to(device)
        model.eval()
        
        flops = FlopCountAnalysis(model, (input_ids,))
        return flops.total()
    except ImportError:
        # Fallback: estimate based on parameter count
        num_params = sum(p.numel() for p in model.parameters())
        # Rough estimate: 2 FLOPs per parameter per token
        batch_size, seq_len = input_shape
        return num_params * 2 * seq_len


def profile_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Profile model execution with PyTorch profiler.
    
    Args:
        model: Model to profile
        input_ids: Input tensor
        output_path: Optional path to save Chrome trace
    
    Returns:
        Profiling statistics
    """
    device = next(model.parameters()).device
    
    if device.type != "cuda":
        return {}
    
    model.eval()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _ = model(input_ids)
    
    # Get key statistics
    key_averages = prof.key_averages()
    
    # Top operations by CUDA time
    top_ops = []
    for event in sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:10]:
        top_ops.append({
            "name": event.key,
            "cuda_time_ms": event.cuda_time_total / 1000,
            "cpu_time_ms": event.cpu_time_total / 1000,
            "calls": event.count,
        })
    
    result = {
        "top_operations": top_ops,
        "total_cuda_time_ms": sum(e.cuda_time_total for e in key_averages) / 1000,
    }
    
    if output_path:
        prof.export_chrome_trace(output_path)
        result["trace_path"] = output_path
    
    return result
