"""
Benchmark Utilities
===================

Tools for running efficiency and accuracy benchmarks.

Reference: Section 6.4 (Efficiency Analysis), Figure 1, Appendix F
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import gc

import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    seq_length: int
    batch_size: int
    
    # Timing
    latency_ms: float
    throughput_tokens_per_sec: float
    
    # Memory
    peak_memory_mb: float
    allocated_memory_mb: float
    
    # Model info
    num_params: int
    
    # Optional accuracy metrics
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """
    Benchmark runner for comparing TEN against baselines.
    
    Measures:
    - Wall-clock training time (Table 1)
    - Throughput (tokens/sec)
    - Memory usage (Table 1)
    - Convergence speed
    - Task-level accuracy/loss
    """
    
    def __init__(
        self,
        device: str = "cuda",
        warmup_iters: int = 10,
        benchmark_iters: int = 100,
        use_amp: bool = True,
    ):
        self.device = torch.device(device)
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.use_amp = use_amp and device == "cuda"
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    def _clear_memory(self):
        """Clear GPU memory cache."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if self.device.type != "cuda":
            return {"peak_mb": 0, "allocated_mb": 0}
        
        return {
            "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        }
    
    def benchmark_forward(
        self,
        model: nn.Module,
        seq_length: int,
        batch_size: int,
        model_name: str = "model",
    ) -> BenchmarkResult:
        """
        Benchmark forward pass latency and throughput.
        
        Args:
            model: Model to benchmark
            seq_length: Sequence length to test
            batch_size: Batch size to test
            model_name: Name for identification
        
        Returns:
            BenchmarkResult with timing and memory metrics
        """
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        input_ids = torch.randint(
            0, 50000, (batch_size, seq_length), 
            device=self.device, dtype=torch.long
        )
        
        self._clear_memory()
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                if self.use_amp:
                    with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                        _ = model(input_ids)
                else:
                    _ = model(input_ids)
        
        # Synchronize before timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        self._clear_memory()
        
        # Benchmark
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(self.benchmark_iters):
                if self.use_amp:
                    with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                        _ = model(input_ids)
                else:
                    _ = model(input_ids)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Compute metrics
        total_time = end_time - start_time
        avg_latency = total_time / self.benchmark_iters * 1000  # ms
        
        total_tokens = batch_size * seq_length * self.benchmark_iters
        throughput = total_tokens / total_time
        
        memory_stats = self._get_memory_stats()
        
        # Get param count
        num_params = sum(p.numel() for p in model.parameters())
        
        return BenchmarkResult(
            model_name=model_name,
            seq_length=seq_length,
            batch_size=batch_size,
            latency_ms=avg_latency,
            throughput_tokens_per_sec=throughput,
            peak_memory_mb=memory_stats["peak_mb"],
            allocated_memory_mb=memory_stats["allocated_mb"],
            num_params=num_params,
        )
    
    def benchmark_training(
        self,
        model: nn.Module,
        seq_length: int,
        batch_size: int,
        model_name: str = "model",
        num_steps: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark training step (forward + backward).
        
        Args:
            model: Model to benchmark
            seq_length: Sequence length
            batch_size: Batch size
            model_name: Name for identification
            num_steps: Number of training steps
        
        Returns:
            BenchmarkResult with training metrics
        """
        model = model.to(self.device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Create dummy data
        input_ids = torch.randint(
            0, 50000, (batch_size, seq_length),
            device=self.device, dtype=torch.long
        )
        labels = input_ids.clone()
        
        self._clear_memory()
        
        # Warmup
        for _ in range(self.warmup_iters):
            optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        self._clear_memory()
        
        # Benchmark training steps
        start_time = time.perf_counter()
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_latency = total_time / num_steps * 1000  # ms
        
        total_tokens = batch_size * seq_length * num_steps
        throughput = total_tokens / total_time
        
        memory_stats = self._get_memory_stats()
        num_params = sum(p.numel() for p in model.parameters())
        
        return BenchmarkResult(
            model_name=model_name,
            seq_length=seq_length,
            batch_size=batch_size,
            latency_ms=avg_latency,
            throughput_tokens_per_sec=throughput,
            peak_memory_mb=memory_stats["peak_mb"],
            allocated_memory_mb=memory_stats["allocated_mb"],
            num_params=num_params,
            loss=loss.item(),
            metadata={"mode": "training"},
        )


def run_benchmark(
    models: Dict[str, nn.Module],
    seq_lengths: List[int] = [512, 1024, 2048, 4096, 8192],
    batch_sizes: List[int] = [1, 8, 32],
    device: str = "cuda",
    mode: str = "forward",
) -> List[BenchmarkResult]:
    """
    Run benchmark across multiple models and configurations.
    
    Args:
        models: Dict of model_name -> model
        seq_lengths: Sequence lengths to test
        batch_sizes: Batch sizes to test
        device: Device to run on
        mode: "forward" or "training"
    
    Returns:
        List of BenchmarkResult
    """
    benchmark = Benchmark(device=device)
    results = []
    
    for model_name, model in models.items():
        for seq_len in seq_lengths:
            for batch_size in batch_sizes:
                try:
                    print(f"Benchmarking {model_name}: seq_len={seq_len}, batch={batch_size}")
                    
                    if mode == "forward":
                        result = benchmark.benchmark_forward(
                            model, seq_len, batch_size, model_name
                        )
                    else:
                        result = benchmark.benchmark_training(
                            model, seq_len, batch_size, model_name
                        )
                    
                    results.append(result)
                    print(f"  Latency: {result.latency_ms:.2f}ms, "
                          f"Throughput: {result.throughput_tokens_per_sec:.0f} tok/s, "
                          f"Memory: {result.peak_memory_mb:.1f}MB")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  OOM for seq_len={seq_len}, batch={batch_size}")
                        results.append(BenchmarkResult(
                            model_name=model_name,
                            seq_length=seq_len,
                            batch_size=batch_size,
                            latency_ms=float('inf'),
                            throughput_tokens_per_sec=0,
                            peak_memory_mb=float('inf'),
                            allocated_memory_mb=float('inf'),
                            num_params=0,
                            metadata={"error": "OOM"},
                        ))
                        benchmark._clear_memory()
                    else:
                        raise
    
    return results


def compare_models(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """
    Generate comparison summary from benchmark results.
    
    Reference: Table 1, Figure 1
    """
    import pandas as pd
    
    # Convert to DataFrame
    data = []
    for r in results:
        data.append({
            "model": r.model_name,
            "seq_len": r.seq_length,
            "batch_size": r.batch_size,
            "latency_ms": r.latency_ms,
            "throughput": r.throughput_tokens_per_sec,
            "memory_mb": r.peak_memory_mb,
            "params": r.num_params,
        })
    
    df = pd.DataFrame(data)
    
    # Compute speedups relative to Transformer
    summary = {}
    
    for seq_len in df["seq_len"].unique():
        seq_df = df[df["seq_len"] == seq_len]
        
        transformer_latency = seq_df[seq_df["model"] == "transformer"]["latency_ms"].values
        transformer_memory = seq_df[seq_df["model"] == "transformer"]["memory_mb"].values
        
        if len(transformer_latency) > 0 and transformer_latency[0] != float('inf'):
            for model in seq_df["model"].unique():
                model_latency = seq_df[seq_df["model"] == model]["latency_ms"].values[0]
                model_memory = seq_df[seq_df["model"] == model]["memory_mb"].values[0]
                
                if model_latency != float('inf'):
                    speedup = transformer_latency[0] / model_latency
                    memory_reduction = transformer_memory[0] / model_memory if model_memory > 0 else float('inf')
                    
                    key = f"{model}_seq{seq_len}"
                    summary[key] = {
                        "speedup": speedup,
                        "memory_reduction": memory_reduction,
                        "latency_ms": model_latency,
                        "memory_mb": model_memory,
                    }
    
    return {"summary": summary, "dataframe": df}
