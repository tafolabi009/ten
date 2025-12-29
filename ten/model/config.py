"""
TEN Configuration Classes
=========================

Configuration dataclasses for TEN and HTEN architectures.

Reference: Paper Section 3.6 (Architecture Details) and Section 6.1 (Experimental Setup)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


@dataclass
class TENConfig:
    """
    Configuration for Temporal Eigenstate Networks (TEN).
    
    Architecture Parameters (Section 3.6):
    - K eigenstates (typically K ~ sqrt(d))
    - Complex eigenvalues λk = exp(αk + i*ωk)
    - Resonance coupling R = I + ε*M with ||ε|| << 1
    
    Default values from Section 6.1:
    - K = 64 eigenstates
    - d = 512 hidden dimension  
    - L = 6 layers
    """
    # Model dimensions
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    hidden_dim: int = 512    # d in paper
    num_eigenstates: int = 64  # K in paper (K ~ sqrt(d))
    num_layers: int = 6      # L layers
    
    # Eigenvalue initialization (Appendix B.2)
    alpha_init_range: Tuple[float, float] = (-3.0, 0.0)  # Decay rates
    omega_init_scale: float = 1.0  # ωk = 2πk/K * scale
    
    # Resonance coupling (Section 3.4)
    # R = I + ε*M where ||ε|| << 1
    resonance_epsilon: float = 0.01
    use_resonance: bool = True
    
    # Feedforward (Section 3.6)
    ffn_hidden_dim: Optional[int] = None  # Default: 4 * hidden_dim
    ffn_dropout: float = 0.1
    
    # Regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    
    # Numerical stability (Appendix B.1)
    eigenvalue_constraint: str = "sigmoid"  # Ensure |λk| <= 1 for stability
    gradient_clip_norm: float = 1.0
    
    # Positional encoding
    max_seq_length: int = 8192
    use_positional_encoding: bool = True
    
    # Initialization (Appendix B.2)
    init_std: float = 0.02
    
    def __post_init__(self):
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = 4 * self.hidden_dim
        
        # Validate K ~ sqrt(d) recommendation
        recommended_k = int(math.sqrt(self.hidden_dim))
        if self.num_eigenstates < recommended_k // 2:
            import warnings
            warnings.warn(
                f"num_eigenstates={self.num_eigenstates} is much smaller than "
                f"recommended sqrt(hidden_dim)={recommended_k}. This may reduce capacity."
            )


@dataclass  
class HTENConfig(TENConfig):
    """
    Configuration for Hierarchical Temporal Eigenstate Networks (HTEN).
    
    Multi-scale processing (Section 5):
    - Processes input at scales s ∈ {1, 2, 4, 8}
    - Each scale has K/|S| eigenstates
    - Total complexity O(TKd log|S|), still linear in T
    
    Default values from Section 6.1:
    - 4 scales {1, 2, 4, 8}
    - K = 16 eigenstates per scale (64 total)
    """
    # Multi-scale parameters (Section 5.1)
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    eigenstates_per_scale: Optional[int] = None  # Default: K / len(scales)
    
    # Scale fusion
    scale_fusion: str = "learned_weights"  # or "concat", "attention"
    
    # Downsampling method
    downsample_method: str = "avg_pool"  # or "conv", "strided"
    
    # Upsampling method  
    upsample_method: str = "linear_interp"  # or "conv_transpose", "nearest"
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.eigenstates_per_scale is None:
            self.eigenstates_per_scale = self.num_eigenstates // len(self.scales)
        
        # Validate that eigenstates divide evenly
        total_eigenstates = self.eigenstates_per_scale * len(self.scales)
        if total_eigenstates != self.num_eigenstates:
            import warnings
            warnings.warn(
                f"num_eigenstates={self.num_eigenstates} doesn't divide evenly by "
                f"{len(self.scales)} scales. Using {total_eigenstates} total eigenstates."
            )
            self.num_eigenstates = total_eigenstates


@dataclass
class TrainingConfig:
    """
    Training hyperparameters from Appendix B.3.
    """
    # Optimizer (AdamW)
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    
    # Schedule
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"
    
    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Training duration
    max_steps: int = 100000
    eval_every: int = 1000
    save_every: int = 5000
    
    # Regularization
    gradient_clip_norm: float = 1.0
    dropout: float = 0.1
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # or "bfloat16"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging
    log_every: int = 100
    use_wandb: bool = True
    wandb_project: str = "temporal-eigenstate-networks"


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarking experiments.
    """
    # Sequence lengths to test
    seq_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    
    # Batch sizes
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    
    # Number of iterations for timing
    warmup_iters: int = 10
    benchmark_iters: int = 100
    
    # Metrics to collect
    measure_throughput: bool = True
    measure_memory: bool = True
    measure_latency: bool = True
    
    # Baselines to compare
    baselines: List[str] = field(default_factory=lambda: ["transformer", "s4"])
