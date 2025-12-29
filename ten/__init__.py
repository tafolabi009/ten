"""
Temporal Eigenstate Networks (TEN)
===================================

A novel O(T) complexity architecture for sequence modeling based on spectral operator theory.

Paper: "Temporal Eigenstate Networks: Linear-Complexity Sequence Modeling via Spectral Decomposition"
Venue: AAAI AIDD 2026 (Applied AI for Drug Discovery Workshop)

Key Features:
- O(TKd) complexity vs O(T²d) for Transformers
- Eigenstate decomposition with learned basis functions
- Stable gradients via Lyapunov-bounded eigenvalue dynamics
- Multi-scale processing via Hierarchical TEN (HTEN)

Modules:
- ten.model: Core TEN and HTEN architectures
- ten.backends.pytorch: PyTorch/CUDA implementation
- ten.backends.jax: JAX/XLA implementation for TPU
- ten.training: Training loops, optimizers, schedulers
- ten.benchmarks: Efficiency and accuracy benchmarking
- ten.evaluation: Drug discovery evaluation tasks
"""

__version__ = "1.0.0"
__author__ = "TEN Research Team"

from ten.model import TEN, HTEN, TENConfig, HTENConfig
from ten.model.language_model import TENForLanguageModeling, HTENForLanguageModeling

__all__ = [
    "TEN",
    "HTEN", 
    "TENConfig",
    "HTENConfig",
    "TENForLanguageModeling",
    "HTENForLanguageModeling",
]
