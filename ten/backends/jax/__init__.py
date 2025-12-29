"""
JAX Backend Module
==================

TPU-optimized implementations using JAX/XLA.
"""

try:
    from ten.backends.jax.ten_jax import (
        TENConfigJax,
        TENJax,
        HTENJax,
        create_ten_state,
        ten_forward,
        hten_forward,
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    TENConfigJax = None
    TENJax = None
    HTENJax = None
    create_ten_state = None
    ten_forward = None
    hten_forward = None

__all__ = [
    "JAX_AVAILABLE",
    "TENConfigJax",
    "TENJax",
    "HTENJax",
    "create_ten_state",
    "ten_forward",
    "hten_forward",
]
