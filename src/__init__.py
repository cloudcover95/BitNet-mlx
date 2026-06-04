# path: BitNet-mlx/src/__init__.py
"""
BitNet-mlx Public API

Exposes the core efficiencies for easy reuse across the JuniorCloud LLC ecosystem
and other repos (JuniorHome, JuniorQuant, JuniorLLM, JuniorAGI, etc.).
"""

# Core Quantization (Black Box)
from .quantization.manifold_quantizer import (
    ManifoldFoldingQuantizer,
    fold_manifold,
    fold_manifold_full,
    abs_mean_quantization,
)

# Hardware Abstraction & Routing
from .compute.hardware_abstraction import HardwareAbstraction

from .compute.tri_state_router import TriStateRouter

# Kernels
try:
    from .kernels.ternary_matmul import ternary_matmul, benchmark_ternary_matmul
except ImportError:
    pass

__all__ = [
    "ManifoldFoldingQuantizer",
    "fold_manifold",
    "fold_manifold_full",
    "abs_mean_quantization",
    "HardwareAbstraction",
    "TriStateRouter",
    "ternary_matmul",
]
