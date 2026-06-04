# path: BitNet-mlx/src/quantization/manifold_quantizer.py
#!/usr/bin/env python3
"""
ManifoldFoldingQuantizer (Direct Projection Style)

Further simplified toward "pure direct projection" philosophy
(inspired by Gemma 4 unified decoder approach).

Core path is now extremely linear:
Raw state → AbsMean scaling → Direct ternary projection

All advanced operations (SVD, TDA, persistence) remain available
but are opt-in to keep the default path as lightweight as possible.

Still maintained as a clean black box.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


def _to_numpy(x: Any) -> np.ndarray:
    if HAS_MLX and isinstance(x, mx.array):
        return np.array(x)
    return np.asarray(x)


def _from_numpy(arr: np.ndarray, like: Any = None) -> Any:
    if HAS_MLX and (like is None or isinstance(like, mx.array)):
        return mx.array(arr)
    return arr


def abs_mean_quantization(state: Any, epsilon: float = 1e-5) -> Any:
    arr = _to_numpy(state)
    scale = np.mean(np.abs(arr)) + epsilon
    quantized = np.round(arr / scale)
    quantized = np.clip(quantized, -1, 1)
    return _from_numpy(quantized, like=state)


def fold_manifold(
    state: Any,
    output_dim: Optional[int] = None,
    include_tda: bool = False,
) -> Dict[str, Any]:
    """
    Core direct projection path (Gemma-like simplicity).
    """
    arr = _to_numpy(state)
    ternary = abs_mean_quantization(arr)

    result = {
        "ternary_embedding": _from_numpy(ternary, like=state),
        "original_shape": arr.shape,
        "method": "direct_manifold_folding",
    }

    if output_dim is not None and arr.ndim >= 2:
        # Optional SVD reduction (still available but not default)
        try:
            U, S, Vt = np.linalg.svd(arr, full_matrices=False)
            k = min(output_dim, len(S))
            reduced = (U[:, :k] * S[:k]) @ Vt[:k, :]
            ternary_reduced = abs_mean_quantization(reduced)
            result["reduced_ternary"] = _from_numpy(ternary_reduced, like=state)
            result["svd_energy"] = float(np.sum(S[:k]) / np.sum(S))
        except Exception:
            pass

    if include_tda:
        from .manifold_quantizer import compute_ternary_tda, get_persistence_signature
        result["tda"] = compute_ternary_tda(ternary)
        result["persistence_signature"] = get_persistence_signature(ternary)

    return result


def fold_manifold_full(state: Any) -> Dict[str, Any]:
    """
    Full-featured version (includes TDA + persistence by default).
    Still uses the simplified direct projection core.
    """
    return fold_manifold(state, include_tda=True)


def compute_ternary_tda(ternary_tensor: Any, max_dim: int = 2) -> Dict[str, Any]:
    arr = _to_numpy(ternary_tensor)
    unique_vals, counts = np.unique(arr, return_counts=True)
    betti_0 = len(unique_vals)
    persistence = {
        str(int(v)): float(c) for v, c in zip(unique_vals, counts)
    }
    return {
        "betti_numbers": {"beta_0": betti_0},
        "persistence": persistence,
    }


def get_persistence_signature(ternary_tensor: Any) -> Dict[str, Any]:
    tda = compute_ternary_tda(ternary_tensor)
    arr = _to_numpy(ternary_tensor)
    return {
        "signature": tda,
        "coherence": float(np.mean(np.abs(arr))),
    }


class ManifoldFoldingQuantizer:
    def __init__(self, output_dim: Optional[int] = None):
        self.output_dim = output_dim
        logging.info("ManifoldFoldingQuantizer initialized (direct projection style)")

    def __call__(self, state: Any, include_tda: bool = True) -> Dict[str, Any]:
        return fold_manifold(state, output_dim=self.output_dim, include_tda=include_tda)


# Simple black-box entry point
_manifold_quantizer = ManifoldFoldingQuantizer()

def fold_manifold_full(state: Any) -> Dict[str, Any]:
    return _manifold_quantizer(state)
