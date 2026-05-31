import mlx.core as mx
from typing import Tuple
from .tda_math import TopologicalManifold

@mx.compile
def compute_tda_ternary_ste(w: mx.array, eps: float = 1e-5) -> Tuple[mx.array, mx.array, mx.array]:
    """
    O(N) Variance-Preserving Ternary Quantization via Composite TDA.
    """
    w_fp32 = w.astype(mx.float32)
    gamma_base = mx.mean(mx.abs(w_fp32), axis=-1, keepdims=True)
    w_q_raw = mx.round(mx.clip(w_fp32 / (gamma_base + eps), -1.0, 1.0))
    
    recon_base = w_q_raw * gamma_base
    var_orig = mx.var(w_fp32, axis=-1, keepdims=True)
    var_recon = mx.var(recon_base, axis=-1, keepdims=True)
    beta = mx.sqrt(var_orig / (var_recon + eps))
    
    # Composite TDA Divergence Evaluation
    tda_div = TopologicalManifold.compute_manifold_divergence(w_fp32, recon_base)
    
    # Exponential topological scaling: Forces structural mapping if divergence > 2%
    tda_scalar = mx.where(tda_div > 0.02, beta * mx.exp(tda_div * 1.5), beta)
    gamma_tda = (gamma_base * tda_scalar).astype(mx.float16)
    
    # Straight-Through Estimator (STE) Gradient Bypass
    w_q_ste = w_fp32 + mx.stop_gradient((w_q_raw * gamma_tda) - w_fp32)
    return w_q_ste.astype(mx.float16), gamma_tda, w_q_raw.astype(mx.int8)
