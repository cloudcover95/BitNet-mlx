# src/bitnet_mlx/core/quantization.py
import mlx.core as mx
from typing import Tuple
from .blackbox_ops import BlackBoxArchitecture

@mx.compile
def compute_asymmetric_ternary_ste(w: mx.array, eps: float = 1e-7) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Asymmetric Binary Projection with STE & Laplacian TDA variance scaling."""
    w_fp32 = w.astype(mx.float32)
    pos_mask, neg_mask = w_fp32 > 0.0, w_fp32 < 0.0
    
    gamma_pos = mx.sum(mx.maximum(w_fp32, 0.0), axis=-1, keepdims=True) / (mx.sum(pos_mask, axis=-1, keepdims=True) + eps)
    gamma_neg = mx.sum(mx.abs(mx.minimum(w_fp32, 0.0)), axis=-1, keepdims=True) / (mx.sum(neg_mask, axis=-1, keepdims=True) + eps)
    
    w_scaled = mx.where(pos_mask, w_fp32 / (gamma_pos + eps), mx.where(neg_mask, w_fp32 / (gamma_neg + eps), 0.0))
    w_q_raw = mx.clip(mx.round(w_scaled), -1.0, 1.0)
    
    recon_base = mx.where(w_q_raw > 0.0, w_q_raw * gamma_pos, mx.where(w_q_raw < 0.0, w_q_raw * gamma_neg, 0.0))
    tda_div = BlackBoxArchitecture.laplacian_spectral_tda(w_fp32, recon_base)
    
    beta = mx.sqrt(mx.var(w_fp32, axis=-1, keepdims=True) / (mx.var(recon_base, axis=-1, keepdims=True) + eps))
    tda_scalar = mx.where(tda_div > 0.015, beta * mx.exp(tda_div * 1.5), beta).astype(mx.float16)
    
    g_p_tda = (gamma_pos * tda_scalar).astype(mx.float16)
    g_n_tda = (gamma_neg * tda_scalar).astype(mx.float16)
    
    w_q_ste = w_fp32 + mx.stop_gradient(recon_base.astype(mx.float16) - w_fp32)
    return w_q_ste, g_p_tda, g_n_tda, w_q_raw.astype(mx.int8)