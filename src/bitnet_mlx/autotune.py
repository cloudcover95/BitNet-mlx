import mlx.core as mx
import logging
from .bitnet_layers import compute_hybrid_ternary_ste

logger = logging.getLogger("BitNet.AutoTune")

class GrokAutoTuner:
@staticmethod
def calculate_optimal_sparsity(w: mx.array, max_delta: float = 0.018, min_variance: float = 0.96) -> tuple:
std_w = mx.var(w).item()
if std_w == 0:
w_q, gamma, w_outliers = compute_hybrid_ternary_ste(w)
return w, w_q, gamma, w_outliers, 0.0

    for sparsity_limit in [0.10, 0.08, 0.05, 0.02, 0.0]:
        cutoff = mx.std(w) * sparsity_limit
        sparsified_w = mx.where(mx.abs(w) < cutoff, 0.0, w)
        w_q, gamma, w_outliers = compute_hybrid_ternary_ste(sparsified_w)
        
        w_recon = (w_q * gamma) + w_outliers
        delta = mx.mean(mx.abs(w - w_recon)).item()
        fidelity = min(mx.var(w_recon).item() / (std_w + 1e-9), 1.0)
        
        if delta < max_delta and fidelity >= min_variance:
            return sparsified_w, w_q, gamma, w_outliers, sparsity_limit
            
    w_q, gamma, w_outliers = compute_hybrid_ternary_ste(w)
    return w, w_q, gamma, w_outliers, 0.0
