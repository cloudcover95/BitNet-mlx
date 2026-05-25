import mlx.core as mx

@mx.compile
def compute_absmean_ternary_ste(w: mx.array, eps: float = 1e-5):
    """
    O(N) AbsMean Ternary Quantization with Straight-Through Estimator.
    Strictly zero-SVD routing. Bounds mapped explicitly to {-1, 0, 1}.
    """
    outlier_cutoff = 3.0 * mx.std(w.astype(mx.float32))
    outlier_mask = mx.abs(w) > outlier_cutoff
    w_core = mx.where(outlier_mask, 0.0, w)
    
    gamma = mx.mean(mx.abs(w_core.astype(mx.float32)), axis=-1, keepdims=True).astype(mx.float16)
    w_q_raw = mx.round(mx.clip(w_core / (gamma + eps), -1.0, 1.0))
    w_q_ste = w_core + mx.stop_gradient(w_q_raw - w_core)
    
    return w_q_ste.astype(mx.float16), gamma, mx.where(outlier_mask, w, 0.0)
