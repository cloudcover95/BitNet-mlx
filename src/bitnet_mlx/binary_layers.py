import mlx.core as mx
import mlx.nn as nn

@mx.compile
def compute_binary_projection(w: mx.array) -> tuple:
    """Extreme absolute binary quantization: $\{-1, 1\}$"""
    eps = 1e-5
    alpha = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    w_bin = mx.sign(w + eps) # Forces to +1 or -1
    return w_bin.astype(mx.int8), alpha
