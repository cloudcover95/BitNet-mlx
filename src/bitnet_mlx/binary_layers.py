import mlx.core as mx
import mlx.nn as nn

@mx.compile
def compute_binary_projection(w: mx.array) -> tuple:
    """
    Extreme absolute binary quantization: $\{-1, 1\}$
    Forces zero-centered weights to boundaries.
    """
    eps = 1e-5
    alpha = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    w_bin = mx.sign(w + eps) 
    return w_bin.astype(mx.int8), alpha
