# path: src/kernels/metal_kernels.py

import mlx.core as mx


def ternary_matmul_metal(input: mx.array, ternary_weight: mx.array, scale: mx.array) -> mx.array:
    """
    Target for custom MLX Metal ternary matmul kernel.
    """
    w = ternary_weight.astype(input.dtype) * scale
    return mx.matmul(input, w.T)
