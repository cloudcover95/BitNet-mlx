# src/bitnet_mlx/core/dynamic_bitlinear.py
import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from .quantization import compute_asymmetric_ternary_ste
from .blackbox_ops import BlackBoxArchitecture
from ..utils.hardware import HardwareGovernor

class DynamicBitLinear(nn.Module):
    """Integrates Zero-Collision Bitwise Routing with Asymmetric TDA quantization."""
    def __init__(self, in_d: int, out_d: int, bias: bool = False, use_hadamard: bool = True) -> None:
        super().__init__()
        self.use_hadamard = use_hadamard
        self.weight = mx.random.uniform(low=-1.0/math.sqrt(in_d), high=1.0/math.sqrt(in_d), shape=(out_d, in_d), dtype=mx.float16)
        self.bias: Optional[mx.array] = mx.zeros((out_d,), dtype=mx.float16) if bias else None
        
    def __call__(self, x: mx.array) -> mx.array:
        q_max = HardwareGovernor.calculate_quantization_ceiling()
        w_q_ste, g_p, g_n, w_raw = compute_asymmetric_ternary_ste(self.weight)
        
        scale_x = mx.max(mx.abs(x), axis=-1, keepdims=True) / q_max
        x_q_ste = x + mx.stop_gradient(mx.clip(mx.round(x / (scale_x + 1e-5)), -q_max, q_max) - x)

        if self.use_hadamard:
            x_q_ste = BlackBoxArchitecture.hadamard_cascade(x_q_ste)

        y = BlackBoxArchitecture.bitwise_sparse_route(x_q_ste, w_raw, g_p, g_n)
        return y + self.bias if self.bias is not None else y