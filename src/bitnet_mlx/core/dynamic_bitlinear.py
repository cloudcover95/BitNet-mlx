import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from .quantization import compute_tda_ternary_ste
from ..utils.hardware import HardwareGovernor

class DynamicBitLinear(nn.Module):
    """Drop-in MLX replacement for nn.Linear executing ternary logic natively."""
    def __init__(self, in_d: int, out_d: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = mx.random.normal((out_d, in_d), dtype=mx.float16) * 0.02
        self.bias: Optional[mx.array] = mx.zeros((out_d,), dtype=mx.float16) if bias else None
        
    def __call__(self, x: mx.array) -> mx.array:
        q_max = HardwareGovernor.calculate_quantization_ceiling()
        w_q_ste, _, _ = compute_tda_ternary_ste(self.weight)
        
        # Adaptive Activation Quantization using Omni-Hardware bounds
        scale_x = mx.max(mx.abs(x), axis=-1, keepdims=True) / q_max
        x_q = mx.clip(mx.round(x / (scale_x + 1e-5)), -q_max, q_max)
        x_q_ste = x + mx.stop_gradient(x_q - x)

        y = mx.matmul(x_q_ste, w_q_ste.T) + mx.matmul(x, (self.weight - w_q_ste).T)
        return y + self.bias if self.bias is not None else y
