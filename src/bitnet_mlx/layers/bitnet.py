import mlx.core as mx
import mlx.nn as nn
import psutil
from ..core.quant import compute_absmean_ternary_ste

class EdgeSubstrate:
    @staticmethod
    def optimal_activation_routing() -> int:
        """Thermal-kinematic routing: Shifts precision based on UMA load / 48V thermal constraints."""
        return 4 if psutil.cpu_percent(interval=0.1) > 85.0 else 8

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, bias: bool = False):
        super().__init__()
        self.weight = mx.random.normal((out_d, in_d), dtype=mx.float16) * 0.02
        self.bias = mx.zeros((out_d,), dtype=mx.float16) if bias else None
        
    def __call__(self, x: mx.array) -> mx.array:
        w_q, g, w_o = compute_absmean_ternary_ste(self.weight)
        
        bits = EdgeSubstrate.optimal_activation_routing()
        q_max = (2 ** (bits - 1)) - 1.0
        
        scale_x = mx.max(mx.abs(x), axis=-1, keepdims=True) / q_max
        x_q = mx.clip(mx.round(x / (scale_x + 1e-5)), -q_max, q_max)
        x_q_ste = x + mx.stop_gradient(x_q - x)

        y = mx.matmul(x_q_ste, w_q.T) * (g.T * scale_x) + mx.matmul(x, w_o.T)
        return y + self.bias if self.bias is not None else y
