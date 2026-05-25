import mlx.core as mx
import mlx.nn as nn
from ..core.quant import compute_absmean_ternary_ste

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, bias: bool = False):
        super().__init__()
        self.weight = mx.random.normal((out_d, in_d), dtype=mx.float16) * 0.02
        self.bias = mx.zeros((out_d,), dtype=mx.float16) if bias else None
        
    def __call__(self, x: mx.array) -> mx.array:
        w_q_ste, g_tda, _ = compute_absmean_ternary_ste(self.weight)
        q_max = 7.0 
        
        scale_x = mx.max(mx.abs(x), axis=-1, keepdims=True) / q_max
        x_q = mx.clip(mx.round(x / (scale_x + 1e-5)), -q_max, q_max)
        x_q_ste = x + mx.stop_gradient(x_q - x)

        y = mx.matmul(x_q_ste, w_q_ste.T) + mx.matmul(x, (self.weight - w_q_ste).T)
        return y + self.bias if self.bias is not None else y
