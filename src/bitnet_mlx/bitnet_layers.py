import mlx.core as mx
import mlx.nn as nn

@mx.compile
def compute_hybrid_ternary(w: mx.array, eps: float = 1e-5) -> tuple:
    """Absolute-mean ternary mapping: gamma = (1/N) * sum(|W_row|)"""
    gamma = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    w_q = mx.round(mx.clip(w / (gamma + eps), -1.0, 1.0))
    return w_q.astype(mx.int8), gamma

@mx.compile
def compute_activation_quantization(x: mx.array, eps: float = 1e-5) -> tuple:
    g_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    x_q = mx.clip(mx.round(x * (127.0 / (g_x + eps))), -127.0, 127.0)
    return x_q.astype(mx.int8), g_x

class DynamicBitLinear(nn.Module):
    """Stateless inference routing for b1.58 AMX execution."""
    def __init__(self, in_d: int, out_d: int, bias: bool = False):
        super().__init__()
        self.w_q = mx.zeros((out_d, in_d), dtype=mx.int8)
        self.gamma = mx.zeros((out_d, 1), dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((out_d,), dtype=mx.float16)
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        x_q, g_x = compute_activation_quantization(x)
        
        # Fused AMX Integer Matrix Multiplication
        y_q = mx.matmul(x_q.astype(mx.int16), self.w_q.T.astype(mx.int16))
        y = y_q.astype(mx.float16) * ((self.gamma.T * g_x) / 127.0)
        
        if self.bias is not None:
            y += self.bias
        return y
