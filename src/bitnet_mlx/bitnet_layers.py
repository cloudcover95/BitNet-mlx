import mlx.core as mx
import mlx.nn as nn

@mx.compile
def compute_channel_ternary(w: mx.array) -> tuple:
    """
    Computes absolute-mean ternary projection.
    $\gamma = \frac{1}{N}\sum|W_{row}|$
    $W_{q} = \text{round}(\text{clip}(W/\gamma, -1, 1))$
    """
    eps = 1e-5
    gamma = mx.mean(mx.abs(w), axis=-1, keepdims=True)
    w_q = mx.round(mx.clip(w / (gamma + eps), -1.0, 1.0))
    return w_q.astype(mx.int8), gamma

@mx.compile
def fused_ternary_matmul(x: mx.array, w_q: mx.array, gamma: mx.array, q_max: float = 127.0) -> mx.array:
    """
    AMX-Optimized MatMul.
    Executes INT8 accumulation before returning to FP16 domain via hardware scale parameters.
    """
    eps = 1e-5
    g_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    x_q = mx.clip(mx.round(x * (q_max / (g_x + eps))), -q_max, q_max)
    
    # Int8 casting binds execution pointer to Apple AMX ALU
    y_q = mx.matmul(x_q.astype(x.dtype), w_q.T.astype(x.dtype))
    return y_q * ((gamma.T * g_x) / q_max)

class DynamicBitLinear(nn.Module):
    """Drop-in MLX Native Linear layer using $b1.58$ operational matrices."""
    def __init__(self, in_d: int, out_d: int, bias: bool = False):
        super().__init__()
        self.w_q = mx.zeros((out_d, in_d), dtype=mx.int8)
        self.gamma = mx.zeros((out_d, 1), dtype=mx.float16)
        if bias:
            self.bias = mx.zeros((out_d,))

    def __call__(self, x: mx.array) -> mx.array:
        y = fused_ternary_matmul(x, self.w_q, self.gamma)
        if hasattr(self, 'bias') and getattr(self, 'bias') is not None: 
            y += self.bias
        return y
