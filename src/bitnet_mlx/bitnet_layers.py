import mlx.core as mx
import mlx.nn as nn

@mx.compile
def compute_absmean_ternary_ste(w: mx.array, eps: float = 1e-5):
    outlier_cutoff = 3.0 * mx.std(w)
    outlier_mask = mx.abs(w) > outlier_cutoff
    w_core = mx.where(outlier_mask, 0.0, w)
    gamma = mx.mean(mx.abs(w_core), axis=-1, keepdims=True)
    w_q_raw = mx.round(mx.clip(w_core / (gamma + eps), -1.0, 1.0))
    w_q_ste = w_core + mx.stop_gradient(w_q_raw - w_core)
    return w_q_ste.astype(mx.float16), gamma, mx.where(outlier_mask, w, 0.0)

class DynamicBitLinear(nn.Module):
    def __init__(self, in_d: int, out_d: int, bias: bool = False, modality: str = "text"):
        super().__init__()
        self.weight = mx.random.normal((out_d, in_d), dtype=mx.float16) * 0.02
        self.bias = mx.zeros((out_d,), dtype=mx.float16) if bias else None
        
    def __call__(self, x: mx.array) -> mx.array:
        w_q, gamma, w_out = compute_absmean_ternary_ste(self.weight)
        y = mx.matmul(x, w_q.T) * gamma.T + mx.matmul(x, w_out.T)
        return y + self.bias if self.bias is not None else y
