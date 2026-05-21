import mlx.core as mx
import mlx.nn as nn
import psutil
class EdgeSubstrate:
@staticmethod
def optimal_activation_bits() -> int:
try:
return 4 if psutil.cpu_percent(interval=0.1) > 85.0 else 8
except Exception:
return 8
@mx.compile
def compute_absmean_ternary_ste(w: mx.array, eps: float = 1e-5):
"""AbsMean Ternary Quantization with STE (O(N))."""
outlier_cutoff = 3.0 * mx.std(w)
outlier_mask = mx.abs(w) > outlier_cutoff
w_core = mx.where(outlier_mask, 0.0, w)
gamma = mx.mean(mx.abs(w_core), axis=-1, keepdims=True)
w_q_raw = mx.round(mx.clip(w_core / (gamma + eps), -1.0, 1.0))
w_q_ste = w_core + mx.stop_gradient(w_q_raw - w_core)
w_outliers = mx.where(outlier_mask, w, 0.0)
return w_q_ste.astype(mx.float16), gamma, w_outliers.astype(mx.float16)
class DynamicBitLinear(nn.Module):
def init(self, in_d: int, out_d: int, bias: bool = False):
super().init()
self.weight = mx.random.normal((out_d, in_d), dtype=mx.float16) * 0.02
self.bias = mx.zeros((out_d,), dtype=mx.float16) if bias else None
def call(self, x: mx.array) -> mx.array:
w_q_ste, gamma, w_outliers = compute_absmean_ternary_ste(self.weight)
active_bits = EdgeSubstrate.optimal_activation_bits()
q_max = (2 ** (active_bits - 1)) - 1.0
scale_x = mx.max(mx.abs(x), axis=-1, keepdims=True) / q_max
x_q = mx.clip(mx.round(x / (scale_x + 1e-5)), -q_max, q_max)
x_q_ste = x + mx.stop_gradient(x_q - x)
y_core = mx.matmul(x_q_ste, w_q_ste.T) * (gamma.T * scale_x)
y = y_core + mx.matmul(x, w_outliers.T)
if self.bias is not None:
y += self.bias
return y
