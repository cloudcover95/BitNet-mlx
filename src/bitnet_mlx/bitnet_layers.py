import mlx.core as mx
import mlx.nn as nn
from .power_matrix import ThermalSubstrate

@mx.compile
def compute_hybrid_ternary_ste(w: mx.array, eps: float = 1e-5) -> tuple:
outlier_cutoff = 3.0 * mx.std(w)
outlier_mask = mx.abs(w) > outlier_cutoff
w_core = mx.where(outlier_mask, 0.0, w)
gamma = mx.mean(mx.abs(w_core), axis=-1, keepdims=True)
w_q_raw = mx.round(mx.clip(w_core / (gamma + eps), -1.0, 1.0))
w_q_ste = w_core + mx.stop_gradient(w_q_raw - w_core)
w_outliers = mx.where(outlier_mask, w, 0.0)
return w_q_ste.astype(mx.float16), gamma, w_outliers.astype(mx.float16)

class DynamicBitLinear(nn.Module):
def init(self, in_d: int, out_d: int, bias: bool = False, activation_bits: int = 8):
super().init()
self.bits = activation_bits
self.weight = mx.random.normal((out_d, in_d), dtype=mx.float16) * 0.02
if bias:
self.bias = mx.zeros((out_d,), dtype=mx.float16)
else:
self.bias = None

def __call__(self, x: mx.array) -> mx.array:
    w_q_ste, gamma, w_outliers = compute_hybrid_ternary_ste(self.weight)
    
    # Thermal-Kinematic Dynamic Precision
    current_state = ThermalSubstrate.query_thermal_state()
    active_bits = 4 if current_state == "a4" else self.bits
    
    q_max = (2 ** (active_bits - 1)) - 1.0
    g_x = mx.max(mx.abs(x), axis=-1, keepdims=True)
    x_q = mx.clip(mx.round(x * (q_max / (g_x + 1e-5))), -q_max, q_max)
    x_q_ste = x + mx.stop_gradient(x_q - x)

    y_core = mx.matmul(x_q_ste, w_q_ste.T) * ((gamma.T * g_x) / q_max)
    y_outliers = mx.matmul(x, w_outliers.T)
    y = y_core + y_outliers
    
    if self.bias is not None:
        y += self.bias
    return y
