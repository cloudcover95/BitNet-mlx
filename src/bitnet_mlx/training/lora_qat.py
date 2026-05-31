# src/bitnet_mlx/training/lora_qat.py
import mlx.core as mx
import mlx.nn as nn
import math
from ..core.dynamic_bitlinear import DynamicBitLinear

class TernaryLoRA(nn.Module):
    """
    Quantization-Aware Training (QAT) proxy layer for Edge-Native Sovereign fine-tuning.
    Freezes the ternary bounds while backpropagating through low-rank FP16 matrices.
    """
    def __init__(self, base_layer: DynamicBitLinear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = alpha / r
        
        # Freeze topological ternary weights
        self.base_layer.freeze()
        
        in_d = self.base_layer.weight.shape[1]
        out_d = self.base_layer.weight.shape[0]
        
        # Trainable low-rank adapters
        self.lora_a = mx.random.uniform(
            low=-1 / math.sqrt(in_d), high=1 / math.sqrt(in_d), shape=(r, in_d), dtype=mx.float16
        )
        self.lora_b = mx.zeros((out_d, r), dtype=mx.float16)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base_layer(x)
        lora_out = mx.matmul(self.dropout(x), self.lora_a.T)
        lora_out = mx.matmul(lora_out, self.lora_b.T) * self.scaling
        return base_out + lora_out