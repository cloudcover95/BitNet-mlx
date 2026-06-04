# path: BitNet-mlx/src/quantization/mlx_quantizer.py
#!/usr/bin/env python3
"""
MLX Quantizer (Kernel Integrated)

Now uses the custom Metal ternary matmul kernel for efficient inference.
"""

import logging
from typing import Any, Dict, Optional

import mlx.core as mx

import numpy as np

from .ternary_pipeline import TernaryPipeline
from ..kernels.ternary_matmul import ternary_matmul

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class MLXQuantizer:
    def __init__(self, output_dim: int = 128):
        self.pipeline = TernaryPipeline(output_dim=output_dim)
        logging.info("MLXQuantizer initialized with custom kernel support")

    def quantize_weights(self, weights: mx.array) -> Dict[str, Any]:
        weights_np = np.array(weights)
        result = self.pipeline.run(weights_np)

        ternary_weights = mx.array(result["projection"]["ternary_weights"])

        return {
            "ternary_weights": ternary_weights,
            "original_shape": weights.shape,
            "ternary_shape": ternary_weights.shape,
            "persistence_score": result["signature"]["persistence_score"],
            "sparsity": result["projection"].get("sparsity", 0.0),
        }

    def quantize_linear_layer(self, weight: mx.array, bias: Optional[mx.array] = None) -> Dict[str, Any]:
        q_weight = self.quantize_weights(weight)
        return {
            "weight": q_weight,
            "bias": bias,
            "is_quantized": True,
        }

    def create_quantized_linear(self, in_features: int, out_features: int) -> Dict[str, Any]:
        weight = mx.random.normal((out_features, in_features))
        return self.quantize_linear_layer(weight)

    def matmul_ternary(self, a: mx.array, ternary_weight: mx.array) -> mx.array:
        """
        Uses the custom optimized Metal kernel for ternary matmul.
        """
        return ternary_matmul(a, ternary_weight, transpose_b=True)

    def forward_quantized_linear(
        self, input_tensor: mx.array, quantized_layer: Dict[str, Any]
    ) -> mx.array:
        """
        Full forward pass for a quantized linear layer using the kernel.
        """
        weight_info = quantized_layer["weight"]
        ternary_w = weight_info["ternary_weights"]
        bias = quantized_layer.get("bias")

        out = self.matmul_ternary(input_tensor, ternary_w)

        if bias is not None:
            out = out + bias

        return out
