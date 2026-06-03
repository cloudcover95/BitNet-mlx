# path: BitNet-mlx/src/inference/quantization_calibration.py
import mlx.core as mx
from typing import Optional, Tuple


class MLXQuantCalibrator:
    """
    Production-grade MLX quantization calibration.
    Implements AbsMean scaling + optional per-channel calibration
    for 1.58-bit ternary (BitNet-style) quantization.

    Fully vectorized. No Python scalar loops.
    """

    def __init__(self, bits: int = 2, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.scale: Optional[mx.array] = None
        self.zero_point: Optional[mx.array] = None

    def calibrate_weights(
        self,
        weight: mx.array,
        per_channel: bool = True
    ) -> Tuple[mx.array, mx.array]:
        """
        Calibrate scale for weight tensor using AbsMean method.

        Args:
            weight: Weight tensor to calibrate
            per_channel: If True, compute per-output-channel scales

        Returns:
            (quantized_weight, scale)
        """
        if per_channel and weight.ndim >= 2:
            # Per-channel (last dim is output channels for linear)
            abs_mean = mx.mean(mx.abs(weight), axis=-1, keepdims=True)
            scale = mx.maximum(abs_mean, mx.array(1e-8))
        else:
            abs_mean = mx.mean(mx.abs(weight))
            scale = mx.maximum(abs_mean, mx.array(1e-8))

        # Ternary quantization: clip to [-1, 1] after scaling
        quantized = mx.clip(mx.round(weight / scale), -1.0, 1.0)

        self.scale = scale
        return quantized, scale

    def calibrate_activations(
        self,
        activation: mx.array,
        running_mean: Optional[mx.array] = None,
        momentum: float = 0.9
    ) -> mx.array:
        """
        Running calibration for activations (useful for dynamic quantization).
        """
        current_mean = mx.mean(mx.abs(activation))

        if running_mean is None:
            running_mean = current_mean
        else:
            running_mean = momentum * running_mean + (1 - momentum) * current_mean

        scale = mx.maximum(running_mean, mx.array(1e-8))
        self.scale = scale
        return scale

    def quantize(self, tensor: mx.array) -> mx.array:
        """Apply previously computed scale to quantize a tensor."""
        if self.scale is None:
            raise RuntimeError("Calibration must be run before quantization")

        return mx.clip(mx.round(tensor / self.scale), -1.0, 1.0)

    def dequantize(self, q_tensor: mx.array) -> mx.array:
        """Dequantize back to original precision."""
        if self.scale is None:
            return q_tensor
        return q_tensor * self.scale
