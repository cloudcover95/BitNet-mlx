# path: BitNet-mlx/src/compute/hardware_abstraction.py
#!/usr/bin/env python3
"""
Hardware Abstraction Layer

Unified interface for different compute backends (MLX, future CUDA,
NPU, custom silicon). Designed to support the long-term vision of a
BitNet-native OS running on diverse edge hardware.

Current focus: Apple Silicon (MLX). Future: NVIDIA, custom AI chips.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ComputeBackend(ABC):
    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        pass

    @abstractmethod
    def ternary_matmul(self, a: Any, ternary_b: Any) -> Any:
        pass

    @abstractmethod
    def quantize(self, tensor: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def device_info(self) -> Dict[str, Any]:
        pass


class MLXBackend(ComputeBackend):
    def __init__(self):
        import mlx.core as mx
        self.mx = mx
        logging.info("MLXBackend initialized")

    def matmul(self, a: Any, b: Any) -> Any:
        return self.mx.matmul(a, b)

    def ternary_matmul(self, a: Any, ternary_b: Any) -> Any:
        from ..kernels.ternary_matmul import ternary_matmul
        return ternary_matmul(a, ternary_b)

    def quantize(self, tensor: Any) -> Dict[str, Any]:
        from ..quantization.mlx_quantizer import MLXQuantizer
        quantizer = MLXQuantizer()
        return quantizer.quantize_weights(tensor)

    def device_info(self) -> Dict[str, Any]:
        return {
            "backend": "mlx",
            "device": str(self.mx.default_device()),
            "unified_memory": True,
        }


class HardwareAbstraction:
    """
    High-level hardware abstraction for BitNet-mlx.
    Allows the same code to run efficiently across current and future hardware.
    """

    def __init__(self):
        self.backends = {}
        self._detect_and_register_backends()

    def _detect_and_register_backends(self):
        try:
            from .hardware_detector import HardwareDetector
            detector = HardwareDetector()
            preferred = detector.get_preferred_backend()

            if preferred == "mlx":
                self.backends["mlx"] = MLXBackend()
            # Future: Add CUDABackend, NPUBackend, etc.

            logging.info(f"Registered backends: {list(self.backends.keys())}")
        except Exception as e:
            logging.warning(f"Backend detection failed: {e}")

    def get_backend(self, name: Optional[str] = None) -> ComputeBackend:
        if name and name in self.backends:
            return self.backends[name]
        # Default to preferred
        return list(self.backends.values())[0] if self.backends else None

    def matmul(self, a: Any, b: Any, backend: Optional[str] = None) -> Any:
        return self.get_backend(backend).matmul(a, b)

    def ternary_matmul(self, a: Any, ternary_b: Any, backend: Optional[str] = None) -> Any:
        return self.get_backend(backend).ternary_matmul(a, ternary_b)

    def quantize(self, tensor: Any, backend: Optional[str] = None) -> Dict[str, Any]:
        return self.get_backend(backend).quantize(tensor)

    def device_info(self, backend: Optional[str] = None) -> Dict[str, Any]:
        return self.get_backend(backend).device_info()
