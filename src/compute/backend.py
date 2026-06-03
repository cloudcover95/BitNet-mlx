# path: BitNet-mlx/src/compute/backend.py
#!/usr/bin/env python3
"""
Compute Backend Abstraction

Defines a common interface for compute backends (MLX, future CUDA, etc.).
This allows BitNet-mlx to support multiple hardware platforms cleanly.

Current implementation: MLX backend for Apple Silicon.
Future: CUDA backend for NVIDIA AI chips.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ComputeBackend(ABC):
    """
    Abstract base class for compute backends.
    """

    @abstractmethod
    def array(self, data: Any) -> Any:
        pass

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        pass

    @abstractmethod
    def norm(self, x: Any, axis: int = -1, keepdims: bool = False) -> Any:
        pass

    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def mean(self, x: Any, axis: int = None, keepdims: bool = False) -> Any:
        pass

    @abstractmethod
    def device_info(self) -> Dict[str, Any]:
        pass


class MLXBackend(ComputeBackend):
    """
    MLX backend implementation for Apple Silicon.
    """

    def __init__(self):
        try:
            import mlx.core as mx
            self.mx = mx
            logging.info("MLXBackend initialized")
        except ImportError:
            raise RuntimeError("mlx is required for MLXBackend")

    def array(self, data: Any) -> Any:
        return self.mx.array(data)

    def matmul(self, a: Any, b: Any) -> Any:
        return self.mx.matmul(a, b)

    def norm(self, x: Any, axis: int = -1, keepdims: bool = False) -> Any:
        return self.mx.linalg.norm(x, axis=axis, keepdims=keepdims)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return self.mx.where(condition, x, y)

    def mean(self, x: Any, axis: int = None, keepdims: bool = False) -> Any:
        return self.mx.mean(x, axis=axis, keepdims=keepdims)

    def device_info(self) -> Dict[str, Any]:
        try:
            return {
                "backend": "mlx",
                "device": str(self.mx.default_device()),
            }
        except Exception:
            return {"backend": "mlx"}
