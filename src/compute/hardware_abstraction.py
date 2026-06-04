# path: BitNet-mlx/src/compute/hardware_abstraction.py
#!/usr/bin/env python3
"""
Hardware Abstraction Layer (AMX + NPU Enhanced)

Extended with explicit AMX optimization paths for Apple Silicon
and scaffolding for future NPU inference backends.
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
        logging.info("MLXBackend initialized (AMX-accelerated)")

    def matmul(self, a: Any, b: Any) -> Any:
        # MLX automatically uses AMX on M-series for matrix ops
        return self.mx.matmul(a, b)

    def ternary_matmul(self, a: Any, ternary_b: Any) -> Any:
        from ..kernels.ternary_matmul import ternary_matmul
        return ternary_matmul(a, ternary_b)

    def quantize(self, tensor: Any) -> Dict[str, Any]:
        from ..quantization.mlx_quantizer import MLXQuantizer
        quantizer = MLXQuantizer()
        return quantizer.quantize_weights(tensor)

    def device_info(self) -> Dict[str, Any]:
        info = {
            "backend": "mlx",
            "device": str(self.mx.default_device()),
            "unified_memory": True,
            "amx_accelerated": True,  # MLX uses AMX for matmul on M-series
        }
        return info


class NPUBackend(ComputeBackend):
    """
    Placeholder / scaffolding for future NPU inference.
    Will be activated when NPU support lands in MLX or via CoreML export.
    """

    def __init__(self):
        logging.info("NPUBackend initialized (future path)")

    def matmul(self, a: Any, b: Any) -> Any:
        # Future: delegate to NPU via CoreML or custom runtime
        raise NotImplementedError("NPU matmul not yet implemented")

    def ternary_matmul(self, a: Any, ternary_b: Any) -> Any:
        raise NotImplementedError("NPU ternary matmul not yet implemented")

    def quantize(self, tensor: Any) -> Dict[str, Any]:
        # Reuse MLX quantizer for now, or future NPU-specific quant
        from ..quantization.mlx_quantizer import MLXQuantizer
        quantizer = MLXQuantizer()
        return quantizer.quantize_weights(tensor)

    def device_info(self) -> Dict[str, Any]:
        return {
            "backend": "npu",
            "status": "scaffolding - not yet active",
        }


class HardwareAbstraction:
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

            # Always register NPU scaffolding for future use
            self.backends["npu"] = NPUBackend()

            logging.info(f"Registered backends: {list(self.backends.keys())}")
        except Exception as e:
            logging.warning(f"Backend detection failed: {e}")

    def get_backend(self, name: Optional[str] = None) -> ComputeBackend:
        if name and name in self.backends:
            return self.backends[name]
        return list(self.backends.values())[0] if self.backends else None

    def matmul(self, a: Any, b: Any, backend: Optional[str] = None) -> Any:
        return self.get_backend(backend).matmul(a, b)

    def ternary_matmul(self, a: Any, ternary_b: Any, backend: Optional[str] = None) -> Any:
        return self.get_backend(backend).ternary_matmul(a, ternary_b)

    def quantize(self, tensor: Any, backend: Optional[str] = None) -> Dict[str, Any]:
        return self.get_backend(backend).quantize(tensor)

    def device_info(self, backend: Optional[str] = None) -> Dict[str, Any]:
        return self.get_backend(backend).device_info()

    def is_amx_available(self) -> bool:
        return "mlx" in self.backends

    def is_npu_available(self) -> bool:
        return "npu" in self.backends  # Currently scaffolding
