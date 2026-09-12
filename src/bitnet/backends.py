# path: src/bitnet/backends.py
from enum import Enum
from typing import Any

class Backend(Enum):
    MLX = "mlx"
    CUDA = "cuda"
    ARM = "arm"
    ASAHI = "asahi"
    CPU = "cpu"

class BackendRouter:
    def __init__(self):
        self._current = Backend.CPU
        self._matmul_impl = None
        self._quant_impl = None
        self.set_backend(self._detect())

    def _detect(self):
        import platform
        mach = platform.machine().lower()
        sysname = platform.system().lower()
        try:
            import mlx.core  # noqa: F401
            return Backend.MLX
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return Backend.CUDA
        except Exception:
            pass
        if mach in ("arm64", "aarch64") and sysname == "linux":
            return Backend.ASAHI
        if mach in ("arm64", "aarch64", "armv8", "armv7l"):
            return Backend.ARM
        return Backend.CPU

    def set_backend(self, backend):
        self._current = backend
        if backend == Backend.MLX:
            try:
                from .kernels.ternary_kernels import ternary_matmul as impl
                from .kernels.ternary_kernels import abs_mean_quantize as qimpl
                self._matmul_impl, self._quant_impl = impl, qimpl
                return
            except Exception:
                backend = Backend.CPU
        if backend == Backend.CUDA:
            from ..kernels.cuda_kernels import cuda_ternary_matmul as impl
            from ..kernels.cuda_kernels import cuda_abs_mean_quantize as qimpl
            self._matmul_impl, self._quant_impl = impl, qimpl
            return
        from ..kernels.arm_asahi_kernels import arm_ternary_matmul as impl
        from ..kernels.arm_asahi_kernels import arm_abs_mean_quantize as qimpl
        self._matmul_impl, self._quant_impl = impl, qimpl

    def ternary_matmul(self, inp, ternary_weight, scale):
        return self._matmul_impl(inp, ternary_weight, scale)

    def abs_mean_quantize(self, weight, eps=1e-8):
        return self._quant_impl(weight, eps)

    def suggest_routing(self, estimated_agent_mb=450):
        return {"backend": self._current.name}

router = BackendRouter()
