# path: BitNet-mlx/src/compute/hardware_detector.py
#!/usr/bin/env python3
"""
Hardware Detector

Detects available compute hardware and provides a unified interface
for future multi-platform support (Apple Silicon MLX, NVIDIA CUDA, etc.).

This is foundational for making BitNet-mlx future-proof across
M-series, M5/Ultra, and next-gen NVIDIA AI edge chips.
"""

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class HardwareDetector:
    """
    Detects and abstracts available compute backends.
    """

    def __init__(self):
        self.backends: Dict[str, bool] = {
            "mlx": HAS_MLX,
            "cuda": self._has_cuda(),
            "cpu": True,  # Always available as fallback
        }
        logging.info(f"Hardware backends detected: {self.backends}")

    def _has_cuda(self) -> bool:
        if not HAS_TORCH:
            return False
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    def get_preferred_backend(self) -> str:
        if self.backends.get("mlx"):
            return "mlx"  # Prefer MLX on Apple Silicon
        elif self.backends.get("cuda"):
            return "cuda"
        return "cpu"

    def get_info(self) -> Dict[str, Any]:
        info = {
            "preferred": self.get_preferred_backend(),
            "available": {k: v for k, v in self.backends.items() if v},
        }

        if HAS_MLX:
            try:
                info["mlx_device"] = str(mx.default_device())
            except Exception:
                pass

        if HAS_TORCH and self.backends.get("cuda"):
            try:
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass

        return info

    def is_apple_silicon(self) -> bool:
        return self.backends.get("mlx", False)

    def is_nvidia_cuda(self) -> bool:
        return self.backends.get("cuda", False)
