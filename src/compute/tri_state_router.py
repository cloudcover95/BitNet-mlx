# path: BitNet-mlx/src/compute/tri_state_router.py
#!/usr/bin/env python3
"""
Tri-State Router (Hardware Agnostic)

Implements the three routing modes from the agnostic architecture:
- User black box (sovereign edge inference)
- Swarm / Agentic (Second Brain + multi-agent)
- Industry fallback (dense when needed)

Works with the HardwareAbstraction layer across MLX, CPU, and future backends.
"""

import logging
from typing import Any, Dict, Optional

try:
    from .hardware_abstraction import HardwareAbstraction
    HAS_ABSTRACTION = True
except ImportError:
    HAS_ABSTRACTION = False
    HardwareAbstraction = None

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


def abs_mean_quantization(weights: Any, epsilon: float = 1e-5) -> Any:
    """
    Core 1.58-bit quantization. Works across MLX, NumPy, and CuPy.
    """
    if HAS_ABSTRACTION:
        backend = HardwareAbstraction().get_backend()
        ts = backend  # simplified
    else:
        import numpy as np
        ts = np

    # Handle both MLX and NumPy/CuPy style
    try:
        scale = ts.mean(ts.abs(weights)) + epsilon
        quantized = ts.round(weights / scale)
        return ts.clip(quantized, -1, 1)
    except AttributeError:
        # Fallback for MLX
        import mlx.core as mx
        scale = mx.mean(mx.abs(weights)) + epsilon
        quantized = mx.round(weights / scale)
        return mx.clip(quantized, -1, 1)


class TriStateRouter:
    def __init__(self, tda_threshold: float = 0.85):
        self.tda_threshold = tda_threshold
        if HAS_ABSTRACTION:
            self.abstraction = HardwareAbstraction()
        else:
            self.abstraction = None
        logging.info("TriStateRouter initialized (hardware agnostic)")

    def route_user_black_box(self, state_tensor: Any) -> Any:
        ternary_state = abs_mean_quantization(state_tensor)
        if hasattr(ternary_state, "sum"):
            return ternary_state.sum(axis=-1)
        return sum(ternary_state)

    def route_swarm_black_box(self, state_tensor: Any, agent_context: Any) -> Any:
        ternary_state = abs_mean_quantization(state_tensor)
        ternary_context = abs_mean_quantization(agent_context)

        # Simple topological intersection (can be expanded with real TDA)
        if hasattr(ternary_state, "__mul__"):
            swarm_manifold = ternary_state * ternary_context
            return swarm_manifold.mean(axis=0)
        return [a * b for a, b in zip(ternary_state, ternary_context)]

    def route_industry_fallback(self, state_tensor: Any) -> Any:
        if self.abstraction:
            backend = self.abstraction.get_backend()
            if hasattr(backend, "matmul"):
                # Use backend matmul if available
                weights = backend.array(state_tensor.shape)  # placeholder
                return backend.matmul(state_tensor, weights.T)

        # Pure NumPy fallback
        import numpy as np
        weights = np.random.standard_normal(state_tensor.shape)
        return np.dot(state_tensor, weights.T)

    def evaluate_and_route(
        self,
        raw_input: Any,
        agent_context: Optional[Any] = None,
        mode: str = "auto",
    ) -> Any:
        if mode == "user":
            return self.route_user_black_box(raw_input)
        elif mode == "swarm" and agent_context is not None:
            return self.route_swarm_black_box(raw_input, agent_context)
        elif mode == "industry":
            return self.route_industry_fallback(raw_input)

        # Auto mode: decide based on ternary coherence
        ternary_baseline = abs_mean_quantization(raw_input)
        if hasattr(ternary_baseline, "mean"):
            coherence = float(ternary_baseline.mean())
        else:
            coherence = sum(abs(x) for x in ternary_baseline) / len(ternary_baseline)

        if coherence >= self.tda_threshold:
            return self.route_user_black_box(raw_input)
        else:
            return self.route_industry_fallback(raw_input)
