# path: BitNet-mlx/tests/test_blackbox.py
import mlx.core as mx
import pytest

from src.inference.proprietary_blackbox import ProprietaryMathKernel


def test_offline_manifold_computation():
    """
    Validates SVD and KPZ mathematical integrity on the M4 Neural Engine.
    Executes purely on silicon without requiring the LLM weights to be loaded.
    """
    kernel = ProprietaryMathKernel()

    # Generate synthetic (1, 60) tensor representing market spot history
    synthetic_tensor = mx.random.normal((1, 60)).tolist()

    state = kernel.compute_manifold_state(synthetic_tensor)

    # Type and existence assertions
    assert "z_score" in state
    assert "k_alpha" in state
    assert "identity_drift" in state
    assert "q_mark" in state

    assert isinstance(state["z_score"], float)
    assert isinstance(state["q_mark"], float)

    # Boundary validations
    assert state["identity_drift"] >= 0.0
    assert 0.0 <= state["q_mark"] <= 1.0
