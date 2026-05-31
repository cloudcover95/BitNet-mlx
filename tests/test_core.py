# tests/test_core.py
import mlx.core as mx
import pytest
from bitnet_mlx.core.quantization import compute_tda_ternary_ste
from bitnet_mlx.utils.hardware import HardwareGovernor
from bitnet_mlx.core.tda_math import TopologicalManifold

def test_spectral_power_iteration():
    """Validates pure-MLX spectral gap estimator resolves properly."""
    matrix = mx.random.normal((128, 128))
    gram = mx.matmul(matrix.T, matrix)
    spectral_norm = TopologicalManifold.power_iteration_spectral_norm(gram)
    assert spectral_norm.item() > 0.0

def test_tda_variance_fidelity():
    """Ensures Spectral TDA scaling guarantees >99% variance preservation on LLM matrix sizes."""
    dim = 1024
    w_fp16 = mx.random.normal((dim, dim), dtype=mx.float16) * 0.05
    w_q_ste, g_tda, _ = compute_tda_ternary_ste(w_fp16)
    
    var_orig = mx.var(w_fp16.astype(mx.float32)).item()
    var_recon = mx.var(w_q_ste.astype(mx.float32)).item()
    
    fidelity = var_recon / (var_orig + 1e-9)
    assert fidelity >= 0.99, f"Variance fidelity collapse: {fidelity*100:.2f}%"

def test_hardware_governor():
    """Ensure dynamic routing doesn't fault."""
    soc = HardwareGovernor.get_soc_topology()
    assert isinstance(soc, str)
    q_max = HardwareGovernor.calculate_quantization_ceiling()
    assert q_max in [3.0, 7.0, 15.0]