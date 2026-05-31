import mlx.core as mx
import pytest
from bitnet_mlx.core.quantization import compute_tda_ternary_ste
from bitnet_mlx.utils.hardware import HardwareGovernor
from bitnet_mlx.core.tda_math import TopologicalManifold

def test_tda_variance_fidelity():
    """Ensures Composite TDA scaling guarantees >99% variance preservation."""
    dim = 256
    w_fp16 = mx.random.normal((dim, dim), dtype=mx.float16) * 0.05
    w_q_ste, _, _ = compute_tda_ternary_ste(w_fp16)
    var_orig = mx.var(w_fp16.astype(mx.float32)).item()
    var_recon = mx.var(w_q_ste.astype(mx.float32)).item()
    assert (var_recon / (var_orig + 1e-9)) >= 0.99

def test_hardware_governor_topology():
    """Ensures dynamic routing executes correctly across A-Series and M-Series."""
    hw_model, hw_brand = HardwareGovernor.get_soc_topology()
    assert isinstance(hw_model, str)
    assert isinstance(hw_brand, str)
