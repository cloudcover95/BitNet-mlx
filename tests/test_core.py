import mlx.core as mx, mlx.nn as nn
from bitnet_mlx.core.quantization import compute_asymmetric_ternary_ste
from bitnet_mlx.training.engine import SovereignTrainer
from bitnet_mlx.core.dynamic_bitlinear import DynamicBitLinear

def test_tda_variance_fidelity():
    w = mx.random.normal((256, 256), dtype=mx.float16) * 0.05
    w_q_ste, _, _, _ = compute_asymmetric_ternary_ste(w)
    fidelity = mx.var(w_q_ste.astype(mx.float32)).item() / (mx.var(w.astype(mx.float32)).item() + 1e-9)
    assert fidelity >= 0.99

def test_training_wrapper():
    model = nn.Sequential(DynamicBitLinear(128, 64))
    
    def mock_dataset_generator(batches=1, bsz=1, seq=128, dim=64):
        for _ in range(batches):
            yield mx.random.normal((bsz, seq), dtype=mx.float16), mx.random.normal((bsz, dim), dtype=mx.float16)

    trained = SovereignTrainer.execute_cycle(model, dataset=mock_dataset_generator(), mode="qlora", epochs=1)
    assert trained is not None
