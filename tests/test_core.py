import mlx.core as mx
from bitnet_mlx.bitnet_layers import compute_absmean_ternary_ste

def test_ternary_boundaries():
    w = mx.random.normal((128, 128))
    w_q_ste, _, _ = compute_absmean_ternary_ste(w)
    vals = set(mx.unique(mx.round(w_q_ste)).tolist())
    assert vals.issubset({-1.0, 0.0, 1.0})
    print("[+] Ternary test passed")
