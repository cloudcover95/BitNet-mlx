import mlx.core as mx
from bitnet_mlx.bitnet_layers import compute_absmean_ternary_ste
from bitnet_mlx.vision import TernaryMultimodalProjector
def test_ternary_boundaries():
w = mx.random.normal((128, 128))
w_q_ste, _, _ = compute_absmean_ternary_ste(w)
vals = set(mx.unique(mx.round(w_q_ste)).tolist())
assert vals.issubset({-1.0, 0.0, 1.0})
print("[+] Ternary test passed")
def test_vision_projector():
projector = TernaryMultimodalProjector(256, 512)
dummy = mx.random.normal((1, 196, 256))
out = projector(dummy)
assert out.shape == (1, 196, 512)
print("[+] Vision projector test passed")
