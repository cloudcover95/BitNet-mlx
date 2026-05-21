import mlx.core as mx
from bitnet_mlx.bitnet_layers import compute_absmean_ternary_ste, compute_ternary_kv_cache
from bitnet_mlx.vlm_backend import TernaryMultimodalProjector

def test_ternary_boundaries():
    w = mx.random.normal((128, 128))
    w_q_ste, _, _ = compute_absmean_ternary_ste(w)
    vals = set(mx.unique(mx.round(w_q_ste)).tolist())
    assert vals.issubset({-1.0, 0.0, 1.0})
    print("[+] Ternary boundary test passed")

def test_kv_cache_compression():
    kv = mx.random.normal((1, 32, 128))
    kv_q, scale = compute_ternary_kv_cache(kv, bits=4)
    assert kv_q.dtype == mx.int8
    print("[+] Phase 2 Ternary KV-Cache test passed")

def test_vision_projector():
    projector = TernaryMultimodalProjector(256, 512)
    dummy = mx.random.normal((1, 196, 256))
    out = projector(dummy)
    assert out.shape == (1, 196, 512)
    print("[+] Multimodal Vision projector test passed")
