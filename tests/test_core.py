import mlx.core as mx
from bitnet_mlx.bitnet_layers import compute_hybrid_ternary_ste, dynamic_kv_cache_quantize
from bitnet_mlx.swarm import BinaryRPC

def test_ste_bounds():
w = mx.random.normal((128, 128))
w_q_ste, gamma, w_outliers = compute_hybrid_ternary_ste(w)
mx.eval(w_q_ste, gamma, w_outliers)
unique_vals = mx.unique(mx.round(w_q_ste)).tolist()
for val in unique_vals:
assert val in [-1, 0, 1], f"Topological breach: {val} outside absolute STE boundary."

def test_binary_rpc_serialization():
t_original = mx.random.normal((64, 64))
payload = BinaryRPC.pack_tensor(t_original)
t_unpacked = BinaryRPC.unpack_tensor(payload)
mx.eval(t_unpacked)
assert mx.array_equal(t_original, t_unpacked).item(), "Binary serialization mapping corrupted."

def test_kv_cache_compression():
kv = mx.random.normal((1, 32, 128))
kv_q, scale = dynamic_kv_cache_quantize(kv, bits=4)
mx.eval(kv_q, scale)
assert mx.max(kv_q).item() <= 7.0 and mx.min(kv_q).item() >= -7.0, "KV-Cache boundary escape detected."
