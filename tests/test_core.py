import mlx.core as mx
from bitnet_mlx.bitnet_layers import compute_hybrid_ternary_ste

def test_ste_bounds():
w = mx.random.normal((128, 128))
w_q_ste, gamma, w_outliers = compute_hybrid_ternary_ste(w)
mx.eval(w_q_ste, gamma, w_outliers)

# STE forward pass operates explicitly on -1, 0, 1 bounds for the core logic
unique_vals = mx.unique(mx.round(w_q_ste)).tolist()
for val in unique_vals:
    assert val in [-1, 0, 1], f"Topological breach: {val} falls outside absolute STE ternary boundary."
