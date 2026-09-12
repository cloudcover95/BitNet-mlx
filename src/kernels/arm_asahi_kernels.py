# path: src/kernels/arm_asahi_kernels.py
"""ARM + Asahi. Same AbsMean as ternary_kernels / FrameForge bitnet_quant."""

def _absmean(xs, eps=1e-8):
    data = [float(x) for x in xs]
    n = len(data) or 1
    scale = sum(abs(x) for x in data) / n + eps
    q = [1 if x / scale > 0.5 else (-1 if x / scale < -0.5 else 0) for x in data]
    return q, scale

def arm_abs_mean_quantize(weight, eps=1e-8):
    xs = list(weight) if not hasattr(weight, "tolist") else list(weight.tolist())
    return _absmean(xs, eps)

def asahi_abs_mean_quantize(weight, eps=1e-8):
    return arm_abs_mean_quantize(weight, eps)

def arm_ternary_matmul(inp, ternary_weight, scale):
    x = list(inp) if not hasattr(inp, "tolist") else list(inp.tolist())
    w = list(ternary_weight) if not hasattr(ternary_weight, "tolist") else list(ternary_weight.tolist())
    acc = 0.0
    for a, b in zip(x, w):
        if b:
            acc += a if b > 0 else -a
    return acc * float(scale)

asahi_ternary_matmul = arm_ternary_matmul
