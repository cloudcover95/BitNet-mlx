# path: src/kernels/cuda_kernels.py
"""CUDA ternary kernels. Torch when present, else stdlib AbsMean."""

def _stdlib_quantize(weight, eps=1e-8):
    xs = list(weight) if not hasattr(weight, "tolist") else list(weight.tolist())
    n = len(xs) or 1
    scale = sum(abs(float(x)) for x in xs) / n + eps
    out = []
    for x in xs:
        v = float(x) / scale
        out.append(1 if v > 0.5 else (-1 if v < -0.5 else 0))
    return out, scale

def cuda_abs_mean_quantize(weight, eps=1e-8):
    try:
        import torch
        if torch.cuda.is_available():
            w = weight if torch.is_tensor(weight) else torch.tensor(weight, device="cuda", dtype=torch.float32)
            w = w.to("cuda")
            scale = torch.mean(torch.abs(w)) + eps
            scaled = w / scale
            q = torch.where(scaled > 0.5, 1, torch.where(scaled < -0.5, -1, 0))
            return q.to(torch.int8), scale
    except Exception:
        pass
    return _stdlib_quantize(weight, eps)

def cuda_ternary_matmul(inp, ternary_weight, scale):
    try:
        import torch
        if torch.cuda.is_available():
            x = inp if torch.is_tensor(inp) else torch.tensor(inp, device="cuda", dtype=torch.float32)
            w = ternary_weight if torch.is_tensor(ternary_weight) else torch.tensor(ternary_weight, device="cuda", dtype=torch.float32)
            s = scale if torch.is_tensor(scale) else torch.tensor(scale, device="cuda", dtype=torch.float32)
            return torch.matmul(x, (w.float() * s).T)
    except Exception:
        pass
    x = list(inp) if not hasattr(inp, "tolist") else list(inp.tolist())
    w = list(ternary_weight) if not hasattr(ternary_weight, "tolist") else list(ternary_weight.tolist())
    acc = 0.0
    for a, b in zip(x, w):
        if b:
            acc += a if b > 0 else -a
    return acc * float(scale)
