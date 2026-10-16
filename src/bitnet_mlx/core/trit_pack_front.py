"""Trit-pack front stage: pack weights to ternary BEFORE the layer pick.

Runs in front of the live layer stack so every weight handed to the layer
picker is already ternary {-1, 0, +1} with a per-tensor scale. Tiers:
  T0 - pure Python trit list
  T1 - numpy bit-plane (pos/neg uint8 planes)
  T2 - native C fallback (optional; falls back to T1 if unavailable)
No network, no model pulls, no sockets. Flags kept off.
"""
from __future__ import annotations

FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
BIND_HOST = "127.0.0.1"


def _absmean(ws):
    ws = list(ws)
    return (sum(abs(float(w)) for w in ws) / len(ws)) if ws else 0.0


def pack_t0(weights):
    """T0: Python trits via absmean (BitNet b1.58 style)."""
    ws = [float(w) for w in weights]
    s = _absmean(ws) or 1.0
    trits = [max(-1, min(1, int(round(w / s)))) for w in ws]
    return {"tier": "T0", "trits": trits, "scale": s, "n": len(trits)}


def pack_t1(weights):
    """T1: numpy bit-planes (pos, neg) packed with np.packbits."""
    import numpy as np
    w = np.asarray(weights, dtype=np.float32).ravel()
    s = float(np.mean(np.abs(w))) if w.size else 0.0
    s = s or 1.0
    t = np.clip(np.rint(w / s), -1, 1).astype(np.int8)
    return {"tier": "T1", "pos": np.packbits(t > 0), "neg": np.packbits(t < 0),
            "scale": s, "n": int(w.size)}


def pack_t2(weights):
    """T2: native C packer if a local extension exists, else T1."""
    try:
        import _trit_pack_native as nat  # optional local build, never fetched
        out = nat.pack(weights)
        out["tier"] = "T2"
        return out
    except Exception:
        out = pack_t1(weights)
        out["tier"] = "T2->T1"
        return out


def unpack(packed):
    """Return list of trits from any tier."""
    if "trits" in packed:
        return list(packed["trits"])
    import numpy as np
    n = packed["n"]
    p = np.unpackbits(packed["pos"])[:n].astype(np.int8)
    q = np.unpackbits(packed["neg"])[:n].astype(np.int8)
    return (p - q).tolist()


PACKERS = {"T0": pack_t0, "T1": pack_t1, "T2": pack_t2}


def pick_tier():
    try:
        import numpy  # noqa: F401
        return "T1"
    except Exception:
        return "T0"


def pack_before_layer(layers, tier=None, layer_picker=None):
    """Pack every layer's weights to ternary, then hand to the layer picker.

    layers: dict name -> iterable of floats. layer_picker(packed_dict, tier)
    is called with already-ternary weights; default returns packed dict.
    """
    tier = tier or pick_tier()
    packer = PACKERS[tier]
    packed = {name: packer(w) for name, w in layers.items()}
    if layer_picker is None:
        return packed
    return layer_picker(packed, tier)


if __name__ == "__main__":
    demo = {"l0": [0.9, -0.05, -1.2, 0.3], "l1": [0.0, 2.0, -2.0, 0.1]}
    ref = {k: pack_t0(v)["trits"] for k, v in demo.items()}
    for t in ("T0", "T1", "T2"):
        out = pack_before_layer(demo, tier=t)
        for k in demo:
            assert unpack(out[k]) == ref[k], (t, k)
    print("ok", ref)
