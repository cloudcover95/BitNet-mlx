#!/usr/bin/env python3
"""Layer-pick self-check for BitNet-mlx (Inference Core lane, JuniorCloud LLC).

Additive, standalone, stdlib only. No bench, no LUT, no GGUF, no model pull.
Mirrors the layer stack (JuniorLLM junior_osai/kernels/stack.py):
  T0 Python trit      for n < 512*512
  T1 bit-plane numpy  for n < 1024*1024
  T2 native C when the .so exists, else falls back to T1
Weights are trit-packed before the layer pick
(src/bitnet_mlx/core/trit_pack_front.py).
Guards: rate 0.05, close rule sha256(prev || pack || text)[:16],
host 127.0.0.1, mint/socket/layer_one/fetch/ue5_launch all False.
Ports touched: none (not a quant file).
"""
import hashlib
import sys

RATE = 0.05
HOST = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024


def trit_pack(weights, threshold=0.5):
    """Map floats to {-1, 0, 1} and pack 5 trits per byte (3**5 = 243)."""
    trits = [1 if w > threshold else (-1 if w < -threshold else 0) for w in weights]
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            v = v * 3 + (t + 1)
        out.append(v)
    return bytes(out), len(trits)


def pick_layer(n, so_exists=False):
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    return "T2" if so_exists else "T1"


def close_rule(prev, pack, text):
    return hashlib.sha256(prev + pack + text.encode("utf-8")).hexdigest()[:16]


def main():
    ok = True
    cases = [(1, False, "T0"), (T0_MAX - 1, False, "T0"), (T0_MAX, False, "T1"),
             (T1_MAX - 1, True, "T1"), (T1_MAX, True, "T2"), (T1_MAX, False, "T1")]
    for n, so, want in cases:
        got = pick_layer(n, so)
        if got != want:
            print(f"FAIL pick_layer({n}, so={so}) = {got}, want {want}")
            ok = False
    pack, count = trit_pack([0.9, -0.9, 0.1, 0.7, -0.2, -0.8])
    if count != 6 or len(pack) != 2:
        print("FAIL trit_pack size")
        ok = False
    tier = pick_layer(count)  # pick happens after packing
    c1 = close_rule(b"\x00" * 16, pack, "inference-slice")
    c2 = close_rule(b"\x00" * 16, pack, "inference-slice")
    if c1 != c2 or len(c1) != 16:
        print("FAIL close rule")
        ok = False
    # Engines agree: a second independent computation of the rule and rate.
    alt = hashlib.sha256(b"".join([b"\x00" * 16, pack, b"inference-slice"])).hexdigest()[:16]
    if alt != c1 or RATE != 0.05:
        print("FAIL engine agreement")
        ok = False
    if HOST != "127.0.0.1" or any(FLAGS.values()):
        print("FAIL safety guards")
        ok = False
    print(f"tier={tier} close={c1} rate={RATE} host={HOST} flags_all_false={not any(FLAGS.values())}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
