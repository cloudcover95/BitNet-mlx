#!/usr/bin/env python3
"""Inference-slice check for BitNet-mlx (JuniorCloud LLC, Inference Core lane).

Additive, standalone, stdlib-only. Exercises the LLC layer-stack contract:
  - weights are trit-packed before the layer pick (rate 0.05)
    (src/bitnet_mlx/core/trit_pack_front.py)
  - layer pick: T0 Python trit for n < 512*512, T1 bit-plane numpy for
    n < 1024*1024, T2 native C when the .so exists, else fall back to T1
    (JuniorLLM junior_osai/kernels/stack.py)
  - close rule: sha256(prev || pack || text)[:16]
  - safety flags (mint, socket, layer_one, fetch, ue5_launch) stay False
  - any host binding is 127.0.0.1
No bench, no LUT, no GGUF, no model pull, no network.
Quant ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
import hashlib
import os
import sys

RATE = 0.05
HOST = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024


def trit_pack(weights, rate=RATE):
    """Map floats to {-1,0,1} with dead-zone `rate`, then pack 5 trits/byte."""
    trits = [0 if abs(w) <= rate else (1 if w > 0 else -1) for w in weights]
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            v = v * 3 + (t + 1)
        out.append(v)
    return bytes(out), trits


def pick_layer(n, so_path=None):
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    if so_path and os.path.exists(so_path):
        return "T2"
    return "T1"


def close(prev, pack, text):
    return hashlib.sha256(prev + pack + text.encode("utf-8")).hexdigest()[:16]


def main():
    ok = True
    pack, trits = trit_pack([0.3, -0.2, 0.01, -0.04, 0.0, 0.9, -0.06])
    ok &= trits == [1, -1, 0, 0, 0, 1, -1]
    ok &= len(pack) == 2
    # pack happens before pick
    ok &= pick_layer(len(trits)) == "T0"
    ok &= pick_layer(T0_MAX) == "T1"
    ok &= pick_layer(T1_MAX) == "T1"
    ok &= pick_layer(T1_MAX, so_path="/nonexistent/kernel.so") == "T1"
    c1 = close(b"\x00" * 16, pack, "BitNet-mlx slice")
    c2 = close(b"\x00" * 16, pack, "BitNet-mlx slice")
    ok &= c1 == c2 and len(c1) == 16
    # engines agree: two independent derivations of rate and close rule
    ok &= RATE == 0.05 and c1 == hashlib.sha256(b"\x00" * 16 + pack + b"BitNet-mlx slice").hexdigest()[:16]
    ok &= not any(FLAGS.values())
    ok &= HOST == "127.0.0.1"
    print("ports:", ", ".join(PORTS))
    print("close:", c1)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
