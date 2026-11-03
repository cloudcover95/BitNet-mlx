#!/usr/bin/env python3
"""Inference slice check for BitNet-mlx: trit-pack weights, then pick the layer tier.

Additive, stdlib-only self-check (JuniorCloud LLC guards). No bench, no LUT,
no GGUF, no model pull, no network. Mirrors the layer stack:
  T0 Python trit  for n < 512*512
  T1 bit-plane numpy for n < 1024*1024
  T2 native C when the .so exists, else falls back to T1
Weights are trit-packed before the tier pick (see
src/bitnet_mlx/core/trit_pack_front.py). Close rule:
sha256(prev || pack || text)[:16], rate 0.05.
"""
import hashlib
import os
import sys

RATE = 0.05
BIND = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024


def trit(x, thr=0.5):
    return 1 if x > thr else (-1 if x < -thr else 0)


def trit_pack(values):
    """Pack trits 5 per byte (3**5 = 243 <= 255)."""
    trits = [trit(v) + 1 for v in values]
    out = bytearray()
    for i in range(0, len(trits), 5):
        chunk = trits[i:i + 5]
        b = 0
        for t in reversed(chunk):
            b = b * 3 + t
        out.append(b)
    return bytes(out)


def trit_unpack(packed, n):
    vals = []
    for b in packed:
        for _ in range(5):
            vals.append(b % 3 - 1)
            b //= 3
    return vals[:n]


def pick_tier(n, so_path=None):
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    if so_path and os.path.exists(so_path):
        return "T2"
    return "T1"  # T2 falls back to T1 when the .so is missing


def close(prev, pack, text):
    return hashlib.sha256(prev + pack + text.encode()).hexdigest()[:16]


def self_check():
    vals = [0.9, -0.8, 0.1, 0.0, -0.3, 0.7, -1.2, 0.6, 0.2, -0.55, 0.51]
    packed = trit_pack(vals)
    assert trit_unpack(packed, len(vals)) == [trit(v) for v in vals], "pack roundtrip"
    assert all(b <= 242 for b in packed), "byte range"
    assert pick_tier(100) == "T0"
    assert pick_tier(T0_MAX) == "T1"
    assert pick_tier(T1_MAX - 1) == "T1"
    assert pick_tier(T1_MAX, so_path="/nonexistent/stack.so") == "T1"
    assert pick_tier(T1_MAX, so_path=__file__) == "T2"
    c1 = close(b"\x00" * 16, packed, "bitnet")
    c2 = close(b"\x00" * 16, packed, "bitnet")
    assert c1 == c2 and len(c1) == 16, "close rule deterministic"
    assert close(b"\x01" * 16, packed, "bitnet") != c1, "close rule chains prev"
    # engines agree: python close vs explicit hashlib on concatenation
    assert c1 == hashlib.sha256(b"\x00" * 16 + packed + b"bitnet").hexdigest()[:16]
    assert RATE == 0.05
    assert BIND == "127.0.0.1"
    assert not any(FLAGS.values()), "flags must stay false"
    return c1


if __name__ == "__main__":
    try:
        tag = self_check()
    except AssertionError as e:
        print("FAIL", e)
        sys.exit(1)
    print("OK rate=%.2f bind=%s close=%s flags=all_false" % (RATE, BIND, tag))
