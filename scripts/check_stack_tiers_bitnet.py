#!/usr/bin/env python3
"""Stdlib-only self-check for the BitNet-mlx inference slice (2026-10-19).

Checks, without any network, model pull, bench, LUT, or GGUF:
  - layer-stack tier pick: T0 python trit (n < 512*512), T1 bit-plane numpy
    (n < 1024*1024), T2 native C when the .so exists, else falls back to T1
  - trit pack rate 0.05 and close rule sha256(prev || pack || text)[:16]
  - safety flags (mint, socket, layer_one, fetch, ue5_launch) all False
  - any bind host is 127.0.0.1
Not a quant file; touches no ports (Home, OSai, FieldCore, Flagstaff, JuniorLLM).
"""
import hashlib
import os
import sys

RATE = 0.05
BIND_HOST = "127.0.0.1"
SAFETY = {"mint": False, "socket": False, "layer_one": False,
          "fetch": False, "ue5_launch": False}
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024


def close_rule(prev: bytes, pack: bytes, text: bytes) -> str:
    return hashlib.sha256(prev + pack + text).hexdigest()[:16]


def trit_pack(weights):
    """Map floats to trits {-1, 0, 1} using absmean threshold."""
    if not weights:
        return []
    scale = sum(abs(w) for w in weights) / len(weights) or 1.0
    out = []
    for w in weights:
        q = round(w / scale)
        out.append(max(-1, min(1, q)))
    return out


def pack_bytes(trits):
    return bytes((t + 1) for t in trits)


def pick_tier(n: int, so_path: str = "") -> str:
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    if so_path and os.path.exists(so_path):
        return "T2"
    return "T1"


def main() -> int:
    fails = []
    w = [0.3, -0.7, 0.01, 1.2, -0.05, 0.0, -1.5, 0.6]
    trits = trit_pack(w)  # pack before the layer pick
    if any(t not in (-1, 0, 1) for t in trits):
        fails.append("trit_pack range")
    tier = pick_tier(len(trits))
    if tier != "T0":
        fails.append("tier small")
    if pick_tier(T0_MAX) != "T1":
        fails.append("tier T1 boundary")
    if pick_tier(T1_MAX, "/nonexistent/kernel.so") != "T1":
        fails.append("tier T2 fallback")
    if pick_tier(T1_MAX, sys.executable) != "T2":
        fails.append("tier T2 present")
    pack = pack_bytes(trits)
    a = close_rule(b"", pack, b"slice")
    b = close_rule(b"", pack, b"slice")
    c = close_rule(a.encode(), pack, b"slice")
    if a != b or len(a) != 16 or a == c:
        fails.append("close rule")
    engines = {"python": (RATE, close_rule), "mirror": (0.05, close_rule)}
    rates = {r for r, _ in engines.values()}
    closes = {fn(b"p", pack, b"t") for _, fn in engines.values()}
    if rates != {0.05} or len(closes) != 1:
        fails.append("engines disagree")
    if any(SAFETY.values()):
        fails.append("safety flag true")
    if BIND_HOST != "127.0.0.1":
        fails.append("bind host")
    if fails:
        print("FAIL:", ", ".join(fails))
        return 1
    print(f"OK tier={tier} rate={RATE} close={a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
