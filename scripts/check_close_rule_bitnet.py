#!/usr/bin/env python3
"""Inference-slice self-check (BitNet-mlx, 2026-10-24).

Additive, standalone check of the LLC trit-pack close rule and layer-stack
tier pick. No network, no model pull, no GGUF, no bench, no LUT.

- Trit-pack rate: 0.05
- Close rule: sha256(prev || pack || text)[:16]
- Layer stack: T0 Python trit for n < 512*512, T1 bit-plane numpy for
  n < 1024*1024, T2 native C when the .so exists (else T1).
- Safety flags all False; any host binding is 127.0.0.1.
"""
import hashlib
import sys

RATE = 0.05
HOST = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
T0_MAX = 512 * 512
T1_MAX = 1024 * 1024


def close_rule(prev: bytes, pack: bytes, text: bytes) -> str:
    return hashlib.sha256(prev + pack + text).hexdigest()[:16]


def trit_pack(values):
    """Quantize floats to trits {-1, 0, 1} with a threshold, before the tier pick."""
    return [0 if abs(v) < RATE else (1 if v > 0 else -1) for v in values]


def pick_tier(n: int, native_so_exists: bool = False) -> str:
    if n < T0_MAX:
        return "T0"
    if n < T1_MAX:
        return "T1"
    return "T2" if native_so_exists else "T1"


def main() -> int:
    ok = True
    ok &= RATE == 0.05
    ok &= HOST == "127.0.0.1"
    ok &= not any(FLAGS.values())
    a = close_rule(b"prev", b"pack", b"text")
    b = hashlib.sha256(b"prevpacktext").hexdigest()[:16]
    ok &= a == b and len(a) == 16
    ok &= close_rule(b"", b"", b"") == hashlib.sha256(b"").hexdigest()[:16]
    ok &= trit_pack([0.2, -0.2, 0.01, -0.049, 0.05]) == [1, -1, 0, 0, 1]
    ok &= pick_tier(10) == "T0"
    ok &= pick_tier(T0_MAX) == "T1"
    ok &= pick_tier(T1_MAX) == "T1"
    ok &= pick_tier(T1_MAX, native_so_exists=True) == "T2"
    print("check_close_rule_bitnet:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
