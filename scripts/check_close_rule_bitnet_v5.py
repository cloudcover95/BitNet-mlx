#!/usr/bin/env python3
"""Offline close-rule self-check for BitNet-mlx (v5). Stdlib only, no network, no model pull.

Close rule: sha256(prev || pack || text)[:16] (hex). Trit rate: 0.05.
Guards: mint, socket, layer_one, fetch, ue5_launch all false. Binds nothing (127.0.0.1 only if ever served).
Layer stack: T0 n<512*512, T1 n<1024*1024, T2 native C if .so exists else T1.
"""
import hashlib
import random
import sys

RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
HOST = "127.0.0.1"


def close_hash(prev: bytes, pack: bytes, text: bytes) -> str:
    return hashlib.sha256(prev + pack + text).hexdigest()[:16]


def pick_layer(n: int, so_exists: bool = False) -> str:
    if n < 512 * 512:
        return "T0"
    if n < 1024 * 1024:
        return "T1"
    return "T2" if so_exists else "T1"


def trit_pack(vals, rate=RATE):
    """Deterministic ternary quantisation: |v| <= rate -> 0, else sign."""
    return bytes((0 if abs(v) <= rate else (1 if v > 0 else 2)) for v in vals)


def selftest() -> int:
    rng = random.Random(5)
    prev = b"\x00" * 8
    chain = []
    for _ in range(200):
        pack = trit_pack([rng.uniform(-1, 1) for _ in range(32)])
        text = bytes(rng.getrandbits(8) for _ in range(16))
        h = close_hash(prev, pack, text)
        assert len(h) == 16
        assert h == hashlib.sha256(prev + pack + text).hexdigest()[:16]
        assert h != close_hash(prev, pack, text + b"x")
        chain.append(h)
        prev = bytes.fromhex(h)
    assert len(set(chain)) == len(chain)
    assert pick_layer(10) == "T0" and pick_layer(512 * 512) == "T1"
    assert pick_layer(1024 * 1024) == "T1" and pick_layer(1024 * 1024, True) == "T2"
    assert not any(FLAGS.values()) and HOST == "127.0.0.1" and RATE == 0.05
    print("PASS check_close_rule_bitnet_v5 rate=%s links=%d" % (RATE, len(chain)))
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
