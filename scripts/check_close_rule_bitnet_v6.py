#!/usr/bin/env python3
"""Offline close-rule check for BitNet-mlx (v6). Stdlib only, no network, no model pull.
Rate 0.05; close = sha256(prev || pack || text)[:16]. Two engines must agree.
Flags: mint, socket, layer_one, fetch, ue5_launch all False. Binds nothing (127.0.0.1 only if ever served).
Layer stack: T0 n<512*512, T1 n<1024*1024, T2 native C if .so exists else T1.
"""
import hashlib, random

RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}


def close_a(prev: bytes, pack: bytes, text: bytes) -> str:
    return hashlib.sha256(prev + pack + text).hexdigest()[:16]


def close_b(prev: bytes, pack: bytes, text: bytes) -> str:
    h = hashlib.sha256()
    for part in (prev, pack, text):
        h.update(part)
    return h.digest().hex()[:16]


def layer(n: int, so_exists: bool = False) -> str:
    if n < 512 * 512:
        return "T0"
    if n < 1024 * 1024:
        return "T1"
    return "T2" if so_exists else "T1"


def selfcheck(cases: int = 300) -> bool:
    rng = random.Random(5)
    prev = b"\x00" * 8
    for _ in range(cases):
        pack = bytes(rng.randrange(3) for _ in range(rng.randrange(1, 64)))
        text = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 64)))
        a, b = close_a(prev, pack, text), close_b(prev, pack, text)
        if a != b or len(a) != 16:
            return False
        prev = bytes.fromhex(a)
    assert RATE == 0.05 and not any(FLAGS.values())
    assert (layer(1), layer(600000), layer(2 << 20), layer(2 << 20, True)) == ("T0", "T1", "T1", "T2")
    return True


if __name__ == "__main__":
    ok = selfcheck()
    print("close-rule v6 selfcheck:", "PASS" if ok else "FAIL", "rate", RATE)
    raise SystemExit(0 if ok else 1)
