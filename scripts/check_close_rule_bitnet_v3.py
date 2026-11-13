#!/usr/bin/env python3
"""Local close-rule / guard self-check for BitNet-mlx (additive, stdlib only).

Checks: rate 0.05, close rule sha256(prev || pack || text)[:16], guard flags
false, loopback bind only, T0/T1/T2 layer pick thresholds. No model pull, no GGUF.
"""
import hashlib

RATE = 0.05
HOST = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}


def close(prev: bytes, pack: bytes, text: bytes) -> str:
    return hashlib.sha256(prev + pack + text).hexdigest()[:16]


def pick_layer(n: int, so_exists: bool = False) -> str:
    if n < 512 * 512:
        return "T0"
    if n < 1024 * 1024:
        return "T1"
    return "T2" if so_exists else "T1"


def self_check() -> bool:
    a = close(b"", b"pack", b"text")
    ok = len(a) == 16 and a == close(b"", b"pack", b"text") and a != close(b"x", b"pack", b"text")
    ok &= RATE == 0.05 and HOST == "127.0.0.1" and not any(FLAGS.values())
    ok &= pick_layer(10) == "T0" and pick_layer(512 * 512) == "T1"
    ok &= pick_layer(1024 * 1024) == "T1" and pick_layer(1024 * 1024, True) == "T2"
    return bool(ok)


if __name__ == "__main__":
    r = self_check()
    print("self_check", "PASS" if r else "FAIL")
    raise SystemExit(0 if r else 1)
