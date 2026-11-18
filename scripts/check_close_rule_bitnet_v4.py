#!/usr/bin/env python3
"""Self-contained close-rule check for BitNet-mlx (v4). Additive, local only.
Rate 0.05; close = sha256(prev || pack || text)[:16]. Binds 127.0.0.1 only; no network."""
import hashlib, sys

RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
HOST = "127.0.0.1"
PORTS = ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]

def tier(n):
    if n < 512 * 512:
        return "T0"
    if n < 1024 * 1024:
        return "T1"
    return "T2"

def close(prev: bytes, pack: bytes, text: bytes) -> str:
    return hashlib.sha256(prev + pack + text).hexdigest()[:16]

def main():
    a = close(b"p", b"k", b"t")
    b = hashlib.sha256(b"pkt").hexdigest()[:16]
    assert a == b and len(a) == 16, "close rule mismatch"
    assert RATE == 0.05
    assert not any(FLAGS.values()), "flags must be false"
    assert HOST == "127.0.0.1"
    assert [tier(n) for n in (1, 512*512, 1024*1024)] == ["T0", "T1", "T2"]
    print("ok rate=%s close=%s ports=%s" % (RATE, a, ",".join(PORTS)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
