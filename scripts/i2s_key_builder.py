"""I2_S key builder + trit pack close. Stdlib only. Rate 0.05 held.
Close rule: sha256(prev || pack || text)[:16].
Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def pack_i2s(trits) -> bytes:
    """Balanced trits (-1/0/1) as 2-bit I2_S codes, MSB first, zero-padded."""
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def unpack_i2s(data: bytes, n: int) -> list[int]:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def close_rule(prev: str, pack: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def build_key(trits, prev: str = "0" * 16, text: str = "i2s|key|0.05") -> dict:
    """Build one I2_S trit-pack key and close it under the standing rule."""
    if BIND != "127.0.0.1" or BIND == "0.0.0.0":
        return {"ok": False, "why": "bind"}
    if MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH:
        return {"ok": False, "why": "flags"}
    pk = pack_i2s(trits)
    tip = close_rule(prev, pk, text)
    return {
        "ok": True,
        "pack": pk.hex(),
        "n": len(trits),
        "prev": prev,
        "tip": tip,
        "rate": RATE,
        "held": True,
        "released": 0,
        "mint": MINT,
        "socket": SOCKET,
        "layer_one": LAYER_ONE,
        "fetch": FETCH,
        "ue5_launch": UE5_LAUNCH,
        "bind": BIND,
        "ports": list(PORTS),
        "text": text,
    }


def rate_step(x: float) -> float:
    return x * RATE


def self_check() -> dict:
    t = [1, 0, -1, -1, 1, 0, 1, -1]
    p = pack_i2s(t)
    assert unpack_i2s(p, len(t)) == t
    a = build_key(t, "0" * 16, "bitnet-mlx|i2s|0.05")
    b = build_key(t, a["tip"], "bitnet-mlx|i2s|0.05|next")
    expected = close_rule("0" * 16, p, "bitnet-mlx|i2s|0.05")
    ok = (
        a["ok"]
        and b["ok"]
        and a["tip"] == expected
        and len(a["tip"]) == 16
        and b["prev"] == a["tip"]
        and a["rate"] == 0.05
        and abs(rate_step(1.0) - 0.05) < 1e-12
        and a["held"] is True
        and a["released"] == 0
        and a["mint"] is False
        and a["socket"] is False
        and a["layer_one"] is False
        and a["fetch"] is False
        and a["ue5_launch"] is False
        and a["bind"] == "127.0.0.1"
        and tuple(a["ports"]) == PORTS
        and close_rule(a["tip"], p, "bitnet-mlx|i2s|0.05|next") == b["tip"]
    )
    return {
        "ok": ok,
        "rate": RATE,
        "close_rule": "sha256(prev || pack || text)[:16]",
        "bind": BIND,
        "ports": list(PORTS),
        "tip": b["tip"] if ok else None,
        "mint": MINT,
        "socket": SOCKET,
        "layer_one": LAYER_ONE,
        "fetch": FETCH,
        "ue5_launch": UE5_LAUNCH,
    }


if __name__ == "__main__":
    row = self_check()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
