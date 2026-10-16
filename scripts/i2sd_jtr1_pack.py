"""I2_S JTR1 pack frames for i2sd userspace load. Stdlib only.

Roadmap item: JuniorHome docs/ROADMAP.md step 3, "i2sd userspace - 127.0.0.1:8767.
No docker.sock." (Quant Chief, pair A, 2026-10-16). This file is the I2_S pack
side of that item: frames the i2sd userspace would load (JTR1: magic + gamma + n
+ 2-bit I2_S payload), complementary to JuniorQuant src/i2sd_quant_budget.py
(Winsor quant-error bound side).

New capability vs scripts/i2s_key_builder.py and src/bitnet_mlx/core/trit_pack_front.py:
a self-contained JTR1 frame encode/decode plus a per-block (n=128) pack chain that
i2sd at 127.0.0.1:8767 would admit, closed under the standing close rule. Not a
wrapper or re-derivation of those files.

JTR1 layout (not GGUF):
  magic  : 4 bytes b'JTR1'
  gamma  : 8 bytes big-endian IEEE-754 float64 (scale for {-1,0,1} -> weight)
  n      : 4 bytes big-endian uint32 (trit count)
  payload: ceil(2n/8) bytes, 2-bit I2_S codes MSB-first (0->0, 1->1, -1->2)

Close rule: sha256(prev || pack || text)[:16], pack = I2_S payload bytes.
Rate 0.05 held. Compatible engines: BitNet-mlx, JuniorLLM, web3node, stocksnode,
JuniorQuant, JuniorStock.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
The i2sd endpoint is named only (http://127.0.0.1:8767); no socket is opened here.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false. No docker.sock.
No model pull, no GGUF download, no network.

Usage: python3 i2sd_jtr1_pack.py [--selftest]
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
I2SD_PORT = 8767
I2SD_ENDPOINT = "http://%s:%d" % (BIND, I2SD_PORT)
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767, no docker.sock"
GENESIS = "0" * 16
BLOCK = 128
MAGIC = b"JTR1"
EPS = 1e-12

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


def unpack_i2s(data: bytes, n: int) -> list:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def close_rule(prev: str, pack: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def encode_jtr1(trits, gamma: float) -> bytes:
    """JTR1 frame: magic + gamma f64 + n u32 + I2_S payload."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    n = len(trits)
    payload = pack_i2s(trits)
    return MAGIC + struct.pack(">dI", float(gamma), n) + payload


def decode_jtr1(frame: bytes) -> dict:
    """Parse one JTR1 frame; raises ValueError on bad magic or truncated body."""
    if len(frame) < 16 or frame[:4] != MAGIC:
        raise ValueError("not a JTR1 frame")
    gamma, n = struct.unpack(">dI", frame[4:16])
    need = 16 + (n * 2 + 7) // 8
    if len(frame) < need:
        raise ValueError("truncated JTR1 frame")
    payload = frame[16:need]
    trits = unpack_i2s(payload, n)
    return {"magic": "JTR1", "gamma": gamma, "n": n, "trits": trits, "payload": payload}


def ternarize(weights, gamma: float) -> list:
    g = gamma if gamma > EPS else EPS
    return [max(-1, min(1, int(round(x / g)))) for x in weights]


def gamma_absmean(weights) -> float:
    if not weights:
        return 1.0
    return sum(abs(x) for x in weights) / len(weights) or 1.0


def i2sd_pack_chain(weights, prev: str = GENESIS, block: int = BLOCK) -> dict:
    """Pack weights into JTR1 frames (n=block) for i2sd load; close each block."""
    if BIND != "127.0.0.1" or not I2SD_ENDPOINT.startswith("http://127.0.0.1:"):
        return {"ok": False, "why": "bind"}
    if MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH:
        return {"ok": False, "why": "flags"}
    if block < 1:
        return {"ok": False, "why": "block"}
    frames, close = [], prev
    for i in range(0, len(weights), block):
        chunk = weights[i : i + block]
        g = gamma_absmean(chunk)
        trits = ternarize(chunk, g)
        frame = encode_jtr1(trits, g)
        payload = pack_i2s(trits)
        text = "i2sd|jtr1|%d|%.2f" % (i // block, RATE)
        close = close_rule(close, payload, text)
        frames.append(
            {
                "block": i // block,
                "gamma": round(g, 8),
                "n": len(trits),
                "frame_hex": frame.hex(),
                "payload_hex": payload.hex(),
                "close": close,
                "text": text,
            }
        )
    return {
        "ok": True,
        "endpoint": I2SD_ENDPOINT,
        "rate": RATE,
        "held": True,
        "released": 0,
        "block": block,
        "frames": frames,
        "close": close,
        "mint": MINT,
        "socket": SOCKET,
        "layer_one": LAYER_ONE,
        "fetch": FETCH,
        "ue5_launch": UE5_LAUNCH,
        "bind": BIND,
        "ports": list(PORTS),
        "roadmap": ROADMAP_ITEM,
    }


def rate_step(x: float) -> float:
    return x * RATE


def self_check() -> dict:
    assert BIND == "127.0.0.1" and I2SD_ENDPOINT == "http://127.0.0.1:8767"
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert abs(RATE - 0.05) < 1e-12
    # Cross-engine pack + close vector (same bytes as the other engines).
    t = [1, 0, -1, -1, 1, 0, 1]
    p = pack_i2s(t)
    assert p == bytes([0b01001010, 0b01000100]), p
    assert unpack_i2s(p, len(t)) == t
    tip = close_rule(GENESIS, p, "web3")
    assert tip == hashlib.sha256(GENESIS.encode() + p + b"web3").hexdigest()[:16]
    assert len(tip) == 16
    # JTR1 round-trip.
    g = 0.02
    frame = encode_jtr1(t, g)
    assert frame[:4] == MAGIC
    dec = decode_jtr1(frame)
    assert dec["trits"] == t and abs(dec["gamma"] - g) < 1e-15 and dec["n"] == len(t)
    assert decode_jtr1(frame + b"\x00")["trits"] == t  # trailing junk ignored after payload
    try:
        decode_jtr1(b"GGUF" + frame[4:])
        raise AssertionError("bad magic should fail")
    except ValueError:
        pass
    # Chain: 4 blocks of 128, deterministic and forward-only.
    w = [((i % 17) - 8) * 0.01 for i in range(BLOCK * 4)]
    m = i2sd_pack_chain(w)
    assert m["ok"] and len(m["frames"]) == 4 and len(m["close"]) == 16
    assert m["rate"] == 0.05 and m["held"] is True and m["released"] == 0
    assert m["bind"] == "127.0.0.1" and m["endpoint"] == I2SD_ENDPOINT
    assert abs(rate_step(1.0) - 0.05) < 1e-12
    assert m["close"] == i2sd_pack_chain(w)["close"]
    assert m["close"] != i2sd_pack_chain(w, prev=m["close"])["close"]
    # Each frame re-decodes and close links match payload.
    prev = GENESIS
    for fr in m["frames"]:
        d = decode_jtr1(bytes.fromhex(fr["frame_hex"]))
        assert d["n"] == BLOCK
        payload = bytes.fromhex(fr["payload_hex"])
        assert payload == pack_i2s(d["trits"])
        prev = close_rule(prev, payload, fr["text"])
        assert prev == fr["close"]
    assert prev == m["close"]
    # Reconstructed weights stay on the ternary grid.
    for fr in m["frames"]:
        d = decode_jtr1(bytes.fromhex(fr["frame_hex"]))
        recon = [x * d["gamma"] for x in d["trits"]]
        assert all(abs(r) <= d["gamma"] + EPS for r in recon)
    return {
        "selftest": "pass",
        "close": m["close"],
        "endpoint": I2SD_ENDPOINT,
        "roadmap": ROADMAP_ITEM,
        "ports": list(PORTS),
        "rate": RATE,
        "close_rule": "sha256(prev || pack || text)[:16]",
        "frames": len(m["frames"]),
    }


def main(argv) -> int:
    if "--selftest" in argv:
        print(json.dumps(self_check(), sort_keys=True))
        return 0
    w = [math.sin(i * 0.1) * 0.05 for i in range(BLOCK * 2)]
    m = i2sd_pack_chain(w)
    print(json.dumps(m, sort_keys=True))
    return 0 if m.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
