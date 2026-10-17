"""Offline UE5 launch-gate I2_S key (watts hysteresis trits). Stdlib only.

Roadmap item: JuniorHome docs/ROADMAP.md step 5, "UE5 launched - JUNIOR_UE5=1 and
watts>=80. Default FrameForge2D. os_route.launch false until then." (Quant Chief,
pair A, slot 2026-10-17 10:05 MT). This file is the I2_S / trit key side of that
item; the quant-error bound side belongs to Winsor & Quant Error in a different
repo path.

New capability: an offline gate key for the UE5 route decision. It takes an env
text (KEY=VALUE lines) and a list of watt readings handed to it as numbers (no
hardware calls, no sensors read), turns each watt reading into a hysteresis trit
around the 80 W threshold using the 0.05 rate as the band:
  +1  watts >= 80 * (1 + 0.05) = 84.0   (clearly above)
   0  76.0 <= watts < 84.0              (inside the band, undecided)
  -1  watts <  80 * (1 - 0.05) = 76.0   (clearly below)
then packs [flag trit] + [watt trits] as a JTR1 frame and closes it into an I2_S
gate key. The gate is armed only when JUNIOR_UE5=1, the latest reading is >= 80 W
and the trailing window (3 readings) holds no -1 trit. Armed -> route "UE5" and
os_route_launch_permitted true (advisory only); otherwise route "FrameForge2D"
(default) and os_route_launch_permitted false.

This file never launches, opens, locates or touches UE5 or any UE5 binary:
UE5_LAUNCH stays False regardless of the gate. It does not wrap, test or
re-derive any other team file; it only reads text and numbers handed to it.

JTR1 layout (not GGUF):
  magic  : 4 bytes b'JTR1'
  gamma  : 8 bytes big-endian IEEE-754 float64 (here the 80.0 W threshold)
  n      : 4 bytes big-endian uint32 (trit count)
  payload: ceil(2n/8) bytes, 2-bit I2_S codes MSB-first (0->0, 1->1, -1->2)

Close rule: sha256(prev || pack || text)[:16], pack = I2_S payload bytes,
text = canonical gate JSON. Rate 0.05 held (hysteresis band).
Compatible engines: BitNet-mlx, JuniorLLM, i2sd userspace, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no hardware calls.

Usage: python3 ue5_launch_gate_i2s_key.py [--selftest] [--env PATH]
       [--watts 81,85,90] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
WATTS_MIN = 80.0
WINDOW = 3
DEFAULT_ROUTE = "FrameForge2D"
GATED_ROUTE = "UE5"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("BitNet-mlx", "JuniorLLM", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #5 UE5 launched (JUNIOR_UE5=1 and watts>=80)"
GENESIS = "0" * 16
MAGIC = b"JTR1"

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def pack_i2s(trits) -> bytes:
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
            code = (b >> s) & 3
            if code == 3:
                raise ValueError("invalid I2_S code 3")
            out.append(_DEC[code])
    return out[:n]


def encode_jtr1(trits, gamma: float) -> bytes:
    trits = list(trits)
    return MAGIC + struct.pack(">dI", float(gamma), len(trits)) + pack_i2s(trits)


def decode_jtr1(frame: bytes) -> dict:
    if frame[:4] != MAGIC:
        raise ValueError("not a JTR1 frame")
    gamma, n = struct.unpack(">dI", frame[4:16])
    payload = frame[16:]
    if len(payload) != (2 * n + 7) // 8:
        raise ValueError("JTR1 payload length mismatch")
    return {"gamma": gamma, "n": n, "trits": unpack_i2s(payload, n), "payload": payload}


def close_rule(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def parse_env(text: str) -> dict:
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip("'\"")
    return env


def flag_trit(env: dict) -> int:
    """+1 JUNIOR_UE5=1, -1 explicitly set to anything else, 0 not stated."""
    if "JUNIOR_UE5" not in env:
        return 0
    return 1 if env["JUNIOR_UE5"] == "1" else -1


def watt_trit(w: float) -> int:
    w = float(w)
    if w != w:  # NaN reads as undecided
        return 0
    if w >= WATTS_MIN * (1.0 + RATE):
        return 1
    if w < WATTS_MIN * (1.0 - RATE):
        return -1
    return 0


def gate(env_text: str, watts, prev: str = GENESIS) -> dict:
    env = parse_env(env_text)
    watts = [float(w) for w in watts]
    ft = flag_trit(env)
    wt = [watt_trit(w) for w in watts]
    tail = wt[-WINDOW:]
    latest_ok = bool(watts) and watts[-1] >= WATTS_MIN
    armed = ft == 1 and latest_ok and -1 not in tail
    trits = [ft] + wt
    frame = encode_jtr1(trits, WATTS_MIN)
    payload = frame[16:]
    doc = {
        "item": ROADMAP_ITEM,
        "flag_trit": ft,
        "watt_trits": wt,
        "watts_latest": watts[-1] if watts else None,
        "watts_min": WATTS_MIN,
        "band": [round(WATTS_MIN * (1 - RATE), 6), round(WATTS_MIN * (1 + RATE), 6)],
        "window": WINDOW,
        "armed": armed,
        "route": GATED_ROUTE if armed else DEFAULT_ROUTE,
        "os_route_launch_permitted": armed,
        "ue5_launch": UE5_LAUNCH,
        "rate": RATE,
        "bind": BIND,
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
        "ports": list(PORTS),
        "engines": list(ENGINES),
    }
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    key = close_rule(prev, payload, text)
    return {"doc": doc, "frame": frame, "text": text, "key": key, "prev": prev}


def selftest() -> int:
    assert RATE == 0.05
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert BIND == "127.0.0.1"
    # I2_S round trip
    tr = [1, 0, -1, 1, -1, 0, 0, 1, -1]
    assert unpack_i2s(pack_i2s(tr), len(tr)) == tr
    f = decode_jtr1(encode_jtr1(tr, 80.0))
    assert f["trits"] == tr and f["gamma"] == 80.0 and f["n"] == 9
    # hysteresis band
    assert watt_trit(84.0) == 1 and watt_trit(83.99) == 0
    assert watt_trit(76.0) == 0 and watt_trit(75.99) == -1
    assert watt_trit(float("nan")) == 0
    # env parsing
    assert flag_trit(parse_env("export JUNIOR_UE5='1'\n")) == 1
    assert flag_trit(parse_env("JUNIOR_UE5=0")) == -1
    assert flag_trit(parse_env("# JUNIOR_UE5=1\n")) == 0
    # armed case
    g = gate("JUNIOR_UE5=1\n", [70, 82, 85, 90])
    d = g["doc"]
    assert d["armed"] and d["route"] == "UE5" and d["os_route_launch_permitted"]
    assert d["ue5_launch"] is False and UE5_LAUNCH is False
    assert decode_jtr1(g["frame"])["trits"] == [1, -1, 0, 1, 1]
    # flag missing -> default FrameForge2D
    g2 = gate("", [90, 90, 90])
    assert not g2["doc"]["armed"] and g2["doc"]["route"] == DEFAULT_ROUTE
    # latest below 80 -> not armed
    assert not gate("JUNIOR_UE5=1", [90, 90, 79.5])["doc"]["armed"]
    # -1 in trailing window -> not armed even if latest >= 80
    assert not gate("JUNIOR_UE5=1", [90, 70, 81])["doc"]["armed"]
    # -1 outside the window does not block
    assert gate("JUNIOR_UE5=1", [50, 81, 82, 83])["doc"]["armed"]
    # no readings -> not armed
    assert not gate("JUNIOR_UE5=1", [])["doc"]["armed"]
    # close rule matches definition and chains
    expect = hashlib.sha256(GENESIS.encode() + g["frame"][16:] + g["text"].encode()).hexdigest()[:16]
    assert g["key"] == expect and len(g["key"]) == 16
    g3 = gate("JUNIOR_UE5=1\n", [70, 82, 85, 90], prev=g["key"])
    assert g3["key"] != g["key"]
    assert gate("JUNIOR_UE5=1\n", [70, 82, 85, 90])["key"] == g["key"]
    print("selftest ok key=%s rate=%.2f route=%s ue5_launch=%s" % (
        g["key"], RATE, d["route"], UE5_LAUNCH))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--env", help="path to KEY=VALUE env text (read only)")
    ap.add_argument("--watts", default="", help="comma-separated watt readings")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    env_text = ""
    if a.env:
        with open(a.env, "r", encoding="utf-8") as fh:
            env_text = fh.read()
    watts = [float(x) for x in a.watts.split(",") if x.strip()]
    g = gate(env_text, watts, a.prev)
    print(json.dumps({"key": g["key"], "prev": g["prev"], "frame_hex": g["frame"].hex(),
                      **g["doc"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
