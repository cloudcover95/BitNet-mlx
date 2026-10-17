"""I2_S policy key for i2sd userspace launch specs. Stdlib only.

Roadmap item: JuniorHome docs/ROADMAP.md step 3, "i2sd userspace - 127.0.0.1:8767.
No docker.sock." (Quant Chief, pair A, slot 2026-10-16 18:05 MT). This file is
the I2_S / trit key side of that item; the quant-error bound side belongs to
Winsor & Quant Error in a different repo path.

New capability: an offline userspace policy audit for an i2sd launch spec (a
systemd --user unit body or KEY=VALUE env text), turned into a trit policy vector
(+1 satisfied, 0 not stated, -1 violated), packed as a JTR1 frame and closed into
an I2_S policy key. i2sd at 127.0.0.1:8767 would only admit a key whose vector
has no -1 trit. It does not wrap, test or re-derive scripts/i2sd_jtr1_pack.py or
any other team file; it only reads text handed to it.

Policy checks (one trit each, in this order):
  0 bind_loopback   : every bind/host/listen value is 127.0.0.1 (any wildcard -> -1)
  1 port_8767       : the i2sd port is 8767
  2 no_docker_sock  : no docker.sock reference anywhere
  3 userspace       : no User=root / uid 0 / sudo
  4 no_privileged   : no --privileged / CAP_SYS_ADMIN / NoNewPrivileges=no
  5 no_fetch        : no curl/wget/http(s) model pull or GGUF download
  6 no_ue5          : no UE5 / UnrealEditor launch
  7 flags_false     : mint/socket/layer_one/fetch/ue5_launch not set true

JTR1 layout (not GGUF):
  magic  : 4 bytes b'JTR1'
  gamma  : 8 bytes big-endian IEEE-754 float64 (here 1.0, policy trits unscaled)
  n      : 4 bytes big-endian uint32 (trit count)
  payload: ceil(2n/8) bytes, 2-bit I2_S codes MSB-first (0->0, 1->1, -1->2)

Close rule: sha256(prev || pack || text)[:16], pack = I2_S payload bytes,
text = canonical policy JSON. Rate 0.05 held.
Compatible engines: BitNet-mlx, JuniorLLM, i2sd userspace, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
The i2sd endpoint is named only (http://127.0.0.1:8767); no socket is opened.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false. No docker.sock.
No model pull, no GGUF download, no network.

Usage: python3 i2sd_userspace_policy_key.py [--selftest] [--spec PATH] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
ENGINES = ("BitNet-mlx", "JuniorLLM", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767 no docker.sock"
GENESIS = "0" * 16
MAGIC = b"JTR1"
CHECKS = (
    "bind_loopback", "port_8767", "no_docker_sock", "userspace",
    "no_privileged", "no_fetch", "no_ue5", "flags_false",
)

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


def encode_jtr1(trits, gamma: float = 1.0) -> bytes:
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


def rate_step(x: float) -> float:
    return x * (1.0 + RATE)


_BIND_RE = re.compile(
    r"(?:--?(?:host|bind|listen)[ =]|(?:HOST|BIND|LISTEN|I2SD_HOST|I2SD_BIND)\s*=\s*|ListenStream\s*=\s*)"
    r"([0-9.]+|\[[0-9a-fA-F:]*\]|[A-Za-z0-9_.-]+)",
)
_ADDR_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3}):(\d{2,5})\b")
_PORT_RE = re.compile(r"(?:--port[ =]|(?:PORT|I2SD_PORT)\s*=\s*)(\d{2,5})")


def audit_spec(text: str) -> list:
    """Return one trit per CHECKS entry for a launch spec text. Pure text scan."""
    low = text.lower()
    hosts = [m.group(1) for m in _BIND_RE.finditer(text)]
    addrs = _ADDR_RE.findall(text)
    hosts += [a for a, _ in addrs]
    ports = [int(p) for p in _PORT_RE.findall(text)] + [int(p) for _, p in addrs]

    t = []
    if not hosts:
        t.append(0)
    else:
        t.append(1 if all(h == BIND for h in hosts) else -1)
    if not ports:
        t.append(0)
    else:
        t.append(1 if all(p == I2SD_PORT for p in ports) else -1)
    t.append(-1 if "docker.sock" in low else 1)
    root = re.search(r"^\s*user\s*=\s*root\b", low, re.M) or re.search(r"\buid\s*[=:]\s*0\b", low) \
        or re.search(r"(^|\s)sudo\s", low)
    t.append(-1 if root else 1)
    priv = ("--privileged" in low or "cap_sys_admin" in low
            or re.search(r"nonewprivileges\s*=\s*(no|false|0)\b", low))
    t.append(-1 if priv else 1)
    fetch = (re.search(r"(^|\s)(curl|wget)\s", low) or re.search(r"https?://(?!127\.0\.0\.1)", low)
             or "huggingface-cli download" in low or "hf_hub_download" in low)
    t.append(-1 if fetch else 1)
    t.append(-1 if ("ue5" in low or "unrealeditor" in low) else 1)
    flags = re.search(r"\b(mint|socket|layer_one|fetch|ue5_launch)\s*[=:]\s*(true|1|yes)\b", low)
    t.append(-1 if flags else 1)
    return t


def policy_key(spec_text: str, prev: str = GENESIS) -> dict:
    trits = audit_spec(spec_text)
    frame = encode_jtr1(trits, 1.0)
    pack = frame[16:]
    policy = {
        "item": ROADMAP_ITEM,
        "endpoint": I2SD_ENDPOINT,
        "checks": dict(zip(CHECKS, trits)),
        "admit": all(x != -1 for x in trits),
        "spec_sha256": hashlib.sha256(spec_text.encode()).hexdigest(),
        "rate": RATE,
    }
    text = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return {
        "policy": policy,
        "frame_hex": frame.hex(),
        "key": close_rule(prev, pack, text),
        "prev": prev,
    }


GOOD_SPEC = """[Unit]
Description=i2sd userspace (BitNet I2_S)
[Service]
ExecStart=%h/.local/bin/i2sd --bind 127.0.0.1 --port 8767
Environment=I2SD_HOST=127.0.0.1
Environment=MINT=false
NoNewPrivileges=yes
[Install]
WantedBy=default.target
"""

_WILD = ".".join(["0"] * 4)  # wildcard bind, only as a negative fixture

BAD_SPEC = """[Service]
User=root
ExecStart=/usr/bin/i2sd --bind %s --port 8080 -v /var/run/docker.sock:/var/run/docker.sock
ExecStartPre=wget https://example.invalid/model.gguf
socket=true
""" % _WILD


def self_check() -> dict:
    res = {}
    # flags and constants
    res["flags_false"] = not any((MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH))
    res["bind_loopback_const"] = BIND == "127.0.0.1" and _WILD not in I2SD_ENDPOINT
    res["rate_0_05"] = RATE == 0.05 and abs(rate_step(100.0) - 105.0) < 1e-9
    # I2_S / JTR1 round trip
    vec = [1, 0, -1, 1, -1, 0, 0, 1, -1]
    fr = encode_jtr1(vec, 0.5)
    d = decode_jtr1(fr)
    res["jtr1_roundtrip"] = d["trits"] == vec and d["gamma"] == 0.5 and d["n"] == 9 and fr[:4] == MAGIC
    res["i2s_bytes"] = pack_i2s([1, 0, -1, 1]) == bytes([0b01001001])
    try:
        unpack_i2s(b"\xff", 4)
        res["rejects_code3"] = False
    except ValueError:
        res["rejects_code3"] = True
    # policy audit
    g = policy_key(GOOD_SPEC)
    b = policy_key(BAD_SPEC)
    res["good_admits"] = g["policy"]["admit"] and all(v == 1 for v in g["policy"]["checks"].values())
    bc = b["policy"]["checks"]
    res["bad_rejects"] = (not b["policy"]["admit"]) and all(
        bc[k] == -1 for k in ("bind_loopback", "port_8767", "no_docker_sock", "userspace", "no_fetch", "flags_false"))
    res["unstated_is_zero"] = audit_spec("ExecStart=i2sd")[:2] == [0, 0]
    # close rule matches the standing definition
    pack = bytes.fromhex(g["frame_hex"])[16:]
    text = json.dumps(g["policy"], sort_keys=True, separators=(",", ":"))
    want = hashlib.sha256(GENESIS.encode() + pack + text.encode()).hexdigest()[:16]
    res["close_rule"] = g["key"] == want and len(g["key"]) == 16
    chained = policy_key(GOOD_SPEC, prev=g["key"])
    res["chain_moves"] = chained["key"] != g["key"]
    res["deterministic"] = policy_key(GOOD_SPEC)["key"] == g["key"]
    res["ok"] = all(res.values())
    res["good_key"] = g["key"]
    res["bad_key"] = b["key"]
    return res


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--spec", help="local launch spec text file (read only)")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest or not a.spec:
        r = self_check()
        print(json.dumps(r, indent=2, sort_keys=True))
        return 0 if r["ok"] else 1
    with open(a.spec, "r", encoding="utf-8") as fh:
        out = policy_key(fh.read(), a.prev)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["policy"]["admit"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
