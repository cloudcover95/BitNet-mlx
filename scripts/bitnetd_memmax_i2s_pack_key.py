"""bitnetd MemoryMax <-> I2_S pack-size key (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" -- hardened
bitnetd.service bound to 127.0.0.1:8765 with a MemoryMax cap (JuniorHome
docs/ROADMAP.md JuniorOS overlay). Quant Chief pair A, slot 2026-10-18 10:05 MT.
Trit Pack & I2_S side: the I2_S key that ties the unit's MemoryMax to the size
of the I2_S weight pack bitnetd will hold. Winsor & Quant Error covers the
quant-error / memcap stream side of the same item in a different file.

New capability (uncovered gap): JuniorLLM scripts/junioros_overlay_jtr1_key.py
audits the unit's hardening flags and PATH/os-release, and web3node's unit
receipt records the unit; neither checks that MemoryMax is actually large
enough for the I2_S pack (and not so loose it caps nothing). This file:
  * parses MemoryMax / MemoryHigh / MemorySwapMax from a bitnetd.service text
    (K/M/G/T suffixes base 1024, bare bytes, "N%" with --host-ram, "infinity"),
  * sizes the I2_S pack: 2 bits per ternary weight in 128-trit blocks
    (32 bytes per block) plus one f32 scale per tensor, or takes the byte size
    of an already-local pack file via os.stat only (never opened, never fetched),
  * required = ceil(pack * (1 + HEADROOM)) + RUNTIME_OVERHEAD,
  * emits one trit per contract check, packs them to JTR1 and an I2_S block,
    and closes the key with the close rule.
It never starts bitnetd, opens a socket, runs a subprocess, or touches the
network. It does not wrap, test, or re-derive any other team file.

Trits (+1 pass, 0 undecided, -1 fail):
  memmax.present     finite MemoryMax parsed (0 missing, -1 unparseable/infinity)
  memmax.covers_pack MemoryMax >= required bytes
  memmax.tight       MemoryMax <= SLACK_CAP * required (cap still means something)
  memhigh.order      MemoryHigh absent or <= MemoryMax
  swap.off           MemorySwapMax=0 (0 when absent)
  bind.loopback      ExecStart/Environment name 127.0.0.1 and port 8765
  bind.no_wildcard   no wildcard address anywhere in the unit
  pack.known         pack size > 0 and whole I2_S blocks

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected); gamma = mean |trit|
I2_S block: code = trit + 1, four per byte MSB first, padded with code 1 to 128.
Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16), pack = JTR1 bytes, text = canonical JSON of the contract.
Rate 0.05: green needs zero -1 trits and at most 5% undecided trits, else hold.
Compatible engines: bitnetd 127.0.0.1:8765, BitNet-mlx, JuniorLLM, i2sd userspace,
JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage:
  python3 scripts/bitnetd_memmax_i2s_pack_key.py --selftest
  python3 scripts/bitnetd_memmax_i2s_pack_key.py --unit bitnetd.service
      (--params N [--tensors T] | --pack-file PATH) [--host-ram SIZE] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
BITNETD_PORT = 8765
WILDCARD = ".".join(["0"] * 4)  # deny-check only; never bound
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("bitnetd 127.0.0.1:8765", "BitNet-mlx", "JuniorLLM", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: hardened bitnetd.service "
    "bind 127.0.0.1:8765 + MemoryMax (I2_S pack-size tie)"
)
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
I2S_BLOCK_BYTES = I2S_BLOCK // 4  # 2 bits per trit
SCALE_BYTES = 4  # one f32 scale per tensor
HEADROOM = 0.25
RUNTIME_OVERHEAD = 256 * 1024 * 1024
SLACK_CAP = 8.0
_UNITS = {"": 1, "B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([BKMGT]?)\s*$", re.I)


# ---------------------------------------------------------------- parsing
def parse_unit(text: str) -> dict:
    """Return {key: [values...]} for the [Service] section (last value wins)."""
    out: dict = {}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" in line and section == "Service":
            k, v = line.split("=", 1)
            out.setdefault(k.strip(), []).append(v.strip())
    return out


def parse_size(value, host_ram=None):
    """Bytes for a systemd size, float('inf') for infinity, None if unparseable."""
    if value is None:
        return None
    v = value.strip()
    if v.lower() == "infinity":
        return float("inf")
    if v.endswith("%"):
        if host_ram is None:
            return None
        try:
            pct = float(v[:-1])
        except ValueError:
            return None
        return int(host_ram * pct / 100.0) if 0 <= pct <= 100 else None
    m = _SIZE.match(v)
    if not m:
        return None
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()])


def i2s_pack_bytes(n_params: int, n_tensors: int = 1) -> int:
    blocks = math.ceil(n_params / I2S_BLOCK) if n_params > 0 else 0
    return blocks * I2S_BLOCK_BYTES + max(n_tensors, 0) * SCALE_BYTES


def pack_file_bytes(path: str) -> int:
    """Size of an already-local pack via os.stat only; 0 if absent."""
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def required_bytes(pack: int) -> int:
    return int(math.ceil(pack * (1.0 + HEADROOM))) + RUNTIME_OVERHEAD


# ---------------------------------------------------------------- checks
def _last(unit: dict, key: str):
    vals = unit.get(key)
    return vals[-1] if vals else None


def contract_trits(unit_text: str, pack: int, host_ram=None, blocks_exact=True):
    unit = parse_unit(unit_text)
    req = required_bytes(pack) if pack > 0 else None
    raw_max = _last(unit, "MemoryMax")
    mmax = parse_size(raw_max, host_ram)
    finite = isinstance(mmax, int)
    if raw_max is None:
        present = 0
    else:
        present = 1 if finite else -1
    covers = 0 if not (finite and req) else (1 if mmax >= req else -1)
    tight = 0 if not (finite and req) else (1 if mmax <= SLACK_CAP * req else -1)
    raw_high = _last(unit, "MemoryHigh")
    if raw_high is None:
        order = 1
    else:
        mh = parse_size(raw_high, host_ram)
        order = 0 if (mh is None or not finite) else (1 if mh <= mmax else -1)
    raw_swap = _last(unit, "MemorySwapMax")
    swap = 0 if raw_swap is None else (1 if parse_size(raw_swap, host_ram) == 0 else -1)
    execs = " ".join(unit.get("ExecStart", []) + unit.get("Environment", []))
    loop = 1 if (BIND in execs and str(BITNETD_PORT) in execs) else -1
    nowild = -1 if WILDCARD in unit_text else 1
    known = 1 if (pack > 0 and blocks_exact) else (-1 if pack > 0 else 0)
    trits = [
        ("memmax.present", present),
        ("memmax.covers_pack", covers),
        ("memmax.tight", tight),
        ("memhigh.order", order),
        ("swap.off", swap),
        ("bind.loopback", loop),
        ("bind.no_wildcard", nowild),
        ("pack.known", known),
    ]
    facts = {
        "memmax_raw": raw_max,
        "memmax_bytes": mmax if finite else None,
        "pack_bytes": pack,
        "required_bytes": req,
        "ratio": round(mmax / req, 4) if (finite and req) else None,
    }
    return trits, facts


# ---------------------------------------------------------------- packing
def pack_jtr1(trits) -> bytes:
    n = len(trits)
    gamma = (sum(abs(t) for t in trits) / n) if n else 0.0
    out = bytearray(MAGIC + struct.pack("<fI", gamma, n))
    for i in range(0, n, 4):
        b = 0
        chunk = trits[i:i + 4]
        for j in range(4):
            code = _JTR[chunk[j]] if j < len(chunk) else 0
            b |= code << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def unpack_jtr1(blob: bytes):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    trits = []
    for k in range(n):
        code = (blob[12 + k // 4] >> (6 - 2 * (k % 4))) & 3
        if code == 3:
            raise ValueError("code 0b11 rejected")
        trits.append(_JTR_DEC[code])
    return gamma, trits


def pack_i2s(trits) -> bytes:
    n = len(trits)
    padded = n + ((-n) % I2S_BLOCK) if n else I2S_BLOCK
    codes = [t + 1 for t in trits] + [1] * (padded - n)
    out = bytearray()
    for i in range(0, padded, 4):
        b = 0
        for j in range(4):
            b |= codes[i + j] << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def unpack_i2s(data: bytes, n: int):
    codes = []
    for b in data:
        for j in range(4):
            codes.append((b >> (6 - 2 * j)) & 3)
    return [c - 1 for c in codes[:n]]


def close_rule(prev: str, pack: bytes, text: str) -> str:
    if len(prev) != 16 or any(c not in "0123456789abcdef" for c in prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits) -> str:
    n = len(trits)
    if n == 0 or any(t < 0 for t in trits):
        return "hold"
    return "green" if sum(1 for t in trits if t == 0) / n <= RATE else "hold"


def build_key(unit_text: str, pack: int, host_ram=None, prev: str = GENESIS, blocks_exact=True):
    named, facts = contract_trits(unit_text, pack, host_ram, blocks_exact)
    trits = [t for _, t in named]
    jtr = pack_jtr1(trits)
    contract = {
        "item": ROADMAP_ITEM,
        "checks": [[k, t] for k, t in named],
        "facts": facts,
        "rate": RATE,
        "headroom": HEADROOM,
        "overhead": RUNTIME_OVERHEAD,
        "slack_cap": SLACK_CAP,
        "bind": f"{BIND}:{BITNETD_PORT}",
        "ports": list(PORTS),
        "engines": list(ENGINES),
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return {
        "key": close_rule(prev, jtr, text),
        "prev": prev,
        "verdict": verdict(trits),
        "checks": named,
        "facts": facts,
        "jtr1_hex": jtr.hex(),
        "i2s_hex": pack_i2s(trits).hex(),
    }


# ---------------------------------------------------------------- selftest
HARDENED = """[Unit]
Description=bitnetd (JuniorOS overlay)

[Service]
ExecStart=/usr/local/bin/bitnetd --host 127.0.0.1 --port 8765
MemoryMax=2G
MemoryHigh=1536M
MemorySwapMax=0
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
"""


def _unit(**kv) -> str:
    base = {"ExecStart": "/usr/local/bin/bitnetd --host 127.0.0.1 --port 8765",
            "MemoryMax": "2G", "MemoryHigh": "1536M", "MemorySwapMax": "0"}
    base.update(kv)
    body = "\n".join(f"{k}={v}" for k, v in base.items() if v is not None)
    return f"[Service]\n{body}\n"


def selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and bool(cond)

    check(RATE == 0.05, "rate 0.05")
    check(not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH), "flags false")
    check(BIND == "127.0.0.1", "bind loopback")
    check(parse_size("2G") == 2 << 30 and parse_size("512M") == 512 << 20, "size suffixes")
    check(parse_size("infinity") == float("inf") and parse_size("lots") is None, "infinity/unparseable")
    check(parse_size("50%", 8 << 30) == 4 << 30 and parse_size("50%") is None, "percent needs host ram")
    # 2B-param ternary model: ceil(2e9/128) blocks * 32 B + 210 f32 scales
    pack = i2s_pack_bytes(2_000_000_000, 210)
    check(pack == math.ceil(2_000_000_000 / 128) * 32 + 210 * 4, "i2s pack bytes")
    check(required_bytes(pack) == math.ceil(pack * 1.25) + RUNTIME_OVERHEAD, "required bytes")
    g = build_key(HARDENED, pack)
    check(g["verdict"] == "green", f"hardened unit green ({g['facts']['ratio']})")
    check(len(g["key"]) == 16 and g["key"] == build_key(HARDENED, pack)["key"], "key deterministic")
    check(build_key(HARDENED, pack, prev="0123456789abcdef")["key"] != g["key"], "prev chains key")
    dt = dict(build_key(_unit(MemoryMax="512M", MemoryHigh=None), pack)["checks"])
    check(dt["memmax.covers_pack"] == -1, "undersized MemoryMax fails")
    dt = dict(build_key(_unit(MemoryMax="64G", MemoryHigh=None), pack)["checks"])
    check(dt["memmax.tight"] == -1 and dt["memmax.covers_pack"] == 1, "bloated MemoryMax fails tight")
    h = build_key(_unit(MemoryMax=None, MemoryHigh=None), pack)
    check(dict(h["checks"])["memmax.present"] == 0 and h["verdict"] == "hold", "missing MemoryMax holds")
    check(dict(build_key(_unit(MemoryMax="infinity"), pack)["checks"])["memmax.present"] == -1,
          "infinity MemoryMax fails")
    check(dict(build_key(_unit(MemoryHigh="3G"), pack)["checks"])["memhigh.order"] == -1,
          "MemoryHigh above MemoryMax fails")
    check(dict(build_key(_unit(MemorySwapMax=None), pack)["checks"])["swap.off"] == 0, "swap absent undecided")
    wild = _unit(ExecStart=f"/usr/local/bin/bitnetd --host {WILDCARD} --port 8765")
    dw = dict(build_key(wild, pack)["checks"])
    check(dw["bind.no_wildcard"] == -1 and dw["bind.loopback"] == -1, "wildcard bind fails")
    check(dict(build_key(_unit(MemoryMax="25%"), pack, host_ram=16 << 30)["checks"])["memmax.covers_pack"] == 1,
          "percent MemoryMax with host ram")
    check(dict(build_key(HARDENED, 0)["checks"])["pack.known"] == 0, "unknown pack undecided")
    trits = [1, 0, -1, 1, 1, -1, 0]
    gm, back = unpack_jtr1(pack_jtr1(trits))
    check(back == trits and abs(gm - 5 / 7) < 1e-6, "jtr1 roundtrip + gamma")
    blk = pack_i2s(trits)
    check(len(blk) == I2S_BLOCK_BYTES and unpack_i2s(blk, len(trits)) == trits, "i2s 128-block roundtrip")
    check(verdict([1] * 20 + [0]) == "green" and verdict([1] * 19 + [0] * 2) == "hold", "rate 0.05 threshold")
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 1.0, 1) + bytes([0xC0]))
        check(False, "0b11 rejected")
    except ValueError:
        check(True, "0b11 rejected")
    try:
        close_rule("xyz", b"", "")
        check(False, "bad prev rejected")
    except ValueError:
        check(True, "bad prev rejected")
    print(f"selftest key {g['key']}")
    print("SELFTEST " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--unit")
    ap.add_argument("--params", type=int)
    ap.add_argument("--tensors", type=int, default=1)
    ap.add_argument("--pack-file")
    ap.add_argument("--host-ram")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.unit or (a.params is None and not a.pack_file):
        ap.error("--unit and one of --params / --pack-file required")
    with open(a.unit, encoding="utf-8") as fh:
        unit_text = fh.read()
    if a.pack_file:
        pack = pack_file_bytes(a.pack_file)
        exact = True
    else:
        pack = i2s_pack_bytes(a.params, a.tensors)
        exact = True
    host = parse_size(a.host_ram) if a.host_ram else None
    out = build_key(unit_text, pack, host if isinstance(host, int) else None, a.prev, exact)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["verdict"] == "green" else 2


if __name__ == "__main__":
    sys.exit(main())
