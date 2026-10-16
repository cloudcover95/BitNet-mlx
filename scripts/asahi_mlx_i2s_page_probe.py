"""Asahi MLX offline probe: 16 KiB page-aligned I2_S trit-pack plan. Stdlib only.

Roadmap item: JuniorHome docs/ROADMAP.md step 4, "Asahi MLX - probe only from
automations." No Quant Chief pair A assignment was visible for the 2026-10-16
10:20 MT slot, so this run takes the standing fallback: #4 as an offline probe.
No hardware calls (no /proc, /sys, device-tree, sysctl, or GPU queries), no
downloads, no network, no subprocess. Every host fact is a caller-supplied
string (--os, --machine, --compat); the probe only classifies what it is told.

Side covered: I2_S / trit-pack key side (Trit Pack & I2_S). The quant-error
bound side belongs to the buddy (Winsor & Quant Error) in a different file.

New capability vs scripts/i2s_key_builder.py and scripts/i2sd_jtr1_pack.py:
  * classify a declared host into an MLX backend expectation
    (darwin/arm64 -> metal; linux/aarch64 Apple SoC under Asahi -> cpu, since
    MLX's Metal backend is macOS-only; anything else -> unsupported),
  * plan JTR1 frames (magic + gamma + n + 2-bit I2_S payload; not GGUF) as
    tiles padded to the host page size (16384 B on Asahi, 16384 B on macOS
    arm64, 4096 B otherwise) so a later loader can mmap each tile page-aligned,
  * key the tile plan with the close rule chained per tile.
Not a wrapper, test, or re-derivation of any other team file.

Close rule: sha256(prev || pack || text)[:16]; pack = tile bytes, text = tile
descriptor JSON. Rate 0.05 held (RATE is the declared drift budget a tile plan
may spend on padding: padding / tile bytes must stay <= RATE, else the plan
widens rows per tile until it fits or reports over_rate).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Compatible engines: BitNet-mlx, JuniorLLM, MLX (CPU on Asahi, Metal on macOS).
Bind 127.0.0.1 only (named, never opened). mint/socket/layer_one/fetch/ue5_launch false.

Usage:
  python3 asahi_mlx_i2s_page_probe.py --selftest
  python3 asahi_mlx_i2s_page_probe.py --os linux --machine aarch64 --compat apple,t8103 --n 4096
"""
from __future__ import annotations

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
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #4 Asahi MLX, probe only from automations"
GENESIS = "0" * 16
MAGIC = b"JTR1"
HEADER = 16  # magic 4 + gamma f64 8 + n u32 4
PAGE_16K = 16384
PAGE_4K = 4096

# Apple SoC device-tree compatibles (caller supplies the string; never read here).
APPLE_SOC = {
    "t8103": "M1",
    "t6000": "M1 Pro",
    "t6001": "M1 Max",
    "t6002": "M1 Ultra",
    "t8112": "M2",
    "t6020": "M2 Pro",
    "t6021": "M2 Max",
    "t6022": "M2 Ultra",
}

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def close_rule(prev: str, pack: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def classify_host(os_name: str, machine: str, compat: str = "") -> dict:
    """Offline classification of a declared host. No hardware is touched."""
    os_name = (os_name or "").strip().lower()
    machine = (machine or "").strip().lower()
    soc = ""
    for part in (compat or "").lower().replace(" ", "").split(","):
        if part in APPLE_SOC:
            soc = part
    arm = machine in ("arm64", "aarch64")
    if os_name == "darwin" and arm:
        return {"host": "macos-apple", "soc": soc, "chip": APPLE_SOC.get(soc, ""),
                "mlx_backend": "metal", "page": PAGE_16K}
    if os_name == "linux" and arm and soc:
        return {"host": "asahi", "soc": soc, "chip": APPLE_SOC[soc],
                "mlx_backend": "cpu", "page": PAGE_16K}
    if os_name == "linux" and arm:
        return {"host": "linux-arm", "soc": "", "chip": "",
                "mlx_backend": "cpu", "page": PAGE_4K}
    return {"host": "other", "soc": soc, "chip": APPLE_SOC.get(soc, ""),
            "mlx_backend": "unsupported", "page": PAGE_4K}


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
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def frame_bytes(n: int) -> int:
    return HEADER + (2 * n + 7) // 8


def _pad_to(nbytes: int, page: int) -> int:
    return -(-nbytes // page) * page


def plan_tiles(n_total: int, page: int, rate: float = RATE) -> dict:
    """Pick trits-per-tile so JTR1 tiles occupy whole pages with padding <= rate.

    Full tiles span pages_per_tile pages; the last tile is padded only to its
    own page boundary. Widens pages_per_tile (fewer headers) until
    padding / planned bytes <= rate. Tiny tensors that cannot meet the rate in
    one page are reported honestly with over_rate=True.
    """
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    best = None
    for pages in (1, 2, 4, 8, 16, 32, 64):
        tile = pages * page
        per = ((tile - HEADER) * 8) // 2
        per -= per % 4  # whole payload bytes
        tiles = -(-n_total // per)
        last = n_total - per * (tiles - 1)
        sizes = [tile] * (tiles - 1) + [_pad_to(frame_bytes(last), page)]
        used = frame_bytes(per) * (tiles - 1) + frame_bytes(last)
        ratio = (sum(sizes) - used) / float(sum(sizes))
        cand = {"pages_per_tile": pages, "tile_bytes": tile, "trits_per_tile": per,
                "tiles": tiles, "last_tile_bytes": sizes[-1],
                "pad_ratio": round(ratio, 6), "over_rate": ratio > rate}
        if best is None or ratio < best["pad_ratio"]:
            best = cand
        if ratio <= rate:
            return cand
    return best


def build_tiles(trits, gamma: float, host: dict, prev: str = GENESIS) -> dict:
    """Page-aligned JTR1 tiles plus a close-rule key chained per tile."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    plan = plan_tiles(len(trits), host["page"])
    per = plan["trits_per_tile"]
    tiles, keys = [], []
    key = prev
    for i in range(plan["tiles"]):
        chunk = trits[i * per:(i + 1) * per]
        body = MAGIC + struct.pack(">dI", float(gamma), len(chunk)) + pack_i2s(chunk)
        tile = body + b"\x00" * (_pad_to(len(body), host["page"]) - len(body))
        desc = json.dumps({"i": i, "n": len(chunk), "page": host["page"],
                           "backend": host["mlx_backend"], "rate": RATE}, sort_keys=True)
        key = close_rule(key, tile, desc)
        tiles.append(tile)
        keys.append(key)
    return {"plan": plan, "tiles": tiles, "keys": keys, "close": key}


def read_tile(tile: bytes) -> dict:
    if tile[:4] != MAGIC:
        raise ValueError("not a JTR1 tile")
    gamma, n = struct.unpack(">dI", tile[4:16])
    return {"gamma": gamma, "n": n, "trits": unpack_i2s(tile[16:frame_bytes(n)], n)}


def _demo_trits(n: int) -> list:
    return [((i * 7 + 3) % 3) - 1 for i in range(n)]


def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and bool(cond)

    check("rate is 0.05", RATE == 0.05)
    check("flags false", not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH))
    check("bind 127.0.0.1", BIND == "127.0.0.1")
    check("ports", PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"))
    ref = hashlib.sha256(b"0" * 16 + b"\x01\x02" + b"t").hexdigest()[:16]
    check("close rule sha256(prev||pack||text)[:16]", close_rule(GENESIS, b"\x01\x02", "t") == ref)

    a = classify_host("linux", "aarch64", "apple,t8103")
    check("asahi M1 -> cpu backend, 16K page", a["host"] == "asahi" and a["chip"] == "M1"
          and a["mlx_backend"] == "cpu" and a["page"] == PAGE_16K)
    m = classify_host("Darwin", "arm64", "")
    check("macOS arm64 -> metal", m["mlx_backend"] == "metal" and m["page"] == PAGE_16K)
    x = classify_host("linux", "x86_64", "")
    check("x86 -> unsupported, 4K page", x["mlx_backend"] == "unsupported" and x["page"] == PAGE_4K)

    for n in (1, 5, 4096):
        check("plan n=%d tiny -> over_rate reported" % n, plan_tiles(n, PAGE_16K)["over_rate"])
    for n in (1000003, 4194304, 9999991):
        p = plan_tiles(n, PAGE_16K)
        check("plan n=%d pad<=rate" % n, not p["over_rate"] and p["pad_ratio"] <= RATE)

    trits = _demo_trits(1000003)
    out = build_tiles(trits, 0.75, a)
    check("tiles page-aligned", all(len(t) % PAGE_16K == 0 for t in out["tiles"]))
    back = []
    for t in out["tiles"]:
        r = read_tile(t)
        back.extend(r["trits"])
        if r["gamma"] != 0.75:
            back = None
            break
    check("round trip", back == trits)
    again = build_tiles(trits, 0.75, a)
    check("key deterministic", again["close"] == out["close"])
    flipped = list(trits)
    flipped[12345] = -flipped[12345] if flipped[12345] else 1
    check("key detects change", build_tiles(flipped, 0.75, a)["close"] != out["close"])
    check("key depends on backend", build_tiles(trits, 0.75, m)["close"] != out["close"])
    print("close", out["close"], "tiles", out["plan"]["tiles"], "pad", out["plan"]["pad_ratio"])
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv) -> int:
    if "--selftest" in argv:
        return selftest()
    args = {"--os": "", "--machine": "", "--compat": "", "--n": "4096", "--gamma": "1.0"}
    it = iter(argv)
    for a in it:
        if a in args:
            args[a] = next(it, "")
    host = classify_host(args["--os"], args["--machine"], args["--compat"])
    out = build_tiles(_demo_trits(int(args["--n"])), float(args["--gamma"]), host)
    print(json.dumps({"roadmap": ROADMAP_ITEM, "host": host, "plan": out["plan"],
                      "close": out["close"], "rate": RATE, "ports": PORTS}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
