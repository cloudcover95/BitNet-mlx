"""16:45 daily audit, BitNet-mlx engine: date-seeded I2_S roundtrip key.

Roadmap anchor: JuniorHome docs/ROADMAP.md, "Automations (current)" ->
"16:45 daily audit" (Chief pair A, slot 2026-10-23 14:05 MT: remaining
engines BitNet-mlx/JuniorQuant). This file covers the BitNet-mlx engine row
on the I2_S / trit pack side. Winsor & Quant Error owns the quant-error bound
side of the same item in a different file and path.

New capability: each audit day D gets its own deterministic synthetic weight
block, seeded by sha256("BitNet-mlx|16:45|D"). The block is ternarized with the
BitNet b1.58 absmean rule (per-row gamma = mean |w|, q = clip(round(w/gamma),
-1, 1)), packed into I2_S rows (4 codes per byte, MSB first), unpacked, and
audited with eight checks that become one trit each (+1 pass, -1 fail,
0 not evaluated):

  t0 roundtrip    unpack(pack(q)) == q exactly
  t1 no_code3     no reserved 0b11 code in any packed byte
  t2 gamma_ok     every row gamma finite and > 0
  t3 row_bytes    packed size == rows * ceil(cols / 4)
  t4 sparsity     zero fraction inside [RATE, 1 - RATE]
  t5 sign_agree   sign(q * gamma) == q for every weight
  t6 slot_utc     16:45 America/Denver maps to 22:45Z (MDT) or 23:45Z (MST)
                  for D (0 if zoneinfo has no tz data on this box)
  t7 policy       bind 127.0.0.1 and mint/socket/layer_one/fetch/ue5_launch off

Days chain: --days N audits the N days ending at --date, and each day's key is
closed onto the previous day's key, so one edited day changes every later key.
No model pull, no GGUF read, no network: the weights are synthetic.

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected); gamma = mean |trit|

Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16), pack = JTR1 bytes, text = canonical JSON of the day report.
Rate 0.05: a day is "green" only with zero -1 trits and at most RATE * n zeros.

Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM (report fields only).
Compatible engines: bitnetd 127.0.0.1:8765, bitnet.cpp I2_S (llama-server),
i2sd userspace 127.0.0.1:8767, BitNet-mlx, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false. Stdlib only.

Usage:
  python3 scripts/daily_audit_1645_bitnet_mlx_roundtrip_i2s_key.py --selftest
  python3 scripts/daily_audit_1645_bitnet_mlx_roundtrip_i2s_key.py --date 2026-10-23 --days 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
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
ENGINES = ("bitnetd 127.0.0.1:8765", "bitnet.cpp I2_S (llama-server)",
           "i2sd userspace 127.0.0.1:8767", "BitNet-mlx", "JuniorLLM")
ENGINE = "BitNet-mlx"
SLOT = (16, 45)
TZ = "America/Denver"
MAGIC = b"JTR1"
GENESIS = "0" * 16
ROWS, COLS = 8, 66          # 66 cols forces a padded tail byte per row
CHECKS = ("roundtrip", "no_code3", "gamma_ok", "row_bytes",
          "sparsity", "sign_agree", "slot_utc", "policy")
CODE = {0: 0b00, 1: 0b01, -1: 0b10}
DECODE = {0b00: 0, 0b01: 1, 0b10: -1}


# ---------------------------------------------------------------- ternary
def day_block(day: dt.date) -> list[list[float]]:
    seed = hashlib.sha256(f"{ENGINE}|16:45|{day.isoformat()}".encode()).digest()
    rng = random.Random(int.from_bytes(seed[:8], "big"))
    return [[rng.gauss(0.0, 1.0) for _ in range(COLS)] for _ in range(ROWS)]


def absmean_ternary(block):
    gammas, q = [], []
    for row in block:
        g = sum(abs(x) for x in row) / len(row)
        gammas.append(g)
        q.append([max(-1, min(1, int(round(x / (g + 1e-8))))) for x in row])
    return gammas, q


def i2s_pack_rows(q) -> bytes:
    out = bytearray()
    for row in q:
        for i in range(0, len(row), 4):
            b = 0
            for j in range(4):
                c = CODE[row[i + j]] if i + j < len(row) else 0
                b |= c << (6 - 2 * j)
            out.append(b)
    return bytes(out)


def i2s_unpack_rows(blob: bytes, rows: int, cols: int):
    per = (cols + 3) // 4
    if len(blob) != rows * per:
        raise ValueError("i2s size mismatch")
    q = []
    for r in range(rows):
        row = []
        for b in blob[r * per:(r + 1) * per]:
            for j in range(4):
                c = (b >> (6 - 2 * j)) & 3
                if c == 3:
                    raise ValueError("reserved code 0b11")
                row.append(DECODE[c])
        q.append(row[:cols])
    return q


def has_code3(blob: bytes) -> bool:
    return any(((b >> s) & 3) == 3 for b in blob for s in (6, 4, 2, 0))


# ---------------------------------------------------------------- slot
def slot_utc_trit(day: dt.date) -> int:
    try:
        from zoneinfo import ZoneInfo
        z = ZoneInfo(TZ)
    except Exception:
        return 0
    local = dt.datetime(day.year, day.month, day.day, *SLOT, tzinfo=z)
    off = local.utcoffset()
    utc = local.astimezone(dt.timezone.utc)
    want = (22, 45) if off == dt.timedelta(hours=-6) else (23, 45)
    if off not in (dt.timedelta(hours=-6), dt.timedelta(hours=-7)):
        return -1
    return 1 if (utc.hour, utc.minute) == want else -1


def policy_trit() -> int:
    off = not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    return 1 if off and BIND == "127.0.0.1" else -1


# ---------------------------------------------------------------- audit
def audit_day(day: dt.date, tamper: str = "") -> list[int]:
    gammas, q = absmean_ternary(day_block(day))
    if tamper == "zero_gamma":
        gammas[0] = 0.0
    blob = i2s_pack_rows(q)
    if tamper == "flip_byte":
        blob = bytes([blob[0] ^ 0b01000000]) + blob[1:]
    if tamper == "code3":
        blob = bytes([blob[0] | 0b11000000]) + blob[1:]
    if tamper == "short":
        blob = blob[:-1]
    t = [0] * len(CHECKS)
    try:
        t[0] = 1 if i2s_unpack_rows(blob, ROWS, COLS) == q else -1
    except ValueError:
        t[0] = -1
    t[1] = -1 if has_code3(blob) else 1
    t[2] = 1 if all(math.isfinite(g) and g > 0 for g in gammas) else -1
    t[3] = 1 if len(blob) == ROWS * ((COLS + 3) // 4) else -1
    flat = [v for row in q for v in row]
    zf = flat.count(0) / len(flat)
    t[4] = 1 if RATE <= zf <= 1 - RATE else -1
    ok = all((1 if v * g > 0 else -1 if v * g < 0 else 0) == v
             for row, g in zip(q, gammas) for v in row)
    t[5] = 1 if ok else -1
    t[6] = slot_utc_trit(day)
    t[7] = policy_trit()
    return t


def jtr1_pack(trits: list[int]) -> bytes:
    n = len(trits)
    gamma = (sum(abs(x) for x in trits) / n) if n else 0.0
    payload = bytearray((n + 3) // 4)
    for i, x in enumerate(trits):
        payload[i // 4] |= CODE[x] << (6 - 2 * (i % 4))
    return MAGIC + struct.pack("<fI", gamma, n) + bytes(payload)


def jtr1_unpack(blob: bytes):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    out = []
    for i in range(n):
        c = (blob[12 + i // 4] >> (6 - 2 * (i % 4))) & 3
        if c == 3:
            raise ValueError("reserved code 0b11")
        out.append(DECODE[c])
    return gamma, out


def close_key(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits: list[int]) -> str:
    if -1 in trits:
        return "fail"
    return "green" if trits.count(0) <= RATE * len(trits) else "hold"


def run(end: dt.date, days: int, tamper_day: dt.date | None = None,
        tamper: str = "") -> dict:
    prev, rows = GENESIS, []
    for k in range(days - 1, -1, -1):
        d = end - dt.timedelta(days=k)
        t = audit_day(d, tamper if d == tamper_day else "")
        pk = jtr1_pack(t)
        rep = {"engine": ENGINE, "item": "ROADMAP Automations 16:45 daily audit",
               "date": d.isoformat(), "slot": "16:45 " + TZ,
               "checks": dict(zip(CHECKS, t)), "verdict": verdict(t),
               "rate": RATE, "bind": BIND, "ports": list(PORTS)}
        text = json.dumps(rep, sort_keys=True, separators=(",", ":"))
        prev = close_key(prev, pk, text)
        rep["key"] = prev
        rows.append(rep)
    return {"days": rows, "key": prev, "engines": list(ENGINES),
            "green": sum(r["verdict"] == "green" for r in rows),
            "fail": sum(r["verdict"] == "fail" for r in rows)}


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def check(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    check("rate is 0.05", RATE == 0.05)
    check("policy flags off, bind 127.0.0.1", policy_trit() == 1)
    pk = jtr1_pack([1, 0, -1])
    check("jtr1 header", pk[:4] == b"JTR1" and len(pk) == 13)
    check("jtr1 roundtrip", jtr1_unpack(pk)[1] == [1, 0, -1])
    try:
        jtr1_unpack(MAGIC + struct.pack("<fI", 1.0, 1) + b"\xc0")
        check("jtr1 rejects 0b11", False)
    except ValueError:
        check("jtr1 rejects 0b11", True)
    check("close rule sha256(prev||pack||text)[:16]",
          close_key(GENESIS, pk, "x") ==
          hashlib.sha256(GENESIS.encode() + pk + b"x").hexdigest()[:16])
    d = dt.date(2026, 10, 23)
    check("day block deterministic", day_block(d) == day_block(d))
    check("day blocks differ by date", day_block(d) != day_block(d - dt.timedelta(days=1)))
    t = audit_day(d)
    check("clean day all pass", t == [1] * len(CHECKS) or t == [1] * 6 + [0, 1])
    for name, idx in (("flip_byte", 0), ("code3", 1), ("zero_gamma", 2), ("short", 3)):
        check("tamper %s caught at t%d" % (name, idx), audit_day(d, name)[idx] == -1)
    try:
        import zoneinfo  # noqa: F401
        zoneinfo.ZoneInfo(TZ)
        check("slot MDT 22:45Z (Jul)", slot_utc_trit(dt.date(2026, 7, 1)) == 1)
        check("slot MST 23:45Z (Jan)", slot_utc_trit(dt.date(2026, 1, 15)) == 1)
    except Exception:
        print("skip slot checks (no tz data)")
    a = run(d, 7)
    b = run(d, 7)
    check("7-day chain deterministic", a["key"] == b["key"])
    check("7-day chain all green", a["green"] == 7 and a["fail"] == 0)
    c = run(d, 7, tamper_day=d - dt.timedelta(days=3), tamper="flip_byte")
    check("tampered mid-day changes final key", c["key"] != a["key"] and c["fail"] == 1)
    check("earlier days unaffected by later tamper",
          [r["key"] for r in c["days"][:3]] == [r["key"] for r in a["days"][:3]])
    print("selftest key", a["key"])
    print("selftest", "PASS" if not fails else "FAIL %d" % len(fails))
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--date", default="")
    ap.add_argument("--days", type=int, default=1)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    end = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
    print(json.dumps(run(end, max(1, a.days)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
