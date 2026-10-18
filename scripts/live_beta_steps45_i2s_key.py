"""Live beta steps 4 and 5 I2_S key: stdlib-import trits + exit-code trits. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 4 (JuniorClimbs
`scripts/stonefield_stdlib_probe.py`, no FastAPI required) and step 5
(`python scripts/backfill_check.py` exits 0); docs/COMPILED_BACKLOG.md
"A - live beta suite". Quant Chief pair A, slot 2026-10-17 18:05 MT.
Trit Pack & I2_S side: the step-4/5 trit pack and its I2_S key. Winsor & Quant
Error covers the quant-error bound side of the same item in a different file.

New capability (not in live_beta_gate_jtr1.py, which only packed a six-gate
verdict JSON for steps 1-3): an offline audit that
  * parses a Python source TEXT handed to it with `ast` (never imports or runs
    it) and gives one balanced trit per top-level imported module:
        +1  module is in sys.stdlib_module_names (or a relative import)
        -1  module is on the forbidden list (fastapi, starlette, uvicorn,
            pydantic, requests, httpx, ...): step 4 says "no FastAPI required"
         0  third-party but not forbidden (unknown)
  * parses an exit-code transcript TEXT handed to it (lines like
    `exit=0`, `exit code: 1`, `returncode 0`) and gives one trit per run:
        +1 exit 0, -1 nonzero exit, 0 line present but code unreadable
  * packs [step-4 import trits] + [step-5 exit trits] as one JTR1 frame and
    closes it into an I2_S key under the close rule.

Verdict: step 4 green when no -1 import trit and the unknown (0) fraction is
<= RATE (0.05). Step 5 green when at least one run is present and the LAST
run trit is +1 with no -1 in the trailing window of 3 runs. Both green ->
release fraction RATE (0.05 cohort ramp step); otherwise 0 (held).

It does not run the probe, backfill_check, juniorctl or any test; no
subprocess, no network, no socket. It does not wrap, test or re-derive any
other team file: it only reads text handed to it (--source / --transcript
paths or the built-in selftest samples).

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected on decode)
  gamma = absmean of the trit vector (|t| summed / n; 0.0 when n == 0)

Close rule: sha256(prev || pack || text)[:16]; prev = 16-hex ASCII,
pack = full JTR1 bytes, text = canonical verdict JSON (sorted keys, compact).
Rate 0.05. Compatible engines: BitNet-mlx, JuniorLLM, i2sd userspace,
JuniorOS overlay. Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no hardware calls.

Usage: python3 scripts/live_beta_steps45_i2s_key.py [--selftest]
       [--source PATH] [--transcript PATH] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import ast
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
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("BitNet-mlx", "JuniorLLM", "i2sd userspace", "JuniorOS overlay")
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md Live beta steps 4-5: stonefield_stdlib_probe "
    "(no FastAPI) + backfill_check exits 0 (COMPILED_BACKLOG A)"
)
MAGIC = b"JTR1"
GENESIS = "0" * 16
WINDOW = 3
FORBIDDEN = frozenset({
    "fastapi", "starlette", "uvicorn", "pydantic", "requests", "httpx",
    "aiohttp", "flask", "django", "gunicorn",
})
_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}
_EXIT_RE = re.compile(r"(?i)\b(?:exit(?:[ _]?code)?|returncode|rc)\b\s*[:=]?\s*(\S+)")


def _stdlib_names() -> frozenset:
    names = getattr(sys, "stdlib_module_names", None)
    if names:
        return frozenset(names)
    return frozenset(sys.builtin_module_names) | frozenset({
        "os", "sys", "json", "re", "ast", "argparse", "hashlib", "struct",
        "pathlib", "typing", "dataclasses", "subprocess", "time", "math",
        "collections", "itertools", "functools", "io", "tempfile", "unittest",
        "logging", "datetime", "csv", "sqlite3", "urllib", "http", "socket",
        "shutil", "textwrap", "string", "random", "statistics",
    })


STDLIB = _stdlib_names()


def import_trits(source: str):
    """Return [(module, trit)] for every import in source, in source order."""
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                out.append(("." * node.level + (node.module or ""), 1, node.lineno))
                continue
            mods = [node.module or ""]
        else:
            continue
        for m in mods:
            top = m.split(".")[0]
            if top in FORBIDDEN:
                t = -1
            elif top in STDLIB or top == "__future__":
                t = 1
            else:
                t = 0
            out.append((m, t, node.lineno))
    out.sort(key=lambda r: r[2])
    return [(m, t) for m, t, _ in out]


def exit_trits(transcript: str):
    out = []
    for line in transcript.splitlines():
        mt = _EXIT_RE.search(line)
        if not mt:
            continue
        tok = mt.group(1).strip().rstrip(".,;)")
        try:
            code = int(tok)
        except ValueError:
            out.append(0)
            continue
        out.append(1 if code == 0 else -1)
    return out


def pack_codes(trits) -> bytes:
    buf = bytearray((2 * len(trits) + 7) // 8)
    for i, t in enumerate(trits):
        if t not in _ENC:
            raise ValueError("not a balanced trit: %r" % (t,))
        buf[(2 * i) // 8] |= _ENC[t] << (6 - (2 * i) % 8)
    return bytes(buf)


def unpack_codes(payload: bytes, n: int):
    out = []
    for i in range(n):
        c = (payload[(2 * i) // 8] >> (6 - (2 * i) % 8)) & 0b11
        if c == 0b11:
            raise ValueError("reserved code 0b11 at index %d" % i)
        out.append(_DEC[c])
    return out


def jtr1_pack(trits) -> bytes:
    n = len(trits)
    gamma = (sum(abs(t) for t in trits) / n) if n else 0.0
    return MAGIC + struct.pack("<fI", gamma, n) + pack_codes(trits)


def jtr1_unpack(blob: bytes):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    payload = blob[12:]
    if len(payload) != (2 * n + 7) // 8:
        raise ValueError("payload length mismatch")
    return gamma, unpack_codes(payload, n)


def close_key(prev: str, pack: bytes, text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    h = hashlib.sha256()
    h.update(prev.encode("ascii"))
    h.update(pack)
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def build(source: str, transcript: str, prev: str = GENESIS) -> dict:
    imps = import_trits(source)
    s4 = [t for _, t in imps]
    s5 = exit_trits(transcript)
    unknown = (s4.count(0) / len(s4)) if s4 else 0.0
    step4 = (-1 not in s4) and unknown <= RATE
    step5 = bool(s5) and s5[-1] == 1 and (-1 not in s5[-WINDOW:])
    trits = s4 + s5
    pack = jtr1_pack(trits)
    verdict = {
        "item": ROADMAP_ITEM,
        "step4_imports": [[m, t] for m, t in imps],
        "step4_forbidden": sorted({m for m, t in imps if t == -1}),
        "step4_unknown_frac": round(unknown, 6),
        "step4_green": step4,
        "step5_exits": s5,
        "step5_green": step5,
        "release": RATE if (step4 and step5) else 0.0,
        "rate": RATE,
        "bind": BIND,
        "ports": list(PORTS),
        "engines": list(ENGINES),
        "mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
        "fetch": FETCH, "ue5_launch": UE5_LAUNCH,
    }
    text = json.dumps(verdict, sort_keys=True, separators=(",", ":"))
    verdict["jtr1_hex"] = pack.hex()
    verdict["key"] = close_key(prev, pack, text)
    verdict["prev"] = prev
    return verdict


_GOOD_SRC = (
    "from __future__ import annotations\n"
    "import json, os\nimport sys\nfrom pathlib import Path\n"
    "from . import local_helper\n"
)
_BAD_SRC = "import json\nfrom fastapi import FastAPI\nimport numpy\n"
_GOOD_TX = "backfill_check run 1\nexit=1\nretry\nexit code: 0\nrc=0\nexit=0\n"
_BAD_TX = "exit=0\nreturncode 2\n"


def selftest() -> int:
    assert RATE == 0.05
    assert BIND == "127.0.0.1"
    assert not any((MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH))
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    # codec round trip incl. odd length
    for v in ([], [1], [-1, 0, 1], [1, -1, 0, 0, 1, -1, 1]):
        g, back = jtr1_unpack(jtr1_pack(v))
        assert back == v, (v, back)
    try:
        unpack_codes(b"\xc0", 1)
        raise AssertionError("0b11 accepted")
    except ValueError:
        pass
    # step 4
    good = import_trits(_GOOD_SRC)
    assert all(t == 1 for _, t in good), good
    bad = dict(import_trits(_BAD_SRC))
    assert bad["fastapi"] == -1 and bad["numpy"] == 0 and bad["json"] == 1
    # step 5
    assert exit_trits(_GOOD_TX) == [-1, 1, 1, 1], exit_trits(_GOOD_TX)
    assert exit_trits(_BAD_TX) == [1, -1]
    assert exit_trits("exit=?\n") == [0]
    # verdicts
    v = build(_GOOD_SRC, _GOOD_TX)
    assert v["step4_green"] and v["step5_green"] and v["release"] == RATE
    v_bad4 = build(_BAD_SRC, _GOOD_TX)
    assert not v_bad4["step4_green"] and v_bad4["release"] == 0.0
    assert v_bad4["step4_forbidden"] == ["fastapi"]
    v_bad5 = build(_GOOD_SRC, _BAD_TX)
    assert not v_bad5["step5_green"] and v_bad5["release"] == 0.0
    assert not build(_GOOD_SRC, "")["step5_green"]
    # close rule recomputed independently
    text = json.dumps({k: v[k] for k in v if k not in ("jtr1_hex", "key", "prev")},
                      sort_keys=True, separators=(",", ":"))
    pack = bytes.fromhex(v["jtr1_hex"])
    want = hashlib.sha256(GENESIS.encode() + pack + text.encode()).hexdigest()[:16]
    assert v["key"] == want
    # chaining and determinism
    v2 = build(_GOOD_SRC, _GOOD_TX, prev=v["key"])
    assert v2["key"] != v["key"]
    assert build(_GOOD_SRC, _GOOD_TX)["key"] == v["key"]
    g, trits = jtr1_unpack(pack)
    assert len(trits) == 6 + 4 and abs(g - 1.0) < 1e-6
    print("selftest ok key=%s release=%s rate=%s" % (v["key"], v["release"], RATE))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--source", help="Python source text file to audit (step 4)")
    ap.add_argument("--transcript", help="exit-code transcript text file (step 5)")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    src = open(a.source, encoding="utf-8").read() if a.source else _GOOD_SRC
    tx = open(a.transcript, encoding="utf-8").read() if a.transcript else _GOOD_TX
    print(json.dumps(build(src, tx, a.prev), sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
