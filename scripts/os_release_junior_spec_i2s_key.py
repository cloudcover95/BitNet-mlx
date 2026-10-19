"""os-release.junior spec-conformance trits -> JTR1 + I2_S 128-block key.

Stdlib only, offline, read-only. No socket, no subprocess, no network.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay", the
os-release.junior part (JuniorLLM rails/linux/os-release.junior, copied by the
overlay installer to /etc and /usr/lib). Quant Chief, pair A, slot
2026-10-18 18:05 MT, "os-release.junior gaps only". This is the I2_S / trit
pack side; the Winsor / quant-error side belongs to Winsor & Quant Error in a
different file and path.

Already covered elsewhere (not re-derived here): ID value, ID_LIKE base
(Debian/Alpine) and no-kernel-blob trits in JuniorLLM
scripts/junioros_overlay_jtr1_key.py (7d40533); the file itself (stocksnode
f49707c); PATH, hardened bitnetd MemoryMax and user-unit parity.

The gap: nothing checks os-release.junior against the os-release(5) format
that systemd, /etc/os-release readers and Python platform.freedesktop_os_release
actually parse, or checks the JUNIOR_* extension keys. A file that "has ID" can
still be unreadable to a strict parser (bad quoting, lowercase key, duplicate
key, CRLF) or advertise a wildcard bitnetd address. New capability, one trit
per check:

  syntax.lines        every non-blank, non-# line is KEY=VALUE
  syntax.key_case     keys are [A-Z][A-Z0-9_]*
  syntax.quoting      unquoted values hold no whitespace/shell specials; quoted
                      values close and escape any inner " $ ` \\
  syntax.no_dup       no key assigned twice
  syntax.newline      no CR; ends with LF (missing final LF -> 0)
  field.name          NAME set (absent -> 0, spec default "Linux")
  field.pretty_name   PRETTY_NAME set (absent -> 0)
  field.id_charset    ID matches [a-z0-9._-]+ (absent -> 0)
  field.id_like       ID_LIKE tokens match [a-z0-9._-]+ and never name ID itself
  field.variant_id    VARIANT_ID charset ok (absent -> 0)
  field.version       VERSION_ID / BUILD_ID / IMAGE_VERSION present with a
                      valid charset (none -> 0: overlay has no version identity)
  field.urls          every *_URL is https://host (http -> 0, other -> -1)
  ext.vendor_prefix   every non-standard key is JUNIOR_-prefixed
  ext.bitnetd         JUNIOR_BITNETD == 127.0.0.1:8765 (wildcard/other -> -1)
  ext.offline         JUNIOR_OFFLINE == "1" (absent -> 0, anything else -> -1)
  ext.charter         JUNIOR_CHARTER names a known relative doc, no ".."
  install.parity      optional /usr/lib copy byte-identical to /etc copy
                      (not given -> 0, differs -> -1)

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  JTR1 codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected)
  gamma = mean |trit| (share of checks that were decided)
I2_S block: code = trit + 1 (-1 -> 0, 0 -> 1, +1 -> 2), four per byte MSB first,
  padded with code 1 (zero) to a 128-trit block, as BitNet I2_S stores ternary.

Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16), pack = full JTR1 bytes, text = canonical JSON report.
Rate 0.05: green needs zero -1 trits and at most 5% undecided (0) trits.
Compatible engines: BitNet-mlx, JuniorLLM, bitnetd 127.0.0.1:8765, i2sd
userspace 127.0.0.1:8767, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (endpoints named, never dialed).
mint/socket/layer_one/fetch/ue5_launch false. No model pull, no GGUF download.

Usage:
  python3 scripts/os_release_junior_spec_i2s_key.py --selftest
  python3 scripts/os_release_junior_spec_i2s_key.py --os-release /etc/os-release.junior \
      [--usr-lib /usr/lib/os-release.junior] [--prev HEX16]
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
BITNETD = f"{BIND}:8765"
I2SD = f"{BIND}:8767"
WILDCARDS = (".".join(["0"] * 4), "[::]", "::", "*")  # deny-check only
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("BitNet-mlx", "JuniorLLM", "bitnetd 127.0.0.1:8765",
           "i2sd userspace 127.0.0.1:8767", "JuniorOS overlay")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: os-release.junior "
                "os-release(5) syntax + JUNIOR_* extension gaps")
CHARTERS = ("docs/BETA_TO_OS.md", "docs/ROADMAP.md", "docs/COMPILED_BACKLOG.md")
STANDARD = frozenset((
    "NAME ID ID_LIKE PRETTY_NAME CPE_NAME VARIANT VARIANT_ID VERSION VERSION_ID "
    "VERSION_CODENAME BUILD_ID IMAGE_ID IMAGE_VERSION HOME_URL DOCUMENTATION_URL "
    "SUPPORT_URL BUG_REPORT_URL PRIVACY_POLICY_URL SUPPORT_END LOGO ANSI_COLOR "
    "VENDOR_NAME VENDOR_URL DEFAULT_HOSTNAME ARCHITECTURE SYSEXT_LEVEL "
    "CONFEXT_LEVEL SYSEXT_SCOPE CONFEXT_SCOPE PORTABLE_PREFIXES").split())
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}

_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_TOKEN = re.compile(r"^[a-z0-9._-]+$")
_VER = re.compile(r"^[A-Za-z0-9._~+-]+$")
_URL = re.compile(r"^(https?)://[A-Za-z0-9.-]+(:\d+)?(/\S*)?$")
_UNQUOTED_BAD = set(" \t\"'$`\\;&|<>()*?[]#~!{}")


# ---------------------------------------------------------------- parsing
def _unquote(raw):
    """Return (value, ok) under os-release(5) shell-style quoting."""
    if raw[:1] in ("'", '"'):
        q = raw[0]
        if len(raw) < 2 or raw[-1] != q:
            return raw, False
        body = raw[1:-1]
        if q == "'":
            return body, "'" not in body
        out, i = [], 0
        while i < len(body):
            c = body[i]
            if c == "\\":
                if i + 1 >= len(body) or body[i + 1] not in '"$`\\':
                    return body, False
                out.append(body[i + 1])
                i += 2
                continue
            if c in '"$`':
                return body, False
            out.append(c)
            i += 1
        return "".join(out), True
    return raw, not any(c in _UNQUOTED_BAD for c in raw)


def parse(text):
    """Parse os-release text into (fields, stats)."""
    fields, seen = {}, {}
    bad_lines = bad_case = bad_quote = 0
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _LINE.match(s)
        if not m:
            bad_lines += 1
            continue
        key, raw = m.group(1), m.group(2)
        if not _KEY.match(key):
            bad_case += 1
        val, ok = _unquote(raw)
        if not ok:
            bad_quote += 1
        seen[key] = seen.get(key, 0) + 1
        fields[key] = val
    dups = sorted(k for k, c in seen.items() if c > 1)
    return fields, {"bad_lines": bad_lines, "bad_case": bad_case,
                    "bad_quote": bad_quote, "dups": dups}


# ---------------------------------------------------------------- checks
def _tri(ok, absent=False):
    return 0 if absent else (1 if ok else -1)


def check(text, usr_lib=None):
    f, st = parse(text)
    rows = []
    rows.append(("syntax.lines", _tri(st["bad_lines"] == 0)))
    rows.append(("syntax.key_case", _tri(st["bad_case"] == 0)))
    rows.append(("syntax.quoting", _tri(st["bad_quote"] == 0)))
    rows.append(("syntax.no_dup", _tri(not st["dups"])))
    if "\r" in text:
        nl = -1
    else:
        nl = 1 if text.endswith("\n") else 0
    rows.append(("syntax.newline", nl))
    rows.append(("field.name", 1 if f.get("NAME") else 0))
    rows.append(("field.pretty_name", 1 if f.get("PRETTY_NAME") else 0))
    ident = f.get("ID")
    rows.append(("field.id_charset", _tri(bool(ident and _TOKEN.match(ident)), ident is None)))
    like = f.get("ID_LIKE")
    if like is None:
        rows.append(("field.id_like", 0))
    else:
        toks = like.split()
        ok = bool(toks) and all(_TOKEN.match(t) for t in toks) and ident not in toks
        rows.append(("field.id_like", _tri(ok)))
    vid = f.get("VARIANT_ID")
    rows.append(("field.variant_id", _tri(bool(vid and _TOKEN.match(vid)), vid is None)))
    vers = [f[k] for k in ("VERSION_ID", "BUILD_ID", "IMAGE_VERSION") if k in f]
    rows.append(("field.version", _tri(bool(vers) and all(_VER.match(v) for v in vers),
                                       not vers)))
    urls = [f[k] for k in sorted(f) if k.endswith("_URL")]
    if not urls:
        rows.append(("field.urls", 0))
    else:
        schemes = [(_URL.match(u).group(1) if _URL.match(u) else None) for u in urls]
        rows.append(("field.urls", -1 if None in schemes else
                     (1 if all(s == "https" for s in schemes) else 0)))
    extra = [k for k in f if k not in STANDARD]
    rows.append(("ext.vendor_prefix", _tri(all(k.startswith("JUNIOR_") for k in extra))))
    b = f.get("JUNIOR_BITNETD")
    if b is None:
        rows.append(("ext.bitnetd", 0))
    else:
        rows.append(("ext.bitnetd", _tri(b == BITNETD)))
    off = f.get("JUNIOR_OFFLINE")
    rows.append(("ext.offline", _tri(off == "1", off is None)))
    ch = f.get("JUNIOR_CHARTER")
    rows.append(("ext.charter", _tri(ch in CHARTERS and ".." not in (ch or ""), ch is None)))
    if usr_lib is None:
        rows.append(("install.parity", 0))
    else:
        rows.append(("install.parity", _tri(usr_lib == text)))
    return rows, f, st


# ---------------------------------------------------------------- packing
def pack_jtr1(trits):
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


def unpack_jtr1(blob):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    if len(blob) != 12 + (2 * n + 7) // 8:
        raise ValueError("bad length")
    trits = []
    for k in range(n):
        code = (blob[12 + k // 4] >> (6 - 2 * (k % 4))) & 3
        if code == 3:
            raise ValueError("code 0b11 rejected")
        trits.append(_JTR_DEC[code])
    return gamma, trits


def pack_i2s(trits):
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


def close_key(prev, pack, text):
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits):
    if not trits or any(t < 0 for t in trits):
        return "hold"
    undecided = sum(1 for t in trits if t == 0) / len(trits)
    return "green" if undecided <= RATE else "hold"


def build(text, usr_lib=None, prev=GENESIS):
    rows, _, st = check(text, usr_lib)
    trits = [t for _, t in rows]
    pack = pack_jtr1(trits)
    i2s = pack_i2s(trits)
    report = {
        "item": ROADMAP_ITEM,
        "rows": rows,
        "dups": st["dups"],
        "verdict": verdict(trits),
        "rate": RATE,
        "bitnetd": BITNETD,
        "i2sd": I2SD,
        "i2s_sha": hashlib.sha256(i2s).hexdigest()[:16],
        "bind": BIND,
        "ports": list(PORTS),
        "engines": list(ENGINES),
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text_c = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["prev"] = prev
    report["key"] = close_key(prev, pack, text_c)
    return report, pack, i2s


# ---------------------------------------------------------------- selftest
# Mirrors JuniorLLM rails/linux/os-release.junior as of 2026-10-18.
_REAL = (
    'NAME="JuniorOS"\n'
    "ID=junioros\n"
    'ID_LIKE="debian alpine"\n'
    'PRETTY_NAME="JuniorOS overlay (BitNet rail)"\n'
    'VARIANT="overlay"\n'
    "VARIANT_ID=overlay\n"
    'HOME_URL="https://github.com/cloudcover95/JuniorLLM"\n'
    'SUPPORT_URL="https://github.com/cloudcover95/JuniorLLM"\n'
    'JUNIOR_BITNETD="127.0.0.1:8765"\n'
    'JUNIOR_OFFLINE="1"\n'
    'JUNIOR_CHARTER="docs/BETA_TO_OS.md"\n'
    "# Overlay only. Not a custom kernel. No OSWorld score.\n"
)
_GOOD = _REAL.replace("VARIANT_ID=overlay\n", "VARIANT_ID=overlay\nVERSION_ID=0.1\n")
_BAD = (
    "NAME=Junior OS\r\n"
    "id=junioros\r\n"
    "ID=JuniorOS\r\n"
    "ID=JuniorOS\r\n"
    'ID_LIKE="debian JuniorOS"\r\n'
    'PRETTY_NAME="say "hi""\r\n'
    "VERSION_ID=0.1 beta\r\n"
    "HOME_URL=ftp://example\r\n"
    "BITNETD_ADDR=x\r\n"
    "JUNIOR_BITNETD=" + WILDCARDS[0] + ":8765\r\n"
    "JUNIOR_OFFLINE=0\r\n"
    "JUNIOR_CHARTER=../etc/passwd\r\n"
    "just garbage\r\n"
)


def selftest():
    assert RATE == 0.05 and BIND == "127.0.0.1" and BITNETD == "127.0.0.1:8765"
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    # quoting rules
    assert _unquote('"a \\"b\\" \\$x"') == ('a "b" $x', True)
    assert _unquote('"a "b""')[1] is False and _unquote('"open')[1] is False
    assert _unquote("a b")[1] is False and _unquote("'it''")[1] is False
    assert _unquote("debian")[1] is True

    # the real overlay file: everything holds except no version identity
    # and no /usr/lib copy handed in -> 0, 0 -> hold (2/17 undecided > 5%)
    rows, f, st = check(_REAL)
    d = dict(rows)
    assert len(rows) == 17, len(rows)
    assert d["field.version"] == 0 and d["install.parity"] == 0
    assert all(v == 1 for k, v in d.items() if k not in ("field.version", "install.parity")), d
    rep, pack, i2s = build(_REAL)
    assert rep["verdict"] == "hold"

    # with VERSION_ID and an identical /usr/lib copy: all +1 -> green
    rep_g, pack_g, _ = build(_GOOD, usr_lib=_GOOD)
    assert all(t == 1 for _, t in rep_g["rows"]) and rep_g["verdict"] == "green"
    # differing /usr/lib copy -> -1
    assert dict(check(_GOOD, usr_lib=_REAL)[0])["install.parity"] == -1

    # broken file: every syntax / field / ext check that can go -1 does
    db = dict(check(_BAD)[0])
    for k in ("syntax.lines", "syntax.key_case", "syntax.quoting", "syntax.no_dup",
              "syntax.newline", "field.id_charset", "field.id_like", "field.version",
              "field.urls", "ext.vendor_prefix", "ext.bitnetd", "ext.offline",
              "ext.charter"):
        assert db[k] == -1, (k, db[k])
    assert check("ID=x\nJUNIOR_BITNETD=localhost:8765")[0][4] == ("syntax.newline", 0)
    assert dict(check('HOME_URL="http://a.b"\n')[0])["field.urls"] == 0
    assert parse(_BAD)[1]["dups"] == ["ID"]

    # JTR1 round trip
    trits = [t for _, t in rep["rows"]]
    g, back = unpack_jtr1(pack)
    assert back == trits and abs(g - 15 / 17) < 1e-6
    assert pack[:4] == MAGIC and len(pack) == 12 + (2 * 17 + 7) // 8
    try:
        unpack_jtr1(b"GGUF" + pack[4:])
        raise AssertionError("GGUF magic accepted")
    except ValueError:
        pass
    # I2_S 128-block
    blk = pack_i2s([-1, 0, 1, 1, 0])
    assert len(blk) == I2S_BLOCK // 4 and blk[0] == 0b00011010 and blk[1] == 0b01010101
    assert len(i2s) == I2S_BLOCK // 4
    # close rule
    pk = pack_jtr1([1, -1, 0])
    want = hashlib.sha256(GENESIS.encode() + pk + b"t").hexdigest()[:16]
    assert close_key(GENESIS, pk, "t") == want
    rep2, _, _ = build(_REAL, prev=rep["key"])
    assert rep2["key"] != rep["key"] and build(_REAL)[0]["key"] == rep["key"]
    try:
        close_key("XYZ", pk, "t")
        raise AssertionError("bad prev accepted")
    except ValueError:
        pass
    return rep["key"]


def _read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--os-release")
    ap.add_argument("--usr-lib")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        key = selftest()
        print(json.dumps({"selftest": "pass", "key": key, "rate": RATE,
                          "engines": list(ENGINES), "ports": list(PORTS)}))
        return 0
    if not a.os_release:
        ap.error("--os-release is required (or --selftest)")
    rep, _, _ = build(_read(a.os_release),
                      _read(a.usr_lib) if a.usr_lib else None, a.prev)
    print(json.dumps(rep, indent=1, sort_keys=True))
    return 0 if rep["verdict"] == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
