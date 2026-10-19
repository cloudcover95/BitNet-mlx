"""JuniorOS overlay "I2_S 2B4T only if GGUF already on disk": header-only gate key.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay", the clause
"I2_S 2B4T only if GGUF already on disk" (Quant Chief pair A, slot 2026-10-19
05:05 MT, "JuniorOS overlay, remaining gaps only"). This is the I2_S / key
side. The Winsor quant-error bracket for the same clause is Winsor & Quant
Error's file; the "No OSWorld score. No kernel blob." receipts belong to other
teams; os-release.junior, MemoryMax, PATH, Debian/Alpine and user-unit parity
already landed. Nothing here imports, wraps, tests or re-derives any team file.

The gap: the overlay may only turn on I2_S 2B4T when a GGUF is already on disk,
but nothing checks that the on-disk file really is a BitNet b1.58 2B4T I2_S
model rather than any blob named *.gguf, and nothing pins which file was used
without hashing ~1.2 GB. Earlier JuniorLLM scripts/i2s_2b4t_key_pack.py stops
at magic + version.

New capability: a bounded, header-only GGUF v2/v3 walk (metadata KV table and
tensor-info table; tensor data is never read, only os.stat for the extent) that
turns the 2B4T claim into trits, a header fingerprint (sha256 of the header
bytes only) that pins the file, and an I2_S key block in the real bitnet.cpp
128-element striped layout (byte gp of a 32-byte block holds elements gp,
32+gp, 64+gp, 96+gp in bit pairs [7:6] [5:4] [3:2] [1:0], code = trit + 1),
not the sequential MSB-first packing used by earlier keys.

Trits (+1 holds, 0 undecided, -1 violates):
  on_disk        a local regular file was given (no URL, never fetched);
                 absent -> 0 and the gate is "closed", which honours "only if"
  magic          first 4 bytes b"GGUF"
  version        u32 version in (2, 3)
  header_bounds  counts sane, every string/array inside the read cap
  arch           general.architecture starts with "bitnet" (missing -> 0)
  i2s_share      ternary projection weights (blk.*.attn_*/ffn_* .weight, 2-D)
                 typed I2_S (ggml type 36): all +1, some 0, none -1
  i2s_count      exactly 210 ternary tensors (30 blocks x 7 projections) +1
  i2s_bytes      each I2_S span >= n/4 + 32 (quants + f32 scale + pad) +1,
                 >= n/4 only 0, smaller -1
  aligned        general.alignment power of two, every offset aligned
  no_overlap     offsets strictly increasing, spans never overlap
  extent         data_start + last offset + last size <= st_size (stat only)
  params_2b      total elements in [1.8e9, 2.8e9]
  blocks         {arch}.block_count equals distinct blk.N indices
  no_fetch       FETCH false and nothing beyond the header was read

JTR1 (not GGUF): b"JTR1" | gamma <f32 LE> | n <u32 LE> | 2-bit codes MSB first
  (0 -> 00, +1 -> 01, -1 -> 10, 11 rejected); gamma = mean |trit|.
I2_S key block: trits padded with 0 to 128, striped as above, 32 bytes/block.
Close rule: key = sha256(prev || pack || text)[:16], prev 16 hex (genesis
"0"*16), pack = JTR1 bytes || I2_S block, text = canonical JSON report.
Rate 0.05: green needs no -1 and at most 5% undecided trits.
Compatible engines: BitNet-mlx, bitnet.cpp I2_S (ggml type 36), JuniorLLM,
bitnetd 127.0.0.1:8765, i2sd userspace 127.0.0.1:8767, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only (endpoints named, never dialed; no socket opened).
mint/socket/layer_one/fetch/ue5_launch false. No model pull, no GGUF download,
no subprocess, no os.environ reads. Read-only on the GGUF.

Usage:
  python3 scripts/gguf_2b4t_on_disk_gate_i2s_key.py --selftest
  python3 scripts/gguf_2b4t_on_disk_gate_i2s_key.py [--gguf PATH] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import sys
import tempfile

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
BITNETD = f"{BIND}:8765"
I2SD = f"{BIND}:8767"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("BitNet-mlx", "bitnet.cpp I2_S (ggml type 36)", "JuniorLLM",
           "bitnetd 127.0.0.1:8765", "i2sd userspace 127.0.0.1:8767",
           "JuniorOS overlay")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: "
                "I2_S 2B4T only if GGUF already on disk")
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
GGML_I2_S = 36
GGUF_MAGIC = b"GGUF"
HEADER_CAP = 64 << 20          # never read more than 64 MiB of header
MAX_COUNT = 1 << 20
PARAMS_2B = (1_800_000_000, 2_800_000_000)
TERNARY_2B4T = 210
TERNARY_SUFFIX = ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                  "attn_output.weight", "ffn_gate.weight", "ffn_up.weight",
                  "ffn_down.weight")
ROWS = ("on_disk", "magic", "version", "header_bounds", "arch", "i2s_share",
        "i2s_count", "i2s_bytes", "aligned", "no_overlap", "extent",
        "params_2b", "blocks", "no_fetch")
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}
# GGUF scalar value types -> struct format
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_BLK = re.compile(r"^blk\.(\d+)\.")


# ------------------------------------------------------------ header reader
class _Cap(Exception):
    pass


class _Reader:
    """Sequential reader over a file object, hard-capped at HEADER_CAP bytes."""

    def __init__(self, fh, cap=HEADER_CAP):
        self.fh, self.cap, self.pos = fh, cap, 0

    def take(self, n):
        if n < 0 or self.pos + n > self.cap:
            raise _Cap("header read cap")
        b = self.fh.read(n)
        if len(b) != n:
            raise _Cap("truncated header")
        self.pos += n
        return b

    def u(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.u("<Q")
        if n > self.cap:
            raise _Cap("string length")
        return self.take(n).decode("utf-8", "replace")

    def value(self, vtype, depth=0):
        if vtype in _SCALAR:
            return self.u(_SCALAR[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9 and depth == 0:
            itype, n = self.u("<I"), self.u("<Q")
            if n > self.cap:
                raise _Cap("array length")
            if itype in _SCALAR:           # skip scalar arrays without keeping them
                self.take(n * struct.calcsize(_SCALAR[itype]))
                return {"array": itype, "n": n}
            if itype == 8:
                for _ in range(n):
                    self.string()
                return {"array": 8, "n": n}
            raise _Cap("nested array")
        raise _Cap(f"value type {vtype}")


def read_header(path, cap=HEADER_CAP):
    """Walk a GGUF header. Returns dict; never reads tensor data."""
    out = {"magic": None, "version": None, "kv": {}, "tensors": [],
           "bounds_ok": False, "header_end": 0, "data_start": 0,
           "alignment": 32, "size": 0, "header_sha": None, "error": None}
    st = os.stat(path)
    out["size"] = st.st_size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        r = _Reader(fh, min(cap, st.st_size))
        orig = r.take

        def take(n):
            b = orig(n)
            h.update(b)
            return b
        r.take = take
        try:
            out["magic"] = r.take(4)
            if out["magic"] != GGUF_MAGIC:
                return out
            out["version"] = r.u("<I")
            if out["version"] not in (2, 3):
                return out
            nt, nkv = r.u("<Q"), r.u("<Q")
            if nt > MAX_COUNT or nkv > MAX_COUNT:
                raise _Cap("count")
            for _ in range(nkv):
                key = r.string()
                out["kv"][key] = r.value(r.u("<I"))
            for _ in range(nt):
                name = r.string()
                nd = r.u("<I")
                if nd > 8:
                    raise _Cap("n_dims")
                dims = [r.u("<Q") for _ in range(nd)]
                out["tensors"].append({"name": name, "dims": dims,
                                       "type": r.u("<I"), "offset": r.u("<Q")})
            al = out["kv"].get("general.alignment", 32)
            out["alignment"] = al if isinstance(al, int) else 0
            out["header_end"] = r.pos
            a = out["alignment"] or 1
            out["data_start"] = r.pos + ((-r.pos) % a)
            out["bounds_ok"] = True
        except (_Cap, struct.error) as e:
            out["error"] = str(e)
    out["header_sha"] = h.hexdigest()[:16]
    out["bytes_read"] = r.pos
    return out


# -------------------------------------------------------------------- trits
def _ne(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def _ternary(t):
    return (len(t["dims"]) == 2 and _BLK.match(t["name"]) is not None
            and t["name"].endswith(TERNARY_SUFFIX))


def check(path):
    rows = dict.fromkeys(ROWS, 0)
    info = {"path": None, "gate": "closed", "header_sha": None}
    rows["no_fetch"] = 1 if not FETCH else -1
    if not path:
        return rows, info
    if "://" in path:
        rows["on_disk"] = rows["no_fetch"] = -1
        info["gate"] = "refused"
        return rows, info
    try:
        st = os.stat(path)
    except OSError:
        return rows, info
    if not stat.S_ISREG(st.st_mode):
        rows["on_disk"] = -1
        info["gate"] = "refused"
        return rows, info
    rows["on_disk"] = 1
    info["path"] = os.path.basename(path)
    hd = read_header(path)
    info["header_sha"] = hd["header_sha"]
    info["gate"] = "open"
    rows["magic"] = 1 if hd["magic"] == GGUF_MAGIC else -1
    if rows["magic"] < 0:
        return rows, info
    rows["version"] = 1 if hd["version"] in (2, 3) else -1
    if rows["version"] < 0:
        return rows, info
    rows["header_bounds"] = 1 if hd["bounds_ok"] else -1
    rows["no_fetch"] = 1 if (not FETCH and hd.get("bytes_read", 0) <= max(
        hd["header_end"], HEADER_CAP)) else -1
    if not hd["bounds_ok"]:
        return rows, info
    kv, ts = hd["kv"], hd["tensors"]
    arch = kv.get("general.architecture")
    rows["arch"] = 0 if not isinstance(arch, str) else (
        1 if arch.lower().startswith("bitnet") else -1)
    tern = [t for t in ts if _ternary(t)]
    i2s = [t for t in tern if t["type"] == GGML_I2_S]
    rows["i2s_share"] = (-1 if not tern or not i2s else
                         1 if len(i2s) == len(tern) else 0)
    rows["i2s_count"] = 1 if len(i2s) == TERNARY_2B4T else 0
    al = hd["alignment"]
    pow2 = al > 0 and (al & (al - 1)) == 0
    rows["aligned"] = 1 if pow2 and all(t["offset"] % al == 0 for t in ts) else -1
    order = sorted(ts, key=lambda t: t["offset"])
    offs = [t["offset"] for t in order]
    data_len = hd["size"] - hd["data_start"]
    ends = offs[1:] + [data_len]
    spans = {id(t): e - o for t, o, e in zip(order, offs, ends)}
    rows["no_overlap"] = 1 if ts and all(b > a for a, b in zip(offs, offs[1:])) \
        and all(s > 0 for s in spans.values()) else -1
    worst = 1
    for t in i2s:
        q = (_ne(t["dims"]) + 3) // 4
        sp = spans[id(t)]
        worst = min(worst, 1 if sp >= q + 32 else 0 if sp >= q else -1)
    rows["i2s_bytes"] = worst if i2s else 0
    last_end = offs[-1] + (((_ne(order[-1]["dims"]) + 3) // 4 + 32)
                           if order[-1]["type"] == GGML_I2_S else 1) if ts else 0
    rows["extent"] = 1 if ts and hd["data_start"] + last_end <= hd["size"] else -1
    params = sum(_ne(t["dims"]) for t in ts)
    rows["params_2b"] = 0 if not ts else (
        1 if PARAMS_2B[0] <= params <= PARAMS_2B[1] else -1)
    bc = kv.get(f"{arch}.block_count") if isinstance(arch, str) else None
    idx = {int(m.group(1)) for m in (_BLK.match(t["name"]) for t in ts) if m}
    rows["blocks"] = 0 if not isinstance(bc, int) else (
        1 if bc == len(idx) and idx == set(range(bc)) else -1)
    info.update({"version": hd["version"], "arch": arch, "tensors": len(ts),
                 "ternary": len(tern), "i2s": len(i2s), "params": params,
                 "alignment": al, "header_bytes": hd["header_end"]})
    return rows, info


# ---------------------------------------------------------------- packing
def pack_jtr1(trits):
    n = len(trits)
    gamma = (sum(abs(t) for t in trits) / n) if n else 0.0
    out = bytearray(MAGIC + struct.pack("<fI", gamma, n))
    for i in range(0, n, 4):
        b = 0
        for j, t in enumerate(trits[i:i + 4]):
            b |= _JTR[t] << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def unpack_jtr1(blob):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    if len(blob) != 12 + (n + 3) // 4:
        raise ValueError("bad length")
    out = []
    for k in range(n):
        c = (blob[12 + k // 4] >> (6 - 2 * (k % 4))) & 3
        if c == 3:
            raise ValueError("code 0b11 rejected")
        out.append(_JTR_DEC[c])
    return gamma, out


def pack_i2s_striped(trits):
    """bitnet.cpp I2_S layout: per 128 elems, byte gp holds gp,32+gp,64+gp,96+gp."""
    n = len(trits)
    padded = n + ((-n) % I2S_BLOCK) if n else I2S_BLOCK
    codes = [t + 1 for t in trits] + [1] * (padded - n)
    out = bytearray()
    for base in range(0, padded, I2S_BLOCK):
        for gp in range(32):
            b = 0
            for g in range(4):
                b |= codes[base + 32 * g + gp] << (6 - 2 * g)
            out.append(b)
    return bytes(out)


def unpack_i2s_striped(blob, n):
    if len(blob) % 32:
        raise ValueError("I2_S block length")
    out = [0] * (len(blob) * 4)
    for k, byte in enumerate(blob):
        base, gp = (k // 32) * I2S_BLOCK, k % 32
        for g in range(4):
            c = (byte >> (6 - 2 * g)) & 3
            if c == 3:
                raise ValueError("code 0b11 rejected")
            out[base + 32 * g + gp] = c - 1
    return out[:n]


def close_key(prev, pack, text):
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits, gate):
    if gate == "closed":
        return "closed"
    if not trits or any(t < 0 for t in trits):
        return "hold"
    return "green" if sum(1 for t in trits if t == 0) / len(trits) <= RATE else "hold"


def build(path=None, prev=GENESIS):
    rows, info = check(path)
    trits = [rows[k] for k in ROWS]
    jtr = pack_jtr1(trits)
    i2s = pack_i2s_striped(trits)
    report = {"item": ROADMAP_ITEM, "rows": rows, "info": info,
              "verdict": verdict(trits, info["gate"]), "rate": RATE,
              "ports": list(PORTS), "engines": list(ENGINES),
              "bind": BIND, "bitnetd": BITNETD, "i2sd": I2SD,
              "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                        "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
              "prev": prev}
    text = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["key"] = close_key(prev, jtr + i2s, text)
    return report, jtr, i2s, text


# --------------------------------------------------------------- selftest
def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_str(k, v):
    return _s(k) + struct.pack("<I", 8) + _s(v)


def _kv_u32(k, v):
    return _s(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def synth_gguf(path, blocks=30, i2s=True, arch="bitnet-b1.58", spoil=None):
    """Write a sparse synthetic 2B4T-shaped GGUF (header real, data is holes)."""
    hid, kvd, ffn, voc, al = 2560, 640, 6912, 128256, 32
    ts = [("token_embd.weight", [hid, voc], 1)]
    for i in range(blocks):
        p = f"blk.{i}."
        ts += [(p + "attn_norm.weight", [hid], 0),
               (p + "attn_q.weight", [hid, hid], GGML_I2_S),
               (p + "attn_k.weight", [hid, kvd], GGML_I2_S),
               (p + "attn_v.weight", [hid, kvd], GGML_I2_S),
               (p + "attn_output.weight", [hid, hid], GGML_I2_S),
               (p + "ffn_norm.weight", [hid], 0),
               (p + "ffn_gate.weight", [hid, ffn], GGML_I2_S),
               (p + "ffn_up.weight", [hid, ffn], GGML_I2_S),
               (p + "ffn_down.weight", [ffn, hid], GGML_I2_S)]
    ts.append(("output_norm.weight", [hid], 0))
    if not i2s:
        ts = [(n, d, 1 if t == GGML_I2_S else t) for n, d, t in ts]
    sizes = {0: 4, 1: 2}
    info, off = [], 0
    for n, d, t in ts:
        ne = _ne(d)
        sz = ne // 4 + 32 if t == GGML_I2_S else ne * sizes[t]
        if spoil == "short" and t == GGML_I2_S and not info[-1:] == []:
            sz = ne // 4 - 32
        info.append((n, d, t, off))
        off += sz + ((-sz) % al)
    kv = [_kv_str("general.architecture", arch),
          _kv_u32("general.alignment", al),
          _kv_u32(f"{arch}.block_count", blocks)]
    head = GGUF_MAGIC + struct.pack("<IQQ", 3, len(info), len(kv)) + b"".join(kv)
    for n, d, t, o in info:
        head += _s(n) + struct.pack("<I", len(d)) + b"".join(
            struct.pack("<Q", x) for x in d) + struct.pack("<IQ", t, o)
    head += b"\0" * ((-len(head)) % al)
    with open(path, "wb") as fh:
        fh.write(head)
        fh.truncate(len(head) + off)
    return len(head) + off


def selftest():
    assert RATE == 0.05 and BIND == "127.0.0.1" and BITNETD == "127.0.0.1:8765"
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert len(PORTS) == 5 and len(ROWS) == 14
    # closed gate: no GGUF on disk honours "only if"
    r, _, _, _ = build(None)
    assert r["verdict"] == "closed" and r["rows"]["on_disk"] == 0
    r, _, _, _ = build("https://example.invalid/model.gguf")
    assert r["verdict"] == "hold" and r["rows"]["no_fetch"] == -1
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "ggml-model-i2_s.gguf")
        synth_gguf(good)
        r, jtr, i2s, text = build(good)
        assert r["verdict"] == "green", r["rows"]
        assert r["info"]["i2s"] == 210 and r["info"]["ternary"] == 210
        assert PARAMS_2B[0] <= r["info"]["params"] <= PARAMS_2B[1]
        assert r["key"] == close_key(GENESIS, jtr + i2s, text)
        sha_good = r["info"]["header_sha"]
        f16 = os.path.join(d, "f16.gguf")
        synth_gguf(f16, i2s=False)
        r2, *_ = build(f16)
        assert r2["rows"]["i2s_share"] == -1 and r2["verdict"] == "hold"
        assert r2["info"]["header_sha"] != sha_good
        llama = os.path.join(d, "llama.gguf")
        synth_gguf(llama, arch="llama")
        assert build(llama)[0]["rows"]["arch"] == -1
        small = os.path.join(d, "small.gguf")
        synth_gguf(small, blocks=4)
        r3 = build(small)[0]["rows"]
        assert r3["params_2b"] == -1 and r3["i2s_count"] == 0 and r3["blocks"] == 1
        short = os.path.join(d, "short.gguf")
        synth_gguf(short, spoil="short")
        r4 = build(short)[0]["rows"]
        assert r4["i2s_bytes"] == -1, r4
        bad = os.path.join(d, "fake.gguf")
        with open(bad, "wb") as fh:
            fh.write(b"PK\x03\x04" + b"\0" * 60)
        assert build(bad)[0]["rows"]["magic"] == -1
        trunc = os.path.join(d, "trunc.gguf")
        with open(good, "rb") as fh:
            blob = fh.read(4096)
        with open(trunc, "wb") as fh:
            fh.write(blob)
        assert build(trunc)[0]["rows"]["header_bounds"] == -1
        assert build(d)[0]["rows"]["on_disk"] == -1
    # JTR1 round trip
    trits = [1, 0, -1, 1, 1, -1, 0]
    g, back = unpack_jtr1(pack_jtr1(trits))
    assert back == trits and abs(g - 5 / 7) < 1e-6
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 1.0, 1) + b"\xc0")
        raise AssertionError("0b11 accepted")
    except ValueError:
        pass
    # I2_S striped block: byte gp holds elems gp, 32+gp, 64+gp, 96+gp
    probe = [0] * 128
    probe[0], probe[32], probe[64], probe[96] = 1, -1, 1, -1
    blk = pack_i2s_striped(probe)
    assert len(blk) == 32 and blk[0] == 0b10001000 and blk[1] == 0b01010101
    long = [((i * 7) % 3) - 1 for i in range(300)]
    assert unpack_i2s_striped(pack_i2s_striped(long), 300) == long
    assert len(pack_i2s_striped(long)) == 96
    # close rule and rate
    want = hashlib.sha256(GENESIS.encode() + b"pk" + b"t").hexdigest()[:16]
    assert close_key(GENESIS, b"pk", "t") == want
    try:
        close_key("XYZ", b"", "t")
        raise AssertionError("bad prev accepted")
    except ValueError:
        pass
    assert verdict([1] * 19 + [0], "open") == "green"     # 5% undecided
    assert verdict([1] * 18 + [0, 0], "open") == "hold"   # 10% undecided
    rep, jtr, i2s, text = build(None)
    return rep["key"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gguf", default=None, help="local GGUF path (header only)")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        key = selftest()
        print(json.dumps({"selftest": "pass", "key": key, "rate": RATE,
                          "ports": list(PORTS), "engines": list(ENGINES)}))
        return 0
    rep, *_ = build(a.gguf, a.prev)
    print(json.dumps(rep, sort_keys=True, indent=2))
    return 0 if rep["verdict"] in ("green", "closed") else 1


if __name__ == "__main__":
    sys.exit(main())
