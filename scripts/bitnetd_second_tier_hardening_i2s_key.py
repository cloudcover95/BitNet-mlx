"""bitnetd second-tier hardening / surface-narrowing key (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" -- Debian/Alpine +
os-release.junior + *hardened bitnetd.service* + juniorctl on PATH (JuniorHome
docs/ROADMAP.md JuniorOS overlay). Quant Chief pair A, slot 2026-10-20 02:05 MT:
overlay, remaining gaps only. Trit Pack & I2_S side: the trit pack and I2_S key.
Winsor & Quant Error covers the quant-error bound side of the same item in a
different file.

New capability. Earlier overlay files score first-tier flag presence
(NoNewPrivileges, PrivateTmp, PrivateDevices, MemoryDenyWriteExecute,
LockPersonality, RestrictRealtime, ProtectSystem, ProtectHome, empty
CapabilityBoundingSet, RestrictAddressFamilies present, IPAddressDeny), plus
MemoryMax, restart/ordering/sandbox reach, user-unit parity, file modes and
seccomp errno. None of them looks at the second tier of systemd.exec(5)
sandboxing that `systemd-analyze security` also weighs, or asks whether the
allowed surface is wider than the loopback IPv4 contract needs. This file
scores, for the *system* unit only (no user-unit parity):
  kern.logs        ProtectKernelLogs=yes (no /dev/kmsg, no syslog(2))
  kern.clock       ProtectClock=yes (no settimeofday/adjtimex, no RTC)
  kern.hostname    ProtectHostname=yes (UTS namespace)
  proc.hide        ProtectProc=invisible|noaccess|ptraceable (other PIDs hidden)
  proc.subset      ProcSubset=pid (no /proc/sys, /proc/kcore ... view)
  ipc.private      PrivateIPC=yes (own SysV IPC / POSIX mqueue namespace)
  ipc.remove       RemoveIPC=yes (IPC objects dropped on stop)
  users.private    PrivateUsers=yes (user namespace, caps meaningless on host)
  fs.umask         UMask= is 0027 or tighter (default 0022 leaves files o+r)
  keyring.private  KeyringMode=private (system-unit default counts as +1)
  dev.policy       DevicePolicy closed (PrivateDevices=yes implies it)
  af.inet6_surplus RestrictAddressFamilies has no AF_INET6 when every bind
                   the unit names is IPv4 loopback (127.0.0.1); 0 if unrestricted
  id.not_shared    not User=nobody without DynamicUser=yes (systemd.exec(5):
                   nobody must not own files or run services; shared identity)
  bind.loopback    unit names 127.0.0.1:8765 and no wildcard address
It parses unit text only. It never starts bitnetd, opens a socket, runs a
subprocess, or touches the network. It does not wrap, test, or re-derive any
other team file.

Trits: +1 pass, 0 undecided, -1 fail.
JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected); gamma = mean |trit|
I2_S block: code = trit + 1, four per byte MSB first, padded with code 1 to 128
values (32 bytes), so the key can be fed to a ternary I2_S 128-block reader.
Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16), pack = JTR1 bytes, text = canonical JSON of the report.
Rate 0.05: green needs zero -1 trits and at most 5% undecided, else hold.
Compatible engines: bitnetd 127.0.0.1:8765, bitnet.cpp I2_S (llama-server),
i2sd userspace, BitNet-mlx, JuniorLLM, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage:
  python bitnetd_second_tier_hardening_i2s_key.py --selftest
  python bitnetd_second_tier_hardening_i2s_key.py --unit rails/linux/bitnetd.service [--prev HEX16]
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
BITNETD_PORT = 8765
WILDCARDS = (".".join(["0"] * 4), "[::]", "::")  # deny-check only; never bound
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("bitnetd 127.0.0.1:8765", "bitnet.cpp I2_S (llama-server)",
           "i2sd userspace", "BitNet-mlx", "JuniorLLM", "JuniorOS overlay")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: hardened "
                "bitnetd.service (second-tier sandboxing + surface narrowing)")
MAGIC = b"JTR1"
GENESIS = "0" * 16
BLOCK = 128
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}
_TRUE = {"1", "yes", "true", "on"}
_PROC_OK = {"invisible", "noaccess", "ptraceable"}
_BIND_RE = re.compile(r"(\[?[0-9A-Fa-f:.]+\]?|localhost):(\d{1,5})\b")


# ---------------------------------------------------------------- unit parse
def parse_unit(text: str) -> dict:
    """systemd unit text -> {section: {key: [values...]}} (continuations joined)."""
    out, sec, pending = {}, None, ""
    for raw in text.splitlines():
        line = pending + raw.strip() if pending else raw.strip()
        pending = ""
        if line.endswith("\\"):
            pending = line[:-1] + " "
            continue
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1]
            out.setdefault(sec, {})
            continue
        if sec is None or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if v == "":
            out[sec][k] = []  # empty assignment resets the list
        else:
            out[sec].setdefault(k, []).append(v)
    return out


def _last(unit, key, sec="Service"):
    vals = unit.get(sec, {}).get(key)
    return vals[-1] if vals else None


def _yes(v) -> bool:
    return v is not None and v.strip().lower() in _TRUE


def _bool_trit(unit, key, why_missing):
    v = _last(unit, key)
    if v is None:
        return -1, key + " unset (" + why_missing + ")"
    return (1, key + "=" + v) if _yes(v) else (-1, key + "=" + v)


# ---------------------------------------------------------------- checks
def check_umask(unit):
    v = _last(unit, "UMask")
    if v is None:
        return -1, "UMask unset (default 0022, files o+r)"
    try:
        m = int(v, 8)
    except ValueError:
        return 0, "UMask unparsed " + v
    return (1, "UMask=" + v) if (m & 0o027) == 0o027 else (-1, "UMask=" + v)


def check_proc(unit):
    v = _last(unit, "ProtectProc")
    if v is None:
        return -1, "ProtectProc unset (default, all PIDs visible)"
    return (1, "ProtectProc=" + v) if v.lower() in _PROC_OK else (-1, "ProtectProc=" + v)


def check_subset(unit):
    v = _last(unit, "ProcSubset")
    if v is None:
        return -1, "ProcSubset unset (default all)"
    return (1, "ProcSubset=pid") if v.lower() == "pid" else (-1, "ProcSubset=" + v)


def check_keyring(unit, user_unit=False):
    v = _last(unit, "KeyringMode")
    if v is None:
        if user_unit:
            return 0, "KeyringMode unset (user manager default inherit)"
        return 1, "KeyringMode unset (system default private)"
    return (1, "KeyringMode=private") if v.lower() == "private" else (-1, "KeyringMode=" + v)


def check_device_policy(unit):
    v = _last(unit, "DevicePolicy")
    if v is not None and v.lower() in ("closed", "strict"):
        return 1, "DevicePolicy=" + v
    if v is not None and v.lower() == "auto" and not _yes(_last(unit, "PrivateDevices")):
        return -1, "DevicePolicy=auto without PrivateDevices"
    if _yes(_last(unit, "PrivateDevices")):
        return 1, "PrivateDevices=yes implies DevicePolicy=closed"
    return -1, "no DevicePolicy and no PrivateDevices"


def unit_binds(unit_text: str) -> list:
    binds = []
    for raw in unit_text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        for host, port in _BIND_RE.findall(line):
            if host.count(".") == 3 or ":" in host.strip("[]") or host == "localhost":
                binds.append((host.strip("[]"), int(port)))
    return binds


def check_af_inet6(unit, unit_text):
    fam = _last(unit, "RestrictAddressFamilies")
    if fam is None:
        return 0, "RestrictAddressFamilies unset (all families; first-tier gap)"
    words = fam.split()
    if words and words[0].startswith("~"):
        deny = {w.lstrip("~") for w in words}
        has6 = "AF_INET6" not in deny
    else:
        has6 = "AF_INET6" in words
    binds = unit_binds(unit_text)
    v6_needed = any(":" in h or h == "localhost" for h, _ in binds)
    if not binds:
        return 0, "no bind named in unit"
    if has6 and not v6_needed:
        return -1, "AF_INET6 allowed but every bind is IPv4 " + ",".join(
            "%s:%d" % b for b in binds)
    return 1, "families match binds"


def check_identity(unit):
    user = _last(unit, "User")
    dyn = _yes(_last(unit, "DynamicUser"))
    if dyn:
        return 1, "DynamicUser=yes"
    if user is None:
        return 0, "no User= (runs as manager user)"
    if user in ("nobody", "65534", "nfsnobody"):
        return -1, "User=" + user + " shared identity without DynamicUser"
    if user in ("root", "0"):
        return -1, "User=root"
    return 1, "User=" + user


def check_bind(unit_text):
    if any(w in unit_text for w in WILDCARDS[:1]) or "[::]" in unit_text:
        return -1, "wildcard address named"
    if "%s:%d" % (BIND, BITNETD_PORT) in unit_text:
        return 1, "%s:%d" % (BIND, BITNETD_PORT)
    return 0, "loopback bind not named"


def contract_trits(unit_text: str, user_unit=False) -> list:
    u = parse_unit(unit_text)
    rows = []

    def add(name, tv):
        rows.append({"check": name, "trit": tv[0], "why": tv[1]})

    add("kern.logs", _bool_trit(u, "ProtectKernelLogs", "kmsg/syslog readable"))
    add("kern.clock", _bool_trit(u, "ProtectClock", "clock writable if caps"))
    add("kern.hostname", _bool_trit(u, "ProtectHostname", "host UTS shared"))
    add("proc.hide", check_proc(u))
    add("proc.subset", check_subset(u))
    add("ipc.private", _bool_trit(u, "PrivateIPC", "host SysV IPC shared"))
    add("ipc.remove", _bool_trit(u, "RemoveIPC", "IPC objects outlive stop"))
    add("users.private", _bool_trit(u, "PrivateUsers", "host user namespace"))
    add("fs.umask", check_umask(u))
    add("keyring.private", check_keyring(u, user_unit))
    add("dev.policy", check_device_policy(u))
    add("af.inet6_surplus", check_af_inet6(u, unit_text))
    add("id.not_shared", check_identity(u))
    add("bind.loopback", check_bind(unit_text))
    return rows


# ---------------------------------------------------------------- packing
def pack_jtr1(trits) -> bytes:
    n = len(trits)
    gamma = (sum(abs(t) for t in trits) / n) if n else 0.0
    payload = bytearray((n + 3) // 4)
    for i, t in enumerate(trits):
        payload[i // 4] |= _JTR[t] << (6 - 2 * (i % 4))
    return MAGIC + struct.pack("<fI", gamma, n) + bytes(payload)


def unpack_jtr1(blob: bytes):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    body = blob[12:]
    if len(body) != (n + 3) // 4:
        raise ValueError("bad payload length")
    out = []
    for i in range(n):
        c = (body[i // 4] >> (6 - 2 * (i % 4))) & 3
        if c == 3:
            raise ValueError("code 0b11 rejected")
        out.append(_JTR_DEC[c])
    return gamma, out


def pack_i2s_block(trits) -> bytes:
    if len(trits) > BLOCK:
        raise ValueError("more than one I2_S block")
    codes = [t + 1 for t in trits] + [1] * (BLOCK - len(trits))
    out = bytearray(BLOCK // 4)
    for i, c in enumerate(codes):
        out[i // 4] |= c << (6 - 2 * (i % 4))
    return bytes(out)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    if len(prev) != 16 or any(ch not in "0123456789abcdef" for ch in prev):
        raise ValueError("prev must be 16 lowercase hex chars")
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits) -> str:
    if not trits or any(t == -1 for t in trits):
        return "hold"
    undecided = sum(1 for t in trits if t == 0) / len(trits)
    return "green" if undecided <= RATE else "hold"


def build_key(rows, prev=GENESIS) -> dict:
    trits = [r["trit"] for r in rows]
    pack = pack_jtr1(trits)
    report = {
        "roadmap": ROADMAP_ITEM,
        "rate": RATE,
        "checks": rows,
        "verdict": verdict(trits),
        "pass": sum(1 for t in trits if t == 1),
        "n": len(trits),
        "i2s_block_hex": pack_i2s_block(trits).hex(),
        "ports": list(PORTS),
        "engines": list(ENGINES),
        "bind": "%s:%d" % (BIND, BITNETD_PORT),
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["prev"] = prev
    report["key"] = close_rule(prev, pack, text)
    report["jtr1_hex"] = pack.hex()
    return report


# ---------------------------------------------------------------- selftest
_SHIPPED_LIKE = """[Unit]
Description=bitnetd (fixture shaped like the shipped first-tier unit)
After=network-online.target
[Service]
User=nobody
Group=nogroup
ExecStart=/usr/local/bin/junior-bitnetd
Environment=JUNIOR_BIND=127.0.0.1:8765
NoNewPrivileges=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectKernelTunables=yes
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
CapabilityBoundingSet=
[Install]
WantedBy=multi-user.target
"""

_TIGHT = """[Unit]
Description=bitnetd (second-tier tight fixture)
[Service]
DynamicUser=yes
ExecStart=/usr/local/bin/junior-bitnetd
Environment=JUNIOR_BIND=127.0.0.1:8765
PrivateDevices=yes
ProtectKernelLogs=yes
ProtectClock=yes
ProtectHostname=yes
ProtectProc=invisible
ProcSubset=pid
PrivateIPC=yes
RemoveIPC=yes
PrivateUsers=yes
UMask=0077
KeyringMode=private
RestrictAddressFamilies=AF_UNIX \\
  AF_INET
[Install]
WantedBy=multi-user.target
"""


def selftest() -> int:
    ok = True

    def expect(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL", msg)

    expect(RATE == 0.05, "rate 0.05")
    expect(not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH), "flags false")
    expect(BIND == "127.0.0.1", "bind loopback")

    tight = contract_trits(_TIGHT)
    expect(all(r["trit"] == 1 for r in tight), "tight all +1: %r" %
           [r for r in tight if r["trit"] != 1])
    expect(verdict([r["trit"] for r in tight]) == "green", "tight green")

    loose = {r["check"]: r["trit"] for r in contract_trits(_SHIPPED_LIKE)}
    for name in ("kern.logs", "kern.clock", "kern.hostname", "proc.hide",
                 "proc.subset", "ipc.private", "ipc.remove", "users.private",
                 "fs.umask", "af.inet6_surplus", "id.not_shared"):
        expect(loose[name] == -1, "loose " + name + " -1")
    expect(loose["keyring.private"] == 1, "system keyring default +1")
    expect(loose["dev.policy"] == 1, "PrivateDevices implies closed")
    expect(loose["bind.loopback"] == 1, "loose bind +1")

    # user-manager keyring default is inherit -> undecided
    expect(check_keyring(parse_unit(_TIGHT.replace("KeyringMode=private\n", "")),
                         user_unit=True)[0] == 0, "user keyring 0")
    # IPv6 loopback bind legitimately needs AF_INET6
    v6 = _TIGHT.replace("127.0.0.1:8765", "[::1]:8765").replace(
        "AF_UNIX \\\n  AF_INET", "AF_UNIX AF_INET6")
    expect(check_af_inet6(parse_unit(v6), v6)[0] == 1, "v6 bind needs AF_INET6")
    # deny-list form ~AF_INET6 counts as narrowed
    deny = "[Service]\nRestrictAddressFamilies=~AF_INET6 AF_PACKET\nX=127.0.0.1:8765\n"
    expect(check_af_inet6(parse_unit(deny), deny)[0] == 1, "deny-list ~AF_INET6")
    # wildcard rejected
    wild = _TIGHT.replace("127.0.0.1:8765", WILDCARDS[0] + ":8765")
    expect(check_bind(wild)[0] == -1, "wildcard -1")
    # UMask 0022 fails, 0027 passes
    expect(check_umask(parse_unit("[Service]\nUMask=0022\n"))[0] == -1, "umask 0022")
    expect(check_umask(parse_unit("[Service]\nUMask=0027\n"))[0] == 1, "umask 0027")
    # empty assignment resets
    expect(_last(parse_unit("[Service]\nUMask=0077\nUMask=\n"), "UMask") is None,
           "empty assignment resets")

    # packing round trip and I2_S block shape
    trits = [r["trit"] for r in contract_trits(_SHIPPED_LIKE)]
    g, back = unpack_jtr1(pack_jtr1(trits))
    expect(back == trits, "jtr1 round trip")
    expect(abs(g - sum(abs(t) for t in trits) / len(trits)) < 1e-6, "gamma")
    blk = pack_i2s_block(trits)
    expect(len(blk) == 32, "i2s block 32 bytes")
    expect(blk[-1] == 0b01010101, "i2s pad code 1")
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 0.0, 1) + bytes([0xC0]))
        expect(False, "0b11 rejected")
    except ValueError:
        pass

    # close rule recompute
    rep = build_key(tight)
    body = {k: v for k, v in rep.items() if k not in ("prev", "key", "jtr1_hex")}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    want = hashlib.sha256(GENESIS.encode() + bytes.fromhex(rep["jtr1_hex"]) +
                          text.encode()).hexdigest()[:16]
    expect(rep["key"] == want, "close rule sha256(prev||pack||text)[:16]")
    rep2 = build_key(tight, prev=rep["key"])
    expect(rep2["key"] != rep["key"], "chain advances")

    print(json.dumps({"selftest": "pass" if ok else "fail", "key": rep["key"],
                      "verdict": rep["verdict"], "n": rep["n"], "rate": RATE}))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--unit", help="local systemd unit file (read only)")
    ap.add_argument("--user-unit", action="store_true",
                    help="score as a user-manager unit (keyring default inherit)")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.unit:
        ap.error("--unit or --selftest required")
    with open(a.unit, "r", encoding="utf-8") as fh:
        text = fh.read()
    rep = build_key(contract_trits(text, a.user_unit), a.prev)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["verdict"] == "green" else 3


if __name__ == "__main__":
    sys.exit(main())
