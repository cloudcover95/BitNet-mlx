"""bitnetd restart / ordering / sandbox-reach key (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" -- Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH; "I2_S 2B4T
only if GGUF already on disk" (JuniorHome docs/ROADMAP.md JuniorOS overlay).
Quant Chief pair A, slot 2026-10-19 14:05 MT: overlay, remaining gaps only.
Trit Pack & I2_S side: the trit pack and I2_S key. Winsor & Quant Error covers
the quant-error bound side of the same item in a different file.

New capability (gaps no team file covers yet). Earlier overlay files score
os-release, PATH, hardening-flag presence, non-root / DynamicUser,
IPAddressAllow loopback, MemoryMax vs pack size, user-unit parity and the GGUF
header gate. None of them asks whether the hardened unit, as written, can
actually *run* its I2_S path and fail the right way. This file checks:
  * refusal exit codes: every non-zero `exit N` in the ExecStart wrapper
    (rails/linux/bitnetd.sh refuses a non-loopback bind with exit 3) against
    Restart= / RestartPreventExitStatus= -- with Restart=on-failure and no
    prevent list, a deliberate refusal turns into a restart burst that ends
    in start-limit-hit instead of a clean terminal failure,
  * boot ordering: After=network-online.target without Wants=/Requires= on it
    is ordering-only and never pulls the target in; a loopback-only service
    needs no network-online dependency at all,
  * sandbox reach of the I2_S branch: whether the unit configures
    JUNIOR_BITNET_CPP / JUNIOR_GGUF at all, and whether a candidate GGUF path
    and bitnet.cpp binary path would be visible / executable inside the unit's
    namespace (ProtectHome, PrivateTmp, PrivateDevices, ProtectSystem,
    InaccessiblePaths, NoExecPaths, TemporaryFileSystem),
  * file modes for User=nobody (os.stat only, only if the path is already
    local; never opened, never fetched),
  * seccomp deny-list action: SystemCallFilter=~@resources with no
    SystemCallErrorNumber= means a denied call (sched_setaffinity, mbind,
    set_mempolicy, setpriority) kills the daemon with SIGSYS rather than
    returning EPERM.
It never starts bitnetd, opens a socket, runs a subprocess, or touches the
network. It does not wrap, test, or re-derive any other team file.

Trits (+1 pass, 0 undecided, -1 fail):
  restart.refusal_terminal  wrapper refusal codes are not restarted
  order.online_pulled       no ordering-only network-online dependency
  order.install_target      WantedBy=multi-user.target (system unit)
  reach.i2s_configured      unit Environment names JUNIOR_BITNET_CPP + JUNIOR_GGUF
  reach.gguf_visible        candidate GGUF path visible inside the sandbox
  reach.cpp_executable      candidate bitnet.cpp path visible and not noexec
  reach.mode_nobody         local files readable (and binary executable) by other
  syscall.deny_errno        denied syscalls return an errno instead of SIGSYS
  bind.loopback             unit names 127.0.0.1 and 8765, no wildcard

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected); gamma = mean |trit|
I2_S block: code = trit + 1, four per byte MSB first, padded with code 1 to 128.
Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16), pack = JTR1 bytes, text = canonical JSON of the report.
Rate 0.05: green needs zero -1 trits and at most 5% undecided, else hold.
Compatible engines: bitnetd 127.0.0.1:8765, bitnet.cpp I2_S (llama-server),
i2sd userspace, BitNet-mlx, JuniorLLM, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage:
  python3 scripts/bitnetd_restart_reach_i2s_key.py --selftest
  python3 scripts/bitnetd_restart_reach_i2s_key.py --unit bitnetd.service
      [--wrapper bitnetd.sh] [--gguf PATH] [--cpp PATH] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
ENGINES = ("bitnetd 127.0.0.1:8765", "bitnet.cpp I2_S (llama-server)",
           "i2sd userspace", "BitNet-mlx", "JuniorLLM", "JuniorOS overlay")
ROADMAP_ITEM = (
    "JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: hardened bitnetd.service "
    "restart/ordering/sandbox reach of the I2_S 2B4T branch"
)
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}
_RESTART_ON_EXIT = {"on-failure", "always", "on-success"}
_HOME_DIRS = ("/home", "/root", "/run/user")
_TMP_DIRS = ("/tmp", "/var/tmp")
_EXIT = re.compile(r"\bexit\s+(\d+)\b")
_SIGNAL_NAMES = re.compile(r"^SIG[A-Z0-9]+$")


# ---------------------------------------------------------------- parsing
def parse_unit(text: str) -> dict:
    """{(section, key): [values...]} in file order. Empty value resets a list."""
    out: dict = {}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" in line and section:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if v == "":
                out[(section, k)] = [""]
            else:
                out.setdefault((section, k), []).append(v)
    return out


def _vals(unit, section, key):
    return [v for v in unit.get((section, key), []) if v != ""]


def _last(unit, section, key):
    vals = unit.get((section, key))
    return vals[-1] if vals else None


def _words(unit, section, key):
    return " ".join(_vals(unit, section, key)).split()


def _yes(v):
    return v is not None and v.lower() in ("yes", "true", "1", "on")


def unit_env(unit) -> dict:
    env = {}
    for line in _vals(unit, "Service", "Environment"):
        for tok in re.findall(r'"[^"]*"|\S+', line):
            tok = tok.strip('"')
            if "=" in tok:
                k, v = tok.split("=", 1)
                env[k] = v
    return env


def wrapper_refusal_codes(text: str) -> list:
    return sorted({int(n) for n in _EXIT.findall(text or "") if int(n) != 0})


def _under(path, prefixes):
    p = os.path.normpath(path)
    for pre in prefixes:
        pre = os.path.normpath(pre.lstrip("-+"))
        if p == pre or p.startswith(pre.rstrip("/") + "/"):
            return pre
    return None


# ---------------------------------------------------------------- checks
def check_restart(unit, wrapper_text):
    codes = wrapper_refusal_codes(wrapper_text)
    restart = (_last(unit, "Service", "Restart") or "no").lower()
    prevent = set()
    for tok in _words(unit, "Service", "RestartPreventExitStatus"):
        if tok.isdigit():
            prevent.add(int(tok))
    if not codes:
        return 0, "no wrapper refusal codes known"
    if restart not in _RESTART_ON_EXIT:
        return 1, "Restart=%s does not restart on exit codes" % restart
    looped = [c for c in codes if c not in prevent]
    if looped:
        return -1, "Restart=%s restarts refusal exit %s (no RestartPreventExitStatus)" % (
            restart, ",".join(map(str, looped)))
    return 1, "refusal exit %s prevented" % ",".join(map(str, codes))


def check_order(unit):
    after = set(_words(unit, "Unit", "After"))
    pulled = set(_words(unit, "Unit", "Wants")) | set(_words(unit, "Unit", "Requires"))
    t = "network-online.target"
    if t in after and t not in pulled:
        return -1, "After=%s without Wants=: ordering-only, never pulled in" % t
    if t in pulled:
        return 0, "pulls %s; loopback-only service does not need it" % t
    return 1, "no network-online dependency (loopback only)"


def check_install(unit):
    wb = _words(unit, "Install", "WantedBy")
    if "multi-user.target" in wb:
        return 1, "WantedBy=multi-user.target"
    return (-1, "no WantedBy") if not wb else (0, "WantedBy=" + " ".join(wb))


def check_configured(env):
    have = [k for k in ("JUNIOR_BITNET_CPP", "JUNIOR_GGUF") if env.get(k)]
    if len(have) == 2:
        return 1, "unit names JUNIOR_BITNET_CPP and JUNIOR_GGUF"
    return 0, "I2_S branch not configured in unit (falls back to i2sd)"


def sandbox_visible(unit, path, need_exec=False):
    """(trit, why) for whether `path` is reachable inside the unit's namespace."""
    if not path:
        return 0, "no candidate path"
    if not path.startswith("/"):
        return -1, "relative path " + path
    ph = (_last(unit, "Service", "ProtectHome") or "no").lower()
    hit = _under(path, _HOME_DIRS)
    if hit:
        if _yes(ph) or ph == "tmpfs":
            return -1, "%s hidden by ProtectHome=%s" % (hit, ph)
    if _under(path, _TMP_DIRS) and _yes(_last(unit, "Service", "PrivateTmp")):
        return -1, "private /tmp hides " + path
    if _under(path, ("/dev",)) and _yes(_last(unit, "Service", "PrivateDevices")):
        return -1, "PrivateDevices hides " + path
    inacc = _words(unit, "Service", "InaccessiblePaths")
    tmpfs = [w.split(":", 1)[0] for w in _words(unit, "Service", "TemporaryFileSystem")]
    hit = _under(path, inacc) or _under(path, tmpfs)
    if hit:
        return -1, "%s masked" % hit
    if need_exec:
        noexec = _under(path, _words(unit, "Service", "NoExecPaths"))
        execok = _under(path, _words(unit, "Service", "ExecPaths"))
        if noexec and not execok:
            return -1, "%s is NoExecPaths" % noexec
    return 1, "visible" + (" and executable" if need_exec else "")


def check_mode(unit, gguf, cpp):
    user = _last(unit, "Service", "User")
    if _yes(_last(unit, "Service", "DynamicUser")) or user in (None, "root", "0"):
        return 0, "mode check applies to a fixed unprivileged User"
    seen = []
    for path, bits in ((gguf, 0o004), (cpp, 0o005)):
        if not path:
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        if (st.st_mode & bits) != bits:
            return -1, "%s not readable%s by %s" % (
                path, "/executable" if bits & 1 else "", user)
        seen.append(path)
    if not seen:
        return 0, "no local file to stat (not fetched)"
    return 1, "other-readable for User=%s" % user


def check_seccomp(unit):
    filters = _vals(unit, "Service", "SystemCallFilter")
    denied = set()
    for f in filters:
        if f.startswith("~"):
            denied.update(f[1:].split())
    errno = _last(unit, "Service", "SystemCallErrorNumber")
    if not denied:
        return 1, "no deny list"
    if errno and errno.lower() != "kill" and not _SIGNAL_NAMES.match(errno):
        return 1, "denied calls return %s" % errno
    if "@resources" in denied:
        return -1, ("~@resources with no SystemCallErrorNumber: sched_setaffinity/"
                    "mbind/set_mempolicy/setpriority kill bitnetd with SIGSYS")
    return 0, "deny list kills on hit; no affinity group denied"


def check_bind(unit_text, env):
    blob = unit_text
    if WILDCARD in blob or "[::]" in blob:
        return -1, "wildcard bind named"
    b = env.get("JUNIOR_BIND", "")
    if b == "%s:%d" % (BIND, BITNETD_PORT) or (BIND in blob and str(BITNETD_PORT) in blob):
        return 1, "%s:%d" % (BIND, BITNETD_PORT)
    return -1, "loopback bind not stated"


def contract_trits(unit_text, wrapper_text="", gguf=None, cpp=None):
    unit = parse_unit(unit_text)
    env = unit_env(unit)
    gguf = gguf or env.get("JUNIOR_GGUF")
    cpp = cpp or env.get("JUNIOR_BITNET_CPP")
    rows = [
        ("restart.refusal_terminal",) + check_restart(unit, wrapper_text),
        ("order.online_pulled",) + check_order(unit),
        ("order.install_target",) + check_install(unit),
        ("reach.i2s_configured",) + check_configured(env),
        ("reach.gguf_visible",) + sandbox_visible(unit, gguf),
        ("reach.cpp_executable",) + sandbox_visible(unit, cpp, need_exec=True),
        ("reach.mode_nobody",) + check_mode(unit, gguf, cpp),
        ("syscall.deny_errno",) + check_seccomp(unit),
        ("bind.loopback",) + check_bind(unit_text, env),
    ]
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
        raise ValueError("bad JTR1 magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    body = blob[12:]
    if len(body) != (n + 3) // 4:
        raise ValueError("bad JTR1 length")
    out = []
    for i in range(n):
        code = (body[i // 4] >> (6 - 2 * (i % 4))) & 3
        if code == 3:
            raise ValueError("JTR1 code 0b11 rejected")
        out.append(_JTR_DEC[code])
    return gamma, out


def pack_i2s_block(trits) -> bytes:
    if len(trits) > I2S_BLOCK:
        raise ValueError("more than one I2_S block")
    codes = [t + 1 for t in trits] + [1] * (I2S_BLOCK - len(trits))
    out = bytearray(I2S_BLOCK // 4)
    for i, c in enumerate(codes):
        out[i // 4] |= c << (6 - 2 * (i % 4))
    return bytes(out)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits) -> str:
    n = len(trits) or 1
    if any(t == -1 for t in trits):
        return "hold"
    return "green" if sum(1 for t in trits if t == 0) / n <= RATE else "hold"


def build_key(rows, prev=GENESIS) -> dict:
    trits = [r[1] for r in rows]
    frame = pack_jtr1(trits)
    block = pack_i2s_block(trits)
    report = {
        "item": ROADMAP_ITEM, "rate": RATE, "verdict": verdict(trits),
        "n": len(trits), "plus": trits.count(1), "zero": trits.count(0),
        "minus": trits.count(-1),
        "checks": [{"check": c, "trit": t, "why": w} for c, t, w in rows],
        "bind": "%s:%d" % (BIND, BITNETD_PORT), "ports": list(PORTS),
        "engines": list(ENGINES),
        "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                  "fetch": FETCH, "ue5_launch": UE5_LAUNCH},
    }
    text = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["key"] = close_rule(prev, frame, text)
    report["prev"] = prev
    report["jtr1_hex"] = frame.hex()
    report["i2s_hex"] = block.hex()
    return report


# ---------------------------------------------------------------- selftest
_WRAPPER = """#!/bin/sh
case "$host" in
  127.0.0.1|localhost|::1) ;;
  *) echo "refusing non-loopback $BIND" >&2; exit 3 ;;
esac
"""

_AS_SHIPPED = """[Unit]
Description=Junior BitNet local inference (loopback)
After=network-online.target

[Service]
Type=simple
User=nobody
Group=nogroup
ExecStart=/usr/local/bin/junior-bitnetd
Restart=on-failure
Environment=JUNIOR_OFFLINE=1
Environment=JUNIOR_BIND=127.0.0.1:8765
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
SystemCallArchitectures=native
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @keyring @resources @mount @reboot @swap @raw-io @module @debug
CapabilityBoundingSet=
MemoryMax=2G

[Install]
WantedBy=multi-user.target
"""

_FIXED = """[Unit]
Description=Junior BitNet local inference (loopback)

[Service]
Type=simple
DynamicUser=yes
ExecStart=/usr/local/bin/junior-bitnetd
Restart=on-failure
RestartPreventExitStatus=3
Environment=JUNIOR_OFFLINE=1 JUNIOR_BIND=127.0.0.1:8765
Environment=JUNIOR_BITNET_CPP=/usr/lib/junioros/bin/llama-server
Environment=JUNIOR_GGUF=/var/lib/junioros/models/ggml-model-i2_s.gguf
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
NoExecPaths=/var/lib
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @resources @mount
SystemCallErrorNumber=EPERM

[Install]
WantedBy=multi-user.target
"""


def selftest() -> int:
    ok = True

    def expect(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL:", msg)

    expect(RATE == 0.05, "rate 0.05")
    expect(not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH), "flags false")
    expect(wrapper_refusal_codes(_WRAPPER) == [3], "wrapper exit 3 found")

    shipped = dict((c, t) for c, t, _ in contract_trits(
        _AS_SHIPPED, _WRAPPER, gguf="/home/nick/models/2b4t.gguf",
        cpp="/tmp/llama-server"))
    expect(shipped["restart.refusal_terminal"] == -1, "shipped restarts exit 3")
    expect(shipped["order.online_pulled"] == -1, "shipped ordering-only online")
    expect(shipped["order.install_target"] == 1, "multi-user install")
    expect(shipped["reach.i2s_configured"] == 0, "shipped I2_S not configured")
    expect(shipped["reach.gguf_visible"] == -1, "/home GGUF hidden")
    expect(shipped["reach.cpp_executable"] == -1, "/tmp binary hidden")
    expect(shipped["syscall.deny_errno"] == -1, "SIGSYS on @resources")
    expect(shipped["bind.loopback"] == 1, "shipped loopback")

    fixed_rows = contract_trits(_FIXED, _WRAPPER)
    fixed = dict((c, t) for c, t, _ in fixed_rows)
    expect(fixed["restart.refusal_terminal"] == 1, "fixed prevents exit 3")
    expect(fixed["order.online_pulled"] == 1, "fixed no online dep")
    expect(fixed["reach.i2s_configured"] == 1, "fixed configured")
    expect(fixed["reach.gguf_visible"] == 1, "/var/lib GGUF visible (read)")
    expect(fixed["reach.cpp_executable"] == 1, "/usr/lib binary executable")
    expect(fixed["syscall.deny_errno"] == 1, "EPERM errno")
    expect(fixed["reach.mode_nobody"] == 0, "DynamicUser mode n/a")
    noexec = sandbox_visible(parse_unit(_FIXED),
                             "/var/lib/junioros/bin/llama-server", need_exec=True)
    expect(noexec[0] == -1, "NoExecPaths blocks exec")
    ro = sandbox_visible(parse_unit(_AS_SHIPPED.replace("ProtectHome=yes",
                                                        "ProtectHome=read-only")),
                         "/home/nick/m.gguf")
    expect(ro[0] == 1, "read-only home stays readable")

    wild = contract_trits(_FIXED.replace("127.0.0.1:8765", WILDCARD + ":8765"), _WRAPPER)
    expect(dict((c, t) for c, t, _ in wild)["bind.loopback"] == -1, "wildcard refused")

    trits = [-1, 0, 1, 1, 0, -1, 1]
    g, back = unpack_jtr1(pack_jtr1(trits))
    expect(back == trits, "JTR1 roundtrip")
    expect(abs(g - 5 / 7) < 1e-6, "gamma mean |trit|")
    expect(pack_jtr1(trits)[:4] == MAGIC and len(pack_jtr1(trits)) == 12 + 2, "JTR1 size")
    blk = pack_i2s_block(trits)
    expect(len(blk) == 32 and blk[0] == 0b00011010, "I2_S block layout")
    expect(blk[-1] == 0b01010101, "I2_S pad code 1")
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 1.0, 1) + bytes([0xC0]))
        expect(False, "0b11 rejected")
    except ValueError:
        pass

    k_shipped = build_key(contract_trits(_AS_SHIPPED, _WRAPPER))
    k_fixed = build_key(fixed_rows)
    expect(k_shipped["verdict"] == "hold", "shipped holds")
    expect(k_fixed["verdict"] == "hold", "fixed holds on one 0 of 9 (> 5%)")
    expect(build_key(fixed_rows)["key"] == k_fixed["key"], "key deterministic")
    expect(build_key(fixed_rows, prev=k_shipped["key"])["key"] != k_fixed["key"],
           "prev chains")
    manual = close_rule(GENESIS, bytes.fromhex(k_fixed["jtr1_hex"]), json.dumps(
        {k: v for k, v in k_fixed.items() if k not in ("key", "prev", "jtr1_hex", "i2s_hex")},
        sort_keys=True, separators=(",", ":")))
    expect(manual == k_fixed["key"] and len(manual) == 16, "close rule sha256[:16]")
    chain = close_rule(k_shipped["key"], bytes.fromhex(k_fixed["jtr1_hex"]), k_fixed["key"])

    print(json.dumps({"selftest": "pass" if ok else "fail",
                      "shipped_key": k_shipped["key"], "fixed_key": k_fixed["key"],
                      "key": chain, "rate": RATE}, sort_keys=True))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--unit")
    ap.add_argument("--wrapper")
    ap.add_argument("--gguf")
    ap.add_argument("--cpp")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.unit:
        ap.error("--unit or --selftest required")
    if not re.fullmatch(r"[0-9a-f]{16}", a.prev):
        ap.error("--prev must be 16 hex chars")
    with open(a.unit, encoding="utf-8") as fh:
        unit_text = fh.read()
    wrapper_text = ""
    if a.wrapper:
        with open(a.wrapper, encoding="utf-8") as fh:
            wrapper_text = fh.read()
    print(json.dumps(build_key(contract_trits(unit_text, wrapper_text, a.gguf, a.cpp),
                               a.prev), sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
