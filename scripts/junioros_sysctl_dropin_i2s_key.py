"""JuniorOS overlay sysctl drop-in key: loopback integrity, BPF/kexec/fs, staging (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" (Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH), JuniorHome
docs/ROADMAP.md JuniorOS overlay. Quant Chief pair A, slot 2026-10-21 05:05 MT:
overlay remaining gaps only. Trit Pack & I2_S side: the trit pack and I2_S key.
Winsor & Quant Error covers the quant-error bound side of the same item in a
different file.

Gap (new capability). The overlay ships rails/linux/sysctl-junior.conf, but
no installer stages it (install-overlay.sh and debian/stage_overlay.sh never copy
it to etc/sysctl.d), and the only earlier check scores three knobs
(kptr_restrict, dmesg_restrict, yama.ptrace_scope). Nothing scores the knobs
that protect the loopback-only bind contract itself, or the rest of the
host-side hardening a dedicated inference box wants. This file parses the
drop-in with sysctl.d(5) rules (comments # and ;, "-" prefix = ignore errors,
"/" or "." separators, later assignment wins) and scores:
  loop.route_localnet.{all,default,lo}  net.ipv4.conf.*.route_localnet is not 1
        (route_localnet=1 lets 127.0.0.0/8 be routed from other interfaces,
        which breaks 127.0.0.1:8765 being loopback-only); unset counts +1
        because the kernel default is 0
  net.rp_filter / net.syncookies          already in the file: now scored
  net.accept_redirects.{v4,v6}            all.accept_redirects = 0
  net.send_redirects                      ipv4 all.send_redirects = 0
  net.accept_source_route                 ipv4 all.accept_source_route = 0
  bpf.unprivileged                        kernel.unprivileged_bpf_disabled >= 1
  bpf.jit_harden                          net.core.bpf_jit_harden = 2
  kexec.disabled                          kernel.kexec_load_disabled = 1
  perf.paranoid                           kernel.perf_event_paranoid >= 2
  fs.protected_{symlinks,hardlinks}       = 1
  fs.protected_{fifos,regular}            >= 1 (2 preferred)
  syntax.no_duplicates                    no key assigned twice (0 if so)
  syntax.no_ignored                       no "-" prefixed (error-ignored) key
  userns.coherent                         user.max_user_namespaces=0 is not
        active while the overlay stages a rootless OCI bundle (it would break
        rootless engines); commented out counts +1
  order.after_distro                      install target name sorts after
        99-sysctl.conf, so Debian /etc/sysctl.conf cannot override it; 0 if the
        name sorts before (overridable), 0 if no target named
  stage.installed                         an installer text you pass copies the
        drop-in into etc/sysctl.d; -1 if installers given and none do
Unset knobs that are not kernel-default-safe score -1 (the drop-in should pin
them). It reads only local files you point it at (or built-in fixtures). It
never runs sysctl, a subprocess, a socket, or the network, and it does not
wrap, test, or re-derive any other team file.

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
  python junioros_sysctl_dropin_i2s_key.py --selftest
  python junioros_sysctl_dropin_i2s_key.py --sysctl rails/linux/sysctl-junior.conf \
      [--installer rails/linux/install-overlay.sh ...] [--prev HEX16]
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
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("bitnetd 127.0.0.1:8765", "bitnet.cpp I2_S (llama-server)",
           "i2sd userspace", "BitNet-mlx", "JuniorLLM", "JuniorOS overlay")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: host sysctl "
                "drop-in (loopback integrity, BPF/kexec/fs hardening, staging)")
MAGIC = b"JTR1"
GENESIS = "0" * 16
BLOCK = 128
DISTRO_LAST = "99-sysctl.conf"
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}
_TARGET_RE = re.compile(r"/etc/sysctl\.d/([A-Za-z0-9_.@-]+\.conf)")
_STAGE_RE = re.compile(r"\b(cp|install)\b[^\n]*sysctl[^\n]*etc/sysctl\.d")

# (check, key, op, bound, unset_trit)
KNOBS = (
    ("loop.route_localnet.all", "net.ipv4.conf.all.route_localnet", "eq", 0, 1),
    ("loop.route_localnet.default", "net.ipv4.conf.default.route_localnet", "eq", 0, 1),
    ("loop.route_localnet.lo", "net.ipv4.conf.lo.route_localnet", "eq", 0, 1),
    ("net.rp_filter", "net.ipv4.conf.all.rp_filter", "ge", 1, -1),
    ("net.syncookies", "net.ipv4.tcp_syncookies", "eq", 1, -1),
    ("net.accept_redirects.v4", "net.ipv4.conf.all.accept_redirects", "eq", 0, -1),
    ("net.accept_redirects.v6", "net.ipv6.conf.all.accept_redirects", "eq", 0, -1),
    ("net.send_redirects", "net.ipv4.conf.all.send_redirects", "eq", 0, -1),
    ("net.accept_source_route", "net.ipv4.conf.all.accept_source_route", "eq", 0, -1),
    ("bpf.unprivileged", "kernel.unprivileged_bpf_disabled", "ge", 1, -1),
    ("bpf.jit_harden", "net.core.bpf_jit_harden", "eq", 2, -1),
    ("kexec.disabled", "kernel.kexec_load_disabled", "eq", 1, -1),
    ("perf.paranoid", "kernel.perf_event_paranoid", "ge", 2, -1),
    ("fs.protected_symlinks", "fs.protected_symlinks", "eq", 1, -1),
    ("fs.protected_hardlinks", "fs.protected_hardlinks", "eq", 1, -1),
    ("fs.protected_fifos", "fs.protected_fifos", "ge", 1, -1),
    ("fs.protected_regular", "fs.protected_regular", "ge", 1, -1),
)


# ---------------------------------------------------------------- parse
def norm_key(k: str) -> str:
    """sysctl.d(5): if the first separator is '/', '/' and '.' swap."""
    k = k.strip()
    i_dot, i_sl = k.find("."), k.find("/")
    if i_sl != -1 and (i_dot == -1 or i_sl < i_dot):
        k = k.translate(str.maketrans("/.", "./"))
    return k


def parse_sysctl(text: str) -> dict:
    vals, dup, ignored, commented = {}, [], [], {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] in "#;":
            body = line[1:].strip()
            if "=" in body:
                k, v = body.split("=", 1)
                k = norm_key(k.lstrip("-"))
                if re.fullmatch(r"[A-Za-z0-9_.*/-]+", k):
                    commented[k] = v.strip()
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("-"):
            k = k[1:]
            ignored.append(norm_key(k))
        k = norm_key(k)
        if k in vals:
            dup.append(k)
        vals[k] = v.strip()  # later assignment wins
    return {"vals": vals, "dup": dup, "ignored": ignored, "commented": commented}


def _int(v):
    try:
        return int(v.split()[0])
    except (ValueError, IndexError, AttributeError):
        return None


def knob_trit(vals, key, op, bound, unset_trit):
    if key not in vals:
        return unset_trit, key + " unset" + (" (kernel default safe)" if unset_trit == 1
                                             else " (drop-in should pin it)")
    n = _int(vals[key])
    if n is None:
        return 0, key + " unparsed " + vals[key]
    ok = (n == bound) if op == "eq" else (n >= bound)
    return (1 if ok else -1), "%s = %d" % (key, n)


def check_userns(p, stages_oci):
    v = p["vals"].get("user.max_user_namespaces")
    if v is None:
        return 1, "user.max_user_namespaces not active (rootless OCI keeps working)"
    n = _int(v)
    if n == 0 and stages_oci:
        return -1, "user.max_user_namespaces=0 while a rootless OCI bundle is staged"
    if n == 0:
        return 1, "user.max_user_namespaces=0 and no rootless OCI staged"
    return 1, "user.max_user_namespaces=" + v


def check_order(text):
    m = _TARGET_RE.search(text)
    if not m:
        return 0, "no /etc/sysctl.d target named"
    name = m.group(1)
    if name > DISTRO_LAST:
        return 1, name + " sorts after " + DISTRO_LAST
    return 0, name + " sorts before " + DISTRO_LAST + " (distro sysctl.conf can override)"


def check_stage(installers):
    if not installers:
        return 0, "no installer text given"
    for name, txt in installers:
        if _STAGE_RE.search(txt):
            return 1, name + " stages drop-in into etc/sysctl.d"
    return -1, "no installer stages the drop-in (" + ",".join(n for n, _ in installers) + ")"


def contract_trits(sysctl_text: str, installers=()) -> list:
    p = parse_sysctl(sysctl_text)
    stages_oci = any("oci/config.json" in t for _, t in installers)
    rows = []

    def add(name, tv):
        rows.append({"check": name, "trit": tv[0], "why": tv[1]})

    for name, key, op, bound, unset in KNOBS:
        add(name, knob_trit(p["vals"], key, op, bound, unset))
    add("syntax.no_duplicates", (0, "duplicate " + ",".join(sorted(set(p["dup"]))))
        if p["dup"] else (1, "no duplicate keys"))
    add("syntax.no_ignored", (0, "error-ignored " + ",".join(p["ignored"]))
        if p["ignored"] else (1, "no '-' prefixed keys"))
    add("userns.coherent", check_userns(p, stages_oci))
    add("order.after_distro", check_order(sysctl_text))
    add("stage.installed", check_stage(list(installers)))
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
_SHIPPED_LIKE = """# JuniorOS host knobs. Install to /etc/sysctl.d/90-junior.conf after review.
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1
kernel.yama.ptrace_scope = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.tcp_syncookies = 1
# user.max_user_namespaces = 0
"""

_TIGHT = """; tight fixture. Install to /etc/sysctl.d/99-zz-junior.conf
net.ipv4.conf.all.route_localnet = 0
net/ipv4/conf/lo/route_localnet = 0
net.ipv4.conf.all.rp_filter = 1
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
kernel.unprivileged_bpf_disabled = 2
net.core.bpf_jit_harden = 2
kernel.kexec_load_disabled = 1
kernel.perf_event_paranoid = 3
fs.protected_symlinks = 1
fs.protected_hardlinks = 1
fs.protected_fifos = 2
fs.protected_regular = 2
"""

_INSTALL_NO = ("install-overlay.sh",
               'cp "$HERE/oci/config.json" "$DEST/usr/lib/junioros/oci/config.json"\n')
_INSTALL_YES = ("install-overlay.sh",
                'cp "$HERE/oci/config.json" "$DEST/usr/lib/junioros/oci/config.json"\n'
                'cp "$HERE/sysctl-junior.conf" "$DEST/etc/sysctl.d/99-zz-junior.conf"\n')


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

    tight = contract_trits(_TIGHT, [_INSTALL_YES])
    expect(all(r["trit"] == 1 for r in tight), "tight all +1: %r" %
           [r for r in tight if r["trit"] != 1])
    expect(verdict([r["trit"] for r in tight]) == "green", "tight green")

    loose = {r["check"]: r["trit"] for r in contract_trits(_SHIPPED_LIKE, [_INSTALL_NO])}
    for name in ("loop.route_localnet.all", "loop.route_localnet.lo",
                 "net.rp_filter", "net.syncookies", "userns.coherent"):
        expect(loose[name] == 1, "loose " + name + " +1")
    for name in ("net.accept_redirects.v4", "bpf.unprivileged", "bpf.jit_harden",
                 "kexec.disabled", "perf.paranoid", "fs.protected_fifos",
                 "stage.installed"):
        expect(loose[name] == -1, "loose " + name + " -1")
    expect(loose["order.after_distro"] == 0, "90- sorts before 99-sysctl.conf")

    # route_localnet=1 breaks the loopback-only bind contract
    rl = contract_trits(_TIGHT.replace("lo/route_localnet = 0", "lo/route_localnet = 1"))
    expect({r["check"]: r["trit"] for r in rl}["loop.route_localnet.lo"] == -1,
           "route_localnet=1 -1")
    # slash form normalises
    expect(norm_key("net/ipv4/conf/eth0.100/rp_filter") ==
           "net.ipv4.conf.eth0/100.rp_filter", "slash/dot swap")
    # active userns=0 with rootless OCI staged is incoherent
    un = _SHIPPED_LIKE.replace("# user.max_user_namespaces = 0", "user.max_user_namespaces = 0")
    expect(check_userns(parse_sysctl(un), True)[0] == -1, "userns=0 + OCI -1")
    expect(check_userns(parse_sysctl(un), False)[0] == 1, "userns=0 no OCI +1")
    # duplicates and ignored keys are undecided, later assignment wins
    d = parse_sysctl("kernel.perf_event_paranoid = 1\n-kernel.perf_event_paranoid = 3\n")
    expect(d["vals"]["kernel.perf_event_paranoid"] == "3", "later wins")
    expect(d["dup"] and d["ignored"], "dup and ignored seen")
    # no installer evidence -> undecided
    expect(check_stage([])[0] == 0, "no installer 0")

    trits = [r["trit"] for r in contract_trits(_SHIPPED_LIKE, [_INSTALL_NO])]
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
    ap.add_argument("--sysctl", help="local sysctl.d drop-in (read only)")
    ap.add_argument("--installer", action="append", default=[],
                    help="local installer script text (read only, repeatable)")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.sysctl:
        ap.error("--sysctl or --selftest required")
    with open(a.sysctl, "r", encoding="utf-8") as fh:
        text = fh.read()
    installers = []
    for path in a.installer:
        with open(path, "r", encoding="utf-8") as fh:
            installers.append((path.rsplit("/", 1)[-1], fh.read()))
    rep = build_key(contract_trits(text, installers), a.prev)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["verdict"] == "green" else 3


if __name__ == "__main__":
    sys.exit(main())
