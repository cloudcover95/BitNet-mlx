"""JuniorOS overlay kconfig.junior closure key: kernel options the overlay's own
consumers demand vs what the fragment lists (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" (Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH), JuniorHome
docs/ROADMAP.md JuniorOS overlay. Quant Chief pair A, slot 2026-10-21 14:05 MT:
overlay remaining gaps only. Trit Pack & I2_S side: the trit pack and I2_S key.
Winsor & Quant Error covers the quant-error bound side of the same item in a
different file.

Gap (new capability). The overlay stages rails/linux/kconfig.junior (copied to
usr/lib/junioros/kconfig.junior by debian/stage_overlay.sh) as the fragment to
merge against a vendor kernel, but nothing checks that the fragment closes over
the kernel features the overlay's own units and bundle need. Each consumer
directive maps to the CONFIG_ symbols it depends on:

  sym                 consumers (why)
  NAMESPACES          unit ProtectSystem/PrivateTmp (mount ns); OCI namespaces
  USER_NS             OCI "user" namespace (rootless bundle)
  PID_NS              OCI "pid" namespace
  NET_NS              OCI "network" namespace
  IPC_NS              OCI "ipc" namespace (runtime create fails without it)
  UTS_NS              OCI "uts" namespace + "hostname": junior-bitnetd
  SECCOMP             unit SystemCallFilter / OCI seccomp-bitnetd.json
  SECCOMP_FILTER      SystemCallFilter, MemoryDenyWriteExecute,
                      RestrictAddressFamilies, RestrictNamespaces,
                      LockPersonality, RestrictRealtime, RestrictSUIDSGID,
                      SystemCallArchitectures (all seccomp-bpf in systemd)
  CGROUPS             unit MemoryMax/TasksMax; OCI linux.resources
  MEMCG               MemoryMax=2G; OCI memory.limit
  CGROUP_PIDS         TasksMax=256; OCI pids.limit (silently unenforced
                      without it: a fork bomb cap that is not there)
  CGROUP_BPF          PrivateDevices=yes implies DevicePolicy=closed, which on
                      the unified cgroup hierarchy is a cgroup BPF program
  BPF_SYSCALL         same device policy program must be loaded
  PROC_FS             OCI /proc mount; unit ProtectKernelTunables remounts
  TMPFS               OCI /dev mount type tmpfs
  INET                bind 127.0.0.1:8765 (AF_INET in RestrictAddressFamilies)
  UNIX                AF_UNIX in RestrictAddressFamilies (journald stdout)
  IPV6                conditional: only if JUNIOR_BIND=::1 (installer and
                      wrapper accept it); default bind is IPv4 -> 0 if absent

Trit per symbol: +1 listed =y in the fragment; -1 demanded by a present
consumer and absent (or =n / "is not set" / =m where a namespace or seccomp
cannot be modular); 0 conditional demand and absent, or listed with no
consumer present (surplus). Three fragment trits: syntax.lines (every line is
CONFIG_X=y|m|n|"..."|number, "# CONFIG_X is not set", comment or blank),
syntax.no_duplicates, and frag.no_kernel_blob (no vmlinuz/bzImage/Image path).
Consumers are detected from the texts you pass (unit, OCI config.json,
installer); with none passed the built-in fixtures are used.

It reads only local files you point it at (or built-in fixtures). It never
runs make, zcat /proc/config.gz, a subprocess, a socket, or the network, and it
does not wrap, test, or re-derive any other team file.

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
  python3 junioros_kconfig_closure_i2s_key.py --selftest
  python3 junioros_kconfig_closure_i2s_key.py --kconfig kconfig.junior \
      [--unit bitnetd.service] [--oci config.json] [--installer install-overlay.sh] \
      [--prev 16HEX]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys

RATE = 0.05
GENESIS = "0" * 16
MAGIC = b"JTR1"
BIND = "127.0.0.1:8765"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False,
         "ue5_launch": False, "bind": BIND}
PORTS = ["Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM"]
ENGINES = ["bitnetd 127.0.0.1:8765", "bitnet.cpp I2_S (llama-server)",
           "i2sd userspace", "BitNet-mlx", "JuniorLLM", "JuniorOS overlay"]
NO_MODULE = {"NAMESPACES", "USER_NS", "PID_NS", "NET_NS", "IPC_NS", "UTS_NS",
             "SECCOMP", "SECCOMP_FILTER", "CGROUPS", "MEMCG", "CGROUP_PIDS",
             "CGROUP_BPF", "BPF_SYSCALL", "PROC_FS", "TMPFS", "INET"}
ORDER = ["NAMESPACES", "USER_NS", "PID_NS", "NET_NS", "IPC_NS", "UTS_NS",
         "SECCOMP", "SECCOMP_FILTER", "CGROUPS", "MEMCG", "CGROUP_PIDS",
         "CGROUP_BPF", "BPF_SYSCALL", "PROC_FS", "TMPFS", "INET", "UNIX",
         "IPV6"]

FIX_KCONFIG = """# Fragment only. Merge against a *vendor* kernel. Do not ship vmlinuz.
CONFIG_NAMESPACES=y
CONFIG_USER_NS=y
CONFIG_PID_NS=y
CONFIG_NET_NS=y
CONFIG_SECCOMP=y
CONFIG_SECCOMP_FILTER=y
CONFIG_CGROUPS=y
CONFIG_MEMCG=y
"""
FIX_UNIT = """[Service]
User=nobody
ProtectSystem=strict
PrivateTmp=yes
PrivateDevices=yes
ProtectKernelTunables=yes
MemoryDenyWriteExecute=yes
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
SystemCallFilter=@system-service
MemoryMax=2G
TasksMax=256
"""
FIX_OCI = json.dumps({
    "hostname": "junior-bitnetd",
    "mounts": [{"destination": "/proc", "type": "proc"},
               {"destination": "/dev", "type": "tmpfs"}],
    "linux": {"namespaces": [{"type": t} for t in
                             ("pid", "ipc", "uts", "mount", "user", "network")],
              "resources": {"memory": {"limit": 2147483648},
                            "pids": {"limit": 256}}},
})
FIX_INSTALLER = """BIND=${JUNIOR_BIND:-127.0.0.1:8765}
case "$host" in 127.0.0.1|localhost|::1) ;; esac
cp "$HERE/seccomp-bitnetd.json" "$DEST/usr/lib/junioros/oci/seccomp-bitnetd.json"
"""

LINE_RE = re.compile(r'^CONFIG_([A-Z0-9_]+)=(y|m|n|"[^"]*"|-?(0x)?[0-9A-Fa-f]+)$')
UNSET_RE = re.compile(r"^# CONFIG_([A-Z0-9_]+) is not set$")


def parse_kconfig(text: str) -> dict:
    vals, dups, bad = {}, [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        u = UNSET_RE.match(line)
        if m:
            sym, val = m.group(1), m.group(2)
        elif u:
            sym, val = u.group(1), "n"
        elif line.startswith("#"):
            continue
        else:
            bad.append(line)
            continue
        if sym in vals:
            dups.append(sym)
        vals[sym] = val
    blob = bool(re.search(r"(^|[\s/])(vmlinuz[^\s]*|bzImage|Image\.gz)\b",
                          "\n".join(l for l in text.splitlines()
                                    if not l.lstrip().startswith("#"))))
    return {"vals": vals, "dups": dups, "bad": bad, "blob": blob}


def unit_directives(text: str) -> dict:
    d = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;[" or "=" not in line:
            continue
        k, v = line.split("=", 1)
        d.setdefault(k.strip(), []).append(v.strip())
    return d


def demands(unit_text: str, oci_text: str, installer_text: str) -> dict:
    """sym -> 'hard' | 'cond' from the consumer texts actually present."""
    need = {}

    def hard(*syms):
        for s in syms:
            need[s] = "hard"

    def cond(s):
        need.setdefault(s, "cond")

    if unit_text:
        u = unit_directives(unit_text)
        yes = lambda k: any(v.lower() in ("yes", "true", "1", "strict", "full")
                            for v in u.get(k, []))
        if yes("ProtectSystem") or yes("PrivateTmp") or yes("ProtectHome"):
            hard("NAMESPACES")
        seccomp_keys = ("SystemCallFilter", "MemoryDenyWriteExecute",
                        "RestrictAddressFamilies", "RestrictNamespaces",
                        "LockPersonality", "RestrictRealtime",
                        "RestrictSUIDSGID", "SystemCallArchitectures")
        if any(k in u for k in seccomp_keys):
            hard("SECCOMP", "SECCOMP_FILTER")
        if "MemoryMax" in u or "TasksMax" in u:
            hard("CGROUPS")
        if "MemoryMax" in u:
            hard("MEMCG")
        if "TasksMax" in u:
            hard("CGROUP_PIDS")
        if yes("PrivateDevices") or "DevicePolicy" in u or "DeviceAllow" in u:
            hard("CGROUPS", "CGROUP_BPF", "BPF_SYSCALL")
        if yes("ProtectKernelTunables"):
            hard("PROC_FS")
        fams = " ".join(u.get("RestrictAddressFamilies", []))
        if "AF_INET" in fams.split():
            hard("INET")
        if "AF_UNIX" in fams.split():
            hard("UNIX")
        if "AF_INET6" in fams.split():
            cond("IPV6")
        for v in u.get("Environment", []):
            if "JUNIOR_BIND=127." in v:
                hard("INET")
            if "JUNIOR_BIND=::1" in v or "JUNIOR_BIND=[::1]" in v:
                hard("IPV6")
    if oci_text:
        cfg = json.loads(oci_text)
        lin = cfg.get("linux", {})
        nsmap = {"pid": "PID_NS", "ipc": "IPC_NS", "uts": "UTS_NS",
                 "user": "USER_NS", "network": "NET_NS", "mount": "NAMESPACES"}
        nss = [n.get("type") for n in lin.get("namespaces", [])]
        if nss:
            hard("NAMESPACES")
        for t in nss:
            if t in nsmap:
                hard(nsmap[t])
        if cfg.get("hostname"):
            hard("UTS_NS")
        res = lin.get("resources", {})
        if res:
            hard("CGROUPS")
        if "memory" in res:
            hard("MEMCG")
        if "pids" in res:
            hard("CGROUP_PIDS")
        for m in cfg.get("mounts", []):
            if m.get("type") == "proc":
                hard("PROC_FS")
            if m.get("type") == "tmpfs":
                hard("TMPFS")
        if "JUNIOR_BIND=127." in oci_text:
            hard("INET")
    if installer_text:
        if "seccomp-bitnetd.json" in installer_text:
            hard("SECCOMP", "SECCOMP_FILTER")
        if "127.0.0.1" in installer_text:
            hard("INET")
        if "::1" in installer_text:
            cond("IPV6")
    return need


def contract_trits(kconfig_text, unit_text="", oci_text="", installer_text=""):
    k = parse_kconfig(kconfig_text)
    need = demands(unit_text, oci_text, installer_text)
    rows = []
    for sym in ORDER:
        val = k["vals"].get(sym)
        dem = need.get(sym)
        listed = val in ("y",) or (val == "m" and sym not in NO_MODULE)
        if listed:
            t = 1 if dem else 0
            why = "listed" if dem else "listed_no_consumer"
        elif dem == "hard":
            t, why = -1, ("absent" if val is None else f"bad_value:{val}")
        else:
            t, why = 0, ("conditional_absent" if dem else "not_demanded")
        if dem is None and not listed:
            continue  # neither demanded nor listed: not part of this closure
        rows.append({"name": f"kconfig.{sym}", "trit": t, "why": why,
                     "demand": dem or "none"})
    for sym in sorted(set(k["vals"]) - set(ORDER)):
        rows.append({"name": f"kconfig.{sym}", "trit": 0,
                     "why": "listed_unmapped", "demand": "none"})
    rows.append({"name": "syntax.lines", "trit": -1 if k["bad"] else 1,
                 "why": ";".join(k["bad"][:3]) or "ok", "demand": "frag"})
    rows.append({"name": "syntax.no_duplicates", "trit": 0 if k["dups"] else 1,
                 "why": ",".join(k["dups"]) or "ok", "demand": "frag"})
    rows.append({"name": "frag.no_kernel_blob", "trit": -1 if k["blob"] else 1,
                 "why": "blob_path" if k["blob"] else "ok", "demand": "frag"})
    return rows


def pack_jtr1(trits) -> bytes:
    n = len(trits)
    gamma = sum(abs(t) for t in trits) / n if n else 0.0
    codes = {0: 0b00, 1: 0b01, -1: 0b10}
    payload = bytearray((n + 3) // 4)
    for i, t in enumerate(trits):
        payload[i // 4] |= codes[t] << (6 - 2 * (i % 4))
    return MAGIC + struct.pack("<fI", gamma, n) + bytes(payload)


def unpack_jtr1(blob: bytes):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    rev = {0b00: 0, 0b01: 1, 0b10: -1}
    out = []
    for i in range(n):
        c = (blob[12 + i // 4] >> (6 - 2 * (i % 4))) & 3
        if c == 0b11:
            raise ValueError("code 0b11 rejected")
        out.append(rev[c])
    return gamma, out


def pack_i2s_block(trits) -> bytes:
    vals = [t + 1 for t in trits[:128]] + [1] * max(0, 128 - len(trits))
    out = bytearray(32)
    for i, c in enumerate(vals):
        out[i // 4] |= c << (6 - 2 * (i % 4))
    return bytes(out)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
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
        "kind": "junioros_kconfig_closure_i2s_key",
        "rate": RATE,
        "n": len(trits),
        "plus": trits.count(1), "zero": trits.count(0), "minus": trits.count(-1),
        "verdict": verdict(trits),
        "rows": rows,
        "jtr1_hex": pack.hex(),
        "i2s_block_hex": pack_i2s_block(trits).hex(),
        "ports": PORTS, "engines": ENGINES, "flags": FLAGS, "prev": prev,
    }
    text = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["key"] = close_rule(prev, pack, text)
    return report


def selftest() -> int:
    fails = []

    def expect(cond, msg):
        if not cond:
            fails.append(msg)

    expect(RATE == 0.05, "rate 0.05")
    expect(FLAGS["bind"].startswith("127.0.0.1:"), "loopback bind")
    expect(not any(FLAGS[k] for k in ("mint", "socket", "layer_one", "fetch",
                                      "ue5_launch")), "flags false")
    rows = contract_trits(FIX_KCONFIG, FIX_UNIT, FIX_OCI, FIX_INSTALLER)
    by = {r["name"]: r["trit"] for r in rows}
    for s in ("NAMESPACES", "USER_NS", "PID_NS", "NET_NS", "SECCOMP",
              "SECCOMP_FILTER", "CGROUPS", "MEMCG"):
        expect(by.get(f"kconfig.{s}") == 1, f"{s} listed +1")
    for s in ("IPC_NS", "UTS_NS", "CGROUP_PIDS", "CGROUP_BPF", "BPF_SYSCALL",
              "PROC_FS", "TMPFS", "INET", "UNIX"):
        expect(by.get(f"kconfig.{s}") == -1, f"{s} missing -1")
    expect(by.get("kconfig.IPV6") == 0, "IPV6 conditional 0")
    expect(by["syntax.lines"] == 1 and by["syntax.no_duplicates"] == 1
           and by["frag.no_kernel_blob"] == 1, "fragment trits pass")
    rep = build_key(rows)
    expect(rep["verdict"] == "hold", "fixture holds")
    # closed fragment goes green
    closed = FIX_KCONFIG + "".join(f"CONFIG_{s}=y\n" for s in ORDER[4:])
    rep2 = build_key(contract_trits(closed, FIX_UNIT, FIX_OCI, FIX_INSTALLER))
    expect(rep2["minus"] == 0 and rep2["verdict"] == "green", "closed green")
    # modular namespace is refused, dup and blob detected
    badfrag = closed.replace("CONFIG_IPC_NS=y", "CONFIG_IPC_NS=m") + \
        "CONFIG_MEMCG=y\nKERNEL=/boot/vmlinuz-6.1\n"
    b = {r["name"]: r["trit"] for r in contract_trits(badfrag, FIX_UNIT, FIX_OCI)}
    expect(b["kconfig.IPC_NS"] == -1, "=m namespace -1")
    expect(b["syntax.no_duplicates"] == 0, "dup 0")
    expect(b["syntax.lines"] == -1, "bad line -1")
    # surplus symbol with no consumer is undecided
    s = {r["name"]: r["trit"] for r in contract_trits(FIX_KCONFIG)}
    expect(s["kconfig.MEMCG"] == 0, "no consumer -> 0")
    # JTR1 + I2_S round trip
    trits = [r["trit"] for r in rows]
    g, back = unpack_jtr1(pack_jtr1(trits))
    expect(back == trits, "jtr1 round trip")
    expect(abs(g - sum(abs(t) for t in trits) / len(trits)) < 1e-6, "gamma")
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 0.0, 1) + bytes([0xC0]))
        expect(False, "0b11 rejected")
    except ValueError:
        pass
    blk = pack_i2s_block(trits)
    expect(len(blk) == 32, "i2s 128-block 32 bytes")
    expect((blk[31] & 3) == 1, "pad code 1")
    # close rule
    body = {k: v for k, v in rep.items() if k != "key"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    want = hashlib.sha256(GENESIS.encode() + pack_jtr1(trits)
                          + text.encode()).hexdigest()[:16]
    expect(rep["key"] == want, "close rule sha256(prev||pack||text)[:16]")
    expect(build_key(rows, prev=rep["key"])["key"] != rep["key"], "chain moves")
    if fails:
        print(json.dumps({"selftest": "fail", "fails": fails}))
        return 1
    print(json.dumps({"selftest": "pass", "key": rep["key"],
                      "verdict": rep["verdict"], "n": rep["n"], "rate": RATE,
                      "plus": rep["plus"], "minus": rep["minus"]}))
    return 0


def _read(path):
    if not path:
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--kconfig")
    ap.add_argument("--unit")
    ap.add_argument("--oci")
    ap.add_argument("--installer")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.kconfig:
        rows = contract_trits(_read(a.kconfig), _read(a.unit), _read(a.oci),
                              _read(a.installer))
    else:
        rows = contract_trits(FIX_KCONFIG, FIX_UNIT, FIX_OCI, FIX_INSTALLER)
    rep = build_key(rows, a.prev)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["verdict"] == "green" else 2


if __name__ == "__main__":
    raise SystemExit(main())
