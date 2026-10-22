"""JuniorOS overlay rootless OCI config.json runtime-spec closure key (JTR1 + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" (Debian/Alpine +
os-release.junior + hardened bitnetd.service + juniorctl on PATH, C4/C6 rootless
OCI bundle), JuniorHome docs/ROADMAP.md JuniorOS overlay. Quant Chief pair A,
slot 2026-10-22 02:05 MT: overlay remaining gaps only. Trit Pack & I2_S side:
the trit pack and I2_S key. Winsor & Quant Error covers the quant-error bound
side of the same item in a different file.

Gap (new capability). The overlay ships rails/linux/oci/config.json and stages
it to usr/lib/junioros/oci/, but the existing checks only scan it for a
loopback bind and the absence of docker.sock / 0.0.0.0. Nothing scores it
against what an OCI runtime (runc / crun, rootless) actually needs to start the
bundle, or against the hardening the overlay promises elsewhere:
  userns.uid_mappings / gid_mappings  a "user" namespace with no
        linux.uidMappings / gidMappings: rootless runc refuses to create it
  userns.uid_in_range                 process.user uid/gid inside a mapping
  seccomp.wired                       rootless.unit() names seccomp-bitnetd.json
        and the installer stages it, but config.json has no linux.seccomp, so
        the runtime never loads it
  ns.cgroup                           cgroup namespace (cgroup v2 isolation)
  resources.memory / pids             already present: now scored
  mask.defaults                       maskedPaths cover the runc default set
        (/proc/acpi, /proc/keys, /proc/scsi, /sys/firmware, ...)
  ro.defaults                         readonlyPaths cover the runc default set
  rootfs.staged                       root.path is relative ("rootfs") but no
        installer stages a rootfs next to config.json
  entry.interpreter                   args[0] junior-bitnetd is a /bin/sh
        script exec'ing python3; undecided unless a rootfs is staged
  tmp.writable                        readonly root with no tmpfs /tmp
  dev.shm                             /dev/shm tmpfs for POSIX shm/semaphores
  net.loopback                        own network ns: lo must come up for
        127.0.0.1:8765; runc/crun bring lo up, undecided by spec alone
plus ociVersion 1.x, readonly root, noNewPrivileges, empty capability sets,
non-root user, uts ns for hostname, env JUNIOR_BIND loopback and matching the
annotation, and no host bind mounts. No recursion: this file does not wrap,
test, or re-derive any other team file.

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
i2sd userspace, BitNet-mlx, JuniorLLM, JuniorOS overlay (runc/crun rootless).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage:
  python junioros_oci_runtime_spec_i2s_key.py --selftest
  python junioros_oci_runtime_spec_i2s_key.py --config rails/linux/oci/config.json \
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
           "i2sd userspace", "BitNet-mlx", "JuniorLLM",
           "JuniorOS overlay (runc/crun rootless OCI)")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: rootless OCI "
                "config.json runtime-spec closure (userns maps, seccomp wiring, "
                "masks, rootfs staging)")
MAGIC = b"JTR1"
GENESIS = "0" * 16
BLOCK = 128
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
RUNC_MASKED = ("/proc/acpi", "/proc/asound", "/proc/kcore", "/proc/keys",
               "/proc/latency_stats", "/proc/timer_list", "/proc/timer_stats",
               "/proc/sched_debug", "/proc/scsi", "/sys/firmware")
# /proc/asound is masked or read-only depending on runtime; accept either.
RUNC_RO = ("/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys",
           "/proc/sysrq-trigger")
_ROOTFS_RE = re.compile(r"junioros/oci/rootfs")


# ---------------------------------------------------------------- checks
def _ns(cfg):
    return {n.get("type") for n in cfg.get("linux", {}).get("namespaces", [])
            if isinstance(n, dict)}


def _mounts(cfg):
    return [m for m in cfg.get("mounts", []) if isinstance(m, dict)]


def _maps(cfg, key):
    m = cfg.get("linux", {}).get(key)
    return m if isinstance(m, list) else []


def _in_map(ident, maps):
    for m in maps:
        try:
            lo, size = int(m["containerID"]), int(m["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if lo <= ident < lo + size:
            return True
    return False


def _host(bind):
    host = bind.rsplit(":", 1)[0] if ":" in bind else bind
    return host


def contract_trits(cfg: dict, installers=()) -> list:
    rows = []

    def add(name, tv):
        rows.append({"check": name, "trit": tv})

    proc = cfg.get("process", {}) if isinstance(cfg.get("process"), dict) else {}
    linux = cfg.get("linux", {}) if isinstance(cfg.get("linux"), dict) else {}
    root = cfg.get("root", {}) if isinstance(cfg.get("root"), dict) else {}
    ns = _ns(cfg)
    mounts = _mounts(cfg)
    dests = {m.get("destination") for m in mounts}
    user = proc.get("user", {}) if isinstance(proc.get("user"), dict) else {}
    uid, gid = user.get("uid"), user.get("gid")

    add("spec.oci_version", 1 if str(cfg.get("ociVersion", "")).startswith("1.") else -1)
    add("root.readonly", 1 if root.get("readonly") is True else -1)
    add("proc.no_new_privileges", 1 if proc.get("noNewPrivileges") is True else -1)
    caps = proc.get("capabilities", {})
    caps_empty = isinstance(caps, dict) and all(
        not caps.get(k) for k in ("bounding", "effective", "inheritable",
                                  "permitted", "ambient"))
    add("proc.caps_empty", 1 if caps_empty else -1)
    add("proc.non_root", 1 if isinstance(uid, int) and uid > 0 and
        isinstance(gid, int) and gid > 0 else -1)

    uidm, gidm = _maps(cfg, "uidMappings"), _maps(cfg, "gidMappings")
    if "user" in ns:
        add("userns.uid_mappings", 1 if uidm else -1)
        add("userns.gid_mappings", 1 if gidm else -1)
        if uidm and gidm and isinstance(uid, int) and isinstance(gid, int):
            add("userns.uid_in_range",
                1 if _in_map(uid, uidm) and _in_map(gid, gidm) else -1)
        else:
            add("userns.uid_in_range", 0)
    else:
        add("userns.uid_mappings", -1)  # rootless needs a user namespace
        add("userns.gid_mappings", -1)
        add("userns.uid_in_range", 0)

    sec = linux.get("seccomp")
    add("seccomp.wired", 1 if isinstance(sec, dict) and sec.get("defaultAction")
        else -1)
    add("ns.cgroup", 1 if "cgroup" in ns else -1)
    add("ns.uts_for_hostname", 1 if not cfg.get("hostname") or "uts" in ns else -1)
    res = linux.get("resources", {}) if isinstance(linux.get("resources"), dict) else {}
    mem = res.get("memory", {}).get("limit") if isinstance(res.get("memory"), dict) else None
    pids = res.get("pids", {}).get("limit") if isinstance(res.get("pids"), dict) else None
    add("resources.memory", 1 if isinstance(mem, int) and mem > 0 else -1)
    add("resources.pids", 1 if isinstance(pids, int) and pids > 0 else -1)

    masked = set(linux.get("maskedPaths", []))
    ro = set(linux.get("readonlyPaths", []))
    mask_need = [p for p in RUNC_MASKED if p != "/proc/asound"]
    add("mask.defaults", 1 if all(p in masked for p in mask_need) and
        ("/proc/asound" in masked or "/proc/asound" in ro) else -1)
    add("ro.defaults", 1 if all(p in ro for p in RUNC_RO) else -1)

    rpath = str(root.get("path", ""))
    staged = any(_ROOTFS_RE.search(text) for _, text in installers)
    if rpath.startswith("/"):
        add("rootfs.staged", 0)
    else:
        add("rootfs.staged", 1 if staged else -1)
    add("entry.interpreter", 0 if not staged else 1)

    tmp_ok = any(m.get("destination") == "/tmp" and m.get("type") == "tmpfs"
                 for m in mounts)
    add("tmp.writable", 1 if tmp_ok or root.get("readonly") is not True else -1)
    add("dev.shm", 1 if "/dev/shm" in dests else 0)
    add("net.loopback", 1 if "network" not in ns else 0)

    env = {}
    for e in proc.get("env", []):
        if isinstance(e, str) and "=" in e:
            k, v = e.split("=", 1)
            env[k] = v
    ann = cfg.get("annotations", {}) if isinstance(cfg.get("annotations"), dict) else {}
    bind = env.get("JUNIOR_BIND", "")
    add("bind.loopback_agree", 1 if bind and _host(bind) in LOOPBACK and
        ann.get("org.junioros.bind", bind) == bind else -1)
    host_bind = any(m.get("type") == "bind" or "bind" in (m.get("options") or [])
                    or "rbind" in (m.get("options") or []) for m in mounts)
    add("mounts.no_host_bind", -1 if host_bind else 1)
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
def _shipped_like() -> dict:
    return {
        "ociVersion": "1.1.0",
        "process": {"terminal": False, "user": {"uid": 1000, "gid": 1000},
                    "args": ["/usr/local/bin/junior-bitnetd"],
                    "env": ["JUNIOR_OFFLINE=1", "JUNIOR_BIND=127.0.0.1:8765"],
                    "noNewPrivileges": True,
                    "capabilities": {k: [] for k in ("bounding", "effective",
                                                     "inheritable", "permitted",
                                                     "ambient")}},
        "root": {"path": "rootfs", "readonly": True},
        "hostname": "junior-bitnetd",
        "mounts": [{"destination": "/proc", "type": "proc", "source": "proc"},
                   {"destination": "/dev", "type": "tmpfs", "source": "tmpfs"}],
        "linux": {"namespaces": [{"type": t} for t in
                                 ("pid", "ipc", "uts", "mount", "user", "network")],
                  "maskedPaths": ["/proc/kcore", "/proc/latency_stats",
                                  "/proc/timer_list", "/proc/timer_stats",
                                  "/proc/sched_debug"],
                  "readonlyPaths": ["/proc/asound", "/proc/bus", "/proc/fs",
                                    "/proc/irq", "/proc/sys", "/proc/sysrq-trigger"],
                  "resources": {"memory": {"limit": 2147483648},
                                "pids": {"limit": 256}}},
        "annotations": {"org.junioros.bind": "127.0.0.1:8765"},
    }


def _tight() -> dict:
    cfg = _shipped_like()
    cfg["mounts"] += [
        {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs",
         "options": ["nosuid", "nodev", "size=65536k"]},
        {"destination": "/dev/shm", "type": "tmpfs", "source": "shm",
         "options": ["nosuid", "noexec", "nodev", "size=65536k"]}]
    lx = cfg["linux"]
    lx["namespaces"] = [{"type": t} for t in
                        ("pid", "ipc", "uts", "mount", "user", "cgroup")]
    lx["uidMappings"] = [{"containerID": 1000, "hostID": 1000, "size": 1}]
    lx["gidMappings"] = [{"containerID": 1000, "hostID": 1000, "size": 1}]
    lx["seccomp"] = {"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}
    lx["maskedPaths"] = list(RUNC_MASKED)
    cfg["root"]["path"] = "/usr/lib/junioros/oci/rootfs"
    return cfg


_INSTALL_NO = ("install-overlay.sh",
               'cp "$HERE/oci/config.json" "$DEST/usr/lib/junioros/oci/config.json"\n')
_INSTALL_YES = ("install-overlay.sh", _INSTALL_NO[1] +
                'cp -a "$HERE/oci/rootfs" "$DEST/usr/lib/junioros/oci/rootfs"\n')


def selftest() -> int:
    ok = True

    def expect(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL", msg, file=sys.stderr)

    expect(RATE == 0.05, "rate 0.05")
    expect(not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH), "flags false")
    expect(BIND == "127.0.0.1", "bind loopback")
    expect(len(PORTS) == 5, "five ports")

    shipped = {r["check"]: r["trit"] for r in contract_trits(_shipped_like(), [_INSTALL_NO])}
    for name in ("userns.uid_mappings", "userns.gid_mappings", "seccomp.wired",
                 "ns.cgroup", "mask.defaults", "rootfs.staged", "tmp.writable"):
        expect(shipped[name] == -1, "shipped-like %s fails" % name)
    for name in ("userns.uid_in_range", "entry.interpreter", "dev.shm", "net.loopback"):
        expect(shipped[name] == 0, "shipped-like %s undecided" % name)
    for name in ("spec.oci_version", "root.readonly", "proc.caps_empty",
                 "proc.non_root", "resources.memory", "resources.pids",
                 "ro.defaults", "bind.loopback_agree", "mounts.no_host_bind"):
        expect(shipped[name] == 1, "shipped-like %s passes" % name)

    tight_rows = contract_trits(_tight(), [_INSTALL_YES])
    tight = {r["check"]: r["trit"] for r in tight_rows}
    expect(all(t == 1 for k, t in tight.items() if k != "rootfs.staged"),
           "tight passes all but absolute-root staging: %r" % tight)
    expect(tight["rootfs.staged"] == 0, "absolute root path undecided")
    rel = _tight()
    rel["root"]["path"] = "rootfs"
    rel_rows = contract_trits(rel, [_INSTALL_YES])
    expect(all(r["trit"] == 1 for r in rel_rows), "relative root + staged = all +1")
    expect(verdict([r["trit"] for r in rel_rows]) == "green", "green verdict")
    expect(verdict([r["trit"] for r in contract_trits(_shipped_like(), [_INSTALL_NO])])
           == "hold", "shipped-like holds")

    bad = _tight()
    bad["process"]["env"][1] = "JUNIOR_BIND=0.0.0.0:8765"
    expect({r["check"]: r["trit"] for r in contract_trits(bad)}["bind.loopback_agree"] == -1,
           "0.0.0.0 bind fails")
    bad = _tight()
    bad["linux"]["uidMappings"] = [{"containerID": 0, "hostID": 1000, "size": 1}]
    expect({r["check"]: r["trit"] for r in contract_trits(bad)}["userns.uid_in_range"] == -1,
           "uid outside map fails")
    bad = _tight()
    bad["mounts"].append({"destination": "/run/docker.sock", "type": "bind",
                          "source": "/run/docker.sock", "options": ["rbind"]})
    expect({r["check"]: r["trit"] for r in contract_trits(bad)}["mounts.no_host_bind"] == -1,
           "host bind mount fails")

    trits = [r["trit"] for r in contract_trits(_shipped_like(), [_INSTALL_NO])]
    blob = pack_jtr1(trits)
    gamma, back = unpack_jtr1(blob)
    expect(back == trits, "JTR1 round trip")
    expect(abs(gamma - sum(abs(t) for t in trits) / len(trits)) < 1e-6, "gamma mean |t|")
    expect(len(pack_i2s_block(trits)) == 32, "I2_S block 32 bytes")
    try:
        unpack_jtr1(MAGIC + struct.pack("<fI", 0.0, 1) + bytes([0xC0]))
        expect(False, "0b11 rejected")
    except ValueError:
        pass

    rep = build_key(contract_trits(_shipped_like(), [_INSTALL_NO]))
    body = {k: v for k, v in rep.items() if k not in ("prev", "key", "jtr1_hex")}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    want = hashlib.sha256(GENESIS.encode() + bytes.fromhex(rep["jtr1_hex"]) +
                          text.encode()).hexdigest()[:16]
    expect(rep["key"] == want, "close rule sha256(prev||pack||text)[:16]")
    rep2 = build_key(contract_trits(_shipped_like(), [_INSTALL_NO]), prev=rep["key"])
    expect(rep2["key"] != rep["key"], "chained prev changes key")
    print(json.dumps({"selftest": "pass" if ok else "fail", "key": rep["key"],
                      "verdict": rep["verdict"], "n": rep["n"], "rate": RATE}))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", help="local OCI config.json (read only)")
    ap.add_argument("--installer", action="append", default=[],
                    help="local installer script text (read only, repeatable)")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.config:
        ap.error("--config or --selftest required")
    with open(a.config, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    installers = []
    for path in a.installer:
        with open(path, "r", encoding="utf-8") as fh:
            installers.append((path.rsplit("/", 1)[-1], fh.read()))
    rep = build_key(contract_trits(cfg, installers), a.prev)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["verdict"] == "green" else 3


if __name__ == "__main__":
    sys.exit(main())
