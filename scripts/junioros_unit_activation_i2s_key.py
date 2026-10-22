"""JuniorOS overlay: bitnetd unit-activation key (JTR1 trit pack + I2_S). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" -- hardened
bitnetd.service + juniorctl on PATH (JuniorHome docs/ROADMAP.md JuniorOS
overlay). Quant Chief pair A, slot 2026-10-22 14:05 MT: overlay remaining gaps,
the bitnetd/i2sd systemd unit-activation gap.
Trit Pack & I2_S side: the trit pack and I2_S key. Winsor & Quant Error covers
the quant-error bound side of the same item in a different file.

New capability. Earlier overlay files score the unit's hardening flags,
restart/ordering/sandbox reach, MemoryMax, the ExecStart -> i2sd install
closure, seccomp, sysctl, kconfig and the OCI bundle. None asks whether the
installed unit is ever *activated* at boot. This file statically reads
rails/linux/install-overlay.sh, rails/linux/bitnetd.service and
rails/linux/debian/preseed.fragment and checks:
  * search path: the system unit lands in a directory systemd searches
    (etc/systemd/system, usr/lib/systemd/system, lib/systemd/system) relative
    to the root the overlay is actually copied onto,
  * enablement: an [Install] WantedBy= target only takes effect once a
    <target>.wants/ symlink or a *.preset "enable" line exists; a plain cp of
    the unit leaves it disabled,
  * preseed hand-off: in a Debian netinst late_command the target system is
    mounted at /target and commands must go through `in-target`; a bare
    `systemctl enable` or a copy to /opt/... acts on the installer ramdisk,
    and a commented example is not a command at all,
  * user unit: usr/lib/systemd/user/bitnetd.service is only started when a
    default.target.wants link or a user preset enables it,
  * port owner: the system and user units both advertise JUNIOR_BIND; at most
    one of them may be enabled for 127.0.0.1:8765,
  * WantedBy names a standard boot target, and the bind is loopback.
It never runs systemctl, starts bitnetd, opens a socket, runs a subprocess or
touches the network. It does not wrap, test, or re-derive any other team file.

Trits (+1 pass, 0 undecided, -1 fail):
  act.search_path        system unit on systemd's unit search path at install root
  act.enabled            wants/ symlink or preset enables the system unit
  act.preseed_target     late_command copies under /target (not installer root)
  act.preseed_in_target  live (uncommented) `in-target systemctl enable bitnetd`
  act.user_enabled       user unit enabled, or user unit not shipped
  act.port_owner         at most one enabled unit owns JUNIOR_BIND
  act.wantedby_standard  WantedBy= is multi-user.target / default.target
  bind.loopback          unit JUNIOR_BIND is 127.0.0.1:8765, no wildcard

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
  python3 scripts/junioros_unit_activation_i2s_key.py --selftest
  python3 scripts/junioros_unit_activation_i2s_key.py \
      --installer rails/linux/install-overlay.sh \
      --unit rails/linux/bitnetd.service \
      [--preseed rails/linux/debian/preseed.fragment] [--prev HEX16] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
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
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md JuniorOS overlay: hardened "
                "bitnetd.service unit activation (enable/preset/preseed)")
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
UNIT = "bitnetd.service"
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}
SYSTEM_DIRS = ("etc/systemd/system", "usr/lib/systemd/system",
               "lib/systemd/system")
USER_DIRS = ("etc/systemd/user", "usr/lib/systemd/user")
PRESET_DIRS = ("etc/systemd/system-preset", "usr/lib/systemd/system-preset",
               "lib/systemd/system-preset")
USER_PRESET_DIRS = ("etc/systemd/user-preset", "usr/lib/systemd/user-preset")
STANDARD_TARGETS = ("multi-user.target", "graphical.target", "default.target")
TRIT_NAMES = ("act.search_path", "act.enabled", "act.preseed_target",
              "act.preseed_in_target", "act.user_enabled", "act.port_owner",
              "act.wantedby_standard", "bind.loopback")


# ---------------------------------------------------------------- parsing
def parse_unit(text: str) -> dict:
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
            out.setdefault((section, k.strip()), []).append(v.strip())
    return out


def _words(unit, section, key):
    return " ".join(v for v in unit.get((section, key), []) if v).split()


def _env(unit):
    env = {}
    for v in unit.get(("Service", "Environment"), []):
        for item in v.split():
            if "=" in item:
                k, val = item.strip('"').split("=", 1)
                env[k] = val
    return env


def _dest_path(tok: str):
    """'"$DEST/etc/x"' -> 'etc/x'; None when not under $DEST."""
    tok = tok.strip().strip('"').strip("'")
    for pre in ("$DEST/", "${DEST}/"):
        if tok.startswith(pre):
            return posixpath.normpath(tok[len(pre):])
    return None


def parse_installer(text: str) -> dict:
    """Statically collect paths and symlinks the installer writes under DEST.

    Recognised: cp SRC... "$DEST/x", cat > "$DEST/x" <<EOF, ln -s[f] T "$DEST/x",
    printf/echo ... > "$DEST/x". Heredoc bodies are kept for preset files.
    """
    files: dict = {}
    links: dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        heredoc = re.match(r"cat\s+>\s*(\S+)\s+<<-?\s*'?\"?(\w+)'?\"?", line)
        if heredoc:
            dst, tag = _dest_path(heredoc.group(1)), heredoc.group(2)
            body = []
            while i < len(lines) and lines[i].strip() != tag:
                body.append(lines[i])
                i += 1
            i += 1
            if dst:
                files[dst] = "\n".join(body)
            continue
        toks = line.split()
        if toks and toks[0] == "cp" and len(toks) >= 3:
            dst = _dest_path(toks[-1])
            if dst:
                srcs = [t for t in toks[1:-1] if not t.startswith("-")]
                if len(srcs) == 1 and not dst.endswith("/"):
                    files[dst] = None
                else:
                    for s in srcs:
                        name = posixpath.basename(s.strip('"'))
                        files[posixpath.join(dst, name)] = None
            continue
        if toks and toks[0] == "ln" and any(t.startswith("-") and "s" in t
                                            for t in toks[1:]):
            args = [t for t in toks[1:] if not t.startswith("-")]
            if len(args) >= 2:
                dst = _dest_path(args[-1])
                if dst:
                    links[dst] = args[0].strip('"')
            continue
        redir = re.search(r">>?\s*(\"?\$\{?DEST\}?/[^\s\"]+\"?)", line)
        if redir and toks and toks[0] in ("echo", "printf"):
            dst = _dest_path(redir.group(1))
            if dst:
                m = re.match(r"(?:echo|printf)\s+(['\"])(.*?)\1", line)
                files[dst] = (m.group(2) if m else "")
    return {"files": files, "links": links}


def parse_preseed(text: str) -> dict:
    """Split late_command material into live and commented commands."""
    live, commented = [], []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        is_comment = s.startswith("#")
        body = s.lstrip("#").strip()
        if body.lower().startswith("late_command example:"):
            continue
        if "preseed/late_command" in body:
            body = body.split("string", 1)[-1].strip()
        cmds = [c.strip() for c in body.split(";") if c.strip()]
        for c in cmds:
            if re.match(r"(in-target\s+)?(cp|systemctl|ln|mkdir|rsync)\b", c):
                (commented if is_comment else live).append(c)
    return {"live": live, "commented": commented}


# ---------------------------------------------------------------- trits
def _preset_enables(text, unit_name):
    for raw in (text or "").splitlines():
        p = raw.split("#", 1)[0].split()
        if len(p) >= 2 and p[0] == "enable" and (
                p[1] == unit_name or (p[1].endswith("*")
                                      and unit_name.startswith(p[1][:-1]))):
            return True
    return False


def _enabled(inst, dirs, preset_dirs, targets, unit_name):
    files, links = inst["files"], inst["links"]
    for d in dirs[:1] + dirs[1:]:
        for t in targets:
            if posixpath.join(d, t + ".wants", unit_name) in links:
                return "wants"
    for path, body in files.items():
        if posixpath.dirname(path) in preset_dirs and path.endswith(".preset"):
            if _preset_enables(body, unit_name):
                return "preset"
    return None


def score(installer: str, unit_text: str, preseed: str | None = None) -> dict:
    inst = parse_installer(installer)
    unit = parse_unit(unit_text)
    ps = parse_preseed(preseed) if preseed is not None else None
    files = inst["files"]
    notes: dict = {}
    t: dict = {}

    sys_paths = [p for p in files if posixpath.basename(p) == UNIT
                 and posixpath.dirname(p) in SYSTEM_DIRS]
    user_paths = [p for p in files if posixpath.basename(p) == UNIT
                  and posixpath.dirname(p) in USER_DIRS]
    wanted = _words(unit, "Install", "WantedBy")

    # where does the overlay root actually land on the target?
    roots = []
    if ps is not None:
        for c in ps["live"] + ps["commented"]:
            m = re.match(r"(?:in-target\s+)?cp\s+(?:-\S+\s+)*\S+\s+(\S+)", c)
            if m:
                roots.append(m.group(1).rstrip("/").rstrip(".").rstrip("/")
                             or "/")
    if not sys_paths:
        t["act.search_path"] = -1
        notes["act.search_path"] = "installer writes no system bitnetd.service"
    elif any(r not in ("/target", "/") for r in roots):
        t["act.search_path"] = -1
        notes["act.search_path"] = (
            "overlay copied onto %s; %s/%s is not on systemd's unit search "
            "path" % (roots[0], roots[0], sys_paths[0]))
    elif not roots and ps is None:
        t["act.search_path"] = 0
        notes["act.search_path"] = "no preseed given; install root unknown"
    else:
        t["act.search_path"] = 1

    how = _enabled(inst, SYSTEM_DIRS, PRESET_DIRS, wanted or STANDARD_TARGETS,
                   UNIT)
    live_enable = ps is not None and any(
        re.match(r"in-target\s+systemctl\s+enable\b.*\bbitnetd", c)
        for c in ps["live"])
    if how or live_enable:
        t["act.enabled"] = 1
        notes["act.enabled"] = how or "in-target systemctl enable"
    else:
        t["act.enabled"] = -1
        notes["act.enabled"] = ("no %s.wants/%s symlink, no preset enable line, "
                                "no live in-target enable: unit lands disabled"
                                % ((wanted or ["multi-user.target"])[0], UNIT))

    if ps is None:
        t["act.preseed_target"] = 0
        t["act.preseed_in_target"] = 0
        notes["act.preseed_target"] = "no preseed given"
    else:
        live_cp = [c for c in ps["live"] if re.match(r"(in-target\s+)?cp\b", c)]
        if live_cp and all(re.search(r"\s/target(/|\s|$)", c) for c in live_cp):
            t["act.preseed_target"] = 1
        else:
            t["act.preseed_target"] = -1
            notes["act.preseed_target"] = (
                "late_command copy is %s; target root is /target"
                % ("commented out" if not live_cp else "outside /target"))
        if live_enable:
            t["act.preseed_in_target"] = 1
        else:
            t["act.preseed_in_target"] = -1
            bare = [c for c in ps["live"] + ps["commented"]
                    if c.startswith("systemctl")]
            notes["act.preseed_in_target"] = (
                "enable is %s" % ("bare/commented `%s` (acts on installer, "
                                  "not /target)" % bare[0] if bare
                                  else "absent"))

    user_on = None
    if user_paths:
        user_on = _enabled(inst, USER_DIRS, USER_PRESET_DIRS,
                           ("default.target",), UNIT)
        t["act.user_enabled"] = 1 if user_on else -1
        if not user_on:
            notes["act.user_enabled"] = ("user unit shipped without "
                                         "default.target.wants link or user "
                                         "preset")
    else:
        t["act.user_enabled"] = 1

    owners = int(t["act.enabled"] == 1) + int(bool(user_on))
    t["act.port_owner"] = 1 if owners == 1 else (-1 if owners > 1 else 0)
    notes["act.port_owner"] = "%d enabled owner(s) of JUNIOR_BIND" % owners

    t["act.wantedby_standard"] = 1 if wanted and all(
        w in STANDARD_TARGETS for w in wanted) else -1

    env = _env(unit)
    bind = env.get("JUNIOR_BIND", "")
    t["bind.loopback"] = 1 if (bind == "%s:%d" % (BIND, BITNETD_PORT)
                               and WILDCARD not in unit_text) else -1

    trits = [t[n] for n in TRIT_NAMES]
    return {"item": ROADMAP_ITEM, "trits": dict(zip(TRIT_NAMES, trits)),
            "notes": notes, "ports": list(PORTS), "engines": list(ENGINES)}


# ---------------------------------------------------------------- pack / key
def jtr1_pack(trits) -> bytes:
    n = len(trits)
    gamma = (sum(abs(x) for x in trits) / n) if n else 0.0
    payload = bytearray((n + 3) // 4)
    for i, x in enumerate(trits):
        payload[i // 4] |= _JTR[x] << (6 - 2 * (i % 4))
    return MAGIC + struct.pack("<fI", gamma, n) + bytes(payload)


def jtr1_unpack(blob: bytes):
    if blob[:4] != MAGIC:
        raise ValueError("bad magic")
    gamma, n = struct.unpack("<fI", blob[4:12])
    out = []
    for i in range(n):
        code = (blob[12 + i // 4] >> (6 - 2 * (i % 4))) & 3
        if code == 3:
            raise ValueError("code 0b11 rejected")
        out.append(_JTR_DEC[code])
    return gamma, out


def i2s_block(trits) -> bytes:
    codes = [x + 1 for x in trits][:I2S_BLOCK]
    codes += [1] * (I2S_BLOCK - len(codes))
    out = bytearray(I2S_BLOCK // 4)
    for i, c in enumerate(codes):
        out[i // 4] |= c << (6 - 2 * (i % 4))
    return bytes(out)


def close_key(prev: str, pack: bytes, text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", prev):
        raise ValueError("prev must be 16 hex chars")
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def verdict(trits) -> str:
    n = len(trits)
    if any(x == -1 for x in trits):
        return "hold"
    return "green" if trits.count(0) <= RATE * n else "hold"


def report(installer, unit_text, preseed=None, prev=GENESIS) -> dict:
    r = score(installer, unit_text, preseed)
    trits = [r["trits"][k] for k in TRIT_NAMES]
    pack = jtr1_pack(trits)
    r.update(verdict=verdict(trits), rate=RATE, pass_count=trits.count(1),
             n=len(trits), i2s_block=i2s_block(trits).hex(),
             flags={"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                    "fetch": FETCH, "ue5_launch": UE5_LAUNCH, "bind": BIND})
    text = json.dumps(r, sort_keys=True, separators=(",", ":"))
    r["prev"] = prev
    r["key"] = close_key(prev, pack, text)
    return r


# ---------------------------------------------------------------- selftest
_UNIT_OK = """[Unit]
Description=t
[Service]
ExecStart=/usr/local/bin/junior-bitnetd
Environment=JUNIOR_BIND=127.0.0.1:8765
[Install]
WantedBy=multi-user.target
"""
_INST_BASE = """#!/bin/sh
set -eu
cp "$HERE/bitnetd.service" "$DEST/etc/systemd/system/bitnetd.service"
cat > "$DEST/usr/lib/systemd/user/bitnetd.service" <<'EOF'
[Install]
WantedBy=default.target
EOF
"""
_INST_FIXED = _INST_BASE + """ln -sf ../bitnetd.service "$DEST/etc/systemd/system/multi-user.target.wants/bitnetd.service"
cat > "$DEST/usr/lib/systemd/system-preset/80-junioros.preset" <<'EOF'
enable bitnetd.service
EOF
"""
_PRESEED_BAD = """d-i pkgsel/include string python3 git systemd
# late_command example:
# cp -a /media/usb/junioros-overlay/. /opt/junioros/
# systemctl enable bitnetd.service
"""
_PRESEED_OK = ("d-i preseed/late_command string cp -a /media/usb/junioros-overlay/."
               " /target/ ; in-target systemctl enable bitnetd.service\n")


def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + name)
        ok = ok and bool(cond)

    check("rate is 0.05", RATE == 0.05)
    check("flags false", not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH))
    check("bind loopback", BIND == "127.0.0.1")
    tr = [1, -1, 0, 1, 1, -1, 0]
    g, back = jtr1_unpack(jtr1_pack(tr))
    check("jtr1 roundtrip", back == tr and abs(g - 5 / 7) < 1e-6)
    check("jtr1 header", jtr1_pack(tr)[:4] == b"JTR1" and len(jtr1_pack(tr)) == 14)
    check("i2s block 32 bytes, pad code 1", len(i2s_block(tr)) == 32
          and i2s_block([])[0] == 0b01010101)
    try:
        jtr1_unpack(MAGIC + struct.pack("<fI", 1.0, 1) + b"\xc0")
        check("0b11 rejected", False)
    except ValueError:
        check("0b11 rejected", True)
    pack = jtr1_pack(tr)
    check("close rule", close_key(GENESIS, pack, "x") ==
          hashlib.sha256(GENESIS.encode() + pack + b"x").hexdigest()[:16])

    bad = report(_INST_BASE, _UNIT_OK, _PRESEED_BAD)
    tb = bad["trits"]
    check("bad: /opt root off search path", tb["act.search_path"] == -1)
    check("bad: unit disabled", tb["act.enabled"] == -1)
    check("bad: commented copy", tb["act.preseed_target"] == -1)
    check("bad: no in-target enable", tb["act.preseed_in_target"] == -1)
    check("bad: user unit disabled", tb["act.user_enabled"] == -1)
    check("bad: no port owner undecided", tb["act.port_owner"] == 0)
    check("bad: hold", bad["verdict"] == "hold")

    good = report(_INST_FIXED, _UNIT_OK, _PRESEED_OK)
    tg = good["trits"]
    check("good: enabled via wants", good["notes"].get("act.enabled") == "wants")
    check("good: user unit off is a fail only if shipped",
          tg["act.user_enabled"] == -1)
    inst_sys_only = _INST_FIXED.replace(
        'cat > "$DEST/usr/lib/systemd/user/bitnetd.service"',
        'cat > "$DEST/usr/share/doc/junioros/bitnetd.user.example"')
    g2 = report(inst_sys_only, _UNIT_OK, _PRESEED_OK)
    check("good: all +1", all(v == 1 for v in g2["trits"].values()))
    check("good: green", g2["verdict"] == "green")
    both = _INST_FIXED + ("ln -s ../../../../usr/lib/systemd/user/bitnetd.service "
                          "\"$DEST/etc/systemd/user/default.target.wants/bitnetd.service\"\n")
    b2 = report(both, _UNIT_OK, _PRESEED_OK)
    check("both enabled -> port owner -1", b2["trits"]["act.port_owner"] == -1)
    wild = _UNIT_OK.replace("127.0.0.1:8765", WILDCARD + ":8765")
    check("wildcard bind -1", report(inst_sys_only, wild, _PRESEED_OK)
          ["trits"]["bind.loopback"] == -1)
    check("no preseed -> undecided", report(inst_sys_only, _UNIT_OK)
          ["trits"]["act.preseed_target"] == 0)
    k1 = report(_INST_BASE, _UNIT_OK, _PRESEED_BAD)["key"]
    k2 = report(_INST_BASE, _UNIT_OK, _PRESEED_BAD, prev=k1)["key"]
    check("key deterministic + chained", k1 == bad["key"] and k1 != k2
          and len(k1) == 16)
    print("selftest", "pass" if ok else "fail", "key", bad["key"])
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--installer")
    ap.add_argument("--unit")
    ap.add_argument("--preseed")
    ap.add_argument("--prev", default=GENESIS)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.installer and a.unit):
        ap.error("--installer and --unit are required (or --selftest)")
    with open(a.installer, encoding="utf-8") as f:
        inst = f.read()
    with open(a.unit, encoding="utf-8") as f:
        unit = f.read()
    pre = None
    if a.preseed:
        with open(a.preseed, encoding="utf-8") as f:
            pre = f.read()
    r = report(inst, unit, pre, a.prev)
    if a.json:
        print(json.dumps(r, indent=2, sort_keys=True))
    else:
        for k in TRIT_NAMES:
            v = r["trits"][k]
            print("%+d %-22s %s" % (v, k, r["notes"].get(k, "")))
        print("%s %d/%d key %s" % (r["verdict"], r["pass_count"], r["n"], r["key"]))
    return 0 if r["verdict"] == "green" else 1


if __name__ == "__main__":
    sys.exit(main())
