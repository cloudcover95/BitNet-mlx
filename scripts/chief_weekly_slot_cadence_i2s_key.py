"""Chief weekly audit: slot reconciliation + weekly cadence key (JTR1 trit pack + I2_S).

Roadmap anchor: JuniorHome docs/ROADMAP.md "Automations (current)" ON item
"JuniorCloud chief weekly Sun/Mon 17:00 MDT (dd0b1c93) - only weekly.
Writes docs/WEEKLY_AUDIT.md". There was no quant file for this item yet, and
the slot was ambiguous: ROADMAP says Sun/Mon 17:00, while every
docs/AUTOMATION_AUDIT.md section lists the weekly audit at Mon 17:00 next to
two other weeklies (omega-obj-weekly Sat 10:00, junioros-core-weekly Sat 11:00).

New capability: read the three docs offline (local copies, never fetched),
reconcile the chief weekly slot across them, check that each
docs/WEEKLY_AUDIT.md section actually landed on that slot on a 7-day cadence,
and close it into a JTR1 trit pack, an I2_S block and a chained key. It also
prints the slot the evidence resolves to, so the Sun/Mon wording can be fixed
by a human. This file only reads; it never writes WEEKLY_AUDIT.md.

Trit Pack & I2_S side: the trit pack and I2_S key. Winsor & Quant Error covers
the quant-error bound side of the same item (fire-time lateness bound) in a
different file.

Trits (+1 holds, 0 undecided, -1 broken):
  slot.roadmap_parsed     ROADMAP names the chief weekly with day(s) + HH:MM
  slot.single_day         ROADMAP names exactly one weekday (Sun/Mon -> 0)
  slot.roadmap_vs_daily   latest daily-audit weekly row matches ROADMAP
                          (+1 exact, 0 day inside an ambiguous set, -1 off)
  slot.daily_stable       the weekly row is identical in every daily section
  slot.weekly_live_match  WEEKLY_AUDIT live list schedule == daily-audit row
  slot.task_id_match      ROADMAP task prefix found in WEEKLY_AUDIT live list
  slot.no_collision       no other weekly shares the chief's day + hour
  slot.target_match       ROADMAP and daily row both name WEEKLY_AUDIT.md
  week.headers_on_slot    every weekly header is on the slot day, slot hour
  week.cadence_7d         consecutive weekly headers are exactly 7 days apart
  week.current            latest weekly covers the last slot at or before asof
  week.daily_precedes     each weekly has a same-day daily audit before it
  week.sections_complete  latest weekly has Automations/GitHub/Shipped/
                          Skipped/Gaps headings

JTR1 layout (not GGUF):
  magic b"JTR1" | gamma <f32 LE> | n <u32 LE> | payload (2-bit codes, MSB first)
  codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10 (0b11 rejected); gamma = mean |trit|
I2_S block: code = trit + 1, four per byte MSB first, padded with code 1 to 128.
Close rule: key = sha256(prev || pack || text)[:16], prev = 16 hex chars
(genesis "0"*16), pack = JTR1 bytes, text = canonical JSON of the report.
Rate 0.05: green needs zero -1 trits and at most 5% undecided, else hold.
Compatible engines: bitnetd 127.0.0.1:8765, bitnet.cpp I2_S (llama-server),
i2sd userspace 127.0.0.1:8767, BitNet-mlx, JuniorLLM.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess.

Usage:
  python3 scripts/chief_weekly_slot_cadence_i2s_key.py --selftest
  python3 scripts/chief_weekly_slot_cadence_i2s_key.py \
      --roadmap docs/ROADMAP.md --daily docs/AUTOMATION_AUDIT.md \
      --weekly docs/WEEKLY_AUDIT.md [--asof "YYYY-MM-DD HH:MM"] \
      [--prev HEX16] [--json]
"""
from __future__ import annotations

import argparse
import datetime as dt
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
ENGINES = ("bitnetd 127.0.0.1:8765", "bitnet.cpp I2_S (llama-server)",
           "i2sd userspace 127.0.0.1:8767", "BitNet-mlx", "JuniorLLM")
ROADMAP_ITEM = ("JuniorHome docs/ROADMAP.md Automations ON: JuniorCloud chief "
                "weekly Sun/Mon 17:00 MDT (dd0b1c93), writes docs/WEEKLY_AUDIT.md")
MAGIC = b"JTR1"
GENESIS = "0" * 16
I2S_BLOCK = 128
TARGET = "WEEKLY_AUDIT.md"
_JTR = {0: 0, 1: 1, -1: 2}
_JTR_DEC = {0: 0, 1: 1, 2: -1}
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
REQUIRED_SECTIONS = ("Automations", "GitHub", "Shipped", "Skipped", "Gaps")
TRIT_NAMES = ("slot.roadmap_parsed", "slot.single_day", "slot.roadmap_vs_daily",
              "slot.daily_stable", "slot.weekly_live_match",
              "slot.task_id_match", "slot.no_collision", "slot.target_match",
              "week.headers_on_slot", "week.cadence_7d", "week.current",
              "week.daily_precedes", "week.sections_complete")

_DAYSET = r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:/(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun))*)"
_SCHED = re.compile(r"^" + _DAYSET + r"\s+(\d{1,2}:\d{2})(?:\s+Denver)?$")


# ---------------------------------------------------------------- parsing
def _hm(s: str) -> tuple:
    h, m = s.split(":")
    return int(h), int(m)


def _cells(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_roadmap(text: str) -> dict:
    out = {"days": None, "time": None, "task": None, "target": None}
    for line in text.splitlines():
        if re.search(r"chief weekly", line, re.I):
            m = re.search(r"chief weekly\W*\s+" + _DAYSET + r"\s+(\d{1,2}:\d{2})",
                          line, re.I)
            if m:
                out["days"] = m.group(1).split("/")
                out["time"] = _hm(m.group(2))
            t = re.search(r"\(`?([0-9a-f]{8})`?\)", line)
            if t:
                out["task"] = t.group(1)
            w = re.search(r"Writes\s+`?([\w./-]+)`?", line)
            if w:
                out["target"] = w.group(1)
            break
    return out


def _row_sched(cells):
    for c in cells[1:]:
        m = _SCHED.match(c)
        if m:
            return m.group(1).split("/"), _hm(m.group(2))
    return None


def parse_daily(text: str) -> list:
    """Return daily-audit sections: date, time, weekly row, other weeklies."""
    secs = []
    cur = None
    for line in text.splitlines():
        h = re.match(r"^# Automation audit (\d{4}-\d{2}-\d{2})(?:\s+~?(\d{1,2}:\d{2}))?",
                     line)
        if h:
            cur = {"date": h.group(1),
                   "time": _hm(h.group(2)) if h.group(2) else None,
                   "weekly": None, "weekly_target": False, "others": []}
            secs.append(cur)
            continue
        if cur is None or not line.lstrip().startswith("|"):
            continue
        cells = _cells(line)
        name = cells[0].lower()
        if "weekly" not in name:
            continue
        sched = _row_sched(cells)
        if "weekly audit" in name:
            cur["weekly"] = sched
            cur["weekly_target"] = any(TARGET in c for c in cells[1:])
        elif sched:
            cur["others"].append((cells[0], sched))
    return secs


def parse_weekly(text: str) -> list:
    secs = []
    cur = None
    for line in text.splitlines():
        h = re.match(r"^# Weekly audit (\d{4}-\d{2}-\d{2}) (\d{1,2}:\d{2})", line)
        if h:
            cur = {"date": h.group(1), "time": _hm(h.group(2)),
                   "sections": [], "live": None, "task": None}
            secs.append(cur)
            continue
        if cur is None:
            continue
        s = re.match(r"^## (.+)$", line)
        if s:
            cur["sections"].append(s.group(1).strip())
            continue
        if line.lstrip().startswith("|"):
            cells = _cells(line)
            if "weekly audit" in cells[0].lower():
                cur["live"] = _row_sched(cells)
                if len(cells) > 1 and re.match(r"^[0-9a-f]{8}", cells[1]):
                    cur["task"] = cells[1][:8]
    return secs


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _wd(d: dt.date) -> str:
    return DAYS[d.weekday()]


# ---------------------------------------------------------------- scoring
def score(roadmap: str, daily: str, weekly: str, asof: str | None = None) -> dict:
    rm = parse_roadmap(roadmap)
    ds = parse_daily(daily)
    ws = parse_weekly(weekly)
    t = {}
    notes = []

    t["slot.roadmap_parsed"] = 1 if rm["days"] and rm["time"] else -1
    rdays = rm["days"] or []
    t["slot.single_day"] = (1 if len(rdays) == 1 else 0 if len(rdays) > 1 else -1)
    if len(rdays) > 1:
        notes.append("ROADMAP names %s; pick one day" % "/".join(rdays))

    rows = [s["weekly"] for s in ds if s["weekly"]]
    latest = rows[-1] if rows else None
    if latest is None or not rdays:
        t["slot.roadmap_vs_daily"] = -1
    elif latest[1] != rm["time"] or not set(latest[0]) <= set(rdays):
        t["slot.roadmap_vs_daily"] = -1
        notes.append("daily row %s %02d:%02d outside ROADMAP slot"
                     % ("/".join(latest[0]), *latest[1]))
    elif set(latest[0]) == set(rdays):
        t["slot.roadmap_vs_daily"] = 1
    else:
        t["slot.roadmap_vs_daily"] = 0
    t["slot.daily_stable"] = (1 if rows and all(r == rows[0] for r in rows)
                              else -1 if rows else 0)

    lw = ws[-1] if ws else None
    if lw is None or lw["live"] is None or latest is None:
        t["slot.weekly_live_match"] = 0
    else:
        t["slot.weekly_live_match"] = 1 if lw["live"] == latest else -1
    if not rm["task"]:
        t["slot.task_id_match"] = 0
    else:
        tasks = [w["task"] for w in ws if w["task"]]
        t["slot.task_id_match"] = (0 if not tasks else
                                   1 if all(x == rm["task"] for x in tasks) else -1)

    if latest is None:
        t["slot.no_collision"] = 0
    else:
        hit = False
        for s in ds:
            for name, (odays, otime) in s["others"]:
                if set(odays) & set(latest[0]) and otime[0] == latest[1][0]:
                    hit = True
                    notes.append("collision with %s" % name)
        t["slot.no_collision"] = -1 if hit else 1

    rt = (rm["target"] or "").endswith(TARGET)
    dt_ok = bool(ds) and any(s["weekly_target"] for s in ds[-1:])
    t["slot.target_match"] = 1 if rt and dt_ok else 0 if (rt or dt_ok) else -1

    # resolved slot: daily-audit row when it agrees with ROADMAP's day set
    slot_days = latest[0] if latest else rdays
    slot_time = latest[1] if latest else rm["time"]
    resolved = ("%s %02d:%02d" % ("/".join(slot_days), *slot_time)
                if slot_days and slot_time else None)

    if not ws or not slot_days or not slot_time:
        for k in ("week.headers_on_slot", "week.cadence_7d", "week.current",
                  "week.daily_precedes", "week.sections_complete"):
            t[k] = 0
    else:
        on = all(_wd(_date(w["date"])) in slot_days and w["time"][0] == slot_time[0]
                 and w["time"] >= slot_time for w in ws)
        t["week.headers_on_slot"] = 1 if on else -1
        dates = [_date(w["date"]) for w in ws]
        if len(dates) < 2:
            t["week.cadence_7d"] = 0
        else:
            gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
            t["week.cadence_7d"] = 1 if all(g == 7 for g in gaps) else -1
            if any(g != 7 for g in gaps):
                notes.append("weekly gaps %s days" % gaps)
        if asof:
            a = dt.datetime.strptime(asof, "%Y-%m-%d %H:%M")
        else:
            cands = [dt.datetime.combine(_date(s["date"]), dt.time(*(s["time"] or (0, 0))))
                     for s in ds]
            cands += [dt.datetime.combine(_date(w["date"]), dt.time(*w["time"]))
                      for w in ws]
            a = max(cands)
        last = None
        for back in range(8):
            d = a.date() - dt.timedelta(days=back)
            if _wd(d) in slot_days and dt.datetime.combine(d, dt.time(*slot_time)) <= a:
                last = d
                break
        t["week.current"] = 1 if last and max(dates) >= last else -1
        if t["week.current"] == -1:
            notes.append("no weekly section for slot %s" % last)
        dmap = {}
        for s in ds:
            dmap.setdefault(s["date"], []).append(s["time"])
        prec = all(any(x is not None and x <= w["time"] for x in dmap.get(w["date"], []))
                   for w in ws)
        t["week.daily_precedes"] = 1 if prec else -1
        secs = " | ".join(lw["sections"])
        t["week.sections_complete"] = (1 if all(r in secs for r in REQUIRED_SECTIONS)
                                       else -1)
        notes.append("asof %s" % a.strftime("%Y-%m-%d %H:%M"))

    trits = [t[n] for n in TRIT_NAMES]
    return {"item": ROADMAP_ITEM, "trits": dict(zip(TRIT_NAMES, trits)),
            "resolved_slot": resolved, "weekly_sections": len(ws),
            "daily_sections": len(ds), "notes": notes,
            "ports": list(PORTS), "engines": list(ENGINES)}


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


def report(roadmap, daily, weekly, asof=None, prev=GENESIS) -> dict:
    r = score(roadmap, daily, weekly, asof)
    trits = [r["trits"][k] for k in TRIT_NAMES]
    pack = jtr1_pack(trits)
    r.update(verdict=verdict(trits), rate=RATE, pass_count=trits.count(1),
             n=len(trits), jtr1=pack.hex(), i2s_block=i2s_block(trits).hex(),
             flags={"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
                    "fetch": FETCH, "ue5_launch": UE5_LAUNCH, "bind": BIND})
    text = json.dumps(r, sort_keys=True, separators=(",", ":"))
    r["prev"] = prev
    r["key"] = close_key(prev, pack, text)
    return r


# ---------------------------------------------------------------- selftest
_RM_OK = ("## Automations (current)\nON:\n"
          "- **JuniorCloud chief weekly** Mon 17:00 MDT (`dd0b1c93`) - only weekly. "
          "Writes `docs/WEEKLY_AUDIT.md`.\n")
_RM_AMBIG = _RM_OK.replace("Mon 17:00", "Sun/Mon 17:00")
_DAILY = """# Automation audit 2026-10-12 16:45 MDT (tip a)

| Name | Schedule (MDT) | Notes |
|---|---|---|
| JuniorCloud daily audit | daily 16:45 | this run |
| JuniorCloud weekly audit | Mon 17:00 | WEEKLY_AUDIT.md |
| omega-obj-weekly | Sat 10:00 | report |

# Automation audit 2026-10-19 16:45 MDT (tip b)

| Name | Schedule (MDT) | Notes |
|---|---|---|
| JuniorCloud weekly audit | Mon 17:00 | WEEKLY_AUDIT.md |
| omega-obj-weekly | Sat 10:00 | report |
| junioros-core-weekly | Sat 11:00 | report |

# Automation audit 2026-10-22 16:45 MDT (tip c)

| Name | Schedule (MDT) | Notes |
|---|---|---|
| JuniorCloud weekly audit | Mon 17:00 | WEEKLY_AUDIT.md |
| omega-obj-weekly | Sat 10:00 | report |
"""
_SECS = "\n".join("## " + s for s in ("Automations (live list)", "GitHub last 7 days",
                                       "Shipped", "Skipped", "Gaps"))
_WEEKLY = ("# Weekly audit 2026-10-12 17:06 MDT\n\n" + _SECS + "\n\n"
           "# Weekly audit 2026-10-19 17:04 MDT\n\n## Automations (live list)\n\n"
           "| name | taskId | schedule | note |\n|---|---|---|---|\n"
           "| JuniorCloud weekly audit | dd0b1c93-e923 | Mon 17:00 Denver | this run |\n\n"
           + _SECS.split("\n", 1)[1] + "\n")


def selftest() -> int:
    fails = []

    def check(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    check("rate is 0.05", RATE == 0.05)
    check("flags false, loopback",
          not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH) and BIND == "127.0.0.1")
    tr = [1, 0, -1, 1, 1]
    pk = jtr1_pack(tr)
    check("jtr1 header", pk[:4] == b"JTR1" and len(pk) == 14)
    g, back = jtr1_unpack(pk)
    check("jtr1 roundtrip", back == tr and abs(g - 0.8) < 1e-6)
    try:
        jtr1_unpack(MAGIC + struct.pack("<fI", 1.0, 1) + b"\xc0")
        check("0b11 rejected", False)
    except ValueError:
        check("0b11 rejected", True)
    check("i2s block 32 bytes, code=trit+1",
          len(i2s_block(tr)) == 32 and i2s_block(tr)[0] == 0b10010010)
    check("close rule", close_key(GENESIS, pk, "x") ==
          hashlib.sha256(GENESIS.encode() + pk + b"x").hexdigest()[:16])
    try:
        close_key("zz", pk, "x")
        check("prev validated", False)
    except ValueError:
        check("prev validated", True)

    good = report(_RM_OK, _DAILY, _WEEKLY)
    check("clean docs all +1 / green",
          good["pass_count"] == len(TRIT_NAMES) and good["verdict"] == "green")
    check("resolved slot Mon 17:00", good["resolved_slot"] == "Mon 17:00")
    amb = report(_RM_AMBIG, _DAILY, _WEEKLY)
    check("Sun/Mon -> single_day 0, vs_daily 0, hold",
          amb["trits"]["slot.single_day"] == 0
          and amb["trits"]["slot.roadmap_vs_daily"] == 0 and amb["verdict"] == "hold"
          and amb["resolved_slot"] == "Mon 17:00")
    off = report(_RM_OK.replace("Mon 17:00", "Sun 17:00"), _DAILY, _WEEKLY)
    check("Sun vs Mon -> roadmap_vs_daily -1",
          off["trits"]["slot.roadmap_vs_daily"] == -1)
    late = report(_RM_OK, _DAILY, _WEEKLY, asof="2026-10-26 18:00")
    check("missed 10-26 slot -> week.current -1", late["trits"]["week.current"] == -1)
    early = report(_RM_OK, _DAILY, _WEEKLY, asof="2026-10-26 16:59")
    check("before 10-26 slot -> week.current +1", early["trits"]["week.current"] == 1)
    gap = _WEEKLY.replace("2026-10-12 17:06", "2026-10-05 17:06")
    check("14-day gap -> cadence -1",
          report(_RM_OK, _DAILY, gap)["trits"]["week.cadence_7d"] == -1)
    coll = _DAILY.replace("| omega-obj-weekly | Sat 10:00 |", "| omega-obj-weekly | Mon 17:30 |")
    check("same day+hour weekly -> collision -1",
          report(_RM_OK, coll, _WEEKLY)["trits"]["slot.no_collision"] == -1)
    sun = _WEEKLY.replace("2026-10-19 17:04", "2026-10-18 17:04")
    check("Sunday header on Mon slot -> headers_on_slot -1",
          report(_RM_OK, _DAILY, sun)["trits"]["week.headers_on_slot"] == -1)
    nogaps = _WEEKLY.replace("## Gaps\n", "")
    check("missing Gaps heading -> sections -1",
          report(_RM_OK, _DAILY, nogaps)["trits"]["week.sections_complete"] == -1)
    task = _WEEKLY.replace("dd0b1c93-e923", "deadbeef-0000")
    check("task id mismatch -> -1",
          report(_RM_OK, _DAILY, task)["trits"]["slot.task_id_match"] == -1)
    k1 = report(_RM_OK, _DAILY, _WEEKLY)["key"]
    k2 = report(_RM_OK, _DAILY, _WEEKLY, prev=k1)["key"]
    check("key deterministic and chained",
          k1 == good["key"] and k2 != k1 and re.fullmatch(r"[0-9a-f]{16}", k2))
    print("selftest key", good["key"])
    print("selftest", "PASS" if not fails else "FAIL %d" % len(fails))
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--roadmap")
    ap.add_argument("--daily")
    ap.add_argument("--weekly")
    ap.add_argument("--asof")
    ap.add_argument("--prev", default=GENESIS)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.roadmap and a.daily and a.weekly):
        ap.error("--roadmap, --daily and --weekly are required (local files only)")
    read = lambda p: open(p, encoding="utf-8").read()
    r = report(read(a.roadmap), read(a.daily), read(a.weekly), a.asof, a.prev)
    if a.json:
        print(json.dumps(r, indent=2, sort_keys=True))
    else:
        for k in TRIT_NAMES:
            print("%+d %s" % (r["trits"][k], k))
        print("resolved slot:", r["resolved_slot"])
        for n in r["notes"]:
            print("note:", n)
        print("verdict %s %d/%d key %s" % (r["verdict"], r["pass_count"], r["n"], r["key"]))
    return 0 if r["verdict"] == "green" else 2


if __name__ == "__main__":
    sys.exit(main())
