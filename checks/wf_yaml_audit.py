#!/usr/bin/env python3
"""WF-YAML audit -- the dead-gate detector (BLOCKER 1 class).

Usage:
  wf_yaml_audit.py FILE [FILE ...]      audit every listed workflow file
  wf_yaml_audit.py --doctor-blocker1 F  rewrite F in place so it carries
                                        BLOCKER 1's original two-line
                                        column-zero form verbatim (MUST_FIRE
                                        setup helper; fails closed if the
                                        fixed needle is not present)

Doctrine wiring:
  (1) argv carrying 0 files is RED -- a check over zero units is UNMEASURED,
      never a pass.
  (2) the caller puts the examined count on the wire; this script prints the
      N/N result and the parser mode actually used.
  (3) the MUST_FIRE lives in .github/workflows/ci.yml: it calls
      --doctor-blocker1 on a copy of ci.yml and demands this audit refuse it.
  (4) unreadable is RED, not empty; missing is not zero.
  (5) PyYAML is used when importable; otherwise this degrades to a structural
      indentation scan for BLOCKER 1's exact shape and SAYS SO plainly in
      every message -- it never claims a full parse it did not perform.
Output avoids the verify_summary-ingested tokens by design: no 'green',
'passed', or 'SUMMARY' on any path, and no suite domain prefix at line start.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BLOCKER1_FIXED = (
    "          mustfire_1=$'launchers: 8/8 watchdog checks green\\ncontracts: 126/126 green'\n"
)
BLOCKER1_BROKEN = (
    "          mustfire_1='launchers: 8/8 watchdog checks green\ncontracts: 126/126 green'\n"
)


def structural_check(text):
    # NOT a YAML parse. Fallback scan for BLOCKER 1's shape only: content of a
    # '|' or '>' block scalar that dedents BELOW the opener's base indent and
    # is then followed by a line back at the block-content indentation -- e.g.
    # a continuation line demoted to column zero.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^( *)[^#\n]*:\s+[|>][-+]?\s*$", lines[i])
        if not m:
            i += 1
            continue
        base = len(m.group(1))
        content = None
        j = i + 1
        while j < len(lines):
            cur = lines[j]
            if cur.strip() == "":
                j += 1
                continue
            ind = len(cur) - len(cur.lstrip(" "))
            if content is None and ind > base:
                content = ind
            if ind <= base:
                nxt = j + 1
                while nxt < len(lines) and lines[nxt].strip() == "":
                    nxt += 1
                if content is not None and ind < base and nxt < len(lines):
                    nind = len(lines[nxt]) - len(lines[nxt].lstrip(" "))
                    if nind >= content:
                        return (
                            f"line {j + 1} at indent {ind} under-cuts the block scalar"
                            f" opened at line {i + 1} (base {base}), then"
                            f" line {nxt + 1} re-indents to {nind} -- the"
                            " column-zero-continuation shape"
                        )
                break
            if content is not None and ind < content:
                break
            j += 1
        i = j if j > i else i + 1
    return None


def audit(paths):
    if not paths:
        print("WF-YAML RED: argv carried 0 files -- the caller measured nothing (doctrine 1)")
        return 1
    try:
        import yaml

        have_yaml = True
    except ImportError:
        yaml = None
        have_yaml = False
    if have_yaml:
        mode = "full YAML parse via PyYAML"
    else:
        mode = (
            "STRUCTURAL INDENTATION SCAN ONLY (PyYAML not importable here) -- "
            "NOT a full YAML parse (doctrine 5)"
        )
    bad = 0
    for p in paths:
        try:
            with Path(p).open(encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"WF-YAML RED-unreadable {p}: {e} -- unreadable is not empty (doctrine 4)")
            bad += 1
            continue
        if have_yaml:
            try:
                yaml.safe_load(text)
            except Exception as e:
                first = str(e).splitlines()
                print(f"WF-YAML RED-unparseable {p}: {first[0] if first else 'unknown YAML error'}")
                bad += 1
        else:
            prob = structural_check(text)
            if prob:
                print(f"WF-YAML RED-structural {p}: {prob}")
                bad += 1
    if bad:
        print(f"WF-YAML RED: {bad} of {len(paths)} workflow file(s) refused under {mode}")
        return 1
    print(
        f"WF-YAML ok: examined {len(paths)} workflow file(s);"
        f" {len(paths)}/{len(paths)} accepted by {mode}"
    )
    return 0


def doctor(path):
    try:
        with Path(path).open(encoding="utf-8") as f:
            t = f.read()
    except OSError as e:
        print(f"WF-YAML DOCTOR RED: unreadable {path}: {e} -- unreadable is not empty (doctrine 4)")
        return 1
    if BLOCKER1_FIXED not in t:
        print(
            f"WF-YAML DOCTOR RED: the fixed mustfire_1 needle is not in {path}"
            " -- cannot rebuild BLOCKER 1; has the fix landed,"
            " or has it been reverted?"
        )
        return 1
    t = t.replace(BLOCKER1_FIXED, BLOCKER1_BROKEN, 1)
    try:
        with Path(path).open("w", encoding="utf-8") as f:
            f.write(t)
    except OSError as e:
        print(f"WF-YAML DOCTOR RED: cannot write {path}: {e}")
        return 1
    print(
        "WF-YAML DOCTOR ok: copy now carries BLOCKER 1's original two-line"
        " column-zero form verbatim"
    )
    return 0


def main(argv):
    if argv[:1] == ["--doctor-blocker1"]:
        if len(argv) != 2:
            print("WF-YAML DOCTOR RED: --doctor-blocker1 takes exactly one file")
            return 1
        return doctor(argv[1])
    return audit(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
