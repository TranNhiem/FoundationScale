#!/usr/bin/env python3
"""BASH-LC sweep -- the inner-shell SOURCE-injection detector (BLOCKER 2 class).

Usage:
  bash_lc_sweep.py LAUNCHER [LAUNCHER ...]        classify every bash -lc site
  bash_lc_sweep.py --reinstate-blocker2 SRC DST   copy SRC to DST with BLOCKER
                                                  2's exact unsafe line put
                                                  back (MUST_FIRE setup helper;
                                                  requires the fixed line to
                                                  occur exactly once in SRC)

Classification:
  unsafe   = double-quoted inner source containing $VAR / ${VAR} / $(...)
             OUTSIDE inner single quotes: the outer shell splices the value
             into the inner shell's SOURCE text (word-splitting; $(...)
             executes).
  audited  = single-quoted source with data arguments ($1-style); the
             :513/:1002 nesting idiom (double-quoted source, every expansion
             inside inner single quotes -- enumerated, not blessed: a single
             quote in the value would reopen it, argv is the fix); or a
             non-literal argument (enumerated for a human).
Pure comment lines are not sites: comment text must never make this sweep red
or inflate its denominator (doctrine 5, symmetric).
Doctrine wiring:
  (1) 0 sites is RED -- both launchers visibly use bash -lc, so zero means the
      sweep broke, i.e. UNMEASURED, not a pass.
  (2) the site count and every site are printed on the wire.
  (3) the MUST_FIRE lives in .github/workflows/ci.yml: it calls
      --reinstate-blocker2 and demands this sweep refuse the copy.
  (4) unreadable input is RED, not empty.
Output avoids the verify_summary-ingested tokens by design: no 'green',
'passed', or 'SUMMARY' on any path, and no suite domain prefix at line start.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BLOCKER2_FIXED = 'bash -lc \'python3 "$1"\' _ "$COT_PROBE_PY"'
BLOCKER2_BROKEN = 'bash -lc "python3 $COT_PROBE_PY"'


def classify(files):
    if len(files) < 2:
        print(
            f"BASH-LC RED: the sweep requires both launchers on argv; got"
            f" {len(files)} -- a partial sweep is UNMEASURED (doctrine 1)"
        )
        return 1
    total = 0
    unsafe = []
    audited = []
    for fn in files:
        try:
            with Path(fn).open(encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError as e:
            print(f"BASH-LC RED: unreadable {fn}: {e} -- unreadable is not empty (doctrine 4)")
            return 1
        for n, line in enumerate(lines, 1):
            if line.lstrip().startswith("#"):
                continue  # pure comment lines are not sites (doctrine 5, symmetric)
            at = line.find("bash -lc")
            if at < 0:
                continue
            total += 1
            j = at + len("bash -lc")
            while j < len(line) and line[j] in " \t":
                j += 1
            if j >= len(line) or line[j] not in "'\"":
                audited.append(
                    f"{fn}:{n} [non-literal argument -- enumerated for a human,"
                    " counted as examined]"
                )
                continue
            if line[j] == "'":
                audited.append(
                    f"{fn}:{n} [single-quoted source; expansions passed as data ($1-style)]"
                )
                continue
            k = j + 1
            body = []
            while k < len(line):
                c = line[k]
                if c == "\\":
                    k += 2
                    continue
                if c == '"':
                    break
                body.append(c)
                k += 1
            body = "".join(body)
            hits = []
            inner_single = False
            idx = 0
            while idx < len(body):
                ch = body[idx]
                if ch == "'":
                    inner_single = not inner_single
                elif ch == "$" and not inner_single:
                    if body[idx : idx + 2] == "$(":
                        hits.append(
                            "$(...) executes in the OUTER shell and its stdout"
                            " splices into inner SOURCE"
                        )
                        idx += 1
                    else:
                        m = re.match(r"\$\{?[A-Za-z_][A-Za-z_0-9]*\}?", body[idx:])
                        if m:
                            hits.append(
                                f"{m.group(0)} expanded bare -- the outer shell"
                                " splices the value into the inner shell's SOURCE"
                                " text (word-splits; $(...) content executes)"
                            )
                            idx += len(m.group(0)) - 1
                idx += 1
            if hits:
                unsafe.append(f"{fn}:{n} {' | '.join(hits)}")
            else:
                audited.append(
                    f"{fn}:{n} [double-quoted source; every expansion nested"
                    " inside inner single quotes -- the :513/:1002 idiom;"
                    " a single quote in the value would reopen it]"
                )
    print(
        f"BASH-LC sweep: examined {total} 'bash -lc' site(s) across {len(files)} launcher file(s)"
    )
    for s in audited:
        print(f"  audited: {s}")
    if total == 0:
        print(
            "BASH-LC RED: 0 sites found, but both launchers visibly use 'bash -lc'"
            " -- zero means the sweep broke, i.e. UNMEASURED (doctrine 1)"
        )
        return 1
    if unsafe:
        print(
            f"BASH-LC RED: {len(unsafe)} unsafe site(s) splice outer-shell"
            " expansions into inner SOURCE text:"
        )
        for u in unsafe:
            print(f"  RED: {u}")
        print('  pass paths as data instead:  bash -lc \'python3 "$1"\' _ "$VAR"')
        return 1
    print(f"BASH-LC ok: {total}/{total} sites audited clean, 0 unsafe")
    return 0


def reinstate(src, dst):
    try:
        with Path(src).open(encoding="utf-8") as f:
            t = f.read()
    except OSError as e:
        print(
            f"BASH-LC MUST_FIRE SETUP RED: unreadable {src}: {e}"
            " -- unreadable is not empty (doctrine 4)"
        )
        return 1
    if t.count(BLOCKER2_FIXED) != 1:
        print(
            f"BASH-LC MUST_FIRE SETUP RED: fixed probe line occurs"
            f" {t.count(BLOCKER2_FIXED)} time(s) in {src}; expected exactly 1"
            " -- cannot isolate the reinstatement; has the fix landed,"
            " or has it been reverted?"
        )
        return 1
    t = t.replace(BLOCKER2_FIXED, BLOCKER2_BROKEN, 1)
    try:
        with Path(dst).open("w", encoding="utf-8") as f:
            f.write(t)
    except OSError as e:
        print(f"BASH-LC MUST_FIRE SETUP RED: cannot write {dst}: {e}")
        return 1
    print(f"BASH-LC MUST_FIRE SETUP ok: unsafe splice reinstated on the copy at {dst}")
    return 0


def main(argv):
    if argv[:1] == ["--reinstate-blocker2"]:
        if len(argv) != 3:
            print("BASH-LC MUST_FIRE SETUP RED: --reinstate-blocker2 takes SRC and DST")
            return 1
        return reinstate(argv[1], argv[2])
    return classify(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
