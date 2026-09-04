#!/usr/bin/env python3
"""Apply the #117 mount-plane fix to the backend, and PROVE the result.

The failure mode specific to THIS fix is not "the edit didn't land" -- B1 covers
that. It is "the edit landed in one arm". The whole defect is that two runtime
arms each carried a private idea of what a run needs; a patch that declares
FS_BIND_PATHS and then materialises it only under singularity reproduces the
defect while looking like the cure, and would pass every generic gate. B7 exists
for that and nothing else.

  B1  every anchor occurs exactly once
  B2  refuted => empty anchor; confirmed => non-empty anchor
  B3  bash -n on the whole spliced backend
  B4  the NOT-DEFECTS survive -- five behaviours that are load-bearing and
      easy to "simplify" away. Each is asserted present BEFORE the edit, so a
      probe that matches nothing cannot silently certify everything.
  B5  every fs_* / run_in_container call resolves to a definition
  B6  contract check: arity and stdin, derived from the definitions (the #120
      gate -- a call can satisfy B5 and still violate the callee's contract)
  B7  BOTH arms materialise from the SHARED array, and the in-container
      verification reports a DENOMINATOR
  B8  a MUST_FIRE drill exists and is reachable
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from apply_113 import _function_bodies  # noqa: E402

GEN = pathlib.Path(__file__).resolve().parent / "h100" / "gen"
SRC = GEN / "fs_container_backend.spliced.sh"
FIX = pathlib.Path(__file__).resolve().parent / "h100" / "fix_117.json"
DST = GEN / "fs_container_backend.bound.sh"

# B4 -- load-bearing behaviours. --no-home is the structural half of the #107
# torch-leak fix (singularity mounts $HOME by default, which is HOW host
# ~/.local/.../torch reached the container), so a patch that "adds binds" by
# dropping it would reopen a closed incident while appearing to fix this one.
NOT_DEFECTS = {
    "--no-home retained (the #107 structural fix)": "--no-home",
    "enroot /dev mount": "/dev:/dev",
    "enroot /sys mount": "/sys:/sys",
    "PYTHONNOUSERSITE forced in-container": "PYTHONNOUSERSITE=1",
    "in-container torch provenance probe": "import torch; print(torch.__file__)",
}


def main() -> int:
    if not FIX.exists():
        print(f"REFUSING: {FIX} absent", file=sys.stderr)
        return 3
    rows = json.loads(FIX.read_text("utf-8"))
    if not rows or rows[0].get("error") or not rows[0].get("content"):
        print(f"REFUSING: task failed: {rows[0].get('error') if rows else 'empty'}",
              file=sys.stderr)
        return 3
    spec = json.loads(rows[0]["content"])
    fixes = spec.get("fixes", [])
    before = SRC.read_text("utf-8")

    dead = [k for k, v in NOT_DEFECTS.items() if v not in before]
    if dead:
        print(f"REFUSING: {len(dead)} of {len(NOT_DEFECTS)} B4 probes match nothing "
              f"in the unedited backend, so they could never detect a regression: "
              f"{dead}", file=sys.stderr)
        return 3

    ok = True
    text = before
    applied, refuted = [], []

    for f in fixes:
        d, verdict = f["defect"], f["verdict"].strip().lower()
        anchor, repl = f["anchor"], f["replacement"]
        if verdict == "refuted":                                            # B2
            if anchor.strip():
                print(f"  FAIL B2  {d} refuted but carries an anchor — ambiguous")
                ok = False
            refuted.append(d)
            print(f"  {d}: REFUTED — {f['rationale'][:220]}")
            continue
        if not anchor.strip():                                              # B2
            print(f"  FAIL B2  {d} confirmed but anchor is EMPTY — no fix landed")
            ok = False
            continue
        n = text.count(anchor)
        if n != 1:                                                          # B1
            print(f"  FAIL B1  {d} anchor occurs {n}x (need exactly 1) — "
                  f"{'fix silently vanishes' if n == 0 else 'wrong site risk'}")
            print(f"           anchor[:120]={anchor[:120]!r}")
            ok = False
            continue
        text = text.replace(anchor, repl, 1)
        applied.append(d)
        print(f"  {d}: applied  ({len(anchor)} -> {len(repl)} chars)")

    print(f"\napplied {len(applied)}/{len(fixes)}: {applied or 'NONE'}"
          + (f"; refuted: {refuted}" if refuted else ""))
    if not applied and not refuted:
        print("  FAIL     an empty fix set is not a successful one"); ok = False

    kept = [k for k, v in NOT_DEFECTS.items() if v in text]
    if len(kept) != len(NOT_DEFECTS):                                       # B4
        print(f"  FAIL B4  {len(kept)}/{len(NOT_DEFECTS)} survived; weakened: "
              f"{sorted(set(NOT_DEFECTS) - set(kept))}")
        ok = False
    else:
        print(f"  PASS B4  all {len(NOT_DEFECTS)} load-bearing behaviours survive")

    nocomment = re.sub(r"(?m)^\s*#.*$", "", text)
    defs = _function_bodies(text)
    called = {m.group(1) for m in re.finditer(
        r"(?:^|\|\||&&|;|\||\$\(|`|\bthen\b|\belse\b|\bdo\b|!)[ \t]*"
        r"((?:fs_[a-z0-9_]+|run_in_container))\b(?!\s*(?:\+?=|\[))",
        nocomment, re.M)}
    unknown = sorted(called - set(defs))
    if unknown:                                                             # B5
        print(f"  FAIL B5  calls {len(unknown)} undefined function(s): {unknown}")
        ok = False
    else:
        print(f"  PASS B5  all {len(called)} called function(s) defined "
              f"(universe of {len(defs)})")

    needs_arg = {n for n, b in defs.items() if re.search(r'\$\{?1[:\-\}\s"]', b)}
    reads_stdin = {n for n, b in defs.items()
                   if re.search(r"(?:\bread\b|\bcat\b|</dev/stdin|\$\(<)", b)}
    viol = []
    for m in re.finditer(r"(?m)^[ \t]*(?:[a-z_]+=\"?\$\()?[ \t]*"
                         r"(fs_[a-z0-9_]+|run_in_container)\b([^\n]*)", nocomment):
        name, rest = m.group(1), m.group(2)
        if name not in defs:
            continue
        args = re.split(r"\|\||&&|[;|)]", rest)[0].strip()
        if name in needs_arg and not args:
            viol.append(f"{name} called with NO argument but dereferences $1")
        if re.search(r"<<-?\s*'?\w", rest) and name not in reads_stdin:
            viol.append(f"{name} fed a heredoc but never reads stdin")
    if viol:                                                                # B6
        for v in sorted(set(viol)):
            print(f"  FAIL B6  {v}")
        ok = False
    else:
        print(f"  PASS B6  {len(defs)} contracts parsed, 0 violations "
              f"({len(needs_arg)} need an arg, {len(reads_stdin)} read stdin)")

    # B7 -- the gate this fix actually needs. Locate each arm by its own marker
    # and require BOTH to reference the shared array. Counting occurrences
    # file-wide would let a singularity-only patch pass.
    ric = re.search(r"(?ms)^run_in_container\(\) \{.*?^\}$", text)
    if not ric:
        print("  FAIL B7  run_in_container vanished"); ok = False
    else:
        body = ric.group(0)
        i_enroot = body.find("--mount")
        i_sing = body.find("--no-home")
        if i_enroot < 0 or i_sing < 0:
            print(f"  FAIL B7  cannot locate both arms (--mount at {i_enroot}, "
                  f"--no-home at {i_sing})")
            ok = False
        else:
            lo, hi = sorted((i_enroot, i_sing))
            arm_a, arm_b = body[lo:hi], body[hi:]
            miss = [n for n, seg in (("enroot-side" if lo == i_enroot else
                                      "singularity-side", arm_a),
                                     ("singularity-side" if lo == i_enroot else
                                      "enroot-side", arm_b))
                    if "FS_BIND_PATHS" not in seg]
            if miss:
                print(f"  FAIL B7  FS_BIND_PATHS is materialised in only ONE arm; "
                      f"missing on the {', '.join(miss)}. A one-arm mount plane "
                      f"IS the defect — the two arms disagreeing about what a run "
                      f"needs.")
                ok = False
            elif not re.search(r"of\s+\$?\{?#?\w*(BIND|declared|N)\w*", body, re.I) \
                    and "declared" not in body:
                print("  FAIL B7  no denominator in the bind verification; "
                      "'k' without 'of N' cannot distinguish 'all bound' from "
                      "'none declared'")
                ok = False
            else:
                print("  PASS B7  both arms materialise FS_BIND_PATHS; "
                      "verification reports a denominator")

    if not re.search(r"FS_REARM\w*BIND|FS_DRILL\w*BIND|MUST_FIRE", text):    # B8
        print("  FAIL B8  no MUST_FIRE drill for the bind plane — a mount check "
              "that has never been seen refusing is unproven")
        ok = False
    else:
        print("  PASS B8  MUST_FIRE bind drill present")

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(text); tmp = fh.name
    rc = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if rc.returncode != 0:                                                  # B3
        print(f"  FAIL B3  bash -n: {rc.stderr.strip()[:300]}"); ok = False
    else:
        print("  PASS B3  bash -n clean")

    if spec.get("gaps"):
        print(f"\ngaps: {spec['gaps'][:1200]}")
    for f in fixes:
        if f["verdict"].strip().lower() != "refuted":
            print(f"observe {f['defect']}: {f['observe'][:180]}")

    if not ok:
        print("\nREFUSING TO WRITE — gates above are red", file=sys.stderr)
        return 5
    DST.write_text(text, "utf-8")
    print(f"\nALL GATES GREEN -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
