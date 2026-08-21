#!/usr/bin/env python3
"""Mutation battery for the FoundationScale verification framework.

A green test suite is not evidence. `243 passed` is exactly the shape of output
the audited estate produced for 472 steps at grad_norm 0.000 — a confident
report from a check that could not have said anything else. So before this
suite is allowed to count as verification, each load-bearing rule is
individually broken and the suite must notice. A mutation that survives is a
rule with no test behind it.

TWO PRECONDITIONS, both of which are this project's own failure mode turned
inward:

1. The suite must be GREEN before we start, or a survivor means nothing: a
   mutant "killed" by a test that was failing anyway says nothing about
   coverage.

2. The baseline must report ZERO SKIPPED TESTS. This is not fussiness. An
   earlier version of this file ran the suite under an interpreter with no
   torch installed, where every checkpoint test skipped. A mutation in
   `dcp.py` could not be killed there no matter how wrong it was, and the
   battery reported it as a surviving mutant: an alarm from a detector never
   connected to the thing it watches. A skipped test is the vacuous-success
   shape, and a battery whose baseline is full of them is measuring nothing
   while printing a reassuring column of `[killed]`. This is a hard failure —
   the battery exits nonzero (2), not a warning. Do not soften it.

   And say so plainly, because it is the most useful thing this file
   documents: this is the SAME defect that was found independently, at the
   same time, in this repository's own CI — 41 tests skipped, pipeline
   green. Two detectors disconnected in two different places, both reporting
   success. The failure this tool exists to catch is not hypothetical and
   not behind us; a suite whose skips nobody adds up drifts back into it by
   default.

ANCHOR DISCIPLINE

Each mutant is applied by replacing one anchor string in the real source. An
anchor that matches other than exactly once is REFUSED and reported as
`not applicable` — never silently skipped, never guessed at — because a
mutation landing in the wrong function tells you nothing about the rule you
meant to break. A refused anchor is not a passing one, and it must never be
able to look like one: ANY `n/a` row makes this file exit nonzero, whether it
is the whole table or one row of forty-two. The partial case is the one that
actually happened here — a run printed `40 killed, 0 alive, 2 n/a` and exited
0, claiming forty-two rules while measuring forty. A stale anchor means the
table no longer describes the module (usually because the module was fixed
underneath it), so re-derive it against the current source; deleting the row
would shrink the battery silently, which is the same trade. Anchors are
validated once when the table is assembled and re-checked here, row by row, at
run time.

BACKUP / RESTORE

Every module under test is backed up BEFORE any of them is touched, all are
restored in a `finally`, and each restore is verified byte-for-byte.
Restoring only the file being mutated when an exception fires would leave an
earlier module mutated on disk — a corrupted tree that looks perfectly fine.
A crash mid-run must not leave a mutant in the tree.

EXIT STATUS

  0  every mutation in the table was applied, and every one was killed
  1  at least one mutant survived — the suite has a gap
  2  preconditions failed (red suite, skipped tests, missing/empty table,
     module unreadable) or any mutation could not be applied — for that
     part of the table, we never measured

CI depends on the difference between 1 and 2: "the suite has a gap" and
"we never measured" are different states and must not share an exit code.

LAYOUT

This file lives at tools/mutate.py and finds the repository from its own
location — no absolute paths, no pinned interpreter, no usernames. Run it
with the Python you intend to verify under:

    python tools/mutate.py

`sys.executable` runs the suite, and precondition 2 is the tripwire that
proves that interpreter can actually see the optional dependencies the tests
need — the refusal, not a blessed path, is the guard. `mutations.json` is
committed beside this file. Run with `--help` for what a surviving mutant
means and what to do about it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

# tools/mutate.py -> tools/ -> repository root. Nothing in this file is allowed
# to contain a machine-specific path: the earlier incarnation pinned one
# developer's absolute checkout and venv, which is exactly the "works on the box
# that wrote it" assumption the audit was about.
ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
SRC = ROOT / "src" / "foundationscale"

# The mutation table is committed alongside this tool. A battery without its
# table is not a battery that found nothing — see load_table().
TABLE = TOOLS / "mutations.json"

# Run the suite with the interpreter running THIS FILE. There is deliberately
# no pinned venv here: the contract is "run tools/mutate.py with the Python you
# intend to verify under", and precondition 2 — zero skipped tests — is the
# tripwire that proves that interpreter can see the project's optional
# dependencies. Under a torch-less interpreter every checkpoint test skips and
# the battery refuses to run.
PY = sys.executable

MODULE_PATHS = {
    "core": "gates/core.py",
    "dcp": "checkpoint/dcp.py",
    "manifest": "provenance/manifest.py",
    "topology": "topology.py",
    "parity": "verify/parity.py",
    "checkpoint_gates": "gates/checkpoint_gates.py",
}

_REQUIRED_KEYS = ("name", "what", "anchor", "replacement")


def run_suite() -> tuple[bool, str, str]:
    """Run the full suite. No `-x`: we want every test a mutant breaks.

    Two mutants killed by the same single test look identical to two mutants
    killed by genuinely different coverage unless the killing tests are named,
    and an early exit throws that away.
    """
    p = subprocess.run(
        [PY, "-m", "pytest", "tests/", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return p.returncode == 0, p.stdout, p.stderr


_COUNT = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|error)")


def tally(out: str) -> dict[str, int]:
    """Parse pytest's summary line into counts."""
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    return {kind: int(n) for n, kind in _COUNT.findall(tail)}


def failing_tests(out: str) -> list[str]:
    return [
        ln.split("::", 1)[-1].split(" ")[0] for ln in out.splitlines() if ln.startswith("FAILED")
    ]


def check_baseline() -> tuple[bool, str]:
    """Green AND complete, or we are not measuring anything."""
    green, out, err = run_suite()
    counts = tally(out)
    if not green:
        detail = out[-2000:]
        if err.strip():
            # Under sys.executable portability the likeliest new red is "this
            # interpreter has no pytest at all", which speaks only on stderr.
            detail += f"\n--- pytest stderr ---\n{err[-1000:]}"
        return False, f"BASELINE IS RED — fix the suite first.\n{detail}"
    skipped = counts.get("skipped", 0)
    if skipped:
        return False, (
            f"BASELINE HAS {skipped} SKIPPED TEST(S) under {PY}.\n"
            "A skipped test cannot kill a mutant, so every mutation it would have "
            "caught gets reported as a rule with no test behind it — a false alarm "
            "from a detector that was never wired up. Install the missing "
            "dependency or fix the skip condition; do not run the battery like this."
        )
    if not counts.get("passed"):
        return False, "BASELINE RAN ZERO TESTS — collection is broken."
    return True, f"baseline: {counts['passed']} passed, 0 skipped"


def die(msg: str) -> NoReturn:
    """Exit 2: this run measured nothing, and must not be mistaken for a result.

    Everything that prevents the battery from starting (no table, malformed
    table, unreadable module) is a precondition failure, not a surviving
    mutant — CI reads 1 as "the suite has a gap" and 2 as "we never measured",
    and the two must never blur.
    """
    print(msg, file=sys.stderr)
    sys.exit(2)


def load_table(only: str | None) -> dict[str, list[dict]]:
    if not TABLE.exists():
        die(
            f"no mutation table at {TABLE} — it is committed alongside this tool.\n"
            "Restore it rather than running without it: a battery with no table "
            "has not 'found nothing', it has not run, and it exits 2 to say so."
        )
    try:
        data = json.loads(TABLE.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        die(f"{TABLE} is not valid JSON: {exc}")
    unknown = set(data) - set(MODULE_PATHS)
    if unknown:
        die(f"table names modules with no path mapping: {sorted(unknown)}")
    if only:
        if only not in data:
            die(f"unknown module {only!r}; have: {', '.join(sorted(data))}")
        data = {only: data[only]}
    if not any(data.values()):
        die(
            f"{TABLE} selects zero mutations — a battery with no mutants proves "
            "nothing, so it does not get to exit 0."
        )
    for mod, muts in sorted(data.items()):
        for m in muts:
            missing = [k for k in _REQUIRED_KEYS if k not in m]
            if missing:
                die(f"{mod}: mutation entry missing key(s) {missing}: {str(m)[:120]!r}")
            if not isinstance(m["anchor"], str) or not isinstance(m["replacement"], str):
                die(f"{mod}/{m.get('name', '?')}: anchor and replacement must be strings")
    return data


EPILOG = """\
exit status — CI relies on 1 and 2 being different:
  0  every applied mutant was killed: each broken rule had a test behind it
  1  at least one mutant SURVIVED: the suite stayed green with a rule broken
  2  never measured: suite red, skipped tests, no table, nothing applied

WHAT A SURVIVING MUTANT MEANS
  The battery deliberately broke one documented rule — flipped a verdict,
  dropped a byte check, widened a tolerance — and the test suite stayed
  green. Whatever that rule protects is currently enforced by nothing, no
  matter how reassuring the pass count underneath it is. `42 passed` over a
  surviving mutant is the same shape as `243 passed` over a dead training
  run.

WHAT TO DO ABOUT A SURVIVOR
  Write the test that fails under the mutation and passes without it, then
  re-run the battery and watch the mutant die. Leave the mutation in
  mutations.json — it is now the specification for the test you owe. Do NOT
  delete or soften the mutation to quiet the battery: that converts "the
  suite has a hole" into "the battery cannot see the hole", which is
  strictly worse, because it also buys confidence.

WHY THE BASELINE MAY REFUSE TO RUN (exit 2)
  No mutant is tried until the full suite passes AND reports zero skipped
  tests. A skipped test cannot kill anything; a battery run on top of skips
  reads live detectors as dead ones. Run this file with the interpreter that
  has the project's optional dependencies installed (e.g. torch for the
  checkpoint modules):

      python tools/mutate.py                whole table
      python tools/mutate.py --module dcp   one module
      python tools/mutate.py --list         show the table, run nothing

mutations.json lives beside this file. Each entry names a module, the rule
being broken, an anchor string that must occur exactly once in that module's
source, and the anchor's replacement; an anchor that does not match exactly
once is refused, never guessed.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="tools/mutate.py",
        description=(
            "Mutation battery: break each load-bearing rule in the framework, one "
            "at a time, and require the test suite to notice. A mutant that "
            "survives is a rule with no test behind it."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--module", help="run one module's mutations only")
    ap.add_argument("--list", action="store_true", help="print the table and exit")
    args = ap.parse_args()

    table = load_table(args.module)

    if args.list:
        for mod, muts in sorted(table.items()):
            print(f"\n{mod}  ({len(muts)} mutations)")
            for m in muts:
                print(f"  {m['name']:34s} {m['what'][:88]}")
        print(f"\n{sum(len(v) for v in table.values())} mutations across {len(table)} module(s)")
        return 0

    ok, msg = check_baseline()
    print(msg)
    if not ok:
        return 2
    print()

    # Back every file up before touching any of them, and restore ALL of them
    # at the end. Restoring only the file being mutated when an exception
    # fires would leave an earlier module mutated on disk — a corrupted tree
    # that looks perfectly fine.
    try:
        backups = {
            SRC / MODULE_PATHS[mod]: (SRC / MODULE_PATHS[mod]).read_text("utf-8") for mod in table
        }
    except OSError as exc:
        print(
            f"cannot back up every module before mutating — {exc}\n"
            f"expected the package at {SRC}; is this tools/ sitting at the "
            f"repository root, next to src/ and tests/?",
            file=sys.stderr,
        )
        return 2

    results: dict[str, dict[str, list]] = {}
    try:
        for mod, muts in table.items():
            path = SRC / MODULE_PATHS[mod]
            orig = backups[path]
            r = results.setdefault(mod, {"killed": [], "alive": [], "na": []})
            print(f"--- {mod}  ({len(muts)} mutations)")
            for m in muts:
                n = orig.count(m["anchor"])
                if n != 1:
                    r["na"].append((m["name"], f"anchor matched {n}x, expected 1"))
                    print(f"  [SKIP ] {m['name']:32s} anchor matched {n}x — not applied")
                    continue
                path.write_text(orig.replace(m["anchor"], m["replacement"], 1), "utf-8")
                green, out, _err = run_suite()
                if green:
                    r["alive"].append((m["name"], m["what"]))
                    print(f"  [ALIVE] {m['name']:32s} suite still green — NOT TESTED")
                else:
                    fails = failing_tests(out)
                    r["killed"].append(m["name"])
                    shown = ", ".join(fails[:3])
                    more = f" (+{len(fails) - 3})" if len(fails) > 3 else ""
                    print(
                        f"  [killed] {m['name']:31s} {len(fails)} test(s): {(shown + more)[:100]}"
                    )
                path.write_text(orig, "utf-8")
    finally:
        for p, text in backups.items():
            p.write_text(text, "utf-8")
            assert p.read_text("utf-8") == text, f"restore failed for {p}!"
        print(f"\n{len(backups)} module(s) restored byte-for-byte.")

    print("\n=== per-module tally ===")
    total_alive = 0
    total_applied = 0
    for mod in sorted(results):
        r = results[mod]
        total_alive += len(r["alive"])
        total_applied += len(r["killed"]) + len(r["alive"])
        applied = len(r["killed"]) + len(r["alive"])
        pct = f"{100 * len(r['killed']) // applied}%" if applied else "n/a"
        print(
            f"  {mod:20s} {len(r['killed']):2d} killed  {len(r['alive']):2d} alive  "
            f"{len(r['na']):2d} n/a   caught={pct}"
        )

    na = [(mod, n, w) for mod, r in results.items() for n, w in r["na"]]
    if na:
        print("\nNOT APPLICABLE — these never ran, so they are not evidence either way:")
        for mod, n, w in na:
            print(f"  {mod}/{n}: {w}")

    if total_alive:
        print("\nSURVIVING MUTANTS — each is a rule the suite does not actually check:")
        for mod in sorted(results):
            for n, w in results[mod]["alive"]:
                print(f"  - {mod}/{n}: {w}")

    # Report every problem, then pick the exit code. A survivor is the more
    # actionable finding, so it owns exit 1 even when anchors were also refused;
    # both were printed above and neither is exit 0.
    if na:
        applied_note = (
            "no mutation was applied at all"
            if total_applied == 0
            else f"{total_applied} of {total_applied + len(na)} mutations were applied"
        )
        print(
            f"\n{len(na)} mutation(s) never ran — {applied_note}. Those rules were "
            f"not measured by this run, and an unapplied mutation is not a passing "
            f"one: exiting {'1 (survivors take precedence)' if total_alive else '2'}. "
            f"Re-derive the stale anchors against the current source, then run again."
        )
    if total_alive:
        return 1
    if na:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
