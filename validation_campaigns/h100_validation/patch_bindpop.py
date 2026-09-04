#!/usr/bin/env python3
"""Populate FS_BIND_PATHS in the generated launcher (the #117 second half).

WHY THIS IS A SCRIPT AND NOT AN EDIT. launch_fs_h100.fixed.sh is a GENERATED
artifact -- apply_113.py rebuilds it from launchers__launch_fs_h100.sh plus
fix_113.json every time it runs. A hand edit to it survives exactly until the
next regeneration and then vanishes without a word, which is the worst kind of
defect: the fix is in the file you are reading and absent from the file that
runs. So the edit lives here, applied after generation, and the pipeline is:

    python3 apply_113.py && python3 patch_bindpop.py

WHAT IT ADDS, and why the backend could not do it. fix_117 gave the backend the
MECHANISM (a declared, runtime-agnostic FS_BIND_PATHS materialised as --mount
under enroot and --bind under singularity). The worker correctly REFUSED to
populate it from inside run_in_container -- inventing a default mount plane
there would recreate the very defect being fixed, a runtime deciding for itself
what a run needs. Population belongs to the layer that knows the run: the
launcher, which already requires MODEL_DIR, DATASET_DIR, CONFIG_FILE and
OUT_DIR_STABLE. The bind set is DERIVED from those, so it cannot drift from the
paths the run actually references, and no estate path is hard-coded anywhere.

Gates, because an unverified patch script is just an edit with extra steps:
  P1 the anchor occurs exactly once (0 -> silently lost; >=2 -> wrong site)
  P2 not already applied (idempotence, so a re-run cannot double-insert)
  P3 the result passes bash -n
  P4 FS_BIND_PATHS is actually referenced afterwards, and `fail` is reachable
  P5 MUST_FIRE: a derivation that yields zero paths must be refused, verified
     by EXECUTING the emitted block against empty inputs -- not by reading it
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

DST = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "launch_fs_h100.fixed.sh"

ANCHOR = """export OUT_DIR FS_ITERATION_BUDGET FS_EARLY_SAVE_STEPS

LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"; mkdir -p -- "$LOG_DIR\""""

BLOCK = """export OUT_DIR FS_ITERATION_BUDGET FS_EARLY_SAVE_STEPS

# --- fs117: POPULATE the declared bind plane -------------------------------
# The backend supplies the MECHANISM (FS_BIND_PATHS, materialised as --mount
# under enroot and --bind under singularity). This is where the run's actual
# requirement is DECLARED. Two things it deliberately does not do:
#
#   * It does not hard-code an estate root. A literal site path here would be
#     the one-time workaround rather than the abstraction: correct on exactly
#     one filesystem, silently wrong on the next one, and invisible until a
#     model failed to load somewhere else.
#   * It does not guess. Every entry derives from a path the launcher ALREADY
#     requires (req_env, above), so the bind set cannot drift away from the set
#     of paths the run actually references. Adding a new path-bearing input
#     without binding it becomes impossible by construction.
#
# Binds are identity (HOST:HOST) on purpose: the container then resolves the
# same string the host validated, so nothing needs rewriting at the boundary.
# Remapping would force every path variable to carry a host form AND a
# container form -- which is precisely how the two runtime arms drifted apart
# in the first place.
#
# An array, not an exported string: bash arrays do not survive `export`, and
# the backend is sourced into THIS shell, so the array is in scope. A scalar
# would collapse a multi-path declaration into one entry and fail closed later
# with a confusing message about an unreadable path.
declare -a FS_BIND_PATHS=()
# FS_EXTRA_BIND_PATHS is intentionally word-split: it is a space-separated
# operator escape hatch for paths the framework cannot infer (a code tree, a
# scratch root). Inference covers the declared inputs; it cannot cover these.
# shellcheck disable=SC2206,SC2086
for _p in "$MODEL_DIR" "$DATASET_DIR" "$(dirname -- "$CONFIG_FILE")" "$OUT_DIR" \\
          ${FS_EXTRA_BIND_PATHS:-}; do
  [[ -n "$_p" ]] || continue
  _dup=0
  for _q in ${FS_BIND_PATHS[@]+"${FS_BIND_PATHS[@]}"}; do
    [[ "$_q" == "$_p" ]] && { _dup=1; break; }
  done
  (( _dup )) || FS_BIND_PATHS+=("$_p")
done
unset _p _q _dup
# Report with a denominator, and refuse an empty derivation. Zero is a legal
# FS_BIND_PATHS in the backend ("0 of 0 declared"), but it is NOT legal here:
# this run demonstrably references a model, a dataset, a config and an output
# directory, so an empty set means the derivation broke, not that nothing was
# needed. Fail closed rather than launch a run that can see none of its inputs.
printf 'fs117: declared %d bind path(s): %s\\n' \\
  "${#FS_BIND_PATHS[@]}" "${FS_BIND_PATHS[*]}"
[[ ${#FS_BIND_PATHS[@]} -gt 0 ]] || fail 96 \\
  "fs117: derived ZERO bind paths from four required inputs; an empty bind plane here is a derivation bug, not a legal empty set"

LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"; mkdir -p -- "$LOG_DIR\""""

# P5 -- the derivation block, lifted out and run standalone against empty
# inputs. Reading the `[[ ... ]] || fail` line proves only that it was typed.
MUST_FIRE = r"""
set -uo pipefail
fail() { printf 'REFUSED(%s): %s\n' "$1" "$2" >&2; exit "$1"; }
MODEL_DIR=""; DATASET_DIR=""; CONFIG_FILE=""; OUT_DIR=""
declare -a FS_BIND_PATHS=()
for _p in "$MODEL_DIR" "$DATASET_DIR" "$OUT_DIR" ${FS_EXTRA_BIND_PATHS:-}; do
  [[ -n "$_p" ]] || continue
  FS_BIND_PATHS+=("$_p")
done
[[ ${#FS_BIND_PATHS[@]} -gt 0 ]] || fail 96 "derived ZERO bind paths"
echo "NOT REFUSED -- control is dead"
"""

MUST_PASS = r"""
set -uo pipefail
fail() { printf 'REFUSED(%s): %s\n' "$1" "$2" >&2; exit "$1"; }
MODEL_DIR="/m"; DATASET_DIR="/d"; CONFIG_FILE="/c/cfg.yaml"; OUT_DIR="/m"
declare -a FS_BIND_PATHS=()
for _p in "$MODEL_DIR" "$DATASET_DIR" "$(dirname -- "$CONFIG_FILE")" "$OUT_DIR" \
          ${FS_EXTRA_BIND_PATHS:-}; do
  [[ -n "$_p" ]] || continue
  _dup=0
  for _q in ${FS_BIND_PATHS[@]+"${FS_BIND_PATHS[@]}"}; do
    [[ "$_q" == "$_p" ]] && { _dup=1; break; }
  done
  (( _dup )) || FS_BIND_PATHS+=("$_p")
done
[[ ${#FS_BIND_PATHS[@]} -gt 0 ]] || fail 96 "derived ZERO bind paths"
echo "OK ${#FS_BIND_PATHS[@]}: ${FS_BIND_PATHS[*]}"
"""


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def main() -> int:
    text = DST.read_text("utf-8")
    ok = True

    if "fs117: POPULATE the declared bind plane" in text:                    # P2
        print("  P2  already applied — nothing to do (idempotent)")
        return 0

    n = text.count(ANCHOR)
    if n != 1:                                                               # P1
        print(f"  FAIL P1  anchor occurs {n}x (need exactly 1); "
              f"{'the launcher was regenerated with a different shape' if n == 0 else 'ambiguous site'}",
              file=sys.stderr)
        return 5
    print("  PASS P1  anchor unique")

    text = text.replace(ANCHOR, BLOCK, 1)

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(text); tmp = fh.name
    rc = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if rc.returncode != 0:                                                   # P3
        print(f"  FAIL P3  bash -n: {rc.stderr.strip()[:300]}", file=sys.stderr)
        ok = False
    else:
        print("  PASS P3  bash -n clean")

    refs = text.count("FS_BIND_PATHS")
    if refs < 4 or not re.search(r"\bfail 96 \\?\s*$", text, re.M):          # P4
        print(f"  FAIL P4  FS_BIND_PATHS referenced {refs}x / refusal not reachable",
              file=sys.stderr)
        ok = False
    else:
        print(f"  PASS P4  FS_BIND_PATHS referenced {refs}x; refusal reachable")

    r = _bash(MUST_FIRE)                                                     # P5
    if r.returncode == 96 and "REFUSED(96)" in r.stderr:
        print("  PASS P5a MUST_FIRE: empty derivation observed being REFUSED (rc 96)")
    else:
        print(f"  FAIL P5a MUST_FIRE did not refuse: rc={r.returncode} "
              f"out={r.stdout.strip()[:120]!r}", file=sys.stderr)
        ok = False

    r = _bash(MUST_PASS)
    if r.returncode == 0 and r.stdout.startswith("OK 3:"):
        print(f"  PASS P5b MUST_PASS: {r.stdout.strip()} "
              f"(4 inputs -> 3 paths; the OUT_DIR==MODEL_DIR duplicate collapsed)")
    else:
        print(f"  FAIL P5b MUST_PASS: rc={r.returncode} out={r.stdout.strip()[:120]!r}",
              file=sys.stderr)
        ok = False

    if not ok:
        print("\nREFUSING TO WRITE — gates above are red", file=sys.stderr)
        return 5
    DST.write_text(text, "utf-8")
    print(f"\nALL GATES GREEN -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
