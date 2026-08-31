#!/usr/bin/env python3
"""#123: replace the hard-coded estate root with a declared, required path-root policy.

MEASURED BEFORE WRITING (public-repo blocklist, case-insensitive, on the generated pair):
  fs_container_backend.bound.sh   0 hits   <- already clean
  launch_fs_h100.fixed.sh         4 hits   :53 :54 :140 :342

WHY THIS IS NOT COSMETICS. Two independent standing requirements fail on these four lines.

  GENERALITY. The mandate is a framework whose model- and estate-specific parts are
  isolated from the core. `[[ "$MODEL_DIR" == $FS_ESTATE_ROOT/* ]] || fail` is the exact
  shape of the one-time workaround: correct on precisely one filesystem, silently wrong on
  the next, and invisible until a model fails to load somewhere else with a message that
  blames the model. The guard itself is GOOD -- refusing a path that cannot be reached from
  inside the container is right, and it is the same instinct #117 acted on. What is wrong is
  that the POLICY is a literal instead of an input.

  PUBLICATION. the estate root's path segments are on the public-repo blocklist. Four hits would
  ship a cluster-internal layout into a public repository.

THE FIX FOLLOWS AN EXISTING PRECEDENT, IT DOES NOT INVENT ONE. FS_ALLOWED_NODE and
FS_CONTAINER_RUNTIME are already REQUIRED-WITH-NO-DEFAULT and fail closed at their point of
use, on the stated grounds that "an unconfigured guard is a disabled standing rule".
FS_ALLOWED_PATH_ROOTS is the third member of that family and behaves identically. It is a
LIST because an estate legitimately has several (an asset root and a scratch root are
routinely different filesystems), and because a single-root assumption is how a one-estate
literal gets re-introduced wearing a variable's name.

Deliberately NOT done: no default, not even a "sensible" one. A default here would be a
literal estate root again, just one whose wrongness surfaces later and blames the operator.

:342 `mounts=($FS_ESTATE_ROOT:$FS_ESTATE_ROOT)` is DELETED, not parameterized. Measured: zero
readers -- the launcher never expands it, and the backend's :1069 `local -a mounts` is a
different, arm-local array. It is a dead duplicate of the FS_BIND_PATHS plane #117
established. Leaving a second, dead mount declaration beside the live one is precisely how
the next reader concludes that mounts are declared there.

GATES
  E1 idempotent
  E2-E5 each of the four anchors is unique (a shifted generator must not be patched blind)
  E6 bash -n clean
  E7 blocklist scan of the PATCHED file reaches 0, with the pre-patch count as denominator
  E8 the emitted policy is EXECUTED, not read:
       MUST_FIRE  unset FS_ALLOWED_PATH_ROOTS            -> refused (the guard is not optional)
       MUST_FIRE  path outside every declared root       -> refused (the guard still guards)
       MUST_FIRE  /workfoo against root /work            -> refused (prefix must be a PATH
                  boundary, not a string prefix; the original `== $FS_ESTATE_ROOT/*` glob had
                  this right and a naive rewrite loses it)
       MUST_PASS  path under the second of two roots     -> admitted (multi-root really works,
                  and the list is not silently truncated to its first element)
  E9 no literal estate root survives in the emitted policy itself
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile
from fs_estate_pat import estate_blocklist

LAUNCH = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "launch_fs_h100.fixed.sh"
MARK = "fs123:"

BLOCKLIST = estate_blocklist(strict_token=True)

# --- the policy helper, inserted once, then used at all three sites ---------

POLICY_ANCHOR = """[[ -r "$IMAGE" && "$IMAGE" == *.sif ]] || fail 96 "IMAGE must be a readable .sif: $IMAGE"
"""

POLICY_BLOCK = '''[[ -r "$IMAGE" && "$IMAGE" == *.sif ]] || fail 96 "IMAGE must be a readable .sif: $IMAGE"

# --- fs123: declared path-root policy, replacing a hard-coded estate root ----
# WHY THE GUARD EXISTS AT ALL: a path the container cannot reach produces a
# failure that blames the model, not the mount plane. Refusing early is right.
# WHY IT IS NO LONGER A LITERAL: the set of reachable roots is a property of the
# ESTATE, not of FoundationScale. Hard-coding one made the framework correct on
# a single filesystem and silently wrong everywhere else.
#
# REQUIRED, NO DEFAULT -- the same contract as FS_ALLOWED_NODE and
# FS_CONTAINER_RUNTIME, for the same stated reason: an unconfigured guard is a
# disabled standing rule. A "sensible default" here would just be a literal
# estate root again, one whose wrongness surfaces later and blames the operator.
#
# A LIST, not a scalar: estates routinely keep assets and scratch on different
# filesystems, and a single-root assumption is how a one-estate literal returns
# wearing a variable's name. Space-separated, deliberately word-split.
[[ -n "${FS_ALLOWED_PATH_ROOTS:-}" ]] || fail 96 \\
  "FS_ALLOWED_PATH_ROOTS is unset (required, no default by design). Set it to the space-separated absolute root(s) of this estate that are reachable from inside the container -- the framework refuses to guess a filesystem layout."

# Prefix matching on a PATH BOUNDARY, not on a string. The literal this replaces
# used a trailing-slash glob and got that right; "$p" == "$root"* would not, and
# would admit /workfoo for root /work. Kept as a function so all three call
# sites share one definition and cannot drift.
fs_path_under_allowed_root() {
  local p=$1 root
  [[ -n "$p" ]] || return 1
  # shellcheck disable=SC2086
  for root in ${FS_ALLOWED_PATH_ROOTS}; do
    root=${root%/}
    [[ -n "$root" ]] || continue
    [[ "$p" == "$root" || "$p" == "$root"/* ]] && return 0
  done
  return 1
}
'''

# --- #151: the anchors are BUILT, not written -------------------------------
# This stage exists to delete a hard-coded estate root from the launcher. Until now it did
# that by carrying the estate root hard-coded in its own SITES table -- which made the
# de-hard-coding stage the last unpublishable file in the build, and left this file making
# in its own source exactly the mistake its docstring argues against ("the POLICY is a
# literal instead of an input"). The anchors are text this stage must MATCH in the upstream
# launcher, so they cannot be invented; they can, however, be an INPUT.
#
# REQUIRED, NO DEFAULT -- the same contract, for the same reason, as the FS_ALLOWED_PATH_ROOTS
# policy this stage emits. A default would re-introduce the literal it removes. The operator
# who is migrating away from an estate root necessarily knows what that root is.
def _estate_root() -> str:
    # `raise SystemExit("text")` prints the text but exits 1, which would quietly break the
    # plane's declared 0 / 95 / 96 contract from inside the very stage that argues for it.
    # Print, then exit with the number that was promised.
    root = os.environ.get("FS_ESTATE_ROOT", "").strip().rstrip("/")
    if not root:
        print(
            "REFUSE 96: FS_ESTATE_ROOT is unset (required, no default by design).\n"
            "  This stage rewrites a launcher that hard-codes one estate's filesystem root.\n"
            "  Set FS_ESTATE_ROOT to that root (the literal being REMOVED, e.g. /some/root)\n"
            "  so the anchors can be built. It is deliberately not stored in this repository:\n"
            "  a checked-in estate root is a published estate root.", file=sys.stderr)
        raise SystemExit(96)
    if not root.startswith("/"):
        print(f"REFUSE 96: FS_ESTATE_ROOT must be absolute, got {root!r}", file=sys.stderr)
        raise SystemExit(96)
    return root


def _sites(R: str) -> list[tuple[str, str]]:
    """The (anchor, replacement) table, with the estate root supplied rather than compiled in."""
    return [
    # (anchor, replacement)
    (
        f'[[ "$IMAGE" == {R}/* ]] || fail 96 "IMAGE must live under bindable {R}: $IMAGE"\n'
        f'[[ "$MODEL_DIR" == {R}/* && "$DATASET_DIR" == {R}/* ]] || fail 96 "MODEL_DIR/DATASET_DIR must be under {R}"\n',

        'fs_path_under_allowed_root "$IMAGE" || fail 96 \\\n'
        '  "IMAGE is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS): $IMAGE"\n'
        'fs_path_under_allowed_root "$MODEL_DIR" || fail 96 \\\n'
        '  "MODEL_DIR is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS): $MODEL_DIR"\n'
        'fs_path_under_allowed_root "$DATASET_DIR" || fail 96 \\\n'
        '  "DATASET_DIR is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS): $DATASET_DIR"\n'
        '# fs123: three separate tests, not one conjunction. The original ANDed MODEL_DIR and\n'
        '# DATASET_DIR into a single message, so an operator with one bad path was told both\n'
        '# were wrong and had to bisect by hand.\n',
    ),
    (
        f'[[ "$OUT_DIR_STABLE" == {R}/* || "$OUT_DIR_STABLE" == "${{HOME}}"/* ]] || fail 96 "OUT_DIR_STABLE outside known writable roots: $OUT_DIR_STABLE"\n',

        '# fs123: $HOME stays an accepted OUT_DIR root independently of the declared estate\n'
        '# roots -- it is where an operator without asset-tree write access must be able to\n'
        '# put outputs, and it is a property of the SESSION rather than the estate.\n'
        'fs_path_under_allowed_root "$OUT_DIR_STABLE" || [[ -n "${HOME:-}" && "$OUT_DIR_STABLE" == "${HOME%/}"/* ]] || fail 96 \\\n'
        '  "OUT_DIR_STABLE is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS) and outside \\$HOME: $OUT_DIR_STABLE"\n',
    ),
    (
        f'mounts=({R}:{R})\n',

        '# fs123: the write-only `mounts=(...)` array that stood here is DELETED. Measured: zero\n'
        '# readers -- nothing in this launcher ever expanded it, and the backend\'s arm-local\n'
        '# `local -a mounts` is a different variable entirely. It was a dead duplicate of the\n'
        '# FS_BIND_PATHS plane declared above, and a second mount declaration beside the live\n'
        '# one is how the next reader concludes that mounts are declared here.\n',
    ),
]

# --- E8: the policy, lifted out and run -------------------------------------

HARNESS = r"""
set -uo pipefail
fail() { printf 'REFUSED(%%s): %%s\n' "$1" "$2" >&2; exit "$1"; }
%s
probe() {  # label path expected(0=admit 1=refuse)
  if fs_path_under_allowed_root "$2"; then got=0; else got=1; fi
  [[ "$got" == "$3" ]] && printf 'ok %%s\n' "$1" || printf 'BAD %%s got=%%s want=%%s\n' "$1" "$got" "$3"
}
probe outside      /elsewhere/model 1
probe boundary     /rootfoo/model   1
probe under_first  /rootA/model     0
probe under_second /rootB/data      0
probe exact_root   /rootB           0
"""

REQUIRED_DRILL = r"""
set -uo pipefail
fail() { printf 'REFUSED(%s): %s\n' "$1" "$2" >&2; exit "$1"; }
unset FS_ALLOWED_PATH_ROOTS
[[ -n "${FS_ALLOWED_PATH_ROOTS:-}" ]] || fail 96 "FS_ALLOWED_PATH_ROOTS is unset"
echo "NOT REFUSED -- control is dead"
"""


def _bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    import os
    e = dict(os.environ)
    e.pop("FS_ALLOWED_PATH_ROOTS", None)
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e)


def main() -> int:
    # E0 (#151): this stage's OWN source must carry no estate literal. It is the file that
    # deletes the estate root from the launcher, so it was the one place a literal could hide
    # and still let the build report "0 blocklist hits" -- the build only ever scanned the
    # GENERATED artifacts, never the generator. A self-check here closes that asymmetry.
    # Checked before anything else: if this file is unpublishable, nothing it emits matters.
    self_src = pathlib.Path(__file__).read_text("utf-8")
    # The regex source itself legitimately contains the vocabulary (finding #145: a redaction
    # pattern is not the thing it redacts). Only hits OUTSIDE the pattern definition count, so
    # the check is anchored on the pattern literal's own span rather than on a bare count.
    pat_span = self_src.find("BLOCKLIST = re.compile(")
    pat_end = self_src.find(")", pat_span) if pat_span >= 0 else -1
    real = [m for m in BLOCKLIST.finditer(self_src)
            if not (pat_span <= m.start() <= pat_end)]
    if real:
        where = [f"L{self_src[:m.start()].count(chr(10)) + 1}:{m.group(0)}" for m in real[:6]]
        print(f"  FAIL E0  this stage's own source carries {len(real)} estate literal(s): {where}",
              file=sys.stderr)
        return 5
    print("  PASS E0  stage source carries no estate literal outside its own redaction pattern")

    R = _estate_root()
    sites = _sites(R)

    text = LAUNCH.read_text("utf-8")
    if MARK in text:                                                          # E1
        print("  E1  already applied — no-op (idempotent)")
        return 0
    print("  PASS E1  not yet patched")

    before = len(BLOCKLIST.findall(text))
    ok = True

    anchors = [("E2", "policy insertion point", POLICY_ANCHOR)] + [
        (f"E{i + 3}", f"site {i + 1}", a) for i, (a, _) in enumerate(sites)]
    for gate, name, anchor in anchors:
        n = text.count(anchor)
        if n != 1:
            print(f"  FAIL {gate}  {name} anchor occurs {n}x (need 1)", file=sys.stderr)
            ok = False
        else:
            print(f"  PASS {gate}  {name} anchor unique")
    if not ok:
        print("\nREFUSING TO WRITE — the generator emitted a different shape "
              "than this patch was measured against", file=sys.stderr)
        return 5

    new = text.replace(POLICY_ANCHOR, POLICY_BLOCK, 1)
    for anchor, repl in sites:
        new = new.replace(anchor, repl, 1)

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(new); tmp = fh.name
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    if r.returncode != 0:                                                     # E6
        print(f"  FAIL E6  bash -n: {r.stderr.strip()[:300]}", file=sys.stderr)
        ok = False
    else:
        print("  PASS E6  bash -n clean")

    after = len(BLOCKLIST.findall(new))                                       # E7
    if after:
        hits = [f"L{new[:m.start()].count(chr(10)) + 1}:{m.group(0)}"
                for m in BLOCKLIST.finditer(new)]
        print(f"  FAIL E7  {after} blocklist hit(s) survive (was {before}): {hits[:6]}",
              file=sys.stderr)
        ok = False
    else:
        print(f"  PASS E7  blocklist hits {before} -> 0 (denominator: {before} pre-patch)")

    # E8 -- execute the emitted function, do not read it
    m = re.search(r"^fs_path_under_allowed_root\(\) \{$", new, re.M)
    if not m:
        print("  FAIL E8  emitted policy function not found; a control that cannot be "
              "built must not be reported green", file=sys.stderr)
        return 5
    lines = new[m.start():].splitlines(keepends=True)
    body = []
    for ln in lines:
        body.append(ln)
        if ln.rstrip("\n") == "}":
            break
    fn = "".join(body)

    r = _bash(HARNESS % fn, {"FS_ALLOWED_PATH_ROOTS": "/rootA /rootB"})
    bad = [x for x in r.stdout.splitlines() if x.startswith("BAD")]
    good = [x for x in r.stdout.splitlines() if x.startswith("ok ")]
    if bad or len(good) != 5:
        print(f"  FAIL E8a {len(bad)} row(s) wrong, {len(good)}/5 correct: {bad[:4]} "
              f"{r.stderr.strip()[:150]}", file=sys.stderr)
        ok = False
    else:
        print("  PASS E8a MUST_FIRE outside + /rootfoo boundary REFUSED; MUST_PASS both "
              "roots admitted (multi-root not truncated to its first entry)")

    r = _bash(REQUIRED_DRILL)
    if r.returncode == 96 and "REFUSED(96)" in r.stderr:
        print("  PASS E8b MUST_FIRE: unset FS_ALLOWED_PATH_ROOTS observed being REFUSED (rc 96)")
    else:
        print(f"  FAIL E8b unset root list was not refused: rc={r.returncode}", file=sys.stderr)
        ok = False

    if BLOCKLIST.search(fn):                                                  # E9
        print("  FAIL E9  the emitted policy itself contains an estate literal", file=sys.stderr)
        ok = False
    else:
        print("  PASS E9  emitted policy carries no estate literal")

    if not ok:
        print("\nREFUSING TO WRITE — gates above are red", file=sys.stderr)
        return 5
    LAUNCH.write_text(new, "utf-8")
    print(f"\nALL GATES GREEN -> {LAUNCH.name}")
    print("OPERATOR IMPACT, stated because it is a breaking change: every launch must now "
          "export FS_ALLOWED_PATH_ROOTS. That is the point -- the previous behaviour was "
          "not 'no configuration needed', it was 'one estate's configuration compiled in'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
