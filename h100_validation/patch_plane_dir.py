#!/usr/bin/env python3
"""Patch the H100 launcher's plane lookup for finding #142.

The launcher carried ``#SBATCH`` directives but located every sibling from
``${BASH_SOURCE[0]}``. A real sbatch submission did not execute that file in
place: Slurm copied it to ``/var/spool/slurmd/jobNNNNN`` and renamed the copy
``slurm_script``. In that directory no siblings exist, whether or not the
legacy backend filename happens to exist somewhere else.

This stage replaces the launcher's first sibling lookup with a four-step
resolver: an optional operator override, the direct in-place case, a named
workload-manager dispatch verified against the real original path, then a
misdirection-free refusal. The stage must leave no artifact behind unless its
static gates, byte-idempotence gate, syntax gate, and three executable controls
are green. ``FS_PLANE_DIR`` is exported because every spawned hop would
otherwise have to solve sbatch staging again.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import stat
import subprocess
import tempfile

TARGET = pathlib.Path("h100") / "gen" / "launch_fs_h100.fixed.sh"
BACKEND_ARTIFACT = "fs_container_backend.bound.sh"
LEGACY_BACKEND = "fs_container_backend.sh"

RESOLVER_BEGIN = "# fs142 plane resolver: BEGIN (self-contained; backend helpers do not exist yet)"
RESOLVER_END = "# fs142 plane resolver: END"

OLD_BLOCK = r'''SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BACKEND="$SCRIPT_DIR/fs_container_backend.sh"
if [[ ! -r "$BACKEND" ]]; then
  printf 'FATAL: fs_container_backend.sh not readable at %s\n' "$BACKEND" >&2
  exit 96
fi
# shellcheck source=launchers/fs_container_backend.sh
source "$BACKEND"
'''

# Exact multiplicity matters before a byte is changed. Finding a second old
# lookup and editing only the first is how a generated tree acquires two
# different answers for the same plane, the #142 drift this stage removes.
REWRITE_ANCHORS = (
    (
        "naive SCRIPT_DIR assignment",
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"',
    ),
    ("legacy backend assignment", 'BACKEND="$SCRIPT_DIR/fs_container_backend.sh"'),
    (
        "legacy unreadable-backend guard",
        # ESCAPING, one level exactly. The launcher holds the two characters backslash-n
        # inside a single-quoted printf format; in a non-raw Python string that is "\\n".
        # The generated anchor carried "\\\\n" -- backslash, backslash, n -- so PRE2 read
        # anchors=[1,1,0,1,1] and the stage refused. Note OLD_BLOCK above gets this right
        # because it is an r-string; this tuple is not, and the two must agree or the
        # multiplicity check and the replacement disagree about the same four lines.
        'if [[ ! -r "$BACKEND" ]]; then\n'
        "  printf 'FATAL: fs_container_backend.sh not readable at %s\\n' \"$BACKEND\" >&2\n"
        "  exit 96\n"
        "fi",
    ),
    ("legacy shellcheck source", "# shellcheck source=launchers/fs_container_backend.sh"),
    ("backend source site", 'source "$BACKEND"'),
)

RESOLVER_BLOCK = r'''# fs142 plane resolver: BEGIN (self-contained; backend helpers do not exist yet)
# Finding #142 was measured on an 8xH100 sbatch job: sbatch stages and renames
# the submitted file, so ${BASH_SOURCE[0]}/. is the spool directory in exactly
# the mode this launcher exists for. SLURM_SUBMIT_DIR was only the submit-time
# CWD and pointed at the plane's parent, so it is deliberately not used here.
# This prologue runs before the backend is parsed. fs_die and every other
# backend helper do not exist yet; Bash builtins, coreutils, and scontrol are
# the complete dependency budget.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BACKEND_NAME="fs_container_backend.bound.sh"
FS142_PLANE_OVERRIDE="${FS_PLANE_DIR:-}"
FS142_PLANE_DIR=""
FS142_PLANE_STEP=0
FS142_CANDIDATE=""
FS142_LAST_BACKEND_DIR="$SCRIPT_DIR"
FS142_STEP1_RESULT="not reached"
FS142_STEP2_RESULT="not reached"
FS142_STEP3_RESULT="not reached"
FS142_STEP4_RESULT="not reached"
FS142_WLM_COMMAND=""
FS142_WLM_KIND="${FS_ALLOCATION:-${SLURM_JOB_ID:+slurm}}"

# Keep the WLM lookup behind a named dispatch. The second workload manager is
# easier to add here than in a launcher whose estate logic has grown into a
# chain of hard-coded Slurm conditionals.
fs142_wlm_command_path() {
  local wlm_kind="$1"
  local batch_info=""
  local -a command_match=()
  case "$wlm_kind" in
  slurm)
    [[ -n "${SLURM_JOB_ID:-}" ]] || return 1
    command -v scontrol >/dev/null 2>&1 || return 1
    batch_info="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)" || return 1
    if [[ "$batch_info" =~ (^|[[:space:]])Command=([^[:space:]]+) ]]; then
      command_match=("${BASH_REMATCH[@]}")
      [[ "${command_match[2]}" == /* ]] || return 1
      printf '%s\n' "${command_match[2]}"
      return 0
    fi
    return 1
    ;;
  *)
    return 127
    ;;
  esac
}

# FS_PLANE_DIR is an optional override, not a required-no-default setting:
# step 2 must keep a direct `bash launch...` invocation working without a new
# operator variable.
FS142_CANDIDATE="$FS142_PLANE_OVERRIDE"
if [[ -n "$FS142_CANDIDATE" ]]; then
  FS142_LAST_BACKEND_DIR="$FS142_CANDIDATE"
  if [[ -r "$FS142_CANDIDATE/$BACKEND_NAME" ]]; then
    if FS142_CANDIDATE="$(cd -- "$FS142_CANDIDATE" >/dev/null 2>&1 && pwd -P)"; then
      FS142_PLANE_DIR="$FS142_CANDIDATE"
      FS142_PLANE_STEP=1
      FS142_STEP1_RESULT="FS_PLANE_DIR=$FS142_PLANE_DIR; backend verified"
    else
      FS142_STEP1_RESULT="FS_PLANE_DIR=$FS142_CANDIDATE; physical directory could not be resolved"
    fi
  else
    FS142_STEP1_RESULT="FS_PLANE_DIR=$FS142_CANDIDATE; $BACKEND_NAME was not readable there"
  fi
else
  FS142_STEP1_RESULT="FS_PLANE_DIR=<unset>; no candidate directory was tested"
fi

# In-place execution remains first-class. `bash launch_fs_h100.fixed.sh`, an
# interactive srun, and unmeasured work managers that do not stage scripts all
# reach this answer without relying on Slurm-specific state.
if [[ -z "$FS142_PLANE_DIR" ]]; then
  FS142_CANDIDATE="$SCRIPT_DIR"
  FS142_LAST_BACKEND_DIR="$FS142_CANDIDATE"
  if [[ -r "$FS142_CANDIDATE/$BACKEND_NAME" ]]; then
    FS142_PLANE_DIR="$FS142_CANDIDATE"
    FS142_PLANE_STEP=2
    FS142_STEP2_RESULT="SCRIPT_DIR=$FS142_PLANE_DIR; backend verified"
  else
    FS142_STEP2_RESULT="SCRIPT_DIR=$FS142_CANDIDATE; $BACKEND_NAME was not readable there"
  fi
else
  FS142_STEP2_RESULT="not tested; step 1 resolved the plane"
fi

# A WLM answer is still only a claim. The measured #142 job supplied the
# original `/home/.../probe_sub/planeprobe.sh` through Command=; accepting that
# answer without checking the sibling beside it would turn a WLM defect into a
# falsely "resolved" plane.
if [[ -z "$FS142_PLANE_DIR" ]]; then
  if [[ -n "$FS142_WLM_KIND" ]]; then
    if FS142_WLM_COMMAND="$(fs142_wlm_command_path "$FS142_WLM_KIND")"; then
      FS142_CANDIDATE="$(dirname -- "$FS142_WLM_COMMAND")"
      FS142_LAST_BACKEND_DIR="$FS142_CANDIDATE"
      if [[ -r "$FS142_CANDIDATE/$BACKEND_NAME" ]]; then
        if FS142_CANDIDATE="$(cd -- "$FS142_CANDIDATE" >/dev/null 2>&1 && pwd -P)"; then
          FS142_PLANE_DIR="$FS142_CANDIDATE"
          FS142_PLANE_STEP=3
          FS142_STEP3_RESULT="$FS142_WLM_KIND returned $FS142_WLM_COMMAND; backend verified in $FS142_PLANE_DIR"
        else
          FS142_STEP3_RESULT="$FS142_WLM_KIND returned $FS142_WLM_COMMAND; physical directory could not be resolved"
        fi
      else
        FS142_STEP3_RESULT="$FS142_WLM_KIND returned $FS142_WLM_COMMAND; $BACKEND_NAME was not readable in $FS142_CANDIDATE"
      fi
    else
      FS142_STEP3_RESULT="$FS142_WLM_KIND returned no original script path"
    fi
  else
    FS142_STEP3_RESULT="no known workload manager was detected, so no script path was returned"
  fi
else
  FS142_STEP3_RESULT="not tested; an earlier step resolved the plane"
fi

# Finding #139 already paid for this lesson: a technically true refusal that
# sends the reader to the wrong property is itself the defect. Here the file
# existed while the directory was wrong, so the refused message names the
# attempted answers before naming the remedy.
if [[ -z "$FS142_PLANE_DIR" ]]; then
  FS142_STEP4_RESULT="refused: no verified backend after steps 1 through 3"
  printf 'FATAL[142]: resolution step 1 (FS_PLANE_DIR), returned: %s\n' "$FS142_STEP1_RESULT" >&2
  printf 'FATAL[142]: resolution step 2 (SCRIPT_DIR), returned: %s\n' "$FS142_STEP2_RESULT" >&2
  printf 'FATAL[142]: resolution step 3 (workload-manager dispatch), returned: %s\n' "$FS142_STEP3_RESULT" >&2
  printf 'FATAL[142]: resolution step 4 (refusal), returned: %s\n' "$FS142_STEP4_RESULT" >&2
  printf 'FATAL[142]: directory actually searched last: %s\n' "$FS142_LAST_BACKEND_DIR" >&2
  printf 'FATAL[142]: hypothesis: this looks like a workload-manager-staged copy of the script, not the original\n' >&2
  printf 'FATAL[142]: remedy: set and export FS_PLANE_DIR=<directory containing %s> before submitting\n' "$BACKEND_NAME" >&2
  exit 96
fi

FS142_STEP4_RESULT="accepted verified backend at $FS142_PLANE_DIR/$BACKEND_NAME"
case "$FS142_PLANE_STEP" in
  1) FS142_RESOLUTION_SOURCE="FS_PLANE_DIR" ;;
  2) FS142_RESOLUTION_SOURCE="SCRIPT_DIR" ;;
  3) FS142_RESOLUTION_SOURCE="$FS142_WLM_KIND" ;;
  *) FS142_RESOLUTION_SOURCE="unknown" ;;
esac
printf 'plane directory resolved: %s (resolution step %s: %s)\n' \
  "$FS142_PLANE_DIR" "$FS142_PLANE_STEP" "$FS142_RESOLUTION_SOURCE"

# Later jobs and every child process inherit the measured answer instead of
# independently repeating the sbatch-staging mistake.
export FS_PLANE_DIR="$FS142_PLANE_DIR"
BACKEND="$FS142_PLANE_DIR/$BACKEND_NAME"
# shellcheck source=fs_container_backend.bound.sh
source "$BACKEND"
# fs142 plane resolver: END
'''

_RESOLVER_RE = re.compile(
    re.escape(RESOLVER_BEGIN) + r".*?" + re.escape(RESOLVER_END) + r"\n",
    re.DOTALL,
)

HARNESS_PRELUDE = "#!/usr/bin/env bash\nset -Eeuo pipefail\nIFS=$'\\n\\t'\numask 027\n\n"

CONTROL_LABELS = (
    "BEHAVIOUR P1 MUST_PASS co-located siblings resolve by step 2",
    "BEHAVIOUR P2 MUST_PASS FS_PLANE_DIR resolves elsewhere by step 1",
    "BEHAVIOUR F1 MUST_FIRE isolated spool shape refuses by step 4",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    target = pathlib.Path(args.root) / TARGET

    gates: list[tuple[str, str, str]] = []

    try:
        original = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        gates.append(("PRE1 target exists and is readable", "FAIL",
                      f"0/1 target(s) readable: {str(exc)[:120]}"))
        _report(gates)
        print("refused to write; 0/1 targets were readable")
        return 2

    gates.append(("PRE1 target exists and is readable", "PASS",
                  f"1/1 target(s), {len(original.encode('utf-8'))} byte(s)"))

    old_state, resolver_blocks, anchor_counts = _patch_state(original)
    anchor_total = len(anchor_counts)
    state_ok = old_state != (resolver_blocks == 1)
    if old_state:
        state_note = f"mode=naive; {anchor_total}/{anchor_total} anchors once; 1/1 old block(s)"
    elif resolver_blocks == 1:
        state_note = "mode=generated; 1/1 resolver block(s); prior result retained"
    else:
        state_note = (
            f"anchors={anchor_counts}; old blocks={original.count(OLD_BLOCK)}; "
            f"resolver blocks={resolver_blocks}"
        )
    gates.append(("PRE2 rewrite anchors identify exactly one recognized state",
                  "PASS" if state_ok else "FAIL",
                  state_note))
    if not state_ok:
        _report(gates)
        print("refused to write; a multiply anchored patch would pick one occurrence silently")
        return 2

    try:
        patched = _rewrite(original)
        resolver_text = _extract_resolver(patched)
    except ValueError as exc:
        gates.append(("A resolver can be extracted from the verified candidate", "FAIL",
                      f"0/1 resolver block(s): {exc}"))
        _report(gates)
        print("refused to write; the candidate resolver is not a single extractable block")
        return 2

    resolver_ok = resolver_text is not None
    positions: list[int] = []
    dispatch_count = 0
    if resolver_text is not None:
        markers = (
            'FS142_PLANE_OVERRIDE=',
            '  FS142_CANDIDATE="$SCRIPT_DIR"',
            'fs142_wlm_command_path "$FS142_WLM_KIND"',
            'FS142_STEP4_RESULT="refused: no verified backend after steps 1 through 3"',
        )
        positions = [resolver_text.find(marker) for marker in markers]
        dispatch_count = resolver_text.count("\n  slurm)")
    ordered = (
        resolver_ok
        and all(position >= 0 for position in positions)
        and positions == sorted(positions)
        and len(set(positions)) == 4
        and dispatch_count == 1
    )
    gates.append(("A resolver is present and all four steps are ordered and reachable",
                  "PASS" if ordered else "FAIL",
                  f"{1 if resolver_ok else 0}/1 resolver block(s); "
                  f"{sum(position >= 0 for position in positions)}/4 step marker(s); "
                  f"{dispatch_count}/1 dispatch entr(y/ies)"))

    names: list[str] = []
    assignments: list[str] = []
    remedy_refs: list[str] = []
    if resolver_text is not None:
        names = re.findall(r'^BACKEND_NAME="([^"\n]+)"$', resolver_text, re.MULTILINE)
        assignments = re.findall(r'^BACKEND="([^"\n]+)"$', resolver_text, re.MULTILINE)
        remedy_refs = re.findall(
            r"FATAL\[142\]: remedy:[^\n]*\\n' (\S+) >&2", resolver_text
        )
    assignment_suffix = assignments[0].rsplit("/", 1)[-1] if len(assignments) == 1 else None
    remedy_suffix = remedy_refs[0].strip('"') if len(remedy_refs) == 1 else None
    pair_ok = (
        len(names) == 1
        and names[0] == BACKEND_ARTIFACT
        and len(assignments) == 1
        and len(remedy_refs) == 1
        and assignment_suffix is not None
        and assignment_suffix == remedy_suffix
    )
    gates.append(("B backend artifact and refusal reference resolve the same pair",
                  "PASS" if pair_ok else "FAIL",
                  f"name={names}; assignment/refusal references=1/1 required; "
                  f"pair match={assignment_suffix == remedy_suffix}"))

    bash_path = shutil.which("bash")
    syntax_ok = False
    behaviour_hold_reason = ""
    if bash_path is None:
        gates.append(("C bash -n clean on the patched candidate", "UNMEASURED",
                      "0/1 candidate(s): no bash executable on PATH"))
        behaviour_hold_reason = "bash executable was unavailable for all three controls"
    else:
        syntax_tmp: str | None = None
        try:
            fd, syntax_tmp = tempfile.mkstemp(
                dir=str(target.parent), prefix=".fs142-candidate-", suffix=".sh"
            )
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(patched)
            proc = subprocess.run(
                [bash_path, "-n", syntax_tmp],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            syntax_ok = proc.returncode == 0
            gates.append(("C bash -n clean on the patched candidate", 
                          "PASS" if syntax_ok else "FAIL",
                          f"1/1 candidate(s); rc={proc.returncode}; "
                          f"stderr: {proc.stderr.strip()[:120] or '<empty>'}"))
            if not syntax_ok:
                behaviour_hold_reason = "syntax gate went red before executable controls were trusted"
        except (OSError, subprocess.SubprocessError) as exc:
            gates.append(("C bash -n clean on the patched candidate", "UNMEASURED",
                          f"0/1 candidate(s): {str(exc)[:120]}"))
            behaviour_hold_reason = "syntax-capable temporary candidate could not be built"
        finally:
            if syntax_tmp is not None:
                pathlib.Path(syntax_tmp).unlink(missing_ok=True)

    exports = re.findall(r'^export FS_PLANE_DIR="\$FS142_PLANE_DIR"$', patched, re.MULTILINE)
    bare_assignments = re.findall(r"^FS_PLANE_DIR=", patched, re.MULTILINE)
    gates.append(("D resolved plane directory is exported, not merely assigned",
                  "PASS" if len(exports) == 1 and not bare_assignments else "FAIL",
                  f"{len(exports)}/1 export line(s); {len(bare_assignments)} bare assignment(s)"))

    # A denominator for the resubmit rewrite. Gate E proves the residual is zero, which is a
    # claim about ABSENCE and is equally satisfied by a launcher that never had a submit chain
    # at all -- the all([]) shape. E1 states how many sites were actually converted, so "clean"
    # and "empty" stop looking alike.
    resubmit_before = original.count(RESUBMIT_OLD)
    resubmit_after = patched.count(RESUBMIT_NEW)
    gates.append(("E1 submit-chain self-reference re-pointed at the resolved plane",
                  "PASS" if resubmit_before > 0 and resubmit_after == resubmit_before
                  else ("UNMEASURED" if resubmit_before == 0 else "FAIL"),
                  f"{resubmit_after}/{resubmit_before} site(s) converted "
                  f"{RESUBMIT_OLD} -> {RESUBMIT_NEW}"))

    outside_resolver = patched.replace(resolver_text, "", 1) if resolver_text is not None else patched
    outside_counts = (
        outside_resolver.count("BASH_SOURCE"),
        len(re.findall(r"\$(?:\{)?SCRIPT_DIR(?:\})?\s*/", outside_resolver)),
        patched.count(LEGACY_BACKEND),
    )
    gates.append(("E no surviving sibling lookup bypasses the resolver",
                  "PASS" if all(count == 0 for count in outside_counts) else "FAIL",
                  f"violations={outside_counts}; 3/3 classes required zero"))

    try:
        reapplied = _rewrite(patched)
        idempotent = reapplied == patched
        delta = 0 if idempotent else len(reapplied.encode("utf-8")) - len(patched.encode("utf-8"))
        gates.append(("IDEM re-application is a byte-exact no-op",
                      "PASS" if idempotent else "FAIL",
                      f"2/2 applications measured; byte delta={delta}"))
    except ValueError as exc:
        gates.append(("IDEM re-application is a byte-exact no-op", "FAIL",
                      f"1/2 applications: {str(exc)[:120]}"))

    # These controls execute the text cut out of the candidate, rather than
    # greping for it. The isolated-directory control is deliberately red:
    # without observing exit 96 and the FS_PLANE_DIR remedy, fail-closed is an
    # unmeasured claim rather than a property of the resolver.
    if syntax_ok and bash_path is not None and resolver_text is not None:
        gates.extend(_behaviour_controls(bash_path, resolver_text))
    else:
        for label in CONTROL_LABELS:
            gates.append((label, "UNMEASURED",
                          f"0/1 control(s): {behaviour_hold_reason or 'resolver extraction failed'}"))

    rc = _finish(gates)
    if rc != 0:
        _report(gates)
        print("refused to write; every gate must be green before the artifact changes")
        return rc

    if patched == original:
        gates.append(("POST installed artifact matches the verified candidate", "PASS",
                      "0 write(s); existing bytes match exactly"))
        _report(gates)
        print(f"already patched {target}; re-application was a byte-exact no-op")
        return 0

    try:
        mode = stat.S_IMODE(target.stat().st_mode)
        _atomic_write(target, patched, mode)
        persisted = target.read_text(encoding="utf-8")
        if persisted != patched:
            gates.append(("POST installed artifact matches the verified candidate", "FAIL",
                          f"persisted delta={len(persisted) - len(patched)} character(s)"))
            _report(gates)
            return 2
        gates.append(("POST installed artifact matches the verified candidate", "PASS",
                      f"1/1 atomic write(s); {len(patched.encode('utf-8'))} byte(s); mode preserved {mode:o}"))
    except (OSError, UnicodeError) as exc:
        gates.append(("POST installed artifact matches the verified candidate", "FAIL",
                      f"0/1 atomic write(s): {str(exc)[:120]}"))
        _report(gates)
        return 2

    _report(gates)
    print(
        f"patched {target}; backend reference={BACKEND_ARTIFACT}; "
        "FS_PLANE_DIR exported; executable controls: 2 MUST_PASS, 1 MUST_FIRE"
    )
    return 0


def _patch_state(text: str) -> tuple[bool, int, list[int]]:
    anchor_counts = [text.count(anchor) for _, anchor in REWRITE_ANCHORS]
    old_state = text.count(OLD_BLOCK) == 1 and all(count == 1 for count in anchor_counts)
    resolver_blocks = len(list(_RESOLVER_RE.finditer(text)))
    return old_state, resolver_blocks, anchor_counts


def _extract_resolver(text: str) -> str | None:
    matches = list(_RESOLVER_RE.finditer(text))
    if len(matches) != 1:
        return None
    return matches[0].group(0)


# The SECOND sibling lookup, and the reason gate E was red at violations=(0, 4, 0).
#
# Installing the resolver fixes how the launcher finds the BACKEND. It does not fix how the
# launcher finds ITSELF. The submit chain resubmits this same file four times -- probe,
# production, resume, post-mortem -- and each of those said "$SCRIPT_DIR/$(basename "$0")".
# SCRIPT_DIR is still assigned inside the resolver (the resolver needs a starting point), so
# every one of those four sites kept resolving against the directory the file happens to be
# sitting in rather than against the plane the resolver just proved. That is #142 exactly, in
# the one code path that MULTIPLIES it: each link of the chain hands its own wrong answer to
# the next job.
#
# Gate E was therefore correct to refuse, and the honest repair is to re-point the four sites,
# not to narrow the gate until it stops noticing them. The rewrite is byte-idempotent: after
# one pass the old spelling does not occur, so the IDEM gate's second application is a no-op.
RESUBMIT_OLD = '"$SCRIPT_DIR/$(basename "$0")"'
RESUBMIT_NEW = '"$FS_PLANE_DIR/$(basename "$0")"'


def _rewrite(text: str) -> str:
    old_state, resolver_blocks, _ = _patch_state(text)
    if old_state and resolver_blocks == 0:
        out = text.replace(OLD_BLOCK, RESOLVER_BLOCK, 1)
    elif not old_state and resolver_blocks == 1:
        match = next(_RESOLVER_RE.finditer(text))
        out = text[:match.start()] + RESOLVER_BLOCK + text[match.end():]
    else:
        raise ValueError(
            f"rewrite state is ambiguous (old={old_state}, resolver blocks={resolver_blocks})"
        )
    return out.replace(RESUBMIT_OLD, RESUBMIT_NEW)


def _clean_control_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "FS_PLANE_DIR",
        "FS_ALLOCATION",
        "SLURM_JOB_ID",
        "BASH_ENV",
        "CDPATH",
    ):
        env.pop(name, None)
    if overrides:
        env.update(overrides)
    return env


def _behaviour_controls(
    bash_path: str, resolver_text: str
) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="fs142-resolver-") as temp_name:
            temp_root = pathlib.Path(temp_name)

            co_located = temp_root / "layout-pass" / "plane"
            co_located.mkdir(parents=True)
            co_script = _write_harness(co_located / "resolver.sh", resolver_text)
            (co_located / BACKEND_ARTIFACT).write_text("# executable control backend\n", encoding="utf-8")

            override_tree = temp_root / "layout-override"
            override_plane = override_tree / "plane"
            override_elsewhere = override_tree / "operator-override"
            override_plane.mkdir(parents=True)
            override_elsewhere.mkdir(parents=True)
            override_script = _write_harness(override_plane / "resolver.sh", resolver_text)
            (override_plane / BACKEND_ARTIFACT).write_text("# losing sibling\n", encoding="utf-8")
            (override_elsewhere / BACKEND_ARTIFACT).write_text("# winning override\n", encoding="utf-8")

            spool = temp_root / "layout-fire" / "var" / "spool" / "slurmd" / "job37274"
            spool.mkdir(parents=True)
            spool_script = _write_harness(spool / "slurm_script", resolver_text)

            results.append(
                _winner_control(
                    bash_path,
                    co_script,
                    {},
                    2,
                    "SCRIPT_DIR",
                    co_located.resolve(),
                    CONTROL_LABELS[0],
                )
            )
            results.append(
                _winner_control(
                    bash_path,
                    override_script,
                    {"FS_PLANE_DIR": str(override_elsewhere)},
                    1,
                    "FS_PLANE_DIR",
                    override_elsewhere.resolve(),
                    CONTROL_LABELS[1],
                )
            )
            results.append(_fire_control(bash_path, spool_script, CONTROL_LABELS[2]))
    except OSError as exc:
        results = [
            (label, "UNMEASURED", f"0/1 control(s): temporary layout failed: {str(exc)[:120]}")
            for label in CONTROL_LABELS
        ]
    return results


def _write_harness(path: pathlib.Path, resolver_text: str) -> pathlib.Path:
    path.write_text(HARNESS_PRELUDE + resolver_text, encoding="utf-8")
    return path


def _winner_control(
    bash_path: str,
    script: pathlib.Path,
    overrides: dict[str, str],
    step: int,
    source: str,
    expected_dir: pathlib.Path,
    label: str,
) -> tuple[str, str, str]:
    try:
        proc = subprocess.run(
            [bash_path, str(script)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_clean_control_env(overrides),
        )
    except subprocess.TimeoutExpired:
        return label, "FAIL", "1/1 control(s); timed out instead of resolving"
    except OSError as exc:
        return label, "UNMEASURED", f"0/1 control(s): {str(exc)[:120]}"

    needle = (
        f"plane directory resolved: {expected_dir} "
        f"(resolution step {step}: {source})"
    )
    ok = proc.returncode == 0 and needle in proc.stdout and proc.stderr == ""
    return (
        label,
        "PASS" if ok else "FAIL",
        f"1/1 control(s); rc={proc.returncode}; winner step {step}={needle in proc.stdout}; "
        f"stderr bytes={len(proc.stderr.encode('utf-8'))}",
    )


def _fire_control(
    bash_path: str, script: pathlib.Path, label: str
) -> tuple[str, str, str]:
    try:
        proc = subprocess.run(
            [bash_path, str(script)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_clean_control_env(),
        )
    except subprocess.TimeoutExpired:
        return label, "FAIL", "1/1 control(s); refused path hung"
    except OSError as exc:
        return label, "UNMEASURED", f"0/1 control(s): {str(exc)[:120]}"

    ordered_needles = (
        "resolution step 1 (FS_PLANE_DIR), returned:",
        "resolution step 2 (SCRIPT_DIR), returned:",
        "resolution step 3 (workload-manager dispatch), returned:",
        "resolution step 4 (refusal), returned:",
        f"directory actually searched last: {script.parent.resolve()}",
        "this looks like a workload-manager-staged copy of the script, not the original",
        "remedy: set and export FS_PLANE_DIR=",
    )
    positions: list[int] = []
    cursor = 0
    for needle in ordered_needles:
        position = proc.stderr.find(needle, cursor)
        positions.append(position)
        if position >= 0:
            cursor = position + len(needle)
    ordered = all(position >= 0 for position in positions) and positions == sorted(positions)
    ok = proc.returncode == 96 and proc.stdout == "" and "FS_PLANE_DIR" in proc.stderr and ordered
    return (
        label,
        "PASS" if ok else "FAIL",
        f"1/1 control(s); rc={proc.returncode}; ordered refusal fields="
        f"{sum(position >= 0 for position in positions)}/{len(ordered_needles)}",
    )


def _atomic_write(target: pathlib.Path, text: str, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        os.chmod(temp_name, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, target)
    except Exception:
        pathlib.Path(temp_name).unlink(missing_ok=True)
        raise


def _finish(gates: list[tuple[str, str, str]]) -> int:
    # An empty gate list must be UNMEASURED even though Python's all([]) is
    # true; success has no denominator until at least one measurement exists.
    if not gates:
        return 3
    if any(status == "FAIL" for _, status, _ in gates):
        return 2
    if any(status == "UNMEASURED" for _, status, _ in gates):
        return 3
    return 0


def _report(gates: list[tuple[str, str, str]]) -> None:
    print("gate table:")
    for label, status, note in gates:
        print(f"  {label}: {status}{f' ({note})' if note else ''}")
    green = sum(status == "PASS" for _, status, _ in gates)
    unmeasured = sum(status == "UNMEASURED" for _, status, _ in gates)
    print(f"gates green: {green}/{len(gates)}")
    if unmeasured:
        print(f"state: UNMEASURED ({unmeasured}/{len(gates)} gate(s) lack a denominator-backed result)")


if __name__ == "__main__":
    raise SystemExit(main())
