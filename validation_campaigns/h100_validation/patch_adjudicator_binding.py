#!/usr/bin/env python3
"""Stage for the adjudicator-binding plane: patch launch_fs_h100.fixed.sh in place (#146).

`h100/gen/launch_fs_h100.fixed.sh` is a generated artifact, so the fix for
finding #146 cannot be a hand edit -- a hand edit lands in the file someone
read, and the next regeneration silently reverts it. The patch therefore
lives here, in the plane, applied by exact-once anchors that REFUSE on a
multi-match, because patching the first of two hits is how half a fix ships.

The stage rewrites three regions of one script: it moves the ADJUDICATORS
parse up beside the other startup containment checks (the consumers sat
above the parse, which is the structural reason the knob was unchecked),
adds the fifth containment call site, folds each spec's dirname into the
FS_BIND_PATHS inference, and stops the invoker from leaking python3's
undeclared exit 2. Every one of those is proven, not assumed: gate D
compares line numbers, and the BEHAVIOUR gate executes the extracted,
patched logic under a real bash with a MUST_FIRE that is observed refusing.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

MARKER = "# fs146 (a):"

# Anchors are matched EXACTLY ONCE each (gate PRE) before anything is
# touched. A blind .replace would pass silently over a regenerated file
# whose text has drifted, and the patched artifact would then be checked
# against a contract it no longer satisfies -- the same failure the junk-
# line strip in the extract stage guards against.

A1_OLD = r'''fs_path_under_allowed_root "$DATASET_DIR" || fail 96 \
  "DATASET_DIR is outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS): $DATASET_DIR"'''

# The parse is re-homed verbatim beside the startup checks. The alternative
# -- moving the checks down -- was rejected: (a) is a defect BECAUSE the bad
# value surfaced late, so a fix that keeps the check late has not fixed (a).
A2_OLD = r'''# Checkpoint adjudicators: finding #68 was zero call sites. They must be configured explicitly;
# absence is not success. Each entry is invoked as: <cmd> <checkpoint_dir> <phase> <out_dir>.
ADJUDICATORS_RAW="${FS_CHECKPOINT_ADJUDICATORS:-}"
[[ -n "$ADJUDICATORS_RAW" ]] || fail 96 "FS_CHECKPOINT_ADJUDICATORS empty; cannot call save adjudicators after saves (all([])!=PASS) -- entries are space/tab/newline-separated, one adjudicator per word"
# fs139: this list used to be split under the global safety IFS while its
# comment said "Each entry is invoked as ..." -- entries, plural, separator
# never stated -- and its refusal then named an adjudicator path, blaming
# the value when the fault was the invisible separator from line 18.
# Prose and parser now agree, in both directions: one adjudicator per word,
# separated by space, tab or newline (space because that is what an
# operator types; tab/newline because that is what the old behaviour forced
# and must keep working). A claim broader than its evidence is a defect
# even when the code is correct -- the stale refusal message was that.
# The assignment-prefix IFS scopes to this read alone, so the global
# safety setting is not weakened for the rest of the script.
IFS=$' \t\n' read -r -a ADJUDICATORS <<< "$ADJUDICATORS_RAW"
[[ "${#ADJUDICATORS[@]}" -gt 0 ]] || fail 96 "zero checkpoint adjudicators configured"
for a in "${ADJUDICATORS[@]}"; do [[ -n "$a" ]] || fail 96 'empty adjudicator token'; done'''

A2_NEW = r'''# fs146: the ADJUDICATORS parse and its refusals MOVED UP beside the other
# startup containment checks. Both consumers of the parsed list -- the (a)
# containment loop and the (b) bind-plane derivation -- sit ABOVE this
# point, so a parse that lived here made the knob uncheckable exactly where
# the other four executed paths were checked. Gate D proves the new order
# by line number; do not re-move the parse without re-homing its consumers.'''

A3_OLD = r'''for _p in "$MODEL_DIR" "$DATASET_DIR" "$(dirname -- "$CONFIG_FILE")" "$OUT_DIR" \
          ${FS_EXTRA_BIND_PATHS:-}; do'''

A3_NEW = r'''for _p in "$MODEL_DIR" "$DATASET_DIR" "$(dirname -- "$CONFIG_FILE")" "$OUT_DIR" \
          ${_adj_dirs[@]+"${_adj_dirs[@]}"} \
          ${FS_EXTRA_BIND_PATHS:-}; do'''

A4_OLD = r'''    if [[ -x "$spec" ]]; then "$spec" "$ckpt" "$phase" "$OUT_DIR" || return $?; ok=$((ok+1)); continue; fi
    if [[ -f "$spec" && "$spec" == *.py ]]; then run_in_container --workdir "$OUT_DIR" -- python3 "$spec" "$ckpt" "$phase" "$OUT_DIR" || return $?; ok=$((ok+1)); continue; fi
    printf 'ADJUDICATOR-REFUSE rc=96 not_executable=%s\n' "$spec" >&2; return 96'''

A4_NEW = r'''    # fs146 (c): capture, then classify -- never propagate blind. The old
    # `|| return $?` let python3's exit 2 (spec invisible in-container, i.e.
    # missing from the bind plane) escape as `END rc=2 phase=adjudicate`, a
    # code this plane does not declare and therefore cannot attribute. 0
    # stays 0, 95 stays 95 (abstention is not failure), 96 stays 96;
    # ANYTHING else maps to 96 with the ORIGINAL code printed. Mapping
    # undeclared codes to 0 was considered and rejected as strictly worse
    # than the defect: a laundered silent pass is all([]) wearing green.
    local _rc=0
    if [[ -x "$spec" ]]; then
      "$spec" "$ckpt" "$phase" "$OUT_DIR" || _rc=$?
    elif [[ -f "$spec" && "$spec" == *.py ]]; then
      run_in_container --workdir "$OUT_DIR" -- python3 "$spec" "$ckpt" "$phase" "$OUT_DIR" || _rc=$?
    else
      printf 'ADJUDICATOR-REFUSE rc=96 not_executable=%s\n' "$spec" >&2
      return 96
    fi
    case $_rc in
      0) ;;
      95|96) return "$_rc" ;;
      *)
        printf 'ADJUDICATOR-REFUSE rc=96 original_rc=%s spec=%s -- undeclared exit code mapped to 96; leading hypothesis: spec is not bound into the container (absent from FS_BIND_PATHS), not an adjudicator failure\n' "$_rc" "$spec" >&2
        return 96
        ;;
    esac
    ok=$((ok+1)); continue'''

# fs117's refusal still says "four required inputs"; after (b) the derivation
# has a fifth input class. Leaving the numeral would be the label-does-not-
# match-measurement defect wearing a prose disguise, so it goes too.
A5_OLD = r'''derived ZERO bind paths from four required inputs'''
A5_NEW = r'''derived ZERO bind paths from the required inputs and the adjudicator dirnames'''

INSERT = r'''# fs146: the ADJUDICATORS parse MOVED here from its original home BELOW the
# bind derivation. Both consumers of the parsed list -- the (a) containment
# check and the (b) bind-plane derivation -- sat above that point, and a
# consumer that cannot be checked until after its input exists is the
# structural cause of #146: the most load-bearing knob in the plane was
# the only unchecked one BECAUSE nothing up here could see it. Moving the
# consumers down was the alternative and was rejected: startup checks buy
# nothing late, and (a) is a defect precisely because a bad value surfaced
# after hours of paid GPU time, not before.

''' + A2_OLD + r'''

# fs146 (a): EVERY spec gets the containment check the other four executed
# paths already had, and the report carries its denominator ("k of n"), so
# an empty or short-circuited sweep can never read as measured -- all([])
# is UNMEASURED, never PASS. The refusal names the offending spec AND the
# declared roots: blame aimed only at the value sends the operator to edit
# the wrong half of the contract.
_adj_seen=0; _adj_ok=0; _adj_bad=""
for _adj in "${ADJUDICATORS[@]}"; do
  _adj_seen=$((_adj_seen+1))
  if fs_path_under_allowed_root "$_adj"; then
    _adj_ok=$((_adj_ok+1))
  else
    _adj_bad="$_adj_bad $_adj"
  fi
done
printf 'fs146: containment %d of %d adjudicator spec(s) under a declared root\n' "$_adj_ok" "$_adj_seen"
[[ "$_adj_ok" -eq "$_adj_seen" ]] || fail 96 \
  "fs146: adjudicator spec(s) outside every declared FS_ALLOWED_PATH_ROOTS entry ($FS_ALLOWED_PATH_ROOTS):${_adj_bad}"
unset _adj_seen _adj_ok _adj_bad
# fs146 (b): each spec's dirname joins the bind inference below. The
# tempting move is a runbook -- "operators must add these to
# FS_EXTRA_BIND_PATHS" -- and it is rejected here on purpose: this knob is
# REQUIRED and framework-known, and routing a required input through the
# escape hatch makes "required" mean "remembered", which is not a control.
# Membership of an allowed root and membership of the bind plane are
# DIFFERENT properties; inference owes declared inputs both. The existing
# de-duplication below is untouched, so a spec under MODEL_DIR adds no
# redundant mount.
declare -a _adj_dirs=()
for _adj in "${ADJUDICATORS[@]}"; do
  _adj_dirs+=("$(dirname -- "$_adj")")
done
unset _adj'''

A1_NEW = A1_OLD + "\n\n" + INSERT

# Application order matters: A2 consumes the ORIGINAL parse block first,
# because A1's replacement re-introduces the same verbatim text one region
# higher; applying A1 first would make A2's anchor match twice and the
# exact-once guard would (correctly) refuse its own patch.
# `consumed` marks anchors that must NOT survive in patched output; A1 is
# retained by construction, so the idempotence probe exempts it.
STEPS = (
    ("A2 move parse out of region 3", A2_OLD, A2_NEW, True),
    ("A1 move parse in beside the containment checks, add check + dirnames", A1_OLD, A1_NEW, False),
    ("A3 fold adjudicator dirnames into FS_BIND_PATHS", A3_OLD, A3_NEW, True),
    ("A4 normalise the invoker's return codes", A4_OLD, A4_NEW, True),
    ("A5 de-numeral the fs117 refusal", A5_OLD, A5_NEW, True),
)

CALL = 'fs_path_under_allowed_root "$'
CALL_ADJ = 'fs_path_under_allowed_root "$_adj"'
DIRS_APPEND = '_adj_dirs+=("$(dirname -- "$_adj")")'
PARSE_LINE = "IFS=$' \\t\\n' read -r -a ADJUDICATORS"
BIND_LINE = 'for _p in "$MODEL_DIR" "$DATASET_DIR"'


class PatchError(RuntimeError):
    pass


def _stale_anchors(text: str) -> list[str]:
    """Consumed anchors that survived AT OR BELOW their own replacement.

    One helper, two callers, on purpose: the same position-blind `text.count(old)` test was
    written twice -- once in _patch and once in the PRE gate -- so fixing one left the other
    reporting a correct patch as half-applied. Two copies of a rule are two chances to fix
    only one of them, which is #150's shape at a smaller scale.
    """
    stale: list[str] = []
    for name, old, new, consumed in STEPS:
        if not consumed:
            continue
        where = text.find(new)
        if where < 0:
            stale.append(f"{name} (replacement text absent)")
        elif old in text[where:]:
            stale.append(name)
    return stale


def _patch(text: str) -> tuple[str, bool]:
    """Return (patched_text, applied). Idempotent: marker present means no-op."""
    if MARKER in text:
        # POSITION, not presence. A2 is a MOVE: it deletes the adjudicator parse from region 3
        # and A1 re-inserts that same block beside the startup containment checks. A bare
        # `text.count(old)` therefore finds the successfully relocated block and calls the move
        # stale -- IDEM went red on every re-run with "consumed anchor(s) survive: ['A2 ...']"
        # while the patch was in fact correct. The anchor is stale only if it survives AT OR
        # BELOW the comment that replaced it; anything above that point is the destination.
        stale = _stale_anchors(text)
        if stale:
            raise PatchError(f"patched marker present but consumed anchor(s) survive: {stale}")
        return text, False
    counts = {name: text.count(old) for name, old, _, _ in STEPS}
    bad = {name: n for name, n in counts.items() if n != 1}
    if bad:
        raise PatchError(f"anchor occurrence counts are not exactly 1: {bad}")
    for _, old, new, _ in STEPS:
        text = text.replace(old, new, 1)
    return text, True


def _line_of(text: str, needle: str) -> int:
    return text[: text.index(needle)].count("\n") + 1


def _extract(text: str, start: str, end_token: str) -> str:
    i = text.index(start)
    j = text.index(end_token, i + len(start))
    return text[i : j + len(end_token)]


def _run_probes(bash: str, patched: str) -> list[tuple[str, bool, str]]:
    """Execute the PATCHED logic, extracted from the artifact text -- never a
    re-implementation, because a probe that tests its own transcription of
    the code measures the transcription, not the artifact."""
    helper = _extract(patched, "fs_path_under_allowed_root() {", "\n}\n")
    logic = _extract(patched, 'ADJUDICATORS_RAW="${FS_CHECKPOINT_ADJUDICATORS:-}"', "unset _adj\n")
    bind = _extract(patched, "declare -a FS_BIND_PATHS=()", "unset _p _q _dup\n")
    invoker = _extract(patched, "run_adjudicators() {", "\n}\n")

    prologue = r"""
set -uo pipefail
fail() { printf 'REFUSE rc=96 %s\n' "$*" >&2; exit 96; }
FS_ALLOWED_PATH_ROOTS='/allowed/shared'
MODEL_DIR=/allowed/shared/model
DATASET_DIR=/allowed/shared/dataset
OUT_DIR=/allowed/shared/out
CONFIG_FILE=/allowed/shared/conf/train.conf
"""
    results: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="fs146_probe_") as td:
        tdp = pathlib.Path(td)

        def run(name: str, body: str) -> subprocess.CompletedProcess:
            path = tdp / f"probe_{name}.sh"
            path.write_text(body, encoding="utf-8")
            return subprocess.run([bash, str(path)], capture_output=True, text=True,
                                  cwd=td, timeout=60)

        # MUST_PASS: a spec under a declared root is accepted AND its dirname
        # lands in the derived bind plane.
        a = run("a", prologue
                + "FS_CHECKPOINT_ADJUDICATORS=/allowed/shared/adj/adjudicate.py\n"
                + helper + logic + bind + r"""
probe_hit=0
for _bq in ${FS_BIND_PATHS[@]+"${FS_BIND_PATHS[@]}"}; do
  [[ "$_bq" == "/allowed/shared/adj" ]] && probe_hit=1
done
[[ "$probe_hit" -eq 1 ]] || { printf 'PROBE-A bind plane missing /allowed/shared/adj (derived: %s)\n' "${FS_BIND_PATHS[*]}" >&2; exit 97; }
printf 'PROBE-A-OK binds=%d %s\n' "${#FS_BIND_PATHS[@]}" "${FS_BIND_PATHS[*]}"
""")
        ok = (a.returncode == 0 and "1 of 1 adjudicator spec(s)" in a.stdout
              and "/allowed/shared/adj" in a.stdout)
        results.append(("MUST_PASS contained spec accepted, dirname bound",
                        ok, f"rc={a.returncode} {a.stdout.strip()[-90:]}"))

        # MUST_FIRE: a spec outside every declared root is OBSERVED refusing,
        # named in its own refusal. A containment check never seen red is not
        # known to work.
        b = run("b", prologue
                + "FS_CHECKPOINT_ADJUDICATORS=/etc/evil/adjudicate.py\n"
                + helper + logic)
        ok = (b.returncode == 96 and "/etc/evil/adjudicate.py" in b.stderr
              and "FS_ALLOWED_PATH_ROOTS" in b.stderr)
        results.append(("MUST_FIRE uncontained spec refused, spec named",
                        ok, f"rc={b.returncode} {b.stderr.strip()[-90:]}"))

        # MUST_FIRE: an adjudicator exiting 2 becomes 96 -- not 2 and not 0 --
        # with the original code printed. A laundered silent pass is worse
        # than the defect, so the probe asserts the code did NOT vanish.
        inv_pre = r"""
set -uo pipefail
OUT_DIR="$PWD/out"; mkdir -p -- "$OUT_DIR"
CKPT="$PWD/checkpoint-001"; mkdir -p -- "$CKPT"
checkpoint_observed=0
"""
        c = run("c", inv_pre
                + "printf '#!/usr/bin/env bash\\nexit 2\\n' > adj.sh; chmod +x adj.sh\n"
                + 'ADJUDICATORS=("$PWD/adj.sh")\n'
                + invoker + r"""
run_adjudicators "$CKPT" save
probe_rc=$?
printf 'PROBE-C rc=%d\n' "$probe_rc"
""")
        ok = (c.returncode == 0 and "PROBE-C rc=96" in c.stdout
              and "original_rc=2" in c.stderr)
        results.append(("MUST_FIRE exit 2 normalised to 96, original printed",
                        ok, f"rc={c.returncode} {c.stdout.strip()} {c.stderr.strip()[-70:]}"))

        # MUST_PASS: an adjudicator that abstains (95) has not failed, and the
        # normaliser must not flatten abstention into refusal.
        d = run("d", inv_pre
                + "printf '#!/usr/bin/env bash\\nexit 95\\n' > adj.sh; chmod +x adj.sh\n"
                + 'ADJUDICATORS=("$PWD/adj.sh")\n'
                + invoker + r"""
run_adjudicators "$CKPT" save
probe_rc=$?
printf 'PROBE-D rc=%d\n' "$probe_rc"
""")
        ok = (d.returncode == 0 and "PROBE-D rc=95" in d.stdout
              and "ADJUDICATOR-REFUSE" not in d.stderr)
        results.append(("MUST_PASS exit 95 preserved (abstention != failure)",
                        ok, f"rc={d.returncode} {d.stdout.strip()} {d.stderr.strip()[-70:]}"))

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    target = root / "h100" / "gen" / "launch_fs_h100.fixed.sh"

    gates: list[tuple[str, bool, str]] = []

    exists = target.is_file() and os.access(target, os.R_OK)
    gates.append(("PRE target exists and is readable", exists, str(target)))
    if not exists:
        return _fail(gates)
    text = target.read_text(encoding="utf-8")

    already = MARKER in text
    if already:
        stale = _stale_anchors(text)
        gates.append(("PRE already-patched input: consumed anchors are gone, none half-applied",
                      not stale, ",".join(stale) or "re-application is a pure no-op"))
    else:
        counts = {n: text.count(o) for n, o, _, _ in STEPS}
        gates.append(("PRE every rewrite anchor occurs EXACTLY once (multi-match refuses)",
                      all(v == 1 for v in counts.values()),
                      ",".join(f"{n.split()[0]}={v}" for n, v in counts.items())))
    if not gates[-1][1]:
        return _fail(gates)

    before_sites = text.count(CALL)
    try:
        patched, applied = _patch(text)
    except PatchError as exc:
        gates.append(("patch application", False, str(exc)[:120]))
        return _fail(gates)

    after_sites = patched.count(CALL)
    adj_sites = patched.count(CALL_ADJ)
    # The COUNT alone cannot say WHAT was added -- a fifth site could be any
    # path -- so the gate asserts the identity of the new site too, and that
    # the four originals are undisturbed (5 = 4 + 1, not 6, not 4).
    gates.append(("A containment call sites 4 -> 5, and the fifth is the adjudicator loop",
                  after_sites == 5 and adj_sites == 1 and before_sites in (4, 5),
                  f"sites {before_sites}->{after_sites}; adjudicator-loop sites {adj_sites}"))

    dirs_idx = patched.find(DIRS_APPEND)
    bind_idx = patched.find(A3_NEW)
    gates.append(("B every spec's dirname reaches the FS_BIND_PATHS derivation",
                  dirs_idx != -1 and bind_idx != -1 and dirs_idx < bind_idx,
                  f"dirname append at byte {dirs_idx}, bind loop at byte {bind_idx}"))

    try:
        parse_ln = _line_of(patched, PARSE_LINE)
        cont_ln = _line_of(patched, CALL_ADJ)
        bind_ln = _line_of(patched, BIND_LINE)
        order_ok = parse_ln < cont_ln < bind_ln
    except ValueError as exc:
        parse_ln = cont_ln = bind_ln = -1
        order_ok = False
    # ORDER is proved by line arithmetic, not by belief: the whole failure
    # mode of #146 was a consumer ordered before its parse, so "we moved it"
    # is a claim that needs a denominator the file itself supplies.
    gates.append(("D ORDER: parse precedes containment check precedes bind derivation",
                  order_ok, f"parse L{parse_ln} < containment L{cont_ln} < bind L{bind_ln}"))

    try:
        inv = _extract(patched, "run_adjudicators() {", "\n}\n")
        # LIVE LINES ONLY. The absence clause looks for the old `|| return $?` propagation the
        # normaliser replaces -- and the replacement's own comment quotes that string in
        # backticks to explain what it removed, so the gate matched its own documentation and
        # reported FAIL on a correct rewrite. Same discriminator as everywhere else in this
        # build: a description of a thing is not the thing, and comment-versus-live-line is
        # the context that separates them. The two presence clauses stay on the full body:
        # they assert the new shape EXISTS, which a comment cannot fake into being executed.
        inv_live = "\n".join(ln for ln in inv.splitlines() if not ln.lstrip().startswith("#"))
        e_ok = ('95|96) return "$_rc"' in inv and "original_rc=" in inv
                and "|| return $?" not in inv_live)
    except ValueError:
        e_ok = False
    gates.append(("E normaliser keeps 0/95/96, maps the undeclared space to 96, prints the original code",
                  e_ok, "static shape; the 2->96 and 95-kept claims are PROVEN below, not here"))

    for _, ok, _ in gates:
        pass  # reporting is deferred to the single table below, as in the plane

    bash = shutil.which("bash")
    if bash is None:
        # Gates C and BEHAVIOUR need a real bash; without one they are not
        # red, they are UNMEASURED -- a declared state with its own exit
        # code, and the artifact must not be written on unmeasured evidence.
        _report(gates)
        print("C bash -n: UNMEASURED (no bash on PATH)")
        print("BEHAVIOUR: UNMEASURED (no bash on PATH)")
        print("refusing to write on UNMEASURED gates -- fail closed")
        return 95

    # TemporaryDirectory, not TemporaryFile: the generated stage asked for a DIRECTORY named
    # *.sh and then wrote text to it (IsADirectoryError). The syntax gate is the one that has
    # to hold before anything is written to the tree, so a crash here is not a cosmetic bug --
    # it takes out the check standing between a malformed rewrite and the shipped launcher.
    with tempfile.TemporaryDirectory(prefix="fs146_bn_") as td:
        tf = str(pathlib.Path(td) / "candidate.sh")
        pathlib.Path(tf).write_text(patched, encoding="utf-8")
        proc = subprocess.run([bash, "-n", tf], capture_output=True, text=True)
    gates.append(("C bash -n clean on the patched artifact",
                  proc.returncode == 0, proc.stderr.strip()[:120]))

    try:
        probes = _run_probes(bash, patched)
        probe_fl = [n for n, ok, _ in probes if not ok]
        gates.append((f"BEHAVIOUR {len(probes) - len(probe_fl)}/{len(probes)} probes green under real bash "
                      "(containment pass+fire, 2->96 fire, 95-kept pass)",
                      not probe_fl, "; ".join(probe_fl) or "all controls observed"))
    except (ValueError, subprocess.SubprocessError) as exc:
        gates.append(("BEHAVIOUR extraction/execution", False, str(exc)[:120]))

    try:
        again, reapplied = _patch(patched)
        idem_ok = (not reapplied) and again == patched
    except PatchError:
        idem_ok = False
    gates.append(("IDEM re-application is a byte-exact no-op", idem_ok,
                  "second pass left the text untouched" if idem_ok else "second pass changed bytes"))

    if not all(ok for _, ok, _ in gates):
        print("refusing to write: a red gate means no artifact ships")
        return _fail(gates)

    # Atomic replace: temp file in the target's own directory, then rename.
    # A half-written launcher is worse than a stale one -- bash -n on a
    # truncated script can even be clean if the cut lands between functions.
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(patched)
    os.replace(tmp, target)

    _report(gates)
    state = "applied" if applied else "already patched; verified byte-identical"
    print(
        f"{state}: {target} -- containment sites {before_sites}->{after_sites}, "
        f"parse L{parse_ln} < containment L{cont_ln} < bind L{bind_ln}, "
        "probes 4/4 green (contained spec bound; uncontained spec refused by name; "
        "exit 2 -> 96 with the original printed; exit 95 preserved)"
    )
    return 0


def _fail(gates: list[tuple[str, bool, str]]) -> int:
    _report(gates)
    return 2


def _report(gates: list[tuple[str, bool, str]]) -> None:
    print("gate table:")
    for label, ok, note in gates:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}{(' (' + note + ')') if note else ''}")
    green = sum(1 for _, ok, _ in gates if ok)
    print(f"{green}/{len(gates)} gates green")


if __name__ == "__main__":
    raise SystemExit(main())
