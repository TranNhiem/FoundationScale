#!/usr/bin/env python3
"""#175: restore the four-state exit contract on the failure path (findings #169, #171, #174).

The plane publishes 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE; UNMEASURED is a
DECLARED state, distinct from failure. Three defects -- all measured on 8xH100
job 37304, all editing the same region -- destroy the contract exactly where
the exit code matters most.

#174 (BLOCKER): the launcher signalled itself. Its failure path handed the
backend's hard-stop helper $$, the launcher's OWN pid, and with no trap
anywhere in either file bash took the default action and died on the spot:
the END line never ran (a 452-line log of a failed run holds zero END and
zero FATAL lines), the exit never ran (sacct shows 37304.batch CANCELLED
ExitCode 0:15 = signal 15, which an orchestrator reads as a human cancel, not
a framework-declared state), and everything below the kill in the helper --
the grace, the KILL, and the enroot force-remove that exists so no orphaned
rank survives -- was unreachable. The helper is not wrong (its real callers,
the full-FT live tripwires, pass the TRAINING pid); the CALL was. So the fix
makes the class unrepresentable: the helper now refuses any pid equal to the
calling shell's own, a new kill-free cleanup helper carries the enroot arm
for callers whose children are already reaped, and the launcher's failure
path calls only the cleanup.

#171: torchrun flattens the trainer's declared state. The trainer's namespace
is 0 measured / 2 ContractError / 3 OperationFailure-or-UNMEASURED; on 37304
it returned 2, torchrun reported 'failed (exitcode: 2) local_rank: 0', srun
surfaced 1, and the old code propagated that 1 verbatim -- a declared
UNMEASURED became a generic failure. The authoritative record survives the
flattening as TEXT: exactly one rank-0-gated summary line carrying the
verdict. Measured caveat: the log is tee'd more than once (37304's single
summary line appears 3x in launch.37304.log), so the parser takes the LAST
match and never COUNTS matches, and tolerates whitespace around the JSON
colon. The new backend verdict mapper (rc, log) -> contract code is
fail-closed (an undeclared death is a 5, never an abstention; a clean exit
that declared nothing is a 95 -- the vacuous-pass hole, closed; an
unreadable log is a 95), sed/grep only because the login node's host python
is 3.6.8 and this runs outside the container, and lives in the backend
because the GB200 launchers need the identical mapping and duplicating it is
how the two drift. The launcher applies it on BOTH arms: the failure arm
prints an END line carrying the raw srun rc AND the mapped code AND the
parsed verdict (the flattening stays visible, not silently corrected), and
the success arm refuses anything that does not map to 0 BEFORE adjudication.

#169: one out-of-contract exit code, and nothing forbade more. The measured
census of the fail choke point was 49x rc 96, 1x rc 95, and 1x rc 124 (the
compose refusal); 124 is in no namespace this plane publishes. The 124 site
becomes 96, and the choke point now rejects any first argument outside
{0,5,95,96} by printing directly and exiting 96 -- never recursing into
itself, and keeping the FATAL[%s]: %s format for legal codes so no log
consumer breaks.

This stage refuses to write when any anchor is absent or multiplied, when the
pre-image does not exhibit all three defects (a MUST_FIRE premise, so the
stage cannot silently no-op on a file it does not recognise), when the
post-image fail census -- a census, n of n, not a spot-check -- reports any
call site outside the four states, when the adjudication tail is not
byte-identical, when bash -n rejects either patched file, when the transform
is not byte-idempotent, or when any of the 12 bash-level controls (synthetic
logs driving the awk-extracted mapper, a self-kill aliveness proof, a live
victim termination proof, and census MUST_FIRE/MUST_PASS pairs) is not
observed. It measures 95 only when a target cannot be read at all.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
LAUNCHER = HERE / "h100" / "gen" / "launch_fs_h100.fixed.sh"
BACKEND = HERE / "h100" / "gen" / "fs_container_backend.bound.sh"
MARK = "fs175:"
CODES = {"0", "5", "95", "96"}
N_CONTROLS = 12

# Needles are ASSEMBLED, never written as one literal: this stage counts them, and a
# source that contains its own needle is inside its own denominator. The same elision
# is applied in this stage's prose; the scanned names appear unsplit only inside the
# payload blocks that exist to carry them into the targets.
HARD_STOP = "fs_hard_stop" + "_training"
MAP_FN = "fs_map_run" + "_verdict"
CLEANUP_FN = "fs_cleanup" + "_orphans"
SUMMARY = "RUN_SUMMARY" + "_JSON"
FAIL124 = "fail " + "124"
SELF_KILL = HARD_STOP + ' "$$" || true'
UNGUARDED = HARD_STOP + "() {\n  local pid=$1\n  kill -TERM"
GUARD_IF = 'if [[ "$pid" == ' + '"$$" ]]'
PROBE_COMMENT = "# Probe must have produced" + " an early-save checkpoint"
OLD_FAIL = ('fail() { local rc="$1"; shift; printf ' + "'FATAL" + "[%s]: %s\\n' "
            + '"$rc" "$*" >&2; exit "$rc"; }')
OLD_BLOCK = (
    'if [[ "$rc" -ne 0 ]]; then\n'
    "  " + HARD_STOP + ' "$$" || true\n'
    "  printf 'END rc=%s phase=train FAILED\\n' \"$rc\" | tee -a \"$RUN_LOG\"\n"
    '  exit "$rc"\n'
    "fi\n"
)
CENSUS_RX = re.compile(r"\bfail[ \t]+(\d+)")

NEW_FAIL = """# fs175: finding #169 -- the plane publishes exactly four states: 0 PASS, 5 RED,
# 95 UNMEASURED, 96 REFUSE. The measured census of this choke point's call sites
# was 49x rc 96, 1x rc 95, and 1x rc 124 (the compose refusal below): 124 is in
# no namespace this plane publishes, and nothing forbade the next violation. So
# the contract is now enforced HERE, at the single exit choke point: an
# out-of-contract first argument is itself a REFUSE. This must NOT recurse
# through fail (a fail that calls fail on bad input never exits), so the bad-rc
# arm prints directly and exits 96. The FATAL[%s]: %s format is unchanged for
# legal codes, so no log consumer breaks.
fail() {
  local rc="${1:-}"
  if [[ $# -gt 0 ]]; then shift; fi
  case "$rc" in
    0|5|95|96) ;;
    *) printf 'FATAL[96]: fail() called with out-of-contract rc=%s (want one of: 0 5 95 96): %s\\n' "$rc" "$*" >&2; exit 96 ;;
  esac
  printf 'FATAL[%s]: %s\\n' "$rc" "$*" >&2
  exit "$rc"
}
"""

NEW_BLOCK = """if [[ "$rc" -ne 0 ]]; then
  # fs175: finding #174 -- the line this replaces handed the hard-stop helper
  # $$, the launcher's OWN pid, and with no trap anywhere bash took SIGTERM's
  # default action and died on the spot: on job 37304 the END line never ran
  # (a 452-line log of a failed run holds zero END and zero FATAL lines), the
  # exit never ran (sacct: 37304.batch CANCELLED ExitCode 0:15 = signal 15,
  # which an orchestrator reads as a human cancel, not a declared state), and
  # the helper's enroot force-remove -- the arm that exists so no orphaned
  # rank survives -- was unreachable. srun has already returned here, so the
  # ranks are already reaped and only container cleanup remains.
  fs_cleanup_orphans || true
  # fs175: finding #171 -- do NOT propagate $rc verbatim: torchrun flattened
  # the trainer's declared state into it (on 37304: trainer rc 2 -> torchrun
  # 'failed (exitcode: 2) local_rank: 0' -> srun 1), so a declared UNMEASURED
  # surfaced as a generic failure. Map (rc, log) through the backend's verdict
  # mapper; the END line carries the raw srun rc AND the mapped code AND the
  # parsed verdict, so the flattening is visible in the log rather than
  # silently corrected.
  fs175_reason="${RUN_LOG}.fs175_reason"
  mapped_rc="$(fs_map_run_verdict "$rc" "$RUN_LOG" 2>"$fs175_reason")" || fail 96 "fs175: verdict mapper unavailable/failed (backend drift); refusing to guess a state"
  map_reason="$(cat "$fs175_reason" 2>/dev/null || true)"; rm -f "$fs175_reason"
  printf 'END rc=%s mapped_rc=%s phase=train FAILED (%s)\\n' "$rc" "$mapped_rc" "$map_reason" | tee -a "$RUN_LOG"
  exit "$mapped_rc"
fi
# fs175: finding #171, success arm -- rc==0 is not PASS until the trainer's
# declaration is checked: a run that exits clean having declared nothing is the
# vacuous-pass hole, and before this block it fell straight through to
# adjudication. Map (rc, log) and refuse anything that is not 0 BEFORE
# adjudicating. This is the whole point of the doctrine and it costs these
# lines only.
fs175_reason="${RUN_LOG}.fs175_reason"
mapped_rc="$(fs_map_run_verdict "$rc" "$RUN_LOG" 2>"$fs175_reason")" || fail 96 "fs175: verdict mapper unavailable/failed (backend drift); refusing to guess a state"
map_reason="$(cat "$fs175_reason" 2>/dev/null || true)"; rm -f "$fs175_reason"
if [[ "$mapped_rc" != 0 ]]; then
  printf 'END rc=%s mapped_rc=%s phase=train FAILED (%s)\\n' "$rc" "$mapped_rc" "$map_reason" | tee -a "$RUN_LOG"
  exit "$mapped_rc"
fi
"""

GUARD_BLOCK = """  # fs175: finding #174 -- refuse suicide. On job 37304 the launcher handed
  # this helper $$, its OWN pid, and with no trap anywhere bash took SIGTERM's
  # default action and died on the spot: the END line never ran (a 452-line
  # log of a failed run holds zero END and zero FATAL lines), the exit never
  # ran (sacct: 37304.batch CANCELLED ExitCode 0:15 = signal 15, which an
  # orchestrator reads as a human cancel, not a framework-declared state), and
  # everything below the kill -- the grace, the KILL, the enroot force-remove
  # that exists so no orphaned rank survives -- was unreachable. The helper's
  # real callers (the full-FT live tripwires) pass the TRAINING pid; killing
  # yourself is never what a caller means, so make it unrepresentable HERE
  # rather than fixing the one call site.
  if [[ "$pid" == "$$" ]]; then
    echo "fs_hard_stop_training: REFUSING to signal pid $pid -- it is the calling shell's own pid; a self-kill is never what a caller means (fs175/#174)" >&2
    return 1
  fi
"""

BACKEND_PAYLOAD = """
# ---------------------------------------------------------------------------
# fs175: verdict mapping + kill-free container cleanup (findings #169/#171/#174)
#
# fs_map_run_verdict <rc> <logfile> -- map an observed srun rc and the run log
# onto the plane's four-state exit contract (0 PASS, 5 RED, 95 UNMEASURED,
# 96 REFUSE). Echoes the mapped code on stdout and a human-readable reason
# (naming the parsed verdict) on stderr, and returns 0 ALWAYS -- the CALLER
# exits, so a mapping problem can never masquerade as a mapped state.
#
# Why this exists: torchrun flattens the trainer's declared state. The
# trainer's namespace is 0 measured / 2 ContractError / 3 OperationFailure-or-
# UNMEASURED; on job 37304 it returned 2, torchrun reported 'failed
# (exitcode: 2) local_rank: 0', and srun surfaced 1 -- a declared UNMEASURED
# became a generic failure. The authoritative record survives the flattening
# as TEXT: exactly one RUN_SUMMARY_JSON line carrying "verdict":"MEASURED" or
# "verdict":"UNMEASURED", rank-0 gated so ranks never race. Measured caveat:
# the log is tee'd more than once -- 37304's single summary line appears 3x in
# launch.37304.log -- so take the LAST match and never COUNT matches.
# Whitespace around the JSON colon is tolerated. sed/grep only, no python: the
# login node's host python is 3.6.8 and this runs OUTSIDE the container. This
# lives in the backend, not a launcher, because it is framework doctrine, not
# an H100 fact -- the GB200 launchers need the identical mapping, and
# duplicating it is how the two drift.
#
# Mapping (fail-closed):
#   verdict UNMEASURED, any rc -> 95  the declaration outranks the rc
#   verdict MEASURED,   rc 0   -> 0   the only pass
#   verdict MEASURED,   rc!=0  -> 5   rank 0 measured but something else died
#   no verdict line,    rc!=0  -> 5   an undeclared death is a FAILURE
#   no verdict line,    rc 0   -> 95  the vacuous-pass hole, closed
#   logfile missing/unreadable -> 95  cannot read the evidence
# ---------------------------------------------------------------------------
fs_map_run_verdict() {
  local rc=$1 logfile=$2
  case "$rc" in (*[!0-9]*|"") rc=1 ;; esac  # a non-numeric rc is a death, not a pass
  if [[ ! -r "$logfile" ]]; then
    echo "fs_map_run_verdict: verdict=NONE rc=$rc mapped=95 -- logfile '$logfile' missing/unreadable; cannot read the evidence" >&2
    echo 95
    return 0
  fi
  local verdict
  verdict="$(grep 'RUN_SUMMARY_JSON' "$logfile" 2>/dev/null | tail -n 1 | sed -n 's/.*"verdict"[[:space:]]*:[[:space:]]*"\\([^"]*\\)".*/\\1/p')"
  case "$verdict" in
    UNMEASURED)
      echo "fs_map_run_verdict: verdict=UNMEASURED rc=$rc mapped=95 -- the trainer declared UNMEASURED; the declaration outranks the rc" >&2
      echo 95 ;;
    MEASURED)
      if [[ "$rc" -eq 0 ]]; then
        echo "fs_map_run_verdict: verdict=MEASURED rc=0 mapped=0 -- the only pass" >&2
        echo 0
      else
        echo "fs_map_run_verdict: verdict=MEASURED rc=$rc mapped=5 -- rank 0 measured but something else died; never launder to 0" >&2
        echo 5
      fi ;;
    *)
      if [[ "$rc" -ne 0 ]]; then
        echo "fs_map_run_verdict: verdict=NONE rc=$rc mapped=5 -- no verdict line; an undeclared death is a FAILURE, not an abstention" >&2
        echo 5
      else
        echo "fs_map_run_verdict: verdict=NONE rc=0 mapped=95 -- no verdict line; exited clean having declared nothing (the vacuous-pass hole, closed)" >&2
        echo 95
      fi ;;
  esac
  return 0
}

# ---------------------------------------------------------------------------
# fs175: fs_cleanup_orphans -- the enroot force-remove arm of
# fs_hard_stop_training WITHOUT any kill, for callers that have already reaped
# their children (srun has returned; the ranks are already gone) and only need
# container cleanup. Force-removal frees the ~25 GiB unpack; the next launch
# re-creates it. Our own provenance record is removed with it -- it attested a
# container lifetime that just ended (keeping it would trip the stale-record
# refusal in fs_enroot_ensure).
# ---------------------------------------------------------------------------
fs_cleanup_orphans() {
  if [[ "${FS_BACKEND:-}" == enroot && -n "${RIC_ACTIVE_CONTAINER:-}" ]]; then
    echo "TRIPWIRE: force-removing enroot container '$RIC_ACTIVE_CONTAINER' so no orphaned rank survives" >&2
    enroot remove --force "$RIC_ACTIVE_CONTAINER" >/dev/null 2>&1 || true
    rm -f "${ENROOT_DATA_PATH:-$HOME/.enroot}/.fs-provenance/$RIC_ACTIVE_CONTAINER.src" || true
    RIC_ACTIVE_CONTAINER=""
  fi
}
"""


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _census(text: str) -> tuple[list[str], list[str]]:
    toks = CENSUS_RX.findall(text)
    return toks, [t for t in toks if t not in CODES]


def _slice_func(text: str, header: str) -> str:
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header):
            out = [ln]
            for ln2 in lines[i + 1:]:
                out.append(ln2)
                if ln2 == "}":
                    return "\n".join(out)
            return ""
    return ""


def _transform_launcher(text: str) -> tuple[str, dict[str, int], bool]:
    if MARK in text:
        return text, {"old_fail": 0, "fail124": 0, "self_kill": 0, "old_block": 0,
                      "map_fn": text.count(MAP_FN)}, True
    counts = {
        "old_fail": text.count(OLD_FAIL),
        "fail124": text.count(FAIL124),
        "self_kill": text.count(SELF_KILL),
        "old_block": text.count(OLD_BLOCK),
        "map_fn": text.count(MAP_FN),
    }
    new = text
    if counts["old_fail"] == 1:
        new = new.replace(OLD_FAIL, NEW_FAIL, 1)
    if counts["fail124"] == 1:
        new = new.replace(FAIL124, "fail 96", 1)
    if counts["old_block"] == 1:
        new = new.replace(OLD_BLOCK, NEW_BLOCK, 1)
    return new, counts, False


def _transform_backend(text: str) -> tuple[str, dict[str, int], bool]:
    if MARK in text:
        return text, {"header": text.count(HARD_STOP + "() {")}, True
    lines = text.splitlines(keepends=True)
    hdr = [i for i, ln in enumerate(lines) if ln.startswith(HARD_STOP + "() {")]
    counts = {"header": len(hdr)}
    if len(hdr) != 1:
        return text, counts, False
    i = hdr[0]
    if (i + 2 >= len(lines) or lines[i + 1] != "  local pid=$1\n"
            or lines[i + 2] != '  kill -TERM "$pid" 2>/dev/null\n'):
        return text, counts, False
    j = None
    for k in range(i + 2, len(lines)):
        if lines[k] in ("}\n", "}"):
            j = k
            break
    if j is None:
        return text, counts, False
    new = lines[:i + 2] + [GUARD_BLOCK] + lines[i + 2:j + 1] + [BACKEND_PAYLOAD] + lines[j + 1:]
    return "".join(new), counts, False


def _run_bash(script: pathlib.Path, cwd: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    e = dict(os.environ)
    for v in ("FS_BACKEND", "RIC_ACTIVE_CONTAINER", "ENROOT_DATA_PATH", "FS_ALLOCATION",
              "RUN_LOG", "BASH_ENV", "ENV"):
        e.pop(v, None)
    e["PATH"] = "/usr/bin:/bin"
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=cwd,
                          env=e, timeout=timeout)


def _controls(post_l: str, post_b: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    ok = 0
    with tempfile.TemporaryDirectory(prefix="fs175-") as td:
        tdp = pathlib.Path(td)
        backend = tdp / "backend.sh"
        backend.write_text(post_b, "utf-8")
        map_sh = tdp / "mapper.sh"
        stop_sh = tdp / "hardstop.sh"
        # Extract each function under test BY NAME with awk (from its ^name() { header
        # to the matching ^}) and source THAT file in a subshell -- never source a
        # whole launcher or backend, which would execute it.
        awk_prog = 'index($0, hdr) == 1 {on = 1} on {print; if ($0 == "}") exit}'
        try:
            for fname, dest in ((MAP_FN, map_sh), (HARD_STOP, stop_sh)):
                r = subprocess.run(["awk", "-v", "hdr=" + fname + "() {", awk_prog, str(backend)],
                                   capture_output=True, text=True, timeout=30)
                dest.write_text(r.stdout, "utf-8")
        except OSError as exc:
            notes.append(f"C0 provenance: FAIL awk could not run ({exc}); controls not interpretable")
            return 0, notes
        if not (map_sh.read_text("utf-8").rstrip().endswith("}")
                and stop_sh.read_text("utf-8").rstrip().endswith("}")):
            notes.append("C0 provenance: FAIL could not awk-extract the verdict mapper / hard-stop "
                         "helper from the patched backend; controls not interpretable")
            return 0, notes
        probe = tdp / "probe.sh"
        probe.write_text("#!/usr/bin/env bash\nset -u\nsource '" + str(map_sh) + "'\ntype "
                         + MAP_FN + " >/dev/null 2>&1\n", "utf-8")
        try:
            p = _run_bash(probe, td)
        except OSError as exc:
            notes.append(f"C0 provenance: FAIL harness bash could not run ({exc}); controls not interpretable")
            return 0, notes
        if p.returncode != 0:
            notes.append(f"C0 provenance: FAIL the awk-extracted mapper does not load under the control "
                         f"bash (rc={p.returncode}); no MUST_FIRE below would be attributable, so the "
                         "controls were not interpretable")
            return 0, notes
        notes.append("C0 provenance: PASS control bash, awk extraction and function sourcing are "
                     "functional (harness provenance only; the counted controls are C1..C12)")

        # Synthetic evidence. line_u is the REAL summary line of job 37304; line_m keeps
        # whitespace around the JSON colon so C2 also covers colon-spacing tolerance.
        line_u = (SUMMARY + ' {"dataset_origin":"UNKNOWN_NOT_RUN","detail":"cannot begin save while '
                  'train is open","phase":"absent","real_data":null,"reason":"contract_refused",'
                  '"verdict":"UNMEASURED"}')
        line_m = SUMMARY + ' {"phase":"train","real_data":true,"verdict" : "MEASURED"}'
        logs = {
            "unmeasured": line_u + "\n",
            "measured": "BEGIN phase=train probe=probe image=x.sif\n" + line_m + "\n",
            "none": "BEGIN phase=train\nranks started; nothing was ever declared\n",
            "dup": line_u + "\n" + line_u + "\n" + line_u + "\n",
            "order": line_m + "\n" + line_u + "\n",
        }
        for lname, content in logs.items():
            (tdp / (lname + ".log")).write_text(content, "utf-8")

        script = tdp / "case.sh"
        errf = tdp / "case.err"

        def run_map(name: str, rc: int, logpath: pathlib.Path, expect: int, note: str,
                    extra=None) -> bool:
            script.write_text(
                "#!/usr/bin/env bash\nset -u\n"
                "source '" + str(map_sh) + "'\n"
                "out=\"$(" + MAP_FN + " " + str(rc) + " '" + str(logpath) + "' 2>'" + str(errf) + "')\"\n"
                "printf 'MAPPED=%s\\n' \"${out:-EMPTY}\"\n", "utf-8")
            r = _run_bash(script, td)
            got = r.stdout.strip()
            if got.startswith("MAPPED="):
                got = got[len("MAPPED="):]
            good = r.returncode == 0 and got == str(expect)
            if good and extra is not None:
                good = bool(extra())
            detail = ""
            if not good:
                err = errf.read_text("utf-8") if errf.exists() else ""
                detail = "  " + (r.stderr.strip() + " " + err.strip())[:140]
            notes.append(f"{name}: rc_in={rc} mapped={got!r} expected={expect} "
                         f"{'PASS' if good else 'FAIL'}{detail} -- {note}")
            return good

        ok += int(run_map("C1 MUST_FIRE declared UNMEASURED + rc 1 (the 37304 case)", 1,
                          tdp / "unmeasured.log", 95,
                          "the declaration outranks the rc: 95, neither the flattened 1 nor a RED 5"))
        ok += int(run_map("C2 MUST_PASS declared MEASURED + rc 0 (spaced JSON colon)", 0,
                          tdp / "measured.log", 0,
                          "the only pass; also proves whitespace around the colon is tolerated"))
        ok += int(run_map("C3 MUST_FIRE declared MEASURED + rc 1", 1, tdp / "measured.log", 5,
                          "rank 0 measured but something else died; never launder to 0"))
        ok += int(run_map("C4 MUST_FIRE no verdict line + rc 1", 1, tdp / "none.log", 5,
                          "an undeclared death is a FAILURE, not an abstention"))
        ok += int(run_map("C5 MUST_FIRE no verdict line + rc 0 (the vacuous-pass hole)", 0,
                          tdp / "none.log", 95, "exited clean having declared nothing"))
        ok += int(run_map("C6 MUST_FIRE missing logfile + rc 1", 1, tdp / "missing.log", 95,
                          "cannot read the evidence; the unreadable-evidence branch must win over "
                          "the rc fallback"))

        def c7_extra() -> bool:
            err = errf.read_text("utf-8") if errf.exists() else ""
            return err.count("verdict=") == 1 and "verdict=UNMEASURED" in err

        ok += int(run_map("C7 MUST_PASS the 37304 line repeated 3x (measured tee duplication)", 1,
                          tdp / "dup.log", 95,
                          "one verdict parsed and the same code as C1: the parser READS the last "
                          "match, it does not COUNT matches", extra=c7_extra))
        ok += int(run_map("C8 MUST_FIRE earlier MEASURED, later UNMEASURED", 1, tdp / "order.log", 95,
                          "last-match ordering: a first-match parser would have said 5"))

        # C9 MUST_FIRE: the self-kill guard refuses and kills nothing. The proof is the
        # calling shell observed ALIVE after the call -- on 37304 this exact call cost the
        # run its END line and its exit code (sacct CANCELLED, ExitCode 0:15 = signal 15).
        script.write_text(
            "#!/usr/bin/env bash\nset -u\n"
            "source '" + str(stop_sh) + "'\n"
            "if " + HARD_STOP + " \"$$\" 2>'" + str(errf) + "'; then\n"
            "  echo UNEXPECTED_SUCCESS\n"
            "  exit 1\n"
            "fi\n"
            "echo STILL_ALIVE\n", "utf-8")
        r = _run_bash(script, td)
        err9 = errf.read_text("utf-8") if errf.exists() else ""
        c9 = r.returncode == 0 and "STILL_ALIVE" in r.stdout and "REFUSING" in err9
        notes.append(f"C9 MUST_FIRE self-kill guard: rc={r.returncode} "
                     f"alive_after_call={'STILL_ALIVE' in r.stdout} refusal_seen={'REFUSING' in err9} "
                     + ("PASS -- the calling shell was observed alive AFTER the call, which is the "
                        "actual claim" if c9 else "FAIL " + (r.stderr.strip() + " " + err9.strip())[:140]))
        ok += int(c9)

        # C10 MUST_PASS: a legitimate caller (the full-FT live tripwires pass the TRAINING
        # pid) still terminates its victim, so the guard did not break the helper's real
        # users. The victim is polled via ps because a reaped-by-nobody corpse is a zombie
        # and kill -0 alone would keep seeing it.
        script.write_text(
            "#!/usr/bin/env bash\nset -u\n"
            "source '" + str(stop_sh) + "'\n"
            "sleep 300 &\n"
            "victim=$!\n"
            "kill -0 \"$victim\" 2>/dev/null || { echo VICTIM_NOT_RUNNING; exit 1; }\n"
            + HARD_STOP + " \"$victim\" 2>/dev/null &\n"
            "helper=$!\n"
            "dead=0\n"
            "for i in $(seq 1 100); do\n"
            "  state=\"$(ps -p \"$victim\" -o stat= 2>/dev/null || true)\"\n"
            "  if [[ -z \"$state\" || \"$state\" == *Z* ]]; then dead=1; break; fi\n"
            "  sleep 0.1\n"
            "done\n"
            "kill -KILL \"$helper\" 2>/dev/null || true\n"
            "wait \"$helper\" 2>/dev/null || true\n"
            "if [[ \"$dead\" == 1 ]]; then wait \"$victim\" 2>/dev/null || true; echo VICTIM_TERMINATED; exit 0; fi\n"
            "kill -KILL \"$victim\" 2>/dev/null || true\n"
            "wait \"$victim\" 2>/dev/null || true\n"
            "echo VICTIM_SURVIVED\n"
            "exit 1\n", "utf-8")
        r = _run_bash(script, td)
        c10 = r.returncode == 0 and "VICTIM_TERMINATED" in r.stdout
        notes.append(f"C10 MUST_PASS legitimate caller still terminates a live victim: rc={r.returncode} "
                     + ("PASS -- the guard did not break the helper's real users"
                        if c10 else "FAIL " + (r.stdout.strip() + " " + r.stderr.strip())[:140]))
        ok += int(c10)

        # C11 MUST_FIRE / C12 MUST_PASS: the census gate itself, against a doctored copy
        # and against the patched launcher.
        doctored = post_l.replace("fail " + "96", FAIL124, 1)
        dtoks, dbad = _census(doctored)
        c11 = doctored != post_l and dbad == ["124"]
        notes.append(f"C11 MUST_FIRE doctored launcher carrying {FAIL124}: census flags {dbad} "
                     f"of {len(dtoks)} call sites "
                     + ("PASS -- the census gate refuses it" if c11
                        else "FAIL -- the census gate did not fire"))
        ok += int(c11)
        toks12, bad12 = _census(post_l)
        c12 = not bad12 and len(toks12) > 0
        notes.append(f"C12 MUST_PASS patched launcher census: {len(toks12) - len(bad12)} of "
                     f"{len(toks12)} call sites in contract "
                     + ("PASS" if c12 else "FAIL " + str(bad12[:3])))
        ok += int(c12)
    return ok, notes


def main() -> int:
    # build_h100_plane.sh invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would have made the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_launcher_exit_discipline.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    for target in (LAUNCHER, BACKEND):
        if not target.exists():
            _stderr(f"UNMEASURED 95: target missing: {target}")
            return 95
    try:
        text_l = LAUNCHER.read_text("utf-8")
        text_b = BACKEND.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: target unreadable: {exc}")
        return 95

    l_marked = MARK in text_l
    b_marked = MARK in text_b
    if l_marked and b_marked:
        print("verdict: already applied; byte-idempotent no-op")
        return 0
    if l_marked != b_marked:
        _stderr(f"REFUSE 96: partial application (launcher marked={l_marked}, backend "
                f"marked={b_marked}); the two files carry one contract and this stage will "
                "not guess which half is stale")
        return 96

    post_l, counts_l, _ = _transform_launcher(text_l)
    post_b, counts_b, _ = _transform_backend(text_b)

    gates = 0
    gres: list[tuple[str, bool, str]] = []

    gres.append(("G1", counts_l["old_fail"] == 1 and counts_l["old_block"] == 1,
                 f"launcher anchors unique: unhardened fail def {counts_l['old_fail']} of 1, "
                 f"old failure block {counts_l['old_block']} of 1"))
    pre_toks, pre_bad = _census(text_l)
    unguarded_pre = text_b.count(UNGUARDED)
    g2 = (counts_l["fail124"] == 1 and counts_l["self_kill"] == 1 and counts_l["map_fn"] == 0
          and unguarded_pre == 1 and pre_bad == ["124"])
    gres.append(("G2", g2,
                 f"MUST_FIRE defect premises, 5 of 5 required: #169 {FAIL124} call site="
                 f"{counts_l['fail124']} (need 1) and pre-image census out-of-contract={pre_bad} "
                 "(need exactly ['124'] -- the measured 49/1/1 census had exactly one), #174 "
                 f"self-kill call={counts_l['self_kill']} (need 1), #174 unguarded hard-stop "
                 f"opening={unguarded_pre} (need 1), #171 verdict mapper absent from launcher="
                 f"{counts_l['map_fn']} (need 0); a pre-image not exhibiting all three defects "
                 "is a file this stage does not recognise"))
    gres.append(("G3", counts_b["header"] == 1 and text_b.count(MAP_FN) == 0
                 and text_b.count(CLEANUP_FN) == 0,
                 f"backend anchors: hard-stop header {counts_b['header']} of 1, mapper pre "
                 f"{text_b.count(MAP_FN)} of 0, cleanup pre {text_b.count(CLEANUP_FN)} of 0"))
    toks, bad = _census(post_l)
    gres.append(("G4", not bad and len(toks) > 0,
                 f"post-image launcher fail census: {len(toks) - len(bad)} of {len(toks)} call "
                 f"sites in contract (out-of-contract: {bad[:3] if bad else 'none'}) -- verified "
                 "by CENSUS, not spot-check"))
    g5checks = [
        ("unhardened fail def gone", post_l.count(OLD_FAIL) == 0),
        ("rc-124 site gone", post_l.count(FAIL124) == 0),
        ("self-kill call gone", post_l.count(SELF_KILL) == 0),
        ("old failure block gone", post_l.count(OLD_BLOCK) == 0),
        ("hardened fail present", post_l.count("want one of: 0 5 95 96") == 1),
        ("mapper called on both arms", post_l.count(MAP_FN + ' "$rc" "$RUN_LOG"') == 2),
        ("cleanup called once", post_l.count(CLEANUP_FN + " || true") == 1),
        ("launcher carries the fs175 mark", post_l.count(MARK) > 0),
    ]
    g5bad = [n for n, c in g5checks if not c]
    gres.append(("G5", not g5bad,
                 f"post-image launcher structure {len(g5checks) - len(g5bad)} of {len(g5checks)}"
                 + (": " + "; ".join(g5bad) if g5bad else "")))
    stop_slice = _slice_func(post_b, HARD_STOP + "() {")
    g6checks = [
        ("hard-stop slice recovered", bool(stop_slice)),
        ("self-kill guard present once", post_b.count(GUARD_IF) == 1),
        ("guard precedes the kill",
         bool(stop_slice) and -1 < stop_slice.find(GUARD_IF) < stop_slice.find('kill -TERM "$pid"')),
        ("TERM arm still reachable below the guard", stop_slice.count('kill -TERM "$pid"') == 1),
        ("KILL escalation still reachable", stop_slice.count('kill -KILL "$pid"') == 1),
        ("enroot force-remove still reachable", stop_slice.count("enroot remove --force") == 1),
        ("mapper defined once", post_b.count(MAP_FN + "() {") == 1),
        ("cleanup defined once", post_b.count(CLEANUP_FN + "() {") == 1),
        ("unguarded opening gone", post_b.count(UNGUARDED) == 0),
        ("backend carries the fs175 mark", post_b.count(MARK) > 0),
    ]
    g6bad = [n for n, c in g6checks if not c]
    gres.append(("G6", not g6bad,
                 f"post-image backend structure {len(g6checks) - len(g6bad)} of {len(g6checks)}"
                 + (": " + "; ".join(g6bad) if g6bad else "")))
    probe_ok = text_l.count(PROBE_COMMENT) == 1 and post_l.count(PROBE_COMMENT) == 1
    suffix_same = (probe_ok
                   and text_l[text_l.index(PROBE_COMMENT):] == post_l[post_l.index(PROBE_COMMENT):])
    gres.append(("G7", suffix_same,
                 "adjudication tail 1 of 1: everything from the probe-checkpoint comment through "
                 "the checkpoint_observed gate (old line 796) to EOF is byte-identical pre/post "
                 "-- the adjudication block was not touched"))
    bn_results = []
    for label, content in (("launcher", post_l), ("backend", post_b)):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as fh:
            fh.write(content)
            tmp = fh.name
        try:
            bn = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True, timeout=30)
        finally:
            os.unlink(tmp)
        bn_results.append((label, bn.returncode == 0, bn.stderr.strip()[:120]))
    gres.append(("G8", all(ok for _, ok, _ in bn_results),
                 f"bash -n clean {sum(1 for _, ok, _ in bn_results if ok)} of {len(bn_results)}: "
                 + "; ".join(f"{lb}={'clean' if ok else er}" for lb, ok, er in bn_results)))
    again_l, _, _ = _transform_launcher(post_l)
    again_b, _, _ = _transform_backend(post_b)
    gres.append(("G9", again_l == post_l and again_b == post_b,
                 f"byte-idempotence on own output {int(again_l == post_l) + int(again_b == post_b)} "
                 "of 2 (a second run is a byte-exact no-op)"))

    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    cok, cnotes = _controls(post_l, post_b)
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls {cok}/{N_CONTROLS}; "
                "writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  2 files would be rewritten, {gates}/{len(gres)} static gates, "
              f"{cok}/{N_CONTROLS} controls")
        return 0
    BACKEND.write_text(post_b, "utf-8")
    LAUNCHER.write_text(post_l, "utf-8")
    print(f"{MARK} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # This stage exists to keep four exit states distinct on the exact path where they
    # collapse, so it must not collapse its own: an unhandled exception is a REFUSE,
    # not a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())