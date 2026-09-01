#!/usr/bin/env python3
"""fs176: the checkpoint-adjudication walk truncated itself, then reported the
truncation as a complete denominator (finding #175).

MEASURED on job 37308 -- a run that PASSED. That is the point. The output tree
held TWO checkpoint saves (the early save the resume proof depends on, and the
final save). Re-running the walker's own find by hand returns both, exit 0. The
launcher processed exactly one, printed a coverage claim of 1, and exited 0;
nothing in the plane could tell. Three faults in one 8-line function:

D1  the loop body ate its own iterator. The per-checkpoint runner was invoked
    INSIDE the streaming read loop over the find -print0 stream; the container
    runtime and the interpreter inherited that stdin and drained it, so the
    second read hit EOF. Only the FIRST directory find emitted was ever
    adjudicated -- here the FINAL save, so the early one was skipped in silence.
D2  the denominator was computed inside the truncated loop, so numerator and
    denominator were cut together and the report stayed self-consistent: a
    denominator derived from the stream it measures cannot ever detect its own
    truncation. The doctrine's own failure mode wearing green.
D3  a refusing adjudicator could not fail the walk. The bare call discarded the
    runner's rc, and the `|| ar=$?` call site suppresses errexit throughout the
    function body, so a per-checkpoint REFUSE (96) or ABSTAIN (95) was dropped
    twice over and the function returned the status of its final printf. The
    consuming gate was a POSITIVITY check, not a denominator check: 1-of-2
    satisfied it.

FIX: collect the whole find stream into an array in a loop whose body forks
nothing; iterate the array with for; feed every runner invocation </dev/null;
measure the found count BEFORE any adjudicator runs and REFUSE 96 if the
processed count differs (an iterator that lost entries is a framework defect,
not a checkpoint result); capture each verdict explicitly, map undeclared codes
to 96 with the original printed, and return the WORST outcome (96 > 95 > 0) by
construction, since the call site suppresses errexit; publish the found count
beside the observed counter and harden the consuming gate with an
observed == found coverage check (fail 5) immediately after the verbatim-kept
positivity refusal; the END line now carries both numbers. Message vocabulary
(ADJUDICATE / ADJUDICATORS / rc=) is kept; fields are ADDED, none renamed.
Zero found stays rc=95 no_checkpoints_found -- UNMEASURED, unchanged.

Static gates state their denominators, including a MUST_FIRE premise (the
pre-image must genuinely exhibit the streaming read loop AND the bare
positivity gate) and a post-image census of walker loops reported as n of n
clean. Controls C1..C8 are bash-level against real temp directories with a
stubbed runner: C1 sees the regression RED against the pre-image slice and
GREEN against the patched slice; C8 proves space- and newline-bearing directory
names survive the array round-trip. Controls need bash and coreutils only --
no python3 inside the shell, no torch, no transformers. Bare invocation
APPLIES: the build driver runs every stage with no arguments.
"""

from __future__ import annotations

import difflib
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "launch_fs_h100.fixed.sh"
MARK = "fs176:"
N_CONTROLS = 8

# Needles are ASSEMBLED, never written as one literal: this stage counts them in
# the target, and a source that contains its own needle is inside its own
# denominator. The payload text is built from the same fragments for the same
# reason, and prose never spells a scanned needle.
WALK = "adjudicate" + "_tree"
RUNADJ = "run_" + "adjudicators"
OBS = "checkpoint" + "_observed"
FOUNDVAR = "checkpoint" + "_found"

STREAM = ("while IFS= read -r -d '' d; do n=$((n+1)); " + RUNADJ
          + ' "$d" "$phase"; done < <(find')
POSGATE = '[[ "$' + OBS + '" -gt 0 ]] || fail 95'
ENDNEEDLE = "checkpoint_saves" + "_adjudicated=%s"
ENDOLD = ("printf 'END rc=0 phase=%s " + ENDNEEDLE + "\\n' "
          + "\"${FS_PHASE:-train}\" \"$" + OBS + "\" | tee -a \"$RUN_LOG\"")
ENDNEW = ("printf 'END rc=0 phase=%s " + ENDNEEDLE + " checkpoint_saves" + "_found=%s\\n' "
          + "\"${FS_PHASE:-train}\" \"$" + OBS + "\" \"$" + FOUNDVAR + "\" | tee -a \"$RUN_LOG\"")
COVGATE_LINE = ('[[ "$' + OBS + '" -eq "$' + FOUNDVAR + '" ]] || fail 5 "fs176: partial '
                'adjudication coverage: observed=${' + OBS + '} found=${' + FOUNDVAR + '}; only a '
                'fraction of the checkpoint saves on disk was adjudicated -- partial coverage is '
                'not PASS"')

INIT_BLOCK = "\n".join([
    OBS + "=0",
    "# fs176: the walker's INDEPENDENT denominator, published before any adjudicator",
    "# runs. Measured on job 37308: two checkpoint saves on disk against a reported",
    "# denominator of 1, the early save skipped, and the run PASSED. The positivity",
    "# refusal below cannot catch that shape; only observed == found can, and the",
    "# coverage check needs this number to exist whether or not a walk truncated.",
    FOUNDVAR + "=0",
])

COVERAGE_BLOCK = "\n".join([
    "# fs176: positivity is not coverage. Job 37308 PASSED with the observed counter",
    "# at 1 against TWO checkpoint saves on disk; the early save the resume proof",
    "# depends on was never adjudicated and the -gt 0 refusal above certified 1-of-2",
    "# as full. Compare against the independent denominator measured BEFORE any",
    "# adjudicator ran: partial coverage is RED, not PASS.",
    COVGATE_LINE,
])

NEW_FUNC = "\n".join([
    WALK + "() {",
    "  # fs176: collect first, iterate second. Measured on job 37308 -- a run that",
    "  # PASSED. The output tree held TWO checkpoint saves (the early save the resume",
    "  # proof depends on, and the final save) while this plane reported a denominator",
    "  # of 1 and exited 0: the old body invoked the per-checkpoint runner INSIDE the",
    "  # streaming read loop, the container runtime and interpreter inherited that",
    "  # stdin and drained it, and iteration 2's read hit EOF, so only the FIRST",
    "  # directory find emitted was ever adjudicated. The count was incremented inside",
    "  # the same truncated loop, so numerator and denominator were cut together and",
    "  # the report stayed self-consistent -- a denominator derived from the stream it",
    "  # measures cannot detect its own truncation. And the bare call discarded the",
    "  # runner's rc while the `|| ar=$?` call site suppresses errexit throughout",
    "  # this body, so a per-checkpoint REFUSE was dropped twice over.",
    "  #",
    "  # Structure now: (1) the collection loop forks NOTHING in its body, so its",
    "  # stdin cannot be stolen; (2) every runner invocation is fed </dev/null, so a",
    "  # child that wants stdin gets devnull, never the framework's iterator; (3) the",
    "  # found count is measured from the array BEFORE any adjudicator runs -- an",
    "  # independent denominator -- and the processed count must equal it or the walk",
    "  # REFUSES 96, because an iterator that lost entries is a framework defect, not",
    "  # a checkpoint result; (4) per-checkpoint verdicts are captured explicitly and",
    "  # the WORST outcome is returned by construction, since the call site suppresses",
    "  # errexit. Message vocabulary (ADJUDICATE / rc=) is kept; fields are ADDED,",
    "  # none renamed.",
    "  local root=\"$1\" phase=\"$2\" d rc=0",
    "  local -a dirs=()",
    "  [[ -d \"$root\" ]] || { printf 'ADJUDICATE rc=96 root_missing=%s\\n' \"$root\" >&2; return 96; }",
    "  while IFS= read -r -d '' d; do dirs+=(\"$d\"); done < <(find \"$root\" -type d \\( -name 'checkpoint*' -o -name 'ckpt*' -o -name 'step_*' \\) -print0 2>/dev/null)",
    "  local found=${#dirs[@]} processed=0 ok=0 abstain=0 refuse=0",
    "  " + FOUNDVAR + "=$found",
    "  [[ \"$found\" -gt 0 ]] || { printf 'ADJUDICATE rc=95 no_checkpoints_found root=%s phase=%s found=0\\n' \"$root\" \"$phase\" >&2; return 95; }",
    "  for d in \"${dirs[@]}\"; do",
    "    rc=0",
    "    " + RUNADJ + " \"$d\" \"$phase\" </dev/null || rc=$?",
    "    processed=$((processed+1))",
    "    case $rc in",
    "      0)  ok=$((ok+1)) ;;",
    "      95) abstain=$((abstain+1)) ;;",
    "      96) refuse=$((refuse+1)) ;;",
    "      *)",
    "        printf 'ADJUDICATE rc=96 original_rc=%s ckpt=%s -- undeclared exit code mapped to 96\\n' \"$rc\" \"$d\" >&2",
    "        refuse=$((refuse+1))",
    "        ;;",
    "    esac",
    "  done",
    "  [[ \"$processed\" -eq \"$found\" ]] || { printf 'ADJUDICATE rc=96 iterator_truncated processed=%s found=%s root=%s phase=%s -- the walk lost entries; a framework defect, not a checkpoint result\\n' \"$processed\" \"$found\" \"$root\" \"$phase\" >&2; return 96; }",
    "  printf 'ADJUDICATE complete root=%s phase=%s adjudicated=%s of %s checkpoint dir(s) ok=%s abstain=%s refuse=%s\\n' \"$root\" \"$phase\" \"$processed\" \"$found\" \"$ok\" \"$abstain\" \"$refuse\"",
    "  if (( refuse > 0 )); then return 96; fi",
    "  if (( abstain > 0 )); then return 95; fi",
    "  return 0",
    "}",
])

FORBIDDEN = [
    ("filesystem-root literal", re.compile(r"/(?:home|Users|root|data|mnt|nfs|lustre|scratch|srv|opt)/")),
    ("IP literal", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("DNS-style host name", re.compile(r"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+\.(?:com|net|org|edu|gov|io|local|lan)\b", re.I)),
]


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _transform(text: str) -> tuple[str, bool]:
    if MARK in text:
        return text, True
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.rstrip("\n") == OBS + "=0":
            lines[i] = INIT_BLOCK + "\n"
            break
    for i, ln in enumerate(lines):
        if ln.startswith(WALK + "() {"):
            j = i
            while j < len(lines) and lines[j].rstrip("\n") != "}":
                j += 1
            if j < len(lines):
                lines[i:j + 1] = [NEW_FUNC + "\n"]
            break
    for i, ln in enumerate(lines):
        if POSGATE in ln:
            lines.insert(i + 1, COVERAGE_BLOCK + "\n")
            break
    for i, ln in enumerate(lines):
        if ENDOLD in ln:
            lines[i] = ln.replace(ENDOLD, ENDNEW)
            break
    return "".join(lines), False


def _added_lines(pre: str, post: str) -> list[str]:
    return [ln[2:] for ln in difflib.ndiff(pre.splitlines(), post.splitlines()) if ln.startswith("+ ")]


def _g8_forbidden(added: list[str]) -> list[str]:
    bad = []
    for ln in added:
        for label, rx in FORBIDDEN:
            if rx.search(ln):
                bad.append(f"{label}: {ln.strip()[:100]}")
    return bad


def _walker_census(text: str) -> tuple[int, int, list[tuple[str, str]]]:
    # Census, not a spot-check: EVERY while-loop that reads a find -print0 stream is
    # classified. A loop is dirty when its body can start a process that reads stdin
    # (the per-checkpoint runner, a container runner, an interpreter, cat) -- the
    # D1 shape. A body of pure builtins/assignments cannot have its stdin stolen.
    lines = text.splitlines()
    segs: list[str] = []
    i = 0
    while i < len(lines):
        if "while" in lines[i] and "read -r -d" in lines[i]:
            j = i
            seg = [lines[i]]
            while "done < <(find" not in lines[j] and j + 1 < len(lines) and j - i < 60:
                j += 1
                seg.append(lines[j])
            if "done < <(find" in lines[j]:
                segs.append("\n".join(seg))
                i = j
        i += 1
    pats = [(RUNADJ, re.compile(re.escape(RUNADJ))),
            ("run_in_container", re.compile(r"\brun_in_container\b")),
            ("python", re.compile(r"\bpython3?\b")),
            ("cat", re.compile(r"\bcat\b"))]
    total = len(segs)
    clean = 0
    dirty: list[tuple[str, str]] = []
    for s in segs:
        m = re.search(r"\bdo\b(.*?)\bdone\b", s, re.S)
        body = m.group(1) if m else s
        hits = [label for label, rx in pats if rx.search(body)]
        if hits:
            dirty.append((s.splitlines()[0].strip()[:60], ",".join(hits)))
        else:
            clean += 1
    return total, clean, dirty


def _awk_slice(text: str, td: pathlib.Path, tag: str) -> str:
    # Extract the single walker function BY NAME with awk, from its header to the
    # matching closing brace at column 0. The slice -- never the whole launcher --
    # is what the controls source.
    whole = td / ("whole-" + tag + ".sh")
    whole.write_text(text, "utf-8")
    prog = "/^" + WALK + "[(][)] [{]/{f=1} f{print} f&&/^[}]/{exit}"
    r = subprocess.run(["awk", prog, str(whole)], capture_output=True, text=True, timeout=30)
    out = r.stdout
    if r.returncode != 0 or not out.startswith(WALK + "() {") or not out.rstrip().endswith("}"):
        return ""
    return out


def _run_walk(func_text: str, stub: str, root: pathlib.Path, td: pathlib.Path,
              tag: str, extra: dict[str, str] | None = None) -> tuple[int, str, str]:
    # Drive the extracted walker in a subshell that mirrors the production call
    # site: set -Eeuo pipefail, the runner stubbed per control, and the walk
    # invoked on the left of `||` so errexit is suppressed throughout its body --
    # the function must return the right code BY CONSTRUCTION.
    ff = td / (tag + "-func.sh")
    ff.write_text(func_text, "utf-8")
    hf = td / (tag + "-harness.sh")
    hf.write_text("\n".join([
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        OBS + "=0",
        FOUNDVAR + "=0",
        RUNADJ + "() {",
        stub,
        "}",
        ". " + shlex.quote(str(ff)),
        "rc=0",
        WALK + " \"$ROOT\" train || rc=$?",
        "printf 'WALK_RC=%s\\n' \"$rc\"",
        "exit 0",
    ]) + "\n", "utf-8")
    env = dict(os.environ)
    env["ROOT"] = str(root)
    for k, v in (extra or {}).items():
        env[k] = v
    r = subprocess.run(["bash", str(hf)], capture_output=True, text=True,
                       env=env, cwd=str(td), timeout=30)
    m = re.search(r"^WALK_RC=(\d+)$", r.stdout, re.M)
    return (int(m.group(1)) if m else -1), r.stdout, r.stderr


def _controls(pre: str, new: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    ok = 0
    with tempfile.TemporaryDirectory(prefix="fs176-") as tds:
        td = pathlib.Path(tds)
        pre_fn = _awk_slice(pre, td, "pre")
        post_fn = _awk_slice(new, td, "post")
        if not pre_fn or not post_fn:
            notes.append("provenance: FAIL could not slice the walker by name with awk from "
                         f"the pre-image ({len(pre_fn)}B) and post-image ({len(post_fn)}B); "
                         "controls not interpretable")
            return 0, notes
        # Uncounted provenance preflight: bash is an external binary here, and a
        # harness that cannot run a trivial walk yields the same nonzero codes a
        # real refusal does, which would confound every MUST_FIRE below.
        root0 = td / "c0-root"
        (root0 / "checkpoint-probe").mkdir(parents=True)
        rc0, out0, err0 = _run_walk(post_fn, "return 0", root0, td, "c0")
        if rc0 != 0 or "adjudicated=1 of 1" not in out0:
            notes.append(f"provenance: FAIL trivial walk rc={rc0} out={out0.strip()[:120]} "
                         f"err={err0.strip()[:120]}; no MUST_FIRE below would be attributable")
            return 0, notes

        def mk3(root: pathlib.Path) -> None:
            root.mkdir()
            for n in ("checkpoint-step-00000050", "checkpoint-step-00000200", "ckpt-final"):
                (root / n).mkdir()

        # C1 MUST_FIRE -- the regression, and it must be seen red. Three matching
        # dirs on disk; the stub DRAINS STDIN, exactly what the container did on
        # job 37308. The pre-image slice must yield 1 (the defect, red); the
        # patched slice must yield 3. Both halves are asserted: a control that
        # only ever sees green is not a control.
        root1 = td / "c1-root"
        mk3(root1)
        drain = "cat >/dev/null\nreturn 0"
        rc_pre, out_pre, _ = _run_walk(pre_fn, drain, root1, td, "c1pre")
        rc_post, out_post, _ = _run_walk(post_fn, drain, root1, td, "c1post")
        good = (rc_pre == 0 and "checkpoint_dirs=1" in out_pre
                and rc_post == 0 and "adjudicated=3 of 3" in out_post)
        ok += int(good)
        notes.append(f"C1 regression (stdin-draining stub, 3 dirs on disk): pre-image rc={rc_pre} "
                     f"red-half={'1-of-3' if 'checkpoint_dirs=1' in out_pre else 'UNEXPECTED'} "
                     f"patched rc={rc_post} green-half={'3-of-3' if 'adjudicated=3 of 3' in out_post else 'UNEXPECTED'} "
                     + ("PASS" if good else "FAIL"))

        # C2 MUST_PASS: three dirs, a well-behaved stub -> processed 3, found 3, rc 0.
        root2 = td / "c2-root"
        mk3(root2)
        rc, out, err = _run_walk(post_fn, "return 0", root2, td, "c2")
        good = rc == 0 and "adjudicated=3 of 3" in out and "ok=3" in out
        ok += int(good)
        notes.append(f"C2 well-behaved stub: rc={rc} expected=0, "
                     f"shape={'3-of-3 ok=3' if good else out.strip()[:120]} "
                     + ("PASS" if good else "FAIL " + err.strip()[:120]))

        # C3 MUST_FIRE: stub refuses 96 on the second dir -> walk returns 96, and
        # the other two are STILL visited: a refusal must not truncate the walk.
        root3 = td / "c3-root"
        mk3(root3)
        visits3 = td / "c3-visits"
        stub96 = ("printf '%s\\n' \"$(basename -- \"$1\")\" >> \"$VISITS\"\n"
                  "if [[ \"$(basename -- \"$1\")\" == \"$BAD\" ]]; then return 96; fi\n"
                  "return 0")
        rc, out, err = _run_walk(post_fn, stub96, root3, td, "c3",
                                 {"VISITS": str(visits3), "BAD": "checkpoint-step-00000200"})
        nvis = len(visits3.read_text("utf-8").splitlines()) if visits3.exists() else 0
        good = rc == 96 and nvis == 3 and "refuse=1" in out
        ok += int(good)
        notes.append(f"C3 one refuser: rc={rc} expected=96, visited={nvis} expected=3, "
                     f"refuse-field={'1' if 'refuse=1' in out else '?'} "
                     + ("PASS" if good else "FAIL"))

        # C4 MUST_FIRE: stub abstains 95 on one dir, 0 elsewhere -> 95, not 0, not 96.
        root4 = td / "c4-root"
        mk3(root4)
        visits4 = td / "c4-visits"
        stub95 = ("printf '%s\\n' \"$(basename -- \"$1\")\" >> \"$VISITS\"\n"
                  "if [[ \"$(basename -- \"$1\")\" == \"$BAD\" ]]; then return 95; fi\n"
                  "return 0")
        rc, out, err = _run_walk(post_fn, stub95, root4, td, "c4",
                                 {"VISITS": str(visits4), "BAD": "checkpoint-step-00000200"})
        nvis = len(visits4.read_text("utf-8").splitlines()) if visits4.exists() else 0
        good = rc == 95 and nvis == 3 and "abstain=1" in out and "refuse=0" in out
        ok += int(good)
        notes.append(f"C4 one abstention: rc={rc} expected=95, visited={nvis} expected=3, "
                     f"abstain-field={'1' if 'abstain=1' in out else '?'} "
                     + ("PASS" if good else "FAIL"))

        # C5 MUST_FIRE: empty root, zero matching dirs -> 95 with the zero
        # denominator stated. UNMEASURED, unchanged.
        root5 = td / "c5-root"
        root5.mkdir()
        (root5 / "not-a-matching-name").mkdir()
        rc, out, err = _run_walk(post_fn, "return 0", root5, td, "c5")
        good = rc == 95 and "no_checkpoints_found" in err and "found=0" in err
        ok += int(good)
        notes.append(f"C5 zero found: rc={rc} expected=95, "
                     f"denominator-stated={'found=0' if 'found=0' in err else '?'} "
                     + ("PASS" if good else "FAIL " + err.strip()[:120]))

        # C6 MUST_FIRE: missing root -> 96.
        rc, out, err = _run_walk(post_fn, "return 0", td / "c6-missing", td, "c6")
        good = rc == 96 and "root_missing" in err
        ok += int(good)
        notes.append(f"C6 missing root: rc={rc} expected=96, "
                     f"root_missing={'yes' if 'root_missing' in err else 'no'} "
                     + ("PASS" if good else "FAIL"))

        # C7 MUST_PASS: precedence -- one 95 and one 96 in the same walk returns
        # 96. Worst wins.
        root7 = td / "c7-root"
        mk3(root7)
        stubmix = ("if [[ \"$(basename -- \"$1\")\" == \"$A95\" ]]; then return 95; fi\n"
                   "if [[ \"$(basename -- \"$1\")\" == \"$B96\" ]]; then return 96; fi\n"
                   "return 0")
        rc, out, err = _run_walk(post_fn, stubmix, root7, td, "c7",
                                 {"A95": "checkpoint-step-00000050", "B96": "ckpt-final"})
        good = rc == 96 and "abstain=1" in out and "refuse=1" in out
        ok += int(good)
        notes.append(f"C7 worst-wins (one 95, one 96): rc={rc} expected=96, "
                     f"fields={'abstain=1 refuse=1' if ('abstain=1' in out and 'refuse=1' in out) else out.strip()[:100]} "
                     + ("PASS" if good else "FAIL"))

        # C8 MUST_FIRE: directory names containing SPACES and NEWLINES survive the
        # array round-trip and are counted once each. The walker already uses
        # -print0; this proves the array preserves it, because a naive $(find)
        # rewrite would pass every other control and silently corrupt this one.
        names8 = ["checkpoint with space", "step_two\nlines", "ckpt.plain"]
        root8 = td / "c8-root"
        root8.mkdir()
        for n in names8:
            (root8 / n).mkdir()
        visits8 = td / "c8-visits"
        stubnul = "printf '%s\\0' \"$1\" >> \"$VISITS\"\nreturn 0"
        rc, out, err = _run_walk(post_fn, stubnul, root8, td, "c8", {"VISITS": str(visits8)})
        got = [g for g in (visits8.read_bytes().split(b"\0") if visits8.exists() else []) if g]
        exp = {str(root8 / n).encode() for n in names8}
        good = rc == 0 and "adjudicated=3 of 3" in out and len(got) == 3 and set(got) == exp
        ok += int(good)
        notes.append(f"C8 hostile names (space + newline): rc={rc} expected=0, "
                     f"visited={len(got)} expected=3, names-intact={'yes' if set(got) == exp else 'no'} "
                     + ("PASS" if good else "FAIL"))
    return ok, notes


def main() -> int:
    # build_h100_plane.sh invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would have made the stage a no-op
    # inside the build while passing by hand -- the #86 orphan shape, one layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_adjudicate_denominator.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    if not TARGET.exists():
        _stderr(f"UNMEASURED 95: target missing: {TARGET}")
        return 95
    try:
        text = TARGET.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: target unreadable: {exc}")
        return 95

    new, already = _transform(text)
    if already:
        print("verdict: already applied; byte-idempotent no-op")
        return 0

    gates = 0
    gres: list[tuple[str, bool, str]] = []
    gres.append(("G1", text.count(WALK + "() {") == 1,
                 f"walker header count={text.count(WALK + '() {')} need=1"))
    prem_stream = STREAM in text
    prem_gate = POSGATE in text
    gres.append(("G2", prem_stream and prem_gate,
                 f"MUST_FIRE premise: streaming read loop with in-body fork present={prem_stream}, "
                 f"bare positivity gate present={prem_gate} -- both required, so the stage cannot "
                 "silently no-op on a file it does not recognise"))
    init_n = sum(1 for ln in text.splitlines() if ln == OBS + "=0")
    gres.append(("G3", init_n == 1 and text.count(ENDOLD) == 1,
                 f"observed-counter init lines={init_n} need=1, END line count={text.count(ENDOLD)} need=1"))
    total, clean, dirty = _walker_census(new)
    gres.append(("G4", total >= 1 and clean == total,
                 f"post-image walker-loop census: {clean} of {total} loops clean (dirty = body can "
                 "start a process that reads stdin)"
                 + ("" if not dirty else " DIRTY: " + "; ".join(f"{h} -> {t}" for h, t in dirty[:2]))))
    found_init_n = sum(1 for ln in new.splitlines() if ln == FOUNDVAR + "=0")
    g5 = (found_init_n == 1
          and new.count(COVGATE_LINE) == 1
          and new.count(ENDNEW) == 1
          and POSGATE in new
          and STREAM not in new
          and "adjudicated=%s of %s" in new
          and new != text)
    gres.append(("G5", g5, f"post-image shape: found-counter init={found_init_n} need=1, "
                           f"coverage gate={new.count(COVGATE_LINE)} need=1, END-carries-both="
                           f"{new.count(ENDNEW)} need=1, positivity-kept-verbatim={POSGATE in new}, "
                           f"defect-loop-gone={STREAM not in new}, changed={new != text}"))
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(new)
        tmp = fh.name
    try:
        bn = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
    finally:
        os.unlink(tmp)
    gres.append(("G6", bn.returncode == 0, "bash -n " + (bn.stderr.strip()[:160] if bn.returncode else "clean")))
    again, _ = _transform(new)
    gres.append(("G7", again == new, "byte-idempotence on own output"))
    bad8 = _g8_forbidden(_added_lines(text, new))
    gres.append(("G8", not bad8, f"added lines carrying host/node/partition/user/fs-root "
                                 f"literals={len(bad8)} " + "; ".join(bad8[:2])))

    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    cok, cnotes = _controls(text, new)
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls {cok}/{N_CONTROLS}; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # This stage exists because a plane collapsed four states into one green lie,
    # so it must not collapse its own: an unhandled exception is a REFUSE, not a
    # bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())