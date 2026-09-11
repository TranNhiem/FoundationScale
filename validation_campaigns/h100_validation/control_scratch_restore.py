#!/usr/bin/env python3
"""
control_scratch_restore.py -- positive control for finding #294 in the
FoundationScale H100 validation campaign.

WHY THIS FILE IS PYTHON, NOT BASH.
  checks/campaign_self_tests.py discovers self-tests as *tracked campaign
  files ending in .py that contain the literal --self-test*. A .sh control
  would be discovered by nothing and would be an orphan on arrival -- the
  exact defect class (#278/#293/#352) this campaign keeps finding. The two
  bash drafts this file merges are used for their LEG LOGIC only; their
  shell mechanics (sed -i portability, grep -c rc absorption, set -e guards)
  are replaced by stdlib Python.

WHAT IT CERTIFIES (#294).
  build_h100_plane.sh "rebuilds from scratch": it rm -f's eight GENERATED,
  GIT-TRACKED artifacts, then its stages regenerate them. On a public clone
  the first stage's private upstream is absent, so the first stage refuses
  (rc 95/96, the stage loop `break`s), NOTHING is regenerated, and the
  operator was left holding eight deletions. The fix snapshots the eight
  into $TMPDIR/h100-build-294.* BEFORE the rm and, at EXIT, restores only
  those still absent -- composed with the pre-existing roll_call_gates
  trap, because bash EXIT traps REPLACE, they never chain.

WHY TWO ARMS.
  Arm A runs the campaign dir copied verbatim (fix armed). Arm B is a
  second copy with BOTH trap installs the fix added stripped back out --
  the pre-fix build. A green on A alone would be consistent with the rm
  never having fired; a red on B alone with the fix never having existed.
  Only the A/B pair makes the green ATTRIBUTABLE: byte-identical trees,
  identical synthetic knobs, one difference (the trap), opposite outcomes.

EXIT CONTRACT (campaign-wide): 0=CLEAR, 5=RED, 95=UNMEASURED, 96=REFUSE.
  This file never exits 1: main() is wrapped, and any uncaught exception
  becomes 95, because a crashing control measured nothing.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXIT_CLEAR = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96
_VALID = (EXIT_CLEAR, EXIT_RED, EXIT_UNMEASURED, EXIT_REFUSE)

# The two anchor lines the #294 fix added, at :518 and :713 of the sibling
# build script. Arm B is built by deleting the first and rewriting the
# second back to its pre-fix form. bash EXIT traps replace, never chain --
# that is exactly why the fix had to compose rather than install.
ANCHOR_BARE = "trap restore_scratch_294 EXIT"
ANCHOR_COMPOSED = "trap 'roll_call_gates || :; restore_scratch_294' EXIT"
COMPOSED_REPLACEMENT = "trap roll_call_gates EXIT"

REACHED_BANNER = "rebuilding from scratch"
ROLLCALL_BANNER = "standing gates:"
SNAPSHOT_GLOB = "h100-build-294.*"
N_ARTIFACTS = 8
ARM_TIMEOUT_S = 600

# SYNTHETIC knobs -- the only ones that gate the rm, and deliberately
# fabricated here so a public runner with no estate secrets gets both arms
# PAST the required-knob guard and into the refuse-LATE path (after the
# rm) that exercises #294. The operator's real estate env file is never
# read; the estate root is a directory this control creates itself.
KNOB_PARTITION = "synthetictestpartition"
KNOB_NONE = "NONE"


class ParseError(Exception):
    """Raised when the rm -f line or its variables cannot be resolved."""


# --------------------------------------------------------------------------
# Instrument: parse the eight artifact paths out of build_h100_plane.sh.
# We do NOT hardcode the eight filenames: a hardcoded list silently stops
# measuring the day a ninth artifact is added. Anything other than one
# unambiguous rm -f line resolving to exactly 8 unique in-tree paths is
# parse failure -> UNMEASURED (95), per contract.
# --------------------------------------------------------------------------
_ASSIGN_RE = re.compile(r"^\s*(?:(?:readonly|export|local)\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_VARUSE_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_VARFULL_RE = re.compile(r"^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$")
_RM_RE = re.compile(r"^\s*rm\s+-f\s+(?P<args>.+?)\s*$")


def _reduce_cd_idioms(value: str) -> str:
    # The canonical `$(cd "$(dirname ...)" && pwd)` prefix is not statically
    # resolvable, but relative to the build script it is "." -- treat it as
    # such so the artifact paths land inside the campaign tree. Balanced-
    # paren scan because the idiom nests $(dirname ...).
    out: list[str] = []
    i = 0
    while i < len(value):
        if value.startswith("$(", i):
            # Start the scan ON the '(', not on the '$'. Starting on '$' leaves
            # depth at 0 through the first character, so the loop guard exits
            # immediately and the group is never entered -- the idiom then
            # survives to _expand() and raises "unresolved shell expansion".
            # Caught by this control's own self-test (#294).
            depth = 0
            j = i + 1
            while j < len(value):
                if value[j] == "(":
                    depth += 1
                elif value[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            inner = value[i + 2 : j - 1]
            if depth == 0 and "cd" in inner and "pwd" in inner and "dirname" in inner:
                out.append(".")
            else:
                out.append(value[i:j])
            i = j
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def _expand(value: str, table: dict[str, str], stack: tuple[str, ...]) -> str:
    value = _reduce_cd_idioms(_unquote(value))

    def repl(m: re.Match[str]) -> str:
        var = m.group(1) or m.group(2)
        if var in stack:
            raise ParseError(f"variable cycle: {' -> '.join(stack + (var,))}")
        if var not in table:
            raise ParseError(f"${var} is used by the rm targets but has no assignment")
        return _expand(table[var], table, stack + (var,))

    for _ in range(32):
        new = _VARUSE_RE.sub(repl, value)
        if new == value:
            break
        value = new
    if "$" in value:
        raise ParseError(f"unresolved shell expansion remains in {value!r}")
    return _unquote(value)


def _resolve_token(tok: str, table: dict[str, str]) -> str:
    m = _VARFULL_RE.match(tok)
    if not m:
        if "$" in tok:
            return _expand(tok, table, ())
        return tok
    var = m.group(1) or m.group(2)
    if var not in table:
        raise ParseError(f"${var} on the rm line has no assignment")
    return _expand(table[var], table, (var,))


def parse_rm_targets(text: str) -> list[str]:
    """Return the rm -f line's targets, resolved to campaign-relative paths."""
    table: dict[str, str] = {}
    for line in text.splitlines():
        m = _ASSIGN_RE.match(line)
        if m:
            table[m.group(1)] = m.group(2)  # last assignment wins, shell-style

    hits: list[list[str]] = []
    errors: list[str] = []
    for line in text.splitlines():
        m = _RM_RE.match(line)
        if not m:
            continue
        args = m.group("args").split("#", 1)[0]
        try:
            tokens = shlex.split(args, posix=True)
            resolved = [_resolve_token(t, table) for t in tokens]
        except (ValueError, ParseError) as exc:
            errors.append(str(exc))
            continue
        rels: list[str] = []
        bad = False
        for r in resolved:
            r2 = os.path.normpath(r)
            if Path(r2).is_absolute() or r2 == ".." or r2.startswith(".." + os.sep):
                errors.append(f"target {r!r} escapes the campaign tree")
                bad = True
                break
            rels.append(r2.replace(os.sep, "/"))
        if bad:
            continue
        if len(rels) == N_ARTIFACTS and len(set(rels)) == N_ARTIFACTS:
            hits.append(rels)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        detail = "; ".join(errors) if errors else "no rm -f candidate lines"
        raise ParseError(
            f"no rm -f line resolves to exactly {N_ARTIFACTS} unique in-tree targets ({detail})"
        )
    raise ParseError(f"{len(hits)} rm -f lines each resolve to {N_ARTIFACTS} targets; ambiguous")


# --------------------------------------------------------------------------
# Instruments: the detectors. A control whose detectors cannot fire
# certifies nothing, so every one of these is exercised over a planted
# fixture with a known answer by --self-test below.
# --------------------------------------------------------------------------
def count_exact_lines(text: str, target: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip() == target)


def strip_anchors(text: str) -> str:
    """Arm-B mutation: delete the bare trap install, rewrite the composed one."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s == ANCHOR_BARE:
            continue
        if s == ANCHOR_COMPOSED:
            indent = line[: len(line) - len(line.lstrip())]
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}{COMPOSED_REPLACEMENT}{nl}")
        else:
            out.append(line)
    return "".join(out)


def count_survivors(tree: Path, rels: list[str]) -> int:
    return sum(1 for r in rels if (tree / r).is_file())


def snapshot_bytes(tree: Path, rels: list[str]) -> dict[str, bytes]:
    snap: dict[str, bytes] = {}
    for r in rels:
        p = tree / r
        if p.is_file():
            snap[r] = p.read_bytes()
    return snap


def byte_mismatches(before: dict[str, bytes], tree: Path, rels: list[str]) -> list[str]:
    bad: list[str] = []
    for r in rels:
        p = tree / r
        if not p.is_file() or before.get(r) != p.read_bytes():
            bad.append(r)
    return bad


def snapshot_leaks(tmpdir: Path) -> list[str]:
    if not tmpdir.is_dir():
        return []
    return sorted(p.name for p in tmpdir.glob(SNAPSHOT_GLOB) if p.is_dir())


def run_arm(build: Path, cwd: Path, env: dict[str, str]):
    """Returns (rc, merged_output, timed_out). macOS has no `timeout` binary,
    so the subprocess timeout lives here, in Python."""
    try:
        proc = subprocess.run(
            ["bash", str(build)],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=ARM_TIMEOUT_S,
            text=True,
            errors="replace",
        )
        return proc.returncode, proc.stdout, False
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return None, out, True
    except OSError as exc:
        return None, f"(control could not exec bash: {exc})", True


def arm_env(work: Path, tmpdir: Path) -> dict[str, str]:
    estate = work / "estate-root-synthetic"
    estate.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["FS_ESTATE_ROOT"] = str(estate)
    env["FS_PARTITION_LITERAL"] = KNOB_PARTITION
    env["FS_ESTATE_IDENT_PAT"] = KNOB_NONE
    env["FS_REDACT_EXTRA"] = KNOB_NONE
    env["TMPDIR"] = str(tmpdir)
    return env


def _aggregate(legs: list[tuple[str, str, str]]) -> int:
    # RED outranks abstention: a measured failure is a stronger statement
    # than "could not measure".
    statuses = [s for _, s, _ in legs]
    if "FAIL" in statuses:
        return EXIT_RED
    if "UNMEASURED" in statuses:
        return EXIT_UNMEASURED
    return EXIT_CLEAR


def emit(legs: list[tuple[str, str, str]], code: int) -> None:
    for name, status, reason in legs:
        print(f"LEG {name} {status}: {reason}", flush=True)
    print()
    print(f"{'LEG':<6}{'STATUS':<12}WHY")
    for name, status, reason in legs:
        print(f"{name:<6}{status:<12}{reason}")
    if code == EXIT_CLEAR:
        print(
            "VERDICT 0: CLEAR -- arm A restored 8/8 byte-identical, arm B "
            "lost 8/8; the green is attributable to the #294 restore and "
            "to nothing else"
        )
    elif code == EXIT_RED:
        print(
            "VERDICT 5: RED -- at least one MUST_PASS/MUST_FIRE leg failed "
            "against the measured expectation"
        )
    else:
        print(
            "VERDICT 95: UNMEASURED -- a precondition could not be "
            "established; this control makes no claim, because a vacuous "
            "green is worse than an abstention"
        )


# --------------------------------------------------------------------------
# --self-test: controls-of-the-control. Runs entirely in a temp dir, no
# estate environment, no network, no repo mutation, well under 10s. Every
# detector is fired both ways over planted fixtures with known answers.
# --------------------------------------------------------------------------
def self_test() -> int:
    fails = 0

    def st(ok: bool, msg: str) -> None:
        nonlocal fails
        print(f"SELF-TEST {'PASS' if ok else 'RED'}: {msg}", flush=True)
        if not ok:
            fails = 1

    fake_names = [f"artifact_{i:02d}.sh" for i in range(1, 9)]
    with tempfile.TemporaryDirectory(prefix="ctrl294-selftest.") as td:
        t = Path(td)

        fake_build = "\n".join(
            [
                "#!/usr/bin/env bash",
                'ROOT="$(cd "$(dirname "$0")" && pwd)"',
                'H100="$ROOT/h100"',
                'GEN="$H100/gen"',
                *[f'V{i}="$GEN/{fake_names[i - 1]}"' for i in range(1, 9)],
                'rm -f "$V1" "$V2" "$V3" "$V4" "$V5" "$V6" "$V7" "$V8"',
                ANCHOR_BARE,
                "echo placeholder",
                ANCHOR_COMPOSED,
                "",
            ]
        )
        rels: list[str] = []
        try:
            rels = parse_rm_targets(fake_build)
            want = [f"h100/gen/{n}" for n in fake_names]
            st(
                rels == want,
                "rm-line parser resolves nested variables through "
                "the cd/pwd idiom to 8 in-tree paths",
            )
        except ParseError as exc:
            st(False, f"rm-line parser raised on a planted fixture: {exc}")

        stripped = strip_anchors(fake_build)
        st(
            count_exact_lines(stripped, ANCHOR_BARE) == 0
            and count_exact_lines(stripped, ANCHOR_COMPOSED) == 0
            and count_exact_lines(stripped, COMPOSED_REPLACEMENT) == 1,
            "strip_anchors removes both anchors and leaves one bare roll-call trap",
        )

        empty = t / "empty"
        full = t / "full" / "h100" / "gen"
        part = t / "part" / "h100" / "gen"
        empty.mkdir()
        full.mkdir(parents=True)
        part.mkdir(parents=True)
        use = rels if rels else [f"h100/gen/{n}" for n in fake_names]
        for r in use:
            (t / "full" / r).parent.mkdir(parents=True, exist_ok=True)
            (t / "full" / r).write_text("x")
        for r in use[:2]:
            (t / "part" / r).parent.mkdir(parents=True, exist_ok=True)
            (t / "part" / r).write_text("x")
        st(count_survivors(t / "empty", use) == 0, "survivor counter fires low (0)")
        st(count_survivors(t / "full", use) == 8, "survivor counter fires high (8)")
        st(count_survivors(t / "part", use) == 2, "survivor counter counts a partial dir (2)")

        before = snapshot_bytes(t / "full", use)
        st(byte_mismatches(before, t / "full", use) == [], "byte compare passes identical trees")
        (t / "full" / use[0]).write_text("tampered")
        st(
            byte_mismatches(before, t / "full", use) == [use[0]],
            "byte compare fires on a one-byte-content difference",
        )

        (t / "h100-build-294.planted").mkdir()
        st(
            snapshot_leaks(t) == ["h100-build-294.planted"],
            "snapshot-leak detector finds a planted h100-build-294.* dir",
        )
        st(snapshot_leaks(t / "empty") == [], "snapshot-leak detector is quiet on an empty TMPDIR")

        log = "header\nrebuilding from scratch\nstanding gates: demo\nstanding gates: demo2\n"
        st("rebuilding from scratch" in log, "reached-site banner detected when planted")
        st(log.count(ROLLCALL_BANNER) == 2, "roll-call counter sees a doubled banner as 2")

        st(_aggregate([("L", "PASS", "")]) == EXIT_CLEAR, "verdict precedence: all-pass -> 0")
        st(
            _aggregate([("L", "UNMEASURED", "")]) == EXIT_UNMEASURED,
            "verdict precedence: unmeasured -> 95",
        )
        st(
            _aggregate([("L", "FAIL", ""), ("L", "UNMEASURED", "")]) == EXIT_RED,
            "verdict precedence: RED outranks abstention -> 5",
        )

    if fails:
        print(
            "SELF-TEST VERDICT 5: an instrument cannot fire; this control certifies nothing",
            flush=True,
        )
        return EXIT_RED
    print("SELF-TEST VERDICT 0: all instruments fire both ways", flush=True)
    return EXIT_CLEAR


# --------------------------------------------------------------------------
# The real two-arm run.
# --------------------------------------------------------------------------
def run_control() -> int:
    legs: list[tuple[str, str, str]] = []

    def leg(name: str, status: str, reason: str) -> None:
        legs.append((name, status, reason))

    def abort_all(reason: str) -> int:
        have = {n for n, _, _ in legs}
        for name in ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"):
            if name not in have:
                leg(name, "UNMEASURED", reason)
        code = _aggregate(legs)
        emit(legs, code)
        return code

    campaign = Path(__file__).resolve().parent
    build_src = campaign / "build_h100_plane.sh"
    if not build_src.is_file():
        return abort_all(
            f"build_h100_plane.sh not found next to this control ({build_src}); nothing to prove"
        )
    text = build_src.read_text()

    # ---- L0 ARM-B CONSTRUCTION ------------------------------------------
    # Both anchors must occur EXACTLY once in the source and ZERO times
    # after the strip. A "pre-fix" arm that was never actually stripped
    # would silently be a second copy of arm A and the A/B attribution
    # would be gone while looking green.
    n_bare = count_exact_lines(text, ANCHOR_BARE)
    n_composed = count_exact_lines(text, ANCHOR_COMPOSED)
    pre_strip = strip_anchors(text)
    z_bare = count_exact_lines(pre_strip, ANCHOR_BARE)
    z_composed = count_exact_lines(pre_strip, ANCHOR_COMPOSED)
    n_repl = count_exact_lines(pre_strip, COMPOSED_REPLACEMENT)
    l0_ok = n_bare == 1 and n_composed == 1 and z_bare == 0 and z_composed == 0 and n_repl == 1
    if l0_ok:
        leg(
            "L0",
            "PASS",
            "anchors occur exactly once each in the source "
            "(:518, :713) and zero times after the strip; "
            "arm B is genuinely the pre-fix build",
        )
    else:
        leg(
            "L0",
            "UNMEASURED",
            f"anchor counts wrong: source bare={n_bare} "
            f"composed={n_composed} (want 1/1); after strip "
            f"bare={z_bare} composed={z_composed} "
            f"replacement={n_repl} (want 0/0/1) -- anchors "
            f"drifted, arm B cannot be construed",
        )

    # ---- artifact discovery (never hardcoded) -----------------------------
    artifacts: list[str] | None = None
    parse_err = ""
    try:
        artifacts = parse_rm_targets(text)
    except ParseError as exc:
        parse_err = str(exc)
    if artifacts is None:
        return abort_all(f"could not derive the artifact set from build_h100_plane.sh: {parse_err}")

    baseline = count_survivors(campaign, artifacts)
    if baseline != N_ARTIFACTS or not l0_ok:
        why = []
        if baseline != N_ARTIFACTS:
            why.append(f"campaign tree holds {baseline}/{N_ARTIFACTS} baseline artifacts")
        if not l0_ok:
            why.append("L0 anchor verification failed; arm B untrustworthy")
        return abort_all("precondition failed: " + "; ".join(why))

    # ---- isolated A/B copies; the real repo tree is never touched ---------
    manager = tempfile.TemporaryDirectory(prefix="ctrl294.")
    work = Path(manager.name)
    try:
        arm_a = work / "A"
        arm_b = work / "B"
        shutil.copytree(campaign, arm_a, symlinks=True)
        shutil.copytree(campaign, arm_b, symlinks=True)
        (arm_b / "build_h100_plane.sh").write_text(pre_strip)

        # The strip is verified AGAIN on the actual arm-B file, not just on a
        # string in memory: trusting a mutation that never took would turn a
        # red L5 into a misread "the fix works".
        b_text = (arm_b / "build_h100_plane.sh").read_text()
        strip_ok = (
            count_exact_lines(b_text, ANCHOR_BARE) == 0
            and count_exact_lines(b_text, ANCHOR_COMPOSED) == 0
        )
        if not strip_ok:
            return abort_all(
                "arm-B written copy still carries a restore_scratch_294 trap; mutation unverified"
            )

        env_a = arm_env(work, work / "tmpA")
        env_b = arm_env(work, work / "tmpB")

        # L3's reference: the bytes as they stand BEFORE each arm's run.
        before_a = snapshot_bytes(arm_a, artifacts)

        rc_a, out_a, to_a = run_arm(arm_a / "build_h100_plane.sh", arm_a, env_a)
        rc_b, out_b, to_b = run_arm(arm_b / "build_h100_plane.sh", arm_b, env_b)

        reached_a = (not to_a) and (REACHED_BANNER in out_a)
        reached_b = (not to_b) and (REACHED_BANNER in out_b)

        # ---- L1 REACHED-SITE ---------------------------------------------
        # MEASUREMENT ON RECORD: an earlier hand-run of this comparison
        # exported no synthetic knobs. Both arms refused at the env guard
        # BEFORE the rm, nothing was deleted, 8/8 "survived" on both arms,
        # and the control reported green over a measurement that never
        # happened. That is why reaching the site is asserted SEPARATELY
        # from the outcome, and why missing banner -> UNMEASURED, never a
        # pass.
        if reached_a and reached_b:
            leg(
                "L1",
                "PASS",
                f"both arms reached the rm site ('{REACHED_BANNER}' present; rc A={rc_a} B={rc_b})",
            )
        else:
            which = []
            if to_a:
                which.append(f"arm A timed out after {ARM_TIMEOUT_S}s")
            elif not reached_a:
                which.append(f"arm A refused upstream of the rm (rc={rc_a})")
            if to_b:
                which.append(f"arm B timed out after {ARM_TIMEOUT_S}s")
            elif not reached_b:
                which.append(f"arm B refused upstream of the rm (rc={rc_b})")
            leg(
                "L1",
                "UNMEASURED",
                "NOT AT THE SITE: "
                + "; ".join(which)
                + " -- an arm that never reached the rm measured nothing, and "
                "any survival count from it would be a green that proves "
                "nothing",
            )

        def gate(ok: bool, name: str, arm: str):
            if not ok:
                leg(
                    name,
                    "UNMEASURED",
                    f"{arm} never reached the rm; the measurement would be vacuous",
                )
            return ok

        # ---- L2 MUST_PASS: arm A leaves 8/8 -------------------------------
        # MEASUREMENT ON RECORD: `roll_call_gates` ends `return "$_rc"`, and
        # `set -e`/errexit is in force inside a trap handler. Without the
        # load-bearing `|| :` at :713, the handler aborts at that return and
        # the restore never runs -- the first cut of the fix lost 8/8 on a
        # refusing build. This leg is where that regression shows up.
        surv_a = count_survivors(arm_a, artifacts)
        if gate(reached_a, "L2", "arm A"):
            if surv_a == N_ARTIFACTS:
                leg(
                    "L2",
                    "PASS",
                    f"arm A leaves {N_ARTIFACTS}/{N_ARTIFACTS} "
                    f"artifacts present after the refused run",
                )
            else:
                leg(
                    "L2",
                    "FAIL",
                    f"arm A leaves only {surv_a}/{N_ARTIFACTS} "
                    f"artifacts; the restore did not put them "
                    f"all back",
                )

        # ---- L3 MUST_PASS: restored bytes are the original bytes ----------
        if gate(reached_a, "L3", "arm A"):
            mism = byte_mismatches(before_a, arm_a, artifacts)
            if not mism:
                leg(
                    "L3",
                    "PASS",
                    f"arm A: all {N_ARTIFACTS} artifacts are "
                    f"BYTE-IDENTICAL to the copy taken before "
                    f"the run",
                )
            else:
                leg(
                    "L3",
                    "FAIL",
                    f"arm A: {len(mism)}/{N_ARTIFACTS} differ "
                    f"from (or are missing versus) the pre-run "
                    f"copy: {' '.join(mism)}",
                )

        # ---- L4 MUST_PASS: no snapshot dir leaks ---------------------------
        if gate(reached_a, "L4", "arm A"):
            leaks = snapshot_leaks(work / "tmpA")
            if not leaks:
                leg(
                    "L4",
                    "PASS",
                    "zero h100-build-294.* entries remain under the TMPDIR arm A was given",
                )
            else:
                leg("L4", "FAIL", f"snapshot dirs leak under arm A's TMPDIR: {' '.join(leaks)}")

        # ---- L5 MUST_FIRE: arm B leaves 0/8 --------------------------------
        # The negative control is what makes A's green attributable. If B
        # also leaves 8/8, the instrument cannot tell the fix from its
        # absence and the whole control is VACUOUS -- that is RED, not PASS:
        # a silent instrument is a measurement failure, not a success.
        surv_b = count_survivors(arm_b, artifacts)
        if gate(reached_b, "L5", "arm B"):
            if surv_b == 0:
                leg(
                    "L5",
                    "PASS",
                    f"arm B leaves 0/{N_ARTIFACTS} with the "
                    f"restore stripped -- the pre-fix loss "
                    f"reproduces, so A's survival is caused by "
                    f"the fix",
                )
            elif surv_b == N_ARTIFACTS:
                leg(
                    "L5",
                    "FAIL",
                    f"arm B leaves {N_ARTIFACTS}/{N_ARTIFACTS} "
                    f"although the fix is stripped; the "
                    f"instrument cannot tell the fix from its "
                    f"absence -- VACUOUS by construction, RED "
                    f"by contract",
                )
            else:
                leg(
                    "L5",
                    "FAIL",
                    f"arm B leaves {surv_b}/{N_ARTIFACTS}; "
                    f"inconsistent with a clean pre-fix "
                    f"refusal path",
                )

        # ---- L6: roll-call banner exactly once on arm A --------------------
        # bash EXIT traps REPLACE, they never chain. Exactly once proves the
        # composed trap neither dropped the pre-existing roll_call_gates
        # handler nor doubled it.
        if gate(reached_a, "L6", "arm A"):
            n_roll = out_a.count(ROLLCALL_BANNER)
            if n_roll == 1:
                leg(
                    "L6",
                    "PASS",
                    f"arm A's '{ROLLCALL_BANNER}' banner appears "
                    f"exactly once -- trap composition intact",
                )
            else:
                leg(
                    "L6",
                    "FAIL",
                    f"arm A's '{ROLLCALL_BANNER}' banner appears "
                    f"{n_roll} times (want exactly 1) -- the "
                    f"composed EXIT trap dropped or doubled the "
                    f"pre-existing handler",
                )

        # ---- L7: arm A's verdict equals arm B's ----------------------------
        # The restore changes what the tree LOOKS like at exit, not the
        # verdict the build reported. Divergent rcs would mean the fix
        # altered the build's own decision, which it must not do.
        if rc_a is None or rc_b is None:
            leg(
                "L7",
                "UNMEASURED",
                "an arm never returned an exit status "
                "(timeout or exec failure); parity cannot "
                "be asserted",
            )
        elif rc_a == rc_b:
            leg(
                "L7",
                "PASS",
                f"exit parity: arm A rc={rc_a} == arm B rc={rc_b} (the calibrated refuse run)",
            )
        else:
            leg(
                "L7",
                "FAIL",
                f"exit divergence: arm A rc={rc_a}, arm B "
                f"rc={rc_b}; the restore changed the build's "
                f"reported verdict, not just the tree",
            )

        code = _aggregate(legs)
        emit(legs, code)
        return code
    finally:
        # Requirement: temp dirs are cleaned on EVERY exit path, green, red,
        # or unmeasured. A failing run is debuggable from the printed legs
        # and the sibling script, not from retained trees.
        manager.cleanup()


def main(argv: list[str]) -> int:
    if argv[:1] == ["--self-test"]:
        return self_test()
    if argv:
        print(
            f"VERDICT 95: UNMEASURED -- unknown arguments {' '.join(argv)!r}; "
            f"this control takes none (or --self-test)"
        )
        return EXIT_UNMEASURED
    return run_control()


if __name__ == "__main__":
    try:
        rc = main(sys.argv[1:])
    except KeyboardInterrupt:
        print(
            "VERDICT 95: UNMEASURED -- interrupted before the measurement completed",
            file=sys.stderr,
        )
        rc = EXIT_UNMEASURED
    except Exception as exc:  # NEVER 1: a crashing control measured nothing.
        print(
            f"VERDICT 95: UNMEASURED -- control raised {type(exc).__name__}: {exc}", file=sys.stderr
        )
        rc = EXIT_UNMEASURED
    if rc not in _VALID:
        rc = EXIT_UNMEASURED
    sys.exit(rc)
