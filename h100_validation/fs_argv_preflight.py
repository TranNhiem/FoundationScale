#!/usr/bin/env python3
"""Preflight an engine launch command's argv on the login node, before sbatch.

WHY THIS EXISTS
---------------
Finding #183: the launcher's login-node path submits four dependent Slurm jobs
(`sbatch` at lines 656, 658, 660, 662) and exits. Every guard that touches the
operator's engine launch command sits BELOW line 724, i.e. inside the allocation:
the first read of FS_ENGINE_LAUNCH_CMD is :817, the unset guard is :819, the
compose call is :841. Below line 656 the name appears only in a comment (:480)
and a printf (:489). So an operator whose launch command is unset, or which
passes a flag the engine entrypoint does not declare, learns this only after the
scheduler grants nodes. This tool is the host-side, pre-submit check. It runs on
a login node whose Python is 3.6.8 and which has no torch, and it never needs an
allocation.

DENOMINATORS
------------
Absence is reported, never assumed. An entrypoint that cannot be read from the
login node is UNMEASURED, never RED: it may exist only inside the container, and
absence of visibility is not evidence of absence. An entrypoint that parses but
declares no --flag gives this tool no denominator. An argv with zero -- tokens
checks nothing -- `all([])` is True, so that is unmeasured, not clean. The
accepted mode set is supplied by the caller (--modes) or derived from the
backend's own `case "$mode" in ...` arm (--backend); this tool never hard-codes
one, because a second oracle drifting from the backend's is the defect class
this campaign keeps finding.

SELF-TEST
---------
--self-test runs 12 mandatory controls against fixtures, the real trainer, and
the real backend. If the real trainer or the real backend is absent, the
controls that need it report unmeasured and the suite exits 95, not 0: a
self-test that silently skipped its only real-artifact controls is the vacuous
case.

Exit codes follow the four-state contract: 0 PASS, 5 RED, 95 UNMEASURED,
96 REFUSE (bad input to this tool itself).
"""

import argparse
import ast
import difflib
import json  # stdlib; required for --json
import os
import re
import shlex
import tempfile
from pathlib import Path
# No annotations, no dataclasses, no walrus, no PEP 585 generics: the host
# interpreter on a login node can predate the training stack by years (measured:
# 3.6.8), and this tool's whole value is that it runs THERE. `from __future__
# import annotations` is a hard SyntaxError on 3.6, so it is deliberately absent.

EXIT_PASS = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

_VERDICT_EXIT = {
    "PASS": EXIT_PASS,
    "RED": EXIT_RED,
    "UNMEASURED": EXIT_UNMEASURED,
    "REFUSE": EXIT_REFUSE,
}

_CHECK_IDS = ("C1", "C2", "C3", "C4", "C5", "C6")

# Tokens that mean "what follows is not yet the entrypoint". Matched on the
# basename so an absolute path to any of them is treated the same as the bare
# name.
_LAUNCHER_BASENAMES = frozenset({"python", "python2", "python3", "torchrun", "srun"})

# The backend declares the accepted modes in one line inside fs_compose_launch:
#   case "$mode" in torchrun|wlm|self) ;; *) fs_die ... ;; esac
# A shell case arm is a stable enough shape to read with a regex; the
# alternative (running bash) is not available on a login node we must not
# assume anything about.
_MODE_CASE_RE = re.compile(
    r'case\s+["\']?\$\{?mode\}?["\']?\s+in\s+([A-Za-z0-9_]+(?:\|[A-Za-z0-9_]+)*)\)')


def _ast_const(node):
    """The literal value of a constant node, tolerating ast.Str for older readers."""
    if isinstance(node, ast.Constant):
        return node.value
    str_t = getattr(ast, "Str", None)
    if str_t is not None and isinstance(node, str_t):
        return node.s
    return None


def declared_flags(src):
    """The --flag strings an entrypoint declares, extracted by AST.

    This reproduces gate_launch_doc.py:493 EXACTLY: walk the tree for Call nodes
    whose func is an Attribute named add_argument; for each positional arg take
    _ast_const; keep the strings starting with '--'. Two oracles that can
    disagree is the defect class this campaign keeps finding, so the self-test
    carries an agreement control against gate_launch_doc.trainer_flags.
    """
    tree = ast.parse(src)
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            value = _ast_const(arg)
            if isinstance(value, str) and value.startswith("--"):
                flags.add(value)
    return flags


def declared_modes(src):
    """The accepted mode set, derived from the backend's own case arm.

    Returns (modes_set, provenance). Derivation is attempted only when the
    source contains `fs_compose_launch` -- otherwise this is not the file we
    think it is. The first line matching a `case` on the mode variable whose
    first arm is a `|`-separated list of bare words followed by `)` wins, and
    the provenance names the line number it was read from. Any failure returns
    an empty set and a provenance explaining what was looked for. There is
    deliberately NO built-in fallback list: a fallback list is the hard-coded
    second oracle this function exists to remove.
    """
    if "fs_compose_launch" not in src:
        return set(), ("no `fs_compose_launch` in the source, so this is not the "
                       "backend file this tool derives the mode set from")
    for lineno, line in enumerate(src.splitlines(), 1):
        m = _MODE_CASE_RE.search(line)
        if m:
            return (set(m.group(1).split("|")),
                    "derived from the backend's case arm at line %d" % lineno)
    return set(), ("looked for a `case \"$mode\" in word|word|...)` arm inside "
                   "fs_compose_launch and found none")


def _is_launcher_token(token):
    """True for python, python3, pythonN.N, torchrun, srun, or a path to one."""
    base = os.path.basename(token)
    if base in _LAUNCHER_BASENAMES:
        return True
    if base.startswith("python"):
        rest = base[len("python"):]
        # pythonN.N: only digits and dots after the name (python3.10, python2.7).
        if rest and all(ch.isdigit() or ch == "." for ch in rest):
            return True
    return False


def resolve_entrypoint(tokens):
    """Return (entrypoint, index, skipped); entrypoint is None when nothing survives.

    skipped is a list of (token, rule) pairs recording why each leading token was
    set aside, so the report can state which token won and by which rule.
    """
    skipped = []
    i = 0
    if i < len(tokens) and _is_launcher_token(tokens[i]):
        skipped.append((tokens[i], "leading launcher/interpreter"))
        i += 1
    while i < len(tokens) and tokens[i].startswith("-"):
        flag = tokens[i]
        skipped.append((flag, "launcher flag preceding the entrypoint"))
        i += 1
        # HEURISTIC, stated because it is one: a `--x value` pair (as opposed to
        # `--x=value`) puts the flag's value in the NEXT token, so that token is
        # skipped too -- but ONLY when it does not end in .py. A token ending in
        # .py is far more likely to be the entrypoint following a valueless flag
        # (e.g. `--standalone`) than a flag value, and mis-skipping it would hide
        # the entrypoint entirely. A flag whose value genuinely ends in .py is
        # still misjudged; that is the price of resolving a command line without
        # the launcher's own parser.
        if flag.startswith("--") and "=" not in flag:
            if i < len(tokens) and not tokens[i].endswith(".py"):
                skipped.append((tokens[i], "value of the preceding --flag (not --flag=value form)"))
                i += 1
    if i < len(tokens):
        return tokens[i], i, skipped
    return None, None, skipped


def _check(cid, state, reason):
    return {"id": cid, "state": state, "reason": reason}


def _finalize(checks, denominator):
    """Fill unrun checks as SKIP, aggregate the verdict, attach the exit code."""
    seen = set(c["id"] for c in checks)
    for cid in _CHECK_IDS:
        if cid not in seen:
            checks.append(_check(cid, "SKIP", "not reached; an earlier check refused"))
    states = [c["state"] for c in checks]
    # ABSENT and SKIP are declared non-states: they neither pass nor measure.
    if "REFUSE" in states:
        verdict = "REFUSE"
    elif "RED" in states:
        verdict = "RED"
    elif "UNMEASURED" in states:
        verdict = "UNMEASURED"
    else:
        verdict = "PASS"
    return {
        "checks": checks,
        "denominator": denominator,
        "verdict": verdict,
        "exit_code": _VERDICT_EXIT[verdict],
    }


def run_checks(launch_cmd, mode, modes, procs_per_node, backend=None):
    """Run C1..C6 against the launch command and return the full report dict."""
    checks = []
    denominator = {
        "argv_flags_checked": 0,
        "declared_flags": 0,
        "entrypoint": None,
        "out_of_scope_tokens": [],
        "unknown_flags": [],
        "accepted_modes": 0,
        "mode_set_provenance": None,
    }

    # C1 -- command present. This is the launcher's :819 guard moved above the
    # first sbatch; the wording names FS_ENGINE_LAUNCH_CMD for parity with it.
    if launch_cmd is None or not launch_cmd.strip():
        checks.append(_check(
            "C1", "REFUSE",
            "FS_ENGINE_LAUNCH_CMD unset or empty; the launcher's own guard for this "
            "sits inside the allocation (:819), which is exactly the failure this "
            "preflight exists to catch before submit"))
        return _finalize(checks, denominator)
    checks.append(_check("C1", "PASS", "launch command present (FS_ENGINE_LAUNCH_CMD)"))

    # C2 -- tokenizes. A command shlex cannot split would fail the same way under
    # the shell, later, at a worse time.
    try:
        tokens = shlex.split(launch_cmd)
    except ValueError as exc:
        checks.append(_check("C2", "REFUSE", "launch command does not tokenize: %s" % exc))
        return _finalize(checks, denominator)
    checks.append(_check("C2", "PASS", "launch command tokenizes into %d token(s)" % len(tokens)))

    # C3 -- mode. The accepted set is the backend's fact, so it is never
    # hard-coded here: an explicit --modes wins, else the set is derived from
    # the backend's own case arm via --backend, else this check has no oracle
    # and is UNMEASURED, never PASS. The source is named in the report every
    # time.
    mode_set = None
    mode_provenance = None
    if modes is not None:
        mode_set = set(modes)
        mode_provenance = "supplied by the caller (--modes)"
    elif backend is not None:
        bpath = Path(str(backend))
        if bpath.is_file() and os.access(str(bpath), os.R_OK):
            try:
                bsrc = bpath.read_text()
            except (OSError, UnicodeError) as exc:
                mode_provenance = "backend %s could not be read (%s)" % (backend, exc)
            else:
                mode_set, mode_provenance = declared_modes(bsrc)
                if not mode_set:
                    mode_set = None
        else:
            mode_provenance = "backend %s is not readable from this login node" % (backend,)
    if mode_set is not None:
        denominator["accepted_modes"] = len(mode_set)
    denominator["mode_set_provenance"] = mode_provenance

    if mode_set is None:
        reason = ("no oracle for the accepted mode set: supply it with --modes, or "
                  "let this tool derive it from the backend with --backend")
        if mode_provenance:
            reason += "; " + mode_provenance
        checks.append(_check("C3", "UNMEASURED", reason))
    elif mode is None or not str(mode).strip():
        checks.append(_check(
            "C3", "UNMEASURED",
            "FS_ENGINE_LAUNCH_MODE is unset, so there is no mode to check against "
            "the accepted mode set %s (%s)" % (sorted(mode_set), mode_provenance)))
    elif mode in mode_set:
        checks.append(_check("C3", "PASS", "mode %r is in the accepted mode set %s (%s)"
                                           % (mode, sorted(mode_set), mode_provenance)))
    else:
        checks.append(_check("C3", "RED", "mode %r is not in the accepted mode set %s "
                                          "(%s) (FS_ENGINE_LAUNCH_MODE)"
                                          % (mode, sorted(mode_set), mode_provenance)))

    # C4 -- procs-per-node. Unset is ABSENT from the denominator and said so: it
    # is not a pass, and calling it UNMEASURED would manufacture alarm over a
    # value the plane treats as optional.
    if procs_per_node is None or not str(procs_per_node).strip():
        checks.append(_check(
            "C4", "ABSENT",
            "procs-per-node unset (FS_ENGINE_PROCS_PER_NODE); absent from the "
            "denominator, not a pass"))
    else:
        try:
            n_procs = int(str(procs_per_node).strip())
        except ValueError:
            n_procs = None
        if n_procs is not None and n_procs > 0:
            checks.append(_check("C4", "PASS", "procs-per-node is a positive integer (%d)" % n_procs))
        else:
            checks.append(_check("C4", "RED", "procs-per-node %r is not a positive integer "
                                              "(FS_ENGINE_PROCS_PER_NODE)" % (procs_per_node,)))

    # C5 -- entrypoint resolution.
    entrypoint, entry_index, skipped = resolve_entrypoint(tokens)
    if entrypoint is None:
        checks.append(_check(
            "C5", "UNMEASURED",
            "no entrypoint token survives after skipping the launcher and its flags"))
        checks.append(_check(
            "C6", "UNMEASURED",
            "entrypoint not resolvable; the entrypoint may exist only inside the "
            "container; absence of visibility is not evidence of absence"))
        return _finalize(checks, denominator)
    if skipped:
        how = "first surviving token after skipping " + ", ".join(
            "%r (%s)" % (tok, rule) for tok, rule in skipped)
    else:
        how = "first token; no launcher or launcher flags preceded it"
    checks.append(_check("C5", "PASS", "entrypoint is %r (%s)" % (entrypoint, how)))

    # C6 -- flag check. Only tokens AFTER the entrypoint are in the denominator;
    # tokens before it belong to torchrun/srun and are explicitly out of scope,
    # so a green here must never be read as covering them.
    argv_after = tokens[entry_index + 1:]
    out_of_scope = tokens[:entry_index]
    denominator["out_of_scope_tokens"] = out_of_scope
    denominator["entrypoint"] = entrypoint
    if out_of_scope:
        scope_note = ("; %d pre-entrypoint token(s) belong to the launcher "
                      "(torchrun/srun) and are out of scope" % len(out_of_scope))
    else:
        scope_note = ""

    # Not resolvable, not readable, or not a .py file: UNMEASURED, never RED.
    container_reason = ("the entrypoint may exist only inside the container; "
                        "absence of visibility is not evidence of absence")
    if not entrypoint.endswith(".py"):
        checks.append(_check("C6", "UNMEASURED", "entrypoint %r is not a .py file; %s%s"
                                                 % (entrypoint, container_reason, scope_note)))
        return _finalize(checks, denominator)
    path = Path(entrypoint)
    if not path.is_file() or not os.access(str(path), os.R_OK):
        checks.append(_check("C6", "UNMEASURED", "entrypoint %r is not readable from this "
                                                 "login node; %s%s"
                                                 % (entrypoint, container_reason, scope_note)))
        return _finalize(checks, denominator)
    try:
        src = path.read_text()
    except (OSError, UnicodeError) as exc:
        checks.append(_check("C6", "UNMEASURED", "entrypoint %r could not be read (%s); %s%s"
                                                 % (entrypoint, exc, container_reason, scope_note)))
        return _finalize(checks, denominator)

    try:
        declared = declared_flags(src)
    except SyntaxError as exc:
        # A file that does not parse tells this tool nothing about its flags;
        # that is a blind spot, not a defect in the operator's argv.
        checks.append(_check("C6", "UNMEASURED", "entrypoint %r does not parse: SyntaxError: %s%s"
                                                 % (entrypoint, exc, scope_note)))
        return _finalize(checks, denominator)

    if not declared:
        checks.append(_check(
            "C6", "UNMEASURED",
            "no `add_argument` call declares a `--flag`, so this tool has no "
            "denominator for this entrypoint" + scope_note))
        return _finalize(checks, denominator)
    denominator["declared_flags"] = len(declared)

    # Split each argv token on the FIRST '=' so `--flag=value` checks `--flag`.
    argv_flags = []
    for tok in argv_after:
        head = tok.split("=", 1)[0]
        if head.startswith("--"):
            argv_flags.append(head)
    denominator["argv_flags_checked"] = len(argv_flags)

    if not argv_flags:
        # Mandatory rule: zero flags checked is unmeasured, not clean --
        # all([]) is True, and a vacuous pass here is exactly the misread this
        # campaign keeps finding.
        checks.append(_check(
            "C6", "UNMEASURED",
            "zero flags checked; `all([])` is True, so this is unmeasured, not clean"
            + scope_note))
        return _finalize(checks, denominator)

    unknown = [f for f in argv_flags if f not in declared]
    if unknown:
        parts = []
        for flag in unknown:
            close = difflib.get_close_matches(flag, sorted(declared), n=1)
            if close:
                parts.append("%s (did you mean %s?)" % (flag, close[0]))
            else:
                parts.append(flag)
        denominator["unknown_flags"] = unknown
        checks.append(_check("C6", "RED", "undeclared flag(s): %s; the entrypoint declares "
                                          "%d flag(s)%s"
                                          % ("; ".join(parts), len(declared), scope_note)))
    else:
        checks.append(_check("C6", "PASS", "%d argv flag(s) checked against %d declared "
                                           "flag(s); every one is declared%s"
                                           % (len(argv_flags), len(declared), scope_note)))
    return _finalize(checks, denominator)


def render_text(report):
    """The plain-text report: one line per check, a denominator, a verdict."""
    lines = []
    for c in report["checks"]:
        lines.append("%s %s  %s" % (c["id"], c["state"], c["reason"]))
    d = report["denominator"]
    den = "DENOMINATOR: %d argv flag(s) checked against %d declared flag(s); entrypoint: %s" % (
        d["argv_flags_checked"], d["declared_flags"], d["entrypoint"] or "(none)")
    if d["out_of_scope_tokens"]:
        den += ("; %d pre-entrypoint token(s) belong to the launcher and were not "
                "checked: %s" % (len(d["out_of_scope_tokens"]), d["out_of_scope_tokens"]))
    den += "; accepted mode set: %d mode(s) (%s)" % (
        d["accepted_modes"], d["mode_set_provenance"] or "no oracle")
    lines.append(den)
    lines.append("ARGV PREFLIGHT %s" % report["verdict"])
    return "\n".join(lines)


_FIXTURE_SRC = """\
import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0)
    return parser
"""


def self_test(trainer_arg, backend_arg=None):
    """Run the mandatory controls against fixtures, the real trainer, and the
    real backend."""
    results = []

    def record(name, ok, observation):
        results.append((name, "ok" if ok else "FAILED", observation))

    def record_unmeasured(name, observation):
        results.append((name, "unmeasured", observation))

    tmpdir = tempfile.mkdtemp(prefix="fs_argv_preflight_")
    fixture = os.path.join(tmpdir, "fixture_entrypoint.py")
    with open(fixture, "w") as fh:
        fh.write(_FIXTURE_SRC)

    # A mode oracle is supplied so C3 can PASS; without one every verdict below
    # would be capped at UNMEASURED and the MUST_PASS controls could not fire.
    mode = "selftest"
    modes = ["selftest"]

    rep = run_checks("%s --beta" % fixture, mode, modes, None)
    text = render_text(rep)
    record("MUST_FIRE/UNKNOWN_FLAG_IS_RED",
           rep["verdict"] == "RED" and "--beta" in text,
           "verdict=%s; '--beta' named in report: %s" % (rep["verdict"], "--beta" in text))

    rep = run_checks("%s --alpha=1" % fixture, mode, modes, None)
    record("MUST_PASS/DECLARED_FLAG_IS_GREEN",
           rep["verdict"] == "PASS",
           "verdict=%s" % rep["verdict"])

    rep = run_checks("/nonexistent/x.py --beta", mode, modes, None)
    record("MUST_FIRE/UNREADABLE_ENTRYPOINT_IS_UNMEASURED_NOT_RED",
           rep["exit_code"] == EXIT_UNMEASURED and rep["exit_code"] != EXIT_RED,
           "exit=%d (required 95, forbidden 5)" % rep["exit_code"])

    rep = run_checks("%s 1" % fixture, mode, modes, None)
    record("MUST_FIRE/ZERO_FLAGS_CHECKED_IS_UNMEASURED",
           rep["exit_code"] == EXIT_UNMEASURED and rep["exit_code"] != EXIT_PASS,
           "exit=%d (required 95, forbidden 0)" % rep["exit_code"])

    rep = run_checks("", mode, modes, None)
    record("MUST_FIRE/EMPTY_COMMAND_IS_REFUSE",
           rep["exit_code"] == EXIT_REFUSE,
           "exit=%d (required 96)" % rep["exit_code"])

    rep = run_checks("torchrun --nproc_per_node=8 %s --alpha 1" % fixture, mode, modes, None)
    text = render_text(rep)
    out = rep["denominator"]["out_of_scope_tokens"]
    named = any(t.startswith("--nproc_per_node") for t in out) and "--nproc_per_node" in text
    record("MUST_PASS/LAUNCHER_FLAGS_ARE_OUT_OF_THE_DENOMINATOR",
           rep["verdict"] == "PASS" and named,
           "verdict=%s; out-of-scope tokens named as not checked: %s" % (rep["verdict"], out))

    # Neither --modes nor --backend, with an otherwise fully-clean command:
    # C3 has no oracle, so the aggregate is UNMEASURED and specifically NOT
    # PASS. A vacuous green here is the misread this campaign keeps finding.
    rep = run_checks("%s --alpha=1" % fixture, mode, None, None)
    record("MUST_FIRE/NO_MODE_ORACLE_IS_UNMEASURED_NOT_PASS",
           rep["verdict"] == "UNMEASURED" and rep["exit_code"] == EXIT_UNMEASURED
           and rep["verdict"] != "PASS",
           "verdict=%s exit=%d (required UNMEASURED/95, forbidden PASS/0)"
           % (rep["verdict"], rep["exit_code"]))

    if trainer_arg:
        trainer = Path(trainer_arg)
    else:
        trainer = Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
    trainer_unmeasured = False
    if not trainer.is_file():
        # A self-test that silently skipped its only real-artifact control is the
        # vacuous case, so this is UNMEASURED and the self-test exits 95, not 0.
        trainer_unmeasured = True
        record_unmeasured("MUST_FIRE/REAL_TRAINER_REJECTS_AN_INVENTED_FLAG",
                          "real trainer not found at %s" % trainer)
    else:
        rep_ok = run_checks("%s --iteration-budget 4" % trainer, mode, modes, None)
        rep_bad = run_checks("%s --not-a-real-flag 1" % trainer, mode, modes, None)
        record("MUST_FIRE/REAL_TRAINER_REJECTS_AN_INVENTED_FLAG",
               rep_ok["verdict"] == "PASS" and rep_bad["verdict"] == "RED",
               "--iteration-budget 4: %s (required PASS); --not-a-real-flag 1: %s "
               "(required RED)" % (rep_ok["verdict"], rep_bad["verdict"]))

    if backend_arg:
        backend = Path(backend_arg)
    else:
        backend = (Path(__file__).resolve().parent / "h100" / "gen"
                   / "fs_container_backend.bound.sh")
    backend_unmeasured = False
    backend_controls = ("MUST_FIRE/AN_UNDECLARED_MODE_IS_RED",
                        "MUST_PASS/A_DECLARED_MODE_IS_NOT_RED",
                        "MUST_FIRE/THE_MODE_SET_IS_DERIVED_NOT_HARDCODED")
    if not backend.is_file():
        # Same rule as the real-trainer control: absent real artifact means
        # UNMEASURED and the self-test exits 95, not 0.
        backend_unmeasured = True
        for name in backend_controls:
            record_unmeasured(name, "real backend not found at %s" % backend)
    else:
        bsrc = backend.read_text()
        derived, derived_prov = declared_modes(bsrc)
        if not derived:
            backend_unmeasured = True
            for name in backend_controls:
                record_unmeasured(name, "could not derive a mode set from %s (%s)"
                                        % (backend, derived_prov))
        else:
            rep = run_checks("%s --alpha=1" % fixture, "nonsense", None, None, str(backend))
            text = render_text(rep)
            listed = all(m in text for m in derived)
            record("MUST_FIRE/AN_UNDECLARED_MODE_IS_RED",
                   rep["verdict"] == "RED" and listed,
                   "verdict=%s (required RED); accepted modes %s listed in report: %s"
                   % (rep["verdict"], sorted(derived), listed))

            pick = sorted(derived)[0]
            rep = run_checks("%s --alpha=1" % fixture, pick, None, None, str(backend))
            c3 = [c for c in rep["checks"] if c["id"] == "C3"][0]
            record("MUST_PASS/A_DECLARED_MODE_IS_NOT_RED",
                   c3["state"] == "PASS",
                   "mode=%r (sorted(derived)[0]); C3=%s (required PASS)"
                   % (pick, c3["state"]))

            # Patch a COPY of the backend: add a fourth alternative `banana` to
            # the case arm and assert `banana` is then accepted. If this tool
            # ever grows a built-in mode list, this control goes red -- it is
            # what makes "derived" a measured claim rather than a comment.
            lines = bsrc.splitlines(True)
            patched = None
            for i, line in enumerate(lines):
                m = _MODE_CASE_RE.search(line)
                if m:
                    lines[i] = line[:m.end(1)] + "|banana" + line[m.end(1):]
                    patched = "".join(lines)
                    break
            if patched is None:
                record("MUST_FIRE/THE_MODE_SET_IS_DERIVED_NOT_HARDCODED",
                       False,
                       "could not locate the case arm in a copy of %s to patch"
                       % backend)
            else:
                copy = os.path.join(tmpdir, "fs_container_backend.with_banana.sh")
                with open(copy, "w") as fh:
                    fh.write(patched)
                rep = run_checks("%s --alpha=1" % fixture, "banana", None, None, copy)
                c3 = [c for c in rep["checks"] if c["id"] == "C3"][0]
                record("MUST_FIRE/THE_MODE_SET_IS_DERIVED_NOT_HARDCODED",
                       c3["state"] == "PASS",
                       "mode 'banana' against the patched copy: C3=%s (required "
                       "PASS; a hard-coded list would be RED)" % c3["state"])

    gate_error = None
    gate = None
    try:
        import gate_launch_doc as gate
    except Exception as exc:  # ImportError, or SyntaxError on a 3.6 login node
        gate_error = exc
    if gate is None:
        # Not a failure -- on the 3.6 login node the gate may not import at all --
        # but the line says `unmeasured`, not `ok`, so a build stage can require `ok`.
        record_unmeasured("MUST_PASS/ORACLE_AGREES_WITH_THE_GATE",
                          "gate_launch_doc is not importable here (%s: %s)"
                          % (type(gate_error).__name__, gate_error))
    elif not trainer.is_file():
        record_unmeasured("MUST_PASS/ORACLE_AGREES_WITH_THE_GATE",
                          "real trainer not found at %s; nothing to compare oracles on" % trainer)
    else:
        src = trainer.read_text()
        ours = declared_flags(src)
        theirs = gate.trainer_flags(src)
        record("MUST_PASS/ORACLE_AGREES_WITH_THE_GATE",
               ours == theirs,
               "declared_flags: %d flag(s); gate_launch_doc.trainer_flags: %d flag(s); "
               "agree: %s" % (len(ours), len(theirs), ours == theirs))

    for name, status, observation in results:
        print("control %s: %s -- %s" % (name, status, observation))
    n_ok = sum(1 for _, status, _ in results if status == "ok")
    n_unmeasured = sum(1 for _, status, _ in results if status == "unmeasured")
    print("SELF-TEST %d/%d ok (%d unmeasured)" % (n_ok, len(results), n_unmeasured))

    if any(status == "FAILED" for _, status, _ in results):
        return EXIT_RED
    if trainer_unmeasured or backend_unmeasured:
        return EXIT_UNMEASURED
    return EXIT_PASS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launch-cmd", default=None,
                        help="engine launch command; falls back to FS_ENGINE_LAUNCH_CMD")
    parser.add_argument("--mode", default=None,
                        help="launch mode; falls back to FS_ENGINE_LAUNCH_MODE")
    parser.add_argument("--procs-per-node", default=None,
                        help="processes per node; falls back to FS_ENGINE_PROCS_PER_NODE")
    parser.add_argument("--modes", nargs="+", default=None,
                        help="the accepted mode set; without it (and without "
                             "--backend) C3 is UNMEASURED, never PASS")
    parser.add_argument("--backend", default=None,
                        help="backend file to derive the accepted mode set from; "
                             "falls back to FS_BACKEND_FILE")
    parser.add_argument("--self-test", action="store_true",
                        help="run the control suite against fixtures, the real "
                             "trainer, and the real backend")
    parser.add_argument("--json", action="store_true",
                        help="emit the report as one JSON object")
    parser.add_argument("--trainer", default=None,
                        help="real trainer path for --self-test; defaults to "
                             "h100/gen/fs_train.fixed.py relative to this file")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test(args.trainer, args.backend)

    launch_cmd = args.launch_cmd
    if launch_cmd is None:
        launch_cmd = os.environ.get("FS_ENGINE_LAUNCH_CMD")
    mode = args.mode
    if mode is None:
        mode = os.environ.get("FS_ENGINE_LAUNCH_MODE")
    procs_per_node = args.procs_per_node
    if procs_per_node is None:
        procs_per_node = os.environ.get("FS_ENGINE_PROCS_PER_NODE")
    backend = args.backend
    if backend is None:
        backend = os.environ.get("FS_BACKEND_FILE")

    report = run_checks(launch_cmd, mode, args.modes, procs_per_node, backend)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
