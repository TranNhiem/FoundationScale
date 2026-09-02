#!/usr/bin/env python3
"""Gate: packaging reachability (closes FoundationScale finding #224).

WHAT IT MEASURES
    Whether names the project DECLARES or ADVERTISES are actually REACHABLE.
    Three denominators, printed on every path:

    (A) console scripts -- every entry point the installed distribution
        declares in the 'console_scripts' group: did the packaging system
        actually write the wrapper, and does its target import and resolve via
        the EntryPoint's own load()?  Existence is resolved against the
        interpreter's own script directory and the distribution's installed
        -file record, NOT against the caller's PATH -- see
        check_console_scripts for why that distinction is the whole point.
        A load() that raises ImportError naming a module OUTSIDE the entry
        point's own package root is UNMEASURED for that script (thin
        environment, not a defect); AttributeError or an ImportError naming
        the entry point's own package root is RED.
    (B) extras -- every extra name the project's own source tells an
        operator to install (pattern ``<dist>[extra]`` inside a quoted
        string) must be a declared 'Provides-Extra'.  Advertised but
        undeclared is RED.  Declared but never advertised is informational
        only (unused is not broken).
    (C) prog= names -- every argparse prog="..." value in the source that
        names something other than a module path must be a declared console
        script.  That mismatch is exactly the #224 shape and is RED.

EXIT CODES
    0   PASS / CLEAR  -- distribution installed, at least one denominator
                         nonzero, zero findings, nothing left unmeasured.
    5   RED           -- a real defect (unreachable console script, phantom
                         extra, undeclared prog= name, unreadable source),
                         a control misbehaved, or an unexpected exception
                         (fail closed: a crash is never clean).
    95  UNMEASURED    -- distribution not installed (suggests
                         `pip install -e .`), or all three denominators are
                         zero (nothing to measure), or a source root is
                         absent, or a console script's target cannot be
                         resolved because the environment is thin.
    96  REFUSE        -- the invocation itself is invalid (bad arguments, a
                         source root that exists but is not a directory).
    Exit 1 is never used; main() is wrapped so any unexpected exception
    exits 5 with a stated reason.

CONTROLS (--self-test)
    Seven, built in a tempfile.TemporaryDirectory so nothing on disk is
    touched and no control reads the caller's environment.  One per axis,
    per outcome:
      MUST_FIRE: prog="totally-unregistered-cmd" not among declared console
                 scripts -> the prog= check must flag it (finding count > 0).
      MUST_FIRE: source advertising <dist>[nosuchextra] -> the extras check
                 must flag it (finding count > 0).
      MUST_PASS: source advertising only declared extras and a prog= equal
                 to a real declared console script -> zero findings.
      MUST_FIRE: a declared script whose wrapper exists in no script
                 directory and in no install record -> RED, resolved == 0.
      MUST_FIRE: a declared script whose wrapper EXISTS but whose target
                 raises AttributeError -> RED.  Existence and resolution are
                 separate questions and both must be able to fail alone.
      MUST_FIRE: a declared script whose target raises ImportError naming a
                 FOREIGN module -> UNMEASURED, and specifically NOT red.
      MUST_PASS: wrapper present, target loads -> zero findings AND
                 resolved == 1, so a detector examining nothing cannot pass.
    Exits 0 only if all controls behave; the MUST_FIRE cases are asserted
    to produce a NONZERO FINDING COUNT, not merely a nonzero exit, so the
    controls cannot be vacuous.  Any control failure exits 5, because then
    every verdict this detector returns is worthless.

    The last four exist because this gate's first version shipped the
    console-script axis with no control at all, and that was the axis that
    was wrong.  An uncontrolled axis is an unmeasured axis wearing a verdict.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import shutil
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

EXIT_CLEAR = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

DEFAULT_DIST = "foundationscale"


class GateArgumentParser(argparse.ArgumentParser):
    def exit(self, status: int = 0, message: str | None = None) -> None:
        # argparse exits 2 on bad arguments; without this remap an operator
        # typo would escape the four-code namespace this gate advertises.
        if message:
            sys.stderr.write(message)
        raise SystemExit(EXIT_CLEAR if status == 0 else EXIT_REFUSE)


def get_console_scripts(dist_name: str) -> list | None:
    """Return declared console-script EntryPoints, or None if not installed.

    importlib.metadata changed between 3.8 (entry_points returns a dict) and
    3.10 (returns EntryPoints with .select); both shapes are handled because
    the gate must run on whatever login-node Python exists, not on the one
    Python the author tested.
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    eps = dist.entry_points
    if hasattr(eps, "select"):
        return list(eps.select(group="console_scripts"))
    return list(eps.get("console_scripts", []))


def get_declared_extras(dist_name: str) -> set[str] | None:
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    metadata = dist.metadata
    raw = metadata.get_all("Provides-Extra") if metadata is not None else None
    return set(raw) if raw else set()


def script_dirs_for_interpreter() -> list[str]:
    """Directories where THIS interpreter's console-script wrappers live.

    `sysconfig.get_path("scripts")` is the directory pip writes wrappers into
    for the interpreter running this gate, whether or not that interpreter's
    environment has been "activated". `os.path.dirname(sys.executable)` is
    added because a venv created with --copies or relocated after creation can
    disagree with the sysconfig scheme.
    """
    dirs = [sysconfig.get_path("scripts"), str(Path(sys.executable).resolve().parent)]
    seen: list[str] = []
    for d in dirs:
        if d and d not in seen:
            seen.append(d)
    return seen


def installed_script_basenames(dist_name: str) -> set[str]:
    """Basenames of files the INSTALLED distribution recorded on disk.

    The RECORD is what the packaging system says it produced, so it answers
    "did pip write this wrapper" without consulting the caller's PATH. Entries
    are relative to site-packages and reach the wrapper via `../../../bin/x`,
    hence the basename comparison. Returns an empty set when the metadata does
    not list files (some install layouts omit RECORD) -- an empty set makes
    this source silent, never authoritative, which is why it is one of two
    sources and not the only one.
    """
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return set()
    files = getattr(dist, "files", None)
    if not files:
        return set()
    return {Path(str(f)).name for f in files}


class ScriptTally(NamedTuple):
    """Per-script outcome counts. resolved + unmeasured + failed == total.

    Kept as counts rather than derived from len(findings) because one script
    can contribute two findings (absent wrapper AND unresolvable target), and
    subtracting a finding count from a script count is how a denominator goes
    negative and stops meaning anything.
    """

    total: int
    resolved: int
    unmeasured: int
    failed: int
    on_path: int


def package_root_of(value: str) -> str:
    # An EntryPoint value looks like "pkg.module:function [extra]"; only its
    # first module segment identifies the distribution's own code.
    module = value.split(":", 1)[0].split("[", 1)[0]
    return module.split(".", 1)[0].strip()


def check_console_scripts(
    dist_name: str,
    scripts: list,
    script_dirs: Sequence[str] | None = None,
    record_basenames: set[str] | None = None,
) -> tuple[list[str], list[str], ScriptTally]:
    """Resolve each declared console script.

    Returns (red_findings, thin_env_unmeasured, tally).
    Wrapper existence and load() resolution are checked separately because pip
    happily installs a wrapper script whose module:function target does not
    exist -- reading pyproject.toml here would prove only that somebody
    typed it, which is the failure this gate exists to catch.

    WRAPPER EXISTENCE IS DELIBERATELY NOT A PATH LOOKUP.  The first draft of
    this gate asked `shutil.which(ep.name)` and reported RED for both of
    FoundationScale's scripts on a tree where pip had installed both wrappers
    correctly and both targets imported cleanly -- the only fact it had
    measured was that the operator had invoked `.venv/bin/python3` by path
    instead of sourcing `activate`.  A verdict that moves with the caller's
    shell environment while the code stands still states no coverage; it is
    the defect #83 fixed for the gates' own interpreter and the one
    [tool.ruff.lint] ignore = ["UP038"] fixes for the linter, recurring here.

    So existence is resolved against two shell-independent sources: the script
    directory belonging to the interpreter running this gate, and the
    distribution's own installed-file record.  PATH membership is still
    counted, but it is reported as operator convenience and can never be RED.
    """
    red: list[str] = []
    unmeasured: list[str] = []
    dirs = list(script_dirs) if script_dirs is not None else script_dirs_for_interpreter()
    record = (
        record_basenames if record_basenames is not None else installed_script_basenames(dist_name)
    )
    resolved = 0
    failed_wrapper = 0
    on_path = 0
    unmeasured_scripts = 0
    for ep in scripts:
        # .exe is checked alongside the bare name so a Windows install is
        # measured rather than reported as universally broken.
        candidates = (ep.name, ep.name + ".exe")
        where = ""
        for directory in dirs:
            for candidate in candidates:
                if (Path(directory) / candidate).exists():
                    where = str(Path(directory) / candidate)
                    break
            if where:
                break
        if not where and record.intersection(candidates):
            where = "recorded by the installed distribution"
        if where:
            resolved += 1
        else:
            failed_wrapper += 1
            red.append(
                "console script '{}' declared by '{}' has no wrapper in the "
                "interpreter's script directory ({}) and none recorded by the "
                "installed distribution".format(
                    ep.name, dist_name, os.pathsep.join(dirs) or "<none>"
                )
            )
        if shutil.which(ep.name):
            on_path += 1
        before = len(red), len(unmeasured)
        try:
            ep.load()
        except AttributeError as exc:
            red.append(f"console script '{ep.name}' target '{ep.value}' does not resolve: {exc}")
        except ImportError as exc:
            own_root = package_root_of(ep.value)
            missing = getattr(exc, "name", "") or ""
            if missing.startswith(own_root):
                red.append(
                    f"console script '{ep.name}' target '{ep.value}' imports its own package "
                    f"but fails inside it ({exc}) -- the target is not reachable"
                )
            else:
                unmeasured.append(
                    f"console script '{ep.name}' target needs optional module '{missing}' not "
                    "present here; undeclared-defect vs thin-environment cannot "
                    "be distinguished"
                )
        except Exception as exc:  # noqa: BLE001 -- fail closed, never skip
            red.append(
                f"console script '{ep.name}' load() raised unexpected {type(exc).__name__}: {exc}"
            )
        # Attribute the load() outcome to THIS script so the tally stays a
        # partition of the denominator. A script whose wrapper is missing AND
        # whose target is broken contributes two findings but is counted once.
        if len(red) > before[0]:
            if where:
                failed_wrapper += 1
                resolved -= 1
        elif len(unmeasured) > before[1] and where:
            unmeasured_scripts += 1
            resolved -= 1
    tally = ScriptTally(
        total=len(scripts),
        resolved=resolved,
        unmeasured=unmeasured_scripts,
        failed=failed_wrapper,
        on_path=on_path,
    )
    return red, unmeasured, tally


def scan_source_roots(
    dist_name: str, roots: Sequence[str]
) -> tuple[set[str], list[tuple[str, str]], list[str], int]:
    """Scan .py files for advertised extras and argparse prog= values.

    Returns (advertised_extras, prog_names[(name, path)], errors, files_scanned).
    Only quoted occurrences of ``<dist>[extra]`` count: the gate measures what
    source TELLS AN OPERATOR to install, not every bracket anywhere in prose
    or code.  Unreadable files are errors, never silent skips (rule 4).
    """
    extra_re = re.compile(
        r"['\"][^'\"]*?\b" + re.escape(dist_name) + r"\[([A-Za-z0-9_.\-]+)\][^'\"]*?['\"]"
    )
    prog_re = re.compile(r"prog\s*=\s*['\"]([^'\"]+)['\"]")
    advertised: set[str] = set()
    progs: list[tuple[str, str]] = []
    errors: list[str] = []
    files_scanned = 0
    for root in roots:
        if not Path(root).is_dir():
            # NOT an error here. An absent root is UNMEASURED and main() owns
            # that classification -- it prints the line and returns 95. Minting
            # a red as well made both fire, and red wins, so `cd /` turned this
            # gate's verdict from CLEAR to RED with no code changed and the
            # header docstring promising 95. Same defect class as the PATH
            # lookup this gate's console-script axis just lost: a verdict that
            # moves with the caller's shell state. Unreadable FILES below stay
            # red, because that is input the gate could not read, not input the
            # operator never pointed it at.
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = str(Path(dirpath) / filename)
                try:
                    with Path(path).open(encoding="utf-8") as handle:
                        text = handle.read()
                except (OSError, UnicodeDecodeError) as exc:
                    errors.append(f"cannot read '{path}': {exc}")
                    continue
                files_scanned += 1
                for match in extra_re.finditer(text):
                    advertised.add(match.group(1))
                for match in prog_re.finditer(text):
                    progs.append((match.group(1), path))
    return advertised, progs, errors, files_scanned


def check_extras(declared: set[str], advertised: set[str]) -> tuple[list[str], set[str]]:
    red = [
        "source advertises '{}[{}]' to operators but the installed "
        "distribution declares no such extra -- the remedy cannot be "
        "followed".format("<dist>", name)
        for name in sorted(advertised - declared)
    ]
    return red, declared - advertised


def check_prog_names(
    declared_scripts: Sequence[str], progs: Sequence[tuple[str, str]]
) -> tuple[list[str], int]:
    """prog= values containing a dot are treated as module paths and excluded;
    every other advertised command name must be a declared console script --
    that exact mismatch is finding #224.
    """
    declared = set(declared_scripts)
    red: list[str] = []
    considered = 0
    seen: set[str] = set()
    for name, path in progs:
        if "." in name:
            continue
        if name in seen:
            continue
        seen.add(name)
        considered += 1
        if name not in declared:
            red.append(
                f"argparse prog='{name}' ({path}) advertises a runnable command "
                "that is not a declared console script"
            )
    return red, considered


def report(kind: str, denominator: int, numerator: int, detail: str = "") -> None:
    # Denominator on every path (rule 2): a bare "N resolve" is noise.
    line = f"{kind}: {numerator} of {denominator}"
    if detail:
        line += " -- " + detail
    print(line)


class _StubEntryPoint:
    """Minimal EntryPoint stand-in for the console-script controls.

    Only .name, .value and .load() are consumed by check_console_scripts, so a
    stub is enough -- and it is what lets the controls exercise the failure
    outcomes (absent wrapper, unresolvable target, thin environment) without
    installing four broken distributions.
    """

    def __init__(self, name: str, value: str, raises: BaseException | None = None) -> None:
        self.name = name
        self.value = value
        self._raises = raises

    def load(self) -> object:
        if self._raises is not None:
            raise self._raises
        return self.load


def run_self_test(dist_name: str) -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pkg-reach-selftest-") as tmp:
        root = str(Path(tmp) / "src")
        Path(root).mkdir(parents=True)
        target = str(Path(root) / "controls_case.py")

        declared_scripts = ["foundationscale-real-cmd"]
        declared_extras = {"train"}

        def write(text: str) -> None:
            with Path(target).open("w", encoding="utf-8") as handle:
                handle.write(text)

        def scan() -> tuple[set[str], list[tuple[str, str]]]:
            advertised, progs, errors, files = scan_source_roots(dist_name, [root])
            if errors or files == 0:
                failures.append(
                    "MUST_PASS framework defect: self-test scan itself "
                    f"errored or saw zero files ({errors})"
                )
            return advertised, progs

        # MUST_FIRE 1: prog= for a command nobody declared.
        write('import argparse\np = argparse.ArgumentParser(prog="totally-unregistered-cmd")\n')
        _, progs = scan()
        red_prog, _ = check_prog_names(declared_scripts, progs)
        # Assert the finding COUNT, not an exit code: a control that exits
        # nonzero for an unrelated reason would otherwise validate a broken
        # detector (rule 3's anti-vacuity clause).
        if len(red_prog) == 0:
            failures.append(
                "MUST_FIRE control failed: prog='totally-unregistered-cmd' produced zero findings"
            )

        # MUST_FIRE 2: advertised extra that is not declared.
        write(f"# To enable: pip install '{dist_name}[nosuchextra]'\n")
        advertised, _ = scan()
        red_extra, _ = check_extras(declared_extras, advertised)
        if len(red_extra) == 0:
            failures.append(
                "MUST_FIRE control failed: advertised [{}] produced zero findings".format(
                    "nosuchextra"
                )
            )

        # MUST_PASS: only declared extras, prog= equal to a real script.
        write(
            "import argparse\n"
            f"# Remedy: pip install '{dist_name}[train]'\n"
            'p = argparse.ArgumentParser(prog="foundationscale-real-cmd")\n'
        )
        advertised, progs = scan()
        red_extra, _ = check_extras(declared_extras, advertised)
        red_prog, considered = check_prog_names(declared_scripts, progs)
        if red_extra:
            failures.append(
                f"MUST_PASS control failed: extras check flagged a declared extra: {red_extra}"
            )
        if red_prog or considered == 0:
            # considered == 0 means the clean prog= name was never even seen,
            # which would make the control vacuous even though nothing fired.
            failures.append(
                f"MUST_PASS control failed: prog findings={red_prog} considered={considered}"
            )

        # ---- console-script axis -------------------------------------------
        # This axis shipped with NO control and was the axis that was wrong: it
        # reported RED for two scripts pip had installed correctly, because it
        # asked the caller's PATH. Four controls, each isolating one outcome,
        # driven through injected script_dirs/record_basenames so the controls
        # themselves cannot depend on how this interpreter was invoked.
        bindir = str(Path(tmp) / "bin")
        Path(bindir).mkdir(parents=True)
        present = str(Path(bindir) / "fs-present-cmd")
        with Path(present).open("w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\n")

        def probe(eps: list[_StubEntryPoint]) -> tuple[list[str], list[str], ScriptTally]:
            return check_console_scripts(
                dist_name, eps, script_dirs=[bindir], record_basenames=set()
            )

        # MUST_FIRE 3: wrapper exists nowhere -> RED on the existence axis.
        red_s, unm_s, tally_s = probe([_StubEntryPoint("fs-ghost-cmd", f"{dist_name}.x:main")])
        if len(red_s) == 0 or tally_s.resolved != 0 or tally_s.failed != 1:
            failures.append(
                f"MUST_FIRE control failed: absent wrapper 'fs-ghost-cmd' "
                f"gave findings={len(red_s)} resolved={tally_s.resolved} failed={tally_s.failed}"
            )

        # MUST_FIRE 4: wrapper present, target unresolvable -> RED on the
        # resolution axis. Proves existence and resolution are separate tests;
        # pip installs a wrapper whether or not its target exists.
        red_s, unm_s, tally_s = probe(
            [
                _StubEntryPoint(
                    "fs-present-cmd",
                    f"{dist_name}.x:gone",
                    raises=AttributeError("module has no attribute 'gone'"),
                )
            ]
        )
        if len(red_s) == 0 or tally_s.resolved != 0:
            failures.append(
                f"MUST_FIRE control failed: present wrapper with broken "
                f"target gave findings={len(red_s)} resolved={tally_s.resolved}"
            )

        # MUST_FIRE 5 / MUST_PASS: a FOREIGN missing module is a thin
        # environment, not a defect -- it must land in unmeasured and NOT in
        # red. Without this the gate could clear its own denominator by
        # reclassifying everything it cannot resolve.
        exc = ImportError("No module named 'transformers'")
        exc.name = "transformers"
        red_s, unm_s, tally_s = probe(
            [_StubEntryPoint("fs-present-cmd", f"{dist_name}.x:main", raises=exc)]
        )
        if len(unm_s) == 0 or len(red_s) != 0 or tally_s.unmeasured != 1:
            failures.append(
                f"MUST_FIRE control failed: foreign ImportError gave "
                f"unmeasured={len(unm_s)} red={len(red_s)} "
                f"tally.unmeasured={tally_s.unmeasured} (expected 1/0/1)"
            )

        # MUST_PASS: wrapper present in the injected script dir, target loads.
        # Asserted with resolved == 1, not merely "no findings": a detector that
        # examined zero scripts would also produce no findings.
        red_s, unm_s, tally_s = probe([_StubEntryPoint("fs-present-cmd", f"{dist_name}.x:main")])
        if red_s or unm_s or tally_s.resolved != 1:
            failures.append(
                f"MUST_PASS control failed: installed wrapper with a "
                f"loadable target gave red={red_s} unmeasured={unm_s} resolved={tally_s.resolved}"
            )

    total_controls = 7
    if failures:
        print(
            f"SELF-TEST DENOMINATOR: {total_controls - len(failures)} of {total_controls} "
            "controls behaved; detector verdicts are worthless until this is fixed"
        )
        for failure in failures:
            print("CONTROL DEFECT: " + failure)
        return EXIT_RED
    report(
        "SELF-TEST DENOMINATOR",
        total_controls,
        total_controls,
        "5x MUST_FIRE produced nonzero finding counts; 2x MUST_PASS stayed "
        "clean over a nonzero denominator",
    )
    return EXIT_CLEAR


def build_parser() -> argparse.ArgumentParser:
    parser = GateArgumentParser(
        prog="packaging_reachability",
        description="Gate: declared/advertised packaging names must be reachable "
        "in the installed distribution (closes #224).",
    )
    parser.add_argument(
        "--dist",
        default=DEFAULT_DIST,
        help="installed distribution name to interrogate (default: %(default)s)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run MUST_FIRE/MUST_PASS controls in a tempdir and exit",
    )
    parser.add_argument(
        "roots", nargs="*", default=None, help="source roots to scan (default: src/)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test(args.dist)

    roots = args.roots if args.roots else ["src/"]
    dist_name = args.dist
    red: list[str] = []
    unmeasured_msgs: list[str] = []

    for root in roots:
        if Path(root).exists() and not Path(root).is_dir():
            # An invocation pointing the scanner at a file is invalid, not
            # unmeasurable: the operator must fix the command line.
            print(f"REFUSE: source root '{root}' exists but is not a directory")
            return EXIT_REFUSE

    scripts = get_console_scripts(dist_name)
    declared_extras = get_declared_extras(dist_name)
    if scripts is None or declared_extras is None:
        # The gate interrogates the INSTALLED distribution; pyproject.toml on
        # disk cannot answer "what did pip produce" (rule 1: zero examined).
        print(
            f"UNMEASURED: distribution '{dist_name}' is not installed; this gate can "
            "only interrogate an installed distribution. Run "
            "`pip install -e .` and re-run. All three denominators are 0/0: "
            "console scripts 0 of 0; extras 0 of 0; prog= names 0 of 0."
        )
        return EXIT_UNMEASURED

    missing_roots = [r for r in roots if not Path(r).is_dir()]
    if missing_roots:
        for root in missing_roots:
            print(
                f"UNMEASURED: source root '{root}' does not exist; advertised-name "
                "checks (extras, prog=) cannot measure what they claim to "
                "measure"
            )

    script_red, script_unmeasured, tally = check_console_scripts(dist_name, scripts)
    red.extend(script_red)

    advertised, progs, scan_errors, files_scanned = scan_source_roots(dist_name, roots)
    red.extend(scan_errors)  # unreadable input is red, never skipped (rule 4)

    extra_red, undeclared_unused = check_extras(declared_extras, advertised)
    red.extend(extra_red)

    script_names = [ep.name for ep in scripts]
    prog_red, prog_considered = check_prog_names(script_names, progs)
    red.extend(prog_red)
    unmeasured_msgs.extend(script_unmeasured)

    # Three denominators, always printed, on every outcome path (rule 2/5).
    report(
        "DENOMINATOR console-scripts",
        tally.total,
        tally.resolved,
        f"{tally.resolved} resolved against the interpreter's script dir or the installed "
        f"record, {tally.unmeasured} unmeasured (thin env), {tally.failed} unreachable",
    )
    if scripts:
        # Reported, never RED. PATH membership is a property of the caller's
        # shell, not of the package; see check_console_scripts' docstring.
        print(
            f"INFO only (never red): {tally.on_path} of {tally.total} console script(s) are on "
            "this shell's PATH -- operator convenience, not a packaging property"
        )
    report(
        "DENOMINATOR extras",
        len(advertised),
        len(advertised) - len(extra_red),
        f"{len(advertised)} source-advertised extra(s) checked against {len(declared_extras)} "
        f"declared by '{dist_name}'",
    )
    report(
        "DENOMINATOR prog-names",
        prog_considered,
        prog_considered - len(prog_red),
        f"{files_scanned} .py file(s) scanned",
    )

    if undeclared_unused:
        print(
            "INFO only (never red): declared but never advertised extras: {}".format(
                ", ".join(sorted(undeclared_unused))
            )
        )
    for msg in unmeasured_msgs:
        print("UNMEASURED item: " + msg)

    if red:
        print(f"RED: {len(red)} finding(s)")
        for finding in red:
            print("FINDING: " + finding)
        return EXIT_RED

    if not scripts and not advertised and prog_considered == 0:
        # all([]) is True: three empty denominators mean nothing was measured.
        print(
            "UNMEASURED: distribution declares zero console scripts and zero "
            "extras, and the source scan found zero prog= names -- this gate "
            "measured nothing and refuses to call that a clean result"
        )
        return EXIT_UNMEASURED
    if missing_roots:
        return EXIT_UNMEASURED
    if unmeasured_msgs:
        print(
            f"UNMEASURED: no defects surfaced, but {len(unmeasured_msgs)} console-script "
            "resolution(s) could not be measured in this environment; "
            "clear is not claimed"
        )
        return EXIT_UNMEASURED

    print("CLEAR: every measured name resolves; denominators printed above")
    return EXIT_CLEAR


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 -- fail closed, never exit 1
        print(
            f"RED: unexpected exception escaped main(): {type(exc).__name__}: {exc} -- this gate "
            "fails closed rather than risk a traceback exit code outside "
            "the 0/5/95/96 namespace"
        )
        sys.exit(EXIT_RED)
