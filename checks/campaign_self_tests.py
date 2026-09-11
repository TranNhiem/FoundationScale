"""Runner and declaration gate for ``validation_campaigns/`` ``--self-test`` modules.

``pyproject.toml`` sets ``testpaths = ["tests"]``, so pytest never collects anything under
``validation_campaigns/``. Campaign modules ship a ``--self-test`` flag that plants controls
and returns 0 only if every control passes. A gate nobody invokes states no coverage, and
this repository has already closed that "orphan gate" defect class more than once. This gate
invokes them.

The population is MEASURED, not hand-listed: a tracked campaign module is in scope when its
AST contains a ``--self-test`` string-literal node. A measured denominator normally carries
the shrinking risk -- delete the self-test and the file leaves the denominator, and the
absence reads as a pass. That hole is closed here by making the two rules two-way:

    R1 says every MEASURED (literal-bearing) file must appear in a DECLARED set.
    R6 says every DECLARED file must be measured as literal-bearing.

Deleting a self-test therefore does not shrink anything quietly: the file drops out of the
measured set while staying in the declared set, and R6 fires MISDECLARED. Adding one does not
pass unnoticed either: it enters the measured set with no declaration, and R1 fires UNDECLARED.
Neither side can move alone. This is why no hand-written ``NO_SELF_TEST`` table exists -- the
62 campaign files with no self-test need no reason strings, because they are outside the claim
rather than excused from it, and the banner reports their count as a measurement.

WHAT IS CLAIMED:
    - Each RUNNABLE module exists on disk, contains a ``--self-test`` string literal in its
      AST, was executed as ``[sys.executable, path, "--self-test"]`` within the per-module
      timeout, and returned exit code 0.
    - RUNNABLE and NOT_RUNNABLE_HERE partition the measured population: R1 undeclared,
      R2 double-claimed, R3 stale, R4 empty reason, R5 generic (reused) reason,
      R6 declared-but-not-literal-bearing.

WHAT IS NOT CLAIMED:
    - NOT_RUNNABLE_HERE is an UNMEASURED population, never a pass: its members are not
      asserted to pass, and nothing is claimed about them beyond the declaration itself.
    - Detection parses each module with ``ast`` and looks for the string literal. A flag name
      built by concatenation or computed at runtime would be invisible to this check, and a
      mention inside a comment is deliberately invisible -- a comment is not a flag.
    - That a self-test's CONTROLS are adequate. This gate claims only that the self-test
      exists, ran, and returned 0. A self-test whose controls abstain and which then exits 0
      passes this gate while measuring nothing; where that is known to be true of a declared
      module, its reason string says so.

EXIT CODES: 0 = CLEAR, 5 = RED, 95 = UNMEASURED, 96 = REFUSE. Never 1.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = (
    "EXIT_CLEAR",
    "EXIT_RED",
    "EXIT_REFUSE",
    "EXIT_UNMEASURED",
    "NOT_RUNNABLE_HERE",
    "RUNNABLE",
    "ModuleResult",
    "Report",
    "build_report",
    "check_declaration_rules",
    "main",
    "measured_population",
    "repo_root",
    "run_self_test",
    "self_test",
    "self_test_literal_count",
    "tracked_campaign_files",
)

EXIT_CLEAR = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96
DEFAULT_TIMEOUT = 300.0
SELF_TEST_FLAG = "--self-test"

RUNNABLE: dict[str, str] = {
    "validation_campaigns/h100_validation/fs_argv_preflight.py": (
        "Twelve controls over the argv-splice preflight, all measured here: the launcher "
        "argv it accepts and each malformed argv it must refuse. Pure stdlib argv parsing, "
        "no estate access and no GPU, so a laptop run and an estate run see the same 12."
    ),
    "validation_campaigns/h100_validation/fs_required_knobs.py": (
        "Fifteen controls over the required-knob table, all measured here. The knob names "
        "are literals in the module rather than read from a live job, so the check is a "
        "pure table comparison that needs neither Slurm nor a container."
    ),
    "validation_campaigns/h100_validation/control_scratch_restore.py": (
        "Fourteen controls over the #294 restore instruments -- the rm-line parser, the "
        "survivor counter, the byte comparator, the snapshot-leak detector, the reached-site "
        "banner and the verdict precedence -- each planted so it fires BOTH ways. The "
        "two-arm measurement itself runs the build script twice and is invoked separately "
        "by `make control-scratch-restore`; the self-test is pure stdlib over "
        "TemporaryDirectory fixtures, so a laptop and the estate see the same 14."
    ),
    "validation_campaigns/nemo_rl_baseline/adjudicate_bench_exit.py": (
        "Twelve controls that plant every verdict plus negative controls in a "
        "TemporaryDirectory and prove exit codes {0, 5, 95, 96} are each distinctly "
        "reachable, so a torchrun-flattened child code cannot be read as a pass; pure "
        "stdlib, no cluster."
    ),
    "validation_campaigns/nemo_rl_baseline/bench_weight_sync.py": (
        "The benchmark harness needs a tray of GPUs and a live process group to BENCHMARK, "
        "but its ten controls cover the declaration and adjudication arithmetic it does "
        "before any collective is formed, and that part is stdlib -- the self-test never "
        "reaches a torch import."
    ),
    "validation_campaigns/nemo_rl_baseline/run_bench_weight_sync.py": (
        "Six controls that spawn fake launchers and prove the record-freshness rule fires: "
        "a pre-existing record the launcher never rewrote must REFUSE rather than "
        "adjudicate cleanly, with the positive control proving that detector can fire."
    ),
    "validation_campaigns/verification_matrix/t1_6_symlinked_shard.py": (
        "Nine controls over the #343 symlinked-shard detector, split three detector / six "
        "instrument. The three that assert the defect is caught (a linked shard refuses, a "
        "dangling link reads as a LINK not as 'missing', the three arms agree on a fixture) "
        "flip RED against a pre-fix tree; the six that certify the instrument -- a regular "
        "shard is NOT refused, the byte-credit counter fires on a link-following stat, stays "
        "silent on lstat and on unwatched paths, patches and restores all three callables, "
        "an absent shard reads as missing, a directory with no index yields UNMEASURED -- "
        "hold in BOTH states, which is what makes the three flips attributable. The fixture "
        "is built from stub shards in a TemporaryDirectory and dcp_meta imports only stdlib "
        "at module scope (torch is function-local), so a laptop and the estate see the same "
        "9. The estate measurement against a real multi-shard checkpoint is a separate run "
        "of the same module with a checkpoint directory as argv."
    ),
}

NOT_RUNNABLE_HERE: dict[str, str] = {
    "validation_campaigns/h100_validation/h100/gen/fs_argv_preflight.py": (
        "The generated copy of the preflight, and it declares itself unmeasurable here "
        "rather than pretending otherwise: run in-tree it reaches 7 of its 12 controls and "
        "exits 95, because from its generated location it doubles its own fixture path "
        "prefix (looking for h100/gen/h100/gen/fs_train.fixed.py) and cannot import "
        "gate_launch_doc, so the remaining 5 abstain. Declared NOT_RUNNABLE_HERE because "
        "95 is the honest verdict for it on this machine and this gate must not launder an "
        "UNMEASURED into a pass; the doubled prefix is a defect in the generator, tracked "
        "separately, and closing it is what would move this file into RUNNABLE."
    ),
}


@dataclass(frozen=True, slots=True)
class ModuleResult:
    """Outcome of one attempted self-test; ``exit_code`` is None when no exit was observed."""

    path: str
    state: str
    exit_code: int | None
    stdout: str
    detail: str


@dataclass(frozen=True, slots=True)
class Report:
    """Aggregate over the runnable population; counts derived once, immutably."""

    results: tuple[ModuleResult, ...]
    passed: int
    red: tuple[str, ...]
    refused: tuple[str, ...]


def repo_root() -> Path:
    """Repository root, from this file's location; no environment and no cwd trust."""
    return Path(__file__).resolve().parent.parent


def tracked_campaign_files(root: Path) -> tuple[list[str], str | None]:
    """Tracked campaign ``.py`` files from the git index; error string on failure.

    The git index is the denominator, never a filesystem walk, because a walk measures the
    machine -- stale build trees, ``__pycache__``, untracked scratch -- rather than the
    repository.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "validation_campaigns"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"git executable unavailable or hung: {exc}"
    if proc.returncode != 0:
        return [], f"git ls-files exited {proc.returncode}: {proc.stderr.strip()}"
    files = sorted(line for line in proc.stdout.splitlines() if line.endswith(".py"))
    return files, None


def self_test_literal_count(path: Path) -> int | None:
    """Count ``--self-test`` string-literal AST nodes; None if unreadable or unparseable.

    A comment mentioning the flag is invisible to ``ast`` on purpose: a comment is not a
    flag, and a gate that counted comments would put a file into its denominator on the
    strength of prose.
    """
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, SyntaxError):
        return None
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == SELF_TEST_FLAG
    )


def measured_population(root: Path, tracked: list[str]) -> tuple[list[str], list[str]]:
    """Split tracked files into (literal-bearing, unparseable) by AST measurement."""
    bearing: list[str] = []
    unparseable: list[str] = []
    for path in tracked:
        found = self_test_literal_count(root / path)
        if found is None:
            unparseable.append(path)
        elif found > 0:
            bearing.append(path)
    return bearing, unparseable


def check_declaration_rules(
    root: Path,
    measured: list[str],
    runnable: dict[str, str],
    not_runnable: dict[str, str],
) -> list[str]:
    """Enforce R1..R6 over the declaration partition; messages name both sides' counts."""
    violations: list[str] = []
    sets = (("RUNNABLE", runnable), ("NOT_RUNNABLE_HERE", not_runnable))
    claims: dict[str, list[str]] = {}
    for set_name, table in sets:
        for path in table:
            claims.setdefault(path, []).append(set_name)
    census = (
        f"measured={len(measured)} RUNNABLE={len(runnable)} NOT_RUNNABLE_HERE={len(not_runnable)}"
    )
    for path in measured:
        if path not in claims:
            violations.append(
                f"R1 UNDECLARED: {path} carries a '{SELF_TEST_FLAG}' literal but is in 0 of "
                f"2 declaration sets ({census})"
            )
    for path, owners in sorted(claims.items()):
        if len(owners) > 1:
            violations.append(
                f"R2 DOUBLE-CLAIMED: {path} claimed by {len(owners)} of 2 sets "
                f"({', '.join(owners)}); the partition requires exactly 1"
            )
    by_reason: dict[str, list[str]] = {}
    for set_name, table in sets:
        for path, reason in sorted(table.items()):
            if not (root / path).is_file():
                violations.append(
                    f"R3 STALE: {path} declared in {set_name} (1 of {len(table)} "
                    f"declaration(s)) but present on disk 0 of 1 time(s)"
                )
                continue
            if reason.strip():
                by_reason.setdefault(reason, []).append(path)
            else:
                violations.append(
                    f"R4 EMPTY REASON: {path} in {set_name} has 0 non-whitespace "
                    f"character(s); a declaration reason requires at least 1"
                )
            found = self_test_literal_count(root / path)
            if found is None:
                violations.append(
                    f"R6 MISDECLARED: {path} in {set_name} cannot be ast-parsed to count "
                    f"'{SELF_TEST_FLAG}' literals (parsed 0 of 1 file(s))"
                )
            elif found == 0:
                violations.append(
                    f"R6 MISDECLARED: {path} declared {set_name} expects at least 1 "
                    f"'{SELF_TEST_FLAG}' literal but the source contains 0; a deleted "
                    f"self-test leaves the measured population and must be seen here, not "
                    f"read as one fewer thing to check"
                )
    for paths in by_reason.values():
        if len(paths) > 1:
            violations.append(
                f"R5 GENERIC REASON: 1 reason string serves {len(paths)} files "
                f"({'; '.join(sorted(paths))}); {len(paths)} distinct reasons are required"
            )
    return violations


def _as_text(value: str | bytes | None) -> str:
    """Normalise subprocess output that may be bytes, str, or missing."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_self_test(path: str, root: Path, timeout: float) -> ModuleResult:
    """Run one module's ``--self-test``; exit 0 is PASS, else FAIL, TIMEOUT, or REFUSE."""
    try:
        proc = subprocess.run(
            [sys.executable, str(root / path), SELF_TEST_FLAG],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ModuleResult(
            path,
            "TIMEOUT",
            None,
            _as_text(exc.stdout),
            f"exceeded {timeout}s timeout; a self-test that hangs has not passed",
        )
    except OSError as exc:
        return ModuleResult(path, "REFUSE", None, "", f"spawn failed: {exc}")
    if proc.returncode == 0:
        return ModuleResult(path, "PASS", 0, proc.stdout, "")
    return ModuleResult(
        path,
        "FAIL",
        proc.returncode,
        proc.stdout,
        f"self-test returned {proc.returncode}, expected 0",
    )


def build_report(results: list[ModuleResult]) -> Report:
    """Fold per-module results into the immutable report."""
    return Report(
        results=tuple(results),
        passed=sum(1 for r in results if r.state == "PASS"),
        red=tuple(f"{r.path}={r.state}" for r in results if r.state in ("FAIL", "TIMEOUT")),
        refused=tuple(f"{r.path}: {r.detail}" for r in results if r.state == "REFUSE"),
    )


def _stub_source(code: int, message: str) -> str:
    """A minimal campaign module returning ``code`` under ``--self-test``."""
    return (
        "import sys\n\n"
        "def main() -> int:\n"
        "    if '--self-test' in sys.argv:\n"
        f"        print({message!r})\n"
        f"        return {code}\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )


def _self_test_declaration_rules(root: Path, camp: str) -> list[tuple[str, bool, str]]:
    """Plant a tree where each of R1..R6 must fire, and one where none may.

    Returns one ``(name, behaved, detail)`` record per control rather than only the
    failures. The count is the point: a suite leg holds this gate's denominator to a
    floor, so a control set that quietly shrinks has to be visible as a smaller number
    and not merely as a still-clean banner.
    """
    controls: list[tuple[str, bool, str]] = []
    stubs = {
        f"{camp}/a_pass.py": _stub_source(0, "passing stub control ok"),
        f"{camp}/b_fail.py": _stub_source(5, "failing stub control ok"),
        f"{camp}/c_undeclared.py": "FLAG = '--self-test'\n",
        f"{camp}/d_double.py": "FLAG = '--self-test'\n",
        f"{camp}/e_empty.py": "FLAG = '--self-test'\n",
        f"{camp}/g_one.py": "FLAG = '--self-test'\n",
        f"{camp}/g_two.py": "FLAG = '--self-test'\n",
        f"{camp}/i_no_literal.py": "VALUE = 99\n",
        f"{camp}/j_comment_only.py": "# mentions --self-test in a comment only\nVALUE = 1\n",
    }
    (root / camp).mkdir(parents=True, exist_ok=True)
    for rel, content in stubs.items():
        (root / rel).write_text(content, encoding="utf-8")
    measured, unparseable = measured_population(root, sorted(stubs))
    controls.append(
        (
            "stubs-parse",
            not unparseable,
            f"{len(unparseable)} of {len(stubs)} planted stub(s) failed to parse: {unparseable}",
        )
    )
    controls.append(
        (
            "comment-only-mention-excluded",
            f"{camp}/j_comment_only.py" not in measured,
            "a --self-test mention in a comment entered the measured population; the ast rule "
            "is reading text, not literals",
        )
    )
    controls.append(
        (
            "no-literal-excluded",
            f"{camp}/i_no_literal.py" not in measured,
            "a file carrying no --self-test literal entered the measured population",
        )
    )
    for required in (f"{camp}/a_pass.py", f"{camp}/c_undeclared.py"):
        controls.append(
            (
                f"literal-bearing-included:{Path(required).name}",
                required in measured,
                f"{required} carries the literal but is not in the measured set",
            )
        )
    runnable = {
        f"{camp}/a_pass.py": "passing stub proves the runner observes a clean exit 0",
        f"{camp}/b_fail.py": "failing stub proves the runner observes a non-zero exit",
        f"{camp}/d_double.py": "double-claim stub for the R2 firing control",
        f"{camp}/g_one.py": "shared reason planted for the R5 control",
        f"{camp}/g_two.py": "shared reason planted for the R5 control",
        f"{camp}/i_no_literal.py": "declared runnable but carries no literal, for R6",
        f"{camp}/z_stale.py": "declared but never written to disk, for the R3 control",
    }
    not_run = {
        f"{camp}/d_double.py": "the other side of the double-claim control",
        f"{camp}/e_empty.py": "   ",
    }
    violations = check_declaration_rules(root, measured, runnable, not_run)
    for rule in (
        "R1 UNDECLARED",
        "R2 DOUBLE-CLAIMED",
        "R3 STALE",
        "R4 EMPTY REASON",
        "R5 GENERIC REASON",
        "R6 MISDECLARED",
    ):
        controls.append(
            (
                f"fires:{rule.split()[0]}",
                any(v.startswith(rule) for v in violations),
                f"{rule} had no firing control; got {violations}",
            )
        )
    clean_measured = [f"{camp}/a_pass.py", f"{camp}/b_fail.py"]
    clean = check_declaration_rules(
        root,
        clean_measured,
        {
            f"{camp}/a_pass.py": "clean pass stub, its own reason",
            f"{camp}/b_fail.py": "clean fail stub, a different reason",
        },
        {},
    )
    controls.append(
        (
            "clean-configuration-silent",
            not clean,
            f"clean configuration produced spurious violation(s): {clean}",
        )
    )
    return controls


def self_test() -> int:
    """Plant a temporary tree and prove every rule R1..R6 fires, plus the runner states.

    The trailing ``SELF-TEST DENOMINATOR`` line is the machine-readable part: the suite
    leg parses it and refuses a tally below a floor, so deleting a control is a RED
    rather than a quieter pass.
    """
    camp = "validation_campaigns"
    controls: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        controls.extend(_self_test_declaration_rules(root, camp))
        got_pass = run_self_test(f"{camp}/a_pass.py", root, timeout=60)
        controls.append(
            (
                "runner-observes-PASS",
                got_pass.state == "PASS" and got_pass.exit_code == 0,
                f"passing stub control: {got_pass.state} exit={got_pass.exit_code}",
            )
        )
        got_fail = run_self_test(f"{camp}/b_fail.py", root, timeout=60)
        controls.append(
            (
                "runner-observes-FAIL",
                got_fail.state == "FAIL" and got_fail.exit_code == 5,
                f"failing stub control: {got_fail.state} exit={got_fail.exit_code}",
            )
        )
        missing = run_self_test(f"{camp}/z_stale.py", root, timeout=60)
        controls.append(
            (
                "absent-module-does-not-PASS",
                missing.state != "PASS",
                "a module absent from disk reported PASS; absence must not read as a pass",
            )
        )
        timed_out = run_self_test(f"{camp}/a_pass.py", root, timeout=1e-6)
        controls.append(
            (
                "runner-observes-TIMEOUT",
                timed_out.state == "TIMEOUT",
                f"a self-test given an unmeetable timeout reported {timed_out.state}; a hang "
                f"must not read as a pass",
            )
        )
    behaved = [name for name, ok, _ in controls if ok]
    failed = [(name, detail) for name, ok, detail in controls if not ok]
    for name, detail in failed:
        print(f"RED campaign_self_tests --self-test: control {name} did not behave: {detail}")
    print(
        f"SELF-TEST DENOMINATOR: {len(behaved)} of {len(controls)} controls behaved; "
        f"R1..R6 each fired, a comment-only mention stayed out of the measured population, "
        f"the clean configuration stayed silent, and PASS / FAIL / TIMEOUT / absent runner "
        f"states were each observed"
    )
    return EXIT_RED if failed else EXIT_CLEAR


def main(argv: list[str] | None = None) -> int:
    """Audit the declaration partition, then execute every RUNNABLE self-test."""
    parser = argparse.ArgumentParser(
        description=(
            "Run declared campaign --self-test modules and audit the declaration partition."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run this gate's own planted controls and exit",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="per-module timeout in seconds (default 300)",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    root = repo_root()
    declared_total = len(RUNNABLE) + len(NOT_RUNNABLE_HERE)
    tracked, error = tracked_campaign_files(root)
    if error is not None:
        print(
            f"REFUSE campaign_self_tests: field tracked population has 0 file(s) vs "
            f"{declared_total} declared path(s): {error}; no filesystem-walk fallback"
        )
        return EXIT_REFUSE
    measured, unparseable = measured_population(root, tracked)
    if unparseable:
        print(
            f"REFUSE campaign_self_tests: field measured population could not be taken -- "
            f"{len(unparseable)} of {len(tracked)} tracked file(s) failed to ast-parse "
            f"({', '.join(unparseable[:3])}); an unparseable file is unmeasured, not absent"
        )
        return EXIT_REFUSE
    violations = check_declaration_rules(root, measured, RUNNABLE, NOT_RUNNABLE_HERE)
    if violations:
        for item in violations:
            print(f"RED campaign_self_tests: {item}")
        print(
            f"RED campaign_self_tests: {len(violations)} declaration violation(s) across "
            f"{len(measured)} measured file(s) and {declared_total} declared path(s)"
        )
        return EXIT_RED
    if not RUNNABLE:
        print(
            f"REFUSE campaign_self_tests: field RUNNABLE has 0 entries against "
            f"{len(NOT_RUNNABLE_HERE)} NOT_RUNNABLE_HERE; all([]) is True, so an empty "
            f"required set would read as a pass -- the founding defect of this codebase"
        )
        return EXIT_REFUSE
    results = [run_self_test(path, root, args.timeout) for path in sorted(RUNNABLE)]
    for result in results:
        for line in result.stdout.splitlines():
            print(f"  {line}")
        line = f"{result.state} {result.path}"
        if result.exit_code is not None:
            line += f" exit={result.exit_code}"
        if result.detail:
            line += f" ({result.detail})"
        print(line)
    report = build_report(results)
    if report.refused:
        print(
            f"REFUSE campaign_self_tests: field results carries {len(report.refused)} of "
            f"{len(report.results)} runnable self-test(s) that could not be spawned: "
            f"{report.refused[0]}"
        )
        return EXIT_REFUSE
    if report.red:
        print(
            f"RED campaign_self_tests: {len(report.red)} of {len(report.results)} runnable "
            f"self-test(s) did not pass: {'; '.join(report.red)}"
        )
        return EXIT_RED
    print(
        f"CLEAR campaign_self_tests: {report.passed} of {len(report.results)} runnable "
        f"self-test(s) passed; {len(NOT_RUNNABLE_HERE)} declared NOT_RUNNABLE_HERE and "
        f"therefore UNMEASURED, not passed; {len(tracked) - len(measured)} of {len(tracked)} "
        f"tracked campaign file(s) carry no '{SELF_TEST_FLAG}' literal and are outside the "
        f"claim by measurement, not by an excuse written down here"
    )
    return EXIT_CLEAR


if __name__ == "__main__":
    sys.exit(main())
