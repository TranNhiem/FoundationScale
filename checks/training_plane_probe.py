#!/usr/bin/env python3
"""Probe: the training plane has two axes, and a zero on one is not a zero on both.

WHY THIS EXISTS (findings #245 and #223)
    Four shipped review documents stated "src/foundationscale/ has no training
    code", citing an ad-hoc six-marker AST probe that was never committed.
    Those six markers read 0 across all git-tracked src/*.py -- and still do.
    But the package DOES train: src/foundationscale/train/loop.py builds a
    transformers.Trainer and calls trainer.train(), importing torch and
    transformers at FUNCTION scope, so every module-scope marker stays at
    zero.  The old probe was literally true and materially misleading: it
    could not distinguish "does not train" from "DELEGATES training to a
    third party".  Because it lived in no committed file, nothing went red
    when train/ landed.  This probe is the committed, controlled replacement.

TWO AXES, REPORTED SEPARATELY, NEVER MERGED
    Axis A -- PRIMITIVES: does src/ IMPLEMENT training itself?
        module_scope_torch_import, nn_module_subclass, forward_definition,
        backward_call, optimizer_step_call, dataloader_construction.
    Axis B -- DELEGATION: does src/ DRIVE a third-party trainer?
        trainer_construction (call target is Trainer / *Trainer),
        fit_or_train_or_save_call (any .train()/.save_model()/.fit()),
        automodel_from_pretrained (AutoModel*.from_pretrained),
        datacollator_construction (DataCollator*), and -- the marker that
        makes the old zero legible -- function_scope_lazy_import of torch /
        transformers / accelerate / datasets / peft.

VERDICT SEMANTICS (the design; do not soften it)
    * Axis A nonzero            -> IMPLEMENTS-PRIMITIVES.  Reported, exit 0.
                                   Implementing primitives is not an error.
    * Axis A zero, Axis B zero  -> NO-TRAINING-PLANE.  A plain finding, exit 0.
    * Axis A zero, Axis B NONZERO -> the #223 condition.  Exit 5 (RED) IF AND
                                   ONLY IF a git-tracked *.md still asserts
                                   a retired bare-absence form (see DOC
                                   ASSERTIONS below).  Otherwise exit 0 with the
                                   two-axis summary.  The zero alone must never
                                   again be publishable as an absence.
    * Zero files scanned        -> 95 UNMEASURED.  Never 0: all([]) is True,
                                   and that vacuity is the bug this repository
                                   exists to prevent.
    * git unavailable / not a repo / index unreadable -> 96 REFUSE, with git's
                                   own stderr.  There is no filesystem fallback.

DENOMINATOR
    `git ls-files -z -- src`, filtered to *.py, decoded utf-8, run from the
    root resolved by `git rev-parse --show-toplevel`.  NEVER rglob: a
    blocklist walk cannot define "the repository", and a stale build/ tree
    already doubled this package's line count once (#244).  The file count
    is printed inside every verdict line.

DOC ASSERTIONS
    ALL git-tracked *.md are scanned (compiled case-insensitive regexes) --
    repo-wide, not just docs/, because README.md carries this claim too --
    for the retired bare-absence forms.  Those phrasings are RETIRED as of
    #245: the corrected form always names both axes.  Each hit is reported as
    file:line.  The scanner is inside its own denominator -- this module's
    source contains those patterns as string literals -- so its own file is
    excluded by RESOLVED path, and no general allowlist is added.

CONTROLS (--self-test)
    Seven, each built in a tempfile.TemporaryDirectory with real source,
    each run through the REAL analysis functions (never a re-implementation),
    each asserting exact expected axis counts:
      MUST_FIRE  planted delegating trainer: imports only inside a function,
                 Trainer(...) then .train().  Axis A stays 0, Axis B >= 1.
                 This is the positive control #223 lacked.
      MUST_FIRE  planted primitive trainer: all six axis-A markers >= 1.
      MUST_PASS  pure-verification stdlib module: both axes 0.
      MUST_PASS  retired phrasings in string literals and comments only:
                 both axes 0 (use vs mention).
      MUST_FIRE  a doc fixture stating a retired form: exactly 1 hit at the
                 right line number.
      MUST_PASS  a doc fixture stating the corrected two-axis form: 0 hits --
                 the negative twin, without which control 5 validates a
                 scanner that matches everything.
      MUST_PASS  empty denominator: analysis reports UNMEASURED and the exit
                 helper maps it to 95, not to a clean pass.
    The harness exits 5 if any control misbehaves, and refuses to pass over
    an empty control list -- a control harness over zero controls is the same
    vacuous pass this whole gate exists to kill.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

EXIT_CLEAR = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

STATUS_UNMEASURED = "UNMEASURED"
STATUS_PRIMITIVES = "IMPLEMENTS-PRIMITIVES"
STATUS_DELEGATES = "DELEGATES-TRAINING"
STATUS_NO_PLANE = "NO-TRAINING-PLANE"

# Axis A: evidence the package implements the training primitives itself.
AXIS_A_MARKERS = (
    "module_scope_torch_import",
    "nn_module_subclass",
    "forward_definition",
    "backward_call",
    "optimizer_step_call",
    "dataloader_construction",
)

# Axis B: evidence the package drives a third-party trainer.  The lazy-import
# marker is the whole point of #245: it is what turns the old probe's zero
# from a misleading absence into a legible delegation.
AXIS_B_MARKERS = (
    "trainer_construction",
    "fit_or_train_or_save_call",
    "automodel_from_pretrained",
    "datacollator_construction",
    "function_scope_lazy_import",
)

_LAZY_IMPORT_ROOTS = frozenset({"torch", "transformers", "accelerate", "datasets", "peft"})
_DRIVE_CALL_NAMES = frozenset({"train", "save_model", "fit"})

# Retired as of #245.  Anything matching one of these in a tracked doc is the
# old probe's zero being republished as an absence.
_RETIRED_ASSERTIONS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"contains?\s+no\s+training\s+code",
        r"zero\s+training\s+code",
        r"no\s+trainer\s+code",
        r"nothing\s+that\s+builds\s+a\s+model",
        # Added by MEASUREMENT, not by imagination.  While #245's corrections
        # were being applied, a freshly generated deliverable reproduced the
        # defect verbatim as "zero training constructs" -- a phrasing none of
        # the four patterns above matches, in a document written AFTER the
        # corrections landed.  A pattern list assembled from the wordings one
        # happens to remember is a denominator that grows only when someone
        # notices; this entry is the one case where the corpus taught the
        # detector.  Note the CORRECTED wording deliberately says "no training
        # PRIMITIVES", which no pattern here matches and control 6 pins.
        r"zero\s+training\s+constructs",
    )
)


class GateArgumentParser(argparse.ArgumentParser):
    def exit(self, status: int = 0, message: str | None = None) -> None:
        # argparse exits 2 on bad arguments; without this remap an operator
        # typo would escape the four-code namespace this probe advertises.
        if message:
            sys.stderr.write(message)
        raise SystemExit(EXIT_CLEAR if status == 0 else EXIT_REFUSE)


class RefusalError(Exception):
    """The invocation itself cannot legitimately produce a verdict (exit 96)."""


@dataclasses.dataclass(frozen=True)
class DocHit:
    path: str
    line: int
    pattern: str
    text: str


@dataclasses.dataclass(frozen=True)
class ProbeReport:
    files_scanned: int
    axis_a: dict[str, int]
    axis_b: dict[str, int]
    unparseable: tuple[str, ...]

    @property
    def axis_a_total(self) -> int:
        return sum(self.axis_a.values())

    @property
    def axis_b_total(self) -> int:
        return sum(self.axis_b.values())

    @property
    def status(self) -> str:
        # Order matters.  An empty sweep is UNMEASURED, never a clean
        # NO-TRAINING-PLANE: all([]) is True and that is the bug.  A
        # delegation is its own verdict, because the axis-A zero that
        # accompanies it was the misleading one.
        if self.files_scanned == 0:
            return STATUS_UNMEASURED
        if self.axis_a_total > 0:
            return STATUS_PRIMITIVES
        if self.axis_b_total > 0:
            return STATUS_DELEGATES
        return STATUS_NO_PLANE


def _terminal_name(func: ast.expr) -> str | None:
    # The call's target is the outermost attribute (a.b.c -> "c") or the bare
    # name.  Subscripts and other callables are not names we can reason about.
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _TrainingPlaneVisitor(ast.NodeVisitor):
    """One pass over one file, filling both axes from the SAME parse.

    A merged detector could never answer "is the zero an absence or a
    delegation?"; keeping two independent counter dicts on one walk is what
    makes the two-axis report cheap and impossible to conflate.
    """

    def __init__(self) -> None:
        self.axis_a: dict[str, int] = {marker: 0 for marker in AXIS_A_MARKERS}
        self.axis_b: dict[str, int] = {marker: 0 for marker in AXIS_B_MARKERS}
        self._function_depth = 0

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name == "forward":
            self.axis_a["forward_definition"] += 1
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            if _terminal_name(base) == "Module":
                self.axis_a["nn_module_subclass"] += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_imports([alias.name for alias in node.names])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # node.module is None for `from . import x`; nothing to classify.
        if node.module is not None:
            self._record_imports([node.module])

    def _record_imports(self, modules: list[str]) -> None:
        for module in modules:
            root = module.split(".")[0]
            if root not in _LAZY_IMPORT_ROOTS:
                continue
            if self._function_depth > 0:
                # THE legibility marker: an import hidden inside a function is
                # precisely the shape that kept the old module-scope probe at
                # zero while train/loop.py trained.
                self.axis_b["function_scope_lazy_import"] += 1
            elif root == "torch":
                self.axis_a["module_scope_torch_import"] += 1

    def visit_Call(self, node: ast.Call) -> None:
        target = _terminal_name(node.func)
        if target is not None:
            if target == "backward":
                self.axis_a["backward_call"] += 1
            if target == "step":
                self.axis_a["optimizer_step_call"] += 1
            if target == "DataLoader":
                self.axis_a["dataloader_construction"] += 1
            if target.endswith("Trainer"):
                self.axis_b["trainer_construction"] += 1
            if target in _DRIVE_CALL_NAMES:
                # Any .train()/.save_model()/.fit() attribute call counts; the
                # tighter "bound from a Trainer construction" form is subsumed.
                self.axis_b["fit_or_train_or_save_call"] += 1
            if target.startswith("DataCollator"):
                self.axis_b["datacollator_construction"] += 1
            if (
                target == "from_pretrained"
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id.startswith("AutoModel")
            ):
                self.axis_b["automodel_from_pretrained"] += 1
        self.generic_visit(node)


def analyze_source_text(text: str, origin: str) -> tuple[dict[str, int], dict[str, int]]:
    tree = ast.parse(text, filename=origin)
    visitor = _TrainingPlaneVisitor()
    visitor.visit(tree)
    return visitor.axis_a, visitor.axis_b


def analyze_sources(paths: list[Path]) -> ProbeReport:
    axis_a = {marker: 0 for marker in AXIS_A_MARKERS}
    axis_b = {marker: 0 for marker in AXIS_B_MARKERS}
    unparseable: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            found_a, found_b = analyze_source_text(text, str(path))
        except (OSError, SyntaxError, ValueError) as exc:
            # A file the parser rejects was NOT measured; the caller must see
            # the gap rather than inherit the file's zero as if it were data.
            unparseable.append(f"{path}: {exc}")
            continue
        for marker in AXIS_A_MARKERS:
            axis_a[marker] += found_a[marker]
        for marker in AXIS_B_MARKERS:
            axis_b[marker] += found_b[marker]
    return ProbeReport(
        files_scanned=len(paths),
        axis_a=axis_a,
        axis_b=axis_b,
        unparseable=tuple(unparseable),
    )


def scanned_docs(paths: list[Path], exclude: Path) -> list[Path]:
    """The doc files this scanner actually reads -- its denominator.

    Split out from ``scan_doc_assertions`` so the count on the wire and the
    set that was read are the same object.  Without this, "no tracked doc
    asserts the retired form" prints identically whether 23 docs were clean
    or the glob matched zero files, and the second case is ``all([])``.
    """
    exclude_resolved = exclude.resolve()
    return [path for path in paths if path.resolve() != exclude_resolved]


def scan_doc_assertions(paths: list[Path], exclude: Path) -> list[DocHit]:
    exclude_resolved = exclude.resolve()
    hits: list[DocHit] = []
    for path in paths:
        # This scanner is inside its own denominator: this module's source
        # carries the retired patterns as string literals (the regexes above
        # and the fixtures below).  It is excluded by RESOLVED path, exactly
        # one file, and no general allowlist is added -- a named exclusion is
        # an admission; a list would be a loophole.
        if path.resolve() == exclude_resolved:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RefusalError(f"cannot read git-tracked doc {path}: {exc}") from exc
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern in _RETIRED_ASSERTIONS:
                if pattern.search(line):
                    hits.append(DocHit(str(path), lineno, pattern.pattern, line.strip()))
    return hits


def exit_code_for(report: ProbeReport, doc_hits: int, docs_scanned: int = -1) -> int:
    # This helper is the verdict semantics, in one place, so the CLI and the
    # self-test can never disagree about what a reading means.
    if report.unparseable:
        return EXIT_RED
    status = report.status
    if status == STATUS_UNMEASURED:
        return EXIT_UNMEASURED
    if status == STATUS_DELEGATES and doc_hits > 0:
        # IF AND ONLY IF: a delegation artifact on axis A's zero is RED only
        # while some tracked document still publishes the retired bare form.
        return EXIT_RED
    if docs_scanned == 0:
        # The doc axis has its OWN denominator, and zero docs is UNMEASURED,
        # not clean (doctrine 1).  Without this branch a broken glob and a
        # genuinely corrected corpus both print "no tracked doc asserts the
        # retired form" and both exit 0 -- the source axis would carry a
        # verdict the doc axis never earned.  ``docs_scanned=-1`` is the
        # not-supplied default used by callers that measure only the source
        # axis; it is deliberately not 0, so "not asked" cannot masquerade
        # as "asked and empty".
        return EXIT_UNMEASURED
    return EXIT_CLEAR


def _decode(raw: bytes, what: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RefusalError(
            f"git output for {what} is not utf-8 ({exc}); refusing to guess"
        ) from exc


def _git(cwd: Path | None, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RefusalError(
            f"git could not be executed ({exc}); this probe defines its denominator by "
            "the index alone, and there is no filesystem fallback"
        ) from exc
    if proc.returncode != 0:
        stderr = _decode(proc.stderr, "stderr").strip() or "<no stderr>"
        raise RefusalError(
            f"git {' '.join(args)} exited {proc.returncode}: {stderr} -- no filesystem "
            "fallback: a blocklist walk cannot define 'the repository' (#244)"
        )
    return _decode(proc.stdout, "stdout")


def _repo_root() -> Path:
    return Path(_git(None, ["rev-parse", "--show-toplevel"]).strip())


def _tracked_files(root: Path, subdir: str, suffix: str) -> list[Path]:
    raw = _git(root, ["ls-files", "-z", "--", subdir])
    names = [name for name in raw.split("\0") if name]
    return sorted(root / name for name in names if name.endswith(suffix))


def _tracked_markdown(root: Path) -> list[Path]:
    """Every git-tracked ``*.md``, not just the ones under ``docs/``.

    Scoping this axis to ``docs/`` was the #205 shape: the instrument is
    instance-scoped while the defect class is plane-wide.  8 of the repo's 23
    tracked markdown files live outside ``docs/`` -- including ``README.md``,
    which is the single most-read document here and the one deliverable 9 is
    about to rewrite with exactly the claim these patterns retire.  A verdict
    that says "no doc asserts the retired form" while structurally unable to
    read ``README.md`` is a true sentence over the wrong denominator.
    """
    raw = _git(root, ["ls-files", "-z"])
    names = [name for name in raw.split("\0") if name]
    return sorted(root / name for name in names if name.endswith(".md"))


def _init_fixture_repo(root: Path) -> bool:
    """Stage a throwaway tree into a real git index for the scan-scope control.

    ``ls-files`` reads the INDEX, so ``add`` is sufficient and no commit (and
    therefore no user identity, no hooks, no signing key) is needed.  Returns
    False rather than raising if git is unavailable: the caller turns that into
    a declared UNMEASURED control, never a silent pass.
    """
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    for args in (["init", "-q"], ["add", "-A"]):
        try:
            proc = subprocess.run(
                ["git", *args], cwd=str(root), capture_output=True, check=False, env=env
            )
        except OSError:
            return False
        if proc.returncode != 0:
            return False
    return True


def report(kind: str, denominator: int, numerator: int, detail: str = "") -> None:
    # Denominator on every path: a bare "0 hits" is the noise that shipped #223.
    line = f"{kind}: {numerator} of {denominator}"
    if detail:
        line += " -- " + detail
    print(line)


def run_self_test() -> int:
    results: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="training-plane-selftest-") as tmp:
        root = Path(tmp)

        # MUST_FIRE 1 -- the positive control #223 lacked.  If this does not
        # fire, the instrument is blind in exactly the way the old probe was.
        delegating = root / "delegating_only.py"
        delegating.write_text(
            textwrap.dedent(
                """\
                def build_and_run(model, args, train_dataset):
                    import torch
                    from transformers import Trainer

                    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
                    trainer.train()
                    return trainer
                """
            ),
            encoding="utf-8",
        )
        r1 = analyze_sources([delegating])
        ok1 = (
            r1.axis_a_total == 0
            and r1.axis_b_total >= 1
            and r1.axis_b["function_scope_lazy_import"] >= 1
            and r1.status == STATUS_DELEGATES
        )
        results.append(
            (
                "MUST_FIRE planted delegating trainer (the #223 positive control)",
                ok1,
                f"axis A total={r1.axis_a_total} (want 0), axis B total={r1.axis_b_total} "
                f"(want >=1), lazy imports={r1.axis_b['function_scope_lazy_import']}",
            )
        )

        # MUST_FIRE 2 -- every one of the six axis-A markers, asserted
        # individually: a detector strong on five markers and blind on the
        # sixth has a hole the size of exactly that primitive.
        primitive = root / "primitive_only.py"
        primitive.write_text(
            textwrap.dedent(
                """\
                import torch
                from torch import nn
                from torch.utils.data import DataLoader


                class M(nn.Module):
                    def forward(self, x):
                        return x


                def run():
                    model = M()
                    opt = torch.optim.SGD(model.parameters(), lr=0.1)
                    loader = DataLoader([1, 2, 3])
                    loss = sum(model(torch.tensor(float(i))) for i in loader)
                    loss.backward()
                    opt.step()
                """
            ),
            encoding="utf-8",
        )
        r2 = analyze_sources([primitive])
        missing2 = [marker for marker in AXIS_A_MARKERS if r2.axis_a[marker] < 1]
        results.append(
            (
                "MUST_FIRE planted primitive trainer (all six axis-A markers)",
                not missing2,
                "markers still zero: " + (", ".join(missing2) if missing2 else "none"),
            )
        )

        # MUST_PASS 3 -- a pure-verification module must not trip either axis.
        verify = root / "pure_verification.py"
        verify.write_text(
            textwrap.dedent(
                """\
                import hashlib
                import json


                def digest(payload: dict) -> str:
                    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
                    return hashlib.sha256(encoded).hexdigest()
                """
            ),
            encoding="utf-8",
        )
        r3 = analyze_sources([verify])
        results.append(
            (
                "MUST_PASS pure-verification stdlib module",
                r3.axis_a_total == 0 and r3.axis_b_total == 0,
                f"axis A={r3.axis_a_total} axis B={r3.axis_b_total} (want 0/0)",
            )
        )

        # MUST_PASS 4 -- use vs mention.  The retired phrasings inside string
        # literals and comments are data to a doc scanner, not calls to an AST
        # probe; an AST detector that fires here is grepping, not parsing.
        mention = root / "mention_only.py"
        mention.write_text(
            textwrap.dedent(
                '''\
                """This package mentions the retired phrasings without performing them."""
                NOTE = "contains no training code"
                # zero training code lives on disk anywhere near here
                SECOND = "no trainer code at all"
                THIRD = "zero training constructs were found"


                def describe() -> str:
                    return "nothing that builds a model"
                '''
            ),
            encoding="utf-8",
        )
        r4 = analyze_sources([mention])
        results.append(
            (
                "MUST_PASS retired phrasings as strings/comments only (use vs mention)",
                r4.axis_a_total == 0 and r4.axis_b_total == 0,
                f"axis A={r4.axis_a_total} axis B={r4.axis_b_total} (want 0/0)",
            )
        )

        # MUST_FIRE 5 -- the doc scanner must catch the retired forms and pin
        # the exact line; a hit without a line number is not actionable.
        #
        # The fixture carries ONE line per entry in _RETIRED_ASSERTIONS, and
        # the assertion is that EVERY pattern fired.  An earlier draft planted
        # a single sentence and checked "1 hit at line 3": that is a control
        # over one pattern reported as a control over the table, so four of
        # five patterns would have sat in no denominator and could have been
        # broken silently.  A pattern that is never exercised is not a
        # detector, it is a comment.  Adding a pattern above without adding
        # its line here turns this leg RED, which is the intended coupling.
        retired_lines = [
            "This package contains no training code.",
            "The package has zero training code of its own.",
            "There is no trainer code in src/.",
            "It ships nothing that builds a model.",
            "The audit found zero training constructs in the package.",
        ]
        retired_doc = root / "retired_assertion.md"
        retired_doc.write_text("\n".join(retired_lines) + "\n", encoding="utf-8")
        hits5 = scan_doc_assertions([retired_doc], exclude=Path(__file__))
        fired5 = {hit.pattern for hit in hits5}
        want5 = {pattern.pattern for pattern in _RETIRED_ASSERTIONS}
        lines5 = sorted({hit.line for hit in hits5})
        ok5 = fired5 == want5 and lines5 == list(range(1, len(retired_lines) + 1))
        missing5 = sorted(want5 - fired5)
        results.append(
            (
                "MUST_FIRE doc fixture stating every retired bare form",
                ok5,
                f"patterns fired {len(fired5)} of {len(want5)}, lines={lines5} "
                f"(want {len(want5)} of {len(want5)}, one hit per line)"
                + (f"; never fired: {missing5}" if missing5 else ""),
            )
        )

        # MUST_PASS 6 -- the negative twin of control 5.  Without it, control 5
        # is satisfied by a scanner that matches every line of every doc.
        corrected_doc = root / "corrected_two_axis.md"
        corrected_doc.write_text(
            "# Review notes\n\nTraining plane, axis A (primitives): zero markers across "
            "src/.  Axis B (delegation): nonzero; the package drives a third-party "
            "trainer from src/foundationscale/train/loop.py.  Both axes are named.\n",
            encoding="utf-8",
        )
        hits6 = scan_doc_assertions([corrected_doc], exclude=Path(__file__))
        results.append(
            (
                "MUST_PASS doc fixture stating the corrected two-axis form",
                len(hits6) == 0,
                f"hits={len(hits6)} (want 0)",
            )
        )

        # MUST_PASS 7 -- an empty sweep is UNMEASURED (95), never a pass.  This
        # is the anti-vacuity clause exercised end to end: analysis classifies,
        # the exit helper maps.
        r7 = analyze_sources([])
        ok7 = r7.status == STATUS_UNMEASURED and exit_code_for(r7, 0) == EXIT_UNMEASURED
        results.append(
            (
                "MUST_PASS empty denominator is UNMEASURED mapped to 95",
                ok7,
                f"status={r7.status} exit={exit_code_for(r7, 0)} (want {STATUS_UNMEASURED}/"
                f"{EXIT_UNMEASURED})",
            )
        )

        # MUST_FIRE 8 -- the DOC axis has its own denominator, and zero docs
        # must not inherit the source axes' clean reading.  This is control 7's
        # argument applied one axis over: the first version of this probe
        # printed "No tracked doc asserts the retired bare form" and exited 0
        # whether it had read 23 docs or none, because only the source
        # denominator was on the wire.  The fixture below is a healthy source
        # tree (delegating, exactly the live repo's shape) paired with an EMPTY
        # doc set, and the assertion is that the pair does NOT come out clean.
        delegating = root / "delegating_for_doc_axis.py"
        delegating.write_text(
            textwrap.dedent(
                """\
                def train() -> None:
                    from transformers import Trainer

                    Trainer(model=None).train()
                """
            ),
            encoding="utf-8",
        )
        r8 = analyze_sources([delegating])
        empty_docs = scanned_docs([], exclude=Path(__file__))
        code8_empty = exit_code_for(r8, 0, len(empty_docs))
        code8_read = exit_code_for(r8, 0, 3)
        ok8 = (
            r8.status == STATUS_DELEGATES
            and len(empty_docs) == 0
            and code8_empty == EXIT_UNMEASURED
            and code8_read == EXIT_CLEAR
        )
        results.append(
            (
                "MUST_FIRE zero docs scanned is UNMEASURED, not the source axes' clean",
                ok8,
                f"status={r8.status} docs=0 exit={code8_empty} (want {EXIT_UNMEASURED}); "
                f"same source tree with docs=3 exit={code8_read} (want {EXIT_CLEAR}) -- the "
                "difference is what proves the doc denominator is load-bearing",
            )
        )

        # MUST_FIRE 9 -- the doc denominator reaches the REPO ROOT, not just
        # docs/.  Widening the scan without a control would only swap one
        # unverified scope for another: the old `ls-files -- docs` and the new
        # bare `ls-files` produce identical output on any tree whose markdown
        # all lives under docs/, so a regression to the narrow form would go
        # unnoticed here and be caught only by a false CLEAR on the live repo.
        # This fixture plants the retired phrasing in a ROOT README.md -- the
        # exact file deliverable 9 rewrites -- and fails unless the scan both
        # enumerates it and hits it.
        scope = root / "scope_repo"
        (scope / "docs").mkdir(parents=True)
        (scope / "README.md").write_text(
            "FoundationScale contains no training code.\n", encoding="utf-8"
        )
        (scope / "docs" / "nested.md").write_text("A clean nested document.\n", encoding="utf-8")
        (scope / "ignored.txt").write_text("no training code\n", encoding="utf-8")
        git_ok = _init_fixture_repo(scope)
        if git_ok:
            scanned9 = _tracked_markdown(scope)
            names9 = sorted(path.name for path in scanned9)
            hits9 = scan_doc_assertions(scanned9, exclude=Path(__file__))
            hit_names9 = [Path(hit.path).name for hit in hits9]
            ok9 = names9 == ["README.md", "nested.md"] and hit_names9 == ["README.md"]
            detail9 = (
                f"scanned={names9} (want ['README.md', 'nested.md'] -- a docs/-only "
                f"denominator returns ['nested.md'] and misses the defect); "
                f"hits={[f'{Path(hit.path).name}:{hit.line}' for hit in hits9]}"
            )
        else:
            ok9 = False
            detail9 = (
                "UNMEASURED: git is unavailable, so the scan-scope control could not run. "
                "This is not a pass -- the widened denominator stays unverified."
            )
        results.append(
            ("MUST_FIRE doc denominator includes root README.md, not only docs/", ok9, detail9)
        )

    # A control harness over zero controls is the same vacuous pass this gate
    # exists to kill; refuse it before reading any result.
    if not results:
        print("SELF-TEST DENOMINATOR: 0 of 0 controls ran -- the harness itself is broken")
        return EXIT_RED

    for name, ok, detail in results:
        state = "PASS" if ok else "FAIL"
        print(f"CONTROL {state}: {name} -- {detail}")

    total = len(results)
    failures = [name for name, ok, _ in results if not ok]
    if failures:
        report(
            "SELF-TEST DENOMINATOR",
            total,
            total - len(failures),
            "detector verdicts are worthless until every control behaves",
        )
        for failure in failures:
            print("CONTROL DEFECT: " + failure)
        return EXIT_RED
    # DERIVED, never stated.  This sentence read "3x MUST_FIRE ... 4x MUST_PASS"
    # as a literal, and adding an eighth control made the summary false while
    # every control still passed -- a claim that drifts from its own evidence
    # while reporting green is precisely the defect class this gate exists for.
    fire = sum(1 for name, _, _ in results if name.startswith("MUST_FIRE"))
    keep = sum(1 for name, _, _ in results if name.startswith("MUST_PASS"))
    if fire + keep != total:
        # Every control must declare its polarity; an unlabelled one is a
        # control in no denominator.
        print(
            f"CONTROL DEFECT: {total - fire - keep} control(s) are labelled neither "
            "MUST_FIRE nor MUST_PASS, so their polarity is unstated"
        )
        return EXIT_RED
    report(
        "SELF-TEST DENOMINATOR",
        total,
        total,
        f"{fire}x MUST_FIRE produced the reading only a real defect produces; "
        f"{keep}x MUST_PASS stayed clean over nonzero denominators",
    )
    # Final line in the SAME wording checks/countables_drift.py prints, because
    # the launcher suite parses it with one shared sed expression.  A gate that
    # invented its own phrasing here would not go red in the harness -- it would
    # fall through to the "unparseable" branch, which is fail-closed but reports
    # the wrong reason, and the next maintainer would debug the parser instead
    # of the gate.  Counts are derived above, never restated.
    print(
        f"self-test denominator: {total} of {total} controls ({fire} MUST_FIRE, {keep} MUST_PASS)"
    )
    return EXIT_CLEAR


def build_parser() -> argparse.ArgumentParser:
    parser = GateArgumentParser(
        prog="training_plane_probe",
        description="Probe: training plane = primitives axis + delegation axis; a zero "
        "on one is never publishable as a zero on both (closes #245/#223).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the seven MUST_FIRE/MUST_PASS controls in a tempdir and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON on stdout instead of the human verdict",
    )
    return parser


def _print_provenance() -> None:
    # A verdict that cannot be attributed to an interpreter is unattributable
    # (#83): the same probe under two interpreters is two different probes.
    version = ".".join(str(part) for part in sys.version_info[:3])
    print(f"INTERPRETER: {sys.executable}")
    print(f"INTERPRETER-VERSION: {version} sys.version_info={tuple(sys.version_info[:5])}")


def _payload(
    status: str,
    code: int,
    denominator: int,
    r: ProbeReport,
    hits: list[DocHit],
    docs_scanned: int,
) -> str:
    return json.dumps(
        {
            "probe": "checks/training_plane_probe.py",
            "status": status,
            "exit_code": code,
            "denominator_src_py": denominator,
            "denominator_docs_md": docs_scanned,
            "axis_a": r.axis_a,
            "axis_a_total": r.axis_a_total,
            "axis_b": r.axis_b,
            "axis_b_total": r.axis_b_total,
            "unparseable": list(r.unparseable),
            "retired_doc_assertions": [dataclasses.asdict(hit) for hit in hits],
            "interpreter": {"executable": sys.executable, "version": sys.version},
        },
        indent=2,
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not args.json:
        _print_provenance()

    try:
        root = _repo_root()
        src_files = _tracked_files(root, "src", ".py")
        doc_files = _tracked_markdown(root)
    except RefusalError as exc:
        # REFUSE, never a fallback walk: the moment "the repository" means
        # "whatever the filesystem happens to hold", the denominator is fiction.
        if args.json:
            print(json.dumps({"status": "REFUSE", "exit_code": EXIT_REFUSE, "reason": str(exc)}))
        else:
            print(f"REFUSE: {exc}")
        return EXIT_REFUSE

    denominator = len(src_files)
    probe_report = analyze_sources(src_files)
    docs_read = scanned_docs(doc_files, exclude=Path(__file__))
    doc_hits = scan_doc_assertions(doc_files, exclude=Path(__file__))
    status = probe_report.status
    code = exit_code_for(probe_report, len(doc_hits), len(docs_read))
    if code == EXIT_UNMEASURED and status != STATUS_UNMEASURED:
        # Distinguish the two roads to 95 in the status word itself, so a
        # reader of the JSON is never left inferring which axis abstained.
        status = "UNMEASURED-DOC-AXIS"

    if args.json:
        print(_payload(status, code, denominator, probe_report, doc_hits, len(docs_read)))
        return code

    for marker in AXIS_A_MARKERS:
        report(f"AXIS A {marker}", denominator, probe_report.axis_a[marker])
    for marker in AXIS_B_MARKERS:
        report(f"AXIS B {marker}", denominator, probe_report.axis_b[marker])

    # The doc scan is a SECOND axis with its OWN denominator, and it is printed
    # whether or not it found anything.  "No tracked doc asserts the retired
    # form" over zero docs and over 23 clean docs are different facts that were
    # printing the same sentence.
    report("AXIS C retired_doc_assertion", len(docs_read), len(doc_hits))

    for bad in probe_report.unparseable:
        print("FINDING unmeasured-file: " + bad)
    for hit in doc_hits:
        print(f"DOC ASSERTION {hit.path}:{hit.line}: {hit.text}")

    if probe_report.unparseable:
        print(
            f"RED over {denominator} git-tracked src/*.py: "
            f"{len(probe_report.unparseable)} file(s) could not be parsed, so those files "
            "were not measured -- clear cannot be claimed over a partial sweep"
        )
        return code

    if status == STATUS_UNMEASURED:
        # Never 0: a probe over zero files has measured nothing, and calling
        # nothing clean is the all([]) == True failure this repo exists to end.
        print(
            f"UNMEASURED over {denominator} git-tracked src/*.py under {root}: the "
            "denominator is zero files, so neither axis has a reading"
        )
        return code

    if status == "UNMEASURED-DOC-AXIS":
        # The source axes read fine; the doc axis read nothing at all.  A
        # CLEAR here would be the source axes' verdict wearing the doc axis's
        # name -- the #245 defect one level up.
        print(
            f"UNMEASURED doc axis under {root}: {len(docs_read)} git-tracked *.md were "
            "scanned, so the claim 'no tracked doc republishes the retired bare form' has an "
            f"empty denominator and is not made. Source axes did read: axis A="
            f"{probe_report.axis_a_total}, axis B={probe_report.axis_b_total} over "
            f"{denominator} src/*.py."
        )
        return code

    if status == STATUS_PRIMITIVES:
        print(
            f"VERDICT {status} over {denominator} git-tracked src/*.py: axis A is nonzero "
            f"({probe_report.axis_a_total} marker hit(s)); src/ implements training "
            "primitives. Reported, not an error."
        )
        return code

    if status == STATUS_NO_PLANE:
        print(
            f"VERDICT {status} over {denominator} git-tracked src/*.py: axis A zero AND "
            "axis B zero -- reported as a plain finding, with both axes named."
        )
        return code

    # STATUS_DELEGATES: the #223 condition.  The axis-A zero is a delegation
    # artifact; whether it is RED depends solely on what the docs still claim.
    if doc_hits:
        print(
            f"RED #223 over {denominator} git-tracked src/*.py: axis A reads 0 while axis "
            f"B reads {probe_report.axis_b_total}; the zero is a delegation artifact, and "
            f"{len(doc_hits)} tracked doc assertion(s) (printed above) still publish the "
            "retired bare-absence form.  The zero alone is not publishable as an absence: "
            "name both axes or remove the claim."
        )
    else:
        print(
            f"VERDICT {status} over {denominator} git-tracked src/*.py: src/ implements "
            f"no training primitives (axis A=0) but drives a third-party trainer (axis "
            f"B={probe_report.axis_b_total} marker hit(s), including "
            f"{probe_report.axis_b['function_scope_lazy_import']} function-scope lazy "
            f"import(s)).  None of {len(docs_read)} git-tracked *.md asserts the "
            "retired bare form; the two-axis "
            "summary above is the publishable statement."
        )
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 -- fail closed, never exit 1
        print(
            f"RED: unexpected exception escaped main(): {type(exc).__name__}: {exc} -- "
            "this probe fails closed rather than risk a traceback exit code outside "
            "the 0/5/95/96 namespace"
        )
        sys.exit(EXIT_RED)
