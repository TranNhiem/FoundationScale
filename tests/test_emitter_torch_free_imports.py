"""Pin the emitter's torch-free import contract, measured every run or not certified.

tools/emit_run_manifest.py's own docstring states the property (listing lines
14-17): it runs on the login node, before one GPU-second is burned, "stdlib +
this package only (the verification plane must run with nothing installed;
the DCP safetensors header parse it relies on is torch-free by design)", and
the #83 addendum (:17-24) says torch is "an OPTIONAL runtime sample, never an
import requirement" -- the function-local ``import torch`` at emitter :762,
inside ``_training_stack_entries`` and guarded by try/except ImportError. One
module-scope ``import torch`` anywhere on the emitter's reachable import
chain turns a free login-node preflight into a failed launch on a paid
allocation, and the first evidence of that edit must not be the launch.

Two measurements were transcript-only until this file (a fact measured only
in a transcript is not a control, doctrine 3):

* M1, CONFIRMED with a corrected denominator: the tree has 7 real torch
  import statements across 3 files, every one function-local; the eighth
  textual grep hit, src/foundationscale/checkpoint/dcp_meta.py:116, is a
  prose comment -- exactly the #84 false-counter class, which is why the
  static half below is an ast census that denominates grammar, never text.
* M2: importing the emitter leaves torch out of sys.modules. Not decidable
  from any listing; pinned here in a FRESH child interpreter, never in the
  pytest process, whose sys.modules is decided by collection order -- pytest
  may already hold torch, or even the emitter itself, before this file runs,
  which is this estate's canonical trap.

Legs
----
STATIC -- census_import_sites ast-walks files and classes every import as
import-time (executes when the module is imported: a direct statement of the
module body, or nested in module-level try/if blocks or class bodies) vs
deferred (inside a function, or under ``if TYPE_CHECKING:``). The census is
deliberately conservative on position -- an import under a module-level
conditional is flagged import-time because the census does not predict
execution; the dynamic leg adjudicates what actually ran. The GREEN leg pins
M1 over every *.py under src/ and tools/ with denominators in the message,
and refuses both an empty file list (all([]) is True -- the failure this
estate exists to catch) and a census that sees ZERO torch sites tree-wide (a
detector that cannot find the shipped :762 sample is measuring nothing, and a
blind detector must not pass). Syntax errors and unreadable files propagate
as RED (doctrine 4: unreadable is not empty, missing is not zero).
DYNAMIC -- run_import_probe executes ``sys.executable -I -c``: the child
imports the target BY PATH, then __import__s the docstring-named torch-free
header parser foundationscale.checkpoint.dcp_meta, then reports which
forbidden-root modules sit in ITS sys.modules. -I strips environment,
PYTHONPATH, user site and cwd, so the parent asserts the child's rc FIRST,
then its post-check marker, then empty stderr -- a dead child is RED, never
"torch absent" -- and requires >=1 foundationscale module in the child (an
unexercised chain is UNMEASURED, doctrine 1).

MUST_FIRE is observed in THIS run, with causes separated and zero dependence
on whether the host has torch installed (no skips):

* static: an injected line-1 ``import torch`` on a byte-copy of the real
  emitter must be flagged at exactly line 1 and move the torch-site counter
  by exactly one (nothing executes -- the host's torch cannot be the cause);
* grammar discrimination: a comment naming 'import torch' must NOT count;
  aliased and trailing-commented module-scope imports MUST count;
* dynamic: a hoisted import of a sentinel THIS FILE writes into tmp_path
  (importable on every host) must land in the child's sys.modules after the
  chain verifiably completes -- no ImportError-shaped red can counterfeit
  that; the SAME doctored copy re-audited against ``torch`` must stay green
  (the detector fires on position, not on everything);
* symmetry (doctrine 5): a never-called FUNCTION-LOCAL import appended to
  the emitter must stay green -- a false alarm costs what a false green
  costs;
* quarantine: a copy carrying a REAL module-scope ``import torch`` must
  never read green. On a torch-less host its red is a refused chain -- right
  colour, wrong reason -- so red-via-detection is asserted additionally only
  when the chain completes; the certified firings are the legs above.

Denominators (doctrine 2) are stated in every assertion message, and each
MUST_PASS leg prints a DENOMINATOR line into the captured run record.

What this control does NOT cover (doctrine 5 -- no claim broader than its
evidence): it cannot prove the emitter RUNS on a bare login node -- the
child is the test host's interpreter under -I (an approximation of
PYTHONNOUSERSITE=1, not the verification plane), so a module-scope dependency
on some OTHER third-party package absent on the plane would read green here.
It executes the module-scope import chain only: argparse/main() paths are
never driven, and ``_training_stack_entries`` is never CALLED -- importing
torch where available is that function's documented job (:779-791), so
driving it is a designed false red. The probe matches the exact root
``torch`` and its submodules; torchvision/torchaudio are deliberately out of
scope. A module-scope ``import torch; del sys.modules["torch"]`` would evade
the dynamic half but not the static one; deliberate smuggling is out of
scope. Five short-lived child interpreters per run are a cost, not a
coverage gap.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
EMITTER_PATH = TOOLS_ROOT / "emit_run_manifest.py"
TORCH_ROOT = "torch"
SENTINEL_NAME = "fs_torch_position_sentinel"
FUNCTION_LOCAL_SENTINEL = "fs_torch_function_local_sentinel"
PROBE_TIMEOUT_SECONDS = 120

# Chain modules the child imports after exec_module: the torch-free-by-design
# DCP header parser the emitter's own docstring names (listing lines 11-17).
CHAIN_MODULES: tuple[str, ...] = ("foundationscale.checkpoint.dcp_meta",)

# Files the static walk must prove it covered: the emitter itself, the
# module-scope chain target quoted from its import block (lines 95-118 hold
# `from foundationscale.provenance.manifest import ...`), and the header
# parser named above. Package __init__ files are NOT required: namespace
# packages are legal, and missing-is-not-zero must not red a healthy tree.
MUST_CENSUS_FILES: tuple[Path, ...] = (
    EMITTER_PATH,
    SRC_ROOT / "foundationscale" / "provenance" / "manifest.py",
    SRC_ROOT / "foundationscale" / "checkpoint" / "dcp_meta.py",
)

# Verdict line marker. Keep in sync with the literal inside _CHILD_PROGRAM;
# the child is self-contained and cannot import this constant.
_MARKER = "FS_TORCHFREE_RESULT "

# The whole dynamic half, executed in a fresh interpreter: import the target
# by path, import the chain modules, then audit which roots landed in ITS
# sys.modules. Import failures are reported in-band with rc 3 -- a dead child
# is never mistaken for a clean one, because the parent asserts rc, marker
# and stderr before reading any verdict.
_CHILD_PROGRAM = """
import importlib.util
import json
import os
import sys

target = sys.argv[1]
extra_paths = sys.argv[2]
forbidden_root = sys.argv[3]
chain = sys.argv[4:]
for entry in extra_paths.split(os.pathsep):
    if entry and entry not in sys.path:
        sys.path.insert(0, entry)
spec = importlib.util.spec_from_file_location("fs_module_under_probe", target)
if spec is None or spec.loader is None:
    print("FS_TORCHFREE_RESULT " + json.dumps({"stage": "spec", "error": target}))
    raise SystemExit(3)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
try:
    spec.loader.exec_module(module)
    for name in chain:
        __import__(name)
except BaseException as exc:
    blob = {"stage": "exec", "error": type(exc).__name__ + ": " + str(exc)}
    print("FS_TORCHFREE_RESULT " + json.dumps(blob))
    raise SystemExit(3) from exc
hits = sorted(
    name for name in sys.modules
    if name == forbidden_root or name.startswith(forbidden_root + ".")
)
fs_modules = sum(
    1
    for name in sys.modules
    if name == "foundationscale" or name.startswith("foundationscale.")
)
blob = {
    "stage": "checked",
    "hits": hits,
    "foundationscale_modules": fs_modules,
    "modules_loaded": len(sys.modules),
}
print("FS_TORCHFREE_RESULT " + json.dumps(blob))
"""


class CensusRefusedError(RuntimeError):
    """Raised when a census is asked to measure zero inputs.

    An import census over zero files must refuse: ``all([])`` is True, and a
    detector that passes over nothing is the failure this estate exists to
    catch (doctrine 1). Missing source roots land here too: missing is not
    zero (doctrine 4).
    """


@dataclasses.dataclass(frozen=True)
class ImportSite:
    """One imported module name and where it sits relative to import time."""

    path: Path
    lineno: int
    module: str
    kind: str  # "import" or "from"
    level: int  # relative-import dots; 0 means absolute
    import_time: bool  # executes when the module itself is imported

    @property
    def is_torch(self) -> bool:
        """True only for an ABSOLUTE import of the ``torch`` root or below."""
        root = TORCH_ROOT
        return (
            self.level == 0
            and (self.module == root or self.module.startswith(root + "."))
        )

    def render(self) -> str:
        """One-line evidence for assertion messages."""
        if self.import_time:
            where = "module scope (executes on import)"
        else:
            where = "deferred (function-local or TYPE_CHECKING)"
        return f"{self.path}:{self.lineno} [{self.kind} {self.module}] {where}"


@dataclasses.dataclass(frozen=True)
class ImportCensus:
    """What the static half examined: denominators first, verdicts derived."""

    files: int
    sites: tuple[ImportSite, ...]

    def torch_sites(self) -> list[ImportSite]:
        """Every torch import the census saw, at any position."""
        return [site for site in self.sites if site.is_torch]

    def import_time_torch_sites(self) -> list[ImportSite]:
        """The verdict input: torch imports that run at module import."""
        return [site for site in self.torch_sites() if site.import_time]


def _guards_type_checking(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:`` and ``if typing.TYPE_CHECKING:``."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _census_one_file(path: Path) -> list[ImportSite]:
    """AST-walk one file into import sites -- position over text.

    ``read_text`` OSError and ``ast.parse`` SyntaxError propagate on purpose:
    an unreadable or unparseable file is RED, not "zero import sites"
    (doctrine 4). Only function bodies and ``if TYPE_CHECKING:`` blocks are
    deferred; every other statement position -- including module-level
    try/if blocks and class bodies -- executes at import time and is classed
    import-time. A regex cannot do this job: it denominates comments as
    import sites (the dcp_meta.py:116 prose hit, the #84 false-counter
    class).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites: list[ImportSite] = []

    def visit(node: ast.AST, *, deferred: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            deferred = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                sites.append(
                    ImportSite(path, node.lineno, alias.name, "import", 0,
                               not deferred)
                )
            return
        elif isinstance(node, ast.ImportFrom):
            sites.append(
                ImportSite(path, node.lineno, node.module or "", "from",
                           node.level, not deferred)
            )
            return
        elif isinstance(node, ast.If) and _guards_type_checking(node.test):
            for stmt in node.body:
                visit(stmt, deferred=True)
            for stmt in node.orelse:
                visit(stmt, deferred=deferred)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, deferred=deferred)

    visit(tree, deferred=False)
    return sites


def census_import_sites(files: Iterable[Path]) -> ImportCensus:
    """Census every import statement in ``files``; REFUSE a zero-file list."""
    ordered = sorted({Path(file) for file in files})
    if not ordered:
        raise CensusRefusedError(
            "import census over zero files is UNMEASURED, never PASS "
            "(all([]) is True; refusing that shape is the estate's job)"
        )
    sites: list[ImportSite] = []
    for path in ordered:
        sites.extend(_census_one_file(path))
    return ImportCensus(files=len(ordered), sites=tuple(sites))


def _tree_python_files() -> list[Path]:
    """Every ``*.py`` under src/ and tools/; a missing root is RED."""
    found: list[Path] = []
    for root in (SRC_ROOT, TOOLS_ROOT):
        if not root.is_dir():
            raise CensusRefusedError(
                f"expected source root missing: {root} -- missing is not zero"
            )
        found.extend(root.rglob("*.py"))
    return found


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """One child-interpreter run: rc, both streams, and its verdict if any."""

    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any] | None


def run_import_probe(
    target: Path,
    forbidden_root: str,
    extra_paths: Iterable[Path] = (),
    chain: Iterable[str] = CHAIN_MODULES,
) -> ProbeResult:
    """Import ``target`` in a FRESH ``-I`` interpreter, then audit the root.

    Nothing pytest (or a developer shell) loaded can leak in: ``-I`` strips
    the environment, PYTHONPATH, user site and cwd, so the child receives
    every sys.path entry it needs explicitly via argv. It never "decides
    green by dying": import failures come back in-band with rc 3, and
    ``_checked_payload`` turns a nonzero rc, a missing marker, or a dirty
    stderr into RED.
    """
    joined = os.pathsep.join(str(path) for path in extra_paths)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD_PROGRAM,
         str(target), joined, forbidden_root, *chain],
        capture_output=True,
        text=True,
        check=False,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    payload: dict[str, Any] | None = None
    for line in proc.stdout.splitlines():
        if line.startswith(_MARKER):
            try:
                payload = json.loads(line[len(_MARKER):])
            except json.JSONDecodeError:
                payload = None
    return ProbeResult(proc.returncode, proc.stdout, proc.stderr, payload)


def _checked_payload(result: ProbeResult, *, context: str) -> dict[str, Any]:
    """Require a child that lived, reached its verdict, and stayed silent."""
    assert result.returncode == 0, (
        f"{context}: probe child exited rc={result.returncode}; a child "
        "that dies before checking is RED, never 'target absent'.\n"
        f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    assert (
        result.payload is not None and result.payload.get("stage") == "checked"
    ), (
        f"{context}: probe child exited 0 but printed no parseable verdict; "
        "the check over its sys.modules never ran, and a check that did not "
        f"run is UNMEASURED, not green (doctrine 1).\nstdout:\n{result.stdout}"
    )
    assert result.stderr == "", (
        f"{context}: probe child wrote {len(result.stderr)} bytes to stderr; "
        "verification-plane silence is part of the contract.\n"
        f"stderr:\n{result.stderr}"
    )
    return result.payload


def _write_module(tmp_path: Path, name: str) -> Path:
    """A bare module the test controls -- importable on every host."""
    path = tmp_path / f"{name}.py"
    path.write_text(
        f'"""Controlled probe module {name}; a docstring and nothing else."""\n',
        encoding="utf-8",
    )
    return path


def _doctored_emitter(tmp_path: Path, injected_import: str) -> tuple[Path, int]:
    """The shipped emitter with one import line injected at LEGAL module scope.

    The probe line lands immediately after the last ``from __future__``
    statement (and the module docstring, if any): still module scope,
    where the detectors must see it, but at a position Python can run.
    Returns the doctored path and the 1-based number of the injected line,
    so callers pin position against a MEASURED value, never against a
    constant the emitter's shape can silently move. The text is compiled
    before it is written; the helper fails closed rather than hand back a
    file Python cannot run -- an illegal injection point is a broken
    control masquerading as detector evidence, never a measurement.
    """
    import ast  # local: the parser is needed only by this measurement

    source = EMITTER_PATH.read_text(encoding="utf-8")
    body = ast.parse(source).body
    index = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        index = 1  # the first statement is the module docstring
    cut = body[0].end_lineno if index else 0
    for node in body[index:]:
        if not (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ):
            break
        cut = node.end_lineno or node.lineno
    lines = source.splitlines(keepends=True)
    lines.insert(cut, f"{injected_import}\n")
    text = "".join(lines)
    injected_lineno = cut + 1
    doctored = tmp_path / EMITTER_PATH.name
    error = None
    try:
        compile(text, str(doctored), "exec")
    except SyntaxError as exc:
        error = exc
    assert error is None, (
        "fixture injection point was illegal: the doctored emitter does "
        f"not compile ({error}); the injected import must land after the "
        "module docstring and every `from __future__` import -- a child "
        "that dies before checking is a broken control, never evidence "
        "of absence"
    )
    doctored.write_text(text, encoding="utf-8")
    return doctored, injected_lineno


def test_tree_census_finds_torch_only_off_the_import_path() -> None:
    """GREEN (static): zero import-time torch imports, denominators stated.

    Pins M1 with its corrected denominator. The walk must provably cover the
    files the property depends on (missing is not zero), and the census must
    provably see torch sites at all: the shipped emitter carries the
    function-local sample at :762, so a census reporting zero torch sites is
    blind, not clean -- and a blind detector must not pass (doctrine 1).
    """
    missing = [str(path) for path in MUST_CENSUS_FILES if not path.is_file()]
    assert not missing, (
        "fail closed: the census cannot vouch for files that do not exist "
        f"(missing is not zero, doctrine 4): {missing}"
    )
    walk = _tree_python_files()
    out_of_walk = [str(path) for path in MUST_CENSUS_FILES if path not in walk]
    assert not out_of_walk, (
        f"the census walk lost files it must cover: {out_of_walk}"
    )
    census = census_import_sites(walk)
    torch_sites = census.torch_sites()
    assert torch_sites, (
        f"examined {census.files} files / {len(census.sites)} import sites "
        "and saw ZERO torch imports; the emitter ships a function-local "
        "torch sample by design (:762), so a census seeing none is measuring "
        "nothing, and a blind detector must not pass"
    )
    bad = census.import_time_torch_sites()
    detail = "\n".join(f"  {site.render()}" for site in bad)
    assert not bad, (
        f"{len(bad)} import-time torch import(s) found among "
        f"{len(torch_sites)} torch sites (of {len(census.sites)} import "
        f"sites across {census.files} files under src/ and tools/):\n"
        f"{detail}"
    )
    print(
        f"DENOMINATOR static leg: {census.files} files, "
        f"{len(census.sites)} import sites, {len(torch_sites)} torch-rooted, "
        "0 at import time"
    )


def test_census_refuses_to_measure_zero_files() -> None:
    """FAIL CLOSED, observed: an empty file list is refused, never passed."""
    with pytest.raises(CensusRefusedError, match="zero files"):
        census_import_sites([])


def test_static_census_counts_grammar_not_text(tmp_path: Path) -> None:
    """DETECTOR DISCRIMINATION, observed: the census denominates grammar.

    Four shapes demanded of any honest census in one measured pass: a comment
    naming 'import torch' must NOT be counted (the real prose hit at
    dcp_meta.py:116); a module-scope ``import torch as t`` with a trailing
    comment MUST be counted; imports under ``if TYPE_CHECKING:`` and inside a
    function count as sites but never as import-time offenders. A census
    that cannot exhibit both behaviours on demand has no control.
    """
    lines = [
        "# a comment naming import torch must NOT be counted",
        "import torch as t  # aliased + trailing comment: MUST count",
        "",
        "if TYPE_CHECKING:",
        "    import torch",
        "",
        "",
        "def _never_called() -> None:",
        "    import torch",
    ]
    probe = tmp_path / "grammar_probe.py"
    probe.write_text("\n".join(lines) + "\n", encoding="utf-8")
    census = census_import_sites([probe])
    torch_sites = census.torch_sites()
    offenders = census.import_time_torch_sites()
    assert len(torch_sites) == 3, (
        "expected exactly 3 torch import statements (aliased module-scope, "
        f"TYPE_CHECKING-guarded, function-local); saw {len(torch_sites)}: "
        + "; ".join(site.render() for site in torch_sites)
    )
    assert [site.lineno for site in offenders] == [2], (
        "the only import-time offender must be the aliased module-scope "
        "import on line 2; flagged "
        f"{[site.lineno for site in offenders]}"
    )


def test_static_detector_fires_on_module_scope_torch(tmp_path: Path) -> None:
    """MUST_FIRE (static), observed: torch itself, by position, no wheel.

    An injected module-scope ``import torch`` on a byte-copy of the real
    emitter -- LEGAL module scope, after the docstring and every
    ``from __future__`` import, so the copy also compiles and runs --
    must be flagged at EXACTLY the measured line the fixture reports
    having injected at, and the torch-site counter must move
    by EXACTLY one -- a counter that cannot see one added import cannot
    certify zero. Nothing executes, so whether this host has torch installed
    cannot be the cause. The same census must clear the shipped emitter
    while seeing its function-local sample (:762): exactly today's shape.
    """
    clean = census_import_sites([EMITTER_PATH])
    clean_torch = clean.torch_sites()
    clean_bad = clean.import_time_torch_sites()
    assert clean_torch and not clean_bad, (
        "precondition: the shipped emitter carries a function-local torch "
        "sample (:762) and no import-time torch; saw "
        f"{len(clean_torch)} torch site(s), import-time: "
        f"{[site.render() for site in clean_bad]}"
    )
    doctored, injected_lineno = _doctored_emitter(
        tmp_path, "import torch  # MUST_FIRE probe: hoisted to module scope"
    )
    fired = census_import_sites([doctored])
    bad = fired.import_time_torch_sites()
    assert [site.lineno for site in bad] == [injected_lineno], (
        "STATIC MUST_FIRE: the census that clears the real emitter must "
        "flag the injected module-scope import at exactly the measured "
        f"injection line [{injected_lineno}]; flagged "
        f"{[site.lineno for site in bad]} ({fired.files} file, "
        f"{len(fired.sites)} import sites examined) -- if empty, the "
        "control is dead"
    )
    assert len(fired.torch_sites()) == len(clean_torch) + 1, (
        f"the census saw {len(clean_torch)} torch site(s) in the real "
        "emitter and must see exactly one more in the doctored copy; saw "
        f"{len(fired.torch_sites())} -- a counter that cannot see one added "
        "import cannot certify zero"
    )


def test_fresh_interpreter_import_leaves_torch_absent() -> None:
    """GREEN (dynamic): the emitter's chain, exercised in a fresh child,
    leaves torch out of the child's sys.modules (M2 pinned).

    rc is asserted first, then the post-check verdict, then stderr: a dead
    child is RED, never 'torch absent'. The child must also prove the chain
    ran -- >=1 foundationscale module loaded, since the emitter's
    module-scope block (lines 95-118) imports
    foundationscale.provenance.manifest -- else the verdict is UNMEASURED.
    """
    assert EMITTER_PATH.is_file(), (
        f"emitter missing at {EMITTER_PATH}; missing is not zero"
    )
    result = run_import_probe(EMITTER_PATH, TORCH_ROOT, [SRC_ROOT])
    payload = _checked_payload(result, context="torch-free import MUST_PASS")
    assert payload["foundationscale_modules"] >= 1, (
        "the child loaded 0 foundationscale modules, so the emitter's "
        "module-scope import chain never ran; 'torch absent' over an "
        "unexercised chain is UNMEASURED (doctrine 1)"
    )
    assert payload["hits"] == [], (
        "importing the emitter's chain pulled torch into a fresh "
        f"interpreter: {payload['hits']} present; "
        f"{payload['modules_loaded']} modules loaded after exec of "
        f"{EMITTER_PATH.name} by path plus chain {list(CHAIN_MODULES)}"
    )
    print(
        "DENOMINATOR runtime leg: emitter by path + "
        f"{len(CHAIN_MODULES)} chain module(s); 0/"
        f"{payload['modules_loaded']} loaded modules torch-rooted; "
        "foundationscale modules in child: "
        f"{payload['foundationscale_modules']}"
    )


def test_dynamic_detector_fires_on_controlled_module_scope_import(
    tmp_path: Path,
) -> None:
    """MUST_FIRE (dynamic), torch-independent by construction.

    Red here can mean only 'module-scope import observed in the child's
    sys.modules': the injected name is a sentinel this test writes into
    tmp_path, importable on every host, and the parent first proves the
    chain completed -- no ImportError-shaped red can stand in for the
    detector. The SAME doctored copy is then re-audited against ``torch``
    and must stay GREEN: the detector fires on position, not on everything,
    and the child did not merely die.
    """
    _write_module(tmp_path, SENTINEL_NAME)
    doctored, _ = _doctored_emitter(
        tmp_path, f"import {SENTINEL_NAME}  # MUST_FIRE: module scope"
    )
    fired = run_import_probe(doctored, SENTINEL_NAME, [SRC_ROOT, tmp_path])
    payload = _checked_payload(fired, context="dynamic MUST_FIRE")
    assert payload["hits"] == [SENTINEL_NAME], (
        "DYNAMIC MUST_FIRE: the detector that clears the real emitter failed "
        "to observe a module-scope import it can always see; hits="
        f"{payload['hits']} across {payload['modules_loaded']} modules "
        "loaded in the child"
    )
    audit = run_import_probe(doctored, TORCH_ROOT, [SRC_ROOT, tmp_path])
    audit_payload = _checked_payload(audit, context="torch colour separation")
    assert audit_payload["hits"] == [], (
        "the doctored copy pulled torch into the child: "
        f"{audit_payload['hits']} (injected sentinel: {SENTINEL_NAME})"
    )


def test_function_local_import_never_executed_stays_green(
    tmp_path: Path,
) -> None:
    """SYMMETRY (doctrine 5): text presence is not execution.

    A never-called FUNCTION-LOCAL import -- exactly the shape of the shipped
    sample at :762 -- appended to a copy of the real emitter must NOT fire
    the dynamic detector: an import that never executes never lands in
    sys.modules. A false alarm costs what a false green costs.
    """
    _write_module(tmp_path, FUNCTION_LOCAL_SENTINEL)
    doctored = tmp_path / EMITTER_PATH.name
    doctored.write_text(
        EMITTER_PATH.read_text(encoding="utf-8")
        + "\n\n"
        + "def _fs_torchfree_never_called() -> None:\n"
        + f"    import {FUNCTION_LOCAL_SENTINEL}  # local: must NOT fire\n",
        encoding="utf-8",
    )
    result = run_import_probe(
        doctored, FUNCTION_LOCAL_SENTINEL, [SRC_ROOT, tmp_path]
    )
    payload = _checked_payload(result, context="function-local symmetry")
    assert payload["hits"] == [], (
        "the detector fired on a FUNCTION-LOCAL import that never executed; "
        "it must read sys.modules after exec (import position), not "
        f"statement text: {payload['hits']}"
    )


def test_doctored_torch_copy_never_reads_green(tmp_path: Path) -> None:
    """QUARANTINE: the estate's worst edit must never pass this detector.

    A byte-copy of the real emitter carrying a REAL module-scope ``import
    torch`` must never read green. Its red is deliberately NOT the certified
    firing (the legs above are): on a torch-less host the child dies on
    ImportError -- right colour, wrong reason, a red that measures the host.
    So the never-green content is asserted unconditionally, and on hosts
    where the chain completes (torch importable here) the red must
    additionally be DETECTION: 'torch' observed in the child's sys.modules,
    with nothing else able to have fired.
    """
    doctored, _ = _doctored_emitter(
        tmp_path, "import torch  # hoisted to module scope: the worst edit"
    )
    result = run_import_probe(doctored, TORCH_ROOT, [SRC_ROOT])
    completed = (
        result.returncode == 0
        and result.payload is not None
        and result.payload.get("stage") == "checked"
    )
    green = completed and result.stderr == "" and not result.payload["hits"]
    assert not green, (
        "a module-scope `import torch` on the emitter's chain read GREEN -- "
        "the detector is a no-op over this estate's worst edit "
        f"(rc={result.returncode}, payload={result.payload})"
    )
    if completed:
        assert "torch" in result.payload["hits"], (
            "the chain completed yet the red is not 'torch observed in the "
            "child' -- so what fired? "
            f"hits={result.payload['hits']}, stderr={result.stderr!r}"
        )
    # Else: torch is absent here and the red is a refused chain -- right
    # colour only by accident. That is why the certified firings rest on the
    # controlled-name and static-position legs above, never on this one.
