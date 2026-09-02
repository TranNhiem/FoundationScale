"""T2_lib_script_boundary#0 staleness control: the wrapper's re-export list
must not silently rot away from the module it re-exports.

The move put the checkpoint-decision API in
``src/foundationscale/gates/adjudication.py`` and left
``tools/live_save_gate.py`` as an argparse wrapper that re-exports every
name. That wrapper's import list is a hand-maintained enumeration, and a
hand-maintained enumeration of another module's surface is exactly the shape
that goes stale without anybody noticing: add a symbol to the library, and
``from tools.live_save_gate import new_thing`` raises ImportError for the
next caller who wants it -- with no test red anywhere in between.
``tests/test_fix44_unmeasured_refusal_record.py`` is a live caller of that
form, so the surface is load-bearing, not decorative.

DENOMINATOR: every module-level name bound in the adjudication module that
is not a dunder, not an imported module object, and not conditionally bound
(see below). The count is asserted non-trivial, because a comparison over an
empty required set is ``all([]) is True`` -- a green that measured nothing.

The conditional-binding carve-out is DERIVED, never allowlisted.
The module imports its probe helpers under try/except, and the names bound
on only one arm of that statement exist or not depending on which arm ran.
Re-exporting one of them makes the wrapper's import succeed or fail
according to the environment -- observed during the move as
``ImportError: cannot import name '_probe_pkg'``, green in one interpreter
and red under pytest. So this control reads the module's own AST and
excludes exactly the names that binding analysis proves are arm-dependent.
Writing that exclusion as a literal list of four names would be excepting
the detector's own residue: the next conditional binding would be re-exported
and this control would still be green.

MUST_FIRE: the comparison is run against a namespace with one required name
removed, and must report that name. A comparison that cannot fail is not a
control.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools.live_save_gate as wrapper  # noqa: E402  (path setup precedes)

from foundationscale.gates import adjudication  # noqa: E402

MODULE_SOURCE = Path(adjudication.__file__)


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _bound_names(nodes: list[ast.stmt]) -> set[str]:
    """Names these statements bind in the MODULE namespace.

    Recurses through compound statements -- including nested try/except, which
    is where ``_probe_pkg`` lives -- but records a def/class by name without
    descending into its body: those bindings are locals, not module names.
    """
    out: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Assign):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, _SCOPES):
            out.add(node.name)
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    out |= _bound_names([child])
            for field in ("handlers", "orelse", "finalbody", "body"):
                for child in getattr(node, field, []) or []:
                    if isinstance(child, ast.excepthandler):
                        out |= _bound_names(child.body)
                        if child.name:
                            out.add(child.name)
    return out


def _conditionally_bound() -> set[str]:
    """Module-level names bound ONLY inside a try/except.

    The rule is deliberately structural rather than an arm-by-arm diff: a name
    the module binds unconditionally somewhere at top level is safe to
    re-export no matter what the try does to it afterwards, and a name that
    exists only because some arm ran is not. The adjudication module already
    honours this -- it declares ``_probe_derive_declared``,
    ``_probe_alias_control`` and ``_PROBE_IMPORT_ERROR`` before the try and
    merely rebinds them inside it, which is why those three stay REQUIRED
    (the suite monkeypatches all three) while ``_imported_*``, ``_probe_pkg``
    and the bound exception name do not.

    Derived from the source, so a new conditional binding -- at any nesting
    depth -- is covered the day it is written.
    """
    tree = ast.parse(MODULE_SOURCE.read_text(encoding="utf-8"))
    tries = [n for n in tree.body if isinstance(n, ast.Try)]
    inside: set[str] = set()
    for node in tries:
        inside |= _bound_names([node])
    outside = _bound_names([n for n in tree.body if not isinstance(n, ast.Try)])
    return inside - outside


def _required_names() -> set[str]:
    conditional = _conditionally_bound()
    return {
        n
        for n in dir(adjudication)
        if not n.startswith("__")
        and n != "annotations"
        and n not in conditional
        and not isinstance(getattr(adjudication, n), types.ModuleType)
    }


def _missing_from(namespace: object, required: set[str]) -> list[str]:
    """The comparison under test, factored out so the MUST_FIRE leg can run
    the SAME function against a doctored namespace -- a control that
    exercises a re-implementation proves nothing about the real one."""
    return sorted(n for n in required if not hasattr(namespace, n))


def test_must_pass_wrapper_reexports_every_unconditional_name() -> None:
    """MUST_PASS: the wrapper's enumeration covers the library's surface."""
    required = _required_names()
    assert len(required) > 50, (
        f"required surface is only {len(required)} names -- the derivation "
        "collapsed, and a comparison over a near-empty set is a vacuous green"
    )
    assert not _missing_from(wrapper, required), (
        f"{len(_missing_from(wrapper, required))} of {len(required)} names are "
        f"missing from the wrapper's re-export list: "
        f"{_missing_from(wrapper, required)} -- regenerate it"
    )


def test_must_pass_reexported_names_are_the_same_objects() -> None:
    """A re-export that rebinds to a COPY would pass a hasattr check and
    still break every monkeypatch aimed through it."""
    required = _required_names()
    divergent = [
        n for n in required if getattr(wrapper, n) is not getattr(adjudication, n)
    ]
    assert not divergent, f"re-exported names bound to different objects: {divergent}"


def test_must_fire_a_dropped_name_is_reported() -> None:
    """MUST_FIRE: remove one required name from a stand-in namespace and the
    same comparison must name it. Observed red, not asserted red."""
    required = _required_names()
    victim = sorted(required)[0]
    doctored = types.SimpleNamespace(
        **{n: getattr(adjudication, n) for n in required if n != victim}
    )
    assert _missing_from(doctored, required) == [victim]


def test_must_pass_conditional_carve_out_is_derived_and_non_empty() -> None:
    """The carve-out must actually be doing work. If the AST analysis stopped
    finding the probe-import block, the exclusion would be empty, every
    arm-dependent name would re-enter the required set, and this suite would
    start failing in one environment and passing in another -- the exact
    non-determinism the move hit. An empty carve-out is that regression."""
    conditional = _conditionally_bound()
    assert conditional, (
        "no conditionally-bound module-level names found: the probe-import "
        "try/except moved or the binding analysis regressed; UNMEASURED"
    )
    required = _required_names()
    assert conditional.isdisjoint(required)
    # The three probe slots are declared BEFORE the try and only rebound
    # inside it, so they are unconditional and the suite monkeypatches all
    # three. A carve-out that swallowed them would silently stop checking the
    # names most likely to break -- the first draft of this control did
    # exactly that and still reported green.
    assert {
        "_probe_derive_declared",
        "_probe_alias_control",
        "_PROBE_IMPORT_ERROR",
    } <= required


def test_must_fire_binding_analysis_sees_a_nested_try(tmp_path) -> None:
    """MUST_FIRE for the analyser itself: ``_probe_pkg`` is bound in a try
    nested inside an except handler. An arm-shallow reader never sees it,
    re-exports it, and the wrapper's import then succeeds or fails by
    environment. Run the real derivation over that exact shape."""
    src = tmp_path / "shaped.py"
    src.write_text(
        "SAFE = 1\n"
        "try:\n"
        "    from x import a as _imported\n"
        "    SAFE = _imported\n"
        "except ImportError as _err:\n"
        "    try:\n"
        "        _pkg = None\n"
        "        SAFE = _pkg\n"
        "    except AttributeError:\n"
        "        pass\n",
        encoding="utf-8",
    )
    global MODULE_SOURCE
    real, MODULE_SOURCE = MODULE_SOURCE, src
    try:
        found = _conditionally_bound()
    finally:
        MODULE_SOURCE = real
    assert found == {"_imported", "_err", "_pkg"}, found
    assert "SAFE" not in found  # bound unconditionally at top level


def test_must_pass_the_cli_half_stayed_in_the_wrapper() -> None:
    """The split is the point of the move: decisions in the library, argparse
    and exit-code mapping in the script. If main() drifted into the library
    the wrapper would be an empty file pretending to be a boundary."""
    assert wrapper.main.__module__ == "tools.live_save_gate"
    assert not hasattr(adjudication, "main")
    assert (
        adjudication.adjudicate_checkpoint.__module__
        == "foundationscale.gates.adjudication"
    )
