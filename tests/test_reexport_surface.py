"""Library/script boundary staleness control: a wrapper's re-export list must
not silently rot away from the module it re-exports.

This repository has TWO boundaries of the same shape, and they are checked by
the same derivation rather than by two hand-written suites:

* ``foundationscale.gates.adjudication`` / ``tools/live_save_gate.py``
  (T2_lib_script_boundary#0) -- the checkpoint-decision API.
* ``foundationscale.gates.probe`` / ``tools/real_checkpoint_probe.py``
  (finding #219) -- the independent-declaration machinery and its REAL-artifact
  alias control.

Each wrapper's import list is a hand-maintained enumeration, and a
hand-maintained enumeration of another module's surface is exactly the shape
that goes stale without anybody noticing: add a symbol to the library, and
``from tools.live_save_gate import new_thing`` raises ImportError for the
next caller who wants it -- with no test red anywhere in between.
``tests/test_fix44_unmeasured_refusal_record.py`` is a live caller of that
form, so the surface is load-bearing, not decorative.

The second pair is here because the first one's fix produced the second one's
defect, twice over. #219's inversion broke four of the adjudication wrapper's
98 names (an ImportError at the top of the file, which is how they were found)
and shipped a probe wrapper enumerating 18 of the library's 29 -- of which the
suite happened to reach for exactly one. Eleven names were one caller away from
the same ImportError with nothing red in between. That is the instance-to-class
move this repo owes every finding (#205): one boundary checked by hand is a
fix, two boundaries checked by one derivation is a gate.

DENOMINATOR: for each pair, every module-level name bound in the LIBRARY module
that is not a dunder, not an imported module object, and not conditionally
bound (see below). The count is asserted non-trivial, because a comparison over
an empty required set is ``all([]) is True`` -- a green that measured nothing.

The conditional-binding carve-out is DERIVED, never allowlisted.
A name bound on only one arm of a try/except exists or not depending on which
arm ran, so re-exporting it makes the wrapper's import succeed or fail
according to the environment -- observed during the T2 move as
``ImportError: cannot import name '_probe_pkg'``, green in one interpreter
and red under pytest. So this control reads each module's own AST and excludes
exactly the names that binding analysis proves are arm-dependent. Writing that
exclusion as a literal list would be excepting the detector's own residue: the
next conditional binding would be re-exported and this control would still be
green. Both libraries currently derive the EMPTY set here, which is the
stronger state and is asserted as such -- see the leg that says so.

MUST_FIRE: the comparison is run against a namespace with one required name
removed, and must report that name; and the binding analysis is run over a
planted nested-try fixture, and must find the arm-dependent names. A comparison
that cannot fail is not a control, and a derivation that returns the empty set
on every real input proves nothing about itself.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools.live_save_gate as wrapper  # noqa: E402  (path setup precedes)
import tools.real_checkpoint_probe as probe_wrapper  # noqa: E402

from foundationscale.gates import adjudication, probe  # noqa: E402

MODULE_SOURCE = Path(adjudication.__file__)

#: (label, library module, wrapper module). The label is what a failure names,
#: so it has to identify the pair without the reader opening this file.
PAIRS = (
    ("adjudication/live_save_gate", adjudication, wrapper),
    ("probe/real_checkpoint_probe", probe, probe_wrapper),
)

#: Minimum required-surface size per pair. Not one shared threshold: the two
#: libraries are an order of magnitude apart in size, and a single number large
#: enough for adjudication would be unmeetable for probe while a number small
#: enough for probe would let adjudication's surface collapse by 90% unnoticed.
#: These are floors against derivation collapse, not pinned counts -- pinning
#: the exact number would turn every legitimate addition into a failure.
_MIN_SURFACE = {
    "adjudication/live_save_gate": 50,
    "probe/real_checkpoint_probe": 20,
}

#: Per-pair named anchors, so the unconditional-surface leg is not just a
#: set-emptiness assertion. ``must_require`` names that have to appear in the
#: DERIVED required set -- the seam names the rest of the suite monkeypatches
#: through, i.e. the ones whose loss would be least visible and most damaging.
#: ``must_not_bind`` names whose REAPPEARANCE would mean the #219 inversion was
#: undone: the sentinel that flagged a failed sideways import, and the sideways
#: import itself pointing back from the library at the script.
_PINNED = {
    "adjudication/live_save_gate": {
        "must_require": {"_probe_derive_declared", "_probe_alias_control"},
        "must_not_bind": {"_PROBE_IMPORT_ERROR", "real_checkpoint_probe"},
    },
    "probe/real_checkpoint_probe": {
        "must_require": {"derive_declared", "run_alias_control"},
        "must_not_bind": {"real_checkpoint_probe", "live_save_gate"},
    },
}


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


def _conditionally_bound(source: Path = MODULE_SOURCE) -> set[str]:
    """Module-level names bound ONLY inside a try/except, in ``source``.

    The rule is deliberately structural rather than an arm-by-arm diff: a name
    the module binds unconditionally somewhere at top level is safe to
    re-export no matter what the try does to it afterwards, and a name that
    exists only because some arm ran is not.

    Both libraries in ``PAIRS`` currently return the empty set here, and that
    is the point of the rule rather than an exemption from it. Adjudication
    used to bind ``_probe_derive_declared``, ``_probe_alias_control`` and
    ``_PROBE_IMPORT_ERROR`` before a try/except ImportError ladder and rebind
    them inside it -- unconditional, so REQUIRED -- while ``_imported_*``,
    ``_probe_pkg`` and the bound exception name existed only on one arm.
    Finding #219 removed the ladder by moving the machinery into the package,
    so the surface is now unconditional by construction and the two surviving
    probe slots are plain top-level imports.

    This function is kept, and kept running over the real modules, because the
    rule outlives the ladder: the day someone reintroduces a conditional
    top-level binding, it is caught. Its own competence is proven on a planted
    fixture in ``test_must_fire_binding_analysis_sees_a_nested_try``, not on
    the real modules -- a derivation that returns the empty set proves nothing
    about itself. The fixture is passed as an ARGUMENT rather than swapped into
    a module global, so the MUST_FIRE cannot leave the analyser aimed at a
    tmp_path if it dies between the swap and the restore.

    Derived from the source, so a new conditional binding -- at any nesting
    depth -- is covered the day it is written.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tries = [n for n in tree.body if isinstance(n, ast.Try)]
    inside: set[str] = set()
    for node in tries:
        inside |= _bound_names([node])
    outside = _bound_names([n for n in tree.body if not isinstance(n, ast.Try)])
    return inside - outside


def _required_names(library: types.ModuleType) -> set[str]:
    conditional = _conditionally_bound(Path(library.__file__))
    return {
        n
        for n in dir(library)
        if not n.startswith("__")
        and n != "annotations"
        and n not in conditional
        and not isinstance(getattr(library, n), types.ModuleType)
    }


def _missing_from(namespace: object, required: set[str]) -> list[str]:
    """The comparison under test, factored out so the MUST_FIRE leg can run
    the SAME function against a doctored namespace -- a control that
    exercises a re-implementation proves nothing about the real one."""
    return sorted(n for n in required if not hasattr(namespace, n))


@pytest.mark.parametrize(("label", "library", "shim"), PAIRS, ids=[p[0] for p in PAIRS])
def test_must_pass_wrapper_reexports_every_unconditional_name(
    label: str, library: types.ModuleType, shim: types.ModuleType
) -> None:
    """MUST_PASS: the wrapper's enumeration covers the library's surface."""
    required = _required_names(library)
    floor = _MIN_SURFACE[label]
    assert len(required) >= floor, (
        f"[{label}] required surface is only {len(required)} names, below the "
        f"floor of {floor} -- the derivation collapsed, and a comparison over "
        "a near-empty set is a vacuous green"
    )
    missing = _missing_from(shim, required)
    assert not missing, (
        f"[{label}] {len(missing)} of {len(required)} names are missing from "
        f"the wrapper's re-export list: {missing} -- regenerate it"
    )


@pytest.mark.parametrize(("label", "library", "shim"), PAIRS, ids=[p[0] for p in PAIRS])
def test_must_pass_reexported_names_are_the_same_objects(
    label: str, library: types.ModuleType, shim: types.ModuleType
) -> None:
    """A re-export that rebinds to a COPY would pass a hasattr check and
    still break every monkeypatch aimed through it."""
    required = _required_names(library)
    divergent = [n for n in required if getattr(shim, n) is not getattr(library, n)]
    assert not divergent, f"[{label}] re-exported names bound to different objects: {divergent}"


@pytest.mark.parametrize(("label", "library", "shim"), PAIRS, ids=[p[0] for p in PAIRS])
def test_must_fire_a_dropped_name_is_reported(
    label: str, library: types.ModuleType, shim: types.ModuleType
) -> None:
    """MUST_FIRE: remove one required name from a stand-in namespace and the
    same comparison must name it. Observed red, not asserted red."""
    required = _required_names(library)
    victim = sorted(required)[0]
    doctored = types.SimpleNamespace(**{n: getattr(library, n) for n in required if n != victim})
    assert _missing_from(doctored, required) == [victim], label


@pytest.mark.parametrize(("label", "library", "shim"), PAIRS, ids=[p[0] for p in PAIRS])
def test_must_pass_the_surface_is_unconditional_by_construction(
    label: str, library: types.ModuleType, shim: types.ModuleType
) -> None:
    """The module binds NOTHING at top level inside a try/except, so the
    re-export surface cannot vary by environment.

    This assertion used to read the other way round. Until finding #219 the
    adjudication module reached sideways into ``tools/real_checkpoint_probe.py``
    behind a try/except ImportError ladder, and the carve-out this suite
    derives existed to keep the ladder's arm-dependent names out of the
    required set. The leg therefore asserted the carve-out was NON-empty,
    using "the analysis still finds the ladder" as a proxy for "the analysis
    still works".

    #219 inverted the dependency: the machinery moved into the package as
    ``foundationscale.gates.probe`` and the ladder is gone. Zero conditional
    bindings is the stronger property -- there is no arm to depend on, so the
    surface is identical in a checkout and in a wheel -- but a leg that merely
    stopped asserting non-emptiness would have become a vacuous green, since
    ``conditional.isdisjoint(required)`` over an empty set is true for free.
    So the emptiness is asserted as a MEASUREMENT with its own message, and
    the burden of proving the analyser can still SEE a conditional binding
    moved entirely onto ``test_must_fire_binding_analysis_sees_a_nested_try``,
    which plants the shape and runs this same derivation over it. That leg is
    what makes this zero mean something.
    """
    conditional = _conditionally_bound(Path(library.__file__))
    assert conditional == set(), (
        f"[{label}] {len(conditional)} module-level name(s) are bound only "
        f"inside a try/except: {sorted(conditional)}. Since #219 neither "
        "library has conditional top-level bindings, and a re-export surface "
        "that depends on which arm ran is environment-dependent by "
        "construction. Either bind it unconditionally or exclude it here "
        "deliberately."
    )
    required = _required_names(library)
    # Not `conditional.isdisjoint(required)` -- that is vacuously true now.
    # The substantive claim is that the seam names the rest of the suite
    # monkeypatches through are in the required set, so each wrapper is checked
    # for exactly the names most likely to break.
    pinned = _PINNED[label]
    assert pinned["must_require"] <= required, (
        f"[{label}] seam names absent from the derived surface: "
        f"{sorted(pinned['must_require'] - required)}"
    )
    revived = {n for n in pinned["must_not_bind"] if hasattr(library, n)}
    assert not revived, (
        f"[{label}] {sorted(revived)} is bound again. #219 removed the "
        "try/except ImportError ladder and its `_PROBE_IMPORT_ERROR` sentinel "
        "by moving the machinery into the package; a live sentinel, or a "
        "library that names one of the tools/ scripts, means the sideways "
        "import has been reintroduced and the package can no longer decide "
        "from a wheel."
    )


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
    # Passed as an argument. This used to swap a module global and restore it
    # in a `finally`, which works but leaves the analyser aimed at a deleted
    # tmp_path for the width of the try -- a shared-state idiom in the one test
    # whose whole job is to prove the analyser is trustworthy.
    found = _conditionally_bound(src)
    assert found == {"_imported", "_err", "_pkg"}, found
    assert "SAFE" not in found  # bound unconditionally at top level


#: (label, wrapper's own module name, the library's public entry point). The
#: entry point is named per pair because "the thing this library exists to do"
#: is not derivable -- but its MODULE is, which is the property under test.
_CLI_SPLIT = {
    "adjudication/live_save_gate": (
        "tools.live_save_gate",
        "adjudicate_checkpoint",
    ),
    "probe/real_checkpoint_probe": (
        "tools.real_checkpoint_probe",
        "derive_declared",
    ),
}


@pytest.mark.parametrize(("label", "library", "shim"), PAIRS, ids=[p[0] for p in PAIRS])
def test_must_pass_the_cli_half_stayed_in_the_wrapper(
    label: str, library: types.ModuleType, shim: types.ModuleType
) -> None:
    """The split is the point of both moves: decisions in the library, argparse
    and exit-code mapping in the script. If main() drifted into the library the
    wrapper would be an empty file pretending to be a boundary; if the entry
    point drifted into the script the library would be the empty one, which is
    exactly the state #219 found and fixed."""
    shim_module, entry_point = _CLI_SPLIT[label]
    assert shim.main.__module__ == shim_module, (
        f"[{label}] main() is defined in {shim.main.__module__}, not the "
        f"wrapper -- the CLI half escaped into the library"
    )
    assert not hasattr(library, "main"), (
        f"[{label}] the library defines main(); argparse and exit-code mapping "
        "belong to the script half of the boundary"
    )
    assert getattr(library, entry_point).__module__ == library.__name__, (
        f"[{label}] {entry_point} is defined in "
        f"{getattr(library, entry_point).__module__}, not {library.__name__} "
        "-- the decision half escaped into the script"
    )
