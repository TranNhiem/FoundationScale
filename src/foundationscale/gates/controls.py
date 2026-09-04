"""Executable self-test for the gate suite.

Why this exists
---------------
The audit's sharpest incident was not in training code but in the tool written to
*detect* silent success: it reported ``all_identity: True`` on a corrupt artifact
because the comparison set was empty and ``all([])`` is ``True``. Every gate in this
repository is a detector of exactly that shape, so each one must carry fixtures of
BOTH kinds — :attr:`~foundationscale.gates.core.ControlKind.MUST_FIRE` inputs that
it provably blocks on, and :attr:`~foundationscale.gates.core.ControlKind.MUST_PASS`
known-healthy inputs it provably does NOT block on — and those fixtures must
actually be run, in CI, on every change. A gate whose controls are never executed
rots into a no-op at the same speed as the code it watches; a gate that blocks on
everything rots in the opposite direction, and gates that block on everything are
the ones that get disabled. Declaring only one kind is itself a CI failure, not an
omission to be generous about: the missing half is behaviour that was verified
zero times, and this runner's whole purpose is to refuse success over zero work.

This module is the runner. It exists to close two holes that ``verify_controls``
alone cannot see:

1. **The registry.** Gates self-register at import time. If the gate package is
   never imported, the registry is empty and there is nothing to verify —
   ``verify_controls`` itself now refuses a zero-gate target set, and
   ``main()`` goes further on the likely cause: it populates the registry
   first, prints its own "registry is empty" line, and exits 1.
2. **The import boundary.** A gate module that raises at import time is a gate
   that never registered and therefore never verified anything. That is recorded
   as a failure here rather than allowing a partial registry to look whole.
3. **The walk boundary.** Registration is import-triggered, so the sweep can
   only ever certify gates in modules it IMPORTED — and for weeks this walk
   descended into ``foundationscale.gates`` alone while ``WeightParityGate``
   registered from the sibling package ``foundationscale.verify.parity``. Its
   three controls were certified over an empty set and the run printed "all
   clear" — the prophecy two paragraphs up, fulfilled one package boundary
   out. The walk now covers the declared gate-package roots, reconciles the
   registry it ends up with against the modules it attempted (a registered
   gate from an unreached module is a named failure, never a quiet bonus),
   and a census of first-party sibling packages refuses silence about any
   package nobody has classified gate-bearing or gate-free. The run also
   prints its certification denominator, because "all controls held" is a
   claim about a count and the count must travel with the claim.

Wired into CI as the ``controls`` job of ``.github/workflows/ci.yml`` and exposed
as the ``foundationscale-controls`` console script. Exits 1 on any failure; exit 0
means every gate was shown, just now, to be capable of blocking a defective input
and of passing a healthy one.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Callable, Collection

from foundationscale.gates.core import REGISTRY, Gate, GateRegistry, verify_controls

_THIS_MODULE = "foundationscale.gates.controls"

_GATE_PACKAGES: tuple[str, ...] = ("foundationscale.gates", "foundationscale.verify")
"""Every first-party package whose modules may define and register gates.

Registration is a side effect of import, so a gate-bearing package missing
from this list contributes NOTHING to the registry — the F1 blind spot, in
which ``WeightParityGate``'s three controls spent weeks certified over an
empty set one sibling package away. The list is deliberately explicit rather
than a walk of the whole ``foundationscale`` tree: torch-backed and otherwise
heavy modules should import into this CI process only when they bear gates,
and a blanket walk would couple this job's exit code to every importable
module in the tree. The blind spot explicitness reopens — the NEXT package
someone forgets to list — is closed structurally instead:
:func:`_unclassified_package_findings` names every first-party sibling on
nobody's list, and :func:`_uncertified_provenance_findings` refuses to
certify any registered gate whose module the walk never attempted, so a
forgotten entry fails loudly the night it lands instead of shipping months
of nine-out-of-ten.
"""

_KNOWN_GATELESS_PACKAGES: frozenset[str] = frozenset(
    {
        "foundationscale.checkpoint",  # torch-backed readers; gates import it lazily
        "foundationscale.provenance",  # run-manifest types; declares no gates
        # The training entry CONSUMES the registry (FoundationScaleSaveGate wires
        # REGISTRY into Trainer's on_save); it defines no Gate subclass and calls
        # no register. Verified by AST over the package, not by reading: zero
        # ClassDef inherits Gate, zero call site is register/add_gate. Were that to
        # change, the provenance reconciliation names it -- the new gate's defining
        # module would not be among the walk's attempted set.
        "foundationscale.train",
        # The model-adapter registry is a CLASSIFIER, not a gate: it answers
        # "dense or MoE, on what evidence" and refuses on malformed config
        # facts, but it registers nothing and blocks nothing -- the emitter and
        # the gates decide what to do with its answer. Its own registry
        # (register_adapter) is a distinct namespace from the gate REGISTRY and
        # is deliberately not plumbed into it. AST-verified over the package:
        # zero ClassDef inherits Gate, zero call site is register/add_gate.
        "foundationscale.models",
    }
)
"""First-party packages affirmatively classified — by a human, in review — as NOT
bearing gates.

Membership is a claim this harness can be wrong about, which is why it is
data in the source and not convention: adding a package here asserts "no
module under this root registers a gate", and the provenance reconciliation
will name the lie loudly if a gate ever appears in one of them and gets
imported through some other channel — its defining module will not be among
the walk's attempted set.
"""

_NO_CONTEXT_TYPE_MARKER = "<no context_type declared — legacy untyped dispatch>"
"""The listing's explicit stand-in when a gate declares no ``context_type``.

The gate listing is the one surface where a human reads the whole gate
population's dispatch declarations at once, and the doctrine is that coverage
is a returned fact, never inferred from the absence of a complaint. Printing
nothing where a declaration is absent would make a forgotten ``context_type``
visually identical to a deliberately untyped gate — silence-as-evidence,
inside the doctrine's own enforcement tool.
"""


def _context_type_label(gate: Gate) -> str:
    """Render one gate's ``context_type`` declaration for the listing.

    The declaration is a ``ClassVar``, so reading it never touches gate-author
    runtime code (``controls()``, by contrast, can raise). The listing can
    therefore describe every registered gate, including the ones whose controls
    it could not run — precisely the gates the listing most needs to show.
    """
    if gate.context_type is None:
        return _NO_CONTEXT_TYPE_MARKER
    return gate.context_type.__name__


def _walk_gate_packages() -> tuple[list[str], list[str], list[str]]:
    """Import every module under each root in :data:`_GATE_PACKAGES`.

    Registration is a side effect of import (the ``@register`` decorator), so a
    module that is never imported contributes nothing to the registry — and a
    controls run over what remains would be a false "all clear". For weeks the
    roots were ``foundationscale.gates`` alone and that false all-clear
    shipped: ``WeightParityGate`` registered from
    ``foundationscale/verify/parity.py``, one sibling package past the
    boundary, exactly the failure this function's own docstring described.
    Modules are imported individually and import exceptions are collected
    rather than raised: a gate module that cannot import on this box is itself
    a finding, and the run must continue so the report shows *all* broken
    modules, not just the first. A root whose own import fails, or that proves
    not to be a package at all, is recorded the same way — every gate beneath
    it is then provably unreached.

    Returns:
        A triple ``(imported, failed, errors)``: module names successfully
        imported, module names whose import was ATTEMPTED and raised (kept as
        data rather than folded into error prose, because the provenance
        reconciliation needs the attempted set and a framework built against
        prose-sniffing does not parse its own sentences to get it), and
        human-readable import-failure strings.

    What this cannot see, stated so it is never mistaken for coverage: a gate
    registered lazily inside a function nothing calls at import time; a gate
    in a third-party package. The first is invisible to any in-process
    mechanism; the second is caught downstream as an uncertified-provenance
    finding -- named and refused, never silently certified and never silently
    skipped.
    """
    imported: list[str] = []
    failed: list[str] = []
    errors: list[str] = []

    def _record_walk_failure(name: str) -> None:
        """Record a package the walk itself could not import, exactly once.

        ``pkgutil.walk_packages`` re-imports every yielded package internally,
        to descend into it; with ``onerror`` unset, a non-ImportError raised
        there escapes :func:`main` as a traceback and the collected report is
        never printed. The loop below always attempts (and, on failure,
        records) its own import of the same package *first*, so a raising
        package reaches this callback already recorded — skip it rather than
        double-count. A raising *module* never reaches this callback at all:
        pkgutil does not re-import non-packages. That asymmetry is why dedup
        by name, not a blanket record, is the correct shape here.
        """
        if not any(error.startswith(f"{name}:") for error in errors):
            errors.append(
                f"{name}: import raised while pkgutil descended into the package — "
                f"any gates defined there never registered"
            )

    for root_name in _GATE_PACKAGES:
        try:
            root = importlib.import_module(root_name)
        except Exception as exc:  # noqa: BLE001 — an unimportable root unreaches every gate under it
            errors.append(
                f"{root_name}: gate-package ROOT import raised {type(exc).__name__}: "
                f"{exc} — no module beneath it was walked, so any gates defined "
                f"there never registered and their controls certified nothing"
            )
            continue
        root_path = getattr(root, "__path__", None)
        if root_path is None:
            errors.append(
                f"{root_name}: gate-package root is a module, not a package — "
                f"there is nothing to walk into, and any gates it defines are "
                f"already registered by the import above"
            )
            continue
        for info in pkgutil.walk_packages(
            root_path, prefix=f"{root_name}.", onerror=_record_walk_failure
        ):
            if info.name == _THIS_MODULE:
                continue
            try:
                importlib.import_module(info.name)
            except Exception as exc:  # noqa: BLE001 — a gate that cannot import did not run
                errors.append(
                    f"{info.name}: import raised {type(exc).__name__}: {exc} — "
                    f"any gates defined there never registered"
                )
                failed.append(info.name)
            else:
                imported.append(info.name)
    return sorted(imported), sorted(failed), errors


def _import_gate_modules() -> tuple[list[str], list[str]]:
    """Import every gate-package module so gates register, returning
    ``(imported, errors)``.

    The old docstring called this a compatibility shim for callers written
    against the two-tuple contract; measured against the tree, there are no
    such callers — a claim about its consumers broader than its evidence,
    doctrine (5) applied to a docstring. Its real, present-day callers are
    this suite's white-box fixtures (the harness suite's ``populated_registry``
    and the fix24 F1 walk test), which want exactly this curated view: the
    walk's three-tuple with the failed-attempt names elided, because they
    assert on nothing those names would refine. :func:`main` still consumes
    the worker's full triple (as its ``walk`` default): the failed-attempt
    names are load-bearing for the provenance reconciliation, and dropping
    them here would re-create "the walk found no errors" / "the walk found
    nothing" ambiguity in a new shape. When the day comes that no caller
    remains, delete this function and nothing else breaks — that is the
    property a callerless compatibility shim never had.
    """
    imported, _failed, errors = _walk_gate_packages()
    return imported, errors


def _uncertified_provenance_findings(
    registry: GateRegistry, attempted: Collection[str]
) -> list[str]:
    """Name every registered gate whose defining module the walk never attempted.

    The registry can only hold what some import executed — but imports arrive
    from more channels than this walk (an operator's sitecustomize, a gate
    module importing a sibling for one helper, a future plugin). Before F1,
    such a gate was silently certified along with everything it sat beside;
    after F1 that silence is recognisably the same defect one boundary down.
    The rule: this harness vouches only for gates it deliberately reached. A
    registered gate from an unattempted module is a NAMED failure, so the
    decision to widen the walk — or to verify that gate's controls in the job
    that imports it — is made by a human, in the open, rather than inherited
    by accident. The registry is a parameter (not read from the module
    global) precisely so this check is testable without polluting the
    process-wide registry.
    """
    findings: list[str] = []
    for gate in registry:
        module = type(gate).__module__
        if module not in attempted:
            findings.append(
                f"{gate.id}: registered from {module}, a module the gate-module "
                f"walk never reached — this harness will not certify what it did "
                f"not deliberately import. Add the owning package to "
                f"_GATE_PACKAGES, or verify this gate's controls in the job that "
                f"imports it"
            )
    return sorted(findings)


def _unclassified_package_findings() -> list[str]:
    """Name every top-level first-party sibling package on nobody's list.

    ``pkgutil.iter_modules`` LISTS package names without executing them, so
    this census has no import side effects — it is the cheap structural
    tripwire for the F1 shape: a brand-new package full of gates, added to
    neither :data:`_GATE_PACKAGES` nor :data:`_KNOWN_GATELESS_PACKAGES`, fails
    HERE, loudly, on the night it lands, instead of shipping months of
    controls runs that never knew it existed. One of two one-line edits
    clears the finding: list the package as gate-bearing, or list it as
    gate-free — a reviewed assertion the provenance reconciliation keeps
    honest. Depth one only: a package nested inside a gate-free root is this
    check's stated limit, not its oversight.
    """
    import foundationscale as root_pkg

    classified = set(_GATE_PACKAGES) | _KNOWN_GATELESS_PACKAGES
    findings: list[str] = []
    for info in pkgutil.iter_modules(root_pkg.__path__, prefix="foundationscale."):
        if info.ispkg and info.name not in classified:
            findings.append(
                f"{info.name}: first-party package is on nobody's list — classify "
                f"it gate-bearing (_GATE_PACKAGES) or gate-free "
                f"(_KNOWN_GATELESS_PACKAGES). An uncounted package boundary is "
                f"how the parity gate shipped uncertified"
            )
    return findings


def _count_controls(registry: GateRegistry) -> tuple[int, list[str], list[str]]:
    """Count declared controls across the registry.

    ``controls()`` is gate-author code and can itself raise; a gate whose control
    list cannot even be built has no proven ability to fire, which :func:`main`
    treats as a failure (``verify_controls`` never sees it, since calling
    ``gate.controls()`` there would raise the same exception — :func:`main`
    therefore excludes its id from the verification pass, or that pass would
    re-raise out of the audit with the whole report still unprinted).

    Returns:
        A triple ``(total, errors, unverifiable_ids)``: the number of controls
        declared, human-readable errors for gates whose ``controls()`` raised
        (carrying the gate id and the exception, so the finding is actionable),
        and the ids of those gates, so :func:`main` can verify the rest without
        handing the known-broken ones back to ``verify_controls``.
    """
    total = 0
    errors: list[str] = []
    unverifiable_ids: list[str] = []
    for gate in registry:
        try:
            total += len(list(gate.controls()))
        except Exception as exc:  # noqa: BLE001 — report, do not abort the audit
            errors.append(
                f"{gate.id}: controls() raised {type(exc).__name__}: {exc} — "
                f"this gate has no proven controls"
            )
            unverifiable_ids.append(gate.id)
    return total, errors, unverifiable_ids


_GateWalk = Callable[[], tuple[list[str], list[str], list[str]]]
"""The gate-module walk :func:`main` executes: import every gate module so
registration happens, returning ``(imported, failed_attempts, errors)``.

A module-level alias rather than an inline annotation, for two reasons: the
triple is unreadable crammed into a signature at the 100-column limit, and the
tests that inject hermetic walks deserve a named contract to aim at — the same
courtesy :data:`_GATE_PACKAGES` extends to its own readers.
"""


def main(walk: _GateWalk | None = None) -> int:
    """Run every registered control and report. Returns the process exit code.

    Args:
        walk: The gate-module walk to execute. Defaults to
            :func:`_walk_gate_packages`. Tests that pin surfaces downstream of
            the walk — the gate listing, the failure tally — over hermetic
            registries inject a hermetic walk here. WHY a parameter rather
            than only a monkeypatch target: fix24 moved the worker this
            function calls, and the listing test's monkeypatched seam went
            SILENTLY dead — its stub returned a two-tuple into a void while
            main() walked the real tree, importing every shipped gate into the
            process registry inside a test whose docstring promised isolation,
            and nothing reddened at the seam itself. A stub that no longer
            intercepts is the same defect class as a control whose detector
            cannot fire. A parameter cannot rot that way: deleting it
            TypeErrors at every caller, and bypassing it inside this body
            reddens
            tests/gates/test_controls_listing.py::test_main_consults_the_injected_walk.

    Exit 1 means at least one of:

    - the registry is empty (a controls run that verified nothing is the exact
      vacuous pass — ``all([]) is True`` — that this framework refuses to ship);
    - a gate module failed to import, so its gates never ran;
    - a gate-package ROOT failed to import or proved not to be a package,
      unreaching every gate beneath it — recorded as one named finding for the
      root rather than silence over the subtree;
    - the import-side reconciliations fired: the registry holds a gate whose
      defining module the walk never reached (the harness vouches only for
      imports it performed itself), or a first-party sibling package is
      classified neither gate-bearing nor gate-free — the F1 blind spot
      standing open again, closable only by a human edit, in review;
    - a gate's ``controls()`` raises, so nothing about it is proven — the gate is
      named with its exception, and the run continues over the remaining gates;
    - a gate declares an incomplete control pair. No MUST_FIRE means its ability
      to block was never shown; no MUST_PASS means its behaviour on a HEALTHY
      input was verified zero times — a gate that blocks on literally everything
      satisfies every MUST_FIRE fixture and used to pass this job green over
      zero healthy-input evaluations. Each missing kind is its own named
      failure, because each is a capability proven for zero inputs (per
      :func:`~foundationscale.gates.core.verify_controls`);
    - one of a gate's controls produced the wrong verdict: a MUST_FIRE fixture
      it passed over; a MUST_PASS fixture it blocked; a MUST_PASS fixture that
      ABSTAINED without an ``expect_skip`` declaration on the Control; or a
      MUST_PASS fixture carrying ``expect_skip`` whose gate reached PASS anyway
      — a stale declaration, failed in the doctrine-(5)-symmetric direction;
    - not one of a gate's MUST_PASS controls reached a real PASS, however
      honestly declared each abstention was: a gate that has never
      affirmatively accepted any input could abstain — or block — on EVERYTHING
      and exit green here (per
      :func:`~foundationscale.gates.core.verify_controls`).

    The listing printed before the verification pass states each registered
    gate's declared :attr:`~foundationscale.gates.core.Gate.context_type`, with
    an explicit marker where none is declared: how much of the population is
    typed is a coverage fact, and coverage facts are stated, never implied by a
    column the reader has to tally by hand.

    Exit 0 means every gate was shown, in this process, to be capable of blocking
    a defective input and of passing a healthy one.
    """
    # Call-time resolution of the seam, never a bound default argument: the
    # default None means "the real walk", looked up as a fresh module global
    # HERE, so an injected walk is an ordinary parameter value AND the
    # pre-fix27 pattern of monkeypatching the private worker name still lands
    # if anyone uses it — both generations of caller then agree on which
    # function actually ran, and neither can be silently bypassed by the other.
    walk = _walk_gate_packages if walk is None else walk
    imported, failed_imports, failures = walk()
    gate_count = len(REGISTRY)
    # Structural reconciliation runs BEFORE verification, over the registry as
    # imported: a gate the walk never reached is uncertified by definition, and
    # an unclassified first-party package is the F1 blind spot standing open
    # again. Both are findings — loud, named, and never resolved by shrinking
    # the denominator.
    attempted = {*imported, *failed_imports, *_GATE_PACKAGES}
    failures.extend(_uncertified_provenance_findings(REGISTRY, attempted))
    failures.extend(_unclassified_package_findings())
    control_count, control_errors, unverifiable_ids = _count_controls(REGISTRY)
    failures.extend(control_errors)

    print("foundationscale-controls — proving each gate can block before we trust it")
    print(f"gate modules imported: {len(imported)}")
    for name in imported:
        print(f"  {name}")
    print(f"gates registered: {gate_count}")
    # Every registered gate's dispatch declaration, sorted for a stable
    # listing: each line names the declared context_type or carries the
    # explicit no-declaration marker. An empty cell would let a forgotten
    # declaration read as a deliberate untyped gate — silence presenting as
    # consent, inside the tool that exists to refuse exactly that.
    undeclared = 0
    for gate in sorted(REGISTRY, key=lambda g: g.id):
        if gate.context_type is None:
            undeclared += 1
        print(f"  {gate.id} — context_type: {_context_type_label(gate)}")
    if gate_count:
        # The population denominator for the column above: "7/9 typed" is a
        # stated fact; a bare column of markers would make the reader count.
        # Zero gates print no ratio — a 0/0 here would restate the vacuous
        # pass that the failure below names correctly instead.
        print(
            f"  context_type declared: {gate_count - undeclared}/{gate_count} gates; "
            f"{undeclared} on untyped legacy broadcast"
        )
    print(f"controls declared: {control_count}")

    if gate_count == 0:
        failures.insert(
            0,
            "registry is empty — no gates ran, so this run verified nothing. "
            "All-clear over zero checks is the vacuous pass this framework exists "
            "to prevent.",
        )
    else:
        # Gates whose controls() could not even be built were recorded above, with
        # gate id and exception. They must NOT be handed back to verify_controls:
        # it calls list(gate.controls()) with no guard, the same exception would
        # escape main() as a traceback, and every collected finding would be lost
        # with the unprinted report. Exclude their ids and run the positive
        # controls for exactly the gates just shown to have a buildable list.
        verifiable_ids = [gate.id for gate in REGISTRY if gate.id not in unverifiable_ids]
        failures.extend(verify_controls(REGISTRY, gate_ids=verifiable_ids))
        # The certification's denominator, printed on every non-empty run:
        # "all controls held" is a claim ABOUT a count, so the count travels
        # with the claim. F1 shipped "OK" where this receipt would have read
        # 9 over a tree holding 10; the reconciliation above is the tripwire,
        # and this line is the receipt a human can eyeball without re-running
        # anything.
        print(f"controls executed for {len(verifiable_ids)} of {gate_count} registered gates")

    if failures:
        print(f"\n{len(failures)} control failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        print("\nresult: FAILED — at least one gate is not proven able to block.")
        return 1

    # Exit 0 now proves three distinct things about the healthy-input half
    # alone, so the banner's word choice is load-bearing: every fixture
    # produced its DECLARED outcome (a declared, reasoned abstention is a
    # first-class declared outcome — writing "passed its known-healthy ones"
    # here would claim affirmations the abstaining fixtures never gave, and
    # doctrine (5) grades that even in a triumphal banner), and every gate
    # additionally reached at least one affirmative PASS.
    print(
        f"\nresult: OK — all {gate_count} registered gates blocked their "
        "deliberately defective inputs; every known-healthy fixture produced "
        "its declared outcome, and every gate reached at least one "
        "affirmative PASS."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
