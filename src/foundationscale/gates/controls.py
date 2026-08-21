"""Executable self-test for the gate suite.

Why this exists
---------------
The audit's sharpest incident was not in training code but in the tool written to
*detect* silent success: it reported ``all_identity: True`` on a corrupt artifact
because the comparison set was empty and ``all([])`` is ``True``. Every gate in this
repository is a detector of exactly that shape, so each one must carry fixtures —
:attr:`~foundationscale.gates.core.ControlKind.MUST_FIRE` inputs that it provably
blocks on — and those fixtures must actually be run, in CI, on every change. A gate
whose controls are never executed rots into a no-op at the same speed as the code
it watches.

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

Wired into CI as the ``controls`` job of ``.github/workflows/ci.yml`` and exposed
as the ``foundationscale-controls`` console script. Exits 1 on any failure; exit 0
means every gate was shown, just now, to be capable of blocking.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

from foundationscale.gates.core import REGISTRY, Gate, GateRegistry, verify_controls

_THIS_MODULE = "foundationscale.gates.controls"

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


def _import_gate_modules() -> tuple[list[str], list[str]]:
    """Import every module under :mod:`foundationscale.gates` so gates register.

    Registration is a side effect of import (the ``@register`` decorator), so a
    module that is never imported contributes nothing to the registry — and a
    controls run over what remains would be a false "all clear". Modules are
    imported individually and import exceptions are collected rather than raised:
    a gate module that cannot import on this box is itself a finding, and the run
    must continue so the report shows *all* broken modules, not just the first.

    Returns:
        A pair ``(imported, errors)`` of module names successfully imported and
        human-readable import-failure strings.
    """
    import foundationscale.gates as package

    imported: list[str] = []
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

    for info in pkgutil.walk_packages(
        package.__path__, prefix=f"{package.__name__}.", onerror=_record_walk_failure
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
        else:
            imported.append(info.name)
    return sorted(imported), errors


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


def main() -> int:
    """Run every registered control and report. Returns the process exit code.

    Exit 1 means at least one of:

    - the registry is empty (a controls run that verified nothing is the exact
      vacuous pass — ``all([]) is True`` — that this framework refuses to ship);
    - a gate module failed to import, so its gates never ran;
    - a gate's ``controls()`` raises, so nothing about it is proven — the gate is
      named with its exception, and the run continues over the remaining gates;
    - a gate has no MUST_FIRE control, or one of its controls produced the wrong
      verdict (per :func:`~foundationscale.gates.core.verify_controls`).

    The listing printed before the verification pass states each registered
    gate's declared :attr:`~foundationscale.gates.core.Gate.context_type`, with
    an explicit marker where none is declared: how much of the population is
    typed is a coverage fact, and coverage facts are stated, never implied by a
    column the reader has to tally by hand.

    Exit 0 means every gate was shown, in this process, to be capable of blocking.
    """
    imported, failures = _import_gate_modules()
    gate_count = len(REGISTRY)
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

    if failures:
        print(f"\n{len(failures)} control failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        print("\nresult: FAILED — at least one gate is not proven able to block.")
        return 1

    print("\nresult: OK — every gate blocked its deliberately defective inputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
