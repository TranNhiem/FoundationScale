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
   never imported, ``verify_controls()`` iterates an empty registry and returns an
   empty failure list — "all clear" over nothing, the vacuous pass one level up.
   ``main()`` populates the registry first and exits 1 if it ends up empty.
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

from foundationscale.gates.core import REGISTRY, GateRegistry, verify_controls

_THIS_MODULE = "foundationscale.gates.controls"


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
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
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


def _count_controls(registry: GateRegistry) -> tuple[int, list[str]]:
    """Count declared controls across the registry.

    ``controls()`` is gate-author code and can itself raise; a gate whose control
    list cannot even be built has no proven ability to fire, which :func:`main`
    treats as a failure (``verify_controls`` never sees it, since calling
    ``gate.controls()`` there would raise the same exception).

    Returns:
        A pair ``(total, errors)`` of the number of controls declared and
        human-readable errors for gates whose ``controls()`` raised.
    """
    total = 0
    errors: list[str] = []
    for gate in registry:
        try:
            total += len(list(gate.controls()))
        except Exception as exc:  # noqa: BLE001 — report, do not abort the audit
            errors.append(
                f"{gate.id}: controls() raised {type(exc).__name__}: {exc} — "
                f"this gate has no proven controls"
            )
    return total, errors


def main() -> int:
    """Run every registered control and report. Returns the process exit code.

    Exit 1 means at least one of:

    - the registry is empty (a controls run that verified nothing is the exact
      vacuous pass — ``all([]) is True`` — that this framework refuses to ship);
    - a gate module failed to import, so its gates never ran;
    - a gate has no MUST_FIRE control, or one of its controls produced the wrong
      verdict (per :func:`~foundationscale.gates.core.verify_controls`).

    Exit 0 means every gate was shown, in this process, to be capable of blocking.
    """
    imported, failures = _import_gate_modules()
    gate_count = len(REGISTRY)
    control_count, control_errors = _count_controls(REGISTRY)
    failures.extend(control_errors)

    print("foundationscale-controls — proving each gate can block before we trust it")
    print(f"gate modules imported: {len(imported)}")
    for name in imported:
        print(f"  {name}")
    print(f"gates registered: {gate_count}")
    print(f"controls declared: {control_count}")

    if gate_count == 0:
        failures.insert(
            0,
            "registry is empty — no gates ran, so this run verified nothing. "
            "All-clear over zero checks is the vacuous pass this framework exists "
            "to prevent.",
        )
    else:
        failures.extend(verify_controls(REGISTRY))

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
