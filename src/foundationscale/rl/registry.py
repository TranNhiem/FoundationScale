"""Algorithm-name registry for concrete RL bindings.

The registry maps a canonical algorithm name to a zero-argument factory
returning a fresh :class:`foundationscale.rl.algorithm.Algorithm`. A lookup
therefore never shares setup state between runs, while registration keeps
the algorithm declarations outside the core wiring implementation.

WHAT THIS MODULE CLAIMS: registered names are unique, available names are
reported in sorted order, and every successful lookup returns a fresh object
passing the runtime :class:`Algorithm` structural check.

WHAT THIS MODULE DOES NOT CLAIM: that a returned algorithm is wired,
semantically correct for a particular run, numerically finite, or compatible
with a loss beyond what the algorithm's own setup handshake measures.
"""

from __future__ import annotations

from collections.abc import Callable

from foundationscale.rl.algorithm import Algorithm

__all__ = (
    "AlgorithmRegistryRefusal",
    "available_algorithm_names",
    "lookup_algorithm",
    "register_algorithm",
    "reset_algorithm_registry",
)


class AlgorithmRegistryRefusal(ValueError):
    """A registration or lookup could not be attributed to one binding.

    WHAT IS CLAIMED: the refusal names the requested key and reports the
    available-name denominator used by the check.

    WHAT IS NOT CLAIMED: anything about the mathematical or wiring validity
    of an algorithm whose name was never successfully registered.
    """


# The only mutable module-level state in this module. Tests reset it through
# reset_algorithm_registry(), never by assigning to this private dictionary.
_REGISTRY: dict[str, Callable[[], Algorithm]] = {}


def _canonical_name(name: str, *, field: str) -> str:
    canonical = name.strip() if isinstance(name, str) else name
    if not isinstance(canonical, str) or not canonical:
        raise AlgorithmRegistryRefusal(
            f"field {field}={name!r}: an algorithm registry key must be a "
            f"non-empty str; absence of a name is not a name"
        )
    return canonical


def available_algorithm_names() -> tuple[str, ...]:
    """Return the registered algorithm names in deterministic order.

    WHAT IS CLAIMED: the returned value is a new, sorted tuple containing
    exactly the names present in the registry at the call.

    WHAT IS NOT CLAIMED: that the return remains valid after another
    registration or reset; callers wanting stability must retain the tuple.
    """
    return tuple(sorted(_REGISTRY))


def register_algorithm(name: str, factory: Callable[[], Algorithm]) -> None:
    """Register one algorithm name and its zero-argument factory.

    A duplicate name is refused instead of overwritten. Overwriting would
    let a later import silently change every report and manifest using the
    earlier name.

    WHAT IS CLAIMED on success: ``name`` resolves to exactly this factory
    until the registry is reset.

    WHAT IS NOT CLAIMED: that ``factory()`` succeeds or produces a useful
    wiring; construction and semantic validity are measured at lookup and
    setup respectively.
    """
    canonical = _canonical_name(name, field="name")
    if not callable(factory):
        raise AlgorithmRegistryRefusal(
            f"field factory={factory!r}: 1 of 1 registration factories for "
            f"{canonical!r} must be callable; a registry entry is a way to "
            f"construct a fresh algorithm, not a shared configured instance"
        )
    if canonical in _REGISTRY:
        names = available_algorithm_names()
        raise AlgorithmRegistryRefusal(
            f"field name={canonical!r}: duplicate registration requested; "
            f"1 of 1 new registrations used an occupied key, while 1 of "
            f"{len(names)} registered names already owns it: {names!r}"
        )
    _REGISTRY[canonical] = factory


def lookup_algorithm(name: str) -> Algorithm:
    """Construct and return the fresh algorithm registered under ``name``.

    WHAT IS CLAIMED: the requested key existed when the lookup started, and
    the constructed object passed the runtime :class:`Algorithm` structural
    check before being returned.

    WHAT IS NOT CLAIMED: that the algorithm has been set up, that its
    required components are present, or that its declared semantics agree
    with a loss. Those are setup-time measurements, not registry lookups.
    """
    canonical = _canonical_name(name, field="name")
    try:
        factory = _REGISTRY[canonical]
    except KeyError as exc:
        names = available_algorithm_names()
        raise AlgorithmRegistryRefusal(
            f"requested key name={canonical!r}: 0 of {len(names)} available "
            f"algorithm names matched; available names ({len(names)}): "
            f"{names!r}"
        ) from exc
    algorithm = factory()
    if not isinstance(algorithm, Algorithm):
        raise AlgorithmRegistryRefusal(
            f"field factory for name={canonical!r}: 1 of 1 constructed "
            f"objects was not an Algorithm "
            f"({type(algorithm).__name__}); the registry may encode only a "
            f"fresh concrete algorithm binding, never a configured component "
            f"or unrelated object"
        )
    return algorithm


def reset_algorithm_registry() -> None:
    """Restore the deterministic default registry.

    The reset removes every entry, including test registrations, and then
    reinstalls the built-in bindings. This is the supported test-reset
    operation for the module's private mutable registry.

    WHAT IS CLAIMED: after return, the registry contains exactly the default
    factories installed by this module.

    WHAT IS NOT CLAIMED: that algorithm objects constructed before the reset
    become invalid; existing references remain owned by their callers.
    """
    _REGISTRY.clear()
    _install_default_algorithms()


def _install_default_algorithms() -> None:
    # Local imports avoid a module-import cycle while keeping the built-in
    # bindings deterministic. These imports are not file or network I/O.
    #
    # Installation happens HERE, not at each algorithm module's import, so
    # that reset_algorithm_registry() restores the same set every time. A
    # family that registered itself on its own import would survive a reset
    # only if something re-imported it, and a guard of the shape
    # ``if name not in available_algorithm_names(): register(...)`` would
    # SKIP silently when a different binding already held the name -- the
    # absence would then read as a successful install.
    from foundationscale.rl.grpo import GRPOAlgorithm
    from foundationscale.rl.policy_gradient import (
        ReinforceBaselineAlgorithm,
        ReinforcePlusPlusAlgorithm,
        RLOOAlgorithm,
    )
    from foundationscale.rl.preference import (
        cpo_algorithm,
        dpo_algorithm,
        ipo_algorithm,
        kto_algorithm,
        orpo_algorithm,
        simpo_algorithm,
    )

    _REGISTRY["grpo"] = GRPOAlgorithm
    _REGISTRY["reinforce_baseline"] = ReinforceBaselineAlgorithm
    _REGISTRY["reinforce_pp"] = ReinforcePlusPlusAlgorithm
    _REGISTRY["rloo"] = RLOOAlgorithm
    # The preference family registers six FACTORIES, not six classes. The other
    # four entries are classes because their constructors take no argument; a
    # preference binding always carries an objective, so the zero-argument
    # callable the registry needs is the family's default-configured factory.
    # Registering `PreferenceAlgorithm` itself would put an unconfigured
    # binding behind a name, and an objective is not a default.
    _REGISTRY["cpo"] = cpo_algorithm
    _REGISTRY["dpo"] = dpo_algorithm
    _REGISTRY["ipo"] = ipo_algorithm
    _REGISTRY["kto"] = kto_algorithm
    _REGISTRY["orpo"] = orpo_algorithm
    _REGISTRY["simpo"] = simpo_algorithm


_install_default_algorithms()
