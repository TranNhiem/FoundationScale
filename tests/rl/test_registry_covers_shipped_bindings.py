"""Registry-coverage audit: every shipped REGISTRABLE rl binding owns a name.

WHAT THIS MODULE PROVES: walking ``pkgutil.iter_modules`` over
``foundationscale.rl`` and keeping every public, locally-defined class ending
in ``Algorithm`` (the ``Algorithm`` protocol excepted) and every local
callable ending in ``_algorithm`` yields a set of bindings that partitions
cleanly into NEEDS_ARGS (``inspect.signature`` reports at least one required
parameter) and REGISTRABLE (zero-argument construction succeeds and returns
an ``Algorithm`` instance), with no third bucket. Every REGISTRABLE binding
must then be reachable by name: the type it constructs directly must equal
the type constructed through ``lookup_algorithm(name)`` for at least one
name in ``available_algorithm_names()``. Every registered name must resolve
to a type some discovered REGISTRABLE binding constructs (no orphans, no
double claims needed beyond that). No submodule import failure is
swallowed -- a failed import is reported by module and exception type, and
zero failures is the healthy, passing state. Against the tree as shipped
this suite FAILS on exactly one binding, ``ppo.PPOAlgorithm``, which is
REGISTRABLE by measurement yet absent from the default registry; adding
``_REGISTRY["ppo"] = PPOAlgorithm`` turns it green.

WHAT THIS MODULE DOES NOT PROVE: reachability by name is the only property
measured here. That a registered algorithm computes correct losses,
advantages, or gradients; that ``setup`` wires roles and refs correctly;
that defaults are sensible -- none of that is claimed; numerical correctness
is another module's proof surface. No table of expected bindings is
maintained: the denominator is DISCOVERED, not declared, so a newly added
algorithm module joins it automatically. The only hard-coded names are the
two non-vacuity anchors in R4, whose job is to forbid an empty walk from
reading as agreement.

CONTROL ARMS: the detector is fed INJECTED snapshots (never a monkeypatched
registry) -- one complete and one missing a known-good entry, differing in
exactly one input -- proving it can fire and is not stuck red; the real
classifier is run over synthetic module-like objects -- one zero-argument
``*_algorithm`` factory returning an ``Algorithm`` instance, one
``*Algorithm`` class with required constructor parameters -- proving the
REGISTRABLE and NEEDS_ARGS buckets each catch the shape they claim to catch.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import cast

from foundationscale import rl as rl_package
from foundationscale.rl.algorithm import Algorithm
from foundationscale.rl.grpo import GRPOAlgorithm
from foundationscale.rl.registry import (
    available_algorithm_names,
    lookup_algorithm,
    reset_algorithm_registry,
)

__all__ = ()


class _Kind(Enum):
    """The three buckets a discovered binding can land in; one must be empty."""

    REGISTRABLE = "registrable"
    NEEDS_ARGS = "needs_args"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class _Discovery:
    """The result of one package walk: bindings found and failures recorded."""

    bindings: dict[str, Callable[..., object]]
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Partition:
    """A clean partition has an EMPTY indeterminate bucket -- asserted in R2."""

    registrable: dict[str, Callable[..., object]]
    needs_args: dict[str, Callable[..., object]]
    indeterminate: dict[str, Callable[..., object]]


def _is_binding(attr_name: str, candidate: object) -> bool:
    """Apply the two discovery rules to one locally-defined attribute.

    The ``Algorithm`` protocol itself satisfies the name rule but a contract
    is not a binding -- nothing constructs it -- so the exclusion keys on
    object identity, which keeps working if a second protocol ever arrives.
    """
    if isinstance(candidate, type):
        return attr_name.endswith("Algorithm") and candidate is not Algorithm
    return attr_name.endswith("_algorithm") and callable(candidate)


def _scan_module(short_name: str, module: ModuleType) -> dict[str, Callable[..., object]]:
    """Collect the bindings one module object ships, keyed "<short>.<attr>".

    The ``__module__`` check counts a re-export at its home module, not
    twice. Takes the module as a parameter so the control arms can hand in a
    synthetic module-like object without touching the real package.
    """
    found: dict[str, Callable[..., object]] = {}
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        candidate = getattr(module, attr_name)
        if getattr(candidate, "__module__", None) != module.__name__:
            continue
        if _is_binding(attr_name, candidate):
            found[f"{short_name}.{attr_name}"] = cast("Callable[..., object]", candidate)
    return found


def _discover(package_name: str, package_path: Iterable[str]) -> _Discovery:
    """Walk the direct submodules of the package and collect every binding.

    Import failures are RECORDED, never re-raised and never dropped: a
    submodule that cannot import is a coverage hole whose name and exception
    type the R5 assertion prints, and a silent swallow is what lets a check
    read success on a denominator it never measured.
    """
    bindings: dict[str, Callable[..., object]] = {}
    failures: list[str] = []
    for info in pkgutil.iter_modules(package_path):
        qualified = f"{package_name}.{info.name}"
        try:
            module = importlib.import_module(qualified)
        except Exception as exc:
            failures.append(f"{qualified}: {type(exc).__name__}: {exc}")
            continue
        bindings.update(_scan_module(info.name, module))
    return _Discovery(bindings=bindings, failures=tuple(failures))


def _classify_binding(candidate: Callable[..., object]) -> _Kind:
    """Classify one binding by measuring it, exactly as the ground truth did.

    NEEDS_ARGS is decided by ``inspect.signature``: any required positional
    or keyword parameter means the registry -- which holds only zero-argument
    factories -- can never hold this binding. REGISTRABLE is decided by
    ACTUALLY CONSTRUCTING the binding with no arguments and checking the
    product against the registry's own ``isinstance(instance, Algorithm)``
    gate. Anything left over -- a signature the interpreter cannot produce,
    a zero-argument call that raises, a product that is not an ``Algorithm``
    -- is INDETERMINATE: a third bucket that R2 requires to be empty, so the
    partition is exhaustive or the suite says why not.
    """
    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return _Kind.INDETERMINATE
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if required:
        return _Kind.NEEDS_ARGS
    try:
        instance = candidate()
    except Exception:
        return _Kind.INDETERMINATE
    if isinstance(instance, Algorithm):
        return _Kind.REGISTRABLE
    return _Kind.INDETERMINATE


def _partition(bindings: Mapping[str, Callable[..., object]]) -> _Partition:
    """Sort every binding into its measured bucket; takes bindings as input."""
    registrable: dict[str, Callable[..., object]] = {}
    needs_args: dict[str, Callable[..., object]] = {}
    indeterminate: dict[str, Callable[..., object]] = {}
    for dotted_name, candidate in sorted(bindings.items()):
        kind = _classify_binding(candidate)
        if kind is _Kind.REGISTRABLE:
            registrable[dotted_name] = candidate
        elif kind is _Kind.NEEDS_ARGS:
            needs_args[dotted_name] = candidate
        else:
            indeterminate[dotted_name] = candidate
    return _Partition(registrable=registrable, needs_args=needs_args, indeterminate=indeterminate)


def _registrable_minus_registered(
    registrable: Mapping[str, Callable[..., object]],
    registered: Mapping[str, type[object]],
) -> tuple[str, ...]:
    """The R1 rule: REGISTRABLE bindings no registered name's type reaches.

    Reachability is measured by CONSTRUCTING -- the type the binding builds
    directly must appear among the types the registered names build -- so a
    name bound to the wrong binding cannot pass, and a name-membership check
    cannot see that lie. Both inputs are parameters so the control arms can
    inject doctored snapshots instead of monkeypatching the registry.
    """
    reached_types = frozenset(registered.values())
    return tuple(
        dotted_name
        for dotted_name, factory in sorted(registrable.items())
        if type(factory()) not in reached_types
    )


def _live_registered_types() -> dict[str, type[object]]:
    """Snapshot the registry's name-to-constructed-type map through its API.

    Resets first: the registry is module-level mutable, and reading it
    without a reset would inherit whatever an earlier test registered. The
    default install -- not a test registration -- is the install under
    measurement.
    """
    reset_algorithm_registry()
    names = available_algorithm_names()
    return {name: type(lookup_algorithm(name)) for name in names}


def _real_partition() -> _Partition:
    """Discover and partition the live package, refusing a poisoned walk."""
    discovery = _discover(rl_package.__name__, rl_package.__path__)
    assert discovery.failures == (), (
        f"{len(discovery.failures)} rl submodule(s) failed to import, so the "
        f"denominator under measurement is short: {list(discovery.failures)}"
    )
    return _partition(discovery.bindings)


def test_every_registrable_binding_is_reachable_by_name() -> None:
    """R1: REGISTRABLE minus registered is empty, missing bindings named.

    This FAILS today naming ``ppo.PPOAlgorithm`` -- the one measured
    REGISTRABLE binding with no registry entry -- and passes after
    ``_REGISTRY["ppo"] = PPOAlgorithm`` lands. The message names the binding,
    not just a count.
    """
    partition = _real_partition()
    registered = _live_registered_types()
    missing = _registrable_minus_registered(partition.registrable, registered)
    assert missing == (), (
        f"{len(missing)} of {len(partition.registrable)} zero-argument-"
        f"constructible Algorithm bindings construct a type none of the "
        f"{len(registered)} registered names reaches: {list(missing)}; add "
        "each to _install_default_algorithms() or give its constructor a "
        "required parameter so it measures NEEDS_ARGS"
    )


def test_the_discovered_set_partitions_into_two_buckets_only() -> None:
    """R2: the walk is non-empty and the partition is exhaustive, no third bucket.

    The denominator is MEASURED, not declared: every discovered binding lands
    in REGISTRABLE or NEEDS_ARGS, the buckets sum to the walk, and the
    INDETERMINATE bucket -- a binding that claims zero-arg shape but fails to
    construct or fails the isinstance gate -- must be empty, because a third
    bucket is exactly where a rotting classification would hide.
    """
    discovery = _discover(rl_package.__name__, rl_package.__path__)
    assert discovery.failures == (), (
        f"{len(discovery.failures)} rl submodule(s) failed to import, so the "
        f"partition is grading a shortened walk: {list(discovery.failures)}"
    )
    assert discovery.bindings, (
        "the package walk produced 0 bindings; an empty walk partitions "
        "trivially and every comparison below would grade a fictional "
        "denominator"
    )
    partition = _partition(discovery.bindings)
    bucket_sizes = (
        len(partition.registrable) + len(partition.needs_args) + len(partition.indeterminate)
    )
    assert bucket_sizes == len(discovery.bindings), (
        f"the three buckets hold {bucket_sizes} of {len(discovery.bindings)} "
        "discovered bindings; a binding escaped every bucket and the partition "
        "is not exhaustive"
    )
    assert partition.indeterminate == {}, (
        f"{len(partition.indeterminate)} binding(s) are INDETERMINATE -- "
        "zero-argument by signature yet failing construction or the "
        "isinstance(Algorithm) gate: "
        f"{sorted(partition.indeterminate)}; the registry could never hold "
        "such a binding even if a name were declared for it"
    )
    assert partition.registrable, (
        "the walk found bindings but NONE is registrable; an empty registrable "
        "denominator would pass R1 vacuously"
    )


def test_no_registered_name_is_an_orphan() -> None:
    """R3: every registered name resolves to a type the walk discovered.

    The reverse direction of R1: a registry entry pointing at something the
    walk cannot see means the WALK is too narrow, and the failure message
    says so -- the fix would be in the discovery rule, not the registry.
    """
    partition = _real_partition()
    registered = _live_registered_types()
    registrable_types = frozenset(type(factory()) for factory in partition.registrable.values())
    orphans = tuple(
        sorted(name for name, reached in registered.items() if reached not in registrable_types)
    )
    assert orphans == (), (
        f"{len(orphans)} of {len(registered)} registered names construct a "
        f"type no discovered REGISTRABLE binding constructs: {list(orphans)}; "
        "an entry pointing at something the walk cannot see means the walk "
        "is too narrow -- widen the discovery rule or register through a "
        "binding the walk can reach"
    )


def test_the_walk_is_anchored_to_grpo_and_ppo() -> None:
    """R4 non-vacuity: the walk must name both anchors by exact dotted name.

    ``all([])`` is True: if a future refactor makes the walk return nothing,
    R1, R2 and R3 would all pass vacuously. Naming a known-good binding
    (grpo) and the currently-orphaned binding (ppo) pins the walk to real
    members. This is the ONLY place a hard-coded name belongs, and it is
    load-bearing precisely because nothing else here declares one.
    """
    discovery = _discover(rl_package.__name__, rl_package.__path__)
    assert discovery.failures == (), (
        f"{len(discovery.failures)} rl submodule(s) failed to import before "
        f"the anchors could be checked: {list(discovery.failures)}"
    )
    for anchor in ("grpo.GRPOAlgorithm", "ppo.PPOAlgorithm"):
        assert anchor in discovery.bindings, (
            f"the anchor {anchor!r} is absent from the {len(discovery.bindings)}"
            " discovered bindings; the walk or the binding predicate is broken "
            "and every comparison in this module is grading a fictional "
            f"denominator (present keys: {sorted(discovery.bindings)})"
        )


def test_no_rl_submodule_import_failure_is_swallowed() -> None:
    """R5: import failures FAIL the suite, named by module and exception type.

    This control demands ZERO failures: the healthy tree has zero, and zero
    passes. A control inverted to require failures would be stuck red on a
    healthy tree -- the bug this module exists, in part, to retire.
    """
    discovery = _discover(rl_package.__name__, rl_package.__path__)
    assert discovery.failures == (), (
        f"{len(discovery.failures)} rl submodule(s) failed to import, so "
        "their bindings were never counted and the denominator is short by "
        f"an unmeasured amount: {list(discovery.failures)}"
    )


def _synthetic_complete_snapshot(
    registrable: Mapping[str, Callable[..., object]],
) -> dict[str, type[object]]:
    """Build a COMPLETE name-to-type snapshot from real registrable bindings.

    Every input to the control arms is derived from the real walk, never
    hand-typed, so a control cannot drift from the thing it certifies.
    """
    return {
        f"injected::{dotted_name}": type(factory())
        for dotted_name, factory in sorted(registrable.items())
    }


def test_control_arm_the_detector_reports_a_dropped_snapshot_entry() -> None:
    """C-DETECT: a snapshot missing one known-good name reports that binding.

    Feeds the R1 rule an INJECTED snapshot -- the complete synthetic map
    minus the grpo entry -- and asserts the rule reports exactly
    ``grpo.GRPOAlgorithm``. Proves the detector can fire. Passes today: the
    live ppo defect is never touched by this arm.
    """
    partition = _real_partition()
    complete = _synthetic_complete_snapshot(partition.registrable)
    dropped = "injected::grpo.GRPOAlgorithm"
    assert dropped in complete, (
        "the control requires the grpo anchor among the registrable "
        "bindings; without it there is no real entry to drop and this arm "
        "proves nothing"
    )
    doctored = {name: reached for name, reached in complete.items() if name != dropped}
    assert len(doctored) == len(complete) - 1, (
        "dropping the anchor removed 0 entries; the snapshot below would be "
        "the complete one and the arm would be vacuous"
    )
    missing = _registrable_minus_registered(partition.registrable, doctored)
    assert missing == ("grpo.GRPOAlgorithm",), (
        f"removing the grpo entry from a complete snapshot of "
        f"{len(complete)} names must make the R1 rule report exactly "
        f"('grpo.GRPOAlgorithm',); got {list(missing)} -- a rule that cannot "
        "fire on a deliberate gap proves nothing when it stays quiet"
    )


def test_control_arm_the_detector_is_quiet_over_a_complete_snapshot() -> None:
    """C-INSTRUMENT: the same rule over a complete snapshot reports nothing.

    Identical to C-DETECT except for exactly one input: the snapshot keeps
    the dropped entry. Proves the detector is not stuck red -- its reds are
    readings of the data, not a broken instrument. Passes today.
    """
    partition = _real_partition()
    complete = _synthetic_complete_snapshot(partition.registrable)
    missing = _registrable_minus_registered(partition.registrable, complete)
    assert missing == (), (
        f"a snapshot covering all {len(partition.registrable)} registrable "
        f"bindings must report no gaps; got {list(missing)} -- the detector "
        "fires on complete data and its quiet runs mean nothing"
    )


def test_control_arm_the_classifier_marks_a_zero_arg_factory_registrable() -> None:
    """C-SELFMATCH: a synthetic zero-arg ``*_algorithm`` factory classifies REGISTRABLE.

    Plants a module-like object exposing one zero-argument ``*_algorithm``
    factory whose product is a real ``Algorithm`` instance, then runs the
    REAL scanner and classifier over it. Proves the classifier matches the
    shape it claims to, rather than grading green because it matches
    nothing.
    """
    module = ModuleType(f"{rl_package.__name__}.synthetic_probe")

    def probe_algorithm() -> Algorithm:
        return GRPOAlgorithm()

    probe_algorithm.__module__ = module.__name__
    module.probe_algorithm = probe_algorithm
    found = _scan_module("synthetic_probe", module)
    assert set(found) == {"synthetic_probe.probe_algorithm"}, (
        f"the real scanner over the synthetic module must find exactly the "
        f"planted factory; found {sorted(found)}"
    )
    kind = _classify_binding(found["synthetic_probe.probe_algorithm"])
    assert kind is _Kind.REGISTRABLE, (
        f"a zero-argument factory returning an Algorithm instance must "
        f"classify REGISTRABLE; got {kind.value} -- the classifier does not "
        "match the shape this module's R1 rule demands of the registry"
    )


def test_control_arm_the_classifier_marks_a_configured_class_needs_args() -> None:
    """C-NEEDSARGS: a synthetic ``*Algorithm`` class with required ctor args is NEEDS_ARGS.

    A same-shape negative control: the planted class passes the name rule
    and the ``__module__`` rule, but its constructor requires parameters, so
    it must NOT land in the REGISTRABLE bucket and so must never be demanded
    of the registry.
    """
    module = ModuleType(f"{rl_package.__name__}.synthetic_probe")

    class ProbeAlgorithm:
        def __init__(self, name: str, objective: str) -> None:
            self.name = name
            self.objective = objective

    ProbeAlgorithm.__module__ = module.__name__
    module.ProbeAlgorithm = ProbeAlgorithm
    found = _scan_module("synthetic_probe", module)
    assert set(found) == {"synthetic_probe.ProbeAlgorithm"}, (
        f"the real scanner over the synthetic module must find exactly the "
        f"planted class; found {sorted(found)}"
    )
    partition = _partition(found)
    assert "synthetic_probe.ProbeAlgorithm" not in partition.registrable, (
        "a class whose constructor requires arguments must never enter the "
        "REGISTRABLE bucket; if it did, R1 would demand a registry name for "
        "something the registry cannot construct"
    )
    assert set(partition.needs_args) == {"synthetic_probe.ProbeAlgorithm"}, (
        f"the planted configured class must classify NEEDS_ARGS; buckets: "
        f"registrable={sorted(partition.registrable)}, "
        f"needs_args={sorted(partition.needs_args)}, "
        f"indeterminate={sorted(partition.indeterminate)}"
    )
