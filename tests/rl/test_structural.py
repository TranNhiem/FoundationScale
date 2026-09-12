"""Structural-conformance tests for ``foundationscale.rl.structural``.

WHAT THIS FILE CLAIMS: that ``structural_report`` closes the hole the
measured background pins -- ``typing.runtime_checkable`` makes
``isinstance()`` check method PRESENCE ONLY and NEVER signatures, so a
mis-signatured adapter passes every setup-time ``isinstance`` gate and
only explodes at the first call as a ``TypeError``. The positive
controls (group A) first assert that the hole is still open
(``isinstance`` is True), then assert that ``structural_report`` fires
anyway.

WHAT THIS FILE DOES NOT CLAIM: anything about how the adapter behind a
double behaves at runtime. The point under test is SHAPE, not behaviour.

Most doubles here are local, minimal classes. Group H is the exception
and is deliberately so: it runs the real shipped ``AdvantageFn`` /
``TemporalAdvantageFn`` pair, because a checker validated only against
doubles written by the same hand that wrote the checker has agreed with
itself. That pair is the measured #396 case -- two protocols that declare
the SAME single member name and differ only in signature, so
``isinstance`` cannot separate them at all.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pytest

from foundationscale.rl.advantage import (
    AdvantageFn,
    GroupNormalisedAdvantage,
    LearnedValueAdvantageEstimation,
    TemporalAdvantageFn,
)
from foundationscale.rl.interfaces import LossFn
from foundationscale.rl.rollout import RolloutSource
from foundationscale.rl.structural import (
    StructuralRefusal,
    StructuralReport,
    structural_report,
)
from foundationscale.rl.value_head import ValueHead
from foundationscale.rl.weightsync import WeightSync

# ---------------------------------------------------------------------------
# Local doubles. Presence-shaped like the protocols, signature-shaped
# deliberately right or wrong per test. Annotations use ``object`` stand-ins
# because the comparison under test is about PARAMETERS (names, kinds,
# requiredness), not about the adapter-side types behind them.
# ---------------------------------------------------------------------------


class _WrongKwargValueHead:
    """ValueHead-shaped by presence, wrong by signature: ``estimate`` demands
    a second required argument no ``ValueHead`` caller passes.

    The mismatch is ARITY, not a parameter name, and that is forced by the
    protocol rather than chosen for variety: ``ValueHead.estimate`` declares
    its parameter POSITIONAL-ONLY, so the name is explicitly not part of the
    contract and a head that renames it to ``features`` is CORRECT -- it
    accepts every call the protocol permits (#397, MEASURED). Modelling the
    defect as a rename here would assert a mismatch that is not one, and the
    test would be pinning the checker's bug rather than the protocol's.

    The sibling doubles for ``RolloutSource`` and ``WeightSync`` below DO
    model a rename, and correctly so: those protocols declare
    POSITIONAL_OR_KEYWORD parameters, so their names ARE part of the contract.
    """

    def estimate(self, features: object, scale: float) -> list[list[float]]:
        return [[0.0]]

    def capabilities(self) -> object:
        return object()


class _WrongKwargRolloutSource:
    """RolloutSource-shaped by presence, wrong by signature: ``inputs`` not ``prompts``."""

    def generate(self, inputs: object) -> object:
        return object()

    def capabilities(self) -> object:
        return object()


class _WrongKwargWeightSync:
    """WeightSync-shaped by presence, wrong by signature: ``name_map`` not ``mapping``."""

    def sync(self, name_map: Mapping[str, str]) -> object:
        return object()

    def capabilities(self) -> object:
        return object()


class _ExactValueHead:
    """Exact ValueHead signature double."""

    def estimate(self, batch: object) -> list[list[float]]:
        return []

    def capabilities(self) -> object:
        return object()


class _OptionalKwargValueHead:
    """ValueHead plus one OPTIONAL keyword-only parameter -- strictly more permissive."""

    def estimate(self, batch: object, *, dtype: str | None = None) -> list[list[float]]:
        _ = dtype
        return []

    def capabilities(self) -> object:
        return object()


class _KwargsCatchAllValueHead:
    """ValueHead with a **kwargs catch-all -- accepts every call site the core can make."""

    def estimate(self, batch: object, **kwargs: object) -> list[list[float]]:
        _ = kwargs
        return []

    def capabilities(self) -> object:
        return object()


class _MissingCapabilities:
    """Has ``estimate`` with the right signature but no ``capabilities`` at all."""

    def estimate(self, batch: object) -> list[list[float]]:
        return []


class _LearnedAdvantageStyleLoss:
    """LossFn-shaped by presence, but adds TWO extra REQUIRED keyword-only parameters.

    Modelled on the measured LearnedValueAdvantageEstimation case: an
    implementation whose ``__call__`` needs ``advantage_estimates`` and
    ``baseline_values`` the core's call site does not supply.
    """

    def __call__(
        self,
        forward_fn: object,
        batch: object,
        *,
        advantage_estimates: object,
        baseline_values: object,
    ) -> object:
        _ = (forward_fn, batch, advantage_estimates, baseline_values)
        return object()

    def declaration(self) -> object:
        return object()


class _PlainClass:
    """A class that is NOT a Protocol -- refusal control input."""

    def estimate(self, batch: object) -> object:
        return object()


@runtime_checkable
class _ZeroMemberProtocol(Protocol):
    """A runtime-checkable Protocol with zero checkable members."""


@runtime_checkable
class _PropertyDeclaringProtocol(Protocol):
    """A protocol whose ``budget`` member is declared as a ``@property``.

    The presence-only rule is a claim about the PROTOCOL side. A protocol
    that declares ``@property`` declares an attribute read, and an attribute
    read has no call signature to compare against -- so presence is the only
    honest check, and the report says so rather than inventing a comparison.
    """

    @property
    def budget(self) -> int: ...

    def estimate(self, batch: object) -> object: ...


class _PropertyDeclaringSubject:
    """Supplies the protocol's property member, with a wrong-typed value.

    ``budget`` returns a ``str`` where the protocol annotates ``int``. That
    is a real defect and mypy is the oracle for it; this module compares
    arity and parameter names, so it must NOT read as a structural mismatch
    here. Reporting it would be this checker claiming a scope it does not
    have.
    """

    @property
    def budget(self) -> str:
        return "unbounded"

    def estimate(self, batch: object) -> object:
        return object()


class _NonCallableCapabilitiesValueHead:
    """``capabilities`` supplied as a property where ``ValueHead`` declares a method.

    ``ValueHead.capabilities`` is invoked -- ``head.capabilities()`` -- so a
    subject offering a readable attribute of the right NAME satisfies
    ``isinstance`` and then raises ``TypeError: 'int' object is not
    callable`` at the first call. That is DECIDABLY wrong, so it is the
    mismatch arm, not the refusal arm.
    """

    def estimate(self, batch: object) -> list[list[float]]:
        return []

    @property
    def capabilities(self) -> int:
        return 42


class _TotallyEmpty:
    """A subject carrying none of any protocol's members.

    Used only where the assertion is about the DENOMINATOR. Every member is
    missing, so ``checked`` is forced to report the full set it intended to
    compare and cannot be flattered by a subject that happens to match.
    """


# Names that cannot be probed by omission because leaving them off -- or
# rather, putting a stand-in function in their place -- corrupts the probe
# CLASS itself rather than testing the protocol. None of them is declarable as
# a protocol member in any sense this module checks, so excluding them narrows
# the probe's mechanics and not its scope.
_PROBE_UNSAFE = frozenset(
    {"__class__", "__dict__", "__weakref__", "__slots__", "__init__", "__new__"}
)


def _isinstance_member_set(protocol: type[object]) -> set[str]:
    """Measure, BEHAVIOURALLY, which names a presence gate over ``protocol``
    requires.

    This is deliberately not a read of ``__protocol_attrs__``: that is the
    value ``structural.py`` itself borrows, and a test that reads the same
    attribute would agree with the implementation by construction and measure
    nothing. Instead each candidate name is probed by omission -- build a
    subject carrying every candidate EXCEPT one, and see whether ``isinstance``
    still says yes. A name whose absence flips the gate to False is, by
    definition, in the set the gate consults.

    Candidates come from the protocol's own class bodies and annotations, so a
    member that ``__protocol_attrs__`` were ever to omit would still be probed
    here and would still show up as load-bearing.

    ``LossFn`` is a Protocol but NOT ``@runtime_checkable`` (MEASURED), so
    ``isinstance`` against it raises rather than answering. It is probed
    through a runtime-checkable subclass, which recomputes the attribute set
    over the same mro. That is not a workaround for an awkward test: it is
    the reason ``structural_report`` accepts non-runtime-checkable protocols
    at all. ``LossFn`` has no runtime gate whatsoever, so a structural check
    is the ONLY structural question anyone can ask about it.
    """
    gate: type[object] = protocol
    if not getattr(protocol, "_is_runtime_protocol", False):
        gate = runtime_checkable(types.new_class("_RuntimeGate", (protocol, Protocol)))

    candidates: set[str] = set()
    for base in protocol.__mro__:
        if base is object or base.__module__ == "typing":
            continue
        candidates.update(vars(base).keys())
        candidates.update(getattr(base, "__annotations__", {}).keys())
    candidates -= set(vars(object).keys())
    candidates -= _PROBE_UNSAFE

    load_bearing: set[str] = set()
    for name in candidates:
        namespace = {other: (lambda self, *args, **kwargs: None) for other in candidates - {name}}
        probe = type("_OmissionProbe", (), namespace)()
        if not isinstance(probe, gate):
            load_bearing.add(name)
    return load_bearing


def _mismatch_text(report: StructuralReport) -> str:
    """Flatten mismatch entries so tests can assert on the named member."""
    return "\n".join(str(entry) for entry in report.mismatches)


# ---------------------------------------------------------------------------
# A. MUST_FIRE positive controls -- the whole point of the module.
# ---------------------------------------------------------------------------


def test_wrong_kwarg_value_head_passes_isinstance_but_fails_structure() -> None:
    """Prove the hole AND the fix for ValueHead: an ``estimate(features=...)``
    object passes ``isinstance(obj, ValueHead)`` (presence-only check, pinned
    here so the test goes red if runtime_checkable ever starts comparing
    signatures), but ``structural_report`` fires and names ``estimate``."""
    obj = _WrongKwargValueHead()

    assert isinstance(obj, ValueHead)  # pin the hole: presence-only, NEVER signatures

    report = structural_report(obj, ValueHead)

    assert not report.ok
    assert "estimate" in _mismatch_text(report)


def test_wrong_kwarg_rollout_source_passes_isinstance_but_fails_structure() -> None:
    """Same hole/fix shape for RolloutSource: ``generate(inputs=...)`` passes
    the isinstance gate, the structural report refuses and names ``generate``."""
    obj = _WrongKwargRolloutSource()

    assert isinstance(obj, RolloutSource)

    report = structural_report(obj, RolloutSource)

    assert not report.ok
    assert "generate" in _mismatch_text(report)


def test_wrong_kwarg_weight_sync_passes_isinstance_but_fails_structure() -> None:
    """Same hole/fix shape for WeightSync: ``sync(name_map=...)`` passes the
    isinstance gate, the structural report refuses and names ``sync``."""
    obj = _WrongKwargWeightSync()

    assert isinstance(obj, WeightSync)

    report = structural_report(obj, WeightSync)

    assert not report.ok
    assert "sync" in _mismatch_text(report)


# ---------------------------------------------------------------------------
# B. The measured real-world case: two extra REQUIRED keyword-only parameters.
# ---------------------------------------------------------------------------


def test_two_extra_required_keyword_only_params_are_a_mismatch() -> None:
    """Prove the LearnedValueAdvantageEstimation-vs-LossFn shape is refused:
    an implementation whose ``__call__`` adds ``advantage_estimates`` and
    ``baseline_values`` as required keyword-only parameters is not a
    compatible LossFn, because the core's call site cannot supply them."""
    report = structural_report(_LearnedAdvantageStyleLoss(), LossFn)

    assert not report.ok
    assert "__call__" in _mismatch_text(report)


# ---------------------------------------------------------------------------
# C. Compatible shapes must report ok.
# ---------------------------------------------------------------------------


def test_exact_signature_double_reports_ok() -> None:
    """Prove the checker is not a false-positive machine: a double with the
    exact protocol signature reports ok with zero mismatches."""
    report = structural_report(_ExactValueHead(), ValueHead)

    assert report.ok
    assert report.mismatches == [] or len(report.mismatches) == 0


def test_optional_keyword_parameter_addition_reports_ok() -> None:
    """Prove LISKOV-tightening passes: adding an OPTIONAL keyword-only
    parameter cannot break any call site the core can legally make."""
    report = structural_report(_OptionalKwargValueHead(), ValueHead)

    assert report.ok


def test_kwargs_catch_all_reports_ok() -> None:
    """Prove a ``**kwargs`` catch-all passes: it accepts every keyword the
    protocol's signature could ever be called with."""
    report = structural_report(_KwargsCatchAllValueHead(), ValueHead)

    assert report.ok


# ---------------------------------------------------------------------------
# D. Missing member is report-valued, never exception-valued.
# ---------------------------------------------------------------------------


def test_missing_member_is_a_report_valued_mismatch() -> None:
    """Prove partial outcomes stay report-valued: an object missing
    ``capabilities`` entirely yields a non-ok report naming ``capabilities``
    matched member instead of raising."""
    obj = _MissingCapabilities()

    assert not isinstance(obj, ValueHead)  # the one isinstance gate that DOES fire

    report = structural_report(obj, ValueHead)

    assert not report.ok
    assert "capabilities" in _mismatch_text(report)


# ---------------------------------------------------------------------------
# E. REFUSAL controls: questions the checker CANNOT answer must raise.
# ---------------------------------------------------------------------------


def test_non_protocol_subject_type_raises_structural_refusal() -> None:
    """Prove the checker refuses when the second argument is not a Protocol at
    all: there is no meaningful 'set of protocol members' to compare against,
    so the question cannot be answered and refusal is the honest outcome."""
    with pytest.raises(StructuralRefusal) as excinfo:
        structural_report(_ExactValueHead(), _PlainClass)

    assert "protocol" in str(excinfo.value).lower()


def test_zero_member_protocol_raises_structural_refusal() -> None:
    """Prove the empty-denominator refusal: a Protocol with zero checkable
    members would make ANY object 'structurally conform' vacuously, including
    ``object()`` -- a denominator of zero is a refusal, not a pass. The
    message must say so."""
    with pytest.raises(StructuralRefusal) as excinfo:
        structural_report(object(), _ZeroMemberProtocol)

    assert "empty" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# F. DENOMINATOR control: `checked` must be the exact member set.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "protocol",
    [LossFn, ValueHead, RolloutSource, WeightSync],
    ids=lambda p: p.__name__,
)
def test_checked_equals_the_member_set_isinstance_consults(
    protocol: type[object],
) -> None:
    """Prove the report's denominator EQUALS the runtime gate's, protocol by
    protocol -- the generic form of the control, and the one that catches a
    dropped member nobody thought to name.

    A per-protocol test asserting a hand-written member set only ever catches
    the members its author remembered. The first version of this module
    derived its denominator by walking ``__mro__`` and skipping ``_``-prefixed
    names, which dropped ``__call__`` -- so ``LossFn`` was compared on 1 of its
    2 members and a subject that could not be CALLED reported ``ok``. The
    oracle here is typing's own attribute set, computed the same way
    ``isinstance`` computes it, so a future member of any of these protocols
    is in this denominator the moment it is declared.
    """
    expected = _isinstance_member_set(protocol)
    assert expected, f"{protocol.__name__} exposes no protocol attributes to compare"

    report = structural_report(_TotallyEmpty(), protocol)

    assert set(report.checked) == expected


def test_call_is_in_the_loss_fn_denominator() -> None:
    """Pin the specific member the original denominator dropped. The generic
    control above would go red if ``__call__`` vanished again, but it would
    not SAY so; this one names it, because ``__call__`` is the member every
    objective protocol in this plane is built around."""
    assert "__call__" in _isinstance_member_set(LossFn)
    assert "__call__" in set(structural_report(_TotallyEmpty(), LossFn).checked)


def test_checked_is_exactly_the_value_head_member_set() -> None:
    """Prove no member is silently dropped from the comparison: ``checked``
    for ValueHead is exactly {estimate, capabilities}. If a member is ever
    skipped, this test -- not a silent green setup -- goes red."""
    report = structural_report(_ExactValueHead(), ValueHead)

    assert set(report.checked) == {"estimate", "capabilities"}


# ---------------------------------------------------------------------------
# G. Property members: present in `checked`, presence-checked only.
# ---------------------------------------------------------------------------


def test_protocol_declared_property_is_presence_checked_only() -> None:
    """Prove a PROTOCOL-declared property sits in the denominator but is
    checked for PRESENCE ONLY: the subject's ``budget`` returns a ``str``
    where the protocol annotates ``int`` -- a real defect, and mypy's to
    report -- yet this report stays green on that member, because an
    attribute read has no call signature to compare. Widening here would be
    the checker claiming a scope it does not have."""
    report = structural_report(_PropertyDeclaringSubject(), _PropertyDeclaringProtocol)

    assert "budget" in set(report.checked)
    assert "budget" not in _mismatch_text(report)
    assert report.ok


def test_non_callable_where_protocol_declares_a_method_is_a_mismatch() -> None:
    """Prove the DECIDABLE-wrong case is reported as a mismatch and never
    laundered into a refusal. ``ValueHead`` declares ``capabilities`` as a
    method, so it is called; a subject offering a readable int of that name
    passes ``isinstance`` and dies with ``TypeError`` at the first call.
    That question was asked and answered -- the answer is no -- so it must
    come back as red, not as CANNOT-MEASURE."""
    report = structural_report(_NonCallableCapabilitiesValueHead(), ValueHead)

    assert not report.ok
    assert "capabilities" in set(report.checked)
    text = _mismatch_text(report)
    assert "capabilities" in text
    assert "non-callable" in text
    assert "int" in text


# ---------------------------------------------------------------------------
# H. The real pair (#396): two protocols isinstance CANNOT separate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subject_name", "protocol", "expected_ok"),
    [
        ("LearnedValueAdvantageEstimation", AdvantageFn, False),
        ("LearnedValueAdvantageEstimation", TemporalAdvantageFn, True),
        ("GroupNormalisedAdvantage", AdvantageFn, True),
        ("GroupNormalisedAdvantage", TemporalAdvantageFn, False),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else str(v),
)
def test_real_advantage_pair_separated_in_all_four_cells(
    subject_name: str,
    protocol: type[object],
    expected_ok: bool,
) -> None:
    """The MUST_FIRE control, on shipped code rather than on doubles.

    ``AdvantageFn`` and ``TemporalAdvantageFn`` declare the SAME single
    member name -- ``compute`` -- and differ only in its parameters
    (MEASURED: ``__protocol_attrs__`` is ``['compute']`` for both). So
    ``isinstance`` reads True in all FOUR cells of this 2x2 and carries zero
    information about which contract a subject actually meets. That is not a
    hypothetical: it is #396.

    ``structural_report`` must get all four right, and the two red cells are
    red for OPPOSITE reasons -- one subject demands parameters the protocol
    never supplies, the other lacks parameters the protocol does supply.
    A checker that only ever tested one direction would pass half of this.
    """
    subjects = {
        "LearnedValueAdvantageEstimation": LearnedValueAdvantageEstimation(),
        "GroupNormalisedAdvantage": GroupNormalisedAdvantage(),
    }
    subject = subjects[subject_name]

    # The premise: isinstance really is blind here. If this ever goes red the
    # protocols have changed shape and the test above it is measuring
    # something else -- so it is asserted, not assumed.
    assert isinstance(subject, protocol)

    report = structural_report(subject, protocol, origin=subject_name)

    assert report.ok is expected_ok
    assert set(report.checked) == {"compute"}
    if not expected_ok:
        assert "compute" in _mismatch_text(report)


def test_the_two_red_cells_fail_in_opposite_directions() -> None:
    """Name the asymmetry the parametrised test only implies.

    Substitution direction matters: an implementation must ACCEPT every call
    the protocol permits. ``LearnedValueAdvantageEstimation`` fails against
    ``AdvantageFn`` because it REQUIRES ``values``/``terminated`` that an
    ``AdvantageFn`` caller will never pass. ``GroupNormalisedAdvantage``
    fails against ``TemporalAdvantageFn`` because it cannot accept those same
    two parameters at all. Same two names, opposite defects -- and the
    messages must say which is which, or the report cannot be acted on.
    """
    extra_required = structural_report(LearnedValueAdvantageEstimation(), AdvantageFn, origin="lv")
    missing_param = structural_report(GroupNormalisedAdvantage(), TemporalAdvantageFn, origin="gn")

    assert "protocol declares no parameter named" in _mismatch_text(extra_required)
    assert "has no parameter named" in _mismatch_text(missing_param)


# ---------------------------------------------------------------------------
# I. POSITIONAL_ONLY is matched by POSITION (#397).
#
# ``ValueHead.estimate`` declares its parameter positional-only, and the ``/``
# is a contract term: it states that the NAME is not part of the seam, so no
# caller may write ``estimate(batch=...)`` and an implementation may name the
# parameter anything. A comparator that looked such a parameter up BY NAME --
# as this module correctly does for every other kind -- would refuse a
# conforming adapter for spelling an unspeakable name differently. These rows
# pin that rule from both sides.
# ---------------------------------------------------------------------------


@runtime_checkable
class _PosOnlyProto(Protocol):
    def estimate(self, batch: int, /) -> int: ...


@runtime_checkable
class _PosOrKwProto(Protocol):
    def estimate(self, batch: int) -> int: ...


class _SubjRenamed:
    def estimate(self, _batch: int) -> int:  # noqa: ARG002
        return 1


class _SubjPosOnly:
    def estimate(self, anything: int, /) -> int:  # noqa: ARG002
        return 1


class _SubjVarArgs:
    def estimate(self, *args: int) -> int:  # noqa: ARG002
        return 1


class _SubjOptionalExtra:
    def estimate(self, b: int, scale: float = 1.0) -> int:  # noqa: ARG002
        return 1


class _SubjNoSlot:
    def estimate(self) -> int:
        return 1


class _SubjRequiredExtra:
    def estimate(self, b: int, scale: float) -> int:  # noqa: ARG002
        return 1


class _SubjKwOnly:
    def estimate(self, *, batch: int) -> int:  # noqa: ARG002
        return 1


@pytest.mark.parametrize(
    ("subject_cls", "protocol", "expected"),
    [
        pytest.param(
            _SubjRenamed,
            _PosOnlyProto,
            True,
            id="posonly-renamed-accepts",
        ),
        pytest.param(
            _SubjPosOnly,
            _PosOnlyProto,
            True,
            id="posonly-posonly-impl-accepts",
        ),
        pytest.param(
            _SubjVarArgs,
            _PosOnlyProto,
            True,
            id="posonly-varargs-accepts",
        ),
        pytest.param(
            _SubjOptionalExtra,
            _PosOnlyProto,
            True,
            id="posonly-optional-extra-accepts",
        ),
        pytest.param(
            _SubjNoSlot,
            _PosOnlyProto,
            False,
            id="posonly-no-slot-must-refuse",
        ),
        pytest.param(
            _SubjRequiredExtra,
            _PosOnlyProto,
            False,
            id="posonly-required-extra-must-refuse",
        ),
        pytest.param(
            _SubjKwOnly,
            _PosOnlyProto,
            False,
            id="posonly-kwonly-must-refuse",
        ),
        pytest.param(
            _SubjRenamed,
            _PosOrKwProto,
            False,
            id="posorkw-renamed-must-refuse",
        ),
    ],
)
def test_positional_only_is_matched_by_position_not_name(
    subject_cls: type[object],
    protocol: type[object],
    expected: bool,
) -> None:
    """Positional-only params bind by position; under `/` the name is free.

    The four must-fire rows are present because a rule that only ever
    accepts is indistinguishable from no rule at all: the permissive rows
    show the check does not over-restrict, the refusing rows show arity and
    positional acceptance still bind, and together they form one instrument.
    """
    report = structural_report(subject_cls(), protocol)
    assert report.ok == expected
    if not expected:
        assert "estimate" in " ".join(report.mismatches)


# ---------------------------------------------------------------------------
# J. Denominator reconciliation: static methods, assigned values, and
#    annotation-only members reach the collector through DIFFERENT arms, and
#    each arm is a distinct place for the old denominator hole to regrow.
# ---------------------------------------------------------------------------


@runtime_checkable
class _StaticParseProto(Protocol):
    """A protocol whose only member is a ``@staticmethod``.

    In the class body the member is a ``staticmethod`` descriptor, not a
    plain function, so the collector must unwrap ``__func__`` before it can
    introspect anything. If that unwrap were dropped, the collector would
    either refuse an answerable question or compare the descriptor's surface
    instead of the declared function's -- this row exercises the unwrap arm
    directly.
    """

    @staticmethod
    def parse(raw: object) -> object: ...


class _SubjMatchingStaticParse:
    """Static-shaped subject whose ``parse`` mirrors the protocol exactly."""

    @staticmethod
    def parse(raw: object) -> object:
        return raw


@runtime_checkable
class _AssignedValueProto(Protocol):
    """Declares ``threshold`` by ASSIGNMENT, plus one method.

    An assigned class-body value is callable-checked before signature
    introspection: a float is not callable, so it must be recorded as a
    presence-only member. Skipping it instead would leave ``checked``
    narrower than the set ``isinstance`` consults -- the original
    denominator hole wearing a different hat, as the collector's own comment
    warns.
    """

    threshold: float = 0.5

    def estimate(self, batch: object) -> object: ...


class _SubjHasThreshold:
    threshold: float = 0.9

    def estimate(self, batch: object) -> object:
        return batch


class _SubjNoThreshold:
    def estimate(self, batch: object) -> object:
        return batch


@runtime_checkable
class _AnnotationOnlyProto(Protocol):
    """``tokens`` is annotated but never assigned, so NO base's vars() holds it.

    Only the reconciliation sweep after the vars() walk can see this member.
    If that sweep were dropped, ``checked`` would silently shrink below the
    set ``isinstance`` checks, and a subject missing ``tokens`` would report
    ``ok`` over a denominator the runtime gate itself would reject.
    """

    tokens: int

    def estimate(self, batch: object) -> object: ...


class _SubjHasTokens:
    tokens: int = 3

    def estimate(self, batch: object) -> object:
        return batch


class _SubjNoTokens:
    def estimate(self, batch: object) -> object:
        return batch


def test_staticmethod_protocol_member_is_unwrapped_and_compared() -> None:
    """Prove the staticmethod unwrap arm yields a comparable signature: an
    exactly-matching static subject must report ok over the full member set,
    which only happens if the collector introspected the declared function
    rather than the descriptor."""
    report = structural_report(_SubjMatchingStaticParse(), _StaticParseProto)

    assert report.ok
    assert set(report.checked) == {"parse"}


@pytest.mark.parametrize(
    ("subject_cls", "protocol", "member", "expected_ok"),
    [
        pytest.param(
            _SubjHasThreshold,
            _AssignedValueProto,
            "threshold",
            True,
            id="assigned-value-present",
        ),
        pytest.param(
            _SubjNoThreshold,
            _AssignedValueProto,
            "threshold",
            False,
            id="assigned-value-absent-must-refuse",
        ),
        pytest.param(
            _SubjHasTokens,
            _AnnotationOnlyProto,
            "tokens",
            True,
            id="annotation-only-present",
        ),
        pytest.param(
            _SubjNoTokens,
            _AnnotationOnlyProto,
            "tokens",
            False,
            id="annotation-only-absent-must-refuse",
        ),
    ],
)
def test_presence_only_members_are_presence_checked(
    subject_cls: type[object],
    protocol: type[object],
    member: str,
    expected_ok: bool,
) -> None:
    """Each non-callable declaration arm is pinned from BOTH sides.

    The present rows keep the members inside ``checked`` while reporting ok,
    proving they were measured rather than skipped; the absent rows prove an
    omitted presence-only member is a report-valued mismatch naming the
    member, never a silent pass over a shrunken denominator. Only pairs like
    this can catch a collector arm that degenerated into unconditionally
    recording presence.
    """
    report = structural_report(subject_cls(), protocol)

    assert member in set(report.checked)
    assert report.ok is expected_ok
    if not expected_ok:
        text = _mismatch_text(report)
        assert f"member '{member}'" in text
        assert "presence-only" in text
        assert "subject offers no attribute" in text


# ---------------------------------------------------------------------------
# K. Borrowing typing's denominator: the legacy reader and the no-reader
#    refusal. Both are runtime-version arms, simulated with monkeypatch so
#    they can be exercised on a 3.12+ interpreter that also supports
#    3.10/3.11 deployments.
# ---------------------------------------------------------------------------


@runtime_checkable
class _LegacyAttrsProto(Protocol):
    def estimate(self, batch: object) -> object: ...


class _SubjLegacyEstimate:
    def estimate(self, batch: object) -> object:
        return batch


def _force_legacy_reader(monkeypatch: pytest.MonkeyPatch, protocol: type) -> None:
    """Make the module's reader fall through to ``typing._get_protocol_attrs``.

    Simulating "this runtime has no ``__protocol_attrs__``" is version-specific,
    and both obvious spellings are wrong on one of the CI legs. MEASURED on
    3.10.20 / 3.11.15 / 3.12.13:

    * 3.10 / 3.11 -- the attribute does not exist at all, so the fallback is
      ALREADY the live path and nothing needs injecting. Assigning ``None``
      actively breaks the row: ``__protocol_attrs__`` is not in
      ``typing._get_protocol_attrs``'s exclusion list on those versions, so the
      legacy reader counts the injected attribute AS A PROTOCOL MEMBER and the
      denominator becomes ``['__protocol_attrs__', 'estimate']`` -- exactly the
      thing this row exists to prove unchanged.
    * 3.12 -- the attribute is real (``{'estimate'}``). Assigning ``None`` is
      clean, and the legacy reader still answers ``['estimate']``. DELETING it
      is not clean: the empty ``Protocol.__protocol_attrs__`` is then inherited,
      so the reader sees ``set()`` rather than ``None``, never falls through,
      and the denominator silently collapses to zero members.

    So: inject only where the attribute is present. The
    ``set(report.checked) == {"estimate"}`` assertion downstream is what keeps
    either failure mode visible rather than merely green.
    """
    if hasattr(protocol, "__protocol_attrs__"):
        monkeypatch.setattr(protocol, "__protocol_attrs__", None)


def test_legacy_get_protocol_attrs_reader_produces_the_same_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hide ``__protocol_attrs__`` to force the 3.10/3.11 reader path.

    ``__protocol_attrs__`` is public only from 3.12; the fallback is the SAME
    computation typing performs for ``isinstance``, so substituting it must
    change nothing about the answer. If the fallback returned an empty or
    partial set, this row would either hit the zero-member refusal or pass
    over a measured denominator of fewer than one member."""
    _force_legacy_reader(monkeypatch, _LegacyAttrsProto)

    report = structural_report(
        _SubjLegacyEstimate(),
        _LegacyAttrsProto,
        origin="legacy-fallback",
    )

    assert report.ok
    assert set(report.checked) == {"estimate"}


def test_runtime_with_no_protocol_attrs_reader_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the neither-reader refusal: deriving the denominator independently
    was the bug this module was repaired for, so a runtime exposing no reader
    at all must raise -- and the message must name BOTH readers it looked
    for, so the deployer knows which capability is missing."""
    _force_legacy_reader(monkeypatch, _LegacyAttrsProto)
    monkeypatch.setattr(typing, "_get_protocol_attrs", None, raising=False)

    with pytest.raises(StructuralRefusal) as excinfo:
        structural_report(_SubjLegacyEstimate(), _LegacyAttrsProto, origin="no-reader")

    message = str(excinfo.value)
    assert "no-reader" in message
    assert "__protocol_attrs__" in message
    assert "typing._get_protocol_attrs" in message


# ---------------------------------------------------------------------------
# L. CANNOT-MEASURE arms: only unanswerable questions may raise, and the
#    AttributeError / anything-else distinction is the whole rule. A dynamic
#    member SERVED by ``__getattr__`` is present; a read raising anything
#    other than AttributeError is a refusal, never a mismatch.
# ---------------------------------------------------------------------------


class _SubjDynamicBudget:
    """Exposes ``budget`` only through ``__getattr__`` -- a genuinely dynamic member."""

    def estimate(self, batch: object) -> object:
        return batch

    def __getattr__(self, name: str) -> object:
        if name == "budget":
            return 7
        raise AttributeError(name)


def test_dynamically_exposed_property_member_counts_as_present() -> None:
    """A static scan cannot see a ``__getattr__``-served member, so the checker
    falls through to a real ``getattr`` -- and a served member must count as
    PRESENT. If the dynamic fallback were dropped, this green row would
    invert into a false missing-member mismatch on ``budget``."""
    report = structural_report(_SubjDynamicBudget(), _PropertyDeclaringProtocol)

    assert report.ok
    assert set(report.checked) == {"budget", "estimate"}


class _SubjBudgetAccessExplodes:
    """``budget`` raises something that is NOT AttributeError when read dynamically."""

    def estimate(self, batch: object) -> object:
        return batch

    def __getattr__(self, name: str) -> object:
        raise RuntimeError(f"budget ledger unavailable: {name}")


def test_non_attribute_error_during_presence_check_is_a_refusal() -> None:
    """AttributeError means ABSENT; anything else means CANNOT MEASURE. A
    crashing attribute read must refuse loudly rather than flatten into a
    missing-member mismatch, because the checker observed nothing about the
    member -- collapsing the two would launder an abstention into a finding."""
    with pytest.raises(StructuralRefusal) as excinfo:
        structural_report(
            _SubjBudgetAccessExplodes(),
            _PropertyDeclaringProtocol,
            origin="exploding-budget",
        )

    message = str(excinfo.value)
    assert "cannot inspect subject member 'budget'" in message
    assert "RuntimeError" in message


@runtime_checkable
class _RunSpeedProto(Protocol):
    def run(self, speed: int) -> int: ...


class _SubjRunGetterExplodes:
    """``run`` is a property whose getter raises -- the read itself is unmeasurable."""

    @property
    def run(self) -> object:
        raise RuntimeError("run registry down")


def test_non_attribute_error_during_callable_lookup_is_a_refusal() -> None:
    """Same CANNOT-MEASURE rule on the callable arm: reading the candidate
    method raises during ``getattr``, so the checker cannot classify the
    member and must refuse -- it may not guess "absent" and may not guess
    "non-callable", both of which are decidable states this object is not in."""
    with pytest.raises(StructuralRefusal) as excinfo:
        structural_report(_SubjRunGetterExplodes(), _RunSpeedProto, origin="getter-explodes")

    assert "attribute access raised RuntimeError" in str(excinfo.value)


class _SubjRunBadSignature:
    """A callable ``run`` carrying a corrupt ``__signature__``.

    ``inspect.signature`` raises ``TypeError`` when ``__signature__`` is not
    a Signature instance. The member is present and callable, yet its surface
    cannot be read: the ONE configuration in which a callable subject member
    still earns a refusal instead of a mismatch.
    """

    def run(self, speed: int) -> int:
        return speed


_SubjRunBadSignature.run.__signature__ = "not-a-signature"  # type: ignore[attr-defined]


def test_callable_subject_member_with_corrupt_signature_is_a_refusal() -> None:
    """Pin the refusal wording: it must name the subject member and the
    introspection failure, so an operator can tell a broken adapter apart
    from a structurally wrong one."""
    # MEASURED: which exception `inspect.signature` raises for a corrupt
    # `__signature__` is interpreter-dependent -- TypeError on 3.10/3.11/3.14,
    # ValueError on 3.12. Ask the interpreter rather than pinning one version's
    # answer, which is what would have taken the 3.12 CI leg RED.
    with pytest.raises((ValueError, TypeError)) as sigexc:
        inspect.signature(_SubjRunBadSignature().run)
    expected = type(sigexc.value).__name__

    with pytest.raises(StructuralRefusal) as excinfo:
        structural_report(_SubjRunBadSignature(), _RunSpeedProto, origin="corrupt-sig")

    message = str(excinfo.value)
    assert "cannot introspect subject member 'run'" in message
    assert f"raised {expected}" in message


@runtime_checkable
class _CorruptSigMemberProto(Protocol):
    def estimate(self, batch: object) -> object: ...


_CorruptSigMemberProto.estimate.__signature__ = "not-a-signature"  # type: ignore[attr-defined]


def test_corrupt_protocol_member_signature_is_a_refusal() -> None:
    """The denominator side of the same rule: when a protocol-declared
    callable cannot be introspected, the comparison surface itself is
    undefined, so the refusal must name the PROTOCOL member rather than blame
    or flatter the subject."""
    with pytest.raises((ValueError, TypeError)) as sigexc:
        inspect.signature(_CorruptSigMemberProto.estimate)
    expected = type(sigexc.value).__name__

    with pytest.raises(StructuralRefusal) as excinfo:
        structural_report(_SubjLegacyEstimate(), _CorruptSigMemberProto, origin="corrupt-proto")

    message = str(excinfo.value)
    assert "cannot introspect protocol member 'estimate'" in message
    assert f"raised {expected}" in message


# ---------------------------------------------------------------------------
# M. Kind and requiredness mismatches inside _compare_callable_signatures:
#    same-name-different-kind, optional-by-name vs required, and the
#    defaulted positional-only slot. Each branch renders its own sentence,
#    and each test pins the exact rendered wording.
# ---------------------------------------------------------------------------


class _SubjSpeedKwOnly:
    """Offers ``speed`` keyword-only where the protocol declares positional-or-keyword."""

    def run(self, *, speed: int) -> int:
        return speed


@runtime_checkable
class _DefaultedSpeedProto(Protocol):
    def run(self, speed: int = 1) -> int: ...


class _SubjSpeedRequired:
    """Same name and kind as the defaulted protocol parameter, but with NO default."""

    def run(self, speed: int) -> int:
        return speed


@runtime_checkable
class _OptionalChunkProto(Protocol):
    def read(self, chunk: int = 8, /) -> int: ...


class _SubjChunkRequired:
    """Occupies the defaulted positional-only slot with a REQUIRED parameter."""

    def read(self, chunk: int) -> int:
        return chunk


@pytest.mark.parametrize(
    ("subject_cls", "protocol", "fragment"),
    [
        pytest.param(
            _SubjSpeedKwOnly,
            _RunSpeedProto,
            "protocol declares POSITIONAL_OR_KEYWORD; subject offers KEYWORD_ONLY",
            id="same-name-different-kind",
        ),
        pytest.param(
            _SubjSpeedRequired,
            _DefaultedSpeedProto,
            "protocol declares an optional POSITIONAL_OR_KEYWORD parameter "
            "(a call may omit it); subject requires that parameter",
            id="optional-by-name-required-in-subject",
        ),
        pytest.param(
            _SubjChunkRequired,
            _OptionalChunkProto,
            "protocol declares an optional POSITIONAL_ONLY parameter "
            "(a call may omit it); subject requires the parameter at that position ('chunk')",
            id="optional-positional-only-required-in-subject",
        ),
    ],
)
def test_kind_and_requiredness_mismatches_are_named_exactly(
    subject_cls: type[object],
    protocol: type[object],
    fragment: str,
) -> None:
    """Three red rows for three distinct comparator branches.

    Each row is exactly one mismatch -- any leaked second finding fails the
    length assertion -- and the asserted fragment is the full rendered
    sentence of its branch. That disciplines the wording: two branches
    collapsing into one shared message would read identically here, and
    identical wording is precisely what makes a report unactionable at
    triage time.
    """
    report = structural_report(subject_cls(), protocol)

    assert not report.ok
    assert len(report.mismatches) == 1
    assert fragment in _mismatch_text(report)
