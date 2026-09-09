"""Adversarial tests for the value-head handshake: claims, estimates, and the seam."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import pytest

from foundationscale.rl.interfaces import ExperienceBatch
from foundationscale.rl.value_head import (
    ValueCapabilities,
    ValueCapabilityRefusal,
    ValueEstimateRefusal,
    ValueHead,
    check_value_capabilities,
    verify_estimates,
)


@dataclass(frozen=True, slots=True)
class _FakeBatch:
    """A plain stand-in carrying only what verify_estimates reads: columns."""

    columns: dict[str, tuple[tuple[int, ...], ...]]


def _make_batch(columns: dict[str, tuple[tuple[int, ...], ...]]) -> ExperienceBatch:
    return cast(ExperienceBatch, _FakeBatch(columns=columns))


def _capabilities(
    *,
    granularity: Literal["token", "sequence"] = "token",
    shares_policy_trunk: bool = False,
    reports_old_values: bool = False,
) -> ValueCapabilities:
    return ValueCapabilities(
        granularity=granularity,
        shares_policy_trunk=shares_policy_trunk,
        reports_old_values=reports_old_values,
    )


# --- ValueCapabilities: the claim half, refused at construction ------------


@pytest.mark.parametrize("bad", ["tok", "", "TOKEN", 1, None])
def test_granularity_refused_outside_the_two_literals(bad: object) -> None:
    # "TOKEN" is case-sensitively wrong: a mutation that casefolded the check
    # dies here. The message must NAME the offending value and both literals.
    with pytest.raises(ValueCapabilityRefusal) as exc_info:
        ValueCapabilities(granularity=bad, shares_policy_trunk=False, reports_old_values=False)
    message = str(exc_info.value)
    assert f"granularity={bad!r}" in message
    assert "'token' or 'sequence'" in message


@pytest.mark.parametrize("field", ["shares_policy_trunk", "reports_old_values"])
@pytest.mark.parametrize("bad", [0, 1])
def test_capability_flag_refuses_an_int_standing_in_for_a_bool(field: str, bad: object) -> None:
    # A truthy 1 read as a capability would launder an unmeasured claim into a
    # declared one, so the int is refused and the reason must be in the
    # message. Passing 0 as well pins that FALSY ints are refused too -- a
    # truthiness-only check would silently admit False-as-0.
    declared: dict[str, object] = {
        "shares_policy_trunk": False,
        "reports_old_values": False,
        field: bad,
    }
    with pytest.raises(ValueCapabilityRefusal) as exc_info:
        ValueCapabilities(granularity="token", **declared)
    message = str(exc_info.value)
    assert f"{field}={bad!r}" in message
    assert "1 is not True" in message
    assert "launder an unmeasured claim into a declared one" in message


def test_both_literals_and_real_bools_are_admitted() -> None:
    token = _capabilities(granularity="token", reports_old_values=True)
    sequence = _capabilities(granularity="sequence", shares_policy_trunk=True)
    assert token.granularity == "token"
    assert token.reports_old_values is True
    assert sequence.granularity == "sequence"
    assert sequence.shares_policy_trunk is True


# --- check_value_capabilities: four refusals, then the consumed tuple ------


@pytest.mark.parametrize("bad", ["tok", "", "token ", 1, None])
def test_check_refuses_a_required_granularity_outside_the_literals(bad: object) -> None:
    with pytest.raises(ValueCapabilityRefusal) as exc_info:
        check_value_capabilities(
            required_granularity=bad,
            needs_old_values=False,
            capabilities=_capabilities(),
        )
    message = str(exc_info.value)
    assert f"required_granularity={bad!r}" in message
    assert "checked against nothing" in message


def test_check_refuses_a_truthy_int_for_needs_old_values() -> None:
    # The head DOES report old values, so if the bool check were weakened to
    # truthiness this call would pass -- the refusal must come from the check
    # itself, and the message says so in its own words.
    with pytest.raises(ValueCapabilityRefusal) as exc_info:
        check_value_capabilities(
            required_granularity="token",
            needs_old_values=1,
            capabilities=_capabilities(reports_old_values=True),
        )
    message = str(exc_info.value)
    assert "needs_old_values=1" in message
    assert "1 is not True" in message
    assert "launder an unmeasured need into a declared one" in message


def test_check_refuses_a_sequence_head_where_tokens_are_required() -> None:
    with pytest.raises(ValueCapabilityRefusal) as exc_info:
        check_value_capabilities(
            required_granularity="token",
            needs_old_values=False,
            capabilities=_capabilities(granularity="sequence"),
            origin="critic-a",
        )
    message = str(exc_info.value)
    assert "critic-a offers sequence-granularity values" in message
    assert "required_granularity='token'" in message
    assert "fabricated per-token measurement" in message


def test_check_refuses_an_old_value_need_the_head_does_not_report() -> None:
    with pytest.raises(ValueCapabilityRefusal) as exc_info:
        check_value_capabilities(
            required_granularity="sequence",
            needs_old_values=True,
            capabilities=_capabilities(granularity="sequence"),
            origin="critic-b",
        )
    message = str(exc_info.value)
    assert "critic-b does not report old values" in message
    assert "needs_old_values=True" in message
    assert "clipping reference" in message


@pytest.mark.parametrize(
    ("needs_old_values", "expected"),
    [
        (False, ("granularity",)),
        (True, ("granularity", "reports_old_values")),
    ],
)
def test_consumed_tuple_drops_reports_old_values_exactly_when_unneeded(
    needs_old_values: bool, expected: tuple[str, ...]
) -> None:
    # The head's CLAIM is held at True on both legs, so the short tuple proves
    # the drop is keyed on the NEED, not the claim: a mutation reading
    # capabilities.reports_old_values dies on the False leg, a mutation
    # returning a constant tuple dies on one leg or the other.
    consumed = check_value_capabilities(
        required_granularity="token",
        needs_old_values=needs_old_values,
        capabilities=_capabilities(reports_old_values=True),
    )
    assert consumed == expected


@pytest.mark.parametrize("shared", [False, True])
def test_shares_policy_trunk_is_declared_and_never_consumed(shared: bool) -> None:
    # Declared for the weight-sync plane and consumed by NEITHER function:
    # the returned tuple is identical for both flag values and never carries
    # the field, which keeps the declared field honest rather than dead.
    consumed = check_value_capabilities(
        required_granularity="token",
        needs_old_values=True,
        capabilities=_capabilities(shares_policy_trunk=shared, reports_old_values=True),
    )
    assert "shares_policy_trunk" not in consumed
    assert consumed == ("granularity", "reports_old_values")


def test_a_token_head_can_serve_a_sequence_requirement() -> None:
    # The granularity refusal is one-directional by design: only "token
    # required against a sequence head" is refused. A mutation making the
    # check symmetric dies here.
    consumed = check_value_capabilities(
        required_granularity="sequence",
        needs_old_values=False,
        capabilities=_capabilities(granularity="token"),
    )
    assert consumed == ("granularity",)


# --- verify_estimates: the measured count, exact on a hand-built batch -----


def test_supervised_count_is_exact_on_a_ragged_but_legal_batch() -> None:
    # Mask (1, 0, 1), (2, 1): ragged rows, and the truthy int 2 pins that the
    # mask is read by TRUTHINESS -- a mutation comparing to 1 literally would
    # skip the position and return 3. The int estimate 1 pins coercion.
    batch = _make_batch({"mask": ((1, 0, 1), (2, 1))})
    estimates = ((0.5, 0.0, 0.25), (1, -2.5))
    assert verify_estimates(estimates, batch, mask_column="mask") == 4


@pytest.mark.parametrize("garbage", [float("nan"), float("inf"), float("-inf")])
def test_a_masked_position_is_not_graded_so_a_non_finite_value_passes(
    garbage: float,
) -> None:
    # THE mask claim: masked positions are never coerced, never compared, and
    # sit in no denominator. A mutation that dropped the mask `continue` dies
    # here on the first non-finite leg; the exact count pins the denominator.
    batch = _make_batch({"mask": ((1, 0, 1), (0, 1))})
    estimates = ((0.5, garbage, 0.25), (garbage, -1.0))
    assert verify_estimates(estimates, batch, mask_column="mask", origin="critic") == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_value_at_a_supervised_position_is_refused(bad: float) -> None:
    # Placed at row 1 position 1 so both coordinates must be named correctly;
    # a mutation hard-coding 0 dies on the coordinate assertion.
    batch = _make_batch({"mask": ((1,), (1, 1))})
    estimates = ((0.0,), (1.0, bad))
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates(estimates, batch, mask_column="mask", origin="critic")
    message = str(exc_info.value)
    assert f"estimates[1][1]={bad!r}" in message
    assert "NaN and inf are not value estimates" in message


@pytest.mark.parametrize("bad", [True, False])
def test_a_bool_at_a_supervised_position_is_refused_before_float(bad: bool) -> None:
    # bool is checked first precisely because isinstance(True, int) is True;
    # a mutation dropping the bool branch would let float(True) == 1.0 stand
    # in for a measured value and count it.
    batch = _make_batch({"mask": ((1, 1),)})
    estimates = ((0.5, bad),)
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates(estimates, batch, mask_column="mask")
    message = str(exc_info.value)
    assert f"estimates[0][1]={bad!r}" in message
    assert "is not a value estimate" in message


def test_an_uncoercible_value_at_a_supervised_position_is_refused() -> None:
    batch = _make_batch({"mask": ((1,),)})
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates((("wide",),), batch, mask_column="mask")
    message = str(exc_info.value)
    assert "estimates[0][0]='wide'" in message
    assert "readable via float(raw)" in message


@pytest.mark.parametrize("bad", [42, None])
def test_a_non_iterable_estimates_return_is_refused_with_its_type_named(
    bad: object,
) -> None:
    batch = _make_batch({"mask": ((1,),)})
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates(bad, batch, mask_column="mask", origin="critic")
    message = str(exc_info.value)
    assert f"type {type(bad).__name__!r}" in message
    assert "not iterable" in message


def test_row_count_disagreement_is_refused_with_both_counts_named() -> None:
    # One estimate row against two mask rows is an estimate of a different
    # batch; both counts must appear or the caller cannot reconcile the two.
    batch = _make_batch({"mask": ((1,), (1,))})
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates(((0.1,),), batch, mask_column="mask", origin="critic")
    message = str(exc_info.value)
    assert "critic returned 1 estimate rows" in message
    assert "'mask' has 2 rows" in message


def test_row_length_disagreement_is_refused_with_index_and_lengths_named() -> None:
    batch = _make_batch({"mask": ((1, 1, 1),)})
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates(((0.1, 0.2),), batch, mask_column="mask")
    message = str(exc_info.value)
    assert "estimates[0] has 2 positions" in message
    assert "row 0 has 3" in message


def test_an_unsized_estimate_row_is_refused_with_the_row_index_named() -> None:
    # The row-count check passes (one row, one mask row), so this refusal can
    # only come from the len() guard inside the loop.
    batch = _make_batch({"mask": ((1,),)})
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates((5,), batch, mask_column="mask")
    message = str(exc_info.value)
    assert "estimates[0]" in message
    assert "not a sized row" in message


def test_zero_supervised_positions_are_refused_never_returned_as_zero() -> None:
    # Every mask entry is 0 and every estimate row is length-legal, so the
    # counts all agree -- yet nothing was measured. UNMEASURED is not zero: a
    # mutation returning the 0 dies because the call raises.
    batch = _make_batch({"mask": ((0, 0), (0,))})
    estimates = ((0.5, 0.6), (0.7,))
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates(estimates, batch, mask_column="mask", origin="critic")
    message = str(exc_info.value)
    assert "0 supervised positions" in message
    assert "across 2 rows" in message
    assert "UNMEASURED is not zero" in message


def test_a_missing_mask_column_is_refused_with_the_carried_columns_named() -> None:
    batch = _make_batch({"not_the_mask": ((1,),)})
    with pytest.raises(ValueEstimateRefusal) as exc_info:
        verify_estimates(((0.1,),), batch, mask_column="mask")
    message = str(exc_info.value)
    assert "mask_column='mask' is absent" in message
    assert "('not_the_mask',)" in message


def test_a_generator_of_rows_is_a_legal_estimates_return() -> None:
    # Only iterability is required of estimates; a mutation demanding a sized
    # Sequence at the boundary dies here.
    batch = _make_batch({"mask": ((1, 1),)})
    estimates = iter(((0.0, 1.0),))
    assert verify_estimates(estimates, batch, mask_column="mask") == 2


# --- the ValueHead seam: presence-checked, signature-blind -----------------


@dataclass(frozen=True, slots=True)
class _HonestHead:
    """A conforming fake: right method names, right call shape, no torch."""

    def estimate(self, _batch: ExperienceBatch) -> Sequence[Sequence[float]]:
        return ((0.0, 0.5),)

    def capabilities(self) -> ValueCapabilities:
        return _capabilities()


def test_value_head_protocol_accepts_a_conforming_fake() -> None:
    head = _HonestHead()
    assert isinstance(head, ValueHead)
    assert not isinstance(object(), ValueHead)
    consumed = check_value_capabilities(
        required_granularity="token",
        needs_old_values=False,
        capabilities=head.capabilities(),
    )
    assert consumed == ("granularity",)


@dataclass(frozen=True, slots=True)
class _MisshapenHead:
    """Same method NAMES as ValueHead, wrong shapes on both."""

    def estimate(self, _batch: object, _required_flag: bool) -> str:
        return "wide"

    def capabilities(self) -> str:
        return "not a ValueCapabilities"


def test_value_head_isinstance_checks_presence_never_signatures() -> None:
    """WHAT IS CLAIMED: the documented hole, pinned. ``@runtime_checkable``
    checks METHOD PRESENCE, never signatures, so _MisshapenHead satisfies
    ``isinstance`` against :class:`ValueHead` even though ``estimate`` needs
    an argument the protocol does not declare and ``capabilities`` returns a
    bare str -- and the wrong shape is discovered only at the call, as a
    TypeError naming the missing argument. This is exactly why the module
    pairs every claim with a measurement rather than trusting the protocol
    check. WHAT IS NOT CLAIMED: that the isinstance success attests
    conformance -- it does not; that is the entire reason
    :func:`check_value_capabilities` and :func:`verify_estimates` exist.
    """
    head = _MisshapenHead()
    assert isinstance(head, ValueHead)
    with pytest.raises(TypeError) as exc_info:
        head.estimate(_make_batch({"mask": ((1,),)}))
    assert "_required_flag" in str(exc_info.value)
