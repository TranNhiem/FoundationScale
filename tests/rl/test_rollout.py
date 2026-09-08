"""Tests for the rollout-source contract and its fail-closed handshake."""

from __future__ import annotations

from typing import Any

import pytest

from foundationscale.rl.interfaces import ExperienceBatch
from foundationscale.rl.rollout import (
    CapabilityRefusal,
    RolloutSource,
    SourceCapabilities,
    check_capabilities,
    verify_generated,
)

# A healthy capability set and a healthy delivered batch: every declared
# column arrives, all required columns share one row count, and the row
# count is non-zero. The dishonest-source controls below break exactly one
# of these properties at a time.
_DECLARED = ("prompt_token_ids", "completion_token_ids", "completion_logprobs")


def _capabilities(names: tuple[str, ...] = _DECLARED) -> SourceCapabilities:
    return SourceCapabilities(columns=names)


def _generated(rows: int = 3) -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "prompt_token_ids": ((1, 2),) * rows,
            "completion_token_ids": ((3,),) * rows,
            "completion_logprobs": ((-0.5,),) * rows,
        }
    )


class _FakeSource:
    """A protocol-conformant source whose CLAIM and DELIVERY are set apart.

    The claim/measurement-gap control needs the two to diverge, which a
    source that derived one from the other could never express.
    """

    def __init__(self, capabilities: SourceCapabilities, batch: ExperienceBatch) -> None:
        self._capabilities = capabilities
        self._batch = batch

    def capabilities(self) -> SourceCapabilities:
        return self._capabilities

    def generate(self, prompts: ExperienceBatch) -> ExperienceBatch:
        return self._batch


class _UnalignedReturnedBatch:
    """Stands in for a delivered object that is NOT a real ExperienceBatch.

    A genuine ExperienceBatch cannot be misaligned -- its constructor
    refuses -- so the disagreement arm of verify_generated is only
    reachable across the seam, where adapter code may hand back anything
    shaped like a batch. This stub is that anything.
    """

    def __init__(self, columns: dict[str, tuple[Any, ...]]) -> None:
        self.columns = columns


def test_of_matches_direct_construction_and_carries_the_truncation_claim() -> None:
    assert SourceCapabilities.of("a", "b") == SourceCapabilities(columns=("a", "b"))
    claimed = SourceCapabilities.of("a", truncation_reported=True)
    assert claimed.truncation_reported is True
    assert SourceCapabilities.of("a").truncation_reported is False


def test_columns_are_normalised_to_a_tuple() -> None:
    capabilities = SourceCapabilities(columns=["a", "b"])  # type: ignore[arg-type]
    assert capabilities.columns == ("a", "b")


def test_rollout_source_is_runtime_checkable() -> None:
    # runtime_checkable exists so setup tooling can refuse a non-source
    # before any step rather than at the first generate() call; pin both
    # answers so the decorator cannot be dropped silently.
    assert isinstance(_FakeSource(_capabilities(), _generated()), RolloutSource)
    assert not isinstance(object(), RolloutSource)


@pytest.mark.parametrize("bad", ["", 7])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_capabilities_refuse_a_non_name_entry_at_every_position(index: int, bad: object) -> None:
    # Parametrised over EVERY position, not one representative: a guard
    # that only checked the first entry would pass a sweep of index 0 and
    # still admit an unnamed capability everywhere else.
    names: list[Any] = ["a", "b", "c"]
    names[index] = bad
    with pytest.raises(CapabilityRefusal) as exc_info:
        SourceCapabilities(columns=tuple(names))
    message = str(exc_info.value)
    assert f"columns[{index}]" in message
    assert repr(bad) in message


def test_capabilities_refuse_a_duplicated_name() -> None:
    with pytest.raises(CapabilityRefusal) as exc_info:
        SourceCapabilities(columns=("a", "b", "a"))
    message = str(exc_info.value)
    assert "'a'" in message
    assert "columns[2]" in message  # where the repeat stood
    assert "columns[0]" in message  # where the original stood


def test_check_capabilities_returns_exactly_the_checked_set() -> None:
    # The returned tuple -- not a superset the caller inferred -- is the
    # denominator the caller checked. Accepting a list and returning the
    # validated tuple pins the normalisation too.
    satisfied = check_capabilities(
        required=["prompt_token_ids", "completion_logprobs"],
        capabilities=_capabilities(),
    )
    assert satisfied == ("prompt_token_ids", "completion_logprobs")


def test_check_capabilities_lists_every_missing_name_and_the_denominator() -> None:
    # Three of five are missing; stopping at the first gap would let a
    # half-met requirement read as a met one, so the refusal names all
    # three and states the denominator.
    with pytest.raises(CapabilityRefusal) as exc_info:
        check_capabilities(
            required=(
                "prompt_token_ids",
                "a",
                "b",
                "c",
                "completion_token_ids",
            ),
            capabilities=_capabilities(),
            origin="<test-source>",
        )
    message = str(exc_info.value)
    assert "3 of 5 required columns are unavailable from <test-source>" in message
    assert "a, b, c" in message
    assert "prompt_token_ids" in message  # the source's claim is shown too


def test_check_capabilities_names_the_default_origin_when_none_given() -> None:
    with pytest.raises(CapabilityRefusal) as exc_info:
        check_capabilities(required=("nope",), capabilities=_capabilities())
    assert "<rollout-source>" in str(exc_info.value)


def test_empty_required_set_refused_at_the_handshake() -> None:
    # An empty requirement makes the handshake vacuously pass -- every
    # requirement in it is trivially met because there are none -- and the
    # refusal must SAY that, because a vacuous pass is the exact failure
    # this framework exists to prevent.
    with pytest.raises(CapabilityRefusal) as exc_info:
        check_capabilities(required=(), capabilities=_capabilities())
    message = str(exc_info.value)
    assert "EMPTY" in message
    assert "vacuous" in message


def test_empty_required_set_refused_at_verification() -> None:
    with pytest.raises(CapabilityRefusal) as exc_info:
        verify_generated(_generated(), required=())
    assert "vacuous" in str(exc_info.value)


@pytest.mark.parametrize("bad", ["", 7])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_required_refuses_a_non_name_entry_at_the_handshake(index: int, bad: object) -> None:
    required: list[Any] = ["a", "b", "c"]
    required[index] = bad
    with pytest.raises(CapabilityRefusal) as exc_info:
        check_capabilities(required=required, capabilities=_capabilities())
    message = str(exc_info.value)
    assert f"required[{index}]" in message
    assert repr(bad) in message


@pytest.mark.parametrize("bad", ["", 7])
def test_required_refuses_a_non_name_entry_at_verification(bad: object) -> None:
    with pytest.raises(CapabilityRefusal) as exc_info:
        verify_generated(
            _generated(),
            required=["prompt_token_ids", bad],  # type: ignore[list-item]
        )
    assert repr(bad) in str(exc_info.value)


def test_a_conformant_source_passes_both_halves_of_the_handshake() -> None:
    source = _FakeSource(_capabilities(), _generated(rows=2))
    satisfied = check_capabilities(
        required=("prompt_token_ids", "completion_token_ids"),
        capabilities=source.capabilities(),
    )
    batch = source.generate(ExperienceBatch(columns={"prompt_token_ids": ((1,), (2,))}))
    assert verify_generated(batch, required=satisfied) == 2


def test_verify_generated_returns_the_measured_row_count() -> None:
    # The returned count is the denominator downstream verdicts must use;
    # pin that it comes from the REQUIRED columns, across more than one of
    # them, so the shared-count path is exercised rather than assumed.
    assert (
        verify_generated(
            _generated(rows=4),
            required=("prompt_token_ids", "completion_logprobs"),
        )
        == 4
    )


def test_verify_generated_names_every_undelivered_column() -> None:
    batch = ExperienceBatch(columns={"prompt_token_ids": ((1,), (2,))})
    with pytest.raises(CapabilityRefusal) as exc_info:
        verify_generated(
            batch,
            required=(
                "prompt_token_ids",
                "completion_token_ids",
                "completion_logprobs",
            ),
            origin="<dishonest>",
        )
    message = str(exc_info.value)
    assert "2 of 3 required columns are absent from the batch returned by <dishonest>" in message
    assert "completion_token_ids, completion_logprobs" in message
    assert "prompt_token_ids" in message  # listed as carried, not as missing


def test_declared_but_undelivered_column_is_caught_after_a_passing_handshake() -> None:
    # The claim/measurement gap this module exists to close: capabilities()
    # CLAIMS all three columns and check_capabilities ACCEPTS the claim,
    # but generate() DELIVERS only two. Nothing at setup can see that;
    # only verify_generated, reading the real batch, refuses it. This is
    # the rollout-side mirror of the item-4 undeclared-sft_loss reading.
    declared = SourceCapabilities.of(
        "prompt_token_ids", "completion_token_ids", "completion_logprobs"
    )
    delivered = ExperienceBatch(
        columns={
            "prompt_token_ids": ((1,), (2,)),
            "completion_token_ids": ((3,), (4,)),
        }
    )
    source = _FakeSource(declared, delivered)
    satisfied = check_capabilities(required=declared.columns, capabilities=source.capabilities())
    assert satisfied == declared.columns
    with pytest.raises(CapabilityRefusal) as exc_info:
        verify_generated(source.generate(ExperienceBatch(columns={})), required=satisfied)
    message = str(exc_info.value)
    assert "1 of 3" in message
    assert "completion_logprobs" in message


def test_verify_generated_refuses_a_misaligned_delivered_batch() -> None:
    # Only reachable with a delivered object that is NOT a real
    # ExperienceBatch (see _UnalignedReturnedBatch); the refusal must name
    # the disagreeing column and give BOTH counts.
    delivered = _UnalignedReturnedBatch(
        {
            "prompt_token_ids": ((1,), (2,), (3,)),
            "completion_token_ids": ((1,), (2,)),
        }
    )
    with pytest.raises(CapabilityRefusal) as exc_info:
        verify_generated(
            delivered,  # type: ignore[arg-type]
            required=("prompt_token_ids", "completion_token_ids"),
        )
    message = str(exc_info.value)
    assert "completion_token_ids" in message
    assert "2 rows" in message
    assert "3 rows" in message


def test_zero_row_batch_refused_because_nothing_was_measured() -> None:
    batch = ExperienceBatch(columns={"prompt_token_ids": (), "completion_token_ids": ()})
    with pytest.raises(CapabilityRefusal) as exc_info:
        verify_generated(batch, required=("prompt_token_ids", "completion_token_ids"))
    message = str(exc_info.value)
    assert "<rollout-source>" in message
    assert "0 rows" in message
    assert "nothing was measured" in message
