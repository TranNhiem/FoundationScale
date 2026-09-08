"""Tests for the PolicyPair container (design section 3.5)."""

import pytest

from foundationscale.rl import PolicyPair, PolicyRoleRefusal


def test_sft_shape_has_no_stubs() -> None:
    # SFT is PolicyPair(train_view=model): no generate view, no references,
    # and nothing fake standing in for either.
    pair = PolicyPair(train_view=object())

    assert pair.roles() == ("train",)
    assert pair.has_generate_view is False


def test_dpo_shape() -> None:
    # DPO is a train view plus one frozen reference under a caller-chosen name.
    pair = PolicyPair(train_view=object(), references={"reference": object()})

    assert pair.has_generate_view is False
    assert pair.roles() == ("train", "reference")


def test_reference_hit_returns_the_registered_view() -> None:
    frozen = object()
    pair = PolicyPair(train_view=object(), references={"reference": frozen})

    assert pair.reference("reference") is frozen


def test_reference_miss_refuses_and_names_available_roles() -> None:
    pair = PolicyPair(
        train_view=object(),
        generate_view=object(),
        references={"reference": object()},
    )

    with pytest.raises(PolicyRoleRefusal) as excinfo:
        pair.reference("missing")

    message = str(excinfo.value)
    assert "missing" in message
    # The message names the REFERENCE roles, not roles(). "train" and
    # "generate" are roles of the pair but are refused as reference keys, so
    # listing them here would name two arguments this method cannot accept.
    # repr, not the bare word: "reference" appears in this message's prose
    # regardless, so the bare word would pass over an EMPTY role list.
    assert repr("reference") in message
    assert "train" not in message
    assert "generate" not in message


def test_a_view_role_is_not_a_reference_role() -> None:
    # "train" is in roles() and is still not answerable by reference(): the two
    # namespaces are deliberately disjoint, and this is the case that would
    # silently return the trainable view if they were not.
    pair = PolicyPair(train_view=object(), generate_view=object())

    for role in ("train", "generate"):
        with pytest.raises(PolicyRoleRefusal):
            pair.reference(role)


def test_train_view_none_is_refused() -> None:
    with pytest.raises(PolicyRoleRefusal):
        PolicyPair(train_view=None)


@pytest.mark.parametrize("bad_key", ["", 3])
def test_reference_key_must_be_a_non_empty_str(bad_key: object) -> None:
    with pytest.raises(PolicyRoleRefusal) as excinfo:
        PolicyPair(train_view=object(), references={bad_key: object()})  # type: ignore[dict-item]

    assert repr(bad_key) in str(excinfo.value)


@pytest.mark.parametrize("shadowing", ["train", "generate"])
def test_reference_key_may_not_shadow_a_container_role(shadowing: str) -> None:
    with pytest.raises(PolicyRoleRefusal) as excinfo:
        PolicyPair(train_view=object(), references={shadowing: object()})

    assert shadowing in str(excinfo.value)


def test_reference_value_none_is_refused() -> None:
    with pytest.raises(PolicyRoleRefusal) as excinfo:
        PolicyPair(train_view=object(), references={"reference": None})

    assert "reference" in str(excinfo.value)


def test_roles_ordering_is_stable_with_several_references() -> None:
    # Role ordering must be reproducible: fixed roles first, references sorted.
    pair = PolicyPair(
        train_view=object(),
        generate_view=object(),
        references={
            "zebra": object(),
            "alpha": object(),
            "mid": object(),
        },
    )

    assert pair.roles() == ("train", "generate", "alpha", "mid", "zebra")


def test_roles_ordering_omits_generate_when_absent() -> None:
    pair = PolicyPair(
        train_view=object(),
        references={"beta": object(), "alpha": object()},
    )

    assert pair.roles() == ("train", "alpha", "beta")


def test_with_reference_returns_a_new_pair_and_leaves_the_original_unchanged() -> None:
    original = PolicyPair(train_view=object(), references={"reference": object()})
    added = object()

    extended = original.with_reference("second", added)

    assert extended is not original
    assert extended.reference("second") is added
    assert "second" in extended.roles()
    assert "second" not in original.roles()
    assert original.roles() == ("train", "reference")


def test_with_reference_refuses_a_duplicate_role() -> None:
    pair = PolicyPair(train_view=object(), references={"reference": object()})

    with pytest.raises(PolicyRoleRefusal) as excinfo:
        pair.with_reference("reference", object())

    assert "reference" in str(excinfo.value)


def test_caller_mutating_its_mapping_does_not_change_the_pair() -> None:
    # The container freezes a dict copy of the mapping it is handed, so the
    # caller's own dict changing afterwards cannot reach inside it.
    refs: dict[str, object] = {"reference": object()}
    pair = PolicyPair(train_view=object(), references=refs)

    refs["injected"] = object()
    refs["reference"] = object()

    assert pair.roles() == ("train", "reference")
    assert pair.reference("reference") is not refs["reference"]
    with pytest.raises(PolicyRoleRefusal):
        pair.reference("injected")
