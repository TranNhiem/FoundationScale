"""Tests that the stated objective-member set and the PreferenceObjective
protocol are the same set.

``preference._OBJECTIVE_MEMBERS`` restates, by hand, the members the protocol
demands. It is stated rather than read from ``PreferenceObjective``'s
``__protocol_attrs__`` because that attribute exists only on Python 3.12+
while this package supports 3.10, so reading it would replace a clean
``AlgorithmWiringRefusal`` with an ``AttributeError`` inside the refusal path
on the two oldest supported interpreters.

A hand-restated set is a mirror of the thing it pins: the two agree by
construction unless something measures them independently. That is what this
module does, and it drives the protocol through the public ``isinstance``
door rather than through the version-gated dunder, so the measurement is
available on every interpreter the package claims.

Both drift directions are covered, and by different legs:

* a member the protocol demands and the constant OMITS makes the full stub
  fail ``isinstance`` -- caught by the exactly-the-stated-members leg;
* a member the constant states and the protocol does NOT demand makes the
  stub still satisfy ``isinstance`` when that member is dropped -- caught by
  the drop-one leg.
"""

from __future__ import annotations

from typing import Any

from foundationscale.rl.preference import (
    _OBJECTIVE_MEMBERS,
    PreferenceObjective,
    _absent_members,
)


def _member_value(name: str) -> Any:
    """Return a presence-satisfying class attribute for one protocol member.

    ``@runtime_checkable`` measures attribute PRESENCE only, so the value has
    to be non-None and of the right callable-ness, never correct. Returning a
    working objective here would make the leg measure the objective instead of
    the member set.
    """
    if name in ("reference_free", "required_columns"):
        return property(lambda self: None)
    return lambda self, *args, **kwargs: None


def _objective_stub(*, omit: str | None = None) -> object:
    """Build an object exposing exactly the stated members, less ``omit``."""
    namespace = {name: _member_value(name) for name in _OBJECTIVE_MEMBERS if name != omit}
    return type("ObjectiveStub", (), namespace)()


def test_stated_member_set_is_not_empty() -> None:
    # A vacuous constant would make every leg below pass by having nothing to
    # measure: the empty stub would satisfy any protocol demanding nothing,
    # and the drop-one loop would run zero times.
    assert len(_OBJECTIVE_MEMBERS) > 0
    assert len(set(_OBJECTIVE_MEMBERS)) == len(_OBJECTIVE_MEMBERS)
    assert tuple(sorted(_OBJECTIVE_MEMBERS)) == _OBJECTIVE_MEMBERS


def test_stub_exposing_exactly_the_stated_members_satisfies_the_protocol() -> None:
    # Drift direction 1: the protocol demands a member the constant omits.
    # The stub exposes the stated set and NOTHING else, so an undeclared
    # demand shows up here as a failing isinstance -- not as a refusal message
    # with a wrong denominator discovered at runtime.
    assert isinstance(_objective_stub(), PreferenceObjective)


def test_dropping_any_one_stated_member_breaks_the_protocol_check() -> None:
    # Drift direction 2: the constant states a member the protocol does not
    # demand. Such a member is invisible to isinstance, so dropping it would
    # leave the stub satisfying the protocol.
    still_satisfying = tuple(
        name
        for name in _OBJECTIVE_MEMBERS
        if isinstance(_objective_stub(omit=name), PreferenceObjective)
    )
    assert still_satisfying == (), (
        f"{len(still_satisfying)} of {len(_OBJECTIVE_MEMBERS)} stated members "
        f"are not demanded by PreferenceObjective: {still_satisfying!r}"
    )


def test_absent_members_names_exactly_the_dropped_member() -> None:
    # The refusal message quotes this helper, so a helper that reported the
    # wrong names would make a correct refusal unactionable.
    for name in _OBJECTIVE_MEMBERS:
        assert _absent_members(_objective_stub(omit=name)) == (name,)


def test_absent_members_over_a_complete_stub_is_empty() -> None:
    assert _absent_members(_objective_stub()) == ()


def test_absent_members_over_an_unrelated_object_names_every_member() -> None:
    # object() exposes __call__ on neither itself nor its type, so every
    # stated member is absent -- the denominator the refusal reports.
    assert _absent_members(object()) == _OBJECTIVE_MEMBERS
