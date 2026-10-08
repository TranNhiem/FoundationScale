"""Golden-fixture parity: prompt_mean_row_weights against the reference weights.

The fixture was generated once, offline, from the upstream implementation the
reduction was ported from, so this test pins exact agreement without importing
that code base. A drift here means the ported denominator no longer matches the
one the reference recipe trains with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundationscale.rl.group_policy_objectives import prompt_mean_row_weights

_FIXTURE = Path(__file__).parent / "fixtures" / "prompt_mean_reference.json"


def _cases() -> list[dict[str, list]]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["cases"]


def test_fixture_is_non_empty_and_well_formed() -> None:
    cases = _cases()
    assert len(cases) == 20
    for case in cases:
        assert len(case["group_ids"]) == len(case["supervised_tokens"]) == len(case["expected"])


@pytest.mark.parametrize("index", range(20))
def test_weights_match_the_reference_exactly(index: int) -> None:
    case = _cases()[index]
    observed = prompt_mean_row_weights(case["group_ids"], case["supervised_tokens"])
    assert list(observed) == pytest.approx(case["expected"], rel=0.0, abs=1e-15)
