"""Pin model-adapter classification as a measurement with paired controls.

The MUST_PASS cases prove that published MoE and dense dialects are accepted with
attributable evidence. The MUST_FIRE cases prove that absence, contradiction, and
mistyped declarations are never coerced into a healthier-looking verdict. Together
they guard the central accounting rule of this package: zero measured units is
UNMEASURED, never a pass.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import foundationscale.models.adapters as model_adapters  # noqa: E402
from foundationscale.models.adapters import (  # noqa: E402
    AdapterRefusal,
    Architecture,
    GemmaAdapter,
    GenericHFAdapter,
    classify_config,
    register_adapter,
    select_adapter,
)


@pytest.fixture(autouse=True)
def _isolate_registered_adapters() -> Iterator[None]:
    """Restore module-level registry state even when a registry test fails."""
    snapshot = list(model_adapters._REGISTERED)
    yield
    # register_adapter mutates process-wide state; without restoration, one test's
    # extension would change which adapter its neighbours measure.
    model_adapters._REGISTERED[:] = snapshot


def test_gemma_moe_signals_are_corroborated_must_pass() -> None:
    """MUST_PASS: Gemma's affirmative flag and measured routed count agree."""
    config = {
        "model_type": "gemma",
        "text_config": {
            "enable_moe_block": True,
            "num_experts": 128,
        },
    }

    result = classify_config(config)

    assert result.architecture is Architecture.MOE
    assert result.num_routed_experts == 128
    assert result.adapter == "gemma"
    assert result.evidence == (
        "text_config.num_experts=128; corroborated by text_config.enable_moe_block=True"
    ), (
        "the affirmative flag must corroborate the measured count, not replace it: "
        "this evidence is what makes the 128-expert verdict auditable"
    )


def test_false_flag_without_count_is_dense_must_pass() -> None:
    """MUST_PASS: an explicit dense declaration is accepted on the flag alone."""
    result = classify_config({"enable_moe_block": False})

    assert result.architecture is Architecture.DENSE
    assert result.num_routed_experts is None
    assert result.evidence == "enable_moe_block=False"


def test_false_flag_and_zero_count_is_positive_dense_must_pass() -> None:
    """MUST_PASS: a dense flag plus a measured zero count is a positive dense claim."""
    config = {
        "enable_moe_block": False,
        "num_experts": 0,
    }

    result = classify_config(config)

    assert result.architecture is Architecture.DENSE
    assert result.num_routed_experts == 0
    assert result.evidence == "enable_moe_block=False; num_experts=0", (
        "a measured zero must remain distinct from an absent count: the former "
        "declares zero routed experts, while the latter measured nothing"
    )


def test_empty_config_is_unmeasured_not_dense_must_fire() -> None:
    """MUST_FIRE: no dialect keys means UNDETERMINED, never inferred dense."""
    result = classify_config({})

    assert result.architecture is Architecture.UNDETERMINED
    assert result.num_routed_experts is None
    assert "absence is unmeasured, not dense" in result.evidence, (
        "an empty config has no measured units; minting zero experts from an "
        "absent key would turn a real MoE into a vacuous dense pass"
    )


def test_dense_flag_against_positive_count_must_fire() -> None:
    """MUST_FIRE: a dense declaration beside a live routed count is a conflict."""
    config = {
        "enable_moe_block": False,
        "text_config": {"num_experts": 8},
    }

    result = classify_config(config)

    assert result.architecture is Architecture.UNDETERMINED
    assert result.num_routed_experts is None
    assert result.evidence == (
        "conflict: dense flag (enable_moe_block=False) vs positive expert count "
        "(text_config.num_experts=8)"
    ), "neither contradictory signal may silently win over the other"


def test_moe_flag_against_zero_count_must_fire() -> None:
    """MUST_FIRE: an affirmative MoE flag beside a measured zero count conflicts."""
    config = {
        "enable_moe_block": True,
        "num_experts": 0,
    }

    result = classify_config(config)

    assert result.architecture is Architecture.UNDETERMINED
    assert result.num_routed_experts is None
    assert result.evidence == (
        "conflict: MoE flag (enable_moe_block=True) vs zero expert count (num_experts=0)"
    ), "an affirmative flag must not manufacture routed experts from a measured zero"


def test_divergent_counts_between_scopes_must_fire() -> None:
    """MUST_FIRE: two different positive counts cannot select one winner."""
    config = {
        "text_config": {"num_experts": 8},
        "num_experts": 16,
    }

    result = classify_config(config)

    assert result.architecture is Architecture.UNDETERMINED
    assert result.num_routed_experts is None
    assert result.evidence == (
        "conflict: divergent positive expert counts (text_config.num_experts=8; num_experts=16)"
    ), (
        "the nested scope is deliberately reported first; changing that order would "
        "change which statement the audit trail presents as primary"
    )


def test_string_flag_is_refused_must_fire() -> None:
    """MUST_FIRE: the quoted string ``"false"`` is refused rather than coerced."""
    with pytest.raises(AdapterRefusal) as excinfo:
        classify_config({"enable_moe_block": "false"})

    message = str(excinfo.value)
    assert "enable_moe_block" in message
    assert "not a JSON boolean" in message, (
        "Python truthiness would read a nonempty quoted 'false' as MoE; the refusal "
        "is the guard against that exact misclassification"
    )


def test_boolean_count_is_refused_must_fire() -> None:
    """MUST_FIRE: boolean ``True`` is not read as the integer count one."""
    with pytest.raises(AdapterRefusal) as excinfo:
        classify_config({"num_experts": True})

    message = str(excinfo.value)
    assert "num_experts" in message
    assert "not a JSON integer routed-expert count" in message, (
        "isinstance(True, int) is true, so checking int before bool would silently "
        "invent a one-expert MoE from a boolean config value"
    )


def test_negative_count_is_refused_must_fire() -> None:
    """MUST_FIRE: routed-expert counts are unsigned."""
    with pytest.raises(AdapterRefusal) as excinfo:
        classify_config({"num_experts": -1})

    message = str(excinfo.value)
    assert "num_experts" in message
    assert "expert counts are unsigned" in message


def test_non_mapping_config_is_refused_must_fire() -> None:
    """MUST_FIRE: a JSON array has no top-level config keys to measure."""
    with pytest.raises(AdapterRefusal) as excinfo:
        classify_config(["not", "a", "config"])

    assert "model config is not a JSON object" in str(excinfo.value), (
        "calling .get on a non-mapping would turn malformed input into an "
        "unattributed implementation error; it must remain a classified refusal"
    )


def test_unknown_family_uses_generic_fallback_must_pass() -> None:
    """MUST_PASS: a family with no bespoke adapter still classifies generically."""
    config = {
        "model_type": "mixtral",
        "num_local_experts": 8,
    }

    result = classify_config(config)

    assert result.architecture is Architecture.MOE
    assert result.num_routed_experts == 8
    assert result.adapter == "generic", (
        "model coverage must not depend on a hand-written adapter for every family; "
        "the dialect table is the declared generic fallback"
    )


def test_equal_counts_across_scopes_do_not_conflict_must_pass() -> None:
    """MUST_PASS: duplicate equal counts corroborate rather than diverge."""
    config = {
        "text_config": {"num_experts": 8},
        "num_experts": 8,
    }

    result = classify_config(config)

    assert result.architecture is Architecture.MOE
    assert result.num_routed_experts == 8
    assert result.evidence == "text_config.num_experts=8; num_experts=8", (
        "the evidence must name both measurements in search order; treating equal "
        "duplicate declarations as a conflict would punish redundant truth"
    )


def test_select_adapter_claims_gemma_config_must_pass() -> None:
    """MUST_PASS: the built-in Gemma adapter claims its own model family."""
    adapter = select_adapter({"model_type": "gemma-4"})

    assert isinstance(adapter, GemmaAdapter)
    assert adapter.name == "gemma"


def test_select_adapter_uses_generic_for_unknown_config_must_pass() -> None:
    """MUST_PASS: an unmatched config falls back to the explicit generic adapter."""
    adapter = select_adapter({"model_type": "unknown-family"})

    assert isinstance(adapter, GenericHFAdapter)
    assert adapter.name == "generic"


def test_register_adapter_refuses_duplicate_name_must_fire() -> None:
    """MUST_FIRE: two adapters cannot share an attribution name."""
    with pytest.raises(AdapterRefusal) as excinfo:
        register_adapter(GemmaAdapter())

    assert "duplicate or reserved model adapter name: gemma" in str(excinfo.value), (
        "duplicate names would make classification evidence ambiguous: the same "
        "adapter string could refer to different rules"
    )


def test_register_adapter_refuses_reserved_generic_name_must_fire() -> None:
    """MUST_FIRE: callers cannot register over the reserved generic fallback."""
    with pytest.raises(AdapterRefusal) as excinfo:
        register_adapter(GenericHFAdapter())

    assert "duplicate or reserved model adapter name: generic" in str(excinfo.value), (
        "the generic adapter must remain the explicit fallback, not a precedence "
        "participant whose registration order changes attribution"
    )


def test_registered_adapter_extends_selection_must_pass() -> None:
    """MUST_PASS: a registered adapter can claim a family outside the built-ins."""

    class CustomAdapter(GenericHFAdapter):
        name = "custom-family"

        def matches(self, config: dict[str, Any]) -> bool:
            return config.get("model_type") == "custom-family"

    config = {
        "model_type": "custom-family",
        "num_local_experts": 4,
    }
    adapter = CustomAdapter()
    register_adapter(adapter)

    selected = select_adapter(config)
    result = classify_config(config)

    assert selected is adapter
    assert result.adapter == "custom-family", (
        "registration is only an extension point if a new matcher is actually "
        "selected and its name is what the classification reports"
    )


def test_explicit_generic_adapter_is_refused_must_fire() -> None:
    """MUST_FIRE: the generic fallback cannot be inserted by an explicit sequence."""
    with pytest.raises(AdapterRefusal) as excinfo:
        select_adapter({}, [GenericHFAdapter()])

    assert "generic fallback must not be registered" in str(excinfo.value), (
        "an explicitly sequenced generic adapter would make fallback behavior depend "
        "on caller ordering instead of remaining the framework's declared default"
    )
