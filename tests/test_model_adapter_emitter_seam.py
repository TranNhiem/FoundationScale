"""Pin the TWO-SIDED coupling between the model adapter and the manifest producer.

tools/emit_run_manifest.py refuses a run when its two readers disagree: the
affirmative MoE flag is read THROUGH the selected adapter
(``_enable_moe_block_flag``), while the routed-expert count is read by
``declared_from_hf_config`` in foundationscale.provenance.manifest. That
refusal only tells the truth when both readers search the SAME scopes and
honour a compatible count vocabulary. Widening one side is not a safe
superset: an adapter that also looked under ``llm_config`` would read
flag=True where the producer found no count, and the emitter would refuse
with "no routed-expert count was found under the keys this tool understands"
-- a verdict that fails closed but states a reason that is FALSE, because the
count does exist, in a scope the producer never looked at. A claim mismatched
to its evidence is a defect here even when the verdict is safe.

Controls and accounting (house rules): the coupling tests reference the
manifest's constants on ONE side only -- a test that expected the literal
"text_config" on both sides would pass by construction and measure nothing,
the ``all([]) is True`` defect in string form. The wide-scope probe is the
MUST_FIRE control proving the scope pin is load-bearing, and the non-boolean
flag test is the MUST_FIRE control for exception translation; the remaining
tests are MUST_PASS pins of seam behaviour from before Gemma semantics moved
behind the adapter. Every test executes; nothing here skips
(FS_FORBID_SKIPS=1), and a broken import fails the suite by name.
"""

from __future__ import annotations

from typing import Any

import pytest
from tools.emit_run_manifest import EmitRefused, _enable_moe_block_flag

from foundationscale.models import adapters
from foundationscale.provenance import manifest

# The nested scope is named once, from its single definition. Building probe
# configs with this constant keeps the behaviour tests measuring the seam
# rather than restating a literal that could rot in two directions at once.
_NESTED = manifest._NESTED_LM_SCOPE_KEY


class _WideScopeProbeAdapter(adapters._Base):
    """Test-only adapter whose flag search covers one scope the producer does not.

    This is the asymmetry the whole file exists to indict, built by hand: if
    the real adapter were ever widened this way, a config nesting both facts
    under the extra scope would classify as MoE-flagged to the emitter while
    the producer honestly reported no count -- and the refusal reason would be
    false. Constructing the divergence here proves test 1 measures something.
    """

    name = "wide-scope-probe"
    _flag_keys = ("enable_moe_block",)

    def enable_moe_block_flag(self, config: dict[str, Any]) -> tuple[bool | None, str]:
        extra = config.get("llm_config")
        if isinstance(extra, dict) and "enable_moe_block" in extra:
            dotted = "llm_config.enable_moe_block"
            return adapters._refuse_unless_bool(dotted, extra[dotted.split(".")[1]]), "llm_config"
        return super().enable_moe_block_flag(config)


def _routed_count(declared: object) -> object:
    """Extract the routed-expert count from the producer's record.

    The record is read by attribute; a mapping is accepted too, and any third
    shape fails loudly with TypeError -- an unrecognised record is a defect to
    name, never a case to coerce (the same rule the adapter itself refuses by).
    """
    sentinel: object = object()
    value = getattr(declared, "num_experts", sentinel)
    if value is not sentinel:
        return value
    if isinstance(declared, dict):
        return declared.get("num_experts")
    raise TypeError(
        f"declared_from_hf_config returned a shape this test cannot read: {type(declared).__name__}"
    )


def test_nested_scopes_tuple_is_the_manifest_constant() -> None:
    # MUST_PASS: the pin itself. Comparing against the manifest's constant --
    # never against a restated "text_config" literal -- is what makes this a
    # measurement: if either side's definition moves without the other, the
    # tuples diverge and this fires.
    assert adapters.NESTED_SCOPES == (_NESTED,), (
        f"adapter scopes {adapters.NESTED_SCOPES!r} drifted from the manifest's "
        f"{(manifest._NESTED_LM_SCOPE_KEY,)!r}: the two readers of the emitter's "
        "two-sided comparison no longer look in the same places, so the emitter "
        "can refuse with a reason that is false (count found vs. count absent)"
    )


def test_dialect_table_covers_every_measured_count_key() -> None:
    # MUST_PASS: wider is permitted (extension keys reach the standalone
    # classifier only), narrower is the defect -- a measured key the producer
    # honours but the table omits would classify a declaring MoE config as
    # "no dialect keys present". The module's import-time guard turns this
    # drift into a collection error; this assertion restates the invariant in
    # a form a reviewer can audit apart from the guard's existence.
    table_count_keys = {row["key"] for row in adapters.DIALECT_TABLE if row["kind"] == "count"}
    missing = tuple(k for k in manifest._EXPERT_COUNT_KEYS if k not in table_count_keys)
    assert not missing, (
        f"measured count keys absent from DIALECT_TABLE: {missing!r}; the producer "
        "honours them, so the adapter would report 'no MoE dialect keys present' "
        "for a config whose MoE status was in fact measured"
    )


def test_wide_scope_probe_diverges_from_the_producer_count() -> None:
    # MUST_FIRE control: a config nesting BOTH facts under the extra scope.
    # The probe reads the flag; the producer, searching only its declared
    # scopes, finds no count. Both halves must hold or this control is dead.
    config = {"llm_config": {"enable_moe_block": True, "n_routed_experts": 8}}
    flag, scope = _WideScopeProbeAdapter().enable_moe_block_flag(config)
    assert flag is True and scope == "llm_config", (
        f"probe reported ({flag!r}, {scope!r}): the wide reader must find the "
        "flag for the asymmetry demonstration to measure anything"
    )
    declared_count = _routed_count(manifest.declared_from_hf_config(config))
    assert declared_count is None, (
        f"producer reported num_experts={declared_count!r} for an llm_config-"
        "nested count: if the producer's scope set ever widens this way, the "
        "pinned asymmetry vanishes and the emitter's disagreement refusal "
        "would silently stop being reachable from this defect class"
    )


def test_seam_reads_a_nested_flag() -> None:
    # MUST_PASS: pre-refactor behaviour -- the nested LM scope is where Gemma
    # family configs state the declaration, and moving the read behind the
    # adapter must not have lost it.
    config = {_NESTED: {"enable_moe_block": True}}
    assert _enable_moe_block_flag(config) == (True, _NESTED)


def test_seam_reads_a_top_level_flag() -> None:
    # MUST_PASS: top level is the other scope the emitter has always searched;
    # an adapter rewrite that only honoured nesting would silently lose it.
    assert _enable_moe_block_flag({"enable_moe_block": True}) == (True, "top level")


def test_seam_reports_absence_as_absence_never_false() -> None:
    # MUST_PASS: (None, "") is the unmeasured shape. Collapsing absence to
    # False would assert a dense declaration no config ever made -- the
    # absence-is-not-evidence rule in one tuple.
    assert _enable_moe_block_flag({"model_type": "gemma3"}) == (None, "")


def test_seam_nested_scope_wins_when_scopes_disagree() -> None:
    # MUST_PASS: scope precedence is part of the dialect contract the emitter
    # documents ("text_config" or "top level"); reversing it would flip real
    # Gemma adjudications without a single flag value changing.
    config = {_NESTED: {"enable_moe_block": False}, "enable_moe_block": True}
    assert _enable_moe_block_flag(config) == (False, _NESTED)


def test_seam_translates_adapter_refusal_into_emit_refused() -> None:
    # MUST_FIRE: a string "false" truthy-parses as MoE, so the adapter refuses
    # -- but callers of the emitter catch EmitRefused. An AdapterRefusal that
    # escaped untranslated would surface as an unhandled error, breaking the
    # seam's contract precisely on the path that exists to fail closed.
    with pytest.raises(EmitRefused) as excinfo:
        _enable_moe_block_flag({"enable_moe_block": "false"})
    assert "not a JSON boolean" in str(excinfo.value), (
        "the translated refusal must carry the adapter's reason verbatim so the "
        "operator sees the coercible value, not a reworded second-hand account"
    )


def test_seam_reads_the_flag_for_non_gemma_configs() -> None:
    # MUST_PASS: the generic fallback adapter carries the flag keys too. This
    # is the regression the refactor could have introduced silently: Gemma
    # semantics moving behind an adapter must not have narrowed flag reading
    # to Gemma-only, or every other family's affirmative declaration would
    # become invisible to the emitter.
    config = {"model_type": "llama", "enable_moe_block": True}
    assert _enable_moe_block_flag(config) == (True, "top level")
