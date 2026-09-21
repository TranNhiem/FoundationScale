"""Hostile regression tests for the per-layer attention span in FlopsModel (#530).

The defect guarded here: ``flops_per_token`` charged
``12 * layers * hidden_size * sequence_length``, i.e. it assumed EVERY layer
attends over the whole sequence. Two of the families this framework is built
for violate that. Qwen3.5 is a 3:1 hybrid whose linear-attention layers carry no
quadratic cost at all -- 27B declares 16 full-attention layers of 64, so the
term was charged roughly 4x -- and Gemma4 interleaves sliding-window layers
whose key-span is bounded by the WINDOW, not by the sequence.

Two of these tests are load-bearing rather than decorative, and each pins a trap
that reads as correct code:

* :func:`test_declared_span_of_zero_is_not_treated_as_undeclared` -- writing the
  fallback as ``span or layers * sequence_length`` breaks an all-linear model
  whose legitimate span is exactly 0, a 100% error in the FLATTERING direction.
* :func:`test_boolean_sliding_window_is_rejected` -- ``isinstance(True, int)`` is
  True in Python, so a flag read as a window charges every sliding layer at a
  span of 1.

The dense path must stay BIT-IDENTICAL: every MFU number already published for a
dense model was produced by the old expression, and a refactor that moved them
by a rounding step would invalidate the comparisons.
"""

from __future__ import annotations

from types import SimpleNamespace

from foundationscale.perf.telemetry import FlopsModel, _attention_span_total

SEQ = 2048
"""The sequence length these shapes were measured at on this estate."""


def _qwen35_layer_types(layers: int) -> list[str]:
    """The Qwen3.5 3:1 hybrid pattern: every fourth layer is full attention.

    Built rather than hardcoded so the ratio is visible; 27B is 64 layers,
    16 of them full, which is what the measured config.json declares.
    """
    return [("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(layers)]


def _flops(**overrides: object) -> FlopsModel:
    fields: dict[str, object] = {
        "parameters": 31_000_000_000,
        "non_embedding_parameters": 30_000_000_000,
        "layers": 48,
        "hidden_size": 5600,
        "sequence_length": SEQ,
    }
    fields.update(overrides)
    return FlopsModel(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 1. the dense path must not move
# --------------------------------------------------------------------------


def test_dense_config_declares_no_span_and_stays_bit_identical() -> None:
    """A config with no layer_types leaves the span undeclared and the number unmoved.

    The expected value is written out as the PRE-#530 expression rather than
    read back off the model, so this fails if the formula changes at all --
    including by a rounding step, which is the failure that would silently
    invalidate every published dense MFU figure.
    """
    config = SimpleNamespace(num_hidden_layers=48, hidden_size=5600)
    assert _attention_span_total(config, 48, SEQ) is None

    model = FlopsModel.from_hf_config(
        config,
        sequence_length=SEQ,
        parameters=31_000_000_000,
        non_embedding_parameters=30_000_000_000,
    )
    assert model is not None
    assert model.attention_span_total is None
    assert model.flops_per_token == 6 * 30_000_000_000 + 12 * 48 * 5600 * SEQ


def test_undeclared_span_falls_back_rather_than_refusing() -> None:
    """Undeclared is a fallback, not an UNMEASURED refusal.

    #529 refuses on an unresolvable expert count because charging every expert
    overstates MFU ~6x. The quadratic term is ~3.5% of the total at these
    shapes, so taking MFU offline for every hybrid over a 3% term would be
    disproportionate -- the fallback is deliberate policy, and this pins it.
    """
    model = _flops()
    assert model.attention_span_total is None
    assert model.unmeasured_reason is None
    assert model.flops_per_token > 0


# --------------------------------------------------------------------------
# 2-3. the hybrid correction
# --------------------------------------------------------------------------


def test_qwen35_hybrid_charges_only_its_full_attention_layers() -> None:
    """Qwen3.5-27B: 16 full of 64 layers, so the span is a quarter of the naive one."""
    layer_types = _qwen35_layer_types(64)
    assert layer_types.count("full_attention") == 16

    config = SimpleNamespace(num_hidden_layers=64, hidden_size=5120, layer_types=layer_types)
    span = _attention_span_total(config, 64, SEQ)
    assert span == 16 * SEQ
    # Exactly one quarter, asserted against the naive product rather than a
    # literal, so the relationship is what is pinned.
    assert span is not None and span * 4 == 64 * SEQ


def test_linear_attention_contributes_exactly_zero() -> None:
    """Asserted directly, not inferred from a total.

    Linear attention's cost is linear in the sequence and is already inside the
    6N parameter term; charging it again in the quadratic term double-counts.
    """
    config = SimpleNamespace(layer_types=["linear_attention"] * 10)
    assert _attention_span_total(config, 10, SEQ) == 0


def test_hybrid_lowers_flops_per_token_against_the_naive_formula() -> None:
    """The correction has to reach the number, not just the field."""
    naive = _flops(layers=64, hidden_size=5120)
    hybrid = _flops(layers=64, hidden_size=5120, attention_span_total=16 * SEQ)
    assert hybrid.flops_per_token < naive.flops_per_token
    assert hybrid.flops_per_token == 6 * 30_000_000_000 + 12 * 5120 * 16 * SEQ


# --------------------------------------------------------------------------
# 4-5. the sliding-window correction
# --------------------------------------------------------------------------


def test_sliding_window_below_sequence_is_charged_at_the_window() -> None:
    config = SimpleNamespace(layer_types=["sliding_attention"] * 4, sliding_window=512)
    assert _attention_span_total(config, 4, SEQ) == 4 * 512


def test_sliding_window_above_sequence_is_capped_at_the_sequence() -> None:
    """A window wider than the sequence cannot make a layer attend past the end."""
    config = SimpleNamespace(layer_types=["sliding_attention"] * 4, sliding_window=8192)
    assert _attention_span_total(config, 4, SEQ) == 4 * SEQ


def test_sliding_layer_without_a_declared_window_falls_back_to_the_sequence() -> None:
    """No window declared means the span is charged in full -- the safe direction."""
    config = SimpleNamespace(layer_types=["sliding_attention"] * 4)
    assert _attention_span_total(config, 4, SEQ) == 4 * SEQ


def test_chunked_attention_is_charged_like_a_sliding_layer() -> None:
    """``chunked_attention`` is a real transformers 5.17.0 literal, not a guess."""
    config = SimpleNamespace(layer_types=["chunked_attention"] * 3, sliding_window=256)
    assert _attention_span_total(config, 3, SEQ) == 3 * 256


def test_boolean_sliding_window_is_rejected() -> None:
    """``isinstance(True, int)`` is True; a flag read as a window would charge 1.

    Going through ``_positive_int`` rather than a bare getattr is what makes
    this hold, and a bare getattr passes every other test in this file.
    """
    config = SimpleNamespace(layer_types=["sliding_attention"] * 2, sliding_window=True)
    assert _attention_span_total(config, 2, SEQ) == 2 * SEQ


# --------------------------------------------------------------------------
# 6-8. the strict acceptance rule -- a partial understanding gets the fallback
# --------------------------------------------------------------------------


def test_layer_type_count_disagreeing_with_layers_is_undeclared() -> None:
    """A partial sum over a mismatched list would be a measurement of nothing."""
    config = SimpleNamespace(layer_types=["full_attention"] * 10)
    assert _attention_span_total(config, 64, SEQ) is None


def test_non_string_entry_makes_the_declaration_undeclared() -> None:
    config = SimpleNamespace(layer_types=["full_attention", 3, "full_attention"])
    assert _attention_span_total(config, 3, SEQ) is None


def test_layer_types_that_is_not_a_sequence_is_undeclared() -> None:
    """A string is iterable and would otherwise be summed character by character."""
    assert _attention_span_total(SimpleNamespace(layer_types="full_attention"), 15, SEQ) is None
    assert _attention_span_total(SimpleNamespace(layer_types=None), 1, SEQ) is None


def test_entirely_unrecognised_vocabulary_is_undeclared_not_reconstructed() -> None:
    """An unknown family must not look understood.

    Without the recognised-anchor guard the ``.get`` default reconstructs
    ``layers * sequence_length`` exactly -- the right number by accident, with a
    declared span standing behind it, so the next family with a genuinely cheap
    layer type would be charged in full and nothing would say so.
    """
    config = SimpleNamespace(layer_types=["mamba", "mamba", "mamba"])
    assert _attention_span_total(config, 3, SEQ) is None


def test_unrecognised_entry_beside_a_recognised_one_is_charged_in_full() -> None:
    """One known anchor is enough to measure; the unknown layer pays full span."""
    config = SimpleNamespace(
        layer_types=["linear_attention", "mamba", "full_attention", "linear_attention"]
    )
    assert _attention_span_total(config, 4, SEQ) == 2 * SEQ


def test_tuple_layer_types_is_accepted() -> None:
    """A frozen or round-tripped config can hand back a tuple rather than a list."""
    config = SimpleNamespace(layer_types=("full_attention", "linear_attention"))
    assert _attention_span_total(config, 2, SEQ) == SEQ


# --------------------------------------------------------------------------
# 9-11. composition, and the zero trap
# --------------------------------------------------------------------------


def test_vlm_reads_layer_types_from_text_config_not_the_flat_config() -> None:
    """The composite-config defect class, one field further in.

    The flat ``layer_types`` here is deliberately WRONG (all full attention) so
    that reading it instead of the text sub-config produces a different number
    rather than the same one.
    """
    text_config = SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=2048,
        layer_types=["full_attention", "full_attention", "linear_attention", "linear_attention"],
    )
    config = SimpleNamespace(text_config=text_config, layer_types=["full_attention"] * 4)

    model = FlopsModel.from_hf_config(
        config,
        sequence_length=SEQ,
        parameters=1_000_000,
        non_embedding_parameters=900_000,
    )
    assert model is not None
    assert model.attention_span_total == 2 * SEQ


def test_hybrid_and_moe_corrections_compose() -> None:
    """Both #529 and #530 apply at once; neither overwrites the other.

    Qwen3.5-35B-A3B is both: a 3:1 hybrid AND 256 routed experts with 8 active.
    """
    span = 10 * SEQ
    routed = 30_000_000_000
    model = _flops(
        layers=40,
        hidden_size=4096,
        attention_span_total=span,
        routed_expert_parameters=routed,
        experts_total=256,
        experts_active=8,
    )
    dense = 6 * 30_000_000_000 + 12 * 4096 * span
    inactive = routed * (256 - 8)
    assert model.flops_per_token == dense - 6 * (inactive // 256)
    # And the span correction is really present: the same model with the span
    # undeclared is strictly more expensive.
    naive = _flops(
        layers=40,
        hidden_size=4096,
        routed_expert_parameters=routed,
        experts_total=256,
        experts_active=8,
    )
    assert model.flops_per_token < naive.flops_per_token


def test_declared_span_of_zero_is_not_treated_as_undeclared() -> None:
    """A fully linear-attention model has a legitimate span of exactly zero.

    ``span or layers * sequence_length`` passes every other test here and fails
    this one by 100%, in the direction that flatters the framework.
    """
    model = _flops(non_embedding_parameters=1000, layers=4, hidden_size=8, attention_span_total=0)
    assert model.flops_per_token == 6 * 1000


def test_from_hf_config_wires_the_span_onto_the_model() -> None:
    """The helper being right is worthless if the constructor never calls it."""
    config = SimpleNamespace(
        num_hidden_layers=64,
        hidden_size=5120,
        layer_types=_qwen35_layer_types(64),
    )
    model = FlopsModel.from_hf_config(
        config,
        sequence_length=SEQ,
        parameters=27_000_000_000,
        non_embedding_parameters=26_000_000_000,
    )
    assert model is not None
    assert model.attention_span_total == 16 * SEQ
    assert model.flops_per_token == 6 * 26_000_000_000 + 12 * 5120 * 16 * SEQ
