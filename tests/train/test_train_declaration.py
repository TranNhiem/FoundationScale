"""Controls for the checkpoint declaration the training entry produces (#225).

The declaration is the DENOMINATOR every checkpoint verdict is computed against.
That makes its failure modes asymmetric, and worth stating before the tests:

  * declare too FEW tensors and a save that dropped the rest still passes --
    a vacuous clear, doctrine 1;
  * declare too MANY and a healthy save reports missing tensors -- a false RED,
    which is what a naive ``state_dict().keys()`` does to every tied-embedding
    model, i.e. to most causal LMs.

Both are covered here, and the tied case is measured against a real number:
MEASURED on hf-internal-testing/tiny-random-gpt2, 65 state_dict keys against 64
tensors in the saved artifact.

The final test is the one that would have caught #225 itself. Every earlier test
asks whether the declaration is internally sensible; that one hands it to the
gate that consumes it and checks the gate can actually adjudicate with it. A
producer and a consumer each internally consistent and never introduced is this
repository's most-repeated defect (#150), and only a round trip refutes it.

No torch, no transformers: ``_declare_checkpoint`` reads a model through three
duck-typed members (``state_dict()``, ``config``, ``_tied_weights_keys``), so
the stubs below are a complete stand-in and the suite runs where CI runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from foundationscale.gates.checkpoint_gates import (
    CheckpointGateContext,
    SaveCompletenessGate,
    TensorMeta,
    matches_expert_family,
    mentions_expert,
)
from foundationscale.train.loop import _declare_checkpoint


class _Tensor:
    """The two members ``_declare_checkpoint`` needs to price a tensor."""

    def __init__(self, n: int = 16, width: int = 2) -> None:
        self._n = n
        self._w = width

    def numel(self) -> int:
        return self._n

    def element_size(self) -> int:
        return self._w


class _Config:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Model:
    def __init__(
        self,
        names: list[str],
        *,
        config: _Config | None = None,
        tied: Any = None,
        sizes: dict[str, _Tensor] | None = None,
    ) -> None:
        self._state = {n: (sizes or {}).get(n, _Tensor()) for n in names}
        self.config = config or _Config(tie_word_embeddings=False)
        if tied is not None:
            self._tied_weights_keys = tied

    def state_dict(self) -> dict[str, _Tensor]:
        return self._state


DENSE_NAMES = [
    "transformer.wte.weight",
    "transformer.h.0.attn.c_attn.weight",
    "transformer.h.0.mlp.c_fc.weight",
    "lm_head.weight",
]


def _moe_names(layers: int = 2, experts: int = 4) -> list[str]:
    out = ["model.embed_tokens.weight"]
    for layer in range(layers):
        for e in range(experts):
            base = f"model.layers.{layer}.mlp.experts.{e}"
            out += [f"{base}.linear_fc1.weight", f"{base}.linear_fc2.weight"]
    return out


# --------------------------------------------------------------------------
# Dense: a POSITIVE declaration, never a default.
# --------------------------------------------------------------------------


def test_dense_declares_zero_experts_with_a_stated_basis() -> None:
    decl, notes = _declare_checkpoint(_Model(DENSE_NAMES))
    assert decl.num_experts == 0
    # 0 is only admissible WITH a basis -- the schema enforces it and #54 is why:
    # absence of a config key must never mint a zero on its own.
    assert decl.moe_layer_basis
    assert "dense" in decl.moe_layer_basis
    # None, not 0. The schema rejects a zero layer count outright; a dense model
    # has no MoE layers to count rather than zero of them.
    assert decl.num_moe_layers is None
    assert decl.expected_expert_bytes is None
    assert set(decl.declared_fqns) == set(DENSE_NAMES)
    assert notes["declaration.expert_named_tensors"] == "0"
    assert notes["declaration.config_expert_keys"] == "(none present)"


def test_declared_fqns_are_sorted_and_deduplicated() -> None:
    # Order is not cosmetic: two runs of the same model must produce byte-equal
    # declarations, or diffing one manifest against another reports noise.
    decl, _ = _declare_checkpoint(_Model(list(reversed(DENSE_NAMES))))
    assert list(decl.declared_fqns) == sorted(DENSE_NAMES)


# --------------------------------------------------------------------------
# Tied weights: the false-RED direction.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tied",
    [
        # transformers 4.x ships a list; 5.x ships an alias -> source mapping.
        # Read by SHAPE, not by version: pinning a version here would make the
        # verdict depend on which transformers happened to be resolved, which is
        # #83/#111's class.
        ["lm_head.weight"],
        {"lm_head.weight": "transformer.wte.weight"},
    ],
    ids=["list-4x", "dict-5x"],
)
def test_tied_alias_is_excluded_and_the_subtraction_is_recorded(tied: Any) -> None:
    model = _Model(DENSE_NAMES, config=_Config(tie_word_embeddings=True), tied=tied)
    decl, notes = _declare_checkpoint(model)
    assert "lm_head.weight" not in decl.declared_fqns
    assert len(decl.declared_fqns) == len(DENSE_NAMES) - 1
    # The subtraction has to be auditable in the artifact. A denominator that
    # silently shrank is indistinguishable from one that was measured small.
    assert notes["declaration.state_dict_keys"] == str(len(DENSE_NAMES))
    assert notes["declaration.tied_excluded"] == "lm_head.weight"


def test_tying_off_keeps_the_alias_because_the_save_will_contain_it() -> None:
    # MUST_FIRE for the exclusion: the same `_tied_weights_keys` with the config
    # flag off must change nothing. The attribute names what WOULD be tied, so
    # reading it alone would drop a tensor the checkpoint really holds -- turning
    # the fix for a false RED into a vacuous PASS one model later.
    model = _Model(DENSE_NAMES, config=_Config(tie_word_embeddings=False), tied=["lm_head.weight"])
    decl, notes = _declare_checkpoint(model)
    assert "lm_head.weight" in decl.declared_fqns
    assert len(decl.declared_fqns) == len(DENSE_NAMES)
    assert notes["declaration.tied_excluded"] == "(none)"


def test_a_tied_name_absent_from_state_dict_is_not_subtracted_twice() -> None:
    model = _Model(
        DENSE_NAMES,
        config=_Config(tie_word_embeddings=True),
        tied=["lm_head.weight", "some.other.head.weight"],
    )
    decl, notes = _declare_checkpoint(model)
    assert len(decl.declared_fqns) == len(DENSE_NAMES) - 1
    assert notes["declaration.tied_excluded"] == "lm_head.weight"


# --------------------------------------------------------------------------
# MoE, and the two-source contract.
# --------------------------------------------------------------------------


def test_moe_declares_the_config_count_and_prices_expert_bytes() -> None:
    names = _moe_names(layers=2, experts=4)
    model = _Model(names, config=_Config(num_experts=4, tie_word_embeddings=False))
    decl, notes = _declare_checkpoint(model)
    assert decl.num_experts == 4
    # 16 experts tensors x 16 elements x 2 bytes.
    assert decl.expected_expert_bytes == 16 * 16 * 2
    assert notes["declaration.priced_expert_tensors"] == "16"
    assert "MoE" in decl.moe_layer_basis


def test_only_verifiable_layouts_are_priced() -> None:
    # An expert-NAMED tensor in a layout the gates cannot parse must be visible
    # to the dense/MoE decision (so the model is not declared dense) and absent
    # from the byte total (so the denominator is not inflated past what any gate
    # can confirm). The two predicates answer different questions, and this is
    # the test that they are not interchangeable.
    odd = "model.layers.0.mlp.experts.gate_router_bias_v2"
    assert mentions_expert(odd) and not matches_expert_family(odd)
    names = [*_moe_names(layers=1, experts=2), odd]
    model = _Model(names, config=_Config(num_experts=2))
    decl, notes = _declare_checkpoint(model)
    assert decl.num_experts == 2
    assert decl.expected_expert_bytes == 4 * 16 * 2
    assert notes["declaration.expert_named_tensors"] == "5"
    assert notes["declaration.priced_expert_tensors"] == "4"
    # It is still DECLARED -- unpriceable is not unowned. A tensor left out of
    # declared_fqns is one the completeness gate will never notice going missing.
    assert odd in decl.declared_fqns


def test_top_k_is_not_read_as_an_expert_count() -> None:
    # `num_experts_per_tok` is on every MoE config and is the router's top-k.
    # Reading it as a count declares 2 experts for a 128-expert layer, and the
    # byte gate then CONFIRMS a checkpoint that is 98% short. Excluded from
    # _EXPERT_COUNT_KEYS on purpose; this is the test that keeps it excluded.
    model = _Model(DENSE_NAMES, config=_Config(num_experts_per_tok=2))
    decl, _ = _declare_checkpoint(model)
    assert decl.num_experts == 0
    assert "dense" in decl.moe_layer_basis


def test_config_says_moe_but_nothing_is_expert_named_declares_unknown() -> None:
    # One source says MoE, the other says dense. Adopting either would be a
    # guess; None makes the gates fail closed, which is the declared behaviour
    # of the schema ("None means NOTHING was declared").
    model = _Model(DENSE_NAMES, config=_Config(num_experts=8))
    decl, _ = _declare_checkpoint(model)
    assert decl.num_experts is None
    assert "UNKNOWN" in decl.moe_layer_basis


def test_disagreeing_expert_count_keys_declare_unknown() -> None:
    names = _moe_names(layers=1, experts=2)
    model = _Model(names, config=_Config(num_experts=8, n_routed_experts=64))
    decl, _ = _declare_checkpoint(model)
    assert decl.num_experts is None
    assert "UNKNOWN" in decl.moe_layer_basis


def test_agreeing_duplicate_keys_are_not_a_conflict() -> None:
    # Several families spell the same number twice. Agreement is agreement.
    names = _moe_names(layers=1, experts=8)
    model = _Model(names, config=_Config(num_experts=8, num_local_experts=8))
    decl, _ = _declare_checkpoint(model)
    assert decl.num_experts == 8


def test_a_model_without_a_config_still_declares_from_its_tensors() -> None:
    model = _Model(DENSE_NAMES)
    del model.config
    decl, notes = _declare_checkpoint(model)
    assert decl.num_experts == 0
    assert notes["declaration.config_expert_keys"] == "(none present)"


# --------------------------------------------------------------------------
# The round trip. This is the test that #225 needed and did not have.
# --------------------------------------------------------------------------


def _ctx(model: _Model, decl: Any, *, drop: str | None = None) -> CheckpointGateContext:
    """The declaration alongside the artifact it is meant to adjudicate."""
    saved = [n for n in decl.declared_fqns if n != drop]
    return CheckpointGateContext(
        tensors=tuple(
            TensorMeta(fqn=n, shape=(4, 4), dtype="bfloat16", storage_id=n, kind="tensor")
            for n in saved
        ),
        declared_fqns=tuple(decl.declared_fqns),
        num_experts=decl.num_experts,
        num_moe_layers=decl.num_moe_layers,
        expected_expert_bytes=decl.expected_expert_bytes,
        origin="test://roundtrip",
    )


def test_the_declaration_is_readable_by_the_gate_that_consumes_it() -> None:
    model = _Model(DENSE_NAMES, config=_Config(tie_word_embeddings=True), tied=["lm_head.weight"])
    decl, _ = _declare_checkpoint(model)
    result = SaveCompletenessGate().check(_ctx(model, decl))
    assert not result.blocking, result.detail
    # A PASS over zero units is the vacuous clear this codebase exists to
    # refuse, so the denominator is asserted, not just the verdict.
    assert result.coverage.checked == len(DENSE_NAMES) - 1 > 0


def test_the_same_declaration_still_catches_a_dropped_tensor() -> None:
    # MUST_FIRE for the round trip above. A declaration the gate accepts but
    # cannot fail with is worth nothing -- it would have made #225 pass.
    model = _Model(DENSE_NAMES, config=_Config(tie_word_embeddings=True), tied=["lm_head.weight"])
    decl, _ = _declare_checkpoint(model)
    result = SaveCompletenessGate().check(_ctx(model, decl, drop="transformer.wte.weight"))
    assert result.blocking
    assert result.evidence["missing"] == ["transformer.wte.weight"]
    # The denominator survives the failure: 3 declared, 2 found. A gate that
    # reported "2 of 2" on a dropped tensor would be measuring the artifact
    # against itself.
    assert result.coverage.checked == len(DENSE_NAMES) - 2
