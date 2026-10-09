# SPDX-License-Identifier: Apache-2.0
"""Part C: agentic_grpo's prompt_mean arm priced through RLTrainer.

WHAT IS CLAIMED: the ``agentic_grpo`` registry entry -- the objective whose
declared reduction is ``prompt_mean`` -- builds its per-row weights with
:func:`prompt_mean_row_weights` over the KEPT rows' own group ids and
supervised-token counts, ONCE per optimizer step, and the loss the step emits
equals the oracle prompt-mean expression over those same kept rows. The
weights are identical whether ``logprob_micro_batch`` prices the whole batch
in one live graph or replays it row-sliced through the leaf-gradient path.
The data-parallel CMD rescales them by ``world_size * P_local / P_global`` --
the peer's active groups folded into ``P_global`` by the collective the
trainer reduces with -- and a step whose ``P_global`` is 0 has no measurable
denominator and is UNMEASURED, never a zero loss priced over rows no gradient
ever touched.

WHAT IS NOT CLAIMED: convergence, real-model loading, equivalence with any
published implementation, or anything about generation quality. The model
below is the deterministic fake the micro-batch / step-metrics harnesses
already use, so every equality here is arithmetic, not sampling luck.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import foundationscale.rl.distributed as distributed_module  # noqa: E402
import foundationscale.rl.trainer as trainer_module  # noqa: E402 (torch importorskip first)
from foundationscale.rl.group_policy_objectives import (  # noqa: E402 (torch importorskip first)
    prompt_mean_row_weights,
)
from foundationscale.rl.trainer import (  # noqa: E402 (torch importorskip first)
    RLTrainConfig,
    RLTrainer,
    TrainerRefusal,
)

_PAD = 0
_TOK_A = 61
_TOK_B = 62
_VOCAB = 96


# --- fake harness, copied from tests/rl/test_trainer_group_family.py ---------


class _FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template = "fake-template"
        self.pad_token_id = _PAD
        self.eos_token = "<eos>"
        self.padding_side = "right"

    def apply_chat_template(self, conversations, tokenize=False, add_generation_prompt=True):
        return [
            ";".join(f"{turn['role']}:{turn['content']}" for turn in conversation)
            for conversation in conversations
        ]

    def __call__(
        self, *, text, return_tensors=None, padding=None, add_special_tokens=None, images=None
    ):
        rows = [[(ord(ch) % 32) + 8 for ch in line] for line in text]
        width = max(len(row) for row in rows)
        input_ids = torch.tensor(
            [[_PAD] * (width - len(row)) + row for row in rows], dtype=torch.long
        )
        return {"input_ids": input_ids, "attention_mask": (input_ids != _PAD).long()}

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [
            "".join("A" if token == _TOK_A else "B" for token in row) for row in sequences.tolist()
        ]


class _FakeModel(torch.nn.Module):
    """Deterministic embedding-logits model: old / current / reference passes
    read IDENTICALLY until the optimiser moves the policy, which is the
    step-1 invariant under test."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(_VOCAB, _VOCAB)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(_VOCAB) * 2.0)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return types.SimpleNamespace(logits=self.embedding(input_ids))

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids,
        attention_mask=None,
        max_new_tokens,
        num_return_sequences,
        do_sample,
        temperature,
        top_p,
        top_k,
        pad_token_id,
        **kwargs,
    ):
        expanded = input_ids.repeat_interleave(num_return_sequences, dim=0)
        continuation = torch.tensor(
            [
                [_TOK_A if row % 2 == 0 else _TOK_B] * max_new_tokens
                for row in range(expanded.shape[0])
            ],
            dtype=torch.long,
        )
        return torch.cat([expanded, continuation], dim=1)


def _w(value: Any) -> Any:
    # The trainer hands the kernel a (rows,) tensor; compare as plain floats.
    return tuple(value.tolist()) if hasattr(value, "tolist") else value


def _install_fake_host(monkeypatch: pytest.MonkeyPatch) -> list[torch.nn.Module]:
    """Patch corpus / prompt surface / model loading; return the load log.

    The loader hands out a FRESH _FakeModel per call and appends it to the
    returned list, so ``loaded[0]`` is the policy and -- iff a reference is
    loaded -- ``loaded[1]`` is the frozen reference copy.
    """
    tokenizer = _FakeTokenizer()
    samples = (
        types.SimpleNamespace(
            sample_id="s0",
            prompt_turns=(("user", "Pick A."),),
            gold="A",
            images=(),
            video=None,
        ),
        types.SimpleNamespace(
            sample_id="s1",
            prompt_turns=(("user", "Now pick the letter A, please."),),
            gold="A",
            images=(),
            video=None,
        ),
    )
    monkeypatch.setattr(
        trainer_module, "load_sharegpt", lambda dataset, gold_key=None: list(samples)
    )
    monkeypatch.setattr(
        trainer_module,
        "resolve_prompt_surface",
        lambda model_id, needs_images: types.SimpleNamespace(
            kind="tokenizer", surface=tokenizer, reason="fake", supports_images=False
        ),
    )
    loaded: list[torch.nn.Module] = []

    def from_pretrained(*args: object, **kwargs: object) -> torch.nn.Module:
        model = _FakeModel()
        loaded.append(model)
        return model

    loader = types.SimpleNamespace(from_pretrained=from_pretrained)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = loader
    fake_transformers.AutoModelForImageTextToText = loader
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return loaded


def _config(algorithm: str, **overrides: Any) -> RLTrainConfig:
    fields: dict[str, Any] = {
        "model": "fake/model",
        "dataset": "unused.jsonl",
        "algorithm": algorithm,
        "group_size": 2,
        "prompts_per_step": 2,
        "max_steps": 2,
        "max_new_tokens": 2,
        "learning_rate": 1e-2,
        "device": "cpu",
    }
    fields.update(overrides)
    return RLTrainConfig(**fields)


# --- continuations with ragged or entirely absent supervision -----------------


def _ragged_generate(
    self: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    num_return_sequences: int,
    **kwargs: Any,
) -> torch.Tensor:
    """_FakeModel.generate variant: one PAD column on rows 1 through 3.

        Generation order is prompt-major ([p0g0, p0g1, p1g0, p1g1]) and the
        response mask counts every non-pad continuation column, so the four rows
        read 2, 1, 1, 1 supervised tokens: group "row-0" denominates its OWN 3
    token?

        read 2, 1, 1, 1 supervised tokens: group "row-0" denominates its own 3
    token?

        tokens and group "row-1" its own 2, over P == 2 active groups. Equal
    tok?

        token totals would reduce any denominator fault to a uniform scale factor
        this fixture could not price; unequal totals make it per-row visible.
    """
    expanded = input_ids.repeat_interleave(num_return_sequences, dim=0)
    continuation = torch.full((expanded.shape[0], max_new_tokens), _TOK_B, dtype=torch.long)
    continuation[1:, -1] = _PAD
    return torch.cat([expanded, continuation], dim=1)


def _all_padded_generate(
    self: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    num_return_sequences: int,
    **kwargs: Any,
) -> torch.Tensor:
    """_FakeModel.generate variant: every continuation column is PAD.

    The rows still score -- the reward reads decoded text and the tokenizer
    decodes pad as a letter like any other token -- while carrying ZERO
    supervised tokens: the all-masked step whose active-group counts are all
    0, so the prompt-mean denominator is P == 0 and UNMEASURED.
    """
    expanded = input_ids.repeat_interleave(num_return_sequences, dim=0)
    continuation = torch.full((expanded.shape[0], max_new_tokens), _PAD, dtype=torch.long)
    return torch.cat([expanded, continuation], dim=1)


class _AlternatingReward:
    """Scores 1, 0, 1, 0 in call order: every group's two rows differ.

    Call order IS generation order ([p0g0, p0g1, p1g0, p1g1]), so both groups
    read [1.0, 0.0] -- real within-group reward variance whatever the decoded
    text says, which is what the padded continuations above need. Four calls
    a step keep the pattern identical on every step.
    """

    def __init__(self) -> None:
        self.calls = 0

    def score(self, *, response: str, gold: str) -> float:
        value = 1.0 if self.calls % 2 == 0 else 0.0
        self.calls += 1
        return value


# --- instrumentation: the step's own priced planes, recorded as-is -----------


def _record_policy_loss_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    real = trainer_module.TensorPolicyLoss

    class _Recording(real):  # type: ignore[misc, valid-type]
        def __call__(self, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return super().__call__(**kwargs)

    monkeypatch.setattr(trainer_module, "TensorPolicyLoss", _Recording)
    return seen


def _record_advantage_results(monkeypatch: pytest.MonkeyPatch, estimator: type) -> list[Any]:
    # The class comes from the trainer's OWN resolved objective, so the patch
    # intercepts the exact call site _one_step uses; nothing is guessed here.
    seen: list[Any] = []
    real = estimator.compute

    def _recording(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(estimator, "compute", _recording)
    return seen


def _record_prompt_mean_weight_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    real = trainer_module.prompt_mean_row_weights

    def _recording(group_ids: Any, supervised_tokens: Any) -> Any:
        weights = real(group_ids, supervised_tokens)
        seen.append(
            {
                "group_ids": tuple(group_ids),
                "counts": tuple(supervised_tokens),
                "weights": weights,
            }
        )
        return weights

    monkeypatch.setattr(trainer_module, "prompt_mean_row_weights", _recording)
    return seen


def _oracle_prompt_mean_loss(
    objective: Any,
    row_weights: tuple[float, ...],
    weight_rows: list[list[float]],
    current: list[list[float]],
    old: list[list[float]],
    mask: list[list[int]],
) -> float:
    """AgenticGRPOLoss's own prompt-mean expression over one step's kept rows.

        WHAT IS CLAIMED: per supervised token ``score = min(ratio * w, clip(ratio,
        low, high) * w)`` at the objective's DECLARED clip bounds and weight with
        ``ratio = exp(current - old)``, reduced as the prompt mean
        :func:`prompt_mean_row_weights` realises -- ``-weight * sum_r w_r * (that
        row's summed scores)`` -- which is the oracle's ``surrogate_total`` priced
        at its own denominator.

        WHAT IS NOT CLAIMED: a re-derivation of the advantage estimator. The
        per-token weight rows are the estimator's own output, taken from the same
    call?

        per-token weight rows are the estimator's own output taken from the same
        call the trainer took them from, so what this can expose is a fault in the
        kernel's pricing or in the row weights, never in the centring.
    """
    low, high = objective.clip_bounds
    surrogate = 0.0
    for row, row_weight in enumerate(row_weights):
        row_total = 0.0
        for position, supervised in enumerate(mask[row]):
            if not supervised:
                continue
            ratio = math.exp(current[row][position] - old[row][position])
            bounded = min(max(ratio, low), high)
            weight = weight_rows[row][position]
            row_total += min(ratio * weight, bounded * weight)
        surrogate += row_weight * row_total
    return -float(objective.weight) * surrogate


# --- agentic_grpo: the prompt_mean denominator, priced and rescaled -----------


def test_prompt_mean_step_loss_matches_the_oracle_over_the_kept_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_host(monkeypatch)
    monkeypatch.setattr(trainer_module, "MCQLetterReward", lambda **kwargs: _AlternatingReward())
    monkeypatch.setattr(_FakeModel, "generate", _ragged_generate)
    objective = RLTrainer(_config("agentic_grpo"))._resolve_objective()
    assert objective.reduction == "prompt_mean", "this test prices the prompt_mean arm"
    assert objective.ratio_scope == "token", "the oracle's ratio is per supervised token"
    advantage_seen = _record_advantage_results(monkeypatch, type(objective.advantage_fn))
    loss_seen = _record_policy_loss_calls(monkeypatch)

    reports = RLTrainer(_config("agentic_grpo")).run()

    assert len(reports) == 2, "both steps carry within-group variance by construction"
    assert len(loss_seen) == len(advantage_seen) == 2, "one kernel call per measured step"
    for report, call, advantage in zip(reports, loss_seen, advantage_seen, strict=True):
        rows = [int(row) for row in advantage.rows]
        assert sorted(rows) == [0, 1, 2, 3], "the estimator priced every scored row"
        mask = [[int(value) for value in row] for row in call["mask"].tolist()]
        counts = [sum(row) for row in mask]
        current = [[float(value) for value in row] for row in call["current_logprobs"].tolist()]
        old = [[float(value) for value in row] for row in call["old_logprobs"].tolist()]
        weight_rows = [[float(value) for value in row] for row in call["advantages"].tolist()]
        # Every row scored in generation order, so the scored row i generated
        # row i and belongs to group i // group_size (group_size=2 below).
        group_ids = [f"row-{index // 2}" for index in rows]
        weights = prompt_mean_row_weights(group_ids, counts)
        # Ragged fixture: rows 0..3 carry 2, 1, 1, 1 supervised tokens, so
        # group "row-0" denominates its own 3 tokens and "row-1" its own 2
        # over P == 2. A uniform 1/P or sequence mean could not separate
        # itself from these numbers row by row.
        assert dict(zip(rows, counts, strict=True)) == {0: 2, 1: 1, 2: 1, 3: 1}
        assert dict(zip(rows, weights, strict=True)) == pytest.approx(
            {0: 1 / 6, 1: 1 / 6, 2: 1 / 4, 3: 1 / 4}
        )
        assert _w(call["reduction_row_weights"]) == weights
        oracle = _oracle_prompt_mean_loss(
            objective,
            row_weights=weights,
            weight_rows=weight_rows,
            current=current,
            old=old,
            mask=mask,
        )
        assert report.rows == 4
        assert report.loss.loss == pytest.approx(oracle, rel=1e-4, abs=1e-6)


def test_prompt_mean_weights_are_per_step_and_micro_batch_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_host(monkeypatch)
    monkeypatch.setattr(trainer_module, "MCQLetterReward", lambda **kwargs: _AlternatingReward())
    monkeypatch.setattr(_FakeModel, "generate", _ragged_generate)
    weights_seen = _record_prompt_mean_weight_calls(monkeypatch)
    loss_seen = _record_policy_loss_calls(monkeypatch)

    whole = RLTrainer(_config("agentic_grpo", logprob_micro_batch=0)).run()
    whole_weights = list(weights_seen)
    whole_loss = list(loss_seen)
    sliced = RLTrainer(_config("agentic_grpo", logprob_micro_batch=1)).run()
    sliced_weights = list(weights_seen[len(whole_weights) :])
    sliced_loss = list(loss_seen[len(whole_loss) :])

    assert len(whole) == len(sliced) == 2
    assert len(whole_weights) == len(sliced_weights) == 2, (
        "the denominator is built once per optimizer step, never once per slice"
    )
    assert whole_weights[0]["counts"] == (2, 1, 1, 1), "per-kept-row counts, in kept order"
    for left, right in zip(whole_weights, sliced_weights, strict=True):
        assert left["group_ids"] == right["group_ids"]
        assert left["counts"] == right["counts"]
        assert left["weights"] == right["weights"]
    for left, right in zip(whole_loss, sliced_loss, strict=True):
        assert _w(left["reduction_row_weights"]) == _w(right["reduction_row_weights"])
    for calls, built in ((whole_loss, whole_weights), (sliced_loss, sliced_weights)):
        for call, weights in zip(calls, built, strict=True):
            assert _w(call["reduction_row_weights"]) == weights["weights"]
    for left, right in zip(whole, sliced, strict=True):
        assert left.rows == right.rows
        assert left.loss.loss == pytest.approx(right.loss.loss, rel=1e-4)


def test_prompt_mean_weights_rescale_by_world_size_and_global_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_host(monkeypatch)
    monkeypatch.setattr(trainer_module, "MCQLetterReward", lambda **kwargs: _AlternatingReward())
    monkeypatch.setattr(_FakeModel, "generate", _ragged_generate)
    objective = RLTrainer(_config("agentic_grpo", max_steps=1))._resolve_objective()
    advantage_seen = _record_advantage_results(monkeypatch, type(objective.advantage_fn))
    loss_seen = _record_policy_loss_calls(monkeypatch)
    remote_active_groups = 1.0
    reductions: list[float] = []

    def _two_rank_all_reduce_sum(value: float, ctx: Any) -> float:
        # Two-rank stand-in: the step's FIRST collective is the prompt-mean
        # denominator reduction and it is the one a peer's active groups
        # enter. Every later reduction keeps this rank's own value, so
        # nothing else in the step moves.
        reductions.append(float(value))
        remote = remote_active_groups if len(reductions) == 1 else 0.0
        return float(value) + remote

    world_size = 2
    ctx = distributed_module.DistContext(
        rank=0,
        world_size=world_size,
        local_rank=0,
        device="cpu",
        # is_distributed=False keeps every OTHER collective the identity this
        # single-process fixture already exercises; the two planes under test
        # are ctx.world_size and the denominator reduction above. Nothing else.
        is_distributed=False,
    )
    monkeypatch.setattr(distributed_module, "init_distributed", lambda sharding: ctx)
    monkeypatch.setattr(distributed_module, "all_reduce_sum", _two_rank_all_reduce_sum)

    reports = RLTrainer(_config("agentic_grpo", max_steps=1)).run()

    assert len(reports) == 1
    assert len(reductions) == 2, (
        "the step reduces twice: the prompt-mean denominator and the report's weight"
    )
    assert reductions[0] == 2.0, "the denominator reduction reports P_local = 2 active groups"
    call, advantage = loss_seen[0], advantage_seen[0]
    rows = [int(row) for row in advantage.rows]
    counts = [int(value) for value in call["mask"].sum(dim=-1).tolist()]
    group_ids = [f"row-{index // 2}" for index in rows]
    base = prompt_mean_row_weights(group_ids, counts)
    local_groups = len({key for key, count in zip(group_ids, counts, strict=True) if count > 0})
    assert local_groups == 2
    p_global = local_groups + remote_active_groups
    scale = float(world_size) * float(local_groups) / p_global
    assert scale == pytest.approx(4 / 3), "world_size * P_local / P_global, 1 peer group"
    expected = tuple(value * scale for value in base)
    assert _w(call["reduction_row_weights"]) == pytest.approx(expected, rel=1e-12)


def test_prompt_mean_zero_global_groups_is_an_unmeasured_step(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Every continuation is pad, so every kept row carries 0 supervised
    # tokens: the prompt-mean denominator is P == 0 and is UNMEASURED, exactly
    # as the all-masked unmeasured-step tests assert -- the run attempted a
    # step and measured nothing, so it is refused as vacuous rather than
    # returned as an empty list that reads like a quiet success.
    _install_fake_host(monkeypatch)
    monkeypatch.setattr(trainer_module, "MCQLetterReward", lambda **kwargs: _AlternatingReward())
    monkeypatch.setattr(_FakeModel, "generate", _all_padded_generate)
    loss_seen = _record_policy_loss_calls(monkeypatch)

    # A kept row always supervises >= 1 token (the advantage stage refuses an
    # all-masked scored row first), so P_global == 0 is reachable only through
    # the collective: every rank reporting 0 active groups. Force that sum.
    monkeypatch.setattr(distributed_module, "all_reduce_sum", lambda value, ctx: 0.0)
    monkeypatch.setattr(_FakeModel, "generate", _ragged_generate)
    trainer = RLTrainer(_config("agentic_grpo"))
    with pytest.raises(TrainerRefusal, match="vacuous"):
        trainer.run()
    err = capsys.readouterr().err
    assert "UNMEASURED step 0" in err
    assert "the prompt_mean denominator is UNMEASURED and never 0.0" in err
    assert loss_seen == [], "the kernel was never priced: no step is claimed"
