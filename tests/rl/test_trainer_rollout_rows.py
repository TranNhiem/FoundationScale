# SPDX-License-Identifier: Apache-2.0
"""Slice 4a: externally generated multi-turn rows into RLTrainer (HF/FSDP2 lane).

WHAT IS CLAIMED: ``RLTrainer._planes_from_batch`` builds the SAME shape of
kept planes (``kept_sequences``/``response_mask``/``attention``) the built-in
encode -> generate -> decode -> score leg builds for a completion-only mask,
and additionally represents an ARBITRARY per-token mask (tool-result tokens
masked mid-response) that leg cannot; a row with ``reward is None`` is
dropped as an abstention, never priced at 0.0; a malformed row (a loss_mask
whose length disagrees with its response, or a batch that supervises 0
tokens after abstention-dropping) is refused by name; ``RLTrainer.run()``
with a ``rollout_source`` configured skips generation entirely, shards whole
groups by the batch's own ``prompt_ids`` column, prices each step through
``_one_step_rows`` / ``_priced_tail``, and calls ``rollout_source.publish``
after every measured step; and ``rollout_source`` is refused together with
ppo in ``_resolve_objective`` before any row is ever read. ``RLTrainConfig.
dataset`` is ``str | None``: declaring it TOGETHER with ``rollout_source``
refuses (exit 96) by name (the rollout source replaces the corpus draw, so a
declared dataset would never be read), declaring NEITHER refuses too (the
built-in leg has no corpus to draw from), and under ``rollout_source`` with
``dataset=None`` no corpus is loaded at all -- ``load_sharegpt`` is never
called.

WHAT IS NOT CLAIMED: convergence, real-model loading, or equivalence with
any published implementation. The fake host is the deterministic
embedding-logits model test_trainer_group_family.py already uses, so every
equality here is arithmetic, not sampling luck. Full generate-leg-vs-rows-leg
equivalence (spec part (a)) is NOT exercised end to end through ``run()`` --
the escape hatch the spec grants instead: ``_planes_from_batch`` is checked
against the built-in leg's OWN completion-only-mask shape (one contiguous
supervised run from the prompt/response boundary onward, same padding and
attention rule), by construction and by direct arithmetic.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from test_trainer_group_family import (  # noqa: E402 (sibling test module, no package)
    _config,
    _FakeModel,
    _install_fake_host,
)

import foundationscale.rl.trainer as trainer_module  # noqa: E402
from foundationscale.rl.interfaces import ExperienceBatch  # noqa: E402
from foundationscale.rl.torch_backend import TensorPolicyLoss  # noqa: E402
from foundationscale.rl.trainer import RLTrainer, TrainerRefusal  # noqa: E402

_PAD = 0


def _batch(**columns: Any) -> ExperienceBatch:
    return ExperienceBatch(columns=columns)


# --- (a) equivalence escape hatch: _planes_from_batch vs the built-in shape --


def test_planes_from_batch_matches_the_built_in_completion_only_mask() -> None:
    """A completion-only mask (every response token supervised) must come out
    in the SAME shape the built-in leg's single prompt/response boundary
    produces: right-padded by length, attention 1 up to each row's own
    length, and response_mask 1 everywhere from that row's own boundary on.
    """
    trainer = RLTrainer(_config("agentic_grpo"))
    batch = _batch(
        prompt_ids=("g0", "g0", "g1", "g1"),
        # Ragged prompt widths (3 vs 2 tokens) -- the built-in leg can only
        # offer this via padding since one encode() call serves every row;
        # per-row planes must still agree with it on the padded shape.
        prompt_token_ids=((10, 11, 12), (10, 11, 12), (20, 21), (20, 21)),
        response_ids=((30, 31), (32, 33), (34, 35), (36, 37)),
        loss_mask=((1, 1), (1, 1), (1, 1), (1, 1)),
        reward=(1.0, 0.0, 1.0, 0.0),
    )

    kept_sequences, response_mask, attention, rewards, group_ids, kept_row_indices = (
        trainer._planes_from_batch(batch, device="cpu", pad_token_id=_PAD)
    )

    assert kept_sequences.tolist() == [
        [10, 11, 12, 30, 31],
        [10, 11, 12, 32, 33],
        [20, 21, 34, 35, _PAD],
        [20, 21, 36, 37, _PAD],
    ]
    assert attention.tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
    ]
    # Hand-derived from "target t predicts position t+1": rows 0/1 (prompt
    # width 3) supervise t=2,3; rows 2/3 (prompt width 2) supervise t=1,2 --
    # one contiguous run from each row's OWN boundary, same as the built-in
    # leg's single-boundary mask, just not sharing one global boundary.
    assert response_mask.tolist() == [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ]
    assert rewards.tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0])
    assert group_ids == ["g0", "g0", "g1", "g1"]
    assert kept_row_indices == [0, 1, 2, 3]


# --- (b) multi-turn mask: a tool-masked token contributes no gradient -------


def test_tool_masked_token_gets_zero_response_mask_and_zero_gradient() -> None:
    """Token 9 is row 0's tool result (loss_mask 0): it is the TARGET of the
    one masked position (predicting it earns no supervision) and, being the
    response's last token, is never read as an INPUT to any kept logit
    column either (the kernel narrows logits to ``width - 1`` columns, so
    the last sequence position is never queried as input). Its embedding
    ROW must see no gradient at all, while the shared prompt token's row
    (read as input at every supervised position, in both rows) does.
    """
    trainer = RLTrainer(_config("agentic_grpo"))
    batch = _batch(
        prompt_ids=("g0", "g0"),
        prompt_token_ids=((1, 2), (1, 2)),
        # Row 0: response token 3 is supervised; tool token 9 is not.
        # Row 1: both response tokens supervised, real within-group variance.
        response_ids=((3, 9), (4, 5)),
        loss_mask=((1, 0), (1, 1)),
        reward=(1.0, 0.0),
    )
    kept_sequences, response_mask, attention, rewards, group_ids, _ = trainer._planes_from_batch(
        batch, device="cpu", pad_token_id=_PAD
    )
    # t predicts position t+1: t=0 predicts the prompt's own last token (not
    # response), t=1 predicts response position 0 (token 3, supervised),
    # t=2 predicts response position 1 (token 9, the masked tool result).
    assert response_mask.tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]

    model = _FakeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    objective = trainer._resolve_objective()
    loss_fn = TensorPolicyLoss(objective=objective)

    from foundationscale.rl.distributed import DistContext

    ctx = DistContext(rank=0, world_size=1, local_rank=0, device="cpu", is_distributed=False)
    report = trainer._priced_tail(
        step=0,
        model=model,
        optimizer=optimizer,
        ref_model=None,
        device="cpu",
        ctx=ctx,
        objective=objective,
        loss_fn=loss_fn,
        kept_sequences=kept_sequences,
        response_mask=response_mask,
        attention=attention,
        modality_kwargs={},
        rewards=rewards,
        group_ids=group_ids,
        null_rank=False,
        rows=None,
    )

    assert report is not None
    grad = model.embedding.weight.grad
    assert grad is not None
    assert torch.equal(grad[9], torch.zeros_like(grad[9])), "the masked tool token must see no grad"
    assert bool(grad[2].abs().any()), "the supervised shared prompt token must see a grad"


# --- (c) None rewards are dropped, never priced as 0.0 ----------------------


def test_planes_from_batch_drops_none_reward_rows(capsys: pytest.CaptureFixture[str]) -> None:
    trainer = RLTrainer(_config("agentic_grpo"))
    batch = _batch(
        prompt_ids=("g0", "g0", "g1"),
        prompt_token_ids=((1, 2), (1, 2), (1, 2)),
        response_ids=((3, 4), (5, 6), (7, 8)),
        loss_mask=((1, 1), (1, 1), (1, 1)),
        reward=(1.0, None, 0.0),
    )

    kept_sequences, _, _, rewards, group_ids, kept_row_indices = trainer._planes_from_batch(
        batch, device="cpu", pad_token_id=_PAD
    )

    assert kept_row_indices == [0, 2]
    assert rewards.tolist() == pytest.approx([1.0, 0.0])
    assert group_ids == ["g0", "g1"]
    assert kept_sequences.shape[0] == 2
    assert "dropped 1 of 3" in capsys.readouterr().err


# --- (d) named refusals: mask/response length mismatch, all-zero -----------


def test_planes_from_batch_refuses_loss_mask_length_mismatch() -> None:
    trainer = RLTrainer(_config("agentic_grpo"))
    batch = _batch(
        prompt_ids=("g0",),
        prompt_token_ids=((1, 2),),
        response_ids=((3, 4, 5),),
        loss_mask=((1, 1),),  # 2 entries for a 3-token response
        reward=(1.0,),
    )
    with pytest.raises(TrainerRefusal, match=r"loss_mask has 2 entries.*response of 3 token"):
        trainer._planes_from_batch(batch, device="cpu", pad_token_id=_PAD)


def test_planes_from_batch_refuses_all_zero_supervision() -> None:
    trainer = RLTrainer(_config("agentic_grpo"))
    batch = _batch(
        prompt_ids=("g0", "g0"),
        prompt_token_ids=((1, 2), (1, 2)),
        response_ids=((3, 4), (5, 6)),
        loss_mask=((0, 0), (0, 0)),
        reward=(1.0, 0.0),
    )
    with pytest.raises(TrainerRefusal, match="supervise 0 response token"):
        trainer._planes_from_batch(batch, device="cpu", pad_token_id=_PAD)


def test_one_step_rows_is_unmeasured_when_every_row_abstains(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = RLTrainer(_config("agentic_grpo"))
    objective = trainer._resolve_objective()
    loss_fn = TensorPolicyLoss(objective=objective)
    batch = _batch(
        prompt_ids=("g0", "g0"),
        prompt_token_ids=((1, 2), (1, 2)),
        response_ids=((3, 4), (5, 6)),
        loss_mask=((1, 1), (1, 1)),
        reward=(None, None),
    )
    model = _FakeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = trainer._one_step_rows(
        0,
        batch,
        model=model,
        optimizer=optimizer,
        ref_model=None,
        objective=objective,
        loss_fn=loss_fn,
        device="cpu",
        pad_token_id=_PAD,
    )

    assert result is None
    assert "UNMEASURED step 0" in capsys.readouterr().err


# --- (e) run() with a fake rollout_source: no generate, publish each step --


class _FakeRolloutSource:
    def __init__(self, batches: list[ExperienceBatch]) -> None:
        self._batches = batches
        self.rollout_calls: list[int] = []
        self.publish_calls: list[int] = []

    def rollout(self, step: int) -> ExperienceBatch:
        self.rollout_calls.append(step)
        return self._batches[step]

    def publish(self, model: Any, tokenizer: Any, ctx: Any, step: int) -> None:
        self.publish_calls.append(step)


def _rows_batch() -> ExperienceBatch:
    # Two groups of two, within-group reward variance in each -- the same
    # shape test_trainer_group_family's own fixtures use to guarantee a
    # measured (non-UNMEASURED) step.
    return _batch(
        prompt_ids=("g0", "g0", "g1", "g1"),
        prompt_token_ids=((1, 2), (1, 2), (1, 2), (1, 2)),
        response_ids=((10, 11), (12, 13), (14, 15), (16, 17)),
        loss_mask=((1, 1), (1, 1), (1, 1), (1, 1)),
        reward=(1.0, 0.0, 1.0, 0.0),
    )


def test_run_with_rollout_source_never_generates_and_publishes_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_host(monkeypatch)

    def _refuse_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("model.generate must never be called on the rollout_source path")

    monkeypatch.setattr(_FakeModel, "generate", _refuse_generate)

    source = _FakeRolloutSource([_rows_batch(), _rows_batch()])
    reports = RLTrainer(
        _config("agentic_grpo", max_steps=2, rollout_source=source, dataset=None)
    ).run()

    assert len(reports) == 2
    assert source.rollout_calls == [0, 1]
    assert source.publish_calls == [0, 1]


# --- (e2) RLTrainConfig.dataset/rollout_source: declared together or not at
# all, and the rollout_source path loads NO corpus ---------------------------


def test_run_refuses_dataset_declared_together_with_rollout_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_host(monkeypatch)
    source = _FakeRolloutSource([_rows_batch()])
    trainer = RLTrainer(
        _config("agentic_grpo", max_steps=1, rollout_source=source, dataset="some.jsonl")
    )
    with pytest.raises(SystemExit) as excinfo:
        trainer.run()
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "dataset='some.jsonl'" in err
    assert "rollout_source" in err


def test_run_refuses_dataset_none_without_rollout_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_fake_host(monkeypatch)
    trainer = RLTrainer(_config("agentic_grpo", dataset=None))
    with pytest.raises(SystemExit) as excinfo:
        trainer.run()
    assert excinfo.value.code == 96
    assert "dataset=None without a rollout_source" in capsys.readouterr().err


def test_run_with_rollout_source_and_no_dataset_loads_no_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_host(monkeypatch)

    def _refuse_load_sharegpt(dataset: Any, gold_key: Any = None) -> Any:
        raise AssertionError(
            "load_sharegpt must never be called when rollout_source replaces the corpus"
        )

    # Overrides _install_fake_host's own (permissive) load_sharegpt stub with
    # one that fails loudly if ever reached.
    monkeypatch.setattr(trainer_module, "load_sharegpt", _refuse_load_sharegpt)

    source = _FakeRolloutSource([_rows_batch()])
    reports = RLTrainer(
        _config("agentic_grpo", max_steps=1, rollout_source=source, dataset=None)
    ).run()

    assert len(reports) == 1
    assert source.rollout_calls == [0]


# --- (f) refusal: rollout_source together with ppo -------------------------


def test_resolve_objective_refuses_ppo_with_rollout_source() -> None:
    trainer = RLTrainer(_config("ppo", rollout_source=object()))
    with pytest.raises(TrainerRefusal, match="ppo"):
        trainer._resolve_objective()


def test_resolve_objective_refuses_estimator_free_with_rollout_source() -> None:
    trainer = RLTrainer(_config("reinforce_baseline", rollout_source=object()))
    with pytest.raises(TrainerRefusal, match="estimator-free"):
        trainer._resolve_objective()
