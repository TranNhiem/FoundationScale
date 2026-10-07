"""CPU tests for the Megatron online-rollout lane (RUNG 2).

Every test imports torch at FUNCTION scope on purpose: the module under test must
import with no torch installed at all, so a module-level ``import torch`` here would
hide the very regression ``test_module_import_does_not_import_torch`` guards. The
repo forbids skips (``FS_FORBID_SKIPS=1``), so nothing is skipped -- the fakes below
stand in for HF generate/tokenizer modules on the CPU-only test box.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import pytest

from foundationscale.rl.megatron.online import (
    OnlineRow,
    broadcast_rows,
    collate_token_batch,
    generate_group_rows,
    greedy_heldout,
    refit_hf_policy,
    render_prompt_ids,
    sample_prompt_indices,
    scored_group_advantages,
    shard_rows,
    step_rollout_metrics,
)


@dataclass(frozen=True, slots=True)
class _FakeSample:
    """Stand-in for foundationscale.rl.corpus.Sample (prompt_turns, gold)."""

    prompt_turns: tuple[tuple[str, str], ...]
    gold: str | None


class _FakeTokenizer:
    """Minimal chat tokenizer: rendered ids per prompt, EOS/pad ids, decode."""

    def __init__(
        self,
        *,
        eos_id: int = 2,
        pad_id: int | None = 0,
        rendered: dict[str, list[int]] | None = None,
        as_dict: bool = False,
    ) -> None:
        self.eos_token_id = eos_id
        self.pad_token_id = pad_id
        self.padding_side = "right"
        self.rendered = rendered or {}
        self.as_dict = as_dict
        self.templates: list[list[dict[str, str]]] = []
        self.decodes: list[list[int]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> Any:
        self.templates.append(list(messages))
        ids = list(self.rendered[messages[-1]["content"]])
        return {"input_ids": [list(ids)]} if self.as_dict else list(ids)

    def decode(self, ids: Any, *, skip_special_tokens: bool) -> str:
        self.decodes.append(list(ids))
        return "<" + ",".join(str(token) for token in ids) + ">"


class _FakeReward:
    """Scripted verifiable scorer: canned verdicts, recorded (response, gold)."""

    def __init__(self, values: list[float | None]) -> None:
        self._values = list(values)
        self.calls: list[tuple[str, str | None]] = []

    def score(self, *, response: str, gold: str | None) -> float | None:
        self.calls.append((response, gold))
        return self._values.pop(0)


class _FakeGenerate:
    """Stand-in for ``hf_model.generate``: scripted continuations, real kwargs."""

    def __init__(self, batches: list[Any]) -> None:
        self._batches = list(batches)
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[Any] = []
        self.evaluated = False
        self.exhausted = False

    def eval(self) -> _FakeGenerate:
        self.evaluated = True
        return self

    def generate(self, input_ids: Any, attention_mask: Any, **kwargs: Any) -> Any:
        import torch

        assert torch.is_grad_enabled() is False, "generation must run under torch.no_grad()"
        assert self._batches, "generate() called once too often"
        self.calls.append(kwargs)
        self.prompts.append(input_ids)
        new = self._batches.pop(0).to(input_ids.device)
        rows = new.shape[0]
        prompt = input_ids if input_ids.shape[0] == rows else input_ids.expand(rows, -1)
        out = torch.cat([prompt, new.to(dtype=torch.long)], dim=1)
        self.exhausted = not self._batches
        return out


class _FakeBridge:
    """Stand-in for the Megatron-Bridge exporter; counts what was consumed."""

    def __init__(self, tensors: list[tuple[str, Any]]) -> None:
        self._tensors = list(tensors)
        self.consumed = 0

    def export_hf_weights(
        self, model_list: Any, cpu: bool = False, show_progress: bool = False
    ) -> Any:
        assert cpu is False and show_progress is False
        for name, tensor in self._tensors:
            self.consumed += 1
            yield name, tensor


def _tiny_hf(tied: bool) -> Any:
    """A tiny torch.nn.Module whose state dict names the fake bridge yields."""
    import torch

    class _Config:
        def __init__(self, value: bool) -> None:
            self.tie_word_embeddings = value

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.config = _Config(tied)

    model = _Tiny()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


# -- sample_prompt_indices -------------------------------------------------


def test_sample_prompt_indices_deterministic_distinct_and_bounded() -> None:
    first = sample_prompt_indices(20, 5, seed=7, step=3)
    assert first == sample_prompt_indices(20, 5, seed=7, step=3)
    assert sample_prompt_indices(20, 5, seed=7, step=4) == sample_prompt_indices(
        20, 5, seed=7, step=4
    )
    assert sample_prompt_indices(20, 5, seed=8, step=3) == sample_prompt_indices(
        20, 5, seed=8, step=3
    )
    assert len(first) == 5
    assert len(set(first)) == 5
    assert all(0 <= index < 20 for index in first)
    assert sorted(sample_prompt_indices(4, 4, seed=1, step=0)) == [0, 1, 2, 3]


def test_sample_prompt_indices_refuses_bad_counts() -> None:
    with pytest.raises(ValueError, match="k 5 exceeds n_items 4"):
        sample_prompt_indices(4, 5, seed=1, step=0)
    with pytest.raises(ValueError, match="k 0 is below"):
        sample_prompt_indices(4, 0, seed=1, step=0)
    with pytest.raises(ValueError, match="k -3 is below"):
        sample_prompt_indices(4, -3, seed=1, step=0)


# -- scored_group_advantages ------------------------------------------------


def test_scored_group_advantages_normalizes_group_and_masks_abstention() -> None:
    rows = [
        OnlineRow((1, 2), (3,), 1.0, 0, True),
        OnlineRow((1, 2), (3, 3), 0.0, 0, True),
        OnlineRow((1, 2), (7, 7), None, 0, False),
        OnlineRow((1, 2), (4,), 2.0, 1, True),
    ]
    advantages, sample_mask = scored_group_advantages(rows)
    half = 0.5 / (0.5 + 1e-4)
    assert advantages[0] == pytest.approx(half)
    assert advantages[1] == pytest.approx(-half)
    # A single scored member is a zero-variance group: advantage 0, measured (mask 1).
    assert advantages[3] == 0.0
    # The abstention is UNMEASURED: it is excluded from the stats and moves nothing.
    assert advantages[2] == 0.0
    assert sample_mask == [1.0, 1.0, 0.0, 1.0]


def test_scored_group_advantages_masks_empty_completion_without_biasing_group() -> None:
    rows = [
        OnlineRow((1,), (), 1.0, 0, True),
        OnlineRow((1,), (9,), 0.0, 0, True),
    ]
    advantages, sample_mask = scored_group_advantages(rows)
    half = 0.5 / (0.5 + 1e-4)
    # Its reward still counts toward the group mean (masking a measured reward out of
    # the mean would bias the very group it belongs to); only the mask stops it.
    assert advantages[0] == pytest.approx(half)
    assert advantages[1] == pytest.approx(-half)
    assert sample_mask == [0.0, 1.0]


def test_scored_group_advantages_all_equal_group_is_flat_but_measured() -> None:
    rows = [
        OnlineRow((1,), (2,), 0.5, 0, False),
        OnlineRow((1,), (3,), 0.5, 0, True),
        OnlineRow((1,), (4,), 0.5, 7, False),
    ]
    advantages, sample_mask = scored_group_advantages(rows)
    assert advantages == [0.0, 0.0, 0.0]
    assert sample_mask == [1.0, 1.0, 1.0]


# -- collate_token_batch ----------------------------------------------------


def test_collate_token_batch_uses_row_ids_and_masks_completion_only() -> None:
    import torch

    rows = [
        OnlineRow((10, 11), (12, 13), 1.0, 0, True),
        OnlineRow((20,), (21,), 0.0, 1, True),
    ]
    batch = collate_token_batch(rows, [0.5, -0.5], [1.0, 1.0], seq_len=8, pad_id=0)
    # No tokenizer: the ids are copied verbatim (0 is pad fill, never a re-tokenisation).
    assert batch["input_ids"].tolist() == [[10, 11, 12, 13], [20, 21, 0, 0]]
    assert batch["input_ids"].dtype == torch.long
    assert batch["loss_mask"].tolist() == [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
    assert batch["loss_mask"].dtype == torch.float32
    assert batch["attention_mask"].tolist() == [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
    assert batch["sample_mask"].tolist() == [1.0, 1.0]
    assert batch["advantages"].tolist() == [0.5, -0.5]
    with pytest.raises(ValueError, match="2 rows but 1 advantages and 2 sample masks"):
        collate_token_batch(rows, [0.5], [1.0, 1.0], seq_len=8, pad_id=0)


def test_collate_token_batch_keeps_eos_completion_token_when_pad_is_eos() -> None:
    rows = [OnlineRow((1, 2), (7, 3), 1.0, 0, True)]
    batch = collate_token_batch(rows, [0.25], [1.0], seq_len=8, pad_id=7)
    # pad_id == eos_id == 7, but 7 is a real completion token: attention must stay 1,
    # and pad-scan masking would have erased exactly the stop token.
    assert batch["input_ids"].tolist() == [[1, 2, 7, 3]]
    assert batch["loss_mask"].tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert batch["attention_mask"].tolist() == [[1.0, 1.0, 1.0, 1.0]]
    assert batch["attention_mask"].sum().item() == 4.0


def test_collate_token_batch_masks_row_when_truncation_drops_completion() -> None:
    rows = [
        OnlineRow((1, 2, 3, 4, 5, 6), (9,), 1.0, 0, True),
        OnlineRow((1, 2, 3), (4, 5), 1.0, 1, True),
    ]
    batch = collate_token_batch(rows, [0.5, -0.5], [1.0, 1.0], seq_len=6, pad_id=0)
    # Row 0 truncates to 6 prompt tokens: no completion survives, so it is unmeasured.
    assert batch["sample_mask"].tolist() == [0.0, 1.0]
    assert batch["loss_mask"].tolist()[0] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Row 1 (3 prompt + 2 completion tokens) fits the window whole and stays measured.
    assert batch["loss_mask"].tolist()[1] == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert batch["advantages"].tolist() == [0.5, -0.5]


# -- shard_rows -------------------------------------------------------------


def test_shard_rows_returns_contiguous_whole_group_slices() -> None:
    rows = [OnlineRow((i,), (i,), 0.0, i // 2, False) for i in range(8)]
    assert shard_rows(rows, dp_size=2, dp_rank=0, group_size=2) == (0, 4)
    assert shard_rows(rows, dp_size=2, dp_rank=1, group_size=2) == (4, 8)
    assert shard_rows(rows, dp_size=4, dp_rank=3, group_size=2) == (6, 8)
    assert shard_rows(rows, dp_size=8, dp_rank=7, group_size=1) == (7, 8)


def test_shard_rows_refuses_uneven_or_group_cutting_shards() -> None:
    rows = [OnlineRow((i,), (i,), 0.0, i // 3, False) for i in range(6)]
    with pytest.raises(ValueError, match="6 rows do not split into 4 equal shards"):
        shard_rows(rows, dp_size=4, dp_rank=0, group_size=1)
    with pytest.raises(ValueError, match="3 rows per shard over dp_size 2"):
        shard_rows(rows, dp_size=2, dp_rank=0, group_size=2)
    with pytest.raises(ValueError, match="dp_rank 2 is outside dp_size 2"):
        shard_rows(rows, dp_size=2, dp_rank=2, group_size=3)
    with pytest.raises(ValueError, match="dp_size 0 is below 1"):
        shard_rows(rows, dp_size=0, dp_rank=0, group_size=2)
    with pytest.raises(ValueError, match="group_size 0 is below 1"):
        shard_rows(rows, dp_size=2, dp_rank=0, group_size=0)


# -- refit_hf_policy --------------------------------------------------------


def test_refit_hf_policy_writes_every_yielded_weight_into_the_hf_state_dict() -> None:
    import torch

    model = _tiny_hf(tied=False)
    ones = torch.ones(8, 4, dtype=torch.float64)  # wrong dtype on purpose
    twos = torch.full((8, 4), 2.0, dtype=torch.float64)
    bridge = _FakeBridge(
        [("embed_tokens.weight", ones), ("lm_head.weight", twos), ("embed_tokens.weight", ones)]
    )
    counts = refit_hf_policy(bridge, [], model, is_writer=True)
    # Distinct names written: 2 of 2 HF entries, 0 left unwritten.
    assert counts == {"written": 2, "unwritten": 0}
    assert bridge.consumed == 3  # even the duplicate yield is consumed to exhaustion
    assert torch.allclose(model.embed_tokens.weight, torch.ones(8, 4))  # dtype preserved
    assert model.embed_tokens.weight.dtype == torch.float32
    assert torch.allclose(model.lm_head.weight, torch.full((8, 4), 2.0))


def test_refit_hf_policy_raises_on_name_missing_after_draining_the_export() -> None:
    import torch

    model = _tiny_hf(tied=True)
    bridge = _FakeBridge(
        [
            ("embed_tokens.weight", torch.ones(8, 4)),
            ("ghost.weight", torch.ones(8, 4)),
        ]
    )
    with pytest.raises(ValueError, match="ghost.weight"):
        refit_hf_policy(bridge, [], model, is_writer=True)
    # The generator is drained BEFORE the raise: peers sit inside a collective.
    assert bridge.consumed == 2


def test_refit_hf_policy_non_writer_drains_the_export_but_writes_nothing() -> None:
    import torch

    model = _tiny_hf(tied=False)
    bridge = _FakeBridge(
        [("embed_tokens.weight", torch.ones(8, 4)), ("lm_head.weight", torch.ones(8, 4))]
    )
    counts = refit_hf_policy(bridge, [], model, is_writer=False)
    assert counts == {"written": 0, "unwritten": 0}
    assert bridge.consumed == 2  # the export is collective: every rank iterates it out
    assert model.embed_tokens.weight.abs().sum().item() == 0.0
    assert model.lm_head.weight.abs().sum().item() == 0.0
    # Production peers hold NO HF copy: the driver passes None on every non-writer rank.
    peer_bridge = _FakeBridge([("embed_tokens.weight", torch.ones(8, 4))])
    assert refit_hf_policy(peer_bridge, [], None, is_writer=False) == {"written": 0, "unwritten": 0}
    assert peer_bridge.consumed == 1


def test_refit_hf_policy_tied_lm_head_is_never_reported_unwritten() -> None:
    import torch

    tied = _tiny_hf(tied=True)
    tied_bridge = _FakeBridge([("embed_tokens.weight", torch.ones(8, 4))])
    assert refit_hf_policy(tied_bridge, [], tied, is_writer=True) == {
        "written": 1,
        "unwritten": 0,
    }

    untied = _tiny_hf(tied=False)
    untied_bridge = _FakeBridge([("embed_tokens.weight", torch.ones(8, 4))])
    # Only the tie case is excluded: an untied head the export skipped IS reported.
    assert refit_hf_policy(untied_bridge, [], untied, is_writer=True) == {
        "written": 1,
        "unwritten": 1,
    }


# -- broadcast_rows / step_rollout_metrics ----------------------------------


def test_broadcast_rows_returns_input_when_dist_is_uninitialized() -> None:
    rows = [OnlineRow((1, 2), (3,), 1.0, 0, True)]
    assert broadcast_rows(rows) == rows
    assert broadcast_rows(rows, src=0) == rows
    with pytest.raises(ValueError, match="rows is None but torch.distributed is not initialized"):
        broadcast_rows(None)


def test_step_rollout_metrics_counts_scored_abstained_and_flat_groups() -> None:
    rows = [
        OnlineRow((1,), (2,), 1.0, 0, True),
        OnlineRow((1,), (3, 3), 0.0, 0, False),
        OnlineRow((1,), (), None, 1, False),
        OnlineRow((1,), (4,), 0.5, 2, True),
        OnlineRow((1,), (5,), 0.5, 2, False),
    ]
    metrics = step_rollout_metrics(rows)
    assert set(metrics) == {
        "reward_mean",
        "scored",
        "unparsed",
        "finished_frac",
        "mean_completion_tokens",
        "flat_groups",
    }
    assert metrics["reward_mean"] == pytest.approx(0.5)  # over the 4 scored rows only
    assert metrics["scored"] == 4.0
    assert metrics["unparsed"] == 1.0
    assert metrics["finished_frac"] == pytest.approx(0.4)
    assert metrics["mean_completion_tokens"] == pytest.approx(1.0)
    # group 0 has {1.0, 0.0} (not flat); group 1 has 0 scored (flat); group 2 all-equal.
    assert metrics["flat_groups"] == 2.0
    assert step_rollout_metrics([]) == {
        "reward_mean": 0.0,
        "scored": 0.0,
        "unparsed": 0.0,
        "finished_frac": 0.0,
        "mean_completion_tokens": 0.0,
        "flat_groups": 0.0,
    }


# -- render_prompt_ids / generate_group_rows / greedy_heldout ----------------


def test_render_prompt_ids_accepts_list_and_batch_encoding_shapes() -> None:
    turns = (("system", "be terse"), ("user", "q1"))
    plain = _FakeTokenizer(rendered={"q1": [7, 8]})
    assert render_prompt_ids(plain, turns) == [7, 8]
    assert plain.templates == [
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "q1"}]
    ]

    encoded = _FakeTokenizer(rendered={"q1": [11, 12, 13]}, as_dict=True)
    assert render_prompt_ids(encoded, (("user", "q1"),)) == [11, 12, 13]


def test_generate_group_rows_cuts_at_eos_scores_and_labels_groups() -> None:
    import torch

    tokenizer = _FakeTokenizer(
        eos_id=2, pad_id=0, rendered={"q1": [10, 11], "q2": [10, 12]}, as_dict=True
    )
    model = _FakeGenerate(
        [
            torch.tensor([[5, 2, 9, 9], [6, 6, 0, 0]]),
            torch.tensor([[7, 7, 2, 2], [8, 8, 8, 8]]),
        ]
    )
    samples = [
        _FakeSample((("user", "q1"),), "A"),
        _FakeSample((("user", "q2"),), "B"),
    ]
    reward = _FakeReward([1.0, None, 0.5, 0.0])
    rows = generate_group_rows(
        model,
        tokenizer,
        samples,
        [1, 0],
        group_size=2,
        max_new_tokens=4,
        temperature=0.8,
        top_p=0.9,
        reward=reward,
        device="cpu",
    )
    assert model.evaluated is True and model.exhausted is True
    assert model.prompts[0].tolist() == [[10, 12]] and model.prompts[1].tolist() == [[10, 11]]
    for call in model.calls:
        assert call["do_sample"] is True
        assert call["temperature"] == 0.8 and call["top_p"] == 0.9 and call["top_k"] == 0
        assert call["num_return_sequences"] == 2 and call["max_new_tokens"] == 4
        assert call["pad_token_id"] == 0 and call["eos_token_id"] == 2
        assert call["use_cache"] is True

    assert [(row.completion_ids, row.finished) for row in rows] == [
        ((5, 2), True),  # cut just after the first EOS, EOS kept
        ((6, 6), False),  # no EOS: budget-truncated, trailing PAD (pad != eos) stripped
        ((7, 7, 2), True),  # pad == eos on nothing here; first EOS at index 2 kept
        ((8, 8, 8, 8), False),
    ]
    assert [row.group for row in rows] == [0, 0, 1, 1]  # position in prompt_indices
    assert rows[0].prompt_ids == (10, 12) and rows[2].prompt_ids == (10, 11)
    assert [row.reward for row in rows] == [1.0, None, 0.5, 0.0]
    assert reward.calls == [("<5,2>", "B"), ("<6,6>", "B"), ("<7,7,2>", "A"), ("<8,8,8,8>", "A")]


def test_generate_group_rows_keeps_trailing_pad_when_pad_is_eos() -> None:
    import torch

    tokenizer = _FakeTokenizer(eos_id=2, pad_id=2, rendered={"q1": [10, 11]})
    model = _FakeGenerate([torch.tensor([[5, 5, 2, 2], [7, 7, 4, 4]])])
    samples = [_FakeSample((("user", "q1"),), None)]
    reward = _FakeReward([1.0, None])
    rows = generate_group_rows(
        model,
        tokenizer,
        samples,
        [0],
        group_size=2,
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        reward=reward,
        device="cpu",
    )
    assert rows[0].completion_ids == (5, 5, 2) and rows[0].finished is True
    # pad == eos: the trailing 4s are neither confirmed fill nor a stop -- kept whole.
    assert rows[1].completion_ids == (7, 7, 4, 4) and rows[1].finished is False


def test_greedy_heldout_left_pads_batches_and_reports_truncation() -> None:
    import torch

    tokenizer = _FakeTokenizer(
        eos_id=2,
        pad_id=0,
        rendered={"q1": [10], "q2": [10, 11], "q3": [10, 11, 12]},
        as_dict=True,
    )
    model = _FakeGenerate([torch.tensor([[2, 9, 9], [5, 5, 2]]), torch.tensor([[4, 4, 4]])])
    samples = [
        _FakeSample((("user", "q1"),), "A"),
        _FakeSample((("user", "q2"),), None),
        _FakeSample((("user", "q3"),), "C"),
    ]
    reward = _FakeReward([1.0, 0.5, None])
    metrics = greedy_heldout(
        model,
        tokenizer,
        samples,
        max_new_tokens=3,
        reward=reward,
        device="cpu",
        batch_size=2,
    )
    first_prompt, first_attn = model.prompts[0], model.calls[0]
    assert first_prompt.tolist() == [[0, 10], [10, 11]]  # LEFT pad
    assert first_attn is not model.prompts[0]
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 3
    assert model.calls[0]["pad_token_id"] == 0 and model.calls[0]["eos_token_id"] == 2
    assert tokenizer.padding_side == "right"  # restored after the call
    assert metrics == {
        "n": 3,
        "correct": 1,
        "unparsed": 1,
        "truncated": 1,
        "unparsed_truncated": 1,  # row 3 abstained AND ran out of budget
        "accuracy": pytest.approx(1.0 / 3.0),
        "parsed_accuracy": 0.5,
        "mean_completion_tokens": pytest.approx(7.0 / 3.0),
    }
    # completions kept: [2] then [5,5,2] then [4,4,4] (row 3 has no EOS in budget)
    assert tokenizer.decodes == [[2], [5, 5, 2], [4, 4, 4]]


def test_greedy_heldout_loose_reward_scores_the_same_decodes_beside_strict() -> None:
    import torch

    tokenizer = _FakeTokenizer(
        eos_id=2,
        pad_id=0,
        rendered={"q1": [10], "q2": [10, 11], "q3": [10, 11, 12]},
        as_dict=True,
    )
    model = _FakeGenerate([torch.tensor([[5, 2, 0], [6, 2, 0], [7, 7, 7]])])
    samples = [
        _FakeSample((("user", "q1"),), "A"),
        _FakeSample((("user", "q2"),), "B"),
        _FakeSample((("user", "q3"),), "C"),
    ]
    strict = _FakeReward([1.0, None, None])
    loose = _FakeReward([1.0, 1.0, 0.0])
    metrics = greedy_heldout(
        model,
        tokenizer,
        samples,
        max_new_tokens=3,
        reward=strict,
        device="cpu",
        batch_size=4,
        loose_reward=loose,
    )
    assert [call for call in loose.calls] == [call for call in strict.calls]
    assert metrics["correct"] == 1 and metrics["unparsed"] == 2
    assert metrics["unparsed_truncated"] == 1  # only row 3 lacks an EOS
    assert metrics["loose_correct"] == 2 and metrics["loose_unparsed"] == 0
    assert metrics["loose_accuracy"] == pytest.approx(2.0 / 3.0)


def test_greedy_heldout_without_loose_reward_emits_no_loose_keys() -> None:
    import torch

    tokenizer = _FakeTokenizer(eos_id=2, pad_id=0, rendered={"q1": [10]}, as_dict=True)
    model = _FakeGenerate([torch.tensor([[5, 2]])])
    metrics = greedy_heldout(
        model,
        tokenizer,
        [_FakeSample((("user", "q1"),), "A")],
        max_new_tokens=2,
        reward=_FakeReward([1.0]),
        device="cpu",
    )
    assert not any(key.startswith("loose_") for key in metrics)


# -- torch-free import contract ---------------------------------------------


def test_module_import_does_not_import_torch() -> None:
    paths = json.dumps(sys.path)
    code = (
        "import json, sys\n"
        f"sys.path[:0] = json.loads({json.dumps(paths)})\n"
        "import foundationscale.rl.megatron.online as online\n"
        "leaked = sorted(\n"
        "    name\n"
        "    for name in ('torch', 'transformers')\n"
        "    if any(name == module or module.startswith(name + '.') for module in sys.modules)\n"
        ")\n"
        "assert not leaked, f'{leaked} pulled in by a bare import of online'\n"
        "assert online.OnlineRow(prompt_ids=(1,), completion_ids=(2,), reward=None, \n"
        "                         group=0, finished=False).reward is None\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_refit_hf_policy_scatters_per_expert_exports_into_fused_hf_experts() -> None:
    """Bridge yields per-expert checkpoint names; transformers 5 holds fused experts."""
    import torch

    experts, inter, hidden = 2, 3, 4

    class _Experts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(experts, 2 * inter, hidden))
            self.down_proj = torch.nn.Parameter(torch.zeros(experts, hidden, inter))

    class _Mlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = _Experts()

    class _Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _Mlp()

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([_Layer()])

    model = _Model()
    yielded = []
    for e in range(experts):
        yielded.append(
            (f"layers.0.mlp.experts.{e}.gate_proj.weight", torch.full((inter, hidden), 10.0 + e))
        )
        yielded.append(
            (f"layers.0.mlp.experts.{e}.up_proj.weight", torch.full((inter, hidden), 20.0 + e))
        )
        yielded.append(
            (f"layers.0.mlp.experts.{e}.down_proj.weight", torch.full((hidden, inter), 30.0 + e))
        )
    counts = refit_hf_policy(_FakeBridge(yielded), [], model, is_writer=True)
    assert counts == {"written": 2, "unwritten": 0}
    gate_up = model.layers[0].mlp.experts.gate_up_proj
    assert gate_up[1, :inter].unique().tolist() == [11.0]  # gate half first
    assert gate_up[1, inter:].unique().tolist() == [21.0]  # then up
    assert model.layers[0].mlp.experts.down_proj[0].unique().tolist() == [30.0]

    # One expert slice short: the fused tensor is NOT credited, so it reports as unwritten.
    partial = _Model()
    counts = refit_hf_policy(_FakeBridge(yielded[:-1]), [], partial, is_writer=True)
    assert counts == {"written": 1, "unwritten": 1}

    # A slice whose shape disagrees with the fused layout refuses instead of copying.
    bad = [("layers.0.mlp.experts.0.down_proj.weight", torch.ones(inter, hidden))]
    with pytest.raises(ValueError, match="slice of layers.0.mlp.experts.down_proj"):
        refit_hf_policy(_FakeBridge(bad), [], _Model(), is_writer=True)
