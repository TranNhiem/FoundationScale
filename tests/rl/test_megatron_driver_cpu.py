"""CPU-only tests for the pure helpers in rl/megatron/pp_step.py and driver.py.

Everything here runs without megatron; the lazy-import contract is exercised
by calling ``require_megatron`` and asserting the named ImportError (on a box
where megatron genuinely absent) or that the call returns (where present --
the test adapts without skipping, since skips fail CI).
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from foundationscale.rl.megatron import driver, pp_step
from foundationscale.rl.megatron.driver import (
    MockRolloutRow,
    collate_mock_batch,
    group_relative_advantages,
    load_mock_rollouts,
    param_hash,
    reduce_step_metrics,
    require_megatron,
)
from foundationscale.rl.megatron.pp_step import (
    family_of,
    local_denominator,
    ratio_and_clip_metrics,
    rescale_to_global_denominator,
)


class _CharTokenizer:
    """Deterministic char-level tokenizer: ord-based ids, no dependencies."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [2 + (ord(c) % 100) for c in text]}


@dataclass(frozen=True)
class _Den:
    tokens: float = 0.0
    sequences: float = 0.0
    pairs: float = 0.0


def test_family_of_mapping_and_refusal() -> None:
    assert family_of("grpo") == "sequence"
    assert family_of("GSPO") == "sequence"
    assert family_of("dapo") == "token"
    assert family_of("dr_grpo") == "dr_grpo"
    assert family_of("dpo") == "preference"
    with pytest.raises(ValueError, match="not one of the 4 declared units"):
        family_of("sft")


def test_local_denominator_units() -> None:
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert local_denominator("token", mask) == pytest.approx(4.0)
    assert local_denominator("sequence", mask) == pytest.approx(2.0)
    sample = torch.tensor([1.0, 0.0, 1.0])
    assert local_denominator("dpo", mask, sample) == pytest.approx(2.0)
    assert local_denominator("token", mask, sample) == pytest.approx(4.0)


def test_rescale_to_global_denominator_token_family() -> None:
    # Kernel returned the microbatch mean over 4 tokens; global is 8 tokens,
    # so this microbatch's additive share is mean * (4/8).
    loss = torch.tensor(2.0, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    den = _Den(tokens=8.0)
    scaled = rescale_to_global_denominator(loss, "token", mask, den)
    assert float(scaled) == pytest.approx(1.0)
    scaled.backward()
    assert float(loss.grad) == pytest.approx(0.5)


def test_rescale_all_masked_microbatch_contributes_zero_but_differentiable() -> None:
    loss = torch.tensor(3.0, requires_grad=True)
    mask = torch.zeros(2, 3)
    scaled = rescale_to_global_denominator(loss, "token", mask, _Den(tokens=4.0))
    assert float(scaled) == 0.0
    scaled.backward()
    assert float(loss.grad) == 0.0


def test_rescale_refuses_zero_global_denominator() -> None:
    mask = torch.ones(1, 2)
    with pytest.raises(ValueError, match="must be positive"):
        rescale_to_global_denominator(torch.tensor(1.0), "token", mask, _Den(tokens=0.0))


def test_rescale_refuses_missing_denominator_field() -> None:
    @dataclass(frozen=True)
    class _Empty:
        pass

    with pytest.raises(ValueError, match="compute_denominators"):
        rescale_to_global_denominator(torch.tensor(1.0), "token", torch.ones(1, 1), _Empty())


def test_ratio_and_clip_metrics() -> None:
    cur = torch.zeros(2, 3)
    old = torch.zeros(2, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = ratio_and_clip_metrics(cur, old, mask, (0.8, 1.2))
    assert float(out["ratio_mean"]) == pytest.approx(1.0)
    assert float(out["clip_fraction"]) == pytest.approx(0.0)
    assert float(out["tokens"]) == pytest.approx(5.0)
    assert all(v.dim() == 0 and not v.requires_grad for v in out.values())

    cur2 = torch.full((1, 2), 1.0)  # ratio e ~= 2.718, above the clip band
    out2 = ratio_and_clip_metrics(cur2, torch.zeros(1, 2), torch.ones(1, 2), (0.8, 1.2))
    assert float(out2["clip_fraction"]) == pytest.approx(1.0)


def test_ratio_and_clip_metrics_fully_masked_returns_zeros() -> None:
    out = ratio_and_clip_metrics(
        torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), (0.8, 1.2)
    )
    assert float(out["tokens"]) == 0.0
    assert float(out["ratio_mean"]) == 0.0


def test_group_relative_advantages() -> None:
    rewards = [1.0, 0.0, 1.0, 0.0, 5.0]
    groups = [0, 0, 0, 0, 1]
    out = group_relative_advantages(rewards, groups, eps=0.0)
    assert out[:4] == pytest.approx([1.0, -1.0, 1.0, -1.0])
    assert out[4] == pytest.approx(0.0)  # singleton group: no signal
    with pytest.raises(ValueError, match="1 id per reward"):
        group_relative_advantages([1.0], [0, 1])


def test_load_mock_rollouts_roundtrip_and_refusals(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"prompt": "p", "completion": "c", "reward": 1.5}) + "\n", encoding="utf-8"
    )
    rows = load_mock_rollouts(path)
    assert rows == [MockRolloutRow(prompt="p", completion="c", reward=1.5)]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "p"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_mock_rollouts(bad)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="0 mock rollout rows"):
        load_mock_rollouts(empty)


def test_collate_mock_batch_masks_and_padding() -> None:
    rows = [
        MockRolloutRow(prompt="ab", completion="cde", reward=1.0),
        MockRolloutRow(prompt="a", completion="bcdefgh", reward=0.0),
    ]
    batch = collate_mock_batch(rows, [0.5, -0.5], _CharTokenizer(), seq_len=64, pad_id=0)
    assert batch["input_ids"].shape == (2, 8)  # 1 + 7 tokens
    assert batch["loss_mask"][0][:2].sum() == 0.0  # prompt carries no supervision
    assert batch["loss_mask"][0][2:5].sum() == 3.0  # completion does
    assert batch["loss_mask"][1].sum() == 7.0
    assert float(batch["sample_mask"].sum()) == 2.0
    assert batch["advantages"].tolist() == [0.5, -0.5]


def test_collate_mock_batch_truncation_keeps_mask_aligned() -> None:
    rows = [MockRolloutRow(prompt="" * 0 + "abcdef", completion="ghij", reward=1.0)]
    batch = collate_mock_batch(rows, [1.0], _CharTokenizer(), seq_len=8, pad_id=0)
    assert batch["input_ids"].shape[1] == 8
    assert batch["loss_mask"].shape[1] == 8


def test_collate_refuses_advantage_count_mismatch() -> None:
    with pytest.raises(ValueError, match="exactly one advantage"):
        collate_mock_batch(
            [MockRolloutRow("p", "c", 1.0)], [], _CharTokenizer(), seq_len=8, pad_id=0
        )


def test_param_hash_is_deterministic_and_sensitive() -> None:
    model_a = torch.nn.Linear(4, 4)
    model_b = torch.nn.Linear(4, 4)
    with torch.no_grad():
        model_b.weight.add_(1.0)
    assert param_hash(model_a) == param_hash(model_a)
    assert param_hash(model_a) != param_hash(model_b)
    with pytest.raises(ValueError, match="0 parameters"):
        param_hash(torch.nn.Module())


def test_require_megatron_named_import_error() -> None:
    if importlib.util.find_spec("megatron") is None:
        with pytest.raises(ImportError, match="requires megatron-core"):
            require_megatron("cpu-test")
    else:
        require_megatron("cpu-test")  # container box: must pass, never skip


def test_build_arg_parser_defaults() -> None:
    args = driver.build_arg_parser().parse_args(
        ["--hf-model", "m", "--rollout-jsonl", "r.jsonl", "--metrics-out", "o.jsonl"]
    )
    assert args.tp == 1 and args.pp == 1 and args.cp == 1
    assert args.sp is False and args.parity_only is False
    assert args.attention_backend is None  # default leaves the provider's choice


def test_build_arg_parser_attention_backend_choices() -> None:
    base = ["--hf-model", "m", "--rollout-jsonl", "r.jsonl", "--metrics-out", "o.jsonl"]
    args = driver.build_arg_parser().parse_args([*base, "--attention-backend", "unfused"])
    assert args.attention_backend == "unfused"
    with pytest.raises(SystemExit):
        driver.build_arg_parser().parse_args([*base, "--attention-backend", "bogus"])


def test_build_arg_parser_offload_rollout_model_is_opt_in() -> None:
    base = ["--hf-model", "m", "--rollout-jsonl", "r.jsonl", "--metrics-out", "o.jsonl"]
    assert driver.build_arg_parser().parse_args(base).offload_rollout_model is False
    args = driver.build_arg_parser().parse_args([*base, "--offload-rollout-model"])
    assert args.offload_rollout_model is True


def test_build_arg_parser_heldout_loose_pattern_is_opt_in() -> None:
    base = ["--hf-model", "m", "--rollout-jsonl", "r.jsonl", "--metrics-out", "o.jsonl"]
    assert driver.build_arg_parser().parse_args(base).heldout_loose_pattern is None
    pattern = r"([A-D])"
    args = driver.build_arg_parser().parse_args([*base, "--heldout-loose-pattern", pattern])
    assert args.heldout_loose_pattern == pattern


def test_pp_step_module_importable_without_megatron() -> None:
    assert pp_step.__name__.endswith("pp_step")
    assert callable(pp_step.make_forward_step)
    assert callable(pp_step.make_logprob_forward_step)


def test_loss_unit_reads_reduction_off_every_registered_objective() -> None:
    # The GPU rung-1 run died here: TensorPolicyLoss declares its unit only via
    # objective.reduction, and make_forward_step looked for a family name.
    from foundationscale.rl.registry import available_algorithm_names, lookup_algorithm
    from foundationscale.rl.torch_backend import TensorPolicyLoss

    seen = 0
    for name in available_algorithm_names():
        objective = getattr(lookup_algorithm(name), "_objective", None)
        reduction = getattr(objective, "reduction", None)
        if reduction not in pp_step._REDUCTION_UNITS:
            continue
        unit = pp_step.loss_unit(TensorPolicyLoss(objective=objective))
        assert unit == pp_step._REDUCTION_UNITS[reduction], name
        seen += 1
    assert seen > 0


def test_loss_unit_explicit_family_wins_and_undeclared_refuses() -> None:
    class Declared:
        family = "gspo"
        objective = SimpleNamespace(reduction="token_mean")

    assert pp_step.loss_unit(Declared()) == "sequence"
    with pytest.raises(ValueError, match="normalization unit"):
        pp_step.loss_unit(SimpleNamespace(objective=SimpleNamespace(reduction=None)))


def test_undo_mcore_loss_average_inverts_the_schedule_scaling() -> None:
    # mcore's forward_step_calc_loss does loss * cp / num_microbatches on a
    # 2-tuple return; applying it to the inverted value must give back the share.
    share = torch.tensor(0.37)
    for nmb, cp in ((1, 1), (4, 1), (8, 2)):
        sent = pp_step.undo_mcore_loss_average(share, nmb, cp)
        assert torch.allclose(sent * cp / nmb, share)
    with pytest.raises(ValueError):
        pp_step.undo_mcore_loss_average(share, 0, 1)


def test_undo_mcore_loss_average_inverts_the_ddp_average() -> None:
    # DDP divides the summed grad by the DP x CP group size; the inverted value
    # after the schedule scaling and that average must give back the share.
    share = torch.tensor(0.37)
    for nmb, cp, dp_cp in ((1, 1, 2), (4, 2, 4), (2, 2, 8)):
        sent = pp_step.undo_mcore_loss_average(share, nmb, cp, dp_cp)
        assert torch.allclose(sent * cp / nmb / dp_cp, share)
    with pytest.raises(ValueError):
        pp_step.undo_mcore_loss_average(share, 1, 1, 0)


def test_cp_shard_index_is_a_load_balanced_partition() -> None:
    assert pp_step.cp_shard_index(8, 2, 0) == [0, 1, 6, 7]
    assert pp_step.cp_shard_index(8, 2, 1) == [2, 3, 4, 5]
    for cp in (1, 2, 4):
        held = [i for r in range(cp) for i in pp_step.cp_shard_index(16 * cp, cp, r)]
        assert sorted(held) == list(range(16 * cp))
    with pytest.raises(ValueError):
        pp_step.cp_shard_index(6, 2, 0)


def test_model_inputs_pad_to_tp_under_sequence_parallel() -> None:
    # SP reduce-scatters the embedding output along the sequence, so an odd
    # row under TP2+SP tripped mcore's "divisible by tensor parallel size".
    batch = {"input_ids": torch.arange(1, 8).view(1, 7)}
    ids, pos, padded = pp_step._model_inputs(batch, 1, 0)
    assert padded == 7 and ids.shape == (1, 7)
    ids, pos, padded = pp_step._model_inputs(batch, 1, 0, sp_tp=2)
    assert padded == 8 and ids[0, -1] == 0 and pos[0].tolist() == list(range(8))
    for cp in (2, 4):
        chunks = [pp_step._model_inputs(batch, cp, r, sp_tp=2) for r in range(cp)]
        assert {c[2] for c in chunks} == {4 * cp}
        assert all(c[0].shape[1] % 2 == 0 for c in chunks)
    assert pp_step.sp_tp(SimpleNamespace(tp=2, sp=True)) == 2
    assert pp_step.sp_tp(SimpleNamespace(tp=2, sp=False)) == 1
    assert pp_step.sp_tp(SimpleNamespace(tp=1, sp=True)) == 1
    assert pp_step.padded_seq_len(7, 1, 1) == 7
    assert pp_step.padded_seq_len(7, 2, 2) == 8
    assert pp_step.padded_seq_len(9, 2, 2) == 16


def test_param_hash_sees_an_update_outside_the_first_parameter_slice() -> None:
    # The old probe hashed params[0][:4096]; an embedding whose leading rows no
    # batch touches is exactly that slice, so real updates were invisible.
    # A sample can still miss one sparse embedding row; every dense layer gets
    # gradient on a real step, which is what the probe must see.
    model = torch.nn.Sequential(torch.nn.Embedding(1000, 8), torch.nn.Linear(8, 8))
    before = param_hash(model)
    with torch.no_grad():
        model[1].bias.add_(1e-3)
    assert param_hash(model) != before


def test_reduce_step_metrics_divides_by_the_global_token_count_once() -> None:
    entries = [
        {"loss": 0.1, "ratio_mean": 1.0, "clip_fraction": 0.0, "tokens": 30.0},
        {"loss": 0.2, "ratio_mean": 1.0, "clip_fraction": 1.0, "tokens": 10.0},
    ]
    out = reduce_step_metrics(entries)
    assert out["ratio_mean"] == pytest.approx(1.0)
    assert out["clip_fraction"] == pytest.approx(0.25)
    assert out["tokens"] == pytest.approx(40.0)
    assert out["loss"] == pytest.approx(0.3)

    def two_identical_ranks(t: torch.Tensor) -> None:
        t.mul_(2.0)

    dp = reduce_step_metrics(entries, two_identical_ranks)
    assert dp["ratio_mean"] == pytest.approx(1.0)
    assert dp["clip_fraction"] == pytest.approx(0.25)
    assert dp["tokens"] == pytest.approx(80.0)
    assert reduce_step_metrics([])["ratio_mean"] == 0.0


def test_restore_tp_attributes_marks_only_unmarked_mcore_weights(monkeypatch):
    """Bridge loads HF weights with perform_initialization off, so mcore-native
    TP layers carry no tensor_model_parallel flag and the grad norm drops tp
    rank 1's vocab shard. TE-style weights that already carry it stay as-is."""
    import sys
    import types

    class _Weighted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4, 2))

    class VocabParallelEmbedding(_Weighted):
        pass

    class ColumnParallelLinear(_Weighted):
        pass

    class RowParallelLinear(_Weighted):
        pass

    layers = types.ModuleType("megatron.core.tensor_parallel.layers")
    layers.VocabParallelEmbedding = VocabParallelEmbedding
    layers.ColumnParallelLinear = ColumnParallelLinear
    layers.RowParallelLinear = RowParallelLinear
    for name in ("megatron", "megatron.core", "megatron.core.tensor_parallel"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel.layers", layers)

    model = torch.nn.Module()
    model.embedding = VocabParallelEmbedding()
    model.output_layer = ColumnParallelLinear()
    model.proj = RowParallelLinear()
    model.tied = ColumnParallelLinear()
    model.tied.weight = None
    model.te = ColumnParallelLinear()
    model.te.weight.tensor_model_parallel = True
    model.te.weight.partition_dim = 0
    model.norm = torch.nn.LayerNorm(2)

    assert driver.restore_tp_attributes([model]) == 3
    assert model.embedding.weight.tensor_model_parallel is True
    assert model.embedding.weight.partition_dim == 0
    assert model.output_layer.weight.partition_dim == 0
    assert model.proj.weight.partition_dim == 1
    assert model.proj.weight.partition_stride == 1
    assert not getattr(model.norm.weight, "tensor_model_parallel", False)
    assert driver.restore_tp_attributes([model]) == 0


def test_objective_loss_or_zero_skips_kernel_on_fully_masked_microbatch() -> None:
    from foundationscale.rl.megatron.pp_step import objective_loss_or_zero

    calls: list[int] = []

    def kernel(**kwargs: object) -> torch.Tensor:
        calls.append(1)
        raise AssertionError("kernel must not see a fully masked microbatch")

    cur = torch.randn(2, 3, requires_grad=True)
    out = objective_loss_or_zero(
        kernel,
        current_logprobs=cur,
        old_logprobs=torch.zeros(2, 3),
        advantages=torch.ones(2),
        mask=torch.zeros(2, 3),
    )
    assert calls == []
    assert out.item() == 0.0
    out.backward()
    assert cur.grad is not None and torch.count_nonzero(cur.grad) == 0


def test_objective_loss_or_zero_calls_kernel_when_supervised() -> None:
    from foundationscale.rl.megatron.pp_step import objective_loss_or_zero

    seen: dict[str, object] = {}

    def kernel(**kwargs: object) -> torch.Tensor:
        seen.update(kwargs)
        return torch.tensor(1.5)

    mask = torch.tensor([[0.0, 1.0, 0.0]])
    ref = torch.zeros(1, 3)
    out = objective_loss_or_zero(
        kernel,
        current_logprobs=torch.zeros(1, 3),
        old_logprobs=torch.zeros(1, 3),
        advantages=torch.ones(1),
        mask=mask,
        reference_logprobs=ref,
    )
    assert out.item() == 1.5
    assert seen["mask"] is mask and seen["reference_logprobs"] is ref


@pytest.mark.parametrize("algorithm", ["dr_grpo", "grpo"])
def test_sharded_rescaled_loss_equals_single_process_loss(algorithm: str) -> None:
    """Splitting a batch across two DP shards, rescaling each shard's kernel
    output and summing must reproduce the kernel on the whole batch. A dr_grpo
    rescale keyed on supervised TOKENS inflated the loss by tokens/row (the
    measured grad_norm tracked response length at ~0.5x on the GB200)."""
    from foundationscale.rl.megatron.normalization import compute_denominators
    from foundationscale.rl.registry import lookup_algorithm
    from foundationscale.rl.torch_backend import TensorPolicyLoss

    torch.manual_seed(0)
    rows, width = 8, 1024
    loss_fn = TensorPolicyLoss(objective=lookup_algorithm(algorithm)._objective)
    unit = pp_step.loss_unit(loss_fn)
    base = torch.randn(rows, width) * 0.1 - 1.0
    cur = base.clone().requires_grad_()
    old = base + torch.randn(rows, width) * 0.01
    ref = base.clone()
    adv = torch.randn(rows)
    lengths = torch.tensor([30, 400, 90, 700, 12, 250, 1000, 60])
    mask = (torch.arange(width)[None, :] < lengths[:, None]).to(torch.float32)
    sample = torch.ones(rows)

    full = float(
        loss_fn(
            current_logprobs=cur,
            old_logprobs=old,
            advantages=adv,
            mask=mask,
            reference_logprobs=ref,
        )
    )
    declared = (float(rows), float(width)) if unit == "dr_grpo" else None
    den = compute_denominators(unit, mask, sample, group=None, declared=declared)
    total = 0.0
    for lo, hi in ((0, 3), (3, 8)):
        raw = loss_fn(
            current_logprobs=cur[lo:hi],
            old_logprobs=old[lo:hi],
            advantages=adv[lo:hi],
            mask=mask[lo:hi],
            reference_logprobs=ref[lo:hi],
        )
        total += float(
            pp_step.rescale_to_global_denominator(
                raw, unit, mask[lo:hi], den, sample_mask=sample[lo:hi]
            )
        )
    assert total == pytest.approx(full, rel=1e-5, abs=1e-7)


def test_iter_microbatches_slices_every_tensor_in_lockstep() -> None:
    batch = {"input_ids": torch.arange(10).view(5, 2), "mask": torch.ones(5, 2)}
    chunks = list(driver._iter_microbatches(batch, 2))
    assert [chunk["input_ids"].shape[0] for chunk in chunks] == [2, 2, 1]
    assert torch.equal(chunks[2]["input_ids"], torch.tensor([[8, 9]]))
    assert all(chunk["mask"].shape[0] == chunk["input_ids"].shape[0] for chunk in chunks)


def test_load_mock_rollouts_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "rollouts.jsonl"
    row = {"prompt": "p", "completion": "c", "reward": 1.0}
    path.write_text(f"\n{json.dumps(row)}\n\n", encoding="utf-8")
    rows = driver.load_mock_rollouts(path)
    assert len(rows) == 1 and rows[0].reward == 1.0


def test_reduce_step_metrics_ignores_non_dict_entries() -> None:
    metrics = driver.reduce_step_metrics(
        ["not-a-dict", {"loss": 0.5, "ratio_mean": 1.0, "clip_fraction": 0.0, "tokens": 4.0}]
    )
    assert metrics["loss"] == pytest.approx(0.5)


def test_build_arg_parser_train_state_is_opt_in() -> None:
    base = ["--hf-model", "m", "--rollout-jsonl", "r.jsonl", "--metrics-out", "o.jsonl"]
    args = driver.build_arg_parser().parse_args(base)
    assert args.train_state_dir == "" and args.train_state_every == 0
    args = driver.build_arg_parser().parse_args(
        [*base, "--train-state-dir", "s", "--train-state-every", "50"]
    )
    assert args.train_state_dir == "s" and args.train_state_every == 50
