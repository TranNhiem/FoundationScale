"""CPU-only tests for ``foundationscale.rl.distributed``.

No GPU, no process group is ever created: every collective is exercised
through world_size == 1 contexts, which are the identity paths the trainers
also hit on a single-process run. FS_FORBID_SKIPS=1 clean -- nothing here
may skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from foundationscale.rl.distributed import (
    DistContext,
    agree_all,
    agree_max,
    all_reduce_mean,
    all_reduce_sum,
    find_decoder_blocks,
    generate_kwargs_for,
    init_distributed,
    save_checkpoint,
    shard_indices,
)
from foundationscale.rl.trainer import RLTrainConfig, RLTrainer, TrainerRefusal


def _ctx(rank: int = 0, world: int = 1) -> DistContext:
    return DistContext(
        rank=rank,
        world_size=world,
        local_rank=rank,
        device=torch.device("cpu"),
        is_distributed=(world > 1),
        owns_process_group=False,
    )


class TestShardIndices:
    def test_equal_counts_disjoint_union_and_remainder(self) -> None:
        # n=7 over 3 ranks: per-rank 2, remainder 1 dropped deterministically.
        parts = [shard_indices(7, _ctx(rank=r, world=3)) for r in range(3)]
        assert [len(p) for p in parts] == [2, 2, 2]
        flat = [i for part in parts for i in part]
        assert len(flat) == len(set(flat)), "slices must be disjoint"
        assert sorted(flat) == [0, 1, 2, 3, 4, 5]
        # The dropped trailing remainder never appears on any rank.
        assert 6 not in flat

    def test_exact_split_drops_nothing(self) -> None:
        parts = [shard_indices(6, _ctx(rank=r, world=2)) for r in range(2)]
        assert parts == [[0, 1, 2], [3, 4, 5]]

    def test_world_one_is_identity(self) -> None:
        assert shard_indices(5, _ctx()) == [0, 1, 2, 3, 4]


def test_init_distributed_none_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.distributed as dist

    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    ctx = init_distributed("none")
    assert ctx.world_size == 1
    assert ctx.rank == 0
    assert ctx.is_distributed is False
    assert not dist.is_initialized()


def test_init_distributed_none_refuses_under_torchrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(ValueError, match="none"):
        init_distributed("none")
    assert not dist.is_initialized(), "a refused init must not create a group"


def test_init_distributed_unknown_sharding_refused() -> None:
    with pytest.raises(ValueError, match="bogus"):
        init_distributed("bogus")


def test_generate_kwargs_for() -> None:
    assert generate_kwargs_for(_ctx(world=4), "fsdp") == {"synced_gpus": True}
    assert generate_kwargs_for(_ctx(), "none") == {}
    assert generate_kwargs_for(_ctx(world=4), "ddp") == {}
    # Degenerate fsdp world of one rank needs no stepping sync.
    assert generate_kwargs_for(_ctx(), "fsdp") == {}


def test_rltrain_config_refuses_unknown_sharding() -> None:
    config = RLTrainConfig(model="m", dataset="d", sharding="bogus")
    with pytest.raises(TrainerRefusal, match="sharding"):
        RLTrainer(config)


def test_preference_config_refuses_unknown_sharding() -> None:
    # PreferenceTrainer now declares `sharding`; an unknown value is refused
    # at trainer construction, before any model is loaded.
    from foundationscale.rl.preference_trainer import PreferenceTrainConfig, PreferenceTrainer
    from foundationscale.rl.trainer import TrainerRefusal

    with pytest.raises(TrainerRefusal, match="sharding"):
        PreferenceTrainer(PreferenceTrainConfig(model="m", dataset="d", sharding="bogus"))


class _Block(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)


class _FakeHFModel(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, n_blocks: int = 2, width: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, width)
        self.blocks = nn.ModuleList(_Block(width) for _ in range(n_blocks))
        self.lm_head = nn.Linear(width, 8, bias=False)
        self.lm_head.weight = self.embed.weight  # tied


def test_find_decoder_blocks_uses_no_split_modules() -> None:
    model = _FakeHFModel(n_blocks=3)
    blocks = find_decoder_blocks(model)
    assert len(blocks) == 3
    assert all(isinstance(b, _Block) for b in blocks)
    # Embeddings and the tied lm_head stay in the root unit, never wrapped.
    assert model.embed not in blocks
    assert model.lm_head not in blocks


class _NoAttrModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.small = nn.ModuleList(_Block() for _ in range(1))
        self.big = nn.ModuleList(_Block() for _ in range(3))


def test_find_decoder_blocks_falls_back_to_largest_module_list() -> None:
    model = _NoAttrModel()
    blocks = find_decoder_blocks(model)
    assert len(blocks) == 3
    assert list(model.big.children()) == blocks


def test_find_decoder_blocks_refuses_when_nothing_matches() -> None:
    class _Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="decoder blocks"):
        find_decoder_blocks(_Bare())


def test_collectives_are_identity_at_world_one() -> None:
    ctx = _ctx()
    assert agree_max(7, ctx) == 7
    assert agree_all(True, ctx) is True
    assert agree_all(False, ctx) is False
    assert all_reduce_mean(2.5, ctx) == 2.5
    assert all_reduce_sum(3.0, ctx) == 3.0


def test_save_checkpoint_world_one_none(tmp_path) -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.saved_with: dict | None = None

        def save_pretrained(self, out_dir, **kwargs) -> None:
            from pathlib import Path

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "model.safetensors").write_text("weights")
            self.saved_with = dict(kwargs)

    class _FakeTokenizer:
        def save_pretrained(self, out_dir) -> None:
            from pathlib import Path

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "tokenizer.json").write_text("{}")

    model = _FakeModel()
    out = tmp_path / "ckpt"
    ok = save_checkpoint(model, _FakeTokenizer(), str(out), _ctx(), sharding="none", step=7)
    assert ok is True
    assert model.saved_with is not None
    assert model.saved_with.get("safe_serialization") is True
    marker = json.loads((out / "fs_rl_checkpoint.json").read_text())
    assert marker["step"] == 7
    assert marker["sharding"] == "none"
    assert marker["world_size"] == 1
    assert (out / "tokenizer.json").exists()


class _FakeShard:
    """A DTensor stand-in: global shape via numel/element_size, full_tensor() logs."""

    def __init__(self, full: torch.Tensor, trace: list[str], name: str) -> None:
        self._full = full
        self._trace = trace
        self._name = name
        self.dtype = full.dtype

    def is_floating_point(self) -> bool:
        return self._full.is_floating_point()

    def numel(self) -> int:
        return self._full.numel()

    def element_size(self) -> int:
        return self._full.element_size()

    def full_tensor(self) -> torch.Tensor:
        self._trace.append(f"gather:{self._name}")
        return self._full.clone()


def _run_fsdp_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rank: int, chunk_bytes: int
) -> tuple[list[str], dict | None]:
    import torch.distributed.checkpoint.state_dict as sd_mod

    import foundationscale.rl.distributed as dist_mod

    trace: list[str] = []
    weights = {
        "embed.weight": torch.randn(8, 4),
        "norm.weight": torch.ones(4),
        "position_ids": torch.arange(4),
    }
    # A plain tensor mixed in: replicated buffers are not DTensors.
    sharded = {
        k: (_FakeShard(v, trace, k) if k != "position_ids" else v) for k, v in weights.items()
    }
    monkeypatch.setattr(sd_mod, "get_model_state_dict", lambda model, options: sharded)
    monkeypatch.setattr(dist_mod, "_save_barrier", lambda ctx: trace.append("save_wait"))
    monkeypatch.setattr(dist_mod, "SAVE_GATHER_CHUNK_BYTES", chunk_bytes)

    class _Model:
        saved: dict | None = None

        def save_pretrained(self, out_dir, state_dict=None, **kwargs) -> None:
            _Model.saved = state_dict

    model = _Model()
    ok = save_checkpoint(
        model, None, str(tmp_path / f"r{rank}"), _ctx(rank=rank, world=3), sharding="fsdp", step=1
    )
    assert ok is True
    return trace, model.saved


@pytest.mark.parametrize("chunk_bytes", [1, 64, 1 << 30])
def test_fsdp_save_ranks_issue_identical_collectives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chunk_bytes: int
) -> None:
    # S6: flush points come from GLOBAL shapes, so rank 0 (which copies to
    # host) and a peer (which keeps nothing) issue the same ordered waits.
    main_trace, main_saved = _run_fsdp_save(monkeypatch, tmp_path, 0, chunk_bytes)
    peer_trace, peer_saved = _run_fsdp_save(monkeypatch, tmp_path, 1, chunk_bytes)
    assert main_trace == peer_trace
    assert main_trace.count("gather:embed.weight") == 1
    assert main_trace[0] == "save_wait", "the entry wait precedes any rank-0-only work"
    assert main_trace[-1] == "save_wait", "the post-save wait is the long-timeout one"
    assert peer_saved is None
    assert main_saved is not None
    assert list(main_saved) == ["embed.weight", "norm.weight", "position_ids"]
    assert main_saved["embed.weight"].dtype == torch.bfloat16
    assert main_saved["norm.weight"].dtype == torch.bfloat16
    assert main_saved["position_ids"].dtype == torch.int64, "integer tensors keep their dtype"


def test_fsdp_save_chunks_bound_the_device_holding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A 1-byte budget flushes after every tensor; a huge one only at the end.
    small, _ = _run_fsdp_save(monkeypatch, tmp_path, 0, 1)
    large, _ = _run_fsdp_save(monkeypatch, tmp_path, 0, 1 << 30)
    # Entry wait + 3 per-tensor flushes + the empty tail flush + the post-save wait.
    assert small.count("save_wait") == 6
    # Entry wait + one tail flush + the post-save wait.
    assert large.count("save_wait") == 3


@pytest.mark.parametrize("world", [2, 4])
def test_fewer_prompts_than_ranks_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: int
) -> None:
    # Sharding is over prompts: prompts_per_step < world_size would give some
    # rank zero prompts, and every step would then abstain. Refuse up front.
    import foundationscale.rl.distributed as dist_mod
    from foundationscale.rl.distributed import DistContext
    from foundationscale.rl.trainer import RLTrainConfig, RLTrainer, TrainerRefusal

    fake = DistContext(rank=0, world_size=world, local_rank=0, device="cpu", is_distributed=False)
    monkeypatch.setattr(dist_mod, "init_distributed", lambda sharding: fake)
    corpus = tmp_path / "c.jsonl"
    row = {
        "answer": "A",
        "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "A"}],
    }
    corpus.write_text(json.dumps(row) + "\n")
    trainer = RLTrainer(
        RLTrainConfig(model="m", dataset=str(corpus), gold_key="answer", prompts_per_step=world - 1)
    )
    with pytest.raises(TrainerRefusal, match="prompts_per_step"):
        trainer.run()


def test_fewer_pairs_than_ranks_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import foundationscale.rl.distributed as dist_mod
    from foundationscale.rl.distributed import DistContext
    from foundationscale.rl.preference_trainer import PreferenceTrainConfig, PreferenceTrainer
    from foundationscale.rl.trainer import TrainerRefusal

    fake = DistContext(rank=0, world_size=4, local_rank=0, device="cpu", is_distributed=False)
    monkeypatch.setattr(dist_mod, "init_distributed", lambda sharding: fake)
    import foundationscale.rl.preference_trainer as pt_mod

    monkeypatch.setattr(pt_mod, "init_distributed", lambda sharding: fake)
    corpus = tmp_path / "p.jsonl"
    corpus.write_text(json.dumps({"prompt": "a", "chosen": "b", "rejected": "c"}) + "\n")
    trainer = PreferenceTrainer(
        PreferenceTrainConfig(model="m", dataset=str(corpus), pairs_per_step=2)
    )
    with pytest.raises(TrainerRefusal, match="pairs_per_step"):
        trainer.run()
