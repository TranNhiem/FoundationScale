"""Refusal tests for the declared-but-unexecutable axes (the #375 batch).

WHAT IS MEASURED: that ``train()`` refuses (96) -- at its OWN refusal site,
BEFORE the Trainer is constructed -- for each declaration the shipped backend
cannot honour:

  * any of tp/pp/ep/cp > 1 (finding #375: every reader of those degrees used
    to be a validator or a recorder while execution was unconditionally
    Trainer+DDP, so the manifest recorded the inverse of the executed
    topology -- tensor_parallel=4, data_parallel=1 for a run that was 4-way
    DDP);
  * ``sharding_strategy`` outside ``("ddp",)``;
  * ``cpu_optimizer_offload=True``.

INSTRUMENT: two independent probes per run, because an exit code is not a
site. ``train()`` contains a SECOND refusal that also exits 96 (the missing
optional-dependency refusal, after dry-run), so ``rc == 96`` alone would pass
on a broken tree whose #375 refusal was deleted -- the vacuous-control defect
of #291/#294/#372. Each test therefore asserts REACHED-SITE (the
``[fs:train:refuse]`` marker, the refusal's own message text, and the absence
of the later ``[fs:train:deps]`` marker) separately from OUTCOME (96, and the
fake Trainer's construction recorder still empty).

The PROCEED arms (all degrees at 1, sharding None/"ddp", offload None/False)
run under the SAME fakes and must reach ``[fs:train:trainer]`` with a recorded
construction. They are the inverse control: they prove the refusal -- and not
some upstream condition that would have stopped every arm anyway -- is what
stopped the refused arms.

torch/transformers/datasets are faked in ``sys.modules``, so the module runs
on a torch-free CPU host in seconds. Nothing here touches the network, and
every path lives under ``tmp_path``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from foundationscale.train import loop


def _base_kwargs(tmp_path) -> dict:
    # profile_name is a placeholder: the refused arms exit before profile
    # resolution runs, and the proceed arms have _resolve_profile patched out.
    # Topology validation is likewise patched -- it is not the unit under test
    # here; the unit under test is train()'s own refusal wiring (#375 batch),
    # and the topology plane has its own suite.
    return {
        "model": "fake-model",
        "dataset": "fake-dataset",
        "output_dir": tmp_path / "out",
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_name": "synthetic-profile",
        "max_steps": 2,
        "save_interval": 1,
    }


def _install_fake_runtime(monkeypatch) -> list:
    """Fake everything UP TO Trainer construction; record constructions.

    Returns the recorder list. A refused run must leave it EMPTY, and a
    proceeding run must leave it non-empty -- asserted both ways, because a
    recorder nobody checks in the negative is a control that cannot fail.
    """
    constructed: list[dict] = []

    class _FakeTopology:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def describe(self) -> str:
            return "fake topology (validation patched out; not the unit under test)"

        def validate_against(self, profile):
            return []

    monkeypatch.setattr(loop, "Topology", _FakeTopology)
    monkeypatch.setattr(
        loop,
        "_resolve_profile",
        lambda cfg: SimpleNamespace(name="synthetic", scheduler="none", gpus_per_node=1),
    )

    class _FakeSplit:
        column_names = ["text"]

        def map(self, fn, batched=False, remove_columns=None):
            return self

        def __len__(self) -> int:
            return 3

    class _FakeTokenizer:
        pad_token = "<pad>"
        eos_token = None

    class _FakeModel:
        # state_dict() -> {} drives _declare_checkpoint down its honest dense
        # branch: no expert keys, no expert-named tensors, agreement by both
        # sources. config=None and no peft wrapper.
        config = None

        def state_dict(self) -> dict:
            return {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(ref):
            return _FakeTokenizer()

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(ref, **kwargs):
            return _FakeModel()

    class _FakeCollator:
        def __init__(self, tokenizer=None, mlm=False):
            self.tokenizer = tokenizer
            self.mlm = mlm

    class _FakeTrainingArguments:
        # NAMED parameters, deliberately: train() introspects this signature
        # and REFUSES any kwarg not named by it (the mechanism the task
        # statement requires every knob to flow through). A bare **kwargs here
        # would name nothing, and every real key would be "dropped" -- which
        # would make the proceed arms exercise the wrong refusal site.
        def __init__(
            self,
            output_dir=None,
            max_steps=None,
            per_device_train_batch_size=None,
            learning_rate=None,
            save_strategy=None,
            save_steps=None,
            seed=None,
            logging_steps=None,
            report_to=None,
            ddp_find_unused_parameters=None,
            bf16=None,
            fp16=None,
            save_safetensors=None,
            optim=None,
            gradient_accumulation_steps=None,
            max_grad_norm=None,
            gradient_checkpointing=None,
            lr_scheduler_type=None,
            warmup_steps=None,
            **extra,
        ):
            self.extra = extra

    class _FakeTrainer:
        def __init__(
            self, model=None, args=None, train_dataset=None, data_collator=None, callbacks=None
        ):
            constructed.append({"model": model, "args": args, "callbacks": list(callbacks or [])})

        def train(self):
            return None

        def save_model(self, path):
            from pathlib import Path

            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            # A minimal safetensors shard (empty JSON header): step 8 globs the
            # artifact format rather than trusting a construction-time flag.
            (p / "model.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")

    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=_FakeAutoModel,
        AutoTokenizer=_FakeAutoTokenizer,
        DataCollatorForLanguageModeling=_FakeCollator,
        Trainer=_FakeTrainer,
        TrainingArguments=_FakeTrainingArguments,
        __version__="0.0-fake",
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: {"train": _FakeSplit()}),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    # A leaked WORLD_SIZE would pull _effective_topology into the run, so both
    # rank-env vars start deleted and any arm that wants a multi-rank runtime
    # sets them itself. LOCAL_WORLD_SIZE is deleted too: on its own it changes
    # nothing (loop.py:427 reads WORLD_SIZE first and returns None when it is
    # unset), but leaving one half of the pair to leak in from the ambient
    # environment makes the arms below depend on where they were run.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    return constructed


ADJUDICATED_CODES = (loop.EXIT_PASS, loop.EXIT_RED, loop.EXIT_UNMEASURED)


@pytest.mark.parametrize("axis", ["tp", "pp", "ep", "cp"])
@pytest.mark.parametrize("world_size", [1, 2], ids=["world1", "world2"])
def test_declared_parallel_degree_above_one_refuses_at_the_375_site(
    tmp_path, monkeypatch, capsys, axis, world_size
):
    """tp/pp/ep/cp=2 INDEPENDENTLY: REFUSE 96 at the #375 site, no Trainer.

    TWO world sizes, and the second one is the whole point. Run only at
    ``world=1`` -- as this test was until it was audited -- ``degree=2``
    satisfies BOTH ``degree > 1`` (what #375 means: the Trainer kwargs carry no
    parallel key, so NO degree above one is executable at any scale) and
    ``degree > world_size`` (ordinary topology arithmetic: you cannot run 2-way
    anything on one rank). One config, two predicates, identical verdict --
    so the arm cannot say which one the shipped code implements, and a rewrite
    that quietly swapped #375's absolute guard for a world-relative one would
    keep every assertion green while making ``tp=2`` executable on a 2-GPU box
    that still cannot execute it.

    ``world=2`` separates them. The runtime can now arithmetically host the
    degree, so a world-relative guard would let the run PROCEED; the absolute
    guard at ``loop.py:1687`` (``getattr(cfg, name) > 1``) must still refuse.
    ``gpus_per_node`` moves to 2 with it, so the DECLARED topology is
    self-consistent with the runtime and the refusal cannot be an artifact of
    declaring one GPU while two ranks exist.

    No ``world=2`` proceed-arm belongs here. The refusals fire at 1687, which
    is upstream of ``_effective_topology`` (1777), so this arm never reaches
    the topology plane -- while a proceeding world=2 run would land in
    ``declared_vs_effective`` against ``_FakeTopology``, which carries no
    degree attributes, and the arm would then be measuring the fake. That
    comparison has its own suite (``test_effective_topology.py``); the
    ``world=1`` inverse control below is what keeps these arms non-vacuous.
    """
    constructed = _install_fake_runtime(monkeypatch)
    kwargs = _base_kwargs(tmp_path)
    if world_size > 1:
        kwargs["gpus_per_node"] = world_size
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(world_size))
    cfg = loop.TrainConfig(**kwargs, **{axis: 2})

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE: train()'s own #375 refusal, not a lookalike 96 from the
    # dependency refusal further down the function.
    assert "[fs:train:refuse]" in out
    assert "finding #375" in out
    # The shipped message names each unwired degree by value ("declared tp=2"),
    # not by rendering a dict. Asserting the dict shape here would pin a form
    # the merge deliberately discarded: a repr's key order is an artifact of
    # insertion, and test_effective_topology.py already has fourteen legs
    # behind the `{axis}={degree}` spelling.
    assert f"declared {axis}=2" in out
    # The deps import runs strictly AFTER the refusal site; its marker must
    # be absent, or the refusal above is not what stopped this run.
    assert "[fs:train:deps]" not in out
    # The refused run still writes its recording (stage="refused") under the
    # reserved manifest name -- refusal is adjudicated and recorded, not lost.
    assert (tmp_path / "out" / loop.MANIFEST_NAME).exists()

    # OUTCOME: 96, and the backend that cannot execute the degree never existed.
    assert rc == loop.EXIT_REFUSE
    assert constructed == [], "a refused run constructed a Trainer"


def test_all_parallel_degrees_at_one_reach_trainer_construction(tmp_path, monkeypatch, capsys):
    """INVERSE CONTROL: tp=pp=ep=cp=1 must get as far as building a Trainer.

    This is the arm that makes the four refusal tests non-vacuous: if some
    upstream condition stopped EVERY run, the refusal arms would pass while
    measuring nothing. Here the same fakes, same config spine, and same
    assertion probes must show the run proceeding THROUGH the refusal site.
    """
    constructed = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path))

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE: Trainer construction was reached, and no refusal of any
    # kind fired on the way.
    assert "[fs:train:trainer]" in out
    assert "[fs:train:refuse]" not in out

    # OUTCOME: a Trainer was actually constructed, and the verdict is an
    # adjudicated one -- 96 here would mean a refusal fired where none should,
    # and 1 is in none of the four declared states.
    assert constructed, "tp=pp=ep=cp=1 must reach Trainer construction"
    assert rc in ADJUDICATED_CODES


@pytest.mark.parametrize("strategy", ["fsdp", "zero3"])
def test_sharding_strategy_without_a_backend_refuses(tmp_path, monkeypatch, capsys, strategy):
    """A declared sharded execution this plane never built: REFUSE 96."""
    constructed = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy=strategy)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE: the message names the declaration that was rejected, so a
    # 96 fired by anything else cannot satisfy this arm.
    assert "[fs:train:refuse]" in out
    assert f"sharding_strategy={strategy!r} is declared" in out
    assert "[fs:train:deps]" not in out

    # OUTCOME
    assert rc == loop.EXIT_REFUSE
    assert constructed == [], "a refused run constructed a Trainer"


@pytest.mark.parametrize("strategy", [None, "ddp"])
def test_undeclared_or_ddp_sharding_proceeds(tmp_path, monkeypatch, capsys, strategy):
    """INVERSE CONTROL: the only honourable sharding declarations proceed."""
    constructed = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), sharding_strategy=strategy)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:trainer]" in out
    assert "[fs:train:refuse]" not in out

    # OUTCOME
    assert constructed, f"sharding_strategy={strategy!r} must reach Trainer construction"
    assert rc in ADJUDICATED_CODES


def test_cpu_optimizer_offload_true_refuses(tmp_path, monkeypatch, capsys):
    """Offload declared, no offload backend wired: REFUSE 96, never claimed."""
    constructed = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), cpu_optimizer_offload=True)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:refuse]" in out
    assert "cpu_optimizer_offload=True is declared" in out
    assert "[fs:train:deps]" not in out

    # OUTCOME
    assert rc == loop.EXIT_REFUSE
    assert constructed == [], "a refused run constructed a Trainer"


@pytest.mark.parametrize("offload", [None, False])
def test_cpu_optimizer_offload_false_or_undeclared_proceeds(tmp_path, monkeypatch, capsys, offload):
    """INVERSE CONTROL: no-offload declarations match what the backend does."""
    constructed = _install_fake_runtime(monkeypatch)
    cfg = loop.TrainConfig(**_base_kwargs(tmp_path), cpu_optimizer_offload=offload)

    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE
    assert "[fs:train:trainer]" in out
    assert "[fs:train:refuse]" not in out

    # OUTCOME
    assert constructed, f"cpu_optimizer_offload={offload!r} must reach Trainer construction"
    assert rc in ADJUDICATED_CODES
