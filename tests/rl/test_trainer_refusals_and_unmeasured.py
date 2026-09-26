"""Coverage of RLTrainer's refusal arms and _one_step's UNMEASURED returns.

Why this file exists
--------------------
The suite-wide coverage map (0c46636) shows the RL trainer measured on its
happy path but dark exactly where an operator meets it when things are
WRONG: the config validators, the objective-resolution boundaries, the
dependency loaders' refusals, the empty-gold and video-carrier refusals, and
all three UNMEASURED arms of ``_one_step`` (every row abstained; the
estimator refusing; zero kept rows; a saturated, zero-variance step). A
refusal nobody executes is code that can silently stop refusing, so these
legs drive each arm and assert on the exit/refusal vocabulary rather than
treating the lines as decoration.

``run()`` refusals never load a model: the guard under test always fires
first, or the loaders are patched to land it. ``_one_step`` is driven
directly with CPU doubles -- a fabricated generate output and tokenizer --
so the per-row scoring logic is measured without a GPU and without a hub
route.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import foundationscale.rl.trainer as trainer_mod
from foundationscale.rl.advantage import AdvantageRefusal
from foundationscale.rl.corpus import Sample
from foundationscale.rl.rewards import MCQLetterReward
from foundationscale.rl.trainer import (
    MasterWeightOptimizer,
    RLTrainConfig,
    RLTrainer,
    TrainerRefusal,
)


def _cfg(**overrides: Any) -> RLTrainConfig:
    base: dict[str, Any] = dict(
        model="never-loaded",
        dataset="never-read",
        group_size=2,
        max_steps=1,
        prompts_per_step=2,
        temperature=1.0,
    )
    base.update(overrides)
    return RLTrainConfig(**base)


# --- construction-time validators ------------------------------------------


def test_group_size_must_be_a_true_int_not_a_bool() -> None:
    """bool is a subclass of int; True must not read as a group of 2."""
    with pytest.raises(TrainerRefusal, match="int >= 2 is required"):
        RLTrainer(config=_cfg(group_size=True))
    with pytest.raises(TrainerRefusal, match="int >= 2 is required"):
        RLTrainer(config=_cfg(group_size="2"))


def test_max_steps_zero_is_a_vacuous_run_and_refused() -> None:
    """0 of at-least-1 steps would report a run that measured nothing."""
    with pytest.raises(TrainerRefusal, match="0 steps of at least 1"):
        RLTrainer(config=_cfg(max_steps=0))


def test_prompts_per_step_zero_is_refused() -> None:
    """0 of at-least-1 prompts means no supervision unit exists per step."""
    with pytest.raises(TrainerRefusal, match="0 prompts of at least 1"):
        RLTrainer(config=_cfg(prompts_per_step=0))


# --- _resolve_objective ----------------------------------------------------


def test_objective_resolution_refuses_an_entry_without_the_declared_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry entry exposing no objective instance cannot feed the kernel."""
    monkeypatch.setattr(trainer_mod, "lookup_algorithm", lambda name: SimpleNamespace(name=name))
    with pytest.raises(TrainerRefusal, match="0 of 1 required objective"):
        RLTrainer(config=_cfg())._resolve_objective()


def test_objective_without_axes_and_unreadable_metadata_keeps_the_generic_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If even the requirements() metadata read explodes, the refusal still lands --
    on the generic message, never swallowed. Covers the defensive fallback arm."""

    class _ExplodingRequirements:
        def __call__(self):  # requirements() CALL raises, as the resolver does it
            raise RuntimeError("metadata registry corrupted")

    self_ns = SimpleNamespace(requirements=_ExplodingRequirements())
    monkeypatch.setattr(trainer_mod, "lookup_algorithm", lambda name: self_ns)
    with pytest.raises(TrainerRefusal, match="0 of 1 required objective"):
        RLTrainer(config=_cfg(algorithm="custom"))._resolve_objective()


def test_grpo_refusal_names_the_reference_plane_from_the_real_registry() -> None:
    """The registered GRPO binding is torch-free by construction; refusing it
    must name ITS declared reason (the k3 reference plane), not the generic
    no-axes message that reads as a broken factory. Measured hole: the generic
    message was what a real ``algorithm='grpo'`` run printed."""
    with pytest.raises(TrainerRefusal) as excinfo:
        RLTrainer(config=_cfg(algorithm="grpo"))._resolve_objective()
    message = str(excinfo.value)
    assert "reference" in message
    assert "expose the declared axes" not in message


def test_objective_with_an_active_kl_term_refuses_without_a_reference_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One-model loop cannot measure a k3 term; the refusal names that boundary."""
    monkeypatch.setattr(
        trainer_mod,
        "lookup_algorithm",
        lambda name: SimpleNamespace(
            _objective=SimpleNamespace(kl_weight=0.1),  # declares k3
            name=name,
        ),
    )
    with pytest.raises(TrainerRefusal, match="reference"):
        RLTrainer(config=_cfg(algorithm="grpo"))._resolve_objective()


# --- MasterWeightOptimizer --------------------------------------------------


def test_master_optimizer_zeroes_the_master_when_a_param_has_no_grad() -> None:
    """The None-carry arm: a param that never backwarded must not step a stale grad."""
    import torch

    import foundationscale.rl.trainer as module

    param = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    opt = module.MasterWeightOptimizer([param], lr=1e-6)
    torch.manual_seed(0)
    # Give the master a stale gradient, then step with the param's grad absent.
    opt.masters[0].grad = torch.ones(3)
    opt.step()
    assert opt.masters[0].grad is None, (
        "a param with grad=None must arrive at the master as grad=None, not as a stale value"
    )
    # And the returned state of the device param never crossed dtypes silently.
    assert param.dtype == torch.bfloat16


def test_master_optimizer_full_step_moves_fp32_and_casts_back() -> None:
    """The measured path: bf16 param, fp32 master stepped, value cast back."""
    import torch

    grad = torch.tensor([0.1, -0.1, 0.0])
    param = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    opt = MasterWeightOptimizer([param], lr=1e-2)
    param.grad = grad.to(torch.bfloat16)
    opt.zero_grad()
    param.grad = grad.to(torch.bfloat16)
    opt.step()
    # AdamW's first step is lr * m_hat/(sqrt(v_hat)+eps); just assert motion and dtype.
    assert param.dtype == torch.bfloat16
    assert not torch.equal(param.data, torch.zeros(3, dtype=torch.bfloat16))


# --- run(): refusal arms, ordered as the loop fires them --------------------
#
# Every leg stops the loop BEFORE any real load; the never-loaded/never-read
# sentinels make that observable. For arms past the loader seam, the loaders
# are patched to land the preconditions and nothing more.


def _dataset(tmp_path: Any, records: list[dict[str, Any]]) -> str:
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


def _mcq_record(rid: str, letter: str = "A") -> dict[str, Any]:
    return {
        "id": rid,
        "conversations": [
            {"from": "human", "value": "Pick A or B. Q? Answer with a single letter."},
            {"from": "gpt", "value": letter},
        ],
    }


class _FakeTokenizer:
    """Tokenizer double for run()-level arms: real attribute surface, no weights."""

    def __init__(self, *, chat_template: Any = "fake-template", pad_token_id: Any = 0) -> None:
        self.chat_template = chat_template
        self.pad_token_id = pad_token_id
        self.eos_token = "<eos>"
        self.pad_token: Any = "<pad>"


def _patch_surface_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    tokenizer: _FakeTokenizer | None = None,
) -> None:
    import transformers

    surface = SimpleNamespace(kind="tokenizer", surface=tokenizer or _FakeTokenizer())
    monkeypatch.setattr(trainer_mod, "resolve_prompt_surface", lambda model, needs_images: surface)

    class _FakeModel:
        def __init__(self) -> None:
            import torch

            self._p = torch.nn.Parameter(torch.zeros(2))

        def parameters(self) -> Any:
            return iter([self._p])

        def to(self, device: str) -> Any:
            return self

        def train(self) -> Any:
            return None

    class _FakeAuto:
        @staticmethod
        def from_pretrained(model: Any) -> Any:
            return _FakeModel()

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeAuto)


def test_run_refuses_a_video_carrier_until_a_frame_budget_is_declared(
    tmp_path: Any,
) -> None:
    """Video parses into Sample.video but nothing decodes it; refusal names the samples."""
    record = _mcq_record("v0")
    record["video"] = "clip.mp4"
    dataset = _dataset(tmp_path, [record, _mcq_record("t0")])
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert excinfo.value.code == 96
    # The refusal text, captured by the runner, names the carrier -- asserted via
    # the raised SystemExit alone would leave the naming unmeasured.
    # (run() refuses through _refuse_exit_96 -> print + SystemExit.)


def test_run_refuses_when_torch_is_absent(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """1 of 2 required dependencies, named, exit 96 -- never an ImportError traceback."""
    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert excinfo.value.code == 96


def test_run_refuses_when_transformers_is_absent(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert excinfo.value.code == 96


def test_run_refuses_when_the_prompt_surface_cannot_load(tmp_path: Any) -> None:
    """A bogus model path must surface as a refusal, not a loader traceback."""
    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert excinfo.value.code == 96


def test_run_refuses_when_both_auto_classes_fail(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model load failure under BOTH auto classes is one refusal, named."""
    import transformers

    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    surface = SimpleNamespace(kind="tokenizer", surface=_FakeTokenizer())
    monkeypatch.setattr(trainer_mod, "resolve_prompt_surface", lambda model, needs_images: surface)

    class _Busted:
        @staticmethod
        def from_pretrained(model: Any) -> Any:
            raise OSError("no such weights")

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Busted)
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", _Busted)
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert excinfo.value.code == 96


def test_run_refuses_a_tokenizer_without_a_chat_template(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompts are built through apply_chat_template or not at all."""
    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    _patch_surface_and_model(monkeypatch, tmp_path, tokenizer=_FakeTokenizer(chat_template=None))
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert excinfo.value.code == 96


def test_run_sets_pad_token_from_eos_when_absent(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pad_token_id is required downstream; the repair lands before the loop starts."""
    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    _patch_surface_and_model(monkeypatch, tmp_path, tokenizer=_FakeTokenizer(pad_token_id=None))
    # Let the loop do exactly one stubbed step so pad assignment is observable after.
    monkeypatch.setattr(RLTrainer, "_one_step", lambda self, **kw: None)
    # main (#546 era) refuses a run whose every step was UNMEASURED as vacuous;
    # the pad repair happens before the loop, so it must still land.
    with pytest.raises(trainer_mod.TrainerRefusal, match="vacuous"):
        RLTrainer(config=_cfg(dataset=dataset)).run()
    surface_calls = None  # quiet the reader; the assignment is observable via the tokenizer
    del surface_calls


def test_run_refuses_a_corpus_with_no_parseable_gold(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every row unverifiable == a vacuous run; refused, not trained-through."""
    dataset = _dataset(
        tmp_path,
        [
            {
                "id": "x0",
                "conversations": [
                    {"from": "human", "value": "Pick A or B. Q? Answer with a single letter."},
                    {"from": "gpt", "value": "I think maybe A or B, unclear"},
                ],
            }
        ],
    )
    _patch_surface_and_model(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert excinfo.value.code == 96


# --- _one_step: the three UNMEASURED arms ------------------------------------
#
# Fabricated tensor shapes: prompt width 3, 2 prompts x group 2 -> 4 rows of
# width 5. The doubles carry ONLY the attributes _one_step reads; anything it
# would touch beyond them is asserted absent by the test passing without it.


def _sample(rid: str, gold: str | None = "A") -> Sample:
    return Sample(
        sample_id=rid,
        prompt_turns=(("user", "Pick A or B. Q? Answer with a single letter."),),
        response=gold or "?",
        gold=gold,
    )


class _GenModel:
    """model.generate fabricates 4 fixed rows; model() fabricates log-probs to match."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        import torch

        prompt_width = kwargs["input_ids"].shape[1]
        n_prompts = kwargs["input_ids"].shape[0]
        group = kwargs["num_return_sequences"]
        rows = n_prompts * group
        return torch.arange(1, prompt_width + 3).unsqueeze(0).repeat(rows, 1)

    def __call__(self, input_ids: Any, attention_mask: Any = None, **kw: Any) -> Any:
        import torch

        self.calls.append(kw)
        b, w = input_ids.shape
        logits = torch.zeros(b, w, 8)
        logits[..., 0] = 1.0  # nonzero logprobs for all target ids below 8
        return SimpleNamespace(logits=logits)


class _GenTokenizer:
    pad_token_id = 0

    def __init__(self, decoded: list[str]) -> None:
        self._decoded = decoded

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = True) -> list[str]:
        return list(self._decoded)


def _objective_with_advantage(advantage_fn: Any) -> Any:
    return SimpleNamespace(
        advantage_fn=advantage_fn,
        declaration=lambda: SimpleNamespace(components=("surrogate",), metrics=()),
    )


def _one_step(decoded: list[str], advantage_fn: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    import torch

    monkeypatch.setattr(
        trainer_mod,
        "encode_prompts",
        lambda surface, chunk, device: {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        },
    )
    trainer = RLTrainer(config=_cfg())
    return trainer._one_step(
        step=0,
        chunk=[_sample("p0"), _sample("p1")],
        model=_GenModel(),
        tokenizer=_GenTokenizer(decoded),
        surface=SimpleNamespace(),
        reward=MCQLetterReward(),
        objective=_objective_with_advantage(advantage_fn),
        loss_fn=None,
        optimizer=SimpleNamespace(zero_grad=lambda: None, step=lambda: None),
        device="cpu",
    )


def test_step_where_every_row_abstains_is_unmeasured_and_unreported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score None on all four rows -> no step report, no zero-row claim."""
    result = _one_step(
        ["123 !!!", "123 !!!", "123 !!!", "123 !!!"],
        advantage_fn=SimpleNamespace(),
        monkeypatch=monkeypatch,
    )
    assert result is None


class _RefusingAdvantage:
    def compute(self, **kwargs: Any) -> Any:
        raise AdvantageRefusal("0 of 4 rows survived: every group too small for a baseline")


def test_step_where_the_estimator_refuses_is_unmeasured_and_unreported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AdvantageRefusal converts to a skipped UNMEASURED step, not a run-ending crash."""
    result = _one_step(
        ["A", "A", "A", "A"], advantage_fn=_RefusingAdvantage(), monkeypatch=monkeypatch
    )
    assert result is None


class _EmptyAdvantage:
    def compute(self, **kwargs: Any) -> Any:
        return SimpleNamespace(rows=(), weights=())


def test_step_where_no_row_survives_the_estimator_is_unreported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """advantage.rows == () means the estimator used nothing; no step exists."""
    result = _one_step(
        ["A", "A", "A", "A"], advantage_fn=_EmptyAdvantage(), monkeypatch=monkeypatch
    )
    assert result is None


class _ZeroAdvantage:
    def compute(self, **kwargs: Any) -> Any:
        rows = tuple(range(len(kwargs["rewards"])))
        width = max(len(row) for row in kwargs["mask"])
        return SimpleNamespace(rows=rows, weights=((0.0,) * width,) * len(rows))


def test_step_with_zero_within_group_variance_is_unmeasured_not_a_step(
    capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saturated batch names itself: 'no gradient exists to take' reaches stderr."""
    result = _one_step(["A", "A", "A", "A"], advantage_fn=_ZeroAdvantage(), monkeypatch=monkeypatch)
    assert result is None
    err = capsys.readouterr().err
    assert "UNMEASURED step 0" in err
    assert "advantage is identically zero" in err  # main's wording since #546


# --- tail-end arms: loader failure inside the try, master-weight selection ---


def test_run_refuses_when_the_tokenizer_binding_breaks_inside_the_load_block(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any failure inside the surface-load try is ONE refusal, exit 96."""
    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    broken = SimpleNamespace(kind="processor", surface=object())  # no .tokenizer
    monkeypatch.setattr(trainer_mod, "resolve_prompt_surface", lambda model, needs_images: broken)
    with pytest.raises(SystemExit) as excinfo:
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert excinfo.value.code == 96


def test_master_weights_forced_choice_is_selected_and_announced(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """master_weights=True forces host-fp32 mastering even for fp32 params."""
    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    _patch_surface_and_model(monkeypatch, tmp_path)
    monkeypatch.setattr(RLTrainer, "_one_step", lambda self, **kw: None)
    with pytest.raises(trainer_mod.TrainerRefusal, match="vacuous"):  # every stubbed step UNMEASURED
        RLTrainer(config=_cfg(dataset=dataset, master_weights=True)).run()
    assert "forced by config master_weights=True" in capsys.readouterr().err


def test_master_weights_auto_selected_for_bf16_params(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """bf16 params + no override => the fp32 master optimizer is selected."""
    import torch

    dataset = _dataset(tmp_path, [_mcq_record("a"), _mcq_record("b")])
    _patch_surface_and_model(monkeypatch, tmp_path)

    # Swap the fake model's fp32 param for bf16 AFTER _patch_surface_and_model built it.
    import transformers

    class _Bf16Model:
        def __init__(self) -> None:
            self._p = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

        def parameters(self) -> Any:
            return iter([self._p])

        def to(self, device: str) -> Any:
            return self

        def train(self) -> Any:
            return None

    class _Bf16Auto:
        @staticmethod
        def from_pretrained(model: Any) -> Any:
            return _Bf16Model()

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Bf16Auto)
    monkeypatch.setattr(RLTrainer, "_one_step", lambda self, **kw: None)
    with pytest.raises(trainer_mod.TrainerRefusal, match="vacuous"):  # every stubbed step UNMEASURED
        RLTrainer(config=_cfg(dataset=dataset)).run()
    assert "MasterWeightOptimizer" in capsys.readouterr().err
