"""The four kernel-axis declarations must reach the kernel, not just the CLI.

loop.py grew nine "declaration axes" whose ``None`` default means NOT DECLARED
(#342's abstention-is-not-a-claim rule). This module measures four of them at
the site each one claims to affect, never at the site that merely parses them:

* ``gradient_checkpointing`` -> the model's own ``is_gradient_checkpointing``,
  read DURING a training step. Asserting only that the flag was accepted while
  the model itself was never touched is precisely the silent no-op this suite
  exists to catch, and reading it after ``trainer.train()`` returns would
  conflate "never enabled" with "enabled and disabled again", because
  transformers is free to call ``gradient_checkpointing_disable()`` at train end.
* ``attn_implementation`` -> ``model.config._attn_implementation`` after a real
  ``from_pretrained``. "eager" is the discriminating value: on every
  transformers release that accepts the knob, the silent default is "sdpa"
  where available, so a dropped declaration and an honoured one land on
  different readings.
* ``lr_scheduler_type`` / ``warmup_steps`` -> ``optimizer.param_groups`` LR
  observed per step-begin. A schedule that silently stayed flat, a silently
  dropped warmup, and a silently swapped scheduler family each move at least
  one asserted reading.

Every test asserts REACHED-SITE separately from OUTCOME (#291, #294, #372: a
control that never arrives at the site it measures is vacuous). No skips
(house rule): when the train extra is absent each test takes the UNMEASURED
arm in ``tiny_run``, which measures the only thing that environment makes
measurable -- the torch-free import contract of loop.py -- and that arm still
differs on a broken tree (a module-scope torch import flips it red).

Everything runs on CPU in seconds against the smallest real GPT2 the
transformers loader will build, synthesized locally: no network, no estate
identifiers, tmp_path only.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# loop.py is torch-free BY CONTRACT -- its REFUSE paths (96) exist to run on
# test runners and login nodes that have no train extra at all -- so importing
# it at module scope here is not a convenience, it is part of the pinned
# behaviour: the UNMEASURED arm below depends on this import NOT pulling torch
# into the process, and asserts exactly that.
from foundationscale.train import loop

_EXIT_REFUSE = loop.EXIT_REFUSE

# The heavy legs below, pinned as data because they are the run's own facts.
_BASE_LR = 1e-3
_WARMUP_STEPS = 5
_SCHEDULER_STEPS = 20
# Observation steps for the scheduler arms: 1 sits inside the declared warmup
# (cosine-with-warmup evaluates it at 1/5 of the base), 6 is the first step
# past the warmup plateau (~0.989 of the base on the fixed tree, and 0.70 on a
# tree that dropped the warmup declaration -- two different readings, one arm).
_EARLY_STEP = 1
_LATE_STEP = 6
# Three consecutive post-warmup steps, read as a CURVATURE probe. The two
# threshold assertions below separate cosine from linear only if the warmup
# declaration was dropped along with the family one; if `warmup_steps` is
# forwarded and `lr_scheduler_type` is not, linear-with-warmup reads 0.200 at
# step 1 and 0.933 at step 6 -- inside BOTH thresholds, so a dropped family
# passes. That is the vacuous half of this test, and it is the exact axis the
# test is named for. The second difference is the instrument that cannot be
# fooled that way: HF's linear lambda is affine in the step, so lr[6]-2*lr[7]+
# lr[8] is EXACTLY zero for it and for any constant schedule, while cosine is
# strictly concave over the first half of its decay and reads -0.020 of base.
# A sign test with a 1e14 margin over float error, not a threshold with 0.05.
_CURVE_STEPS = (6, 7, 8)
# Halfway between the two families in units of base LR. Cosine: -0.0200x.
# Linear or flat: 0.0000x, by the arithmetic and not by measurement.
_CURVATURE_CEILING = -0.005


def _missing_train_dep() -> str | None:
    """find_spec PROBES WITHOUT IMPORTING.

    Importing torch inside the availability check would both burn the
    torch-free contract this module pins and make every verdict below depend
    on whether the probe ran (house rule: torch is never imported at module
    scope). ``accelerate`` is probed because modern transformers refuses to
    construct a Trainer without it -- loop.py turns that absence into a
    named REFUSE (96), and this probe keeps the absent-extra arm from being
    surprised by it.
    """
    for name in ("torch", "transformers", "datasets", "accelerate"):
        if importlib.util.find_spec(name) is None:
            return name
    return None


def _assert_torch_free_surface(missing: str) -> None:
    """The shared UNMEASURED arm for every test in this module.

    An environment without foundationscale[train] cannot observe a scheduler
    or an attention backend, so skipping would be the honest-looking choice --
    and skips are forbidden. What this environment CAN observe is the contract
    the refusal paths in loop.py are built on: the module must import without
    torch, and the probe above must not have imported torch either. Both
    assertions differ on a broken tree -- a module-scope torch import anywhere
    in loop.py flips them red on exactly the hosts the REFUSE path serves.
    """
    assert "torch" not in sys.modules, (
        f"train leg is UNMEASURED here ({missing} absent), yet torch is in "
        "sys.modules before foundationscale.train.loop was touched: this "
        "process does not present the torch-free surface the REFUSE paths "
        "are designed for -- the probe or a conftest imported it at scope"
    )
    importlib.import_module("foundationscale.train.loop")
    assert "torch" not in sys.modules, (
        f"train leg is UNMEASURED here ({missing} absent), and importing "
        "foundationscale.train.loop dragged torch into the process: the "
        "module is no longer torch-free at scope, which silently downgrades "
        "every dep-absence REFUSE (96) into a bare ImportError traceback and "
        "exit 1 -- a code in none of the four declared states (#171)"
    )


def _write_tiny_model(model_dir: Path) -> None:
    """Genuinely tiny, genuinely real: a 1-layer GPT2 plus a WordLevel tokenizer.

    Both are constructed and ``save_pretrained``d locally so nothing reaches a
    hub or a cache (no estate identifiers, no network, seconds on CPU). The
    tokenizer is a tokenizers backend wrapped in PreTrainedTokenizerFast --
    material ``AutoTokenizer.from_pretrained`` loads without any model id.
    GPT2 is chosen because it ties embeddings (so the tied-alias handling in
    the declaration path is exercised for real) and supports eager/sdpa
    attention, which is what the attn axis needs to measure.
    """
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
        "delta": 7,
        "epsilon": 8,
        "zeta": 9,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    config = transformers.GPT2Config(
        vocab_size=len(vocab),
        n_layer=1,
        n_head=2,
        n_embd=8,
        n_positions=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.GPT2LMHeadModel(config)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def _write_tiny_dataset(path: Path) -> None:
    """A .jsonl with a ``text`` column -- the thin path's one dataset contract."""
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    lines = [{"text": " ".join(words[(i + j) % len(words)] for j in range(5))} for i in range(24)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


@pytest.fixture
def tiny_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """A complete, launchable train fixture -- or the UNMEASURED arm.

    Returns ``None`` when the train extra is absent; the UNMEASURED arm then
    runs HERE (inside the fixture) rather than as a skip, so every consuming
    test sees an environment verdict instead of no verdict at all.
    """
    missing = _missing_train_dep()
    if missing is not None:
        _assert_torch_free_surface(missing)
        return None

    # If the test runner actually was launched under torchrun, the declared
    # topology (1x1) would be compared against that runtime and could block
    # for reasons orthogonal to these axes. These tests measure kernel wiring
    # on one CPU process; the declared-vs-effective axis has its own suite.
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    # Topology-vs-profile validation is deliberately NOT this module's subject:
    # measuring it from here would couple four kernel-axis verdicts to a
    # separate instrument. It is patched to an explicit empty finding list --
    # stated in the fixture, visible at the patch site, never silently bypassed
    # because the partition scan or profile schema drifted.
    monkeypatch.setattr(
        loop.Topology,
        "validate_against",
        lambda self, profile: [],
    )

    model_dir = tmp_path / "tiny-model"
    _write_tiny_model(model_dir)
    dataset_path = tmp_path / "tiny.jsonl"
    _write_tiny_dataset(dataset_path)

    def make_config(name: str, **overrides: Any) -> loop.TrainConfig:
        kwargs: dict[str, Any] = {
            "model": str(model_dir),
            "dataset": str(dataset_path),
            "output_dir": tmp_path / "runs" / name,
            # Fail-closed fields mirror the CLI: one machine, one GPU.
            "nodes": 1,
            "gpus_per_node": 1,
            # A synthetic profile object, not profile_by_name: which built-in
            # profiles the package ships is irrelevant to what this module
            # measures, and depending on one would make these verdicts move
            # when the profile registry changes.
            "profile": SimpleNamespace(name="synthetic-direct", scheduler="none", gpus_per_node=1),
            # "sft" is what the loop actually implements (CLM collator), so
            # declaring it is a true statement, not a plausible default --
            # the CLI's own reasoning, applied here.
            "objective": "sft",
            "max_steps": 4,
            "per_device_batch_size": 1,
            "learning_rate": _BASE_LR,
            # No mid-run saves: the save gate and the final checkpoint are not
            # this module's subject either, and a blocked mid-run save would
            # stop training before the probes below have sampled.
            "save_interval": 50,
            "seed": 7,
        }
        kwargs.update(overrides)
        return loop.TrainConfig(**kwargs)

    return SimpleNamespace(make_config=make_config)


@pytest.fixture
def capture_trainer(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Wrap transformers.Trainer so tests can read the OBJECTS the loop built.

    The loop re-imports ``Trainer`` from the transformers module at train time
    (the heavy imports live inside ``train()`` by contract), so replacing the
    module attribute is enough and nothing in the loop is patched. Optional
    probe callbacks are appended AFTER construction, on top of the callback
    list the loop itself supplied -- the save-gate and objective-gate
    callbacks still run, unmodified.
    """

    def install(callback: Any = None) -> list[Any]:
        import transformers

        real_trainer = transformers.Trainer
        held: list[Any] = []

        class _CapturingTrainer(real_trainer):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                if callback is not None:
                    self.add_callback(callback)
                held.append(self)

        monkeypatch.setattr(transformers, "Trainer", _CapturingTrainer)
        return held

    return install


def test_gradient_checkpointing_declaration_reaches_the_model(
    tiny_run: Any, capture_trainer: Any
) -> None:
    """Declare gradient_checkpointing=True; the MODEL must report it, mid-step.

    A Trainer that accepted the flag without ever calling
    ``model.gradient_checkpointing_enable`` is the exact deficit class this
    suite exists to catch, so the assertion lands on the model's own state,
    sampled by a callback at every step-begin: transformers may disable
    checkpointing again at train end, and a post-hoc read could not separate
    "never enabled" from "enabled then disabled".
    """
    if tiny_run is None:
        return  # UNMEASURED arm ran inside the fixture
    import transformers

    samples: list[bool | None] = []

    class _GradientCheckpointingProbe(transformers.TrainerCallback):
        def on_step_begin(
            self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any
        ) -> Any:
            if model is not None:
                samples.append(
                    None
                    if not hasattr(model, "is_gradient_checkpointing")
                    else bool(model.is_gradient_checkpointing)
                )
            return control

    held = capture_trainer(callback=_GradientCheckpointingProbe())
    cfg = tiny_run.make_config("gc", gradient_checkpointing=True, max_steps=4)
    rc = loop.train(cfg)

    # REACHED-SITE, separately from OUTCOME (#291/#294/#372): a Trainer was
    # constructed, the run was not refused, and at least one step actually
    # observed a model exposing the attribute. Without these three, the final
    # assertion could pass vacuously over zero samples.
    assert held, "REACHED-SITE fails: no Trainer was ever constructed"
    assert rc != _EXIT_REFUSE, (
        f"REACHED-SITE fails: the run was REFUSED ({rc}) before training; on the "
        "pinned train extra, gradient_checkpointing is an accepted "
        "TrainingArguments knob, so a refusal here means the declaration was "
        "rejected, which is itself a verdict worth seeing"
    )
    assert samples and None not in samples, (
        f"REACHED-SITE fails: the probe recorded {len(samples)} step begin(s) but "
        "never found a model reporting is_gradient_checkpointing -- the "
        "assertion below would be measuring nothing"
    )

    # OUTCOME: no sampled step ran with the mechanism off. On the broken tree
    # this suite guards against -- the declared value never entering
    # TrainingArguments (#342's silent-drop shape) -- the engine default
    # (False) applies and EVERY sample is False, so this arm flips.
    false_steps = sum(1 for s in samples if not s)
    assert false_steps == 0, (
        "declared gradient_checkpointing=True, but the model reports "
        f"is_gradient_checkpointing False on {false_steps}/{len(samples)} "
        "observed step(s): the declaration was accepted with no effect -- the "
        "precise silent no-op this axis exists to refuse to tolerate"
    )


def test_declared_attn_implementation_appears_on_the_loaded_model(
    tiny_run: Any, capture_trainer: Any
) -> None:
    """Declare attn_implementation="eager"; the loaded model must carry it.

    The reading is ``model.config._attn_implementation`` -- transformers' own
    record of what the load actually selected -- never the argv, and never the
    ``from_pretrained`` SIGNATURE. Signature introspection was the original
    instrument here and it was wrong in every release: the Auto class is a
    dispatcher declared ``(*model_args, **kwargs)``, so the parameter is never
    named, and a guard reading that as "the release does not support it"
    refused 100% of declared values while this test stayed green by asserting
    the refusal. An axis certified entirely by its own dead branch.

    The UNDECLARED control is what makes the "eager" reading mean anything: it
    measures what this stack picks when nothing is declared. If that is also
    "eager" (a build with no sdpa), the two arms cannot be told apart and the
    test says UNMEASURED instead of claiming a pass.
    """
    if tiny_run is None:
        return

    # CONTROL arm first: no declaration at all. Its reading is the silent
    # default -- the value a tree that parsed "eager" and dropped it would
    # produce. This is measured on THIS host, not assumed to be "sdpa".
    control_held = capture_trainer()
    control_cfg = tiny_run.make_config("attn-control", max_steps=2)
    control_rc = loop.train(control_cfg)
    assert control_rc != _EXIT_REFUSE, (
        f"REACHED-SITE fails: the undeclared control run was REFUSED "
        f"({control_rc}); with no attn_implementation declared there is "
        "nothing on this axis to refuse"
    )
    assert control_held, "REACHED-SITE fails: the control built no Trainer"
    default_impl = getattr(control_held[0].model.config, "_attn_implementation", None)
    assert default_impl is not None, (
        "REACHED-SITE fails: this release publishes no _attn_implementation "
        "reading, so the declaration below could not be verified by anything"
    )
    if default_impl == "eager":
        # Not a pass and not a failure: on this build the declared value and
        # the silent default coincide, so no reading can separate a wired axis
        # from a dropped one. Stated, not swallowed (#291/#294/#372).
        pytest.fail(
            "UNMEASURED: the undeclared default is already 'eager' on this "
            "build, so declaring 'eager' cannot be distinguished from dropping "
            "the declaration. Re-take this arm on a build with sdpa available, "
            "or declare a value this stack does not default to"
        )

    held = capture_trainer()
    cfg = tiny_run.make_config("attn-eager", attn_implementation="eager", max_steps=2)
    rc = loop.train(cfg)

    # REACHED-SITE. A REFUSE here is the regression this test exists to catch:
    # the loader demonstrably accepts the knob (the control just read a real
    # value off the same class), so refusing means the guard is guessing again.
    assert rc != _EXIT_REFUSE, (
        f"REACHED-SITE fails: the run was REFUSED ({rc}) although this stack "
        f"accepts the parameter -- the undeclared control loaded and reported "
        f"{default_impl!r}"
    )
    assert held, "REACHED-SITE fails: no Trainer was ever constructed"

    # OUTCOME. On the broken tree (declaration parsed, never forwarded) the
    # reading is the control's default, not "eager": the arm flips.
    selected = getattr(held[0].model.config, "_attn_implementation", None)
    assert selected == "eager", (
        "declared attn_implementation='eager', but the loaded model reports "
        f"{selected!r} (the undeclared default on this build is "
        f"{default_impl!r}): the declaration never reached from_pretrained, "
        "or something overrode it on the way in"
    )


def test_unusable_attn_implementation_refuses_96_instead_of_falling_back(
    tiny_run: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A kernel this build cannot provide is REFUSED (96), never fallen back to.

    The declared value is "flash_attention_2" against the REAL loader. On a
    build without flash-attn, transformers raises ImportError; on an
    unsupported NAME it raises ValueError. Both are the same event to the
    operator -- this build cannot give you that kernel -- and the contract is
    that both land on 96, not on RED 5 (a run that failed) and not on 0 (a run
    that quietly trained under sdpa while the manifest claimed flash
    attention). That last one is #342's defect class and is the reason this
    arm exists at all.

    This test used to patch in a "legacy-shaped loader" whose signature named
    no ``attn_implementation``, and assert the refusal fired BEFORE any load.
    The double was the defect: no real loader names that parameter -- the Auto
    class is ``(*model_args, **kwargs)`` in every release -- so the fixture
    modelled a shape that does not exist, and the pre-load refusal it pinned
    fired on every declaration, real ones included. The oracle is now the
    load's own outcome, so the loader IS reached; what must not happen is
    training continuing after it could not honour the declaration.
    """
    if tiny_run is None:
        return

    cfg = tiny_run.make_config("attn-refuse", attn_implementation="flash_attention_2", max_steps=2)
    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE for the instrument: is flash-attn actually absent here? If
    # some future image installs it, "cannot provide" is false on this build
    # and the correct verdict flips to a successful, honoured run -- asserting
    # 96 then would pin an environment fact, not a contract (#252).
    flash_available = False
    try:
        import flash_attn  # noqa: F401

        flash_available = True
    except Exception:  # noqa: BLE001 -- any import failure means unavailable
        flash_available = False
    if flash_available:
        assert rc != _EXIT_REFUSE, (
            "flash-attn IS installed on this build, so a declared "
            "flash_attention_2 must be HONOURED, not refused"
        )
        return

    # OUTCOME, two separable ways the broken tree diverges: it returns
    # something other than 96 (it trains on under the silent fallback and
    # exits 0, or scores the rejected declaration as a failed run and exits
    # 5), and it emits no refusal marker naming the knob.
    assert rc == _EXIT_REFUSE, (
        f"expected REFUSE ({_EXIT_REFUSE}) for a declared attention kernel "
        f"this build cannot provide; got {rc}. RED (5) here would report a "
        "failed run where there was a rejected request; 0 would mean the run "
        "trained under some other kernel while claiming this one"
    )
    assert "fs:train:refuse" in out and "attn_implementation" in out, (
        "the refusal must be EVIDENT: stdout names neither the refuse marker "
        "nor the knob, so a log reader could not tell this run was refused "
        "on the declared attention implementation"
    )


def test_silently_overridden_attn_implementation_refuses_96(
    tiny_run: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accepted by the call, different on the model: still a REFUSE (96).

    This is the branch the other two arms cannot reach. A loader that raises is
    caught by the arm above; a loader that honours the value is measured by the
    arm below. The dangerous case is the one in between -- the call returns
    normally and the loaded model reports something else, which is exactly what
    a silent fallback looks like from the outside. Nothing about "no exception
    was raised" proves the kernel was honoured, so the post-load reading is
    what decides, and disagreement is refused rather than trained through.

    The double here strips the kwarg and calls the REAL loader, so the returned
    model is a genuine transformers model whose config carries the genuine
    default -- not a stub asserting its own answer (#252: the double must not
    be narrower than the thing it stands in for).
    """
    if tiny_run is None:
        return
    import transformers

    real_loader = transformers.AutoModelForCausalLM.from_pretrained

    def _dropping_loader(model_ref: str, **kwargs: Any) -> Any:
        kwargs.pop("attn_implementation", None)
        return real_loader(model_ref, **kwargs)

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_dropping_loader),
    )

    cfg = tiny_run.make_config("attn-overridden", attn_implementation="eager", max_steps=2)
    rc = loop.train(cfg)
    out = capsys.readouterr().out

    # REACHED-SITE: the double must actually produce a DIFFERENT reading, or
    # this arm measures nothing. Taken from the same loader, undeclared.
    default_impl = getattr(_dropping_loader(cfg.model).config, "_attn_implementation", None)
    assert default_impl not in (None, "eager"), (
        "REACHED-SITE fails: with the declaration stripped this build reports "
        f"{default_impl!r}, so a dropped 'eager' is indistinguishable from an "
        "honoured one and there is nothing here to refuse"
    )

    assert rc == _EXIT_REFUSE, (
        f"expected REFUSE ({_EXIT_REFUSE}) when the loaded model reports "
        f"{default_impl!r} after 'eager' was declared; got {rc}. A 0 here is "
        "the silent fallback itself: trained under one kernel, recorded as "
        "having been asked for another"
    )
    assert "fs:train:refuse" in out and "attn_implementation" in out, (
        "the refusal must be EVIDENT: stdout names neither the refuse marker nor the knob"
    )


def test_declared_scheduler_and_warmup_move_the_learning_rate(
    tiny_run: Any, capture_trainer: Any
) -> None:
    """Declare cosine + warmup_steps; the OPTIMIZER's LR must trace them.

    Cosine-with-warmup on the declared numbers gives, to three useful decimal
    places: step-begin 1 -> 0.200 of the base (inside warmup), step-begin 6 ->
    0.989 (first step past the 5-step warmup, at the schedule's peak). A tree
    that accepted both knobs and dropped them runs the engine default
    (linear, zero warmup): 0.950 and 0.700 at those same steps. Every
    assertion below therefore has a broken-tree arm that comes out DIFFERENT,
    including the trivial one -- a schedule that silently stayed flat comes
    out RED on the inequality alone, which is the floor this axis guarantees.
    """
    if tiny_run is None:
        return
    import transformers

    lrs: dict[int, float] = {}

    class _LearningRateProbe(transformers.TrainerCallback):
        # At on_step_begin the Trainer has stepped the scheduler exactly
        # global_step times and not once more, so the value read here is the
        # schedule evaluated AT global_step -- not sampled a step early or
        # late, and not reconstructed from the log stream, which the cadence
        # bound (max(1, min(10, max_steps))) would not make observable at
        # step 1 anyway.
        def on_step_begin(
            self, args: Any, state: Any, control: Any, optimizer: Any = None, **kw: Any
        ) -> Any:
            if optimizer is not None:
                lrs[int(state.global_step)] = float(optimizer.param_groups[0]["lr"])
            return control

    held = capture_trainer(callback=_LearningRateProbe())
    cfg = tiny_run.make_config(
        "scheduler",
        lr_scheduler_type="cosine",
        warmup_steps=_WARMUP_STEPS,
        learning_rate=_BASE_LR,
        max_steps=_SCHEDULER_STEPS,
    )
    rc = loop.train(cfg)

    # REACHED-SITE: a Trainer was built, the run was not refused (cosine and
    # warmup_steps are unremarkable to the pinned transformers), and BOTH
    # observation steps exist in the record. Step 1 stays inside the declared
    # warmup; step 6 is clear of it and -- with logging_steps=10 at
    # max_steps=20 -- clear of the objective gate's first-log dispatch at
    # step 10 too, so nothing can have stopped the run before the late sample.
    assert held, "REACHED-SITE fails: no Trainer was ever constructed"
    assert rc != _EXIT_REFUSE, (
        f"REACHED-SITE fails: the run was REFUSED ({rc}); on the pinned train "
        "extra, cosine + warmup_steps are accepted declarations"
    )
    missing = [s for s in (_EARLY_STEP, _LATE_STEP, *_CURVE_STEPS) if s not in lrs]
    assert not missing, (
        "REACHED-SITE fails: probe record covers steps "
        f"{sorted(lrs)}; without {missing} every comparison below is vacuous"
    )

    # OUTCOME 1 (the floor): the rate at step 1 differs from a later step's.
    # A schedule that silently stayed flat -- knobs dropped and the engine
    # default itself somehow constant -- comes out RED here.
    assert lrs[_EARLY_STEP] != lrs[_LATE_STEP], (
        f"the scheduler never moved the learning rate: {lrs[_EARLY_STEP]} at "
        f"step {_EARLY_STEP} equals {lrs[_LATE_STEP]} at step {_LATE_STEP}"
    )
    # OUTCOME 2 (warmup declaration honoured): inside the warmup the cosine
    # step-1 reading is 1/5 of base. Strictly below half of base separates it
    # from the dropped-declaration engine default (linear, no warmup: 0.950
    # of base at step 1).
    assert lrs[_EARLY_STEP] < 0.5 * _BASE_LR, (
        f"step {_EARLY_STEP} LR is {lrs[_EARLY_STEP]:.3e} -- at or above "
        f"0.5x the declared base {_BASE_LR}. Cosine-with-warmup reads "
        f"~0.200x here; the engine default reads ~0.950x, which is where a "
        "silently dropped warmup declaration lands"
    )
    # OUTCOME 3 (BOTH declarations dropped): the step-6 reading sits near the
    # cosine peak (~0.989x base), where the bare engine default -- linear with
    # NO warmup -- reads ~0.700x. This leg is honest about what it separates:
    # the pair, not the family. Forward `warmup_steps` alone and linear reads
    # 0.933x here, which clears 0.85x; that hole is what OUTCOME 4 closes, and
    # this assertion is kept because the both-dropped arm is a real broken
    # tree and this is the cheapest reading that catches it.
    assert lrs[_LATE_STEP] > 0.85 * _BASE_LR, (
        f"step {_LATE_STEP} LR is {lrs[_LATE_STEP]:.3e} -- below 0.85x the "
        f"declared base {_BASE_LR}. Declared cosine reads ~0.989x here; "
        "linear-without-warmup reads ~0.700x, which is where a tree that "
        "dropped BOTH the family and the warmup declaration lands"
    )
    # OUTCOME 4 (scheduler FAMILY honoured, independent of warmup): shape, not
    # level. HF's linear lambda is affine in the step, so its second difference
    # over three consecutive post-warmup steps is exactly 0 -- and so is a flat
    # schedule's. Cosine is strictly concave across the first half of its decay
    # and reads about -0.020x base here. Dropping `lr_scheduler_type` while
    # `warmup_steps` still reaches TrainingArguments moves this from -0.020x to
    # 0.000x, which no threshold on a single reading can see.
    early, mid, late = (lrs[s] for s in _CURVE_STEPS)
    curvature = early - 2.0 * mid + late
    assert curvature < _CURVATURE_CEILING * _BASE_LR, (
        f"second difference over steps {list(_CURVE_STEPS)} is "
        f"{curvature / _BASE_LR:+.4f}x base -- at or above "
        f"{_CURVATURE_CEILING:+.4f}x. Declared cosine reads -0.0200x; a "
        "dropped lr_scheduler_type reads 0.0000x, because linear (and flat) "
        "have no curvature at all. Readings: "
        f"{[f'{lrs[s] / _BASE_LR:.4f}x' for s in _CURVE_STEPS]}"
    )
