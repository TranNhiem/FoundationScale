"""Pinned telemetry arms for ``foundationscale.train.loop.train()``.

The lines under test are the ones a CPU box can never reach on its own (the CUDA
peak-memory reads) and the small filters in the per-step loss reader that decide
whether a number the trainer logged is a measurement or noise. Every test here
runs the real ``train()`` against the same on-disk tiny model/tokenizer pair and
``text``-column dataset the train execution suite already uses; only the CUDA
probe surface and the one trainer hook that feeds log_history are doubled, and
each doubled input names the branch it is meant to reach.

No skips: the train extra is installed in CI precisely so this module executes,
and torch.cuda is monkeypatched before ``train()`` can touch it.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from foundationscale.train.loop import (
    EXIT_PASS,
    MANIFEST_NAME,
    TrainConfig,
    train,
)

# Same single-node profile shape as tests/test_train_entry.py: the declared 1x1
# topology satisfies it exactly, so no prologue finding can stand in for the arm
# a given test is trying to reach.
PROFILE_DATA: dict = {
    "name": "reference-single-node",
    "scheduler": "slurm",
    "partitions": ["batch"],
    "node_pattern": r"compute-0[1-8]",
    "gpus_per_node": 1,
    "nccl_socket_ifname": "eth0",
    "ib_hca_pattern": "mlx5_*",
    "mnnvl_available": False,
    "container_runtime": "none",
    "container_image": "",
    "filesystem_roots": ["/tmp"],
    "max_nodes": 8,
}

# vocab_size is 256 below; the tokenizer fixture emits ids 0..255 only, so an
# embedding index out of range cannot disguise a fixture defect as a loop defect.
_SPECIAL_TOKENS = ("<unk>", "<pad>", "<eos>")
_WORD_TOKENS = tuple(f"w{i}" for i in range(253))


def _function_like(original: Any, body: Callable[..., Any]) -> Any:
    """A probe double with the original's surface, per the #83/#111 fixture rule.

    ``functools.wraps`` carries ``__wrapped__`` and the introspection surface
    torch reaches for; an ``lru_cache`` original gets an ``lru_cache`` double so
    ``cache_info``/``cache_clear`` answer the same way.
    """

    stub: Any = functools.wraps(original)(body)
    if hasattr(original, "cache_info"):
        stub = functools.lru_cache(maxsize=None)(stub)
    return stub


@pytest.fixture(scope="module")
def _offline_module() -> None:
    """Offline flags for the module-scoped model build, which predates any
    function-scoped fixture's monkeypatching.

    ``from_config`` touches no network, but pinning the flags here too means
    the fact is guarded twice -- an env fixture is cheap, an accidental hub
    call in CI is a hang measured in minutes.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.setenv("TRANSFORMERS_OFFLINE", "1")
    mp.setenv("HF_DATASETS_OFFLINE", "1")
    yield
    mp.undo()


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory, _offline_module: None) -> Path:
    """A real, loadable causal LM + tokenizer on disk, built once per module.

    Constructed from a config OBJECT, never a hub id: AutoModelForCausalLM.
    from_config allocates random weights for a 2-layer Llama whose embeddings
    are 256 x 32, small enough that one training step on CPU takes well under
    a second. The tokenizer is a WordLevel model over the same 256-token vocab,
    wrapped in PreTrainedTokenizerFast so ``save_pretrained`` produces a plain
    tokenizer.json that AutoTokenizer.from_pretrained reads offline. Both live
    in the SAME directory because that is the layout train() assumes when it
    resolves cfg.model.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

    model_dir = tmp_path_factory.mktemp("tiny-model")

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(str(model_dir))

    vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + _WORD_TOKENS)}
    wordlevel = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    wordlevel.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=wordlevel,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    fast.save_pretrained(str(model_dir))

    return model_dir


@pytest.fixture
def profile_path(tmp_path: Path) -> Path:
    path = tmp_path / "cluster_profile.json"
    path.write_text(json.dumps(PROFILE_DATA), encoding="utf-8")
    return path


@pytest.fixture
def text_dataset(tmp_path: Path) -> Path:
    """16 rows with a "text" column of whitespace-joined in-vocab words.

    WordLevel with a Whitespace pre-tokenizer splits on spaces, so the text
    must be written pre-segmented -- feeding natural sentences would map every
    word to <unk> and the fixture would silently stop measuring the model it
    claims to train.
    """
    path = tmp_path / "train.jsonl"
    rows = [
        {"text": f"w{(i * 7) % 253} w{(i * 7 + 1) % 253} w{(i * 7 + 2) % 253} w{(i * 7 + 3) % 253}"}
        for i in range(16)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _make_train_config(
    out_dir: Path,
    profile_file: Path,
    model_dir: Path,
    dataset_file: Path,
) -> TrainConfig:
    """Construct the config directly, naming only fields TrainConfig declares.

    The candidate-spelling helper this replaces could not fail when the schema
    changed: its "required field missing" check excluded keyword-only
    parameters, so it reported success while omitting every required argument.
    Required fields are named here, one by one, so a renamed or dropped field
    is a TypeError at construction instead of a silently different run.
    """
    cfg = TrainConfig(
        model=str(model_dir),
        dataset=str(dataset_file),
        output_dir=out_dir,
        nodes=1,
        gpus_per_node=1,
        profile_path=profile_file,
        max_steps=3,
        logging_steps=1,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        precision="fp32",
        save_interval=2,
        seed=17,
        dry_run=False,
        # Declared, because these configs drive a REAL trainer and the objective
        # gate refuses an undeclared run at its first observed step. Omitting it
        # here would make every telemetry test red for a reason that has nothing
        # to do with what the test is pinning.
        objective="sft",
    )
    assert Path(cfg.output_dir) == out_dir, (
        "wanted TrainConfig to bind the per-run output_dir the test created, "
        f"got {cfg.output_dir!r} when {str(out_dir)!r} was passed"
    )
    return cfg


def _patch_cuda_surface(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    allocated: int = 0,
    reserved: int = 0,
) -> list[str]:
    """Double only the surfaces train() reads; never leave them real.

    The list records every call in order so the peak test can assert the reset
    happened before the reads, which is the ordering the source declares.

    The availability lie is confined to loop's OWN probe,
    ``loop._cuda_availability`` -- NOT to ``torch.cuda.is_available``. Telling
    the whole of torch that CUDA exists is what a CPU-only wheel cannot
    survive: ``transformers.Trainer`` reads the same flag and moves the model
    to ``cuda``, which raises ``AssertionError('Torch not compiled with CUDA
    enabled')`` and adjudicates the run RED before any telemetry is written.
    The probe IS the input the claim conditions on ("when the availability
    probe reports CUDA, reset precedes both peak reads"), so doubling it is
    the narrowest instrument that reaches the branch, and the CPU arm passes
    ``available=False`` with the same UNMEASURED reason the real probe returns
    on this machine -- the two arms then differ on exactly one input.

    The three peak functions are doubled on ``torch.cuda`` itself in BOTH
    arms: on the CPU arm they must never be called, and a recording double is
    what makes an accidental real read visible instead of silent.
    """

    import torch

    from foundationscale.train import loop as loop_module

    calls: list[str] = []

    def _availability(_torch_module: Any) -> tuple[bool, str]:
        calls.append("is_available")
        if available:
            return True, ""
        return False, (
            "UNMEASURED: torch.cuda.is_available() is False, so this run has "
            "no CUDA peak-memory counter to read"
        )

    def _reset() -> None:
        calls.append("reset_peak_memory_stats")

    def _max_allocated() -> int:
        calls.append("max_memory_allocated")
        return allocated

    def _max_reserved() -> int:
        calls.append("max_memory_reserved")
        return reserved

    monkeypatch.setattr(
        loop_module,
        "_cuda_availability",
        _function_like(loop_module._cuda_availability, _availability),
        raising=True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        _function_like(torch.cuda.reset_peak_memory_stats, _reset),
        raising=True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        _function_like(torch.cuda.max_memory_allocated, _max_allocated),
        raising=True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        _function_like(torch.cuda.max_memory_reserved, _max_reserved),
        raising=True,
    )
    return calls


def _wrap_trainer_train(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Any], Any] | None,
) -> None:
    """Run the real trainer, then adjust the train-time surface _telemetry reads.

    Mutating after the real ``train()`` call keeps optimization honest while the
    fixture controls the exact instrument the loss-curve reader consults next.
    """

    import transformers

    original = transformers.Trainer.train

    def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original(self, *args, **kwargs)
        if mutate is not None:
            mutate(self)
        return output

    monkeypatch.setattr(transformers.Trainer, "train", _patched, raising=True)


def _run_one(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
    *,
    cuda: tuple[bool, int, int] | None,
    mutate_trainer: Callable[[Any], Any] | None,
) -> tuple[int, Path, list[str]]:
    """Run the real ``train()`` once and return its exit, out_dir and call log.

    The recorder is returned BY IDENTITY rather than copied into a local list:
    the doubles append to the list ``_patch_cuda_surface`` created, so a
    ``calls.extend(...)`` into a second list snapshots it while still empty and
    every later append lands on the object the test cannot see. The ordering
    assertion then reads ``[]`` forever -- a test that can only fail by raising
    on an empty list, never by observing a wrong order.
    """
    run_dir = tmp_path_factory.mktemp("telemetry-run")
    out_dir = run_dir / "out"
    if cuda is not None:
        available, allocated, reserved = cuda
        calls = _patch_cuda_surface(
            monkeypatch,
            available=available,
            allocated=allocated,
            reserved=reserved,
        )
    else:
        # Force the genuine CPU shape: the probe says no, and any accidental
        # real peak read would have to get past a double that records it.
        calls = _patch_cuda_surface(monkeypatch, available=False)
    _wrap_trainer_train(monkeypatch, mutate_trainer)
    cfg = _make_train_config(out_dir, profile_path, tiny_model_dir, text_dataset)
    code = train(cfg)
    return code, out_dir, calls


def _walk_json(node: Any, wanted: str) -> Iterator[Any]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == wanted:
                yield value
            yield from _walk_json(value, wanted)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_json(item, wanted)


def _manifest_entry(out_dir: Path, key: str) -> dict[str, Any]:
    """The telemetry ENTRY the manifest recorded for ``key``.

    Each entry is ``{"key", "value", "source", "unit"}`` -- the value and the
    provenance of that value travel together, which is the whole point of the
    section. The two are read separately below rather than flattened, so a
    test that wants "4096, measured" cannot be satisfied by "4096" alone.
    """

    path = out_dir / MANIFEST_NAME
    found = list(_walk_json(json.loads(path.read_text(encoding="utf-8")), key))
    assert found, f"wanted manifest key {key!r} to be present under {path}, got none"
    entry = found[0]
    assert isinstance(entry, dict) and "value" in entry and "source" in entry, (
        f"wanted the {key!r} telemetry entry to carry a value and its source, got {entry!r}"
    )
    return entry


def _assert_no_peak_counter_was_read(calls: list[str], arm: str) -> None:
    """The CPU arm's control: the probe IS consulted and NO counter is read.

    Without this the sentence "any accidental real peak read would have to get
    past a double that records it" is an unverified claim in a fixture
    docstring -- the doubles record every call precisely so it can be measured.
    Two things are asserted, because either alone can pass for the wrong
    reason: ``is_available`` must appear (an arm that never probed did not
    reach the decision the CUDA arm pins, so an empty log would be a vacuous
    pass), and none of the three peak functions may appear at all.
    """

    assert "is_available" in calls, (
        f"wanted the {arm} arm to consult the availability probe at least once -- "
        f"an arm that never probed never reached the branch under test; got {calls}"
    )
    read = [call for call in calls if call != "is_available"]
    assert read == [], (
        f"wanted the {arm} arm to read NO CUDA peak counter, because a CPU-only "
        f"box has none to read; got {read} inside call log {calls}"
    )


def test_cuda_peaks_are_measured_when_the_probe_reports_cuda(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
) -> None:
    """CUDA-reported machine: reset precedes reads, reads land as measured."""

    code, out_dir, calls = _run_one(
        tmp_path_factory,
        monkeypatch,
        profile_path,
        tiny_model_dir,
        text_dataset,
        cuda=(True, 4096, 8192),
        mutate_trainer=None,
    )
    assert code == EXIT_PASS, f"wanted a GREEN run on the doubled-CUDA plane, got exit {code}"
    reset_at = calls.index("reset_peak_memory_stats")
    alloc_at = calls.index("max_memory_allocated")
    res_at = calls.index("max_memory_reserved")
    assert reset_at < alloc_at and reset_at < res_at, (
        "wanted reset_peak_memory_stats before both peak reads so the figure prices "
        f"training, got call order {calls}"
    )
    allocated = _manifest_entry(out_dir, "peak_memory_allocated_bytes")
    reserved = _manifest_entry(out_dir, "peak_memory_reserved_bytes")
    # Value and source are asserted SEPARATELY and by equality: a substring
    # test over the rendered entry would also be satisfied by the key name,
    # and `4096 in entry` over a dict tests its KEYS, which never match.
    assert (allocated["value"], allocated["source"]) == (4096, "measured"), (
        "wanted peak_memory_allocated_bytes to carry the doubled counter 4096 as "
        f"measured, got {allocated!r}"
    )
    assert (reserved["value"], reserved["source"]) == (8192, "measured"), (
        "wanted peak_memory_reserved_bytes to carry the doubled counter 8192 as "
        f"measured, got {reserved!r}"
    )


def test_missing_metrics_mapping_marks_each_metric_unmeasured_not_zero(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
) -> None:
    """A TrainOutput whose .metrics is not a dict must mint no numbers."""

    import transformers

    original = transformers.Trainer.train

    def _train_then_blank(self: Any, *args: Any, **kwargs: Any) -> Any:
        # TrainOutput is a NamedTuple, so it cannot be mutated in place --
        # even object.__setattr__ raises "can't set attribute". The metrics
        # hole is therefore produced by RETURNING a replacement, which is
        # also the shape a real trainer variant would take: train() reads
        # only the returned object, never an attribute hung on the Trainer.
        output = original(self, *args, **kwargs)
        assert hasattr(output, "_replace"), (
            "wanted Trainer.train() to return a NamedTuple whose metrics field "
            f"can be replaced, got {type(output).__name__} with no _replace"
        )
        replaced = output._replace(metrics=None)
        assert replaced.metrics is None, (
            f"wanted the fixture to clear TrainOutput.metrics, got {replaced.metrics!r}"
        )
        return replaced

    monkeypatch.setattr(transformers.Trainer, "train", _train_then_blank, raising=True)
    code, out_dir, calls = _run_one(
        tmp_path_factory,
        monkeypatch,
        profile_path,
        tiny_model_dir,
        text_dataset,
        cuda=None,
        mutate_trainer=None,
    )
    assert code == EXIT_PASS, f"wanted unread metrics to stay unscored inside GREEN, got {code}"
    _assert_no_peak_counter_was_read(calls, "missing-metrics")
    runtime = _manifest_entry(out_dir, "train_runtime_s")
    loss = _manifest_entry(out_dir, "train_loss")
    assert runtime["source"] == "unmeasured" and "no metrics mapping" in str(runtime["value"]), (
        "wanted train_runtime_s to name the missing metrics mapping as the instrument, "
        f"got {runtime!r}"
    )
    assert loss["source"] == "unmeasured" and loss["value"] != 0, (
        f"wanted train_loss present-but-unmeasured rather than a minted zero, got {loss!r}"
    )


def test_absent_trainer_state_takes_the_early_loss_curve_return(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
) -> None:
    """trainer.state None reaches the first guarded return of _loss_curve."""

    def _drop_state(trainer_self: Any) -> None:
        object.__setattr__(trainer_self, "state", None)

    code, out_dir, calls = _run_one(
        tmp_path_factory,
        monkeypatch,
        profile_path,
        tiny_model_dir,
        text_dataset,
        cuda=None,
        mutate_trainer=_drop_state,
    )
    assert code == EXIT_PASS, f"wanted a stateless trainer to stay GREEN, got exit {code}"
    _assert_no_peak_counter_was_read(calls, "absent-state")
    curve = _manifest_entry(out_dir, "train_loss_curve")
    assert curve["source"] == "unmeasured" and "no state attribute" in str(curve["value"]), (
        f"wanted the loss curve entry to name the absent trainer.state instrument, got {curve!r}"
    )


def test_loss_curve_filters_stray_bool_nonnumeric_and_fractional_rows(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
) -> None:
    """Each `continue` is fed by one hostile row; the curve keeps only step 1.

    Row meanings, in the exact order _loss_curve reads them:
      * "stray-log-row" -> non-dict entry, `if not isinstance(entry, dict): continue`.
      * bool loss row -> bool flag masquerading as an int (bool subclasses int).
      * string loss row -> a label, not a numeric measurement (`(int, float)` check).
      * step 2.5 row -> a float step that is not an integer logging step.
    The bool row is LAST on purpose: without the bool filter it would overwrite
    step 1 with loss 1.0, so the expected 0.25 can fail.
    """

    hostile_history: list[Any] = [
        {"loss": 0.75, "step": 1},  # the one intact measurement the filter must keep
        "stray-log-row",  # reaches the non-dict continue
        {"loss": "loss-is-a-label-here", "step": 3},  # reaches the non-numeric continue
        {"loss": 0.5, "step": 2.5},  # reaches the non-integer float-step continue
        {"loss": True, "step": 1},  # reaches the bool-filter continue, ordered last
    ]

    def _seed_history(trainer_self: Any) -> None:
        trainer_self.state.log_history = list(hostile_history)

    code, out_dir, calls = _run_one(
        tmp_path_factory,
        monkeypatch,
        profile_path,
        tiny_model_dir,
        text_dataset,
        cuda=None,
        mutate_trainer=_seed_history,
    )
    assert code == EXIT_PASS, (
        f"wanted hostile-but-filterable log rows to stay inside a GREEN run, got exit {code}"
    )
    _assert_no_peak_counter_was_read(calls, "hostile-rows")
    curve = _manifest_entry(out_dir, "train_loss_curve")
    assert (curve["value"], curve["source"]) == ([[1, 0.75]], "measured"), (
        "wanted only the intact step-1 measurement to survive all four continue "
        f"branches, recorded as measured, got {curve!r}"
    )


def test_duplicate_step_keeps_the_later_record_for_a_resumed_run(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
) -> None:
    """Assignment-over-append: a resumed run logs a step twice, later wins."""

    def _seed_duplicate(trainer_self: Any) -> None:
        trainer_self.state.log_history = [
            {"loss": 0.9, "step": 1},
            {"loss": 0.9, "step": 2},
            {"loss": 0.1, "step": 1},  # resumed rerun of step 1: the run ended here
        ]

    code, out_dir, calls = _run_one(
        tmp_path_factory,
        monkeypatch,
        profile_path,
        tiny_model_dir,
        text_dataset,
        cuda=None,
        mutate_trainer=_seed_duplicate,
    )
    assert code == EXIT_PASS, f"wanted a resumed-history run to stay GREEN, got exit {code}"
    _assert_no_peak_counter_was_read(calls, "duplicate-step")
    curve = _manifest_entry(out_dir, "train_loss_curve")
    assert (curve["value"], curve["source"]) == ([[1, 0.1], [2, 0.9]], "measured"), (
        f"wanted the later step-1 record to replace the earlier one, got {curve!r}"
    )


def test_unreadable_history_and_nonlist_history_are_named_unmeasured(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
) -> None:
    """A non-list log_history reaches the guarded return before any loop runs.

    The paired two-run comparison for a history that RAISES lives in the next
    test; this one pins the cheap guard (tuple instead of list) so its reason
    string cannot be confused with the exception arm.
    """

    def _tuple_history(trainer_self: Any) -> None:
        trainer_self.state.log_history = ({"loss": 0.5, "step": 1},)

    code, out_dir, calls = _run_one(
        tmp_path_factory,
        monkeypatch,
        profile_path,
        tiny_model_dir,
        text_dataset,
        cuda=None,
        mutate_trainer=_tuple_history,
    )
    assert code == EXIT_PASS, f"wanted a non-list history to stay GREEN, got exit {code}"
    _assert_no_peak_counter_was_read(calls, "non-list-history")
    curve = _manifest_entry(out_dir, "train_loss_curve")
    assert curve["source"] == "unmeasured" and "no log_history list" in str(curve["value"]), (
        f"wanted the non-list guard to name log_history as the unread instrument, got {curve!r}"
    )


def test_telemetry_exception_does_not_change_the_run_verdict(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
) -> None:
    """The _loss_curve `except Exception` arm must never turn a run RED.

    A list subclass passes ``isinstance(history, list)`` and then raises inside
    ``for entry in history``: exactly the shape that reaches the telemetry
    ``except`` arm rather than the non-list guard. The claim under test is a
    comparison, so the only verdict assertion is control-vs-doubled from two
    real runs.
    """

    class _ExplodingHistory(list):
        def __iter__(self) -> Iterator[Any]:
            raise RuntimeError("log_history iterator detonated by the fixture")

    control_code, _, control_calls = _run_one(
        tmp_path_factory,
        monkeypatch,
        profile_path,
        tiny_model_dir,
        text_dataset,
        cuda=None,
        mutate_trainer=None,
    )

    # Fresh monkeypatch scope: the first run's patches are already undone.
    def _explode_history(trainer_self: Any) -> None:
        trainer_self.state.log_history = _ExplodingHistory()

    doubled_code, _, doubled_calls = _run_one(
        tmp_path_factory,
        monkeypatch,
        profile_path,
        tiny_model_dir,
        text_dataset,
        cuda=None,
        mutate_trainer=_explode_history,
    )
    _assert_no_peak_counter_was_read(control_calls, "exploding-history control")
    _assert_no_peak_counter_was_read(doubled_calls, "exploding-history doubled")
    assert doubled_code == control_code, (
        "wanted a telemetry failure to leave the verdict unchanged from the same "
        f"run with working telemetry, got control exit {control_code} vs "
        f"telemetry-failure exit {doubled_code}"
    )
