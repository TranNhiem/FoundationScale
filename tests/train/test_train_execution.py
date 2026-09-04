"""Execution tests for ``foundationscale.train.loop.train()``.

Everything in ``tests/test_train_entry.py`` deliberately stops BEFORE the
optional-dependency import block; this module exists to cover the lines after
it -- model/tokenizer/dataset construction, Trainer execution, the final
safetensors save, and the save-gate adjudication. It runs the REAL code path
on CPU against a tiny causal LM and tokenizer built in-process, so it needs
no network and no hub model id.

Defect class pinned overall: the "green board over code that never ran"
class -- 263 lines that shipped unexercised for a release. Each test below
also pins a specific refusal/RED arm so a regression in one of them cannot
masquerade as the happy path.

No skips anywhere: this module runs under FS_FORBID_SKIPS=1, where the train
extra is installed exactly so these tests execute.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from foundationscale.train.loop import (
    EXIT_PASS,
    EXIT_RED,
    EXIT_REFUSE,
    EXIT_UNMEASURED,
    TrainConfig,
    train,
)

# Same shape of profile fixture as tests/test_train_entry.py: a single-node
# reference profile the declared 1x1 topology satisfies exactly, so the
# prologue emits no blocking findings and execution reaches the deps import.
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

# The model config below declares vocab_size=256, so the tokenizer must emit
# ids strictly below 256 or every training step would index the embedding out
# of range and the failure would look like a training defect when it was a
# fixture defect. 3 special tokens + 253 word tokens == 256 exactly.
_SPECIAL_TOKENS = ("<unk>", "<pad>", "<eos>")
_WORD_TOKENS = tuple(f"w{i}" for i in range(253))


# Whatever `_offline_and_cpu_env` displaced, recorded at patch time so the
# control below can compare the double against the real thing. It cannot read
# the original itself: the autouse fixture has already run by the time any test
# body executes, so `torch.backends.mps.is_available` is the double there.
_ORIGINAL_MPS_PROBES: dict[str, Any] = {}


def _false_probe_like(original: Any) -> Any:
    """A zero-arg ``False`` stub that is not NARROWER than the probe it replaces.

    A test double is a claim that it can stand in for the original, and an
    attribute the original has and the double lacks falsifies that claim
    silently. `lambda: False` looked adequate because the only thing this
    module calls the probe FOR is its return value -- but torch does not only
    call these probes, it INTROSPECTS them. `torch/_dynamo/variables/torch.py`
    reads ``torch.backends.mps.is_available.__wrapped__`` at import time while
    building its constant-fold table, and the real `is_available` is an
    `_lru_cache_wrapper` (which has `__wrapped__`) while a bare lambda is a
    plain function (which does not) -- narrower by six attributes.

    The consequence was a verdict that depended on the machine, the #83/#111
    class this framework rejects. The first import of `torch._dynamo` is lazy
    and happens from deep inside `transformers.modeling_utils`; if it lands
    inside the patched window it hits the double, raises AttributeError, and
    transformers' LazyModule converts that into
    ``ModuleNotFoundError: Could not import module 'is_kubeflow_available'`` --
    a message naming neither MPS, nor dynamo, nor this fixture. Locally, torch
    2.13 did not read the attribute and the suite was green; CI resolved torch
    2.14, which does, and three matrix jobs went red on a test that had passed
    on the author's machine minutes earlier. The latency was the fixture's, not
    torch's: the double was already narrower when it was written.

    So the double is built the way the original was, rather than by naming
    `__wrapped__` as a special case -- an allowlist of one attribute would go
    stale the next time torch introspects a different one. `cache_info` is the
    discriminator because that is what distinguishes the two shapes actually
    present here (`is_available` is cached, `is_built` is not), and wrapping
    the lambda means everything reachable through `__wrapped__` still answers
    False. `test_cpu_pin_double_is_not_narrower_than_the_probe` measures the
    non-narrowing rather than assuming it.
    """
    stub: Any = lambda: False  # noqa: E731 -- must be a plain function, not a def, to match shape
    if hasattr(original, "cache_info"):
        stub = functools.lru_cache(maxsize=None)(stub)
    return stub


@pytest.fixture(autouse=True)
def _offline_and_cpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee offline-CPU execution regardless of the runner it is on.

    The offline variables are the contract this module runs under: nothing in
    it may contact a hub. CUDA_VISIBLE_DEVICES is blanked so a CI runner that
    happens to have a GPU still trains on CPU -- otherwise the same test would
    exercise a different device depending on the machine, which is the class
    of environment-dependent verdict this framework exists to reject. The
    torchrun/scheduler variables are removed so no leaked runtime evidence
    steers the declared-vs-effective comparison in the prologue.

    Blanking CUDA_VISIBLE_DEVICES covers NVIDIA and nothing else, and a claim
    that is narrower than its wording is the defect, not a rounding error: on
    an Apple-silicon developer machine `TrainingArguments` otherwise picks MPS
    and the same test then exercises a different backend than CI's CPU. The
    env-var route does not close it -- ACCELERATE_USE_CPU=1 moves the batch to
    CPU and leaves the model on MPS, which fails with "Placeholder storage has
    not been allocated on MPS device!" rather than falling back. Device
    selection in both transformers and accelerate reads torch.backends.mps, so
    that is patched directly; `raising=False` because the module is absent on
    some builds and a fixture that explodes on a Linux runner would trade one
    machine-dependent outcome for another.

    What this does NOT cover, stated rather than left implied. Memory PINNING
    is decided separately, from `torch.accelerator.current_accelerator()` -- a
    C-level probe no Python patch reaches -- so on an Apple-silicon host the
    DataLoader still emits "'pin_memory' ... not supported on MPS now" and
    disables pinning. That is a host-allocation optimisation, not a compute
    placement: the model and the batch are on CPU either way, and
    `test_fixture_pins_execution_to_cpu` asserts that rather than assuming it.
    Accelerators other than CUDA and MPS (xpu, hpu) are likewise not excluded;
    no runner in this project has one, so such a control would itself be
    unexercised here, and an unexercised control is worse than a stated gap.
    """
    import torch

    if hasattr(torch.backends, "mps"):
        for _probe in ("is_available", "is_built"):
            _original = getattr(torch.backends.mps, _probe, None)
            _ORIGINAL_MPS_PROBES[_probe] = _original
            monkeypatch.setattr(
                torch.backends.mps, _probe, _false_probe_like(_original), raising=False
            )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    for var in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_JOB_ID",
    ):
        monkeypatch.delenv(var, raising=False)


def test_fixture_pins_execution_to_cpu(tmp_path: Path) -> None:
    """Control for the autouse fixture: prove CPU placement, do not assume it.

    Without this leg a fixture that quietly stopped working would fail nothing.
    The suite would keep passing while exercising whatever backend the host
    happens to have, which is the machine-dependent-verdict class (#83, #111)
    rather than a coverage gap -- and it would be invisible, because every
    other test here asserts on artifacts, not on where they were computed.

    Two assertions, because either one alone can be laundered.
    `TrainingArguments` resolves its device through accelerate's PartialState,
    a process-wide singleton: whichever caller runs FIRST fixes the answer for
    the rest of the process, so a later CPU reading may be inherited from an
    earlier test rather than produced by the fixture. The direct probe cannot
    be masked that way, and the resolved device is what the Trainer will act
    on. Neither is redundant; they fail on different breakages.
    """
    import torch
    from transformers import TrainingArguments

    if hasattr(torch.backends, "mps"):
        assert torch.backends.mps.is_available() is False, (
            "autouse fixture did not neutralise the MPS availability probe"
        )

    resolved = TrainingArguments(output_dir=str(tmp_path / "probe")).device
    assert resolved.type == "cpu", f"expected CPU placement, resolved {resolved}"


def test_cpu_pin_double_is_not_narrower_than_the_probe() -> None:
    """Control for `_false_probe_like`: the double must not be narrower.

    MUST_PASS arm: for every probe the fixture displaced, every attribute the
    original exposes is present on the double. Stated as a set difference, not
    as a check for the one attribute that broke -- torch reads `__wrapped__`
    today and the point is that the double survives whatever it reads next.

    MUST_FIRE arm: the naive `lambda: False` this replaced IS narrower than
    `is_available`, so the assertion above is a real measurement and not a
    tautology that any object would satisfy. Without it a `_false_probe_like`
    that returned the original unchanged would also pass, and the control
    would certify nothing.

    Denominator: the probes actually patched, reported on failure. Zero
    patched probes is UNMEASURED, not a pass -- on a torch build with no
    `mps` module the fixture patches nothing and this control must say so
    rather than exit green over an empty set.
    """
    import torch

    if not hasattr(torch.backends, "mps"):
        pytest.fail(
            "UNMEASURED: torch.backends has no `mps`, so the fixture patched 0 probes and "
            "this control has an empty denominator. An empty set satisfies every claim "
            "(`all([])` is True); it is not evidence the double is adequate. Re-scope this "
            "control to whatever the fixture displaces on this build, or delete both."
        )

    assert _ORIGINAL_MPS_PROBES, (
        "UNMEASURED: no originals were recorded, so the fixture did not run or stopped "
        "recording. The comparison below would be over an empty denominator."
    )

    for name, original in _ORIGINAL_MPS_PROBES.items():
        double = getattr(torch.backends.mps, name)
        missing = sorted(set(dir(original)) - set(dir(double)))
        assert not missing, (
            f"the {name} double is NARROWER than the probe it replaces: missing {missing}. "
            f"torch introspects these at import time, so a missing attribute surfaces as an "
            f"unrelated ModuleNotFoundError several frames away. Build the double to match "
            f"the original's shape in _false_probe_like rather than excepting the attribute."
        )
        assert double() is False, f"the {name} double must report False, not {double()!r}"

    cached = _ORIGINAL_MPS_PROBES.get("is_available")
    assert cached is not None and hasattr(cached, "cache_info"), (
        "MUST_FIRE precondition gone: is_available is no longer a cached wrapper, so the "
        "narrowing this control detects can no longer be constructed. Re-derive the arm "
        "against whatever shape the probe has now -- do not delete it."
    )
    naive_gap = sorted(set(dir(cached)) - set(dir(lambda: False)))
    assert "__wrapped__" in naive_gap, (
        "MUST_FIRE did not fire: a bare `lambda: False` is no longer narrower than "
        f"is_available (gap {naive_gap}), so the MUST_PASS arm above proves nothing."
    )


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


@pytest.fixture()
def profile_path(tmp_path: Path) -> Path:
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(PROFILE_DATA), encoding="utf-8")
    return path


@pytest.fixture()
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


def _cfg(
    tmp_path: Path,
    profile_file: Path,
    *,
    model: str,
    dataset: str,
    **overrides: object,
) -> TrainConfig:
    """One real training step: the smallest config that still runs step 4-9."""
    kwargs: dict = {
        "model": model,
        "dataset": dataset,
        "output_dir": tmp_path / "out",
        "nodes": 1,
        "gpus_per_node": 1,
        "profile_path": profile_file,
        "max_steps": 1,
        "per_device_batch_size": 2,
        "save_interval": 50,
        "dry_run": False,
    }
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_train_happy_path_executes_and_saves_safetensors(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins steps 4-9: real deps import, one Trainer step, real final save,
    real save-gate adjudication over a real manifest.

    Defect class: the 263-line dead block itself -- code that shipped never
    executed. Accepts EXIT_PASS or EXIT_UNMEASURED rather than only EXIT_PASS
    because adjudication over a 2-layer random-weight checkpoint is a verdict
    of the gate registry, and the registry's membership is not this test's
    fixture to control; what IS deterministic here is that the final save
    produced safetensors shards (a precondition loop.py itself enforces
    before adjudicating) and that the return is inside the declared
    0/5/95/96 contract. The observed code is printed so the run log records
    which arm fired -- a verdict that is asserted can never be laundered
    into "probably passed".
    """
    cfg = _cfg(tmp_path, profile_path, model=str(tiny_model_dir), dataset=str(text_dataset))
    rc = train(cfg)
    out = capsys.readouterr().out

    final_dir = Path(cfg.output_dir) / "final"
    shards = sorted(final_dir.glob("*.safetensors"))
    # Step 8 refuses to adjudicate a save with zero shards (that vacuity is
    # doctrine 1), so asserting the shards is asserting evidence, not hope.
    assert shards, f"no safetensors shards under {final_dir}; train output:\n{out}"
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out
    verdict = "EXIT_PASS" if rc == EXIT_PASS else "EXIT_UNMEASURED"
    print(f"fs-test: tiny-model adjudication returned {rc} ({verdict})")
    # Markers prove the path actually ran rather than refusing early:
    # deps imported, data tokenized, trainer built, one step ran, saved.
    for marker in (
        "[fs:train:deps]",
        "[fs:train:data]",
        "[fs:train:trainer]",
        "[fs:train:run]",
        "[fs:train:saved]",
        "[fs:train:adjudicate]",
    ):
        assert marker in out, f"missing {marker}; train output:\n{out}"
    assert "Traceback" not in out


def test_train_refuses_dataset_without_text_column(
    tiny_model_dir: Path,
    profile_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins the REFUSE arm at the column check: a split without 'text' is a
    caller error, not a trainer crash.

    Defect class: a typed refusal escaping as an exception -- the column check
    exists so a misnamed column returns 96 with the columns NAMED, instead of
    dying inside tokenization as an unclassifiable RED or a traceback.
    """
    bad = tmp_path / "notext.jsonl"
    rows = [{"content": f"w{i} w{i + 1}"} for i in range(16)]
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    cfg = _cfg(tmp_path, profile_path, model=str(tiny_model_dir), dataset=str(bad))
    rc = train(cfg)
    out = capsys.readouterr().out

    assert rc == EXIT_REFUSE, out
    assert "[fs:train:refuse]" in out
    assert "text" in out  # the missing column is named in the refusal
    assert "content" in out  # and so are the columns that WERE present
    # The refusal happens before the trainer is built: no step ran, nothing saved.
    assert "[fs:train:run]" not in out
    assert "Traceback" not in out


def test_train_returns_red_when_model_dir_is_unconstructible(
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins the RED arm at model/dataset construction: an empty directory has
    no config.json, so from_pretrained raises and train() must map that to 5.

    Defect class: expected-construction-failure leaving the declared exit
    namespace -- step 5 wraps construction exactly so a missing config is
    adjudicated RED with the exception recorded, never exit 1 with a
    traceback. Using an EMPTY directory (not a random name) matters: a name
    that fails to resolve could take several different exception routes, while
    "directory exists, config absent" is a single deterministic one.
    """
    empty_model = tmp_path / "empty-model"
    empty_model.mkdir()

    cfg = _cfg(tmp_path, profile_path, model=str(empty_model), dataset=str(text_dataset))
    rc = train(cfg)
    out = capsys.readouterr().out

    assert rc == EXIT_RED, out
    assert "[fs:train:red]" in out
    assert "construction failed" in out
    assert "Traceback" not in out
    # RED here is pre-training: no checkpoint, no final save to misadjudicate.
    assert not (Path(cfg.output_dir) / "final").exists()


def test_dry_run_passes_after_full_prologue_with_zero_artifacts(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins the dry_run arm: the ENTIRE validation prologue runs, then the
    function returns EXIT_PASS before the deps import -- proof the dry-run
    contract means "zero GPUs, zero training artifacts", not "fake success".

    Defect class: the pretend run -- a dry run that writes checkpoints would
    train a caller to trust artifacts that never passed a save gate, and a
    dry run that skips the prologue would validate nothing. Asserting the
    ABSENCE of [fs:train:deps] pins the exact return point inside train().
    """
    cfg = _cfg(
        tmp_path,
        profile_path,
        model=str(tiny_model_dir),
        dataset=str(text_dataset),
        dry_run=True,
    )
    rc = train(cfg)
    out = capsys.readouterr().out

    assert rc == EXIT_PASS, out
    assert "dry-run PASS" in out
    assert "[fs:train:validated]" in out  # prologue really ran
    assert "[fs:train:manifest]" in out  # the dry_run stage manifest was written
    assert "[fs:train:deps]" not in out  # return happened BEFORE the import block
    # No checkpoint material may exist: the dry-run promise is structural,
    # not conventional, so it is asserted against the filesystem.
    final_dir = Path(cfg.output_dir) / "final"
    assert not final_dir.exists()
    assert sorted(Path(cfg.output_dir).glob("**/*.safetensors")) == []
