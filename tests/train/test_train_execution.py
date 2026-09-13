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
    _TELEMETRY_UNITS,
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
_ORIGINAL_CUDA_PROBES: dict[str, object] = {}
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
    # #372: CUDA needs the SAME direct patch, for the same reason the docstring
    # above gives for MPS -- the env-var route does not close it. Blanking
    # CUDA_VISIBLE_DEVICES only takes effect if CUDA has not yet initialised,
    # and in a full-suite run an earlier module has already touched torch, so
    # the value is cached and the blanking is a no-op.
    #
    # MEASURED: on GB200 (aarch64, torch 2.14.0+cu130, cuda True) this test
    # failed with `resolved cuda:0` while passing on every x86 GitHub runner
    # and on the developer Mac. It passed there because those hosts have NO
    # CUDA DEVICE -- the assertion was satisfied by absent hardware, not by
    # anything the fixture did. A control that can only pass where it is
    # vacuous is worse than no control: it occupies the slot a real one needs.
    if hasattr(torch, "cuda"):
        for _probe in ("is_available", "device_count"):
            _original = getattr(torch.cuda, _probe, None)
            if _original is None:
                continue
            _ORIGINAL_CUDA_PROBES[_probe] = _original
            _double = (
                _false_probe_like(_original) if _probe == "is_available" else (lambda *a, **k: 0)
            )
            monkeypatch.setattr(torch.cuda, _probe, _double, raising=False)
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

    Several assertions, because each alone can be laundered.
    `TrainingArguments` resolves its device through accelerate's PartialState,
    a process-wide singleton: whichever caller runs FIRST fixes the answer for
    the rest of the process, so a later CPU reading may be inherited from an
    earlier test rather than produced by the fixture. The direct probe cannot
    be masked that way, and the resolved device is what the Trainer will act
    on. Neither is redundant; they fail on different breakages.

    SUBSTITUTION is asserted separately from OUTCOME, and that separation is
    the whole point of the #372 re-take. `is_available() is False` and
    `device.type == "cpu"` are both satisfiable by a host that simply has no
    accelerator -- true of every x86 GitHub runner and of the developer Mac --
    so on those hosts they measure the machine, not the fixture. The
    `_ORIGINAL_*_PROBES` records are written only by the fixture's patch loops,
    so asserting they are populated fails on ANY host the moment a loop is
    deleted. That is what makes this leg non-vacuous off-GPU; the outcome
    assertions are what make it bite on-GPU.

    MEASURED on a GB200 host (aarch64, torch 2.14.0+cu130) with the instrument
    confirmed live beforehand at `cuda True, device_count 1`: this leg green
    there, while the SAME host with the fixture absent resolves `cuda` in a
    fresh process. So the green here is attributable to the fixture rather
    than to absent hardware -- which is the claim the x86 runners cannot make
    on their own, and the reason the substitution assertions above exist.
    """
    import torch
    from transformers import TrainingArguments

    if hasattr(torch.backends, "mps"):
        assert _ORIGINAL_MPS_PROBES, (
            "autouse fixture never recorded an MPS probe; its patch loop did not run"
        )
        assert torch.backends.mps.is_available() is False, (
            "autouse fixture did not neutralise the MPS availability probe"
        )

    # The CUDA arm, symmetric with MPS above. Before this, _ORIGINAL_CUDA_PROBES
    # was written by the fixture and read by nothing -- the arm added to fix #372
    # was itself uncontrolled.
    if hasattr(torch, "cuda"):
        assert _ORIGINAL_CUDA_PROBES, (
            "autouse fixture never recorded a CUDA probe; its patch loop did not run"
        )
        assert torch.cuda.is_available() is False, (
            "autouse fixture did not neutralise the CUDA availability probe"
        )
        assert torch.cuda.device_count() == 0, (
            "autouse fixture did not neutralise the CUDA device_count probe"
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
        # Declared, because these configs drive a REAL trainer and the objective
        # gate refuses an undeclared run at its first observed step. Omitting it
        # here would make every execution test red for a reason that has nothing
        # to do with what the test is pinning. The refusal arm is worth covering
        # -- it is covered deliberately, in the callback's own module, rather
        # than incidentally by every test that happens to run the loop.
        "objective": "sft",
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
    # The objective gate's ARM is deterministic here even though the save
    # gate's verdict is not: max_steps=1 binds logging_steps=1, so a loss IS
    # logged and the gate reads it. Asserting PASS rather than "the marker is
    # present" is what makes this a control for the cadence: the knob was once
    # a bare 10, under which this 1-step run emitted no training log at all and
    # the gate landed on its no-loss backstop -- UNMEASURED, on a run that was
    # perfectly healthy. A revert to a constant cadence turns this line red.
    obj = [ln for ln in out.splitlines() if "[fs:train:objective_gate]" in ln]
    assert len(obj) == 1, f"expected exactly one objective-gate line, got {obj}"
    assert "PASS" in obj[0] and "first observed step" in obj[0], obj[0]
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


# --- #400: run-manifest outcome telemetry -------------------------------------


def _read_run_manifest(output_dir: Any) -> dict[str, Any]:
    """Locate and JSON-load the run manifest ``train()`` wrote under ``output_dir``.

    ``MANIFEST_NAME`` is imported inside this helper, not at module scope: the
    module header imports only what the pre-existing tests needed, and the
    reserved basename is loop.py's constant to rename, never this suite's to
    retype. The done-stage emission overwrites the train-stage manifest in
    place (loop.py says so at the first emission), so after ``train()``
    returns this one path carries the final manifest -- telemetry included.
    """
    from foundationscale.train.loop import MANIFEST_NAME

    path = Path(output_dir) / MANIFEST_NAME
    assert path.exists(), f"no run manifest at {path}; the done-stage emission never ran"
    return json.loads(path.read_text(encoding="utf-8"))


def test_telemetry_entries_are_value_and_status_records_not_bare_scalars(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins the telemetry section's SHAPE: every entry is a two-part record.

    Without this test a reader would take "the manifest has a telemetry
    section" to mean the section carries the run's numbers. A regression that
    wrote bare scalars -- or a constant placeholder under every key --
    satisfies that weaker reading while carrying no information: a section of
    sevens has the same keys as a section of measurements. The (value,
    source) record is what lets a reader tell a measured number from a stated
    reason for its absence, so the assertion is on the record shape AND on at
    least one entry being genuinely ``measured`` -- either check alone can be
    laundered by a writer that fills both halves with filler.
    """
    cfg = _cfg(tmp_path, profile_path, model=str(tiny_model_dir), dataset=str(text_dataset))
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    telemetry = _read_run_manifest(cfg.output_dir)["telemetry"]
    assert telemetry, "the done manifest's telemetry section is empty"
    sources = set()
    for key, entry in telemetry.items():
        assert isinstance(entry, dict), (
            f"telemetry[{key!r}] is a bare {type(entry).__name__} ({entry!r}), not a "
            f"(value, source) record -- a section of scalars passes a section-exists "
            f"check while proving nothing was measured"
        )
        assert "value" in entry and "source" in entry, (
            f"telemetry[{key!r}] is not a two-part record: keys {sorted(entry)}"
        )
        sources.add(entry["source"])
    assert sources <= {"measured", "derived", "unmeasured"}, (
        f"telemetry sources {sorted(sources)} fall outside the vocabulary loop.py emits"
    )
    assert "measured" in sources, (
        "no telemetry entry is measured; on a real one-step run the trainer's own "
        "TrainOutput.metrics must measure at least train_runtime_s and train_loss"
    )


def test_train_loss_curve_is_a_measured_series_of_per_step_losses(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins ``train_loss_curve`` as a REAL series: >= 2 measured [step, loss] pairs.

    This test configures max_steps=3 with logging_steps=1 -- three optimizer
    steps with a loss logged at every one -- because the module-default
    max_steps=1 could only ever produce a one-point curve, and a one-point
    curve is the defect shape itself. The step count is raised HERE, in the
    config this test builds, rather than the assertion weakened to >= 1: a
    control that cannot fail is the defect this suite exists to refuse.

    Without this test a reader would take the presence of "train_loss_curve"
    in telemetry as evidence that a curve was recorded. On real GB200 hardware
    four verification-matrix adjudicators read a "loss curve" of length ONE --
    the aggregate train_loss wearing a series' name -- and returned GREEN on
    it. One scalar is not a curve: it cannot show that a knob moved training,
    and it cannot show that two settings agree. The aggregate is deliberately
    NOT compared against the curve's losses by float inequality, because a
    genuine coincidence is legal; the load-bearing claim is structural, that
    the series has more than one point.
    """
    cfg = _cfg(
        tmp_path,
        profile_path,
        model=str(tiny_model_dir),
        dataset=str(text_dataset),
        max_steps=3,
        logging_steps=1,
    )
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    telemetry = _read_run_manifest(cfg.output_dir)["telemetry"]
    assert "train_loss_curve" in telemetry, (
        "train_loss_curve is absent from telemetry; the per-step loss series is the "
        "evidence that a knob moved training, and its absence reads as silence"
    )
    entry = telemetry["train_loss_curve"]
    assert entry["source"] == "measured", (
        f"train_loss_curve has source {entry['source']!r} on a run configured for "
        f"three logged steps; anything else means the emitter could not read the "
        f"trainer's log history on a healthy run"
    )
    curve = entry["value"]
    assert isinstance(curve, list), (
        f"train_loss_curve carries a {type(curve).__name__}, not a list of "
        f"[step, loss] pairs; the series shape is the whole point of the entry"
    )
    assert len(curve) >= 2, (
        f"train_loss_curve has {len(curve)} point(s): a one-point curve is the "
        f"aggregate train_loss wearing a series' name -- the exact shape that "
        f"produced four false GREENs on real GB200 hardware. logging_steps and "
        f"max_steps are the knobs that determine the series length; this test "
        f"configures logging_steps=1 and max_steps=3 precisely so the curve "
        f"cannot degenerate, so a short series here means the emitter stopped "
        f"reading the trainer's log history"
    )
    steps: list[int] = []
    for point in curve:
        assert isinstance(point, list) and len(point) == 2, (
            f"curve point {point!r} is not a 2-element [step, loss] list; the "
            f"series is consumed element-by-element by adjudicators, so a "
            f"misshapen point fails downstream as an unrelated unpacking error"
        )
        step, loss = point
        assert isinstance(step, int) and not isinstance(step, bool), (
            f"curve step {step!r} is a {type(step).__name__}, not an int; bool "
            f"is an int subclass, so the exclusion is stated explicitly rather "
            f"than left to isinstance's quiet coercion"
        )
        assert isinstance(loss, int | float) and not isinstance(loss, bool), (
            f"curve loss {loss!r} is a {type(loss).__name__}, not a number; a "
            f"non-numeric loss cannot show that two settings agree"
        )
        steps.append(step)
    # strict=False, stated rather than defaulted: this is the pairwise idiom, so
    # the operands differ in length by one BY CONSTRUCTION and strict=True would
    # raise on every well-formed curve. B905 wants the choice made explicitly,
    # and the explicit choice here is the permissive one.
    assert all(earlier < later for earlier, later in zip(steps, steps[1:], strict=False)), (
        f"curve steps {steps} are not strictly ascending; a repeated or "
        f"out-of-order step means the emitter is not reading the log history in "
        f"order, and a curve with duplicate steps is not a curve over steps"
    )


def test_train_loss_curve_is_unmeasured_with_a_reason_when_no_step_is_logged(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MUST-FIRE arm for the curve's unmeasured branch: absence is SAID, not silent.

    The emitter is driven down its own unmeasured branch with the cheapest
    honest lever the loop already exposes -- no seam is added to loop.py. A
    DECLARED logging_steps=50 against max_steps=1 is honoured as declared
    (pinned in test_logging_steps_effective_is_derived_and_matches_the_resolved_cadence),
    so the single optimizer step falls between log points and the trainer's
    log_history carries no per-step loss for the emitter to read -- the same
    shape the happy-path test's comment records for the historical bare-10
    cadence, under which a 1-step run emitted no training log at all.

    Without this test a reader would believe an unreadable history surfaces
    somewhere. The vacuous shapes are an entry that is MISSING (silence reads
    as "nothing to say") and an entry present as [] with source "measured" (a
    claim that an empty series was measured); both pass a key-exists check
    while telling the adjudicator nothing, and the second is the false-GREEN
    class with a series' name attached.
    """
    cfg = _cfg(
        tmp_path,
        profile_path,
        model=str(tiny_model_dir),
        dataset=str(text_dataset),
        max_steps=1,
        logging_steps=50,
    )
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    telemetry = _read_run_manifest(cfg.output_dir)["telemetry"]
    assert "train_loss_curve" in telemetry, (
        "train_loss_curve is absent from telemetry on a run whose cadence logged "
        "no per-step loss; an unreadable history must be SAID, not omitted -- a "
        "missing key reads as silence, and silence is the vacuous pass"
    )
    entry = telemetry["train_loss_curve"]
    assert entry["source"] == "unmeasured", (
        f"train_loss_curve has source {entry['source']!r} on a run whose cadence "
        f"logged no per-step loss; stamping an unreadable series 'measured' -- "
        f"above all as an empty list -- claims an instrument reading nobody made"
    )
    reason = entry["value"]
    assert isinstance(reason, str) and reason.strip(), (
        f"train_loss_curve carries value {reason!r} on the unmeasured arm; the "
        f"arm must record a non-empty reason -- an empty list here would be the "
        f"vacuous measured pass wearing an unmeasured stamp"
    )
    assert reason.startswith("UNMEASURED: "), (
        f"the unmeasured reason {reason!r} does not begin 'UNMEASURED: '; the "
        f"prefix is what lets a reader tell a stated absence from a measured "
        f"value without parsing the sentence"
    )
    assert any(
        token in reason for token in ("logging_steps", "max_steps", "log_history", "history")
    ), (
        f"the recorded reason {reason!r} names neither the cadence knobs "
        f"(logging_steps / max_steps) nor the history that could not be read; a "
        f"reason that does not say WHAT was unavailable leaves the reader unable "
        f"to act on it"
    )


def test_telemetry_units_come_from_the_telemetry_units_table(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins unit PROVENANCE: each recorded unit is ``_TELEMETRY_UNITS[key]``.

    Without this test a reader would take the unit beside a value as
    authoritative. The emitter attaches units from one table; a second copy
    of that table typed into this test would make the test agree with itself
    -- emitter and table could drift apart and nothing would go red. So the
    test reads the imported table, and it compares KEY SETS, not only the
    keys that happen to be present: a metric added to the table but never
    emitted (or emitted but never added) changes exactly one side of the
    equality. ``train_loss`` is the deliberate exception the table's own
    comment names -- a unitless scalar, emitted with unit None -- so the
    key-set comparison runs over the entries that CARRY a unit.
    """
    cfg = _cfg(tmp_path, profile_path, model=str(tiny_model_dir), dataset=str(text_dataset))
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    telemetry = _read_run_manifest(cfg.output_dir)["telemetry"]
    assert telemetry, "the done manifest's telemetry section is empty"
    emitted_units = {key: entry.get("unit") for key, entry in telemetry.items()}
    for key, unit in emitted_units.items():
        assert unit == _TELEMETRY_UNITS.get(key), (
            f"telemetry[{key!r}] carries unit {unit!r}; the table says "
            f"{_TELEMETRY_UNITS.get(key)!r} -- the emitter stopped reading _TELEMETRY_UNITS"
        )
    carried = {key for key, unit in emitted_units.items() if unit is not None}
    assert carried == set(_TELEMETRY_UNITS), (
        f"telemetry keys carrying a unit {sorted(carried)} drifted from _TELEMETRY_UNITS "
        f"{sorted(_TELEMETRY_UNITS)}; a key added to one side and not the other is "
        f"invisible to per-key assertions"
    )


def test_peak_memory_is_recorded_unmeasured_with_a_reason_when_cuda_is_absent(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins the CPU arm of the peak-memory probe: UNMEASURED, with a reason.

    Without this test a reader of a CPU run's manifest could meet a 0 and
    read it as "training allocated no memory" -- a measurement no instrument
    made -- or meet a missing key and read it as "nothing to say". Zero is a
    claim about the world; absence is silence; the truth on this host is that
    no CUDA peak counter exists to read, and the manifest must SAY that. The
    autouse fixture pins CUDA unavailable, so both peak entries must be
    present, stamped unmeasured, and carrying a non-empty reason string.
    """
    cfg = _cfg(tmp_path, profile_path, model=str(tiny_model_dir), dataset=str(text_dataset))
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    telemetry = _read_run_manifest(cfg.output_dir)["telemetry"]
    for key in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        assert key in telemetry, (
            f"{key} is absent from telemetry; absence reads as silence, not UNMEASURED"
        )
        entry = telemetry[key]
        assert entry["source"] == "unmeasured", (
            f"{key} has source {entry['source']!r}; on a CUDA-less run the peak counter "
            f"does not exist, so any other status claims a measurement nobody made"
        )
        assert isinstance(entry["value"], str) and entry["value"].strip(), (
            f"{key} carries value {entry['value']!r}; the unmeasured arm must record a "
            f"non-empty reason -- a numeric 0 here would be a measurement, not a statement"
        )


def test_the_two_absent_cuda_branches_record_distinguishable_reasons(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins that the two no-peak-counter branches record WHICH one fired.

    Without this test a reader would assume an UNMEASURED peak entry says why.
    "This torch build exposes no torch.cuda module" and
    "torch.cuda.is_available() is False" are different facts about a machine
    with different remedies; one shared reason string -- or a branch falling
    through to the other's message -- passes every status-level check while
    leaving the reader unable to tell a build property from a runtime one.

    The no-module arm is driven through ``_cuda_availability`` rather than
    through ``train()``, and that is a measurement rather than a shortcut:
    deleting ``cuda`` from the real torch module breaks transformers long
    before control reaches the telemetry block, so the run adjudicates RED on
    an AttributeError raised inside the dependency and the branch is never
    entered. Reaching it needs the decision to be a unit with inputs, which is
    why it is one. The default arm stays end-to-end AND asserts that the reason
    the RUN recorded is the identical string the unit returns -- without that
    equality the unit could drift into a private opinion no manifest ever
    carries, and this test would be pinning dead code.
    """
    import torch

    from foundationscale.train.loop import _cuda_availability

    cfg_default = _cfg(
        tmp_path / "available-false",
        profile_path,
        model=str(tiny_model_dir),
        dataset=str(text_dataset),
    )
    rc = train(cfg_default)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out
    entry_default = _read_run_manifest(cfg_default.output_dir)["telemetry"][
        "peak_memory_allocated_bytes"
    ]
    assert entry_default["source"] == "unmeasured", entry_default
    reason_default = entry_default["value"]

    available, unit_reason = _cuda_availability(torch)
    assert not available, (
        "precondition: the autouse fixture pins this run to CPU, so is_available() "
        "must be False here -- on a box with a visible device this arm measures "
        "nothing and its reason assertion below could not fire"
    )
    assert unit_reason == reason_default, (
        f"the reason the RUN recorded ({reason_default!r}) is not the reason the unit "
        f"produces ({unit_reason!r}); they have drifted, so the no-module assertion "
        f"below is about a string no manifest will ever carry"
    )

    class _TorchWithoutCuda:
        """A torch-shaped object with no ``cuda`` attribute: the branch's input."""

    _, reason_no_module = _cuda_availability(_TorchWithoutCuda())

    assert "is_available() is False" in reason_default, (
        f"the present-but-unavailable arm did not fire or its reason changed: {reason_default!r}"
    )
    assert "no torch.cuda module" in reason_no_module, (
        f"the no-module arm did not fire or its reason changed: {reason_no_module!r}"
    )
    assert reason_default != reason_no_module, (
        "the two absent-CUDA branches record the same reason; a reader cannot tell "
        "'no cuda module in this torch build' from 'cuda present but unavailable'"
    )


def _altered_telemetry_value(value: Any) -> Any:
    """Return a different value of the SAME kind as ``value``.

    A flat replacement -- every entry becomes 424242 -- is rejected by the
    telemetry record itself: an UNMEASURED entry's value IS its reason and must
    stay a non-empty string, so the flat form dies in validation before the
    fingerprint is ever computed, and the test would report a schema complaint
    while claiming to measure fingerprint scope. Altering each value in a
    type-appropriate way keeps every record legal while still moving every
    telemetry byte, which is the condition the fingerprint must be blind to.

    The list branch is what keeps that promise for ``train_loss_curve``: its
    value IS a list of [step, loss] pairs, and the string fallthrough would
    turn it into ``f"{value} (fs400-altered)"``. That still moves bytes, so
    the fingerprint test would stay green -- but it would no longer be
    altering a series as a series, and a future reader would take the string
    as the intended alteration of the entry. A list must stay a list of the
    same length, with each element altered by this same function, so the
    recursion below is the contract, not an implementation detail.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return 424242
    if isinstance(value, list):
        return [_altered_telemetry_value(element) for element in value]
    return f"{value} (fs400-altered)"


def test_altered_telemetry_value_maps_a_list_to_an_altered_list() -> None:
    """Positive control for the helper's list branch: it is LIVE, not merely present.

    Without this test the branch above could be deleted or shadowed by the
    string fallthrough and nothing would go red: a stringified list still
    moves bytes, so the fingerprint test's alteration arm would stay green
    while quietly stopping testing what it says it tests. Asserting the shape
    of the alteration directly -- list in, list of the same length out, every
    element changed -- is what makes the branch's existence measured rather
    than assumed.
    """
    original = [[0, 2.91], [1, 2.88], [2, 2.85]]
    altered = _altered_telemetry_value(original)
    assert isinstance(altered, list), (
        f"_altered_telemetry_value turned a list into a {type(altered).__name__}; "
        f"the alteration must stay the same KIND as the value, or the fingerprint "
        f"test's alteration arm no longer alters a series as a series"
    )
    assert len(altered) == len(original), (
        f"the altered list has {len(altered)} elements against {len(original)}; "
        f"an alteration that changes the length is not the same kind of value"
    )
    for before, after in zip(original, altered, strict=True):
        assert isinstance(after, list) and len(after) == len(before), (
            f"element {before!r} came back as {after!r}; a [step, loss] pair must "
            f"come back as an altered pair, not a scalar or a string"
        )
        assert after != before, (
            f"element {before!r} survived the alteration unchanged as {after!r}; "
            f"an alteration that changes nothing would let the fingerprint "
            f"equality hold over identical bytes and prove nothing"
        )


def test_telemetry_does_not_move_the_manifest_fingerprint(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins the fingerprint's SCOPE: configuration in, outcome out.

    Without this test a reader would take the manifest fingerprint as a
    configuration identity. Telemetry is an OUTCOME of the run; if outcome
    bytes entered the fingerprint, two identical configurations would
    fingerprint differently merely because one ran slower, and every tool
    that diffs runs by fingerprint would report phantom configuration
    changes. The control arm matters as much: altering something that IS
    configuration must move the fingerprint, or the equality is vacuous.
    """
    cfg = _cfg(tmp_path, profile_path, model=str(tiny_model_dir), dataset=str(text_dataset))
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    manifest = _read_run_manifest(cfg.output_dir)
    assert manifest["telemetry"], (
        "precondition: the done manifest carries a non-empty telemetry section, so "
        "removing or altering it changes real bytes"
    )
    # The entry point is RunManifest.fingerprint(), a method over the record --
    # not a module function over a dict -- so the manifest is rebuilt from the
    # bytes actually written before each hash. Imported here rather than at
    # module scope: this is the only test in the module that needs it.
    from foundationscale.provenance.manifest import RunManifest

    with_telemetry = RunManifest.from_dict(manifest).fingerprint()

    stripped = RunManifest.from_dict(
        {k: v for k, v in manifest.items() if k != "telemetry"}
    ).fingerprint()
    assert stripped == with_telemetry, (
        "removing the telemetry section moved the fingerprint, so outcome bytes sit "
        "inside a configuration identity: two identical configurations would "
        "fingerprint differently merely because one ran on a busier machine"
    )

    altered_payload = dict(manifest)
    altered_payload["telemetry"] = {
        key: {**entry, "value": _altered_telemetry_value(entry["value"])}
        for key, entry in manifest["telemetry"].items()
    }
    assert altered_payload["telemetry"] != manifest["telemetry"], (
        "precondition: the alteration changed nothing, so the equality below would "
        "hold over identical bytes and prove nothing"
    )
    altered = RunManifest.from_dict(altered_payload).fingerprint()
    assert altered == with_telemetry, (
        "changing every telemetry VALUE moved the fingerprint; absence and alteration "
        "must BOTH be invisible to it, or a slower re-run of the same configuration "
        "reads as a different computation"
    )

    # Control arm: the two equalities above are satisfied by a fingerprint that
    # never moves at all. Alter something that IS configuration and require it to
    # move, so the equalities are evidence of scope rather than of constancy.
    config_key = sorted(manifest["config"])[0]
    config_payload = dict(manifest)
    config_payload["config"] = {
        **manifest["config"],
        config_key: {**manifest["config"][config_key], "value": "fs400-control-arm-sentinel"},
    }
    moved = RunManifest.from_dict(config_payload).fingerprint()
    assert moved != with_telemetry, (
        f"altering the configuration key {config_key!r} did NOT move the fingerprint, so "
        "the two equalities above hold over a constant and prove nothing about scope"
    )


def test_written_manifest_reloads_without_an_unknown_key_finding_for_telemetry(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins that telemetry is a DECLARED schema section, not a tolerated extra.

    Without this test a reader would assume that because the writer puts a
    telemetry section in, the reader owns it. loop.py's own docstring says
    unknown top-level keys are preserved and resurfaced as findings -- and
    calls that behaviour correct; if telemetry were serialized as an extra
    key, EVERY well-formed run would report an unknown-key finding, and
    operators would learn to ignore findings, the exact outcome the findings
    channel exists to prevent. The reload goes through
    ``provenance.manifest.load`` -- the reader every manifest consumer goes
    through, imported inside this body per the module's import rule -- not
    through a second json.load that would validate nothing. That loader does
    not raise on an unrecognised key; it stashes it and resurfaces it as a
    finding, so a tolerated-extra telemetry section would reload CLEAN and
    only the findings list would say otherwise. That list is what this test
    reads.
    """
    cfg = _cfg(tmp_path, profile_path, model=str(tiny_model_dir), dataset=str(text_dataset))
    rc = train(cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    from foundationscale.provenance.manifest import load as load_manifest
    from foundationscale.train.loop import MANIFEST_NAME

    manifest_path = Path(cfg.output_dir) / MANIFEST_NAME
    reloaded = load_manifest(manifest_path)
    assert reloaded.telemetry, (
        "the reloaded manifest carries no telemetry section; a loader that silently "
        "drops the section is indistinguishable from one that never flagged it"
    )
    finding_texts = [f if isinstance(f, str) else str(f) for f in reloaded.findings]
    unknown_telemetry = [
        text for text in finding_texts if "telemetry" in text and "unknown" in text.lower()
    ]
    assert not unknown_telemetry, (
        f"reloading the written manifest reported unknown-key finding(s) for the "
        f"telemetry section: {unknown_telemetry}; the section must be schema-declared, "
        f"not tolerated"
    )


def test_logging_steps_effective_is_derived_and_matches_the_resolved_cadence(
    tiny_model_dir: Path,
    profile_path: Path,
    text_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pins the one DERIVED telemetry entry: status derived, value resolved.

    Without this test a reader would take ``logging_steps_effective`` for the
    cadence the operator requested. It is the cadence the trainer RESOLVED, and
    the two differ in exactly one direction: a declared value is honoured as
    declared, and only an ABSENT one is bound against max_steps. That
    asymmetry is deliberate -- it is the fix for a 1-step run that logged no
    loss at all under a default cadence of 10 -- so both arms are here. An
    all-declared test would pass against an implementation that ignored the
    fallback entirely; an all-absent one would pass against an implementation
    that overrode the operator. Nothing OBSERVED either value, so both are
    stamped derived: measured would claim an instrument reading that never
    happened.
    """
    declared_cfg = _cfg(
        tmp_path / "declared",
        profile_path,
        model=str(tiny_model_dir),
        dataset=str(text_dataset),
        logging_steps=7,
    )
    rc = train(declared_cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    declared = _read_run_manifest(declared_cfg.output_dir)["telemetry"]["logging_steps_effective"]
    assert declared["source"] == "derived", (
        f"logging_steps_effective is stamped {declared['source']!r}; the value was "
        f"computed from the config, not observed, and the status must say so"
    )
    assert declared["value"] == 7, (
        f"logging_steps_effective is {declared['value']!r} for a run that DECLARED 7; "
        f"a declared cadence is honoured as declared, and silently rewriting it to a "
        f"max_steps-derived number would make the operator's setting unfindable"
    )

    fallback_cfg = _cfg(
        tmp_path / "fallback",
        profile_path,
        model=str(tiny_model_dir),
        dataset=str(text_dataset),
        logging_steps=None,
        max_steps=1,
    )
    rc = train(fallback_cfg)
    out = capsys.readouterr().out
    assert rc in (EXIT_PASS, EXIT_UNMEASURED), out

    fallback = _read_run_manifest(fallback_cfg.output_dir)["telemetry"]["logging_steps_effective"]
    assert fallback["source"] == "derived", (
        f"the fallback cadence is stamped {fallback['source']!r}; it was derived from "
        f"max_steps, and the status must not claim it was read off an instrument"
    )
    assert fallback["value"] == 1, (
        f"logging_steps_effective is {fallback['value']!r} for a 1-step run that "
        f"declared no cadence; the fallback must bind to max_steps, because a bare "
        f"default of 10 logs no loss at all in a 1-step run and the objective gate's "
        f"verdict depends on a loss existing at the only step"
    )
