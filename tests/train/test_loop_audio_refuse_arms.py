"""REFUSE arms for the audio-modality refusal gates in ``train()``.

The audio arms sit ahead of any model call: ``train()`` refuses a run whose
declared modality set the training plane cannot honour, and it does so by
REMARKING -- not by raising -- so the verdict is EXIT_REFUSE (96) with a
reason text written to ``run_manifest.json`` under ``cfg.output_dir``. Those
are the four arms pinned here:

1. two non-text modalities at once -> "One non-text modality", naming both
   ``FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN`` and ``FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN``;
2. ``FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN`` declared but the dataset has no such
   column -> the "audio column 'X' is declared but dataset" line;
3. a declared audio column that IS present but whose rows name a family the
   model registry has never heard of (this module's tiny Llama is deliberately
   an unregistered family) -> the audio-support refusal, which must still name
   ``audio``;
4. a CONTROL with neither env var set: the same tiny model / same dataset runs
   the ordinary path and is NOT refused for an audio reason. The control is
   what makes the other three load-bearing -- any implementation that returned
   96 unconditionally would pass arms 1-3 and must fail here.

No skips anywhere: this module runs under FS_FORBID_SKIPS=1 on CPU-only boxes.
The autouse fixture pins CUDA/MPS probes to doubles that are not narrower than
the originals (see ``_false_probe_like``); the verbatim fixture set from
``tests/train/test_loop_refuse_arms.py`` is copied here because this module
must not import from a sibling test module.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from foundationscale.train.loop import (
    EXIT_RED,
    EXIT_REFUSE,
    TrainConfig,
    train,
)

# Same shape of profile fixture as tests/test_train_entry.py: a single-node
# reference profile the declared 1x1 topology satisfies exactly, so the
# prologue emits no blocking findings and execution reaches the refusal gates.
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

# vocab_size=256 below, so the tokenizer must emit ids strictly below 256 or a
# fixture defect would masquerade as a training defect. 3 + 253 == 256 exactly.
_SPECIAL_TOKENS = ("<unk>", "<pad>", "<eos>")
_WORD_TOKENS = tuple(f"w{i}" for i in range(253))

_AUDIO_ENV = "FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN"
_IMAGE_ENV = "FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN"


# Whatever the autouse fixture displaced, recorded at patch time: a test body
# cannot read the original itself because the fixture has already run by then.
_ORIGINAL_CUDA_PROBES: dict[str, object] = {}
_ORIGINAL_MPS_PROBES: dict[str, Any] = {}


def _false_probe_like(original: Any) -> Any:
    """A zero-arg ``False`` stub that is not NARROWER than the probe it replaces.

    torch introspects these probes -- ``torch/_dynamo/variables/torch.py`` reads
    ``__wrapped__`` while building its constant-fold table -- so a bare lambda
    is narrower than the cached original and the failure then surfaces, or does
    not, according to when the lazy dynamo import happens to land. Wrapping the
    stub in ``lru_cache`` when the original is cached keeps every attribute
    reachable through ``__wrapped__`` answering False.
    """
    stub: Any = lambda: False  # noqa: E731 -- must be a plain function to match shape
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


@pytest.fixture(autouse=True)
def _offline_and_cpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every hardware/network reach to a double; the real arms are wired by raise.

    Nothing here may touch torch.cuda for real: this suite's contract is to
    pass on a CPU-only box. The environment-fault arms do not fake hardware --
    they raise a wrapped OSError at the call site under test, which is the
    shape a wrapped fault takes in the libraries this plane calls.
    """
    _ORIGINAL_CUDA_PROBES["is_available"] = torch.cuda.is_available
    monkeypatch.setattr(torch.cuda, "is_available", _false_probe_like(torch.cuda.is_available))
    for name in ("is_available", "is_built"):
        original = getattr(torch.backends.mps, name)
        _ORIGINAL_MPS_PROBES[name] = original
        monkeypatch.setattr(torch.backends.mps, name, _false_probe_like(original))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


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

    The family deliberately does NOT match any audio-capable family this
    training plane has audio support registered for -- that is what makes the
    audio-support refusal reachable in this module (arm 3).
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
    claims to train. This is the dataset that declares NO audio column, which
    is exactly the declarata required for arm 2.
    """
    path = tmp_path / "train.jsonl"
    rows = [
        {"text": f"w{(i * 7) % 253} w{(i * 7 + 1) % 253} w{(i * 7 + 2) % 253} w{(i * 7 + 3) % 253}"}
        for i in range(16)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture()
def audio_dataset(tmp_path: Path) -> Path:
    """Rows with a text column, an answer column, and an ``audio`` column.

    The ``audio`` values are string paths that are never opened: the audio
    refusal gates sit ahead of any decode, so what they measure is the
    DECLARATION, not the payload.
    """
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "text": f"w{(i * 7) % 253} w{(i * 7 + 1) % 253}",
            "answer": f"w{(i * 7 + 2) % 253}",
            "audio": f"/nonexistent/audio-{i}.wav",
        }
        for i in range(16)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture()
def cfg(
    tmp_path: Path,
    tiny_model_dir: Path,
    text_dataset: Path,
    profile_path: Path,
) -> TrainConfig:
    """Field set mirrors the config fixture of tests/test_train_execution.py.

    Only fields TrainConfig actually declares: the cluster profile arrives as
    ``profile_path`` (a Path to the JSON file the ``profile_path`` fixture
    writes, not as a dict), the batch-size field is ``per_device_batch_size``,
    the save cadence is ``save_interval``, and the required topology fields
    ``nodes``/``gpus_per_node`` are given explicitly. ``objective`` is declared
    because the control arm lets the REAL Trainer.train() run, and the
    objective gate refuses an undeclared run at its first observed step --
    omitting it would make that arm red for a reason that has nothing to do
    with what the arm is pinning.
    """
    return TrainConfig(
        model=str(tiny_model_dir),
        dataset=str(text_dataset),
        output_dir=tmp_path / "output",
        nodes=1,
        gpus_per_node=1,
        profile_path=profile_path,
        objective="sft",
        precision="fp32",
        attn_implementation="eager",
        max_steps=2,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_interval=50,
        dry_run=False,
    )


def _manifest_path(cfg: TrainConfig) -> Path:
    return Path(cfg.output_dir) / "run_manifest.json"


def _read_manifest(cfg: TrainConfig) -> dict[str, Any]:
    path = _manifest_path(cfg)
    assert path.exists(), (
        f"wanted the run to write a run_manifest.json at {path} before "
        f"returning; a refusal without a manifest leaves no reason on disk to "
        f"audit against"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _refusal_text(manifest: dict[str, Any]) -> str:
    """Concatenate every string leaf of the manifest into one searchable corpus.

    The refusal reason lives in a field this module does NOT pin by name --
    only its words -- so a flattened search over the whole document keeps the
    test from over-constraining the manifest schema while still holding the
    reason text to its contract.
    """
    parts: list[str] = []
    stack: list[Any] = [manifest]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return "\n".join(parts)


def test_two_non_text_modalities_refuse_citing_both_columns(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AUDIO and IMAGE at once -> 96, 'One non-text modality', both columns named.

    The gate is a budget check (one non-text modality per run), so the
    refusal text must say so AND name both offending declarations -- naming
    only one of ``audio``/``image`` would tell the operator which knob to turn
    down but not which kind of run they were trying to throw together.
    """
    monkeypatch.delenv(_AUDIO_ENV, raising=False)
    monkeypatch.delenv(_IMAGE_ENV, raising=False)
    monkeypatch.setenv(_AUDIO_ENV, "audio")
    monkeypatch.setenv(_IMAGE_ENV, "image")

    verdict = train(cfg)
    assert verdict == EXIT_REFUSE, (
        f"wanted train() to adjudicate an audio+image run as REFUSE (exit "
        f"{EXIT_REFUSE}): only one non-text modality may be declared per run, "
        f"so this one is unscored rather than failed; got exit {verdict}"
    )

    manifest = _read_manifest(cfg)
    assert manifest["config"]["stage"]["value"] == "refused", (
        f"wanted run_manifest.json to record the run as refused (status/stage "
        f"== 'refused') once the modality budget check has spoken; got "
        f"{manifest.get('status')!r}/{manifest.get('stage')!r} in {manifest}"
    )

    # The refusal sentence is printed by _mark; the manifest records the stage.
    text = capsys.readouterr().out + _refusal_text(manifest)
    assert "One non-text modality" in text, (
        f"wanted the refusal text to say 'One non-text modality' so the "
        f"operator sees the budget rule being enforced, not a dataset bug; "
        f"got:\n{text}"
    )
    for column in ("audio", "image"):
        assert column in text, (
            f"wanted the refusal text to name the {column!r} column it is "
            f"refusing alongside the other one, so the operator knows which "
            f"declaration to drop; got:\n{text}"
        )


def test_declared_audio_column_missing_from_dataset_refuses(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AUDIO declared, dataset has no such column -> 96 naming the column and the dataset.

    The refusal is a declarata/contents mismatch, so the text is contractually
    obliged to name both sides of the disagreement: the column the operator
    asked for and the dataset that does not carry it.
    """
    monkeypatch.delenv(_IMAGE_ENV, raising=False)
    monkeypatch.delenv(_AUDIO_ENV, raising=False)
    monkeypatch.setenv(_AUDIO_ENV, "audio")

    verdict = train(cfg)
    assert verdict == EXIT_REFUSE, (
        f"wanted train() to refuse a run whose declared audio column "
        f"'audio' does not exist in the dataset (exit {EXIT_REFUSE}): the "
        f"declared modality is unsatisfiable, so the run is unscored rather "
        f"than failed; got exit {verdict}"
    )

    manifest = _read_manifest(cfg)
    assert manifest["config"]["stage"]["value"] == "refused", (
        f"wanted run_manifest.json to record the run as refused once the "
        f"audio column check has spoken; got {manifest.get('status')!r}/"
        f"{manifest.get('stage')!r} in {manifest}"
    )

    # The refusal sentence is printed by _mark; the manifest records the stage.
    text = capsys.readouterr().out + _refusal_text(manifest)
    assert "audio column 'audio' is declared but dataset" in text, (
        f"wanted the refusal text to contain exactly "
        f"\"audio column 'audio' is declared but dataset\", naming the "
        f"column AND the dataset it is missing from; got:\n{text}"
    )


def test_audio_column_present_but_family_unregistered_refuses(
    cfg: TrainConfig,
    audio_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AUDIO declared and present -> 96 via the audio-support refusal.

    The tiny Llama here is deliberately an UNREGISTERED family for the audio
    support table, so the run cannot be served even though the dataset column
    resolves -- this is the arm that proves the refusal is keyed on what the
    MODEL can carry, not on whether the dataset offers it. The audio paths in
    the rows are never opened; the gate sits ahead of any decode.
    """
    monkeypatch.delenv(_IMAGE_ENV, raising=False)
    monkeypatch.delenv(_AUDIO_ENV, raising=False)
    monkeypatch.setenv(_AUDIO_ENV, "audio")

    cfg = dataclasses.replace(cfg, **{"dataset": str(audio_dataset)})
    verdict = train(cfg)

    assert verdict == EXIT_REFUSE, (
        f"wanted train() to refuse an audio run on a model whose family has "
        f"no audio support registered (exit {EXIT_REFUSE}): the audio tower "
        f"this run would need is not something the training plane can build "
        f"for this family; got exit {verdict}"
    )

    manifest = _read_manifest(cfg)
    # The refusal sentence is printed by _mark; the manifest records the stage.
    text = capsys.readouterr().out + _refusal_text(manifest)
    assert "audio" in text, (
        f"wanted the refusal text to name 'audio' so the operator can see "
        f"which modality is unsupported for this model family; got:\n{text}"
    )


def test_no_modalities_declared_is_not_refused_for_audio(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: no modality env vars -> the audio gates MUST stay silent.

    This is the load-bearing control for the whole module: any implementation
    that returned 96 unconditionally would pass every arm above and must fail
    here. The assertion is deliberately narrow -- it pins only that the run is
    not REFUSEd for an audio reason -- so it cannot fail for a loss-flame or a
    save defect unrelated to what this module measures.
    """
    monkeypatch.delenv(_IMAGE_ENV, raising=False)
    monkeypatch.delenv(_AUDIO_ENV, raising=False)

    verdict = train(cfg)

    assert verdict != EXIT_REFUSE, (
        f"wanted a run with no modality declaration to escape the audio "
        f"refusal gates (exit must not be {EXIT_REFUSE}): nothing about this "
        f"run names a non-text modality, so there is nothing to refuse here; "
        f"got exit {verdict}"
    )

    if verdict in (EXIT_RED, 0):
        manifest_path = _manifest_path(cfg)
        if manifest_path.exists():
            text = _refusal_text(json.loads(manifest_path.read_text(encoding="utf-8")))
            for needle in (
                "One non-text modality",
                _AUDIO_ENV,
                _IMAGE_ENV,
                "audio_support_refusal",
            ):
                assert needle not in text, (
                    f"wanted the control run's manifest to carry NO audio "
                    f"refusal language, but found {needle!r} in:\n{text}"
                )
