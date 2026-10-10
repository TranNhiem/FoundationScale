"""The video frame-budget arms of ``train()``: what a declared budget refuses and admits.

A video column alone refuses (pinned in test_modality_refusal.py). With
``FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES`` beside it the column becomes frames on the
image arm, and the arms here pin the three refusals that path adds -- a budget that
is not a budget, a declared column the dataset lacks, a clip that cannot become
frames -- plus the arm that gets PAST the fold, so a fold that refused everything
would fail here. The image arm's processor is replaced by the tiny model's own
tokenizer: these arms measure the declaration and the fold, which sit before any
collator, and the tiny Llama has no processor to resolve.

Fixtures are copied from tests/train/test_loop_audio_refuse_arms.py, because a test
module must not import from a sibling.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from foundationscale.train.loop import (
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

_VIDEO_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN"
_FRAMES_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES"


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


@pytest.fixture()
def video_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "video.jsonl"
    rows = [{"text": f"<video> w{i} w{i + 1}", "clip": f"clip{i}.mp4"} for i in range(4)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture()
def tokenizer_surface(monkeypatch: pytest.MonkeyPatch, tiny_model_dir: Path) -> None:
    import types

    from transformers import AutoTokenizer

    from foundationscale.rl import prompt_surface

    tok = AutoTokenizer.from_pretrained(str(tiny_model_dir))
    monkeypatch.setattr(
        prompt_surface,
        "resolve_prompt_surface",
        lambda *_a, **_k: types.SimpleNamespace(surface=tok),
    )


def _declare(monkeypatch: pytest.MonkeyPatch, frames: str) -> None:
    monkeypatch.setenv(_VIDEO_ENV, "clip")
    monkeypatch.setenv(_FRAMES_ENV, frames)
    monkeypatch.delenv("FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN", raising=False)
    monkeypatch.delenv("FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN", raising=False)


def test_a_budget_that_is_not_a_budget_refuses_before_anything_loads(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _declare(monkeypatch, "0")
    assert train(cfg) == EXIT_REFUSE
    assert "video frame budget is not a budget" in capsys.readouterr().out
    assert _read_manifest(cfg)["config"]["stage"]["value"] == "refused"


def test_a_declared_video_column_the_dataset_lacks_refuses(
    cfg: TrainConfig,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_surface: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _declare(monkeypatch, "2")
    assert train(cfg) == EXIT_REFUSE
    out = capsys.readouterr().out
    assert "video column 'clip' is declared but dataset" in out
    assert "frame budget f2-uniform-s0" in out


def test_a_clip_that_cannot_become_frames_refuses(
    cfg: TrainConfig,
    video_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_surface: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from foundationscale import video

    def broken(*_a: Any, **_k: Any) -> list[str]:
        raise video.VideoDecodeError("no decoder here")

    monkeypatch.setattr(video, "video_frame_paths", broken)
    _declare(monkeypatch, "2")
    run = dataclasses.replace(cfg, dataset=str(video_dataset))
    assert train(run) == EXIT_REFUSE
    assert "could not become frames: no decoder here" in capsys.readouterr().out


def test_a_decodable_corpus_gets_past_the_fold(
    cfg: TrainConfig,
    video_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_surface: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The control: with frames available the fold does NOT refuse.

    Whatever the stand-in surface then does downstream is not this module's claim;
    what is pinned is that every clip was asked for and no fold refusal was printed.
    """
    from foundationscale import video

    asked: list[str] = []

    def frames(path: Any, budget: Any, cache_dir: Any, **_k: Any) -> list[str]:
        asked.append(str(path))
        return [f"{cache_dir}/f{i}.jpg" for i in range(budget.frames)]

    monkeypatch.setattr(video, "video_frame_paths", frames)
    _declare(monkeypatch, "2")
    run = dataclasses.replace(cfg, dataset=str(video_dataset))
    # Downstream of the fold the stand-in frames are not real images, and the
    # collator's own refusal (SystemExit 96) is a different layer's claim.
    with contextlib.suppress(SystemExit):
        train(run)
    out = capsys.readouterr().out
    assert len(asked) == 4 and asked[0].endswith("clip0.mp4")
    assert "could not become frames" not in out
    assert "is declared but dataset" not in out
