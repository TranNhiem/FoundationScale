"""The conversation path's video wiring in ``train()``: declaration-only arms.

Native video (``FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN`` +
``FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN`` + ``FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES``)
is a DIFFERENT arm from the plain image-fold path tests/train/test_loop_video_arms.py
pins: a clip stays one video (train/conversation.py's frames_for hook), never N
separate images folded onto an image column. These arms cover what is reachable
WITHOUT a working chat-template processor -- the declared-column-vs-dataset-
columns check, and the DATA line announcing which path a declared video takes --
mirroring test_loop_video_arms.py's own split between "declaration-only" (pinned
here, on a CPU box with a bare tokenizer) and "full collation" (GPU proof,
validation_campaigns/vlm_2xh200), the same split test_conversation_loop_wiring.py's
own docstring states for the conversation axes generally.

Fixtures are copied from tests/train/test_loop_video_arms.py (itself copied from
test_loop_audio_refuse_arms.py), because a test module must not import from a
sibling.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from foundationscale.train.loop import EXIT_REFUSE, TrainConfig, train

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

_SPECIAL_TOKENS = ("<unk>", "<pad>", "<eos>")
_WORD_TOKENS = tuple(f"w{i}" for i in range(253))

_CONVERSATIONS_ENV = "FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN"
_OVERLONG_ENV = "FOUNDATIONSCALE_TRAIN_OVERLONG"
_VIDEO_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN"
_FRAMES_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES"

_ORIGINAL_CUDA_PROBES: dict[str, object] = {}
_ORIGINAL_MPS_PROBES: dict[str, Any] = {}


def _false_probe_like(original: Any) -> Any:
    stub: Any = lambda: False  # noqa: E731 -- must be a plain function to match shape
    if hasattr(original, "cache_info"):
        stub = functools.lru_cache(maxsize=None)(stub)
    return stub


@pytest.fixture(scope="module")
def _offline_module() -> None:
    mp = pytest.MonkeyPatch()
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.setenv("TRANSFORMERS_OFFLINE", "1")
    mp.setenv("HF_DATASETS_OFFLINE", "1")
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _offline_and_cpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
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
def conversations_dataset(tmp_path: Path) -> Path:
    """4 rows of {"conversations": [...]}, each human turn carrying a <video> marker.

    No "clip" (or any video) column at all -- the declaration-vs-dataset-columns
    check this module pins needs exactly that gap.
    """
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "conversations": [
                {"from": "human", "value": f"<video> w{i} w{i + 1}"},
                {"from": "gpt", "value": f"w{i + 2}"},
            ]
        }
        for i in range(4)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture()
def cfg(
    tmp_path: Path,
    tiny_model_dir: Path,
    conversations_dataset: Path,
    profile_path: Path,
) -> TrainConfig:
    return TrainConfig(
        model=str(tiny_model_dir),
        dataset=str(conversations_dataset),
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


def _declare_conversation_video(monkeypatch: pytest.MonkeyPatch, frames: str) -> None:
    monkeypatch.setenv(_CONVERSATIONS_ENV, "conversations")
    monkeypatch.setenv(_OVERLONG_ENV, "drop")
    monkeypatch.setenv(_VIDEO_ENV, "clip")
    monkeypatch.setenv(_FRAMES_ENV, frames)
    monkeypatch.delenv("FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN", raising=False)
    monkeypatch.delenv("FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN", raising=False)


def test_video_declared_with_conversations_is_sampled_natively_not_on_an_image_column(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The DATA line names the native path -- never 'on image column', which
    would mean the plain-image-fold auto-fallback mis-fired for a declaration
    that asked for the conversation path instead."""
    _declare_conversation_video(monkeypatch, "2")
    assert train(cfg) == EXIT_REFUSE  # the dataset has no "clip" column -- see below
    out = capsys.readouterr().out
    assert "sampled NATIVELY for the conversation path" in out
    assert "on image column" not in out


def test_video_column_declared_but_dataset_lacks_it_refuses_under_conversations(
    cfg: TrainConfig,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_surface: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _declare_conversation_video(monkeypatch, "2")
    assert train(cfg) == EXIT_REFUSE
    out = capsys.readouterr().out
    assert "video column 'clip' is declared alongside conversations" in out
    assert "silent-drop defect" in out
