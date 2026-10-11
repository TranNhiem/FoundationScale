# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the offline preference trainer's image/video support.

These exercise the media declaration surface added to ``PreferenceTrainConfig``
and ``PreferenceTrainer``: the config validation (partial ``video_*``
declarations refused, same doctrine as ``adapter``/``adapter_*``), the
``run()``-level carrier check (undeclared media still refuses "text-only",
declared media is let through to the processor path), and the processor-
rendered batch builder ``_encode_batch_media`` (completion-only label
masking via ``assistant_label_mask``, chosen/rejected sharing one record's
rendered media, KTO's single-row-per-record shape, the never-truncate
overlong policy) and ``_forward_logprobs``'s reuse of
``rl.trainer._narrow_modality_kwargs_by_row`` for Qwen-style flattened-patch
micro-batch slicing. None of this needs a real vision tower or a downloaded
checkpoint: a small deterministic fake processor (mirroring
``tests/train/test_conversation_collator.py``'s ``_FakeConversationProcessor``)
stands in for ``Gemma4UnifiedProcessor``/``Qwen3VLProcessor``.

WHAT IS CLAIMED: an image-declared paired batch's two sides (chosen,
rejected) share byte-identical rendered pixel content; the derived
supervision mask covers exactly the completion tokens; a flattened-patch
``pixel_values``/``image_grid_thw`` pair survives ``logprob_micro_batch``
row-range slicing with the correct per-row patch block; an overlong media
row refuses rather than truncates; and declaring ``image_column``/
``video_column`` is required for a carried field to escape the historical
text-only refusal, which is otherwise byte-identical to before this file's
feature existed (see ``tests/rl/test_preference_trainer.py``'s
``test_record_carrying_an_image_field_refuses_exit_96``, unmodified).

WHAT IS NOT CLAIMED: that a real transformers processor or vision tower
produces these exact token-expansion counts, or anything about GPU behaviour
-- that is the ``validation_campaigns/vlm_2xh200`` job's claim.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl import preference_trainer as pt  # noqa: E402
from foundationscale.rl.preference_torch import TensorPreferenceLoss  # noqa: E402
from foundationscale.rl.preference_trainer import (  # noqa: E402
    PreferenceTrainConfig,
    PreferenceTrainer,
)
from foundationscale.rl.registry import reset_algorithm_registry  # noqa: E402
from foundationscale.rl.trainer import TrainerRefusal  # noqa: E402
from foundationscale.train.conversation import derive_turn_markers  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover -- PIL is a declared test dependency elsewhere too
    Image = None  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def _default_registry() -> None:
    reset_algorithm_registry()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _cfg(model: str, dataset: str, **overrides: Any) -> PreferenceTrainConfig:
    kwargs: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "pairs_per_step": 2,
        "max_steps": 1,
        "learning_rate": 1e-3,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return PreferenceTrainConfig(**kwargs)


def _trainer(**overrides: Any) -> PreferenceTrainer:
    overrides.setdefault("max_length", 64)
    return PreferenceTrainer(_cfg("m", "d.jsonl", **overrides))


def _make_image(tmp_path: Path, name: str) -> str:
    assert Image is not None, "Pillow is required for the multimodal preference tests"
    path = tmp_path / name
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Config validation: image_column / video_column / video_* partial-declaration
# ---------------------------------------------------------------------------


def test_config_image_column_blank_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="non-empty string"):
        PreferenceTrainer(_cfg("m", "d.jsonl", image_column="   "))


def test_config_video_column_blank_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="non-empty string"):
        PreferenceTrainer(_cfg("m", "d.jsonl", video_column="   "))


def test_config_video_frames_without_video_column_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="partial video"):
        PreferenceTrainer(_cfg("m", "d.jsonl", video_frames=4))


def test_config_video_sampling_without_video_column_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="partial video"):
        PreferenceTrainer(_cfg("m", "d.jsonl", video_sampling="other"))


def test_config_video_max_side_without_video_column_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="partial video"):
        PreferenceTrainer(_cfg("m", "d.jsonl", video_max_side=64))


def test_config_video_cache_dir_without_video_column_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="partial video"):
        PreferenceTrainer(_cfg("m", "d.jsonl", video_cache_dir="/tmp/cache"))


def test_config_video_column_with_zero_frames_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="invalid"):
        PreferenceTrainer(_cfg("m", "d.jsonl", video_column="video"))


def test_config_video_column_with_non_int_frames_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="an integer is required"):
        PreferenceTrainer(_cfg("m", "d.jsonl", video_column="video", video_frames=2.5))


def test_config_video_column_with_bool_max_side_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="an integer is required"):
        PreferenceTrainer(
            _cfg("m", "d.jsonl", video_column="video", video_frames=4, video_max_side=True)
        )


def test_config_declared_video_column_builds_frame_budget() -> None:
    trainer = _trainer(video_column="video", video_frames=4, video_max_side=256)
    assert trainer._needs_media is True
    assert trainer._video_budget is not None
    assert trainer._video_budget.frames == 4
    assert trainer._video_budget.max_side == 256


def test_config_image_column_alone_sets_needs_media_with_no_video_budget() -> None:
    trainer = _trainer(image_column="image")
    assert trainer._needs_media is True
    assert trainer._video_budget is None


def test_config_no_media_columns_is_the_historical_text_only_trainer() -> None:
    trainer = _trainer()
    assert trainer._needs_media is False
    assert trainer._video_budget is None
    assert trainer._processor is None
    assert trainer._turn_markers is None


# ---------------------------------------------------------------------------
# run(): the carrier check now keys off declared_columns, not "any media"
# ---------------------------------------------------------------------------


def test_run_undeclared_video_refuses_text_only_even_with_image_declared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Declaring image_column does not amnesty an UNDECLARED video field."""
    rows = [
        {
            "prompt": "<image> describe",
            "chosen": "a",
            "rejected": "b",
            "image": "x.png",
            "video": "y.mp4",
        }
    ]
    dataset = _write_jsonl(tmp_path / "d.jsonl", rows)
    cfg = _cfg(str(tmp_path / "no-such-model"), str(dataset), image_column="image")

    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    captured = capsys.readouterr()

    assert excinfo.value.code == 96
    assert "text-only" in captured.err
    assert "undeclared column" in captured.err
    assert "line 1 (video)" in captured.err


def test_run_declared_image_column_passes_the_carrier_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A declared image_column lets the record past the carrier check; the
    run then fails LATER, at processor load on a bogus model path -- proving
    the carrier check (not a lucky model load) is what changed."""
    rows = [{"prompt": "<image> describe", "chosen": "a", "rejected": "b", "image": "x.png"}]
    dataset = _write_jsonl(tmp_path / "d.jsonl", rows)
    cfg = _cfg(str(tmp_path / "no-such-model"), str(dataset), image_column="image")

    with pytest.raises(SystemExit) as excinfo:
        PreferenceTrainer(cfg).run()
    captured = capsys.readouterr()

    assert excinfo.value.code == 96
    assert "preference record field(s) carry image or video data" not in captured.err
    assert "AutoProcessor failed to load" in captured.err


# ---------------------------------------------------------------------------
# _encode_batch_media: a small deterministic fake processor
# ---------------------------------------------------------------------------


def _content_seed(image: Any) -> float:
    """A float keyed on an image's PIXEL CONTENT, not its object identity.

    Two independent PIL loads of the SAME file produce two different Python
    objects with identical bytes; two DIFFERENT files produce different
    bytes. ``_FakeProcessor`` uses this (rather than call order) to fill a
    flattened-patch block, so the tests below can tell "same declared media,
    rendered twice" from "two different images" by comparing tensors.
    """
    return float(sum(image.tobytes()) % 97)


class _FakeProcTokenizer:
    def __init__(self, processor: _FakeProcessor) -> None:
        self._p = processor

    @property
    def all_special_ids(self) -> list[int]:
        return sorted(self._p.special_ids)

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [self._p.intern(w) for w in text.split()]}


class _FakeProcessor:
    """A tiny, deterministic stand-in for Gemma4UnifiedProcessor/Qwen3VLProcessor.

    Mirrors ``tests/train/test_conversation_collator.py``'s
    ``_FakeConversationProcessor`` (same template/render shape), trimmed to
    one style and extended to emit Qwen-style FLATTENED-PATCH
    ``pixel_values``/``image_grid_thw`` (rather than one row per image) so
    the narrowing tests below exercise the same block-vs-row distinction
    ``rl.trainer._narrow_modality_kwargs_by_row`` was built for.
    """

    def __init__(self, *, image_expansion: int = 3, video_expansion: int = 2) -> None:
        self.image_expansion = image_expansion
        self.video_expansion = video_expansion
        self.pad_token_id = 0
        self.image_token_id = 1
        self.video_token_id = 2
        self.vocab: dict[str, int] = {}
        self.rev_vocab: dict[int, str] = {}
        self.special_ids: set[int] = {0}
        self._next_id = 3
        self.tokenizer = _FakeProcTokenizer(self)
        for role in ("user", "assistant", "system"):
            self.intern(f"<|im_start|>{role}")
        self.intern("<|im_end|>")

    def intern(self, word: str) -> int:
        is_structural = word == "<|im_end|>" or word.startswith("<|im_start|>")
        if word not in self.vocab:
            self.vocab[word] = self._next_id
            self.rev_vocab[self._next_id] = word
            self._next_id += 1
        word_id = self.vocab[word]
        if is_structural:
            self.special_ids.add(word_id)
        return word_id

    def _header_words(self, role: str) -> list[str]:
        return [f"<|im_start|>{role}", "<NL>"]

    def _end_words(self) -> list[str]:
        return ["<|im_end|>", "<NL>"]

    def _render_words(self, messages: list[dict[str, Any]], *, enable_thinking: bool) -> list[str]:
        words: list[str] = []
        for message in messages:
            words.extend(self._header_words(message["role"]))
            for block in message["content"]:
                if block["type"] == "text":
                    words.extend(block["text"].split())
                elif block["type"] == "image":
                    words.append("<image_ph>")
                elif block["type"] == "video":
                    words.append("<video_ph>")
                else:  # pragma: no cover -- normalize_conversation never emits another type
                    raise AssertionError(f"unexpected content block type {block['type']!r}")
            words.extend(self._end_words())
        return words

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        return_dict: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> Any:
        words = self._render_words(messages, enable_thinking=enable_thinking)
        if tokenize:
            return {"input_ids": [self.intern(w) for w in words]}
        return " ".join(words)

    def __call__(
        self,
        *,
        text: list[str],
        return_tensors: str = "pt",
        padding_side: str = "right",
        padding: Any = False,
        max_length: int | None = None,
        images: list[list[Any]] | None = None,
        videos: list[list[Any]] | None = None,
        video_metadata: list[list[dict[str, Any]]] | None = None,
        do_sample_frames: bool | None = None,
    ) -> dict[str, Any]:
        rows: list[list[int]] = []
        image_seeds: list[float] = []
        video_seeds: list[float] = []
        for index, row_text in enumerate(text):
            remaining_images = list(images[index]) if images else []
            remaining_videos = list(videos[index]) if videos else []
            ids: list[int] = []
            for word in row_text.split():
                if word == "<image_ph>" and remaining_images:
                    image = remaining_images.pop(0)
                    image_seeds.append(_content_seed(image))
                    ids.extend([self.image_token_id] * self.image_expansion)
                elif word == "<video_ph>" and remaining_videos:
                    frame = remaining_videos.pop(0)[0]
                    video_seeds.append(_content_seed(frame))
                    ids.extend([self.video_token_id] * self.video_expansion)
                else:
                    ids.append(self.intern(word))
            rows.append(ids)
        row_lengths = [len(r) for r in rows]
        if padding == "max_length":
            assert max_length is not None
            width = max([max_length, *row_lengths])
        else:
            width = max(row_lengths, default=0)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        for index, ids in enumerate(rows):
            n = len(ids)
            if n == 0:
                continue
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            if padding_side == "left":
                input_ids[index, width - n :] = ids_tensor
                attention_mask[index, width - n :] = 1
            else:
                input_ids[index, :n] = ids_tensor
                attention_mask[index, :n] = 1
        out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        total_images = sum(len(row) for row in (images or []))
        if images is not None and total_images > 0:
            # Flattened patches: image_expansion patches per image,
            # concatenated row-major. Each image's block is seeded from the
            # IMAGE'S OWN CONTENT (not its position in the batch), so two
            # renders of the SAME declared file produce the SAME block
            # (chosen/rejected sharing one record's media) while two
            # different files produce different ones.
            blocks = [
                torch.arange(self.image_expansion * 2, dtype=torch.float32).reshape(
                    self.image_expansion, 2
                )
                + seed
                for seed in image_seeds
            ]
            out["pixel_values"] = torch.cat(blocks, dim=0)
            out["image_grid_thw"] = torch.tensor(
                [[1, 1, self.image_expansion]] * total_images, dtype=torch.long
            )
        total_videos = sum(len(row) for row in (videos or []))
        if videos is not None and total_videos > 0:
            assert do_sample_frames is False
            blocks = [
                torch.arange(self.video_expansion * 2, dtype=torch.float32).reshape(
                    self.video_expansion, 2
                )
                + seed
                for seed in video_seeds
            ]
            out["pixel_values_videos"] = torch.cat(blocks, dim=0)
            out["video_grid_thw"] = torch.tensor(
                [[1, 1, self.video_expansion]] * total_videos, dtype=torch.long
            )
        return out


def _media_trainer(processor: _FakeProcessor, **overrides: Any) -> PreferenceTrainer:
    overrides.setdefault("max_length", 64)
    overrides.setdefault("image_column", "image")
    trainer = _trainer(**overrides)
    trainer._processor = processor
    trainer._turn_markers = derive_turn_markers(processor)
    return trainer


def _image_record(image_path: str, **fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "prompt": "<image> describe this",
        "chosen": "a cat",
        "rejected": "a dog",
        "image": image_path,
        "_line": 1,
    }
    record.update(fields)
    return record


def test_encode_batch_media_label_mask_covers_only_the_completion(tmp_path: Path) -> None:
    processor = _FakeProcessor()
    trainer = _media_trainer(processor)
    image_path = _make_image(tmp_path, "photo.png")
    batch = trainer._encode_batch(
        records=(_image_record(image_path),),
        tokenizer=None,
        pad_token_id=0,
        device="cpu",
    )
    assert batch.input_ids.shape[0] == 2  # chosen, rejected
    assert batch.modality_kwargs is not None
    assert "pixel_values" in batch.modality_kwargs

    # Decode which tokens are supervised on each row and check they are
    # exactly that row's completion words PLUS the end-of-turn marker
    # (assistant_label_mask's span is inclusive of the end marker) --
    # nothing from the header or the (image-bearing) prompt turn.
    assert trainer._turn_markers is not None
    end_width = len(trainer._turn_markers.end_ids)
    for row_index, completion in enumerate(("a cat", "a dog")):
        width = int(batch.attention_mask[row_index].sum())
        mask_row = batch.completion_mask[row_index, : width - 1]
        supervised_positions = int(mask_row.sum())
        expected = len(completion.split()) + end_width
        assert supervised_positions == expected, (
            f"row {row_index}: expected {expected} supervised positions "
            f"({len(completion.split())} completion words + {end_width} "
            f"end-marker tokens), got {supervised_positions}"
        )


def test_encode_batch_media_chosen_and_rejected_share_the_record_image(tmp_path: Path) -> None:
    processor = _FakeProcessor()
    trainer = _media_trainer(processor)
    image_path = _make_image(tmp_path, "photo.png")
    batch = trainer._encode_batch(
        records=(_image_record(image_path),),
        tokenizer=None,
        pad_token_id=0,
        device="cpu",
    )
    assert batch.modality_kwargs is not None
    pixel_values = batch.modality_kwargs["pixel_values"]
    image_grid_thw = batch.modality_kwargs["image_grid_thw"]
    # One image per row (chosen, rejected), both from the SAME declared
    # file: two flattened-patch blocks, byte-identical to each other.
    assert image_grid_thw.shape[0] == 2
    expansion = processor.image_expansion
    block_0 = pixel_values[:expansion]
    block_1 = pixel_values[expansion : 2 * expansion]
    assert torch.equal(block_0, block_1), (
        "chosen and rejected rendered DIFFERENT pixel blocks for the same "
        "declared image; they must share the record's own media"
    )


def test_encode_batch_media_two_records_two_distinct_images(tmp_path: Path) -> None:
    processor = _FakeProcessor()
    trainer = _media_trainer(processor)
    image_a = _make_image(tmp_path, "a.png")
    image_b = _make_image(tmp_path, "b.png")
    batch = trainer._encode_batch(
        records=(
            _image_record(image_a, _line=1),
            _image_record(image_b, chosen="a bird", rejected="a fish", _line=2),
        ),
        tokenizer=None,
        pad_token_id=0,
        device="cpu",
    )
    assert batch.input_ids.shape[0] == 4  # 2 records x (chosen, rejected)
    assert batch.modality_kwargs is not None
    assert batch.modality_kwargs["image_grid_thw"].shape[0] == 4


def test_encode_batch_media_kto_builds_one_row_per_record(tmp_path: Path) -> None:
    processor = _FakeProcessor()
    trainer = _media_trainer(processor, algorithm="kto")
    image_path = _make_image(tmp_path, "photo.png")
    record = {
        "prompt": "<image> describe this",
        "completion": "a cat",
        "label": True,
        "image": image_path,
        "_line": 1,
    }
    batch = trainer._encode_batch(
        records=(record,),
        tokenizer=None,
        pad_token_id=0,
        device="cpu",
    )
    assert batch.input_ids.shape[0] == 1
    assert batch.desirable is not None
    assert bool(batch.desirable[0]) is True


def test_encode_batch_media_marker_mismatch_refuses(tmp_path: Path) -> None:
    """``normalize_conversation``'s own marker check is reached from here
    too -- no second, divergent validation was written for this path."""
    processor = _FakeProcessor()
    trainer = _media_trainer(processor)
    image_path = _make_image(tmp_path, "photo.png")
    record = _image_record(image_path, prompt="describe this")  # no <image> marker
    with pytest.raises(SystemExit) as excinfo:
        trainer._encode_batch(
            records=(record,),
            tokenizer=None,
            pad_token_id=0,
            device="cpu",
        )
    assert excinfo.value.code == 96


def test_encode_batch_media_overlong_row_refuses_without_truncating(tmp_path: Path) -> None:
    processor = _FakeProcessor()
    trainer = _media_trainer(processor, max_length=6)
    image_path = _make_image(tmp_path, "photo.png")
    record = _image_record(
        image_path,
        prompt="<image> describe this picture in great detail please",
        chosen="a very long chosen answer with many words in it",
        rejected="a dog",
    )
    with pytest.raises(TrainerRefusal, match="never truncated"):
        trainer._encode_batch(
            records=(record,),
            tokenizer=None,
            pad_token_id=0,
            device="cpu",
        )


def test_encode_batch_media_video_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Video rows render through the SAME pipeline, via ``frames_for_row``
    (monkeypatched here to avoid needing a real clip/decoder on CPU)."""
    processor = _FakeProcessor()
    trainer = _media_trainer(
        processor,
        image_column=None,
        video_column="video",
        video_frames=2,
    )
    frame_a = Image.new("RGB", (4, 4), color=(1, 2, 3))
    frame_b = Image.new("RGB", (4, 4), color=(4, 5, 6))

    def _fake_frames_for_row(data: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "frames": [frame_a, frame_b],
            "fps": 2.0,
            "frames_indices": [0, 1],
            "duration": 1.0,
        }

    monkeypatch.setattr(pt, "frames_for_row", _fake_frames_for_row)
    record = {
        "prompt": "<video> describe this clip",
        "chosen": "a cat runs",
        "rejected": "a dog runs",
        "video": "clip.mp4",
        "start": 0.0,
        "end": 1.0,
        "_line": 1,
    }
    batch = trainer._encode_batch(
        records=(record,),
        tokenizer=None,
        pad_token_id=0,
        device="cpu",
    )
    assert batch.input_ids.shape[0] == 2
    assert batch.modality_kwargs is not None
    assert "pixel_values_videos" in batch.modality_kwargs
    assert batch.modality_kwargs["video_grid_thw"].shape[0] == 2


def test_encode_batch_media_mixed_records_build_correct_per_row_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MEASURED regression (Qwen3.6-27B GPU proof): ONE step over an image
    record, a video record and a text-only record together must produce
    image_counts/video_counts that correctly mark which of the 6 OUTPUT rows
    (3 records x chosen/rejected) carry which modality, and a micro-batch
    slice spanning the image/video boundary must still reach the model
    (rather than raising the 'Video features and video tokens do not match'
    error narrowing by raw training-row range produced)."""
    processor = _FakeProcessor()
    trainer = _media_trainer(processor, image_column="image", video_column="video", video_frames=2)
    image_path = _make_image(tmp_path, "photo.png")
    frame_a = Image.new("RGB", (4, 4), color=(1, 2, 3))
    frame_b = Image.new("RGB", (4, 4), color=(4, 5, 6))

    def _fake_frames_for_row(data: Any, **kwargs: Any) -> dict[str, Any]:
        return {"frames": [frame_a, frame_b], "fps": 2.0, "frames_indices": [0, 1], "duration": 1.0}

    monkeypatch.setattr(pt, "frames_for_row", _fake_frames_for_row)
    records = (
        _image_record(image_path, _line=1),
        {
            "prompt": "<video> describe this clip",
            "chosen": "a cat runs",
            "rejected": "a dog runs",
            "video": "clip.mp4",
            "start": 0.0,
            "end": 1.0,
            "_line": 2,
        },
        {"prompt": "plain text question", "chosen": "a", "rejected": "b", "_line": 3},
    )
    batch = trainer._encode_batch(records=records, tokenizer=None, pad_token_id=0, device="cpu")

    assert batch.input_ids.shape[0] == 6  # 3 records x (chosen, rejected)
    assert batch.image_counts == (1, 1, 0, 0, 0, 0)
    assert batch.video_counts == (0, 0, 1, 1, 0, 0)

    # The slice [1, 3) crosses from the image record's rejected row into the
    # video record's chosen row -- exactly the heterogeneous boundary that
    # broke on GPU before image_counts/video_counts existed.
    model = _RecordingTinyLM()
    trainer._forward_logprobs(model=model, batch=batch, start=1, end=3)
    received = model.received[0]
    assert received["image_grid_thw"].shape[0] == 1  # row 1's own image only
    assert received["video_grid_thw"].shape[0] == 1  # row 2's own video only


# ---------------------------------------------------------------------------
# _forward_logprobs: reuse of rl.trainer._narrow_modality_kwargs_by_row
# ---------------------------------------------------------------------------


class _RecordingTinyLM(torch.nn.Module):
    """A toy causal LM that records every forward's extra (modality) kwargs."""

    def __init__(self, vocab: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.head = torch.nn.Linear(8, vocab)
        self.received: list[dict[str, Any]] = []

    def forward(self, input_ids: Any, attention_mask: Any = None, **kwargs: Any) -> Any:
        self.received.append(kwargs)
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def test_forward_logprobs_narrows_flattened_patches_per_row_range() -> None:
    """4 rows, one flattened-patch image (3 patches) each; slicing rows
    [1, 3) must hand the model exactly rows 1 and 2's own 3-patch blocks,
    not a raw [3:9] cut of the underlying tensor by row count."""
    trainer = _trainer()
    row_count, width, expansion = 4, 5, 3
    input_ids = torch.randint(3, 60, (row_count, width))
    attention_mask = torch.ones((row_count, width), dtype=torch.long)
    pixel_values = torch.arange(row_count * expansion * 2, dtype=torch.float32).reshape(
        row_count * expansion, 2
    )
    image_grid_thw = torch.tensor([[1, 1, expansion]] * row_count, dtype=torch.long)
    batch = pt._EncodedBatch(
        target_ids=input_ids[:, 1:],
        completion_mask=torch.ones((row_count, width - 1)),
        attention_mask=attention_mask,
        input_ids=input_ids,
        desirable=None,
        modality_kwargs={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
    )
    model = _RecordingTinyLM()
    trainer._forward_logprobs(model=model, batch=batch, start=1, end=3)

    assert len(model.received) == 1
    received = model.received[0]
    assert set(received) == {"pixel_values", "image_grid_thw"}
    assert received["image_grid_thw"].shape[0] == 2
    expected_block = pixel_values[expansion : 3 * expansion]
    assert torch.equal(received["pixel_values"], expected_block), (
        "narrowing rows [1, 3) must select rows 1 and 2's own 3-patch "
        "blocks (patches [3:9]), not a row-count-based [1:3] slice of the "
        "flattened patch tensor"
    )


def test_forward_logprobs_narrows_a_per_token_key_alongside_flattened_patches() -> None:
    """A per-TOKEN key (e.g. Qwen3VL's ``mm_token_type_ids``, shape
    (rows, width) -- one entry per column, NOT per image) must be narrowed
    by plain ROW slicing in the SAME call that block-narrows the flattened
    patches -- the two keys use DIFFERENT rules and must not be confused.
    Chosen and rejected rows of one pair are typically different widths
    (different completion lengths); this uses the GLOBAL batch width
    (already right-padded, as ``_encode_batch``/``_encode_batch_media``
    both produce), so the per-token key's own column count matches
    ``input_ids`` regardless of any one row's true completion length."""
    trainer = _trainer()
    row_count, width, expansion = 4, 6, 3
    input_ids = torch.randint(3, 60, (row_count, width))
    attention_mask = torch.ones((row_count, width), dtype=torch.long)
    pixel_values = torch.arange(row_count * expansion * 2, dtype=torch.float32).reshape(
        row_count * expansion, 2
    )
    image_grid_thw = torch.tensor([[1, 1, expansion]] * row_count, dtype=torch.long)
    # Per-token key: row r, column c -> value r * 100 + c, so a wrong
    # (shifted or row-mismatched) slice is detectable by exact value.
    mm_token_type_ids = torch.tensor(
        [[row * 100 + col for col in range(width)] for row in range(row_count)]
    )
    batch = pt._EncodedBatch(
        target_ids=input_ids[:, 1:],
        completion_mask=torch.ones((row_count, width - 1)),
        attention_mask=attention_mask,
        input_ids=input_ids,
        desirable=None,
        modality_kwargs={
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        },
    )
    model = _RecordingTinyLM()
    trainer._forward_logprobs(model=model, batch=batch, start=1, end=3)

    received = model.received[0]
    assert torch.equal(received["mm_token_type_ids"], mm_token_type_ids[1:3]), (
        "the per-token key must be narrowed by plain row range [1:3], "
        "independent of the flattened-patch block narrowing done for "
        "pixel_values/image_grid_thw in the SAME call"
    )
    assert received["image_grid_thw"].shape[0] == 2


def test_forward_logprobs_heterogeneous_batch_narrows_each_modality_by_its_own_rows() -> None:
    """MEASURED regression (Qwen3.6-27B, a mixed image+video+text pair
    file): image_grid_thw/video_grid_thw count MEDIA, not training rows --
    a 4-row batch where only row 0 carries an image and only row 2 carries
    a video must narrow each modality by ITS OWN media-row range (translated
    from image_counts/video_counts), never by the raw [start, end) training-
    row range directly. Before this fix, forwarding rows [0, 2) handed the
    model video_grid_thw sliced [0:2) even though video_grid_thw has only 1
    row total (row 2's video) -- out of alignment with input_ids, and the
    real failure this reproduces: 'Video features and video tokens do not
    match, tokens: 0, features: 2400' the first time a video row shared a
    micro-batch slice with a non-video row."""
    trainer = _trainer()
    row_count, width = 4, 5
    # row 0: image (3 patches). row 1: text only. row 2: video (2 patches).
    # row 3: image (3 patches).
    image_counts = (1, 0, 0, 1)
    video_counts = (0, 0, 1, 0)
    input_ids = torch.randint(3, 60, (row_count, width))
    attention_mask = torch.ones((row_count, width), dtype=torch.long)
    # image_grid_thw/pixel_values have ONE row per IMAGE (2 images total:
    # rows 0 and 3), not one row per training row.
    pixel_values = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2 * 3, 2)
    image_grid_thw = torch.tensor([[1, 1, 3], [1, 1, 3]], dtype=torch.long)
    # pixel_values_videos/video_grid_thw have ONE row (1 video total: row 2).
    pixel_values_videos = torch.arange(1 * 2 * 2, dtype=torch.float32).reshape(1 * 2, 2)
    video_grid_thw = torch.tensor([[1, 1, 2]], dtype=torch.long)
    batch = pt._EncodedBatch(
        target_ids=input_ids[:, 1:],
        completion_mask=torch.ones((row_count, width - 1)),
        attention_mask=attention_mask,
        input_ids=input_ids,
        desirable=None,
        modality_kwargs={
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        },
        image_counts=image_counts,
        video_counts=video_counts,
    )
    model = _RecordingTinyLM()

    # Rows [0, 2): row 0's image (image-index 0) only; no video at all --
    # the video keys must be OMITTED, not passed as empty tensors (MEASURED:
    # an empty pixel_values_videos/video_grid_thw pair still reaches the
    # vision tower and crashes on the degenerate input).
    trainer._forward_logprobs(model=model, batch=batch, start=0, end=2)
    received = model.received[0]
    assert torch.equal(received["image_grid_thw"], image_grid_thw[0:1])
    assert torch.equal(received["pixel_values"], pixel_values[0:3])
    assert "video_grid_thw" not in received
    assert "pixel_values_videos" not in received

    # Rows [2, 4): row 2's video (video-index 0) and row 3's image
    # (image-index 1, the SECOND image block) -- neither is image-index 0
    # or an empty video.
    trainer._forward_logprobs(model=model, batch=batch, start=2, end=4)
    received = model.received[1]
    assert torch.equal(received["video_grid_thw"], video_grid_thw[0:1])
    assert torch.equal(received["pixel_values_videos"], pixel_values_videos[0:2])
    assert torch.equal(received["image_grid_thw"], image_grid_thw[1:2])
    assert torch.equal(received["pixel_values"], pixel_values[3:6])

    # Row [1, 2) alone: the text-only row -- NEITHER modality present. This
    # is the exact micro_batch=1 shape that crashed on GPU (a lone row with
    # no media of either kind sharing a run with image/video-bearing rows).
    trainer._forward_logprobs(model=model, batch=batch, start=1, end=2)
    received = model.received[2]
    assert "image_grid_thw" not in received
    assert "pixel_values" not in received
    assert "video_grid_thw" not in received
    assert "pixel_values_videos" not in received


def test_forward_logprobs_classifies_per_media_keys_with_no_grid_thw_pair() -> None:
    """MEASURED regression (gemma-4-12B-it processor, 2026-10-11): gemma has
    NO image_grid_thw/video_grid_thw at all -- its own per-image/per-video
    keys are named image_position_ids/video_position_ids, each shaped
    (n_media, ...), i.e. one row per IMAGE/VIDEO, exactly like Qwen's
    image_grid_thw but under a completely different, unlisted name. Before
    shape-based classification, these fell through to the training-row
    'generic' branch and were narrowed by the WRONG axis the first time a
    text-only row shared a step with two image rows: 'RuntimeError: start
    (2) + length (1) exceeds dimension size (2)' (image_position_ids has 2
    rows -- one per image -- but row-index 2 was a valid TRAINING row).
    _classify_modality_key must recognise ANY key whose row-0 width matches
    the media total, not just two hardcoded names."""
    trainer = _trainer()
    row_count, width = 4, 5
    # row 0: image. row 1: image. row 2: text only. row 3: text only --
    # mirrors the exact row composition the real crash needed (two image
    # rows ahead of a text row, narrowing row index 2 or beyond).
    image_counts = (1, 1, 0, 0)
    input_ids = torch.randint(3, 60, (row_count, width))
    attention_mask = torch.ones((row_count, width), dtype=torch.long)
    # gemma-shaped: ONE ROW PER IMAGE, no paired count key at all.
    pixel_values = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    image_position_ids = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    batch = pt._EncodedBatch(
        target_ids=input_ids[:, 1:],
        completion_mask=torch.ones((row_count, width - 1)),
        attention_mask=attention_mask,
        input_ids=input_ids,
        desirable=None,
        modality_kwargs={"pixel_values": pixel_values, "image_position_ids": image_position_ids},
        image_counts=image_counts,
        video_counts=None,
    )
    model = _RecordingTinyLM()

    # Row [2, 3): the first TEXT-only row, training-row index 2 -- exactly
    # the index that overran a 2-row image tensor in the measured crash.
    trainer._forward_logprobs(model=model, batch=batch, start=2, end=3)
    assert "pixel_values" not in model.received[0]
    assert "image_position_ids" not in model.received[0]

    # Row [1, 2): row 1's own image (image-index 1, the SECOND row).
    trainer._forward_logprobs(model=model, batch=batch, start=1, end=2)
    received = model.received[1]
    assert torch.equal(received["pixel_values"], pixel_values[1:2])
    assert torch.equal(received["image_position_ids"], image_position_ids[1:2])


def test_forward_logprobs_passes_no_extra_kwargs_when_modality_kwargs_is_none() -> None:
    trainer = _trainer()
    input_ids = torch.randint(3, 60, (2, 4))
    batch = pt._EncodedBatch(
        target_ids=input_ids[:, 1:],
        completion_mask=torch.ones((2, 3)),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
        input_ids=input_ids,
        desirable=None,
    )
    model = _RecordingTinyLM()
    trainer._forward_logprobs(model=model, batch=batch, start=0, end=2)
    assert model.received == [{}]


# ---------------------------------------------------------------------------
# End to end on CPU: one real DPO step over an image-declared paired batch
# ---------------------------------------------------------------------------


class _TinyMultimodalLM(torch.nn.Module):
    """Ignores pixel_values/image_grid_thw (toy embed+head), but must ACCEPT
    them: this is what proves the plumbing reaches the model call at all."""

    def __init__(self, vocab: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.head = torch.nn.Linear(8, vocab)

    def forward(self, input_ids: Any, attention_mask: Any = None, **kwargs: Any) -> Any:
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def test_one_step_dpo_image_pair_step0_loss_is_ln2(tmp_path: Path) -> None:
    import math

    processor = _FakeProcessor()
    trainer = _media_trainer(processor, algorithm="dpo", max_length=64)
    image_path = _make_image(tmp_path, "photo.png")
    records = (_image_record(image_path),)

    model = _TinyMultimodalLM()
    probe = model.embed.weight
    trainer._probe_parameter = probe
    trainer._probe_initial = probe.detach().to(device="cpu", dtype=torch.float32, copy=True)

    report = trainer._one_step(
        step=0,
        records=records,
        policy_model=model,
        reference_model=copy.deepcopy(model),
        tokenizer=None,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device="cpu",
        pad_token_id=0,
        loss_fn=TensorPreferenceLoss(objective=trainer._objective),
    )

    assert report is not None
    assert report.rows == 1
    assert abs(report.loss.loss - math.log(2)) < 1e-4, (
        f"step-0 image-pair DPO loss {report.loss.loss!r} is not ln(2) "
        f"within 1e-4 on a policy==reference step"
    )
    assert trainer.history[0]["weight_delta_l2"] > 0.0, (
        "weight_delta_l2 is 0 after one optimiser step; the image-pair "
        "batch ran through the loss but the model did not move"
    )
