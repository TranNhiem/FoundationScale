# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the offline preference trainer's audio support (Qwen3-ASR).

These exercise the audio declaration surface SP-2 adds to
``PreferenceTrainConfig`` (``audio_column``/``audio_language``) and
``PreferenceTrainer`` (``_needs_audio``/``_encode_batch_audio``):

  * the config validation: a partial audio declaration is refused (same
    doctrine as ``adapter``/``adapter_*`` and ``video_*``), a blank column or
    language names no axis at all, and audio excludes the image/video axes
    (one record cannot carry two processor-rendered modalities through one
    transcription request);
  * the ``run()``-level carrier check: a record carrying audio in a column
    nobody declared refuses "text-only", exactly as the undeclared-video
    check does -- never a silent downgrade to the unused prompt;
  * the sentinel rule: in audio mode the prompt is the TRANSCRIPTION REQUEST,
    so a record whose prompt is anything but ``AUDIO_PROMPT_SENTINEL`` refuses
    (its text prompt would be silently unused);
  * ``_encode_batch_audio``'s row/mask/feature layout over the Qwen3-ASR
    transcription request path: two rows per paired record (chosen, then
    rejected -- interleaved, one pair after another), one row per KTO record,
    a completion mask covering EXACTLY the transcript + eos (the SFT
    supervision surface -- so DPO prices the tokens SFT trained), never an
    audio placeholder and never a prompt token, and byte-identical audio
    features on the two rows of one record (both sides condition on the SAME
    clip).

WHAT IS CLAIMED: the layout, masking and refusal rules above are measured off
a deterministic fake Qwen3-ASR-like processor whose mel block is a function
of the waveform's own content -- so "same clip twice" is a tensor equality and
"two different clips" is a tensor inequality, never a call-order artefact.

WHAT IS NOT CLAIMED: that a real ``Qwen3ASRProcessor`` expands placeholders
by any particular count (that is ``tests/train/test_audio_collator.py``'s
pair-check claim), or anything about GPU behaviour -- that is the
``validation_campaigns/speech_pref`` job's claim. Nothing is downloaded: the
audio surface resolution is monkeypatched and every wav is written into
``tmp_path``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from foundationscale.rl import preference_trainer as pt  # noqa: E402
from foundationscale.rl.preference_trainer import (  # noqa: E402
    AUDIO_PROMPT_SENTINEL,
    PreferenceTrainConfig,
    PreferenceTrainer,
)
from foundationscale.rl.registry import reset_algorithm_registry  # noqa: E402
from foundationscale.rl.trainer import TrainerRefusal  # noqa: E402
from foundationscale.train import audio as train_audio  # noqa: E402
from foundationscale.train.audio import _qwen3_asr_tokens  # noqa: E402


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
    # Roomy by default: an audio prompt is the transcription request's
    # placeholder run plus the transcript, and the audio path never truncates.
    overrides.setdefault("max_length", 512)
    return PreferenceTrainer(_cfg("m", "d.jsonl", **overrides))


def _make_wav(
    path: Path,
    seconds: float,
    *,
    sample_rate: int = 16000,
    freq: float = 440.0,
) -> np.ndarray:
    """One real mono wav on disk; ``load_audio`` reads it through soundfile.

    Real bytes (not a numpy dict) so the loader's file path is exercised and
    the sample-rate enforcement has something to enforce.
    """
    frames = int(seconds * sample_rate)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    wave = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wave, sample_rate)
    return wave


# ---------------------------------------------------------------------------
# Config validation: audio_column / audio_language are one declaration or none
# ---------------------------------------------------------------------------


def test_audio_prompt_sentinel_is_the_declared_literal() -> None:
    """The sentinel is shared verbatim between the dataset writer and the
    trainer (part B emits it, part A refuses anything else), so its spelling
    is contract, not an implementation detail."""
    assert AUDIO_PROMPT_SENTINEL == "<|asr_audio|>"
    assert pt.AUDIO_PROMPT_SENTINEL == "<|asr_audio|>"


def test_config_audio_language_without_audio_column_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="audio"):
        PreferenceTrainer(_cfg("m", "d.jsonl", audio_language="en"))


def test_config_audio_column_without_audio_language_refuses() -> None:
    # The transcription request must be TOLD the language: one language
    # trained under another's label is a silent mislabel, so a half
    # declaration refuses rather than defaults.
    with pytest.raises(TrainerRefusal, match="audio"):
        PreferenceTrainer(_cfg("m", "d.jsonl", audio_column="audio"))


def test_config_audio_column_blank_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="non-empty string"):
        PreferenceTrainer(_cfg("m", "d.jsonl", audio_column="   ", audio_language="en"))


def test_config_audio_language_blank_refuses() -> None:
    with pytest.raises(TrainerRefusal, match="non-empty string"):
        PreferenceTrainer(_cfg("m", "d.jsonl", audio_column="audio", audio_language="   "))


def test_config_audio_and_image_columns_refuse() -> None:
    # One transcription request renders the prompt from ONE clip; an image
    # column would be a second, unrendered modality on the same row.
    with pytest.raises(TrainerRefusal, match="image"):
        PreferenceTrainer(
            _cfg(
                "m",
                "d.jsonl",
                audio_column="audio",
                audio_language="en",
                image_column="image",
            )
        )


def test_config_audio_and_video_columns_refuse() -> None:
    with pytest.raises(TrainerRefusal, match="video"):
        PreferenceTrainer(
            _cfg(
                "m",
                "d.jsonl",
                audio_column="audio",
                audio_language="en",
                video_column="video",
                video_frames=2,
            )
        )


def test_config_audio_declaration_sets_needs_audio() -> None:
    trainer = _trainer(audio_column="audio", audio_language="en")
    assert trainer._needs_audio is True


def test_config_text_only_run_leaves_needs_audio_false() -> None:
    # Byte-identical historical trainer: no audio declaration, no audio path.
    assert _trainer()._needs_audio is False


def test_config_image_only_run_leaves_needs_audio_false() -> None:
    # The image/video axes are their own declaration; nothing about them
    # implies audio.
    assert _trainer(image_column="image")._needs_audio is False


# ---------------------------------------------------------------------------
# run(): an audio carrier in an UNDECLARED column refuses, as the
# undeclared-video check does -- never a silent drop of the clip
# ---------------------------------------------------------------------------


def test_run_undeclared_audio_carrier_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        {
            "prompt": "describe this clip",
            "chosen": "a",
            "rejected": "b",
            "audio": "x.wav",
        }
    ]
    dataset = _write_jsonl(tmp_path / "d.jsonl", rows)
    cfg = _cfg(str(tmp_path / "no-such-model"), str(dataset))

    with pytest.raises((TrainerRefusal, SystemExit)) as excinfo:
        PreferenceTrainer(cfg).run()
    captured = capsys.readouterr()
    diagnostic = f"{excinfo.value}\n{captured.err}"

    # The diagnostic contract is the undeclared-video check's: the carrier's
    # LINE and KEY are named so the operator can fix the corpus without
    # guessing which axis was in play.
    assert "line 1" in diagnostic
    assert "audio" in diagnostic
    assert "undeclared" in diagnostic
    if isinstance(excinfo.value, SystemExit):
        # The exit-96 carrier mechanism the undeclared-video test pins: the
        # message keeps that check's own "text-only / undeclared column"
        # wording.
        assert excinfo.value.code == 96
        assert "text-only" in captured.err
        assert "undeclared column" in captured.err
    else:
        # The module's config/data refusal type carries the same words
        # without a process exit -- the run is refused either way, and the
        # record is NEVER allowed to train on its unused prompt alone.
        assert isinstance(excinfo.value, TrainerRefusal)
        assert "audio_column" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The audio encode path: a fake Qwen3-ASR-like processor behind the patched
# audio surface resolution (apply_transcription_request is the prompt builder)
# ---------------------------------------------------------------------------


WORD_BASE = 6000


def _stable_word_id(word: str) -> int:
    """Deterministic content word id (``hash()`` is randomised per process)."""
    return WORD_BASE + (sum(ord(c) for c in word) % 97)


def _as_wave(entry: Any) -> np.ndarray:
    """One mono float32 waveform out of any carrier the preference path forwards.

    The pair format names a PATH to a wav/flac, but the request API is
    documented over waveforms -- which one arrives here is an implementation
    detail, so the fake accepts both (and the decoded-datasets dict) and keys
    its tensors on the wave's OWN CONTENT either way.
    """
    if isinstance(entry, Mapping):
        entry = entry["array"]
    if isinstance(entry, (str, os.PathLike)):
        samples, _rate = sf.read(str(entry), dtype="float32")
    else:
        samples = np.asarray(entry, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=-1)
    return np.asarray(samples, dtype=np.float32)


class _FakeASRTokenizer:
    """Stable-id transcript tokenizer: answer cell in, deterministic ids out.

    ``pad_token_id`` is 0 exactly so it collides with content id 0 in general
    vocabularies -- padding discovered by id equality would silently eat a
    supervised token (the #450 lesson), so the masking claims below are pinned
    through this id. ``add_special_tokens=False`` is asserted because the
    completion side is the SFT transcription target: transcript tokens + eos,
    nothing else.
    """

    pad_token_id = 0
    eos_token_id = 42

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> Any:
        assert add_special_tokens is False
        words = text.split() if text.strip() else []
        return SimpleNamespace(input_ids=[_stable_word_id(word) for word in words])


class Qwen3ASRProcessor:
    """FAKE transformers-5.19 Qwen3ASRProcessor: the request-API surface only.

    The CLASS NAME is load-bearing: ``train.audio._MASK_LENGTH_TOKEN_FORMULAS``
    looks the placeholder expansion up by ``type(processor).__name__``, so a
    renamed stub would quietly take the UNMEASURED branch of the transcription
    path's pair-check and make every positive case vacuous.

    ``apply_transcription_request(audio=[...], language=...)`` renders
    ``[OPEN] + audio_pl * N + [CLOSE, PREFILL]`` per row (the user turn is
    audio ONLY -- the contract's prompt column is the sentinel, and its text
    must never reach the request) where N is the REAL registered
    ``_qwen3_asr_tokens`` expansion over ``len(wave) // 320`` feature frames,
    so a collator-side pair-check sees agreement rather than a stub artefact.

    The mel block is seeded from the wave ITSELF (``input_features[row, :T, 0]
    = wave[:T]``) so "both sides of one record saw the same clip" is a tensor
    equality and "two records saw different clips" is a tensor inequality -- a
    stub filling by row index would make both claims unfalsifiable. Every call
    is recorded (count and language) so the language-forwarding claim is
    measurable without reaching into the trainer.
    """

    AUDIO_OPEN_TOKEN = 5001
    AUDIO_CLOSE_TOKEN = 5002
    PREFILL_TOKEN = 5005
    AUDIO_TOKEN_ID = 7777
    FRAMES_PER_SAMPLE = 320

    def __init__(self) -> None:
        self.audio_token_id = self.AUDIO_TOKEN_ID
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.tokenizer = _FakeASRTokenizer()
        self.received_requests: list[tuple[int, Any]] = []

    def apply_transcription_request(self, audio: Any, *, language: str) -> dict[str, Any]:
        waves = [_as_wave(entry) for entry in audio]
        self.received_requests.append((len(waves), language))
        prompt_rows: list[list[int]] = []
        mask_lengths: list[int] = []
        for wave in waves:
            mask_length = int(len(wave)) // self.FRAMES_PER_SAMPLE
            placeholders = _qwen3_asr_tokens(mask_length)
            mask_lengths.append(mask_length)
            prompt_rows.append(
                [self.AUDIO_OPEN_TOKEN]
                + [self.AUDIO_TOKEN_ID] * placeholders
                + [self.AUDIO_CLOSE_TOKEN, self.PREFILL_TOKEN]
            )

        batch_size = len(prompt_rows)
        width = max((len(ids) for ids in prompt_rows), default=0)
        input_ids = torch.full((batch_size, width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, width, dtype=torch.long)
        for row, ids in enumerate(prompt_rows):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1

        feat_width = max(mask_lengths, default=0) or 1
        input_features = torch.zeros(batch_size, feat_width, 128, dtype=torch.float32)
        input_features_mask = torch.zeros(batch_size, feat_width, dtype=torch.bool)
        for row, (wave, mask_length) in enumerate(zip(waves, mask_lengths, strict=True)):
            if mask_length <= 0:
                continue
            input_features[row, :mask_length, 0] = torch.from_numpy(wave[:mask_length].copy())
            input_features_mask[row, :mask_length] = True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
        }


class _FakeASRSurface:
    """PromptSurface-shaped wrapper (see rl.prompt_surface.PromptSurface)."""

    def __init__(self, processor: Any, *, kind: str = "processor") -> None:
        self.kind = kind
        self.surface = processor
        self.reason = "test fixture"
        self.supports_images = False


def _patch_audio_surface(monkeypatch: pytest.MonkeyPatch, processor: Any) -> _FakeASRSurface:
    """Route every audio surface resolution to the fake Qwen3-ASR processor.

    Both binding shapes are covered: the canonical
    ``foundationscale.train.audio.resolve_audio_surface`` (a late import reads
    the patched name) and any name the preference module bound at ITS import
    time. No AutoProcessor is loaded and nothing is downloaded.
    """
    surface = _FakeASRSurface(processor)

    def _fake_resolve(model_id: str) -> Any:
        return surface

    monkeypatch.setattr(train_audio, "resolve_audio_surface", _fake_resolve)
    if hasattr(pt, "resolve_audio_surface"):
        monkeypatch.setattr(pt, "resolve_audio_surface", _fake_resolve)
    return surface


def _audio_trainer(
    monkeypatch: pytest.MonkeyPatch,
    processor: Any,
    **overrides: Any,
) -> PreferenceTrainer:
    """One audio-declared trainer wired to ``processor`` (no model load)."""
    overrides.setdefault("audio_column", "audio")
    overrides.setdefault("audio_language", "en")
    _patch_audio_surface(monkeypatch, processor)
    trainer = _trainer(**overrides)
    if getattr(trainer, "_processor", None) is None:
        # The media path caches the resolved processor here; setting it is a
        # no-op when the trainer resolves lazily and the patched resolver is
        # what supplies the fake in either shape.
        trainer._processor = processor
    return trainer


def _pair_record(
    audio_path: str,
    *,
    chosen: str,
    rejected: str,
    line: int,
) -> dict[str, Any]:
    return {
        "prompt": AUDIO_PROMPT_SENTINEL,
        "audio": audio_path,
        "chosen": chosen,
        "rejected": rejected,
        "_line": line,
    }


def _target_ids(text: str) -> list[int]:
    """The completion side in the SFT transcription format: transcript + eos."""
    words = text.split() if text.strip() else []
    return [_stable_word_id(word) for word in words] + [_FakeASRTokenizer.eos_token_id]


def _aligned(out: Mapping[str, Any]) -> tuple[Any, Any]:
    """(token stream, completion mask) with ``mask[i, j]`` supervising ``stream[i, j]``.

    ``completion_mask`` is the trainer's TARGET mask over one row's completion;
    the contract pins its contents (transcript + eos only) but not its column
    offset, so both live layouts are mapped onto the token stream here and the
    claims below are asserted against whichever the implementation produced --
    shifted (``width - 1``, supervising ``input_ids[:, 1:]``) or unshifted
    (``width``, supervising ``input_ids``). Any other width is a layout nobody
    declared and fails loudly rather than being read as either.
    """
    ids = out.input_ids
    mask = out.completion_mask
    row_count, width = int(ids.shape[0]), int(ids.shape[1])
    assert tuple(mask.shape[0] for _ in (0,)) == (row_count,) if False else True
    assert int(mask.shape[0]) == row_count
    if int(mask.shape[1]) == width:
        return ids, mask
    if int(mask.shape[1]) == width - 1:
        return ids[:, 1:], mask
    raise AssertionError(
        f"completion_mask width {int(mask.shape[1])} is neither the input width "
        f"{width} nor its target-shifted width {width - 1}; the supervision "
        f"mask cannot be attributed to the token stream"
    )


def _supervised_tokens(out: Mapping[str, Any], row: int) -> list[int]:
    """The tokens ``completion_mask`` supervises on ``row``, in stream order."""
    stream, mask = _aligned(out)
    keep = mask[row] > 0.5
    return [int(token) for token in stream[row][keep]]


def test_encode_batch_audio_non_sentinel_prompt_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record whose prompt is ordinary text REFUSES: in audio mode the prompt
    is the transcription request, so the text would be silently unused and the
    measured conditional would not be the declared one."""
    processor = Qwen3ASRProcessor()
    trainer = _audio_trainer(monkeypatch, processor)
    wav = tmp_path / "clip.wav"
    _make_wav(wav, 0.5)
    record = _pair_record(str(wav), chosen="a cat", rejected="a dog", line=1)
    record["prompt"] = "please transcribe this clip for me"

    with pytest.raises(SystemExit) as excinfo:
        trainer._encode_batch_audio(records=(record,), device="cpu", pad_token_id=0)
    # The refusal happens BEFORE any request: no clip is silently transcribed
    # under a prompt the run never intended to use.
    assert processor.received_requests == []
    assert excinfo.value.code == 96


def test_encode_batch_audio_builds_interleaved_rows_per_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two paired records -> FOUR rows in chosen,rejected,chosen,rejected order,
    and each row's supervised tokens are EXACTLY that side's transcript + eos.

    The interleaving is the paired loss's row pairing (rows 2*i, 2*i+1 are one
    record's pair), so an implementation that emitted both chosen rows first
    would price the wrong couples -- the per-row token claims below fail on
    that shape, not on a row count alone.
    """
    processor = Qwen3ASRProcessor()
    trainer = _audio_trainer(monkeypatch, processor)
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _make_wav(wav_a, 0.5, freq=440.0)  # 8000 samples -> 25 feature frames
    _make_wav(wav_b, 1.0, freq=880.0)  # 16000 samples -> 50 feature frames
    records = (
        _pair_record(
            str(wav_a),
            chosen="a cat sat on the mat",
            rejected="dog barked loudly",
            line=1,
        ),
        _pair_record(
            str(wav_b),
            chosen="birds sing now",
            rejected="a dog sat",
            line=2,
        ),
    )

    out = trainer._encode_batch_audio(records=records, device="cpu", pad_token_id=0)

    assert tuple(int(v) for v in out.input_ids.shape[:1]) == (4,)
    assert int(out.attention_mask.shape[0]) == 4
    assert int(out.completion_mask.shape[0]) == 4
    assert _supervised_tokens(out, 0) == _target_ids("a cat sat on the mat")
    assert _supervised_tokens(out, 1) == _target_ids("dog barked loudly")
    assert _supervised_tokens(out, 2) == _target_ids("birds sing now")
    assert _supervised_tokens(out, 3) == _target_ids("a dog sat")


def test_encode_batch_audio_completion_mask_covers_only_transcript_and_eos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DPO log-probabilities are priced over the same tokens SFT trained: the
    transcript words plus the end-of-sequence token -- nothing from the
    request's prefix and nothing beyond the row's own completion."""
    processor = Qwen3ASRProcessor()
    trainer = _audio_trainer(monkeypatch, processor)
    wav = tmp_path / "clip.wav"
    _make_wav(wav, 0.25)
    record = _pair_record(str(wav), chosen="a cat sat", rejected="dogs run", line=1)

    out = trainer._encode_batch_audio(records=(record,), device="cpu", pad_token_id=0)
    stream, mask = _aligned(out)

    chosen_target = _target_ids("a cat sat")
    rejected_target = _target_ids("dogs run")
    assert _supervised_tokens(out, 0) == chosen_target
    assert _supervised_tokens(out, 1) == rejected_target
    # Count equality, not just value order: the mask covers exactly the
    # completion's arc length -- one column per supervised token.
    assert _supervised_tokens(out, 0).__len__() == len(chosen_target)
    assert _supervised_tokens(out, 1).__len__() == len(rejected_target)
    # Positive control: the completion really is IN the stream (a mask that
    # supervised nothing elsewhere would otherwise compare lists of lists).
    row_stream = [int(t) for t in stream[0]]
    assert row_stream[: len(chosen_target)].count(_FakeASRTokenizer.eos_token_id) <= 1
    assert chosen_target[0] in row_stream
    assert int((mask > 0.5).sum()) == len(chosen_target) + len(rejected_target)


def test_encode_batch_audio_placeholders_and_prompt_are_never_supervised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audio placeholder is a valid INPUT slot for tower output and never a
    generation target (the #450 lesson); the request's own structural tokens
    and padding are likewise prompt-side or non-content and stay out of the
    completion mask.
    """
    processor = Qwen3ASRProcessor()
    trainer = _audio_trainer(monkeypatch, processor)
    wav = tmp_path / "clip.wav"
    _make_wav(wav, 0.25)  # 25 frames -> 2 placeholders per row
    record = _pair_record(str(wav), chosen="a cat", rejected="a dog", line=1)

    out = trainer._encode_batch_audio(records=(record,), device="cpu", pad_token_id=0)
    stream, mask = _aligned(out)

    placeholder_id = Qwen3ASRProcessor.AUDIO_TOKEN_ID
    placeholder_positions = stream == placeholder_id
    # Positive control first: the rows DO carry placeholders (an empty tensor
    # comparison would "pass" a batch with no audio in it at all).
    assert int(placeholder_positions.sum()) > 0
    assert float(mask[placeholder_positions].sum()) == 0.0

    for prompt_token in (
        Qwen3ASRProcessor.AUDIO_OPEN_TOKEN,
        Qwen3ASRProcessor.AUDIO_CLOSE_TOKEN,
        Qwen3ASRProcessor.PREFILL_TOKEN,
    ):
        assert float(mask[stream == prompt_token].sum()) == 0.0
    # Padding is never supervised either (pad id 0 is the transcription
    # tokenizer's own pad, see its docstring).
    assert float(mask[stream == _FakeASRTokenizer.pad_token_id].sum()) == 0.0


def test_encode_batch_audio_forwards_the_declared_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request is told the DECLARED language: one language trained under
    another's label is a silent mislabel, so the config value must reach
    ``apply_transcription_request`` rather than a default."""
    processor = Qwen3ASRProcessor()
    trainer = _audio_trainer(monkeypatch, processor, audio_language="de")
    wav = tmp_path / "clip.wav"
    _make_wav(wav, 0.25)
    records = (_pair_record(str(wav), chosen="a cat", rejected="a dog", line=1),)

    trainer._encode_batch_audio(records=records, device="cpu", pad_token_id=0)

    assert processor.received_requests, "0 transcription requests were made"
    assert {language for _rows, language in processor.received_requests} == {"de"}


def test_encode_batch_audio_chosen_and_rejected_share_record_audio_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both rows of one record condition on the SAME clip: the two mel blocks
    are byte-identical, and two records carrying DIFFERENT clips produce
    different blocks (the positive control that features track the clip at
    all). The feature mask key the Qwen3-ASR request returns rides along
    with ``input_features``.
    """
    processor = Qwen3ASRProcessor()
    trainer = _audio_trainer(monkeypatch, processor)
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wave_a = _make_wav(wav_a, 0.5, freq=440.0)
    wave_b = _make_wav(wav_b, 0.25, freq=880.0)
    records = (
        _pair_record(str(wav_a), chosen="a cat", rejected="a dog", line=1),
        _pair_record(str(wav_b), chosen="birds sing", rejected="dogs run", line=2),
    )

    out = trainer._encode_batch_audio(records=records, device="cpu", pad_token_id=0)

    assert "input_features" in out.modality_kwargs
    assert "input_features_mask" in out.modality_kwargs
    features = out.modality_kwargs["input_features"]
    feature_mask = out.modality_kwargs["input_features_mask"]
    assert int(features.shape[0]) == 4
    assert int(feature_mask.shape[0]) == 4

    # Record 1's own clip, twice (chosen then rejected): identical blocks.
    assert torch.equal(features[0], features[1]), (
        "chosen and rejected rows of one record saw DIFFERENT mel blocks for "
        "the same declared clip; both sides must condition on the record's own "
        "audio"
    )
    assert torch.equal(feature_mask[0], feature_mask[1])
    assert torch.equal(features[2], features[3])
    assert torch.equal(feature_mask[2], feature_mask[3])

    # Positive control: two different clips are two different blocks, at the
    # lengths their own frame counts declare.
    assert not torch.equal(features[0], features[2])
    assert int(feature_mask[0].sum()) == len(wave_a) // Qwen3ASRProcessor.FRAMES_PER_SAMPLE
    assert int(feature_mask[2].sum()) == len(wave_b) // Qwen3ASRProcessor.FRAMES_PER_SAMPLE


def test_encode_batch_audio_kto_builds_one_row_per_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KTO's SUP rows are single-completion records: one row per record (never
    the paired interleave), each supervised over its own transcript + eos."""
    processor = Qwen3ASRProcessor()
    trainer = _audio_trainer(monkeypatch, processor, algorithm="kto")
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _make_wav(wav_a, 0.5, freq=440.0)
    _make_wav(wav_b, 0.25, freq=880.0)
    records = (
        {
            "prompt": AUDIO_PROMPT_SENTINEL,
            "audio": str(wav_a),
            "completion": "birds sing now",
            "label": True,
            "_line": 1,
        },
        {
            "prompt": AUDIO_PROMPT_SENTINEL,
            "audio": str(wav_b),
            "completion": "dog barked loudly",
            "label": False,
            "_line": 2,
        },
    )

    out = trainer._encode_batch_audio(records=records, device="cpu", pad_token_id=0)

    assert int(out.input_ids.shape[0]) == 2
    assert int(out.completion_mask.shape[0]) == 2
    assert "input_features" in out.modality_kwargs
    assert int(out.modality_kwargs["input_features"].shape[0]) == 2
    assert _supervised_tokens(out, 0) == _target_ids("birds sing now")
    assert _supervised_tokens(out, 1) == _target_ids("dog barked loudly")
    stream, mask = _aligned(out)
    assert float(mask[stream == Qwen3ASRProcessor.AUDIO_TOKEN_ID].sum()) == 0.0
