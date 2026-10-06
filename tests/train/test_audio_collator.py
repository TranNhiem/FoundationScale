"""The TRAIN plane's audio carrier: the audio collator and the feature-drop probe (#P1-SLICE2).

WHY THIS MODULE EXISTS. The pixel-side collator (#410/#450, see
tests/rl/test_prompt_surface_train_images.py) taught that a declared modality
column can travel from the dataset to the forward and disappear without a
single signal. This module applies the same rigour to audio: a run declaring
FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN must never see its `input_features` silently
swapped for a text-only batch, no silently truncated waveform, and no silently
unmasked placeholder in the labels.

WHY EACH LEG IS SHAPED THIS WAY. The stub processor emits `input_features`
ONLY when apply_chat_template receives an audio block -- a stub that always
emits the key would pass on a collator that drops the audio content, which is
exactly the defect under test. The stub's OWN `_compute_audio_num_tokens`
disagrees with the ACTUAL placeholder count when
`placeholder_prediction_error != 0`, so the MUST_FIRE pair-check leg fails on
the #P1-SLICE2 defect's shape and not on a stub artefact. The real tiny wav
files come from `soundfile` in `tmp_path` so `load_audio` runs through its real
`_read_soundfile` -> `sf.read` path -- real bytes on disk, real sample-rate
enforcement, real AudioLoadError reasons.

torch is installed in CI and is used here (the FAKE needs to emit real tensors
the collator can clone / mask / slice -- a fake sub-tensor that hides the clone
would also hide a collator that aliases input_ids). Nothing is skipped.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from foundationscale.train.audio import (
    AUDIO_FEATURES_KEY,
    AudioLoadError,
    load_audio,
    max_audio_seconds,
    refuse_if_audio_features_dropped,
    train_audio_collator_or_refuse,
)


class _FakeAudioProcessor:
    """FAKE gemma-4-shaped processor with a falsifiable contract built in.

    * `apply_chat_template` emits `input_features` ONLY for rows that carry an
      audio block. A stub that always emits the key makes every MUST_PASS leg
      pass on a collator that drops the audio content -- the exact defect under
      test -- so the emission is conditional here and the audio block's
      presence is recorded for the legs to assert on. `add_generation_prompt`
      is honoured exactly as a chat-template must: on a user-only conversation
      it APPENDS an assistant-turn opener, which is what the collator's second
      call uses to find the prompt/answer boundary.
    * `_compute_audio_num_tokens(S)` returns `S // 320 + placeholder_prediction_error`.
      When `placeholder_prediction_error != 0` it DISAGREES with the actual
      number of placeholders that `apply_chat_template` inserted (which always
      uses `S // 320`), so the pair-check MUST_FIRE leg genuinely fires and not
      on a stub artefact. `include_compute=False` builds a processor with NO
      `_compute_audio_num_tokens` at all (method genuinely absent, so the
      getattr-is-None skip branch is exercised) rather than setting it to None.
    * token ids are DETERMINISTIC across runs and processes: a word maps to
      `WORD_BASE + sum(ord(c) for c in word) % 97`, so a leg can compute the
      expected token id in the test and compare. Python's `hash()` is NOT
      usable here (randomised per process).
    """

    AUDIO_OPEN_TOKEN = 5001
    AUDIO_CLOSE_TOKEN = 5002
    USER_CLOSE_TOKEN = 5003
    ANSWER_OPEN_TOKEN = 5005
    ANSWER_EOS_TOKEN = 5004
    WORD_BASE = 6000
    AUDIO_TOKEN_ID = 7777
    SAMPLES_PER_TOKEN = 320

    def __init__(
        self,
        *,
        placeholder_prediction_error: int = 0,
        include_compute: bool = True,
    ) -> None:
        self.audio_token_id = self.AUDIO_TOKEN_ID
        self.audio_token = "<|audio|>"
        self.boa_token = "<|audio_start|>"
        self.eoa_token = "<|audio_end|>"
        self.audio_seq_length = 750
        self.audio_ms_per_token = 40
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.placeholder_prediction_error = placeholder_prediction_error
        self.received_convos: list[Any] = []
        self.include_input_features = True
        if include_compute:
            # Bound at instance construction so the method is genuinely present
            # on the instance when called and genuinely absent when
            # include_compute=False (deliberately not `= None`, so an
            # `is None` guard in the collator would NOT skip the check).
            self._compute_audio_num_tokens = self._actual_compute

    def _actual_compute(self, audio_waveform: Any, sampling_rate: int) -> int:
        # Same signature as transformers 5.5.0's Gemma4Processor helper.
        assert sampling_rate == self.feature_extractor.sampling_rate
        return len(audio_waveform) // self.SAMPLES_PER_TOKEN + self.placeholder_prediction_error

    @classmethod
    def _stable_word_id(cls, word: str) -> int:
        return cls.WORD_BASE + (sum(ord(c) for c in word) % 97)

    def apply_chat_template(
        self,
        messages: Any,
        *,
        tokenize: bool,
        return_dict: bool,
        return_tensors: str,
        padding: bool,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert tokenize is True
        assert return_dict is True
        assert return_tensors == "pt"
        assert padding is True
        self.received_convos.append(messages)
        convos = messages
        per_row_ids: list[list[int]] = []
        per_row_audio_totals: list[int] = []
        for convo in convos:
            ids: list[int] = []
            audio_total = 0
            for msg in convo:
                role = msg["role"]
                if role == "assistant":
                    ids.append(self.ANSWER_OPEN_TOKEN)
                for block in msg["content"]:
                    if block["type"] == "audio":
                        wave = block["audio"]
                        # ACTUAL placeholder count, derived from the wave alone.
                        # NEVER from _compute_audio_num_tokens: the two must be
                        # able to disagree for the pair-check leg to be real.
                        actual = int(len(wave) // self.SAMPLES_PER_TOKEN)
                        ids.append(self.AUDIO_OPEN_TOKEN)
                        ids.extend([self.AUDIO_TOKEN_ID] * actual)
                        ids.append(self.AUDIO_CLOSE_TOKEN)
                        audio_total += actual
                    elif block["type"] == "text":
                        text = block["text"].strip()
                        if text:
                            ids.extend(self._stable_word_id(word) for word in text.split())
                if role == "user":
                    ids.append(self.USER_CLOSE_TOKEN)
                elif role == "assistant":
                    ids.append(self.ANSWER_EOS_TOKEN)
            if add_generation_prompt and convo and convo[-1]["role"] == "user":
                # The chat-template contract: append the assistant-turn opener
                # AFTER a trailing user turn when asked to generate. This is
                # exactly how the single-call alternative would render the
                # prompt-only sequence, so the prompt boundary inferred from
                # this call collides EXACTLY with the boundary in the full
                # rendering (which appends the ANSWER_OPEN_TOKEN as part of the
                # assistant message's own render).
                ids.append(self.ANSWER_OPEN_TOKEN)
            per_row_ids.append(ids)
            per_row_audio_totals.append(audio_total)

        batch_size = len(per_row_ids)
        max_tokens = max((len(ids) for ids in per_row_ids), default=0)
        input_ids = torch.zeros(batch_size, max_tokens, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.long)
        for index, ids in enumerate(per_row_ids):
            if ids:
                input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[index, : len(ids)] = 1

        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.include_input_features:
            # FALSIFIABILITY: emitted ONLY when include_input_features is True,
            # so a leg can build a fake that drops the feature key and drive
            # the drop probe's MUST_FIRE case from the collator's own output.
            feat_tokens = max(per_row_audio_totals, default=0) or 1
            input_features = torch.zeros(batch_size, feat_tokens, 128, dtype=torch.float32)
            input_features_mask = torch.zeros(batch_size, feat_tokens, dtype=torch.bool)
            for index, total in enumerate(per_row_audio_totals):
                input_features_mask[index, :total] = True
            out["input_features"] = input_features
            out["input_features_mask"] = input_features_mask
        return out


class _FakeSurface:
    """PromptSurface-shaped wrapper (see rl.prompt_surface.PromptSurface)."""

    def __init__(self, processor: Any, *, kind: str = "processor") -> None:
        self.kind = kind
        self.surface = processor
        self.reason = "test fixture"
        self.supports_images = False


def _make_wav(path: Path, seconds: float, sample_rate: int = 16000) -> np.ndarray:
    frames = int(seconds * sample_rate)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    wave = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(path), wave, sample_rate)
    return wave


def _stable_word_id(word: str) -> int:
    return _FakeAudioProcessor._stable_word_id(word)


def _prompt_len(wave_samples: int, user_text: str) -> int:
    """Prompt length in tokens per the fake's rendering.

    user-only with add_generation_prompt=True renders:
        [AUDIO_OPEN, audio_pl x N, AUDIO_CLOSE,
         word(user), word(prompt)..., USER_CLOSE, ANSWER_OPEN]
    where N = wave_samples // SAMPLES_PER_TOKEN. Length = 4 + N + word_count.
    """
    n_audio = wave_samples // _FakeAudioProcessor.SAMPLES_PER_TOKEN
    words = user_text.strip().split() if user_text.strip() else []
    return 4 + n_audio + len(words)


def _answer_len(answer_text: str) -> int:
    """Assistant answer length in tokens: per-word tokens + ANSWER_EOS.

    The ANSWER_OPEN_TOKEN sits at the end of the prompt (it opens the assistant
    turn) and is masked; only the answer words and the stop token are
    supervised at the tail.
    """
    words = answer_text.strip().split() if answer_text.strip() else []
    return 1 + len(words)


# ---------------------------------------------------------------------------
# MUST_PASS: one batch end to end. Keys carry input_features; labels mask
# prompt + audio placeholders + pad and keep the assistant target; the
# coverage tracker closes at COVERED.
# ---------------------------------------------------------------------------


def test_one_batch_keeps_input_features_and_masks_prompt_audio_and_pad(
    tmp_path: Path,
) -> None:
    processor = _FakeAudioProcessor()
    collate = train_audio_collator_or_refuse(
        _FakeSurface(processor), audio_column="audio", max_length=256
    )

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wave_a = _make_wav(wav_a, 0.5)  # 8000 samples -> 8000 // 320 = 25 audio_pl
    wave_b = _make_wav(wav_b, 1.0)  # 16000 samples -> 50 audio_pl

    rows = [
        {
            "audio": str(wav_a),
            "text": "prompt one",
            "answer": "target one here",
        },
        {
            "audio": str(wav_b),
            "text": "prompt two",
            "answer": "target two",
        },
    ]
    batch = collate(rows)

    # The KEYS contract: input_features must be in the batch dict the model
    # would actually receive. Must be a dict-batch carrying the audio tower's
    # mel features -- not a text-only encode sneaking through.
    assert "input_features" in batch
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    # Coverage closed over the two rows. rows_expected grew with rows seen;
    # rows_checked is 2; verdict COVERED.
    assert collate.coverage.rows_checked == 2
    assert collate.coverage.rows_expected == 2
    assert collate.coverage.verdict() == "COVERED"

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    p_a = _prompt_len(len(wave_a), "prompt one")
    a_a = _answer_len("target one here")
    p_b = _prompt_len(len(wave_b), "prompt two")
    a_b = _answer_len("target two")

    real_a = p_a + a_a
    real_b = p_b + a_b
    seq_len = max(real_a, real_b)
    assert input_ids.shape == (2, seq_len)
    assert attention_mask.shape == (2, seq_len)
    assert labels.shape == (2, seq_len)

    # --- Prompt: -100 from the first-attended slot to prompt_len -------------
    assert torch.all(labels[0, :p_a] == -100)
    assert torch.all(labels[1, :p_b] == -100)

    # --- Audio placeholders across the WHOLE row: -100 ---------------------
    assert not torch.any(labels == _FakeAudioProcessor.AUDIO_TOKEN_ID)

    # --- Padding: -100 wherever attention_mask == 0 -------------------------
    assert torch.all(labels[attention_mask == 0] == -100)
    if seq_len > real_a:
        assert torch.all(attention_mask[0, real_a:] == 0)
        assert torch.all(labels[0, real_a:] == -100)

    # --- Assistant answer: survives supervised -----------------------------
    assert torch.all(labels[0, p_a : p_a + a_a] == input_ids[0, p_a : p_a + a_a])
    assert torch.all(labels[1, p_b : p_b + a_b] == input_ids[1, p_b : p_b + a_b])
    assert not torch.any(labels[0, p_a : p_a + a_a] == -100)
    assert not torch.any(labels[1, p_b : p_b + a_b] == -100)

    # Positive control: the answer words ARE the expected stable ids -- so
    # this leg genuinely reads the assistant cell and not a duplicated one.
    expected_a_tail = [_stable_word_id(w) for w in ["target", "one", "here"]]
    expected_a_tail.append(_FakeAudioProcessor.ANSWER_EOS_TOKEN)
    assert input_ids[0, p_a : p_a + a_a].tolist() == expected_a_tail


def test_text_fields_override_reads_the_right_cells(tmp_path: Path) -> None:
    processor = _FakeAudioProcessor()
    collate = train_audio_collator_or_refuse(
        _FakeSurface(processor),
        audio_column="audio",
        max_length=256,
        text_fields=("prompt", "target"),
    )
    wav = tmp_path / "x.wav"
    wave = _make_wav(wav, 0.5)
    rows = [
        {
            "audio": str(wav),
            "prompt": "prompt one",
            "target": "target one here",
            # decoy fields defaulting to the DEFAULT text_fields names -- if
            # the override were ignored, these would land in the batch instead.
            "text": "DECOY USER PROMPT WITH MORE WORDS",
            "answer": "DECOY ANSWER WITH MORE WORDS THAN THE REAL ONE",
        }
    ]
    batch = collate(rows)

    p = _prompt_len(len(wave), "prompt one")
    a = _answer_len("target one here")
    expected_tail = [_stable_word_id(w) for w in ["target", "one", "here"]]
    expected_tail.append(_FakeAudioProcessor.ANSWER_EOS_TOKEN)
    # Positive control: the tail MATCHES the overridden "target" cell and NOT
    # the default "answer" cell (whose token length alone would differ).
    assert batch["input_ids"][0, p : p + a].tolist() == expected_tail


# ---------------------------------------------------------------------------
# MUST_FIRE: one refusal per defect class. Each leg asserts exit code 96 and
# the exact identifiers the refusal message must carry (row id, reason,
# column-name) so the operator can act without guessing.


# ---------------------------------------------------------------------------


def test_missing_wav_is_refused_naming_row_id_reason_and_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    processor = _FakeAudioProcessor()
    collate = train_audio_collator_or_refuse(
        _FakeSurface(processor), audio_column="audio", max_length=256
    )
    # NO monkeypatch here on purpose: load_audio goes through the real
    # soundfile.read path, which raises on a missing file and is wrapped as
    # reason="unreadable". A silent drop would be #371 one layer down.
    rows = [
        {
            "audio": str(tmp_path / "nope.wav"),
            "text": "prompt one",
            "answer": "target one here",
        }
    ]
    with pytest.raises(SystemExit) as excinfo:
        collate(rows)
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "train-row[0]" in err
    assert "unreadable" in err
    assert "'audio'" in err


def test_sample_rate_mismatch_is_refused_naming_row_id_reason_and_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    processor = _FakeAudioProcessor()
    collate = train_audio_collator_or_refuse(
        _FakeSurface(processor), audio_column="audio", max_length=256
    )
    wav_path = tmp_path / "wrong_rate.wav"
    # 22.05 kHz against the fake's 16 kHz feature-extractor rate; load_audio
    # refuses "sample_rate_mismatch" -- it deliberately does NOT resample (a
    # silently resampled row's seconds, placeholder counts and spectrum all
    # move while the row looks untouched).
    _make_wav(wav_path, 0.5, sample_rate=22050)
    rows = [
        {
            "audio": str(wav_path),
            "text": "prompt one",
            "answer": "target one here",
        }
    ]
    with pytest.raises(SystemExit) as excinfo:
        collate(rows)
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "train-row[0]" in err
    assert "sample_rate_mismatch" in err
    assert "'audio'" in err


def test_placeholder_count_mismatch_refuses_naming_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # FAILING INPUT: processor._compute_audio_num_tokens disagrees with the
    # ACTUAL placeholder count apply_chat_template inserted (off by +1). P0
    # measured these equal on gemma-4 (147/121/312/248 on both sides); a
    # disagreement means the processor/model pair is out of contract and the
    # batch cannot be trusted to line up with the audio tower's expected
    # output length. Refusing 96 names the row and both numbers.
    processor = _FakeAudioProcessor(placeholder_prediction_error=1)
    collate = train_audio_collator_or_refuse(
        _FakeSurface(processor), audio_column="audio", max_length=256
    )
    wav = tmp_path / "a.wav"
    _make_wav(wav, 0.5)  # 8000 samples -> actual 25 placeholders; expected 26.
    rows = [
        {
            "audio": str(wav),
            "text": "prompt one",
            "answer": "target one here",
        }
    ]
    with pytest.raises(SystemExit) as excinfo:
        collate(rows)
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "train-row[0]" in err
    assert "_compute_audio_num_tokens" in err
    assert "25" in err  # ACTUAL placeholder count
    assert "26" in err  # EXPECTED placeholder count


def test_no_compute_method_means_the_pair_check_is_skipped_not_invented(
    tmp_path: Path,
) -> None:
    processor = _FakeAudioProcessor(include_compute=False)
    # Absent, not merely None: the getattr fallback branch is what runs here.
    assert not hasattr(processor, "_compute_audio_num_tokens")
    collate = train_audio_collator_or_refuse(
        _FakeSurface(processor), audio_column="audio", max_length=256
    )
    wav = tmp_path / "a.wav"
    _make_wav(wav, 0.5)
    rows = [
        {
            "audio": str(wav),
            "text": "prompt one",
            "answer": "target one here",
        }
    ]
    batch = collate(rows)
    # The batch is accepted; the pair check is reported neither as pass nor
    # as run -- same UNMEASURED doctrine as max_audio_seconds returning None.
    assert "input_features" in batch
    assert collate.coverage.rows_checked == 1


def test_batch_wider_than_max_length_refuses_never_truncates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # FAILING INPUT: a 0.5 s wav + a 2-word prompt + a 3-word answer render
    # 25 + 4 + 2 + 4 = 35 tokens; a max_length=5 bound is unmeasurably small.
    # The audio path DELIBERATELY does not truncate: truncation would drop
    # measured sound between the loader and the forward (the silent-drop
    # defect). Refuse 96 naming BOTH numbers.
    processor = _FakeAudioProcessor()
    collate = train_audio_collator_or_refuse(
        _FakeSurface(processor), audio_column="audio", max_length=5
    )
    wav = tmp_path / "a.wav"
    _make_wav(wav, 0.5)
    rows = [
        {
            "audio": str(wav),
            "text": "prompt one",
            "answer": "target one here",
        }
    ]
    with pytest.raises(SystemExit) as excinfo:
        collate(rows)
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "'audio'" in err
    assert "5" in err  # max_length
    assert "35" in err  # actual encoded width


def test_refuse_if_audio_features_dropped_returns_none_when_the_key_is_present() -> None:
    # The complementary leg: without it the probe could refuse EVERY batch
    # and still pass -- the vacuous-gate shape this repo keeps finding.
    result = refuse_if_audio_features_dropped(
        {"input_ids": [1], "attention_mask": [1], AUDIO_FEATURES_KEY: [1]},
        audio_column="audio",
    )
    assert result is None


def test_refuse_if_audio_features_dropped_fires_when_the_key_is_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: a collator whose output has text keys only. The refusal
    # must name BOTH the declared audio column and the dropped feature key, or
    # the operator cannot tell which axis vanished.
    with pytest.raises(SystemExit) as excinfo:
        refuse_if_audio_features_dropped(
            {"input_ids": [1], "attention_mask": [1]},
            audio_column="audio",
        )
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "'audio'" in err
    assert AUDIO_FEATURES_KEY in err
    assert "DROPPED" in err


def test_non_string_keys_are_normalised_before_the_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The refusal message does sorted(keys); without str() normalisation that
    # sort raises TypeError on mixed types and the probe crashes INSTEAD of
    # refusing -- the drop it exists to report would be masked by its own
    # report.
    mixed_present = {1: "a", AUDIO_FEATURES_KEY: "b"}
    assert refuse_if_audio_features_dropped(mixed_present.keys(), audio_column="audio") is None
    with pytest.raises(SystemExit) as excinfo:
        refuse_if_audio_features_dropped({1: "a", "input_ids": "b"}.keys(), audio_column="audio")
    assert excinfo.value.code == 96
    assert AUDIO_FEATURES_KEY in capsys.readouterr().err


def test_the_feature_key_override_is_honoured_in_both_directions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An override that is accepted but ignored would refuse (or pass) against
    # the wrong spelling. Both directions are pinned: the custom key satisfies
    # the probe AND the DEFAULT key no longer does.
    result = refuse_if_audio_features_dropped(
        {"mels": [1]}, audio_column="audio", feature_key="mels"
    )
    assert result is None
    with pytest.raises(SystemExit) as excinfo:
        refuse_if_audio_features_dropped(
            {AUDIO_FEATURES_KEY: [1]}, audio_column="audio", feature_key="mels"
        )
    assert excinfo.value.code == 96
    assert "'mels'" in capsys.readouterr().err


def test_a_collator_thats_seen_no_rows_reports_vacuous_coverage(tmp_path: Path) -> None:
    # Complementary leg: a zero-row batch must NOT be a pass. AudioCoverage
    # treats 0 rows checked as VACUOUS regardless of what else accumulated --
    # a manifest proving nothing about the audio it saw is not a manifest.
    processor = _FakeAudioProcessor()
    collate = train_audio_collator_or_refuse(
        _FakeSurface(processor), audio_column="audio", max_length=256
    )
    batch = collate([])
    assert batch == {}
    assert collate.coverage.rows_checked == 0
    assert collate.coverage.verdict() == "VACUOUS"


def test_audio_load_error_carries_the_reason_and_row_id_directly() -> None:
    # Pins that the AudioLoadError the collator wraps names its row and reason
    # -- the two identifiers the exit-96 refusal message must print for the
    # operator to act. Drives the class directly because the collator is what
    # catches it and refuses; the surface it exposes for the refusal wording
    # is exactly these two attributes.
    err = AudioLoadError("row-xyz", "too_long")
    assert err.row_id == "row-xyz"
    assert err.reason == "too_long"
    assert "too_long" in str(err)
    assert "row-xyz" in str(err)


def test_load_audio_directly_reports_the_sample_rate_mismatch_for_the_tests_row(
    tmp_path: Path,
) -> None:
    # Direct companion to the collator's MUST_FIRE leg: confirms the loader's
    # own error surfaces the reason the collator must forward. A regression in
    # the loader would otherwise be masked by the collator's exit-96 wrapper
    # and both halves of the pair would shift together.
    wav = tmp_path / "wrong_rate.wav"
    _make_wav(wav, 0.5, sample_rate=22050)
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(str(wav), target_sr=16000, row_id="row-abc", max_seconds=30.0)
    assert excinfo.value.row_id == "row-abc"
    assert excinfo.value.reason == "sample_rate_mismatch"


def test_max_audio_seconds_reads_the_caps_off_the_processor() -> None:
    # Companion contract: the collator calls max_audio_seconds(processor) for
    # the row's time bound, so its formula must not drift from the processor's
    # declaration (750 * 40 ms = 30.0 s for the fake's gemma-4-shaped spec).
    processor = _FakeAudioProcessor()
    assert max_audio_seconds(processor) == pytest.approx(30.0)
    # And the UNMEASURED arm: when either attribute is unreadable, the return
    # is None -- UNMEASURED, not unlimited-as-measured.
    assert max_audio_seconds(SimpleNamespace()) is None
