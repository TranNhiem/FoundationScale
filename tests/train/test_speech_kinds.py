"""tests/train/test_speech_kinds.py"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from foundationscale.train.speech_kinds import (
    SPEECH_KINDS,
    WHISPER_WINDOW_SECONDS,
    auto_class_name,
    load_speech_model,
    speech_model_kind,
    speech_support_refusal,
    train_ctc_collator_or_refuse,
    train_seq2seq_collator_or_refuse,
)

# ---------------------------------------------------------------------------
# kind detection table + architectures fallback (never guess)
# ---------------------------------------------------------------------------


def test_speech_model_kind_table_maps_every_declared_family() -> None:
    """MUST_PASS: every table row resolves to a kind in SPEECH_KINDS and the closed
    table is exhaustive here."""
    cases = {
        "whisper": "seq2seq",
        "parakeet_ctc": "ctc",
        "parakeet": "ctc",
        "wav2vec2": "ctc",
        "hubert": "ctc",
        "gemma4": "audio_llm",
        "qwen2_audio": "audio_llm",
    }
    for model_type, expected in cases.items():
        assert speech_model_kind(SimpleNamespace(model_type=model_type)) == expected
        assert expected in SPEECH_KINDS


def test_speech_model_kind_architectures_fallback_settles_ctc_and_whisper_seq2seq() -> None:
    """MUST_PASS: when model_type is absent/unrecognised the architectures list decides
    ForCTC and Whisper cond-gen; unknown names give None (never guess)."""
    assert (
        speech_model_kind(SimpleNamespace(model_type="unknown", architectures=["ParakeetForCTC"]))
        == "ctc"
    )
    assert (
        speech_model_kind(SimpleNamespace(model_type="unknown", architectures=["X", "YForCTC"]))
        == "ctc"
    )
    assert (
        speech_model_kind(
            SimpleNamespace(model_type="unknown", architectures=["WhisperForConditionalGeneration"])
        )
        == "seq2seq"
    )
    # Gemma4ForConditionalGeneration alone does NOT settle a kind (audio_llm is
    # reached via model_type).
    assert (
        speech_model_kind(
            SimpleNamespace(model_type="unknown", architectures=["Gemma4ForConditionalGeneration"])
        )
        is None
    )
    assert speech_model_kind(SimpleNamespace(architectures=["SomethingForCausalLM"])) is None


def test_speech_model_kind_none_when_no_declared_geometry() -> None:
    """MUST_PASS: a config whose own declarations name no geometry returns None so the
    CALLER refuses (never guess)."""
    assert speech_model_kind(SimpleNamespace()) is None
    assert speech_model_kind(SimpleNamespace(model_type="llama", architectures=None)) is None
    assert (
        speech_model_kind(SimpleNamespace(model_type="gpt2", architectures=["GPT2LMHeadModel"]))
        is None
    )


def test_speech_model_kind_table_outranks_architectures_artefact() -> None:
    """MUST_PASS: when the two axes disagree the model_type table wins (declaration
    outranks a saved class name)."""
    assert (
        speech_model_kind(SimpleNamespace(model_type="gemma4", architectures=["ParakeetForCTC"]))
        == "audio_llm"
    )
    assert (
        speech_model_kind(
            SimpleNamespace(
                model_type="wav2vec2", architectures=["WhisperForConditionalGeneration"]
            )
        )
        == "ctc"
    )


def test_auto_class_name_by_kind_and_raises_outside() -> None:
    """MUST_PASS: each of the 3 kinds has its Auto class name; anything else raises
    ValueError (pure lookup must not kill the interpreter)."""
    assert auto_class_name("audio_llm") == "AutoModelForCausalLM"
    assert auto_class_name("seq2seq") == "AutoModelForSpeechSeq2Seq"
    assert auto_class_name("ctc") == "AutoModelForCTC"
    with pytest.raises(ValueError):
        auto_class_name("asr_llm")
    with pytest.raises(ValueError):
        auto_class_name("")


# ---------------------------------------------------------------------------
# load_speech_model -- unknown-kind refusal only (no real model is constructed)
# ---------------------------------------------------------------------------


def test_load_speech_model_unknown_kind_refuses_exit_96() -> None:
    """MUST_FIRE: an unknown speech kind refuses via SystemExit(96) naming the kind and
    SPEECH_KINDS (its geometry is undeclared)."""
    with pytest.raises(SystemExit) as exc:
        load_speech_model("mystery_kind", "some/path")
    assert exc.value.code == 96


def test_load_speech_model_empty_kind_refuses_exit_96() -> None:
    """MUST_FIRE: load_speech_model refuses (96) for a kind outside SPEECH_KINDS --
    e.g. empty string -- rather than guessing a geometry."""
    with pytest.raises(SystemExit) as exc:
        load_speech_model("", "some/path")
    assert exc.value.code == 96
    assert list(SPEECH_KINDS) == ["audio_llm", "seq2seq", "ctc"]


# ---------------------------------------------------------------------------
# speech_support_refusal: per-kind gates with the missing piece named
# ---------------------------------------------------------------------------


def _ok_audio_llm_processor() -> SimpleNamespace:
    """audio_llm processor exposing only what it must: audio_token/audio_token_id and
    feature_extractor.sampling_rate."""
    return SimpleNamespace(
        audio_token="<audio>",
        audio_token_id=8190,
        feature_extractor=SimpleNamespace(sampling_rate=16000),
    )


def _ok_seq2seq_processor() -> SimpleNamespace:
    """seq2seq processor exposing feature_extractor.sampling_rate + tokenizer (no
    audio placeholder requirement)."""
    return SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000),
        tokenizer=SimpleNamespace(toke_ids=1),
    )


def _ok_ctc_processor() -> SimpleNamespace:
    """ctc processor exposing feature_extractor.sampling_rate + tokenizer."""
    return SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000),
        tokenizer=SimpleNamespace(),
    )


def _good_processor(kind: str) -> SimpleNamespace:
    assert kind in SPEECH_KINDS
    return {
        "audio_llm": _ok_audio_llm_processor,
        "seq2seq": _ok_seq2seq_processor,
        "ctc": _ok_ctc_processor,
    }[kind]()


def test_speech_support_refusal_accepts_a_replete_processor_per_kind() -> None:
    """MUST_PASS: each of the 3 kinds accepts a processor exposing its own required
    surface (boa/eoa NOT required for audio_llm)."""
    for kind in ("audio_llm", "seq2seq", "ctc"):
        assert speech_support_refusal(kind, _good_processor(kind)) is None


def test_speech_support_refusal_names_the_first_missing_piece_per_kind() -> None:
    """MUST_FIRE: an incomplete processor produces ONE refusal naming the missing piece
    first (fixed order), including boa/eoa absence never refused for audio_llm."""
    # audio_llm -- no audio_token
    r = speech_support_refusal(
        "audio_llm",
        SimpleNamespace(
            audio_token=None,
            audio_token_id=9,
            feature_extractor=SimpleNamespace(sampling_rate=16000),
        ),
    )
    assert r is not None and "audio_token" in r and "audio_llm" in r
    # audio_llm -- non-int audio_token_id (bool must not read as int)
    r = speech_support_refusal(
        "audio_llm",
        SimpleNamespace(
            audio_token="<audio>",
            audio_token_id=True,
            feature_extractor=SimpleNamespace(sampling_rate=16000),
        ),
    )
    assert r is not None and "audio_token_id" in r
    # audio_llm -- boa/eoa absent is NOT a refusal (Gemma-4 surround convention only)
    proc = SimpleNamespace(
        audio_token="<audio>",
        audio_token_id=5,
        feature_extractor=SimpleNamespace(sampling_rate=8000),
    )
    assert speech_support_refusal("audio_llm", proc) is None

    # seq2seq -- no feature_extractor
    r = speech_support_refusal("seq2seq", SimpleNamespace(tokenizer=SimpleNamespace()))
    assert r is not None and "feature_extractor" in r
    # seq2seq -- no tokenizer (the label path)
    r = speech_support_refusal(
        "seq2seq", SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    )
    assert r is not None and "tokenizer" in r
    # ctc -- no tokenizer
    r = speech_support_refusal(
        "ctc", SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    )
    assert r is not None and "tokenizer" in r


def test_speech_support_refusal_bad_sampling_rate_and_unknown_kind_and_none_processor() -> None:
    """MUST_FIRE: non-positive/non-int sampling_rate, an unknown kind and a None
    processor each refuse (refusal names the bad number / SPEECH_KINDS)."""
    # unknown kind
    r = speech_support_refusal("hand_clap", _ok_ctc_processor())
    assert r is not None and "hand_clap" in r
    # None processor
    assert speech_support_refusal("ctc", None) is not None
    # sampling_rate not a positive int (bool counts as invalid here too)
    for bad in (True, 0, -1, "16000", None, 3.5):
        proc = SimpleNamespace(
            feature_extractor=SimpleNamespace(sampling_rate=bad), tokenizer=SimpleNamespace()
        )
        r = speech_support_refusal("ctc", proc)
        assert r is not None and "sampling_rate" in r, f"sampling_rate={bad!r} must refuse"
    # feature_extractor missing for audio_llm too
    r = speech_support_refusal("audio_llm", SimpleNamespace(audio_token="<a>", audio_token_id=3))
    assert r is not None and "feature_extractor" in r


# ---------------------------------------------------------------------------
# shared helpers: real 16 kHz wav rows in tmp_path
# ---------------------------------------------------------------------------


def _write_wav(tmp_path, name: str = "clip.wav", seconds: float = 0.25, sr: int = 16000):
    """Write a tiny 16 kHz mono sine wav and return its path -- a real soundfile
    carrier for these rows."""
    t = np.arange(int(sr * seconds)) / sr
    waveform = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    path = tmp_path / name
    sf.write(str(path), waveform.astype(np.float32), sr)
    return str(path), seconds


def _audio_processor(sr: int = 16000) -> SimpleNamespace:
    """A minimal feature_extractor/tokenizer stub returning real torch tensors, with
    sampling_rate for each geometry's gate."""
    feature_extractor = SimpleNamespace(sampling_rate=sr)
    tokenizer = SimpleNamespace()
    return SimpleNamespace(feature_extractor=feature_extractor, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# seq2seq collator
# ---------------------------------------------------------------------------


class _Seq2SeqFeatExtractor:
    """Fake Whisper feature extractor: 3 mel frames per input row, optional
    attention_mask, real torch tensors."""

    def __init__(self, sampling_rate: int = 16000, emit_mask: bool = False):
        self.sampling_rate = sampling_rate
        self.emit_mask = emit_mask
        self.calls: list = []

    def __call__(self, waves, sampling_rate: int, return_tensors: str, **kwargs):
        self.calls.append((waves, sampling_rate))
        batch = len(waves)
        out = {"input_features": torch.zeros(batch, 3, dtype=torch.float32)}
        if self.emit_mask:
            out["attention_mask"] = torch.ones(batch, 3, dtype=torch.long)
        return SimpleNamespace(**out)


class _Seq2SeqTokenizer:
    """Fake Whisper tokenizer: returns the given per-row id rows as a Mapping-like with
    input_ids, or the provided keys."""

    def __init__(self, id_rows, keys: tuple = ("input_ids",)):
        self.id_rows = id_rows
        self.keys = keys
        self.texts: list = []
        self.kwargs: list = []

    def __call__(self, texts, **kwargs):
        self.texts.append(list(texts))
        self.kwargs.append(kwargs)
        out = {"input_ids": self.id_rows}
        # allow tests to omit input_ids entirely
        out = {k: v for k, v in out.items() if k in self.keys}
        return SimpleNamespace(**out)


class _BadRowLoader:
    """Not used directly -- kept as a marker that load errors are exercised via
    AudioLoadError provocations below."""


def _seq2seq_processor(extractor, tokenizer):
    return SimpleNamespace(feature_extractor=extractor, tokenizer=tokenizer)


def test_seq2seq_collator_pads_labels_with_100_and_coverage_counts(tmp_path) -> None:
    """MUST_PASS: labels pad with -100 (the ignore index), real 16 kHz rows load, and
    coverage counts expected+ok per row."""
    path_a, _ = _write_wav(tmp_path, "a.wav", 0.25)
    path_b, _ = _write_wav(tmp_path, "b.wav", 0.25)
    extractor = _Seq2SeqFeatExtractor(emit_mask=True)
    decoder_start = 50258
    tokenizer = _Seq2SeqTokenizer(id_rows=[[decoder_start, 1, 2], [decoder_start, 3]])
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(extractor, tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=448,
        decoder_start_token_id=decoder_start,
        language="en",
        task="transcribe",
    )
    rows = [
        {"audio": path_a, "answer": "hello there"},
        {"audio": path_b, "answer": "hi"},
    ]
    batch = collate(rows)
    labels = batch["labels"]
    assert isinstance(labels, torch.Tensor)
    # decoder start present on ALL rows -> stripped; rows [1,2] and [3] padded to
    # len 2 with -100
    assert labels.tolist() == [[1, 2], [3, -100]]
    assert (labels == -100).any(), "MUST pad bare rows with the ignore index"
    assert batch["input_features"].shape[0] == 2
    assert batch["attention_mask"].shape == (2, 3)
    cov = collate.coverage
    assert cov.rows_expected
    assert cov.rows_checked == 2, "coverage counts each loaded row"
    assert cov.seconds_total == pytest.approx(0.5, abs=1e-3)


def test_seq2seq_keeps_decoder_start_when_not_universal(tmp_path) -> None:
    """MUST_PASS: when even ONE row lacks the decoder start token, NOTHING is stripped --
    every row stays exactly as tokenised."""
    path, _ = _write_wav(tmp_path, "a.wav")
    path2, _ = _write_wav(tmp_path, "b.wav")
    extractor = _Seq2SeqFeatExtractor(emit_mask=False)
    decoder_start = 50258
    tokenizer = _Seq2SeqTokenizer(id_rows=[[decoder_start, 7], [8, 9]])  # second row lacks start
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(extractor, tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=448,
        decoder_start_token_id=decoder_start,
    )
    batch = collate([{"audio": path, "answer": "x"}, {"audio": path2, "answer": "y"}])
    # first row keeps its start token; nothing is stripped anywhere
    assert batch["labels"].tolist() == [[decoder_start, 7], [8, 9]]
    assert "attention_mask" not in batch, "never invent a mask the extractor did not emit"


def test_seq2seq_strips_decoder_start_only_when_universal(tmp_path) -> None:
    """MUST_PASS: with all rows carrying the decoder start token, exactly one leading
    token is removed per row."""
    path, _ = _write_wav(tmp_path, "a.wav")
    path2, _ = _write_wav(tmp_path, "b.wav")
    decoder_start = 50258
    tokenizer = _Seq2SeqTokenizer(id_rows=[[decoder_start, 4, 5, 6], [decoder_start, 4]])
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(_Seq2SeqFeatExtractor(), tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=448,
        decoder_start_token_id=decoder_start,
    )
    batch = collate([{"audio": path, "answer": "x"}, {"audio": path2, "answer": "y"}])
    assert batch["labels"].tolist() == [[4, 5, 6], [4, -100, -100]]


def test_seq2seq_over_length_labels_refuse_exit_96_naming_row(tmp_path) -> None:
    """MUST_FIRE: one row wider than max_label_length refuses (96) naming the row --
    labels are never truncated silently."""
    path, _ = _write_wav(tmp_path, "a.wav")
    path2, _ = _write_wav(tmp_path, "b.wav")
    decoder_start = 50258
    tokenizer = _Seq2SeqTokenizer(id_rows=[[5, 6, 7], [8, 9, 10]])
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(_Seq2SeqFeatExtractor(), tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=2,
        decoder_start_token_id=decoder_start,
    )
    with pytest.raises(SystemExit) as exc:
        collate([{"audio": path, "answer": "a longer text"}, {"audio": path2, "answer": "b"}])
    assert exc.value.code == 96


def test_seq2seq_load_error_refuses_naming_row_and_reason(tmp_path) -> None:
    """MUST_FIRE: a row the loader rejects is a REFUSED run naming its row id and a
    counted reason (AudioLoadError.reason) -- never a shorter batch."""
    path, _ = _write_wav(tmp_path, "a.wav")
    # Missing declared carrier -> load_audio raises AudioLoadError with a counted reason.
    missing = str(tmp_path / "nope.wav")

    def raiser(*args, **kwargs):
        from foundationscale.train.audio import AudioLoadError

        raise AudioLoadError("missing file", None)

    import foundationscale.train.audio as audio_mod
    from foundationscale.train.audio import AudioLoadError

    original = audio_mod.load_audio

    def fake_load(value, *, target_sr, row_id, max_seconds):
        if value == missing:
            raise AudioLoadError("missing file", value)
        return original(value, target_sr=target_sr, row_id=row_id, max_seconds=max_seconds)

    audio_mod.load_audio = fake_load
    try:
        tokenizer = _Seq2SeqTokenizer(id_rows=[[5, 6], [7, 8]])
        collate = train_seq2seq_collator_or_refuse(
            _seq2seq_processor(_Seq2SeqFeatExtractor(), tokenizer),
            audio_column="audio",
            answer_field="answer",
            max_label_length=448,
            decoder_start_token_id=50258,
        )
        with pytest.raises(SystemExit) as exc:
            collate([{"audio": path, "answer": "x"}, {"audio": missing, "answer": "y"}])
        assert exc.value.code == 96
    finally:
        audio_mod.load_audio = original


def test_seq2seq_over_length_seconds_refuses(tmp_path) -> None:
    """MUST_FIRE: a waveform longer than the Whisper 30 s window refuses rather than
    truncating (measured window WHISPER_WINDOW_SECONDS)."""
    assert WHISPER_WINDOW_SECONDS == 30.0
    # A REAL 31 s file through the real strict loader: no patching, so the test
    # proves the collator hands the window to load_audio and refuses on too_long.
    path, _ = _write_wav(tmp_path, "long.wav", seconds=31.0)
    tokenizer = _Seq2SeqTokenizer(id_rows=[[5, 6]])
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(_Seq2SeqFeatExtractor(), tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=448,
        decoder_start_token_id=50258,
        language="en",
        task="transcribe",
    )
    with pytest.raises(SystemExit) as exc:
        collate([{"audio": path, "answer": "x"}])
    assert exc.value.code == 96


def test_seq2seq_refuses_when_processor_incomplete_before_any_row() -> None:
    """MUST_FIRE: a seq2seq processor missing its tokenizer refuses (96) at construction,
    before the corpus is touched (the refusal names the missing piece)."""
    with pytest.raises(SystemExit) as exc:
        train_seq2seq_collator_or_refuse(
            SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000)),
            audio_column="audio",
            answer_field="answer",
            max_label_length=448,
            decoder_start_token_id=1,
        )
    assert exc.value.code == 96


def test_seq2seq_tokenizer_without_input_ids_refuses(tmp_path) -> None:
    """MUST_FIRE: a tokenizer that returns no input_ids refuses (96) instead of training
    against an objective that is not the task."""
    path, _ = _write_wav(tmp_path, "a.wav")
    tokenizer = _Seq2SeqTokenizer(id_rows=[], keys=())
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(_Seq2SeqFeatExtractor(), tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=448,
        decoder_start_token_id=50258,
    )
    with pytest.raises(SystemExit) as exc:
        collate([{"audio": path, "answer": "x"}])
    assert exc.value.code == 96


def test_seq2seq_label_row_count_mismatch_refuses(tmp_path) -> None:
    """MUST_FIRE: fewer tokenizer label rows than texts refuses -- a denominator
    mismatch would mislabel the batch."""
    path, _ = _write_wav(tmp_path, "a.wav")
    path2, _ = _write_wav(tmp_path, "b.wav")
    tokenizer = _Seq2SeqTokenizer(id_rows=[[5, 6]])  # 2 texts, 1 label row
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(_Seq2SeqFeatExtractor(), tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=448,
        decoder_start_token_id=50258,
    )
    with pytest.raises(SystemExit) as exc:
        collate([{"audio": path, "answer": "x"}, {"audio": path2, "answer": "y"}])
    assert exc.value.code == 96


def test_seq2seq_extractor_without_input_features_refuses(tmp_path) -> None:
    """MUST_FIRE: an extractor whose output lost input_features refuses (96) -- a
    declared audio axis that is not executed is a refusal."""
    path, _ = _write_wav(tmp_path, "a.wav")

    class EmptyExtractor:
        def __init__(self):
            self.sampling_rate = 16000

        def __call__(self, waves, sampling_rate, return_tensors, **kwargs):
            return SimpleNamespace(spectrograms=torch.zeros(1, 3))

    tokenizer = _Seq2SeqTokenizer(id_rows=[[5, 6]])
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(EmptyExtractor(), tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=448,
        decoder_start_token_id=50258,
    )
    with pytest.raises(SystemExit) as exc:
        collate([{"audio": path, "answer": "x"}])
    assert exc.value.code == 96


# ---------------------------------------------------------------------------
# ctc collator
# ---------------------------------------------------------------------------


class _CTCFeatExtractor:
    """Fake CTC feature extractor emitting input_features and (optionally)
    attention_mask as torch tensors, recording calls."""

    def __init__(self, sampling_rate: int = 16000, include_mask: bool = True, frames: int = 4):
        self.sampling_rate = sampling_rate
        self.include_mask = include_mask
        self.frames = frames
        self.calls: list = []

    def __call__(self, waves, sampling_rate, return_tensors, return_attention_mask=False, **kwargs):
        self.calls.append({"return_attention_mask": return_attention_mask, "n": len(waves)})
        out = {"input_features": torch.zeros(len(waves), self.frames, dtype=torch.float32)}
        if self.include_mask:
            out["attention_mask"] = torch.ones(len(waves), self.frames, dtype=torch.long)
        return SimpleNamespace(**out)


class _CTCTokenizer:
    """Fake CTC tokenizer honouring add_special_tokens=False and returning input_ids
    rows per transcript (or a custom keys set)."""

    def __init__(self, id_rows, keys: tuple = ("input_ids",)):
        self.id_rows = id_rows
        self.keys = keys
        self.calls: list = []

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        self.calls.append({"texts": list(texts), "add_special_tokens": add_special_tokens})
        out = {"input_ids": self.id_rows}
        out = {k: v for k, v in out.items() if k in self.keys}
        return SimpleNamespace(**out)


def test_ctc_labels_pad_with_pad_token_id_never_100(tmp_path) -> None:
    """MUST_PASS: CTC labels pad with the CTC blank (pad_token_id) and carry NO -100
    anywhere; attention_mask is required and present."""
    path, _ = _write_wav(tmp_path, "a.wav")
    path2, _ = _write_wav(tmp_path, "b.wav")
    blank = 0
    extractor = _CTCFeatExtractor(include_mask=True)
    tokenizer = _CTCTokenizer(id_rows=[[1, 2, 3], [4]])
    collate = train_ctc_collator_or_refuse(
        _seq2seq_processor(extractor, tokenizer),
        audio_column="audio",
        answer_field="answer",
        pad_token_id=blank,
        max_seconds=None,
    )
    batch = collate([{"audio": path, "answer": "ab"}, {"audio": path2, "answer": "c"}])
    labels = batch["labels"]
    assert labels.tolist() == [[1, 2, 3], [4, blank, blank]]
    assert (labels != -100).all(), (
        "CTC labels must NEVER be -100 anywhere (that is the seq2seq ignore index)"
    )
    assert "attention_mask" in batch and batch["attention_mask"].shape[0] == 2
    assert extractor.calls[0]["return_attention_mask"] is True
    # coverage counts each loaded row ok
    cov = collate.coverage
    assert cov.rows_expected
    assert cov.rows_checked == 2


def test_ctc_empty_label_row_refuses(tmp_path) -> None:
    """MUST_FIRE: a transcript that tokenises to ZERO targets refuses (96) -- an empty
    CTC row would supervise the head to emit frames for nothing."""
    path, _ = _write_wav(tmp_path, "a.wav")
    extractor = _CTCFeatExtractor()
    tokenizer = _CTCTokenizer(id_rows=[[]])  # zero tokens -> refused, not trained
    collate = train_ctc_collator_or_refuse(
        _seq2seq_processor(extractor, tokenizer),
        audio_column="audio",
        answer_field="answer",
        pad_token_id=0,
        max_seconds=None,
    )
    with pytest.raises(SystemExit) as exc:
        collate([{"audio": path, "answer": ""}])
    assert exc.value.code == 96


def test_ctc_extractor_without_attention_mask_refuses(tmp_path) -> None:
    """MUST_FIRE: a CTC extractor that emits no attention_mask refuses (96) -- the CTC
    loss derives input lengths FROM that mask and must not be fed unmeasured lengths."""
    path, _ = _write_wav(tmp_path, "a.wav")
    extractor = _CTCFeatExtractor(include_mask=False)
    tokenizer = _CTCTokenizer(id_rows=[[5, 6]])
    collate = train_ctc_collator_or_refuse(
        _seq2seq_processor(extractor, tokenizer),
        audio_column="audio",
        answer_field="answer",
        pad_token_id=0,
        max_seconds=None,
    )
    with pytest.raises(SystemExit) as exc:
        collate([{"audio": path, "answer": "hello"}])
    assert exc.value.code == 96


def test_ctc_refuses_bad_pad_token_id_and_missing_parts_before_decode() -> None:
    """MUST_FIRE: a pad_token_id that is not a non-negative int refuses (96), as does a
    processor missing its tokenizer (each names its defect)."""
    path, _ = (
        _write_wav(__import__("pathlib").Path("/tmp"), "z.wav") if False else (None, None)
    )  # not used below
    with pytest.raises(SystemExit) as exc:
        train_ctc_collator_or_refuse(
            _seq2seq_processor(_CTCFeatExtractor(), _CTCTokenizer(id_rows=[[1]])),
            audio_column="audio",
            answer_field="answer",
            pad_token_id=True,
            max_seconds=None,
        )
    assert exc.value.code == 96
    with pytest.raises(SystemExit) as exc2:
        train_ctc_collator_or_refuse(
            SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000)),
            audio_column="audio",
            answer_field="answer",
            pad_token_id=0,
            max_seconds=None,
        )
    assert exc2.value.code == 96


def test_ctc_load_error_refuses_naming_row_and_reason(tmp_path) -> None:
    """MUST_FIRE: a CTC row the loader rejects is a REFUSED run naming its row id +
    counted reason (never a silently shorter batch)."""
    missing = str(tmp_path / "absent.wav")

    import foundationscale.train.audio as audio_mod
    from foundationscale.train.audio import AudioLoadError

    original = audio_mod.load_audio

    def fake_load(value, *, target_sr, row_id, max_seconds):
        raise AudioLoadError("missing file", value)

    audio_mod.load_audio = fake_load
    try:
        extractor = _CTCFeatExtractor()
        tokenizer = _CTCTokenizer(id_rows=[[5, 6]])
        collate = train_ctc_collator_or_refuse(
            _seq2seq_processor(extractor, tokenizer),
            audio_column="audio",
            answer_field="answer",
            pad_token_id=0,
            max_seconds=None,
        )
        with pytest.raises(SystemExit) as exc:
            collate([{"audio": missing, "answer": "x"}])
        assert exc.value.code == 96
    finally:
        audio_mod.load_audio = original


def test_ctc_coverage_counts_expected_rows_with_audio(tmp_path) -> None:
    """MUST_PASS: the ctc collator's coverage tracks rows_expected(), rows_with_audio
    == loaded rows and total seconds -- counters shared across worker processes."""
    path, expected_seconds = _write_wav(tmp_path, "r.wav", 0.5)
    path2, _ = _write_wav(tmp_path, "s.wav", 0.25)
    extractor = _CTCFeatExtractor()
    tokenizer = _CTCTokenizer(id_rows=[[1, 2], [3]])
    collate = train_ctc_collator_or_refuse(
        _seq2seq_processor(extractor, tokenizer),
        audio_column="audio",
        answer_field="answer",
        pad_token_id=0,
        max_seconds=1.0,
    )
    batch = collate([{"audio": path, "answer": "hi"}, {"audio": path2, "answer": "yo"}])
    assert batch["labels"].shape[0] == 2
    cov = collate.coverage
    assert cov.rows_expected
    assert cov.rows_checked == 2
    assert cov.seconds_total == pytest.approx(0.75, abs=1e-3)


def test_ctc_and_seq2seq_vocabs_differ_in_padding_contract(tmp_path) -> None:
    """MUST_PASS: identical rows through both geometries yield -100 padding for seq2seq
    and blank padding for ctc -- one batch cannot hide both contracts."""
    path, _ = _write_wav(tmp_path, "v.wav")
    path2, _ = _write_wav(tmp_path, "w.wav")

    decoder_start = 50258
    seq_tokenizer = _Seq2SeqTokenizer(id_rows=[[decoder_start, 5, 6], [decoder_start, 7]])
    seq_collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(_Seq2SeqFeatExtractor(), seq_tokenizer),
        audio_column="audio",
        answer_field="answer",
        max_label_length=448,
        decoder_start_token_id=decoder_start,
    )
    seq_batch = seq_collate([{"audio": path, "answer": "x"}, {"audio": path2, "answer": "y"}])
    assert seq_batch["labels"].tolist() == [[5, 6], [7, -100]]

    ctc_tokenizer = _CTCTokenizer(id_rows=[[5, 6], [7]])
    ctc_collate = train_ctc_collator_or_refuse(
        _seq2seq_processor(_CTCFeatExtractor(), ctc_tokenizer),
        audio_column="audio",
        answer_field="answer",
        pad_token_id=99,
        max_seconds=None,
    )
    ctc_batch = ctc_collate([{"audio": path, "answer": "x"}, {"audio": path2, "answer": "y"}])
    assert ctc_batch["labels"].tolist() == [[5, 6], [7, 99]]
    assert (ctc_batch["labels"] != -100).all()


# --- Whisper prompt prefix, plane refusal, collator dispatch, loader -----------------


class _PrefixTokenizer(_Seq2SeqTokenizer):
    """A Whisper-like tokenizer that owns a prompt prefix and a language vocabulary."""

    START, EN, TRANSCRIBE, NOTS = 50258, 50259, 50360, 50364

    def __init__(self, id_rows, multilingual: bool = True):
        super().__init__(id_rows)
        self.unk_token_id = 0
        self.multilingual = multilingual
        self.prefix_tokens = [self.START, self.NOTS]
        self.set_calls: list = []

    def convert_tokens_to_ids(self, token):
        return self.EN if (self.multilingual and token == "<|en|>") else self.unk_token_id

    def set_prefix_tokens(self, language=None, task=None, predict_timestamps=None):
        self.set_calls.append((language, task, predict_timestamps))
        lang = [self.EN] if language else []
        self.prefix_tokens = [self.START, *lang, self.TRANSCRIBE, self.NOTS]


def test_seq2seq_multilingual_tokenizer_without_a_declared_language_refuses(tmp_path) -> None:
    """MUST_FIRE: the language token is declared, never guessed."""
    tok = _PrefixTokenizer(id_rows=[[50258, 50259, 50360, 50364, 7]])
    with pytest.raises(SystemExit) as exc:
        train_seq2seq_collator_or_refuse(
            _seq2seq_processor(_Seq2SeqFeatExtractor(), tok),
            audio_column="audio",
            max_label_length=448,
            decoder_start_token_id=50258,
        )
    assert exc.value.code == 96


def test_seq2seq_declared_language_sets_and_verifies_the_prompt(tmp_path) -> None:
    """MUST_PASS: labels carry <|en|><|transcribe|><|notimestamps|> after the start strip."""
    path, _ = _write_wav(tmp_path)
    tok = _PrefixTokenizer(id_rows=[[50258, 50259, 50360, 50364, 7, 8]])
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(_Seq2SeqFeatExtractor(), tok),
        audio_column="audio",
        max_label_length=448,
        decoder_start_token_id=50258,
        language="en",
    )
    batch = collate([{"audio": path, "answer": "x"}])
    assert tok.set_calls == [("en", "transcribe", False)]
    assert batch["labels"][0].tolist()[:3] == [50259, 50360, 50364]


def test_seq2seq_labels_missing_the_declared_prompt_refuse(tmp_path) -> None:
    """MUST_FIRE: a label row that does not start with the prompt is mislabelled."""
    path, _ = _write_wav(tmp_path)
    tok = _PrefixTokenizer(id_rows=[[50258, 50364, 7]])  # no language/task tokens
    collate = train_seq2seq_collator_or_refuse(
        _seq2seq_processor(_Seq2SeqFeatExtractor(), tok),
        audio_column="audio",
        max_label_length=448,
        decoder_start_token_id=50258,
        language="en",
    )
    with pytest.raises(SystemExit) as exc:
        collate([{"audio": path, "answer": "x"}])
    assert exc.value.code == 96


def test_speech_plane_refusal_order() -> None:
    """MUST_FIRE x2 / MUST_PASS: unknown kind, then missing audio tower, then processor."""
    from foundationscale.train.speech_kinds import speech_plane_refusal

    audio_family = SimpleNamespace(name="whisper", towers=(("model.encoder", "audio"),))
    textual = SimpleNamespace(name="txt", towers=(("model.visual", "image"),))
    proc = _seq2seq_processor(_Seq2SeqFeatExtractor(), _Seq2SeqTokenizer(id_rows=[[1]]))
    assert "not a known speech kind" in (speech_plane_refusal(None, audio_family, proc) or "")
    assert "declares no audio tower" in (speech_plane_refusal("seq2seq", textual, proc) or "")
    assert "no registered family" in (speech_plane_refusal("seq2seq", None, proc) or "")
    assert speech_plane_refusal("seq2seq", audio_family, proc) is None


def test_build_speech_collator_dispatches_by_kind_and_caps_whisper_labels(
    tmp_path, monkeypatch
) -> None:
    """MUST_PASS: each kind gets its collator; seq2seq label cap = min(max_length, 448)."""
    from foundationscale.train import speech_kinds as sk

    seen: dict = {}
    monkeypatch.setattr(
        sk, "train_seq2seq_collator_or_refuse", lambda p, **kw: seen.setdefault("s2s", kw)
    )
    monkeypatch.setattr(
        sk, "train_ctc_collator_or_refuse", lambda p, **kw: seen.setdefault("ctc", kw)
    )
    import foundationscale.train.audio as audio_mod

    monkeypatch.setattr(
        audio_mod, "train_audio_collator_or_refuse", lambda s, **kw: seen.setdefault("llm", kw)
    )
    surface = SimpleNamespace(surface=object())
    whisper_cfg = SimpleNamespace(max_target_positions=448, decoder_start_token_id=50258)
    sk.build_speech_collator(
        "seq2seq",
        surface,
        audio_column="a",
        max_length=1024,
        model_config=whisper_cfg,
        language="en",
    )
    assert seen["s2s"]["max_label_length"] == 448 and seen["s2s"]["language"] == "en"
    sk.build_speech_collator(
        "ctc",
        surface,
        audio_column="a",
        max_length=1024,
        model_config=SimpleNamespace(pad_token_id=1024),
        language=None,
    )
    assert seen["ctc"]["pad_token_id"] == 1024
    sk.build_speech_collator(
        "audio_llm",
        surface,
        audio_column="a",
        max_length=777,
        model_config=SimpleNamespace(),
        language=None,
    )
    assert seen["llm"]["max_length"] == 777
    with pytest.raises(SystemExit):
        sk.build_speech_collator(
            "tts",
            surface,
            audio_column="a",
            max_length=1,
            model_config=SimpleNamespace(),
            language=None,
        )


def test_load_speech_model_uses_the_named_architecture_for_audio_llms(monkeypatch) -> None:
    """MUST_PASS: audio_llm loads config.architectures[0]; other kinds use the Auto class."""
    import types

    from foundationscale.train import speech_kinds as sk

    calls: list = []

    def cls(name):
        return SimpleNamespace(from_pretrained=lambda path, **kw: calls.append((name, path)))

    fake = types.SimpleNamespace(
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda p: SimpleNamespace(
                architectures=["Qwen2AudioForConditionalGeneration"]
            )
        ),
        Qwen2AudioForConditionalGeneration=cls("Qwen2AudioForConditionalGeneration"),
        AutoModelForCausalLM=cls("AutoModelForCausalLM"),
        AutoModelForSpeechSeq2Seq=cls("AutoModelForSpeechSeq2Seq"),
        AutoModelForCTC=cls("AutoModelForCTC"),
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake)
    sk.load_speech_model("audio_llm", "/m")
    sk.load_speech_model("seq2seq", "/m")
    sk.load_speech_model("ctc", "/m")
    assert [c[0] for c in calls] == [
        "Qwen2AudioForConditionalGeneration",
        "AutoModelForSpeechSeq2Seq",
        "AutoModelForCTC",
    ]


def test_frozen_batchnorm_keeps_running_stats_and_still_trains_affine() -> None:
    """MUST_PASS: running stats do not move in train mode; the affine weight gets a gradient."""
    import torch

    from foundationscale.train.speech_kinds import freeze_batchnorm_statistics

    model = torch.nn.Sequential(torch.nn.Conv1d(2, 3, 1), torch.nn.BatchNorm1d(3))
    before = model[1].running_mean.clone()
    assert freeze_batchnorm_statistics(model) == 1
    model.train()
    out = model(torch.randn(4, 2, 5) * 10 + 3)
    out.sum().backward()
    assert torch.equal(model[1].running_mean, before)
    assert model[1].weight.grad is not None
    assert freeze_batchnorm_statistics(torch.nn.Linear(2, 2)) == 0


def test_group_by_duration_is_declared_and_refuses_without_the_column() -> None:
    """MUST_PASS/MUST_FIRE: undeclared -> None; declared -> sampler; no column -> refusal."""
    from foundationscale.train.speech_kinds import group_by_duration_settings

    assert group_by_duration_settings(None, ["audio", "duration"]) is None
    assert group_by_duration_settings("", ["audio", "duration"]) is None
    assert group_by_duration_settings("1", ["audio", "duration"]) == {
        "train_sampling_strategy": "group_by_length",
        "length_column_name": "duration",
    }
    refusal = group_by_duration_settings("1", ["audio", "answer"])
    assert isinstance(refusal, str) and "duration" in refusal


def test_apply_group_by_duration_updates_kwargs_or_refuses() -> None:
    """MUST_PASS/MUST_FIRE: applied in place with an announcement; refusal; undeclared no-op."""
    from foundationscale.train.speech_kinds import GROUP_BY_DURATION_ENV, apply_group_by_duration

    kw: dict = {}
    line = apply_group_by_duration(kw, {GROUP_BY_DURATION_ENV: "1"}, ["audio", "duration"])
    assert line is not None and line.startswith("[   ok]")
    assert kw["train_sampling_strategy"] == "group_by_length"
    kw2: dict = {}
    refusal = apply_group_by_duration(kw2, {GROUP_BY_DURATION_ENV: "1"}, ["audio"])
    assert refusal is not None and not refusal.startswith("[   ok]") and kw2 == {}
    assert apply_group_by_duration({}, {}, ["duration"]) is None


def test_full_train_announcement_names_the_declared_scope() -> None:
    """The announcement names the scope from the family, not a fixed 'language model'."""
    from foundationscale.train.speech_kinds import full_train_announcement

    fam = SimpleNamespace(adapter_scope_prefixes=("encoder.layers",))
    line = full_train_announcement("audio", ["encoder.subsampling"], fam)
    assert "encoder.subsampling" in line and "['encoder.layers']" in line


def test_lora_audio_plan_announces_or_refuses() -> None:
    """MUST_PASS/MUST_FIRE: measured wrap points -> modules + announcement; none -> refusal."""
    from foundationscale.families.registry import REGISTRY
    from foundationscale.train.speech_kinds import lora_audio_plan

    by_name = {spec.name: spec for spec in REGISTRY}
    modules, line = lora_audio_plan(by_name["parakeet_ctc"], "audio")
    assert modules == ["encoder.subsampling"] and "['encoder.layers']" in line
    none, refusal = lora_audio_plan(SimpleNamespace(towers=(("a", "audio"),)), "audio")
    assert none == [] and "declares no adapter_full_train" in refusal


def test_build_speech_collator_checked_probes_and_resets(monkeypatch) -> None:
    """MUST_PASS: probe rows are collated, then coverage is reset; MUST_FIRE: no features."""
    from foundationscale.train import speech_kinds as sk

    calls: list = []

    class _Col:
        def __init__(self, keys):
            self.keys_ = keys
            self.coverage = SimpleNamespace(reset=lambda: calls.append("reset"))

        def __call__(self, rows):
            calls.append(len(rows))
            return dict.fromkeys(self.keys_)

    monkeypatch.setattr(sk, "build_speech_collator", lambda *a, **k: _Col(["input_features"]))
    col = sk.build_speech_collator_checked(
        "ctc",
        object(),
        [{}, {}],
        audio_column="a",
        max_length=1,
        model_config=SimpleNamespace(),
        language=None,
    )
    assert calls == [2, "reset"] and isinstance(col, _Col)
    monkeypatch.setattr(sk, "build_speech_collator", lambda *a, **k: _Col(["input_ids"]))
    with pytest.raises(SystemExit):
        sk.build_speech_collator_checked(
            "ctc",
            object(),
            [{}],
            audio_column="a",
            max_length=1,
            model_config=SimpleNamespace(),
            language=None,
        )
