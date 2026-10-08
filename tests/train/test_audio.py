"""P1 slice 1: the audio contract is a conjunction, and every refusal names its cause.

The claim under test: a declared audio column is accepted ONLY where the family
carries an audio tower, the processor can place and delimit the placeholders, and
the feature extractor states the rate every waveform must match -- and every waveform
handed on is the caller's samples, verbatim bar the mono mixdown, with its seconds
counted. Nothing here reaches a model or a GPU, deliberately: reaching this decision
by any other route has so far meant reaching hardware first.

THE CONTROLS ARE THE POINT, and there are three kinds:

  each leg, alone          every MUST_FIRE below breaks exactly one leg of the
                           conjunction and names exactly one missing piece. A
                           generic refusal ("audio is not supported") would put the
                           operator back where T1-23 left them: the right exit code
                           attached to the wrong explanation.
  the strict loader        a rate mismatch refuses rather than resamples, and a row
                           past ``max_seconds`` refuses rather than truncates. The
                           manifest describes what the tower saw, so the loader may
                           not improve a row in silence -- and the round trip through
                           either accepted carrier must be the SAME numbers.
  the denominator          AudioCoverage counts what refuses to load: a dropped row
                           is ACCOUNTED (absent from the batch, present in the
                           total), and zero rows checked is VACUOUS even when every
                           row was accounted.

No skips. Skips are failures in this repo.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from foundationscale.train.audio import (
    AUDIO_COLUMN_ENV,
    AUDIO_LOAD_REASONS,
    AudioCoverage,
    AudioLoadError,
    audio_placeholder_counts,
    audio_support_refusal,
    load_audio,
    max_audio_seconds,
    placeholder_mismatches,
)

# --- test doubles: shapes, never classes -------------------------------------


def _family_with_audio() -> SimpleNamespace:
    """A gemma-4-shaped family: the registry spells tower_modalities as (prefix, modality)."""
    return SimpleNamespace(
        name="gemma4",
        tower_modalities=(
            ("model.vision_tower", "image"),
            ("model.audio_tower", "audio"),
            ("model.embed_vision", "image"),
        ),
    )


def _family_without_audio() -> SimpleNamespace:
    """qwen3.5-shaped: a vision tower, and no audio tower anywhere in the registry."""
    return SimpleNamespace(name="qwen3.5", tower_modalities=(("model.visual", "image"),))


def _family_modality_words() -> SimpleNamespace:
    """The other accepted spelling: bare modality strings, as an exercise set speaks them."""
    return SimpleNamespace(name="gemma4", tower_modalities={"audio"})


def _processor(**overrides: Any) -> SimpleNamespace:
    """A gemma-4-shaped processor: audio tokens, an id to count, a rate, and a cap."""
    fields: dict[str, Any] = {
        "audio_token": "<audio>",
        "boa_token": "<boa>",
        "eoa_token": "<eoa>",
        "audio_token_id": 32000,
        "feature_extractor": SimpleNamespace(sampling_rate=16000),
        "audio_seq_length": 750,
        "audio_ms_per_token": 40,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _write_wav(directory: Path, name: str, samples: np.ndarray, sampling_rate: int) -> Path:
    """A real WAV on disk. float32 on purpose: WAV has no DOUBLE subtype and PCM_16
    would round the mixdown leg's 1.0/-1.0 fixture into a falsehood."""
    path = directory / name
    sf.write(str(path), samples.astype(np.float32), sampling_rate, subtype="FLOAT")
    return path


# --- the acceptance gate (audio_support_refusal) -----------------------------


def test_must_pass_gemma4_family_and_full_processor_entrain_no_refusal() -> None:
    """MUST_PASS: the whole conjunction holds, so the runner is told to accept."""
    assert audio_support_refusal(_family_with_audio(), _processor()) is None


def test_must_pass_modality_words_spelling_of_tower_modalities_also_entrains_none() -> None:
    """MUST_PASS (control for the pair form): both spellings of tower_modalities read.

    A literal ``"audio" in family.tower_modalities`` over the registry's
    ``(prefix, modality)`` pairs is ALWAYS False, so this pair of tests is what keeps
    the accept path from being dead code that only a test double can reach.
    """
    assert audio_support_refusal(_family_modality_words(), _processor()) is None


def test_must_fire_unregistered_family_names_the_gap_and_the_variable() -> None:
    """MUST_FIRE leg 1: no family, so no measurement of an audio tower exists."""
    msg = audio_support_refusal(None, _processor())
    assert msg is not None, "an unregistered family must refuse: its towers are unknowable"
    assert "not registered" in msg
    assert AUDIO_COLUMN_ENV in msg


def test_must_fire_family_without_an_audio_tower_names_the_family_and_what_it_has() -> None:
    """MUST_FIRE leg 1: a registered family that simply has no audio tower."""
    msg = audio_support_refusal(_family_without_audio(), _processor())
    assert msg is not None
    assert "audio tower" in msg
    assert "qwen3.5" in msg, "the refusal names the family, so a wrong checkpoint is visible"
    assert "image" in msg, "what the family DOES connect is more useful than a bare no"
    assert AUDIO_COLUMN_ENV in msg


def test_must_fire_missing_processor_names_the_processor() -> None:
    """MUST_FIRE leg 2: no processor, so no tokens and no rate can be read."""
    msg = audio_support_refusal(_family_with_audio(), None)
    assert msg is not None and "processor" in msg
    assert "gemma4" in msg
    assert AUDIO_COLUMN_ENV in msg


@pytest.mark.parametrize("attr", ["audio_token", "boa_token", "eoa_token"])
def test_must_fire_each_missing_token_names_that_token(attr: str) -> None:
    """MUST_FIRE leg 3, one per token: placeholder placement needs all three words."""
    msg = audio_support_refusal(_family_with_audio(), _processor(**{attr: None}))
    assert msg is not None and attr in msg
    assert AUDIO_COLUMN_ENV in msg


def test_must_fire_absent_audio_token_id_names_it() -> None:
    """MUST_FIRE leg 3: without an id, nothing can count the placeholders it places."""
    msg = audio_support_refusal(_family_with_audio(), _processor(audio_token_id=None, **{}))
    assert msg is not None and "audio_token_id" in msg
    assert AUDIO_COLUMN_ENV in msg


@pytest.mark.parametrize("value", ["7", 7.0, True])
def test_must_fire_non_int_audio_token_id_names_it(value: Any) -> None:
    """MUST_FIRE leg 3: an id that is not an int is not an id.

    ``True`` is the interesting one -- ``isinstance(True, int)`` is True in Python,
    and Token 1 would put every audio block in an unrelated word's slots.
    """
    msg = audio_support_refusal(_family_with_audio(), _processor(audio_token_id=value))
    assert msg is not None and "audio_token_id" in msg
    assert AUDIO_COLUMN_ENV in msg


def test_must_fire_missing_feature_extractor_names_the_rate_it_hides() -> None:
    """MUST_FIRE leg 4, clipped at the extractor: absent and None are the same gap."""
    deleted = _processor()
    del deleted.feature_extractor
    for processor in (_processor(feature_extractor=None), deleted):
        msg = audio_support_refusal(_family_with_audio(), processor)
        assert msg is not None and "feature_extractor" in msg
        assert "sampling_rate" in msg
        assert AUDIO_COLUMN_ENV in msg


@pytest.mark.parametrize("rate", [None, 0, -16000, 16000.5, "16000", True])
def test_must_fire_every_non_int_sampling_rate_names_sampling_rate(rate: Any) -> None:
    """MUST_FIRE leg 4: the rate is the exact-match target for every row, so it must
    be a positive int -- a float rate is not rounded into service, 0 excludes every
    waveform, and a bool is a config typo wearing a number's clothes."""
    msg = audio_support_refusal(
        _family_with_audio(), _processor(feature_extractor=SimpleNamespace(sampling_rate=rate))
    )
    assert msg is not None and "sampling_rate" in msg
    assert AUDIO_COLUMN_ENV in msg


def test_the_first_failing_leg_is_the_one_named() -> None:
    """Two broken legs produce ONE report, and it is the first in the check order.

    Ordering is what makes the text a function of the state rather than of which
    attribute happened to be read first.
    """
    msg = audio_support_refusal(_family_without_audio(), None)
    assert msg is not None
    assert "audio tower" in msg
    assert "no audio processor" not in msg, "the second broken leg must not share the report"


def test_the_refusal_states_the_way_out_in_the_loops_words() -> None:
    """The wording is half the claim (#490's T1-23 was a correct code and a wrong story)."""
    msg = audio_support_refusal(_family_without_audio(), _processor())
    assert msg is not None
    assert msg.endswith(f"Unset {AUDIO_COLUMN_ENV} to train text-only on the same corpus")
    assert "requires a 'text' column" not in msg, "the exact confound that muddied T1-23's 96"


# --- the measured cap (max_audio_seconds) ------------------------------------


def test_must_pass_gemma4_cap_is_seventy_five_by_four_is_thirty_seconds() -> None:
    """MUST_PASS: 750 audio slots at 40 ms measure 30.0 s (Gemma-4's P0 numbers)."""
    assert max_audio_seconds(_processor()) == pytest.approx(30.0)


def test_must_fire_missing_audio_ms_per_token_is_unmeasured_not_invented() -> None:
    """MUST_FIRE: the millisecond figure lives on the config in some releases.

    UNMEASURED, not "unbounded": the caller decides, and a guessed cap would become a
    number in the manifest that no measurement produced.
    """
    processor = _processor()
    del processor.audio_ms_per_token
    assert max_audio_seconds(processor) is None


def test_must_fire_missing_audio_seq_length_is_unmeasured() -> None:
    """MUST_FIRE: one factor missing is the same as no product being measurable."""
    processor = _processor()
    del processor.audio_seq_length
    assert max_audio_seconds(processor) is None


def test_must_fire_garbage_measures_become_unmeasured_not_zero() -> None:
    """MUST_FIRE: a cap of 0.0 s would refuse every real row, and True would mean 1 ms.

    Same direction as above: anything this function cannot measure reads as None.
    """
    assert max_audio_seconds(_processor(audio_ms_per_token=True)) is None
    assert max_audio_seconds(_processor(audio_seq_length="750")) is None
    assert max_audio_seconds(_processor(audio_seq_length=float("nan"))) is None


# --- the loader (load_audio) -------------------------------------------------


def test_must_pass_path_carries_float32_mono_and_exact_duration(tmp_path: Path) -> None:
    """MUST_PASS: 4000 frames at 16 kHz is 0.25 s, and the manifest quotes this number."""
    path = _write_wav(tmp_path, "clip.wav", np.zeros(4000, dtype=np.float32), 16000)
    wave, duration = load_audio(str(path), target_sr=16000, row_id="row-1", max_seconds=30.0)
    assert wave.dtype == np.float32
    assert wave.shape == (4000,)
    assert abs(duration - 0.25) < 1e-6, f"duration {duration} drifted; the manifest reports it"


def test_must_pass_os_pathlike_is_a_path_too(tmp_path: Path) -> None:
    """MUST_PASS: ``pathlib.Path`` is the other spelling of a filesystem path corpora use."""
    path = _write_wav(tmp_path, "pathtype.wav", np.zeros(1600, dtype=np.float32), 16000)
    _wave, duration = load_audio(Path(path), target_sr=16000, row_id="row-2", max_seconds=None)
    assert abs(duration - 0.1) < 1e-6


def test_must_pass_stereo_mixes_down_over_frames_and_not_channels(tmp_path: Path) -> None:
    """MUST_PASS: (frames, channels) becomes one dimension by averaging the LAST axis.

    The fixture averages exactly to 0.0, so a channel PICK instead of a mixdown fails
    loudly -- and the frame count pins that the wrong axis was not averaged (which
    would halve the duration into the manifest).
    """
    stereo = np.stack([np.full(200, 1.0), np.full(200, -1.0)], axis=1).astype(np.float32)
    path = _write_wav(tmp_path, "stereo.wav", stereo, 16000)
    wave, duration = load_audio(str(path), target_sr=16000, row_id="row-3", max_seconds=1.0)
    assert wave.shape == (200,), "a (frames, channels) waveform must not escape the loader"
    assert np.allclose(wave, 0.0), "the mixdown must average, not choose a channel"
    assert abs(duration - 200 / 16000) < 1e-6


def test_must_pass_decoded_dict_form_is_accepted() -> None:
    """MUST_PASS: HF datasets' decoded Audio column, {"array", "sampling_rate"}."""
    wave, duration = load_audio(
        {"array": np.zeros(3200, dtype=np.float32), "sampling_rate": 16000},
        target_sr=16000,
        row_id="row-4",
        max_seconds=30.0,
    )
    assert wave.shape == (3200,) and wave.dtype == np.float32
    assert abs(duration - 0.2) < 1e-6


def test_must_pass_decoded_dict_stereo_array_mixes_down_too() -> None:
    """MUST_PASS (control): the dict carrier is held to the same shape rule as the file."""
    array = np.stack([np.full(100, 0.5), np.full(100, -0.5)], axis=1).astype(np.float32)
    wave, _duration = load_audio(
        {"array": array, "sampling_rate": 16000}, target_sr=16000, row_id="row-5", max_seconds=None
    )
    assert wave.shape == (100,)
    assert np.allclose(wave, 0.0)


def test_must_pass_stored_dict_form_reads_the_path(tmp_path: Path) -> None:
    """MUST_PASS: the stored form, {"path"}, decodes from the file it names."""
    path = _write_wav(tmp_path, "stored.wav", np.zeros(1600, dtype=np.float32), 16000)
    _wave, duration = load_audio(
        {"path": str(path)}, target_sr=16000, row_id="row-6", max_seconds=None
    )
    assert abs(duration - 0.1) < 1e-6


def test_must_pass_stored_dict_form_decodes_from_bytes_when_the_path_is_elsewhere(
    tmp_path: Path,
) -> None:
    """MUST_PASS: "bytes" is authoritative, because the named cache need not exist here.

    Without this the loader would be correct on the corpus's own box and red on every
    box that streams it -- the export lesson, one layer down.
    """
    path = _write_wav(tmp_path, "cached.wav", np.zeros(1600, dtype=np.float32), 16000)
    wave, duration = load_audio(
        {"path": str(tmp_path / "evicted" / "row.wav"), "bytes": path.read_bytes()},
        target_sr=16000,
        row_id="row-7",
        max_seconds=None,
    )
    assert wave.shape == (1600,)
    assert abs(duration - 0.1) < 1e-6


def test_must_pass_integer_pcm_is_scaled_the_way_the_file_carrier_scales_it() -> None:
    """MUST_PASS: identical samples must give identical numbers from both carriers."""
    wave, _duration = load_audio(
        {"array": np.array([32767], dtype=np.int16), "sampling_rate": 16000},
        target_sr=16000,
        row_id="row-8",
        max_seconds=None,
    )
    assert float(wave[0]) == pytest.approx(32767 / 32768)


def test_must_fire_wrong_sample_rate_refuses_instead_of_resampling(tmp_path: Path) -> None:
    """MUST_FIRE: the strict-by-default rate rule, which is a whole design decision.

    A silent resample would change the measured input (seconds, placeholders,
    spectrum) while the row looked untouched. Refusing names the mismatch instead.
    """
    path = _write_wav(tmp_path, "8k.wav", np.zeros(800, dtype=np.float32), 8000)
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(str(path), target_sr=16000, row_id="row-9", max_seconds=None)
    err = excinfo.value
    assert err.reason == "sample_rate_mismatch"
    assert err.row_id == "row-9"


def test_must_fire_row_over_max_seconds_is_counted_too_long(tmp_path: Path) -> None:
    """MUST_FIRE: past the cap the processor would truncate silently. FS counts it."""
    path = _write_wav(tmp_path, "long.wav", np.zeros(4000, dtype=np.float32), 16000)
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(str(path), target_sr=16000, row_id="row-10", max_seconds=0.1)
    err = excinfo.value
    assert err.reason == "too_long"
    assert err.row_id == "row-10"


def test_must_fire_row_under_min_seconds_is_counted_too_short(tmp_path: Path) -> None:
    """MUST_FIRE: the floor is symmetric with the cap, and just as un-pasted-over."""
    path = _write_wav(tmp_path, "short.wav", np.zeros(4000, dtype=np.float32), 16000)
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(str(path), target_sr=16000, row_id="row-11", min_seconds=0.5, max_seconds=None)
    err = excinfo.value
    assert err.reason == "too_short"
    assert err.row_id == "row-11"


def test_must_fire_zero_length_array_is_empty() -> None:
    """MUST_FIRE: zero samples is its own reason -- not "too_short", which lies about
    being measurable."""
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(
            {"array": np.zeros(0, dtype=np.float32), "sampling_rate": 16000},
            target_sr=16000,
            row_id="row-12",
            max_seconds=30.0,
        )
    err = excinfo.value
    assert err.reason == "empty"
    assert err.row_id == "row-12"


def test_must_fire_garbage_bytes_are_unreadable(tmp_path: Path) -> None:
    """MUST_FIRE: a truncated header is one counted row, not a crash and not silence."""
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"this is not audio, it is a comment")
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(str(path), target_sr=16000, row_id="row-13", max_seconds=None)
    err = excinfo.value
    assert err.reason == "unreadable"
    assert err.row_id == "row-13"


def test_must_fire_garbage_bytes_inside_the_stored_form_are_unreadable() -> None:
    """MUST_FIRE (control): the bytes carrier fails into the same bucket as the file."""
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(
            {"path": "/nonexistent/row.wav", "bytes": b"\x00\x01nonsense"},
            target_sr=16000,
            row_id="row-14",
            max_seconds=None,
        )
    err = excinfo.value
    assert err.reason == "unreadable"
    assert err.row_id == "row-14"


def test_must_fire_an_int_is_unsupported_value() -> None:
    """MUST_FIRE: a scalar is not a carrier, and "anything else" has a name."""
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(42, target_sr=16000, row_id="row-15", max_seconds=None)
    err = excinfo.value
    assert err.reason == "unsupported_value"
    assert err.row_id == "row-15"


@pytest.mark.parametrize(
    "value",
    [
        {"label": "x"},
        {"array": np.zeros(1600, dtype=np.float32)},
        {"path": None},
        {"array": ["a", "b"], "sampling_rate": 16000},
    ],
)
def test_must_fire_every_unreadable_mapping_is_unsupported_value(value: Any) -> None:
    """MUST_FIRE: no "array" pair, no "path", a non-path, and an array numpy cannot
    see as sound -- four shapes, one countable reason."""
    with pytest.raises(AudioLoadError) as excinfo:
        load_audio(value, target_sr=16000, row_id="row-16", max_seconds=None)
    err = excinfo.value
    assert err.reason == "unsupported_value"
    assert err.row_id == "row-16"


def test_must_fire_a_reason_outside_the_vocabulary_is_refused_at_construction() -> None:
    """MUST_FIRE: "TooLong" and "too_long" would merge into two invisible buckets.

    The closed vocabulary is what makes AudioCoverage's refused map totalable at all.
    """
    with pytest.raises(ValueError, match="not one of"):
        AudioLoadError("row-17", "TooLong")
    assert issubclass(AudioLoadError, ValueError), "a bad row is a data error, not a crash"
    assert set(AUDIO_LOAD_REASONS) >= {
        "unreadable",
        "empty",
        "sample_rate_mismatch",
        "too_long",
        "too_short",
        "unsupported_value",
    }


# --- the manifest (AudioCoverage) ---------------------------------------------


def test_must_fire_nothing_checked_is_vacuous() -> None:
    """MUST_FIRE: 0 rows checked blocks, ahead of every other verdict (gate doctrine)."""
    assert AudioCoverage(rows_expected=3).verdict() == "VACUOUS"


def test_must_fire_refused_only_rows_are_still_vacuous() -> None:
    """MUST_FIRE: every row accounted for and still nothing measured -- a batch of
    drops says nothing about the path the audio would have taken."""
    cov = AudioCoverage(rows_expected=2)
    cov.record_refused("unreadable")
    cov.record_refused("unreadable")
    assert cov.refused == {"unreadable": 2}, "one reason merges into one bucket"
    assert cov.verdict() == "VACUOUS"


def test_must_fire_undercovered_counts_refusals_toward_the_denominator() -> None:
    """MUST_FIRE: checked + refused is the accounted total; 1 of 4 is not 2 of 4."""
    cov = AudioCoverage(rows_expected=4)
    cov.record_ok(0.1)
    cov.record_refused("empty")
    assert cov.verdict() == "UNDERCOVERED"


def test_must_fire_more_rows_than_expected_is_overcovered() -> None:
    """MUST_FIRE: extra rows are as much a lie about the manifest as missing ones."""
    cov = AudioCoverage(rows_expected=1)
    cov.record_ok(1.0)
    cov.record_ok(1.0)
    assert cov.verdict() == "OVERCOVERED"


def test_must_pass_exact_breadth_is_covered() -> None:
    """MUST_PASS: the accept path of the ladder, and the control for the three above."""
    cov = AudioCoverage(rows_expected=2)
    cov.record_ok(0.5)
    cov.record_ok(0.5)
    assert cov.verdict() == "COVERED"


def test_must_pass_a_counted_drop_is_accounted_not_absent() -> None:
    """MUST_PASS: tolerant mode's whole contract -- the drop is in the total.

    Without the refused map entering the denominator, a loader that refused 9 of 10
    rows would report COVERED the moment one row decoded.
    """
    cov = AudioCoverage(rows_expected=3)
    cov.record_ok(0.1)
    cov.record_ok(0.2)
    cov.record_refused("too_long")
    assert cov.verdict() == "COVERED"


def test_the_manifest_names_its_keys_and_is_json_ready() -> None:
    """The manifest writer must never meet a numpy scalar or a live accumulator here."""
    cov = AudioCoverage(rows_expected=3, sampling_rate=16000)
    cov.record_ok(1.5)
    cov.record_ok(0.5)
    cov.record_refused("unreadable")
    manifest = cov.as_manifest()
    assert set(manifest) == {
        "rows_expected",
        "rows_checked",
        "rows_refused",
        "seconds_total",
        "sampling_rate",
        "refused",
        "placeholder_rows_verified",
        "placeholder_rows_unmeasured",
        "verdict",
    }
    json.dumps(manifest)
    assert manifest["rows_checked"] == 2
    assert manifest["rows_refused"] == 1
    assert manifest["seconds_total"] == 2.0
    assert manifest["sampling_rate"] == 16000
    assert manifest["refused"] == {"unreadable": 1}
    assert manifest["verdict"] == "COVERED"


def test_the_manifest_is_a_snapshot_and_not_a_live_view() -> None:
    """A manifest row that drifts as the loader continues contradicts itself in print."""
    cov = AudioCoverage(rows_expected=2)
    cov.record_ok(1.0)
    manifest = cov.as_manifest()
    cov.record_ok(1.0)
    assert manifest["rows_checked"] == 1


# --- placeholder accounting --------------------------------------------------


def test_placeholder_counts_are_per_row_and_need_no_tensor() -> None:
    """Token ids, not tensors: the count runs before any model exists."""
    rows: list[list[int]] = [[1, 2, 3], [], [7, 7, 7, 7], [7]]
    assert audio_placeholder_counts(rows, 7) == [0, 0, 4, 1]


def test_placeholder_agreement_is_an_empty_mismatch() -> None:
    """P0's measurement (147/121/312/248 on both sides) checked here without a GPU."""
    assert placeholder_mismatches([147, 121, 312, 248], [147, 121, 312, 248]) == []


def test_placeholder_mismatches_name_the_rows_where_counts_disagree() -> None:
    got = placeholder_mismatches([147, 2, 248], [147, 121, 247])
    assert got == [(1, 2, 121), (2, 248, 247)], "row index, placeholders, tower length"


def test_placeholder_a_denominator_mismatch_is_not_a_pass() -> None:
    """zip() would answer "no row disagreed" over rows that were never compared."""
    with pytest.raises(ValueError, match="denominator mismatch"):
        placeholder_mismatches([1, 2], [1, 2, 3])


# --- the torch-free import contract -------------------------------------------


def test_the_module_imports_under_a_bare_interpreter_with_no_heavy_stack() -> None:
    """If deleted: the validation plane starts requiring the training stack to exist.

    In-process asserts cannot see a module name BEFORE importing the module that is
    supposed not to reach it, so a fresh interpreter is the only honest instrument.
    The assert lives in the child so a failure names the stack that leaked in.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        "import foundationscale.train.audio, sys; "
        "loaded = [m for m in ('torch', 'numpy', 'soundfile', 'transformers') "
        "if m in sys.modules]; "
        "assert not loaded, f'bare import pulled in {loaded}'"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"child stdout={done.stdout!r} stderr={done.stderr!r}"


def test_reset_zeroes_every_count_and_keeps_the_object() -> None:
    """The survival probe's rows must leave no trace in the training-row record."""
    cov = AudioCoverage(rows_expected=2, sampling_rate=16000)
    cov.record_ok(1.0)
    cov.record_refused("too_long")
    cov.placeholder_rows_verified = 1
    cov.placeholder_rows_unmeasured = 1
    same = cov
    cov.reset()
    assert same is cov
    assert cov.as_manifest() == AudioCoverage(rows_expected=0).as_manifest()


def test_full_train_modules_come_from_the_family_and_only_from_audio_towers() -> None:
    """The measured declaration is used as-is; another modality's modules are not audio's."""
    from foundationscale.families.registry import REGISTRY
    from foundationscale.train.audio import audio_full_train_modules

    gemma4 = next(spec for spec in REGISTRY if spec.name == "gemma4")
    assert audio_full_train_modules(gemma4) == [
        "model.audio_tower",
        "model.embed_audio.embedding_projection",
    ]
    mixed = SimpleNamespace(
        towers=(("v.tower", "image"), ("a.tower", "audio")),
        adapter_full_train=("v.tower", "a.tower.proj"),
    )
    assert audio_full_train_modules(mixed) == ["a.tower.proj"]
    assert audio_full_train_modules(SimpleNamespace(towers=(("a.tower", "audio"),))) == []
    assert audio_full_train_modules(None) == []


def test_full_train_modules_for_the_measured_speech_families() -> None:
    """Whisper and Qwen2-Audio wrap their audio roots; Parakeet its subsampling front end."""
    from foundationscale.families.registry import REGISTRY
    from foundationscale.train.audio import audio_full_train_modules

    by_name = {spec.name: spec for spec in REGISTRY}
    assert audio_full_train_modules(by_name["whisper"]) == ["model.encoder"]
    assert audio_full_train_modules(by_name["qwen2_audio"]) == [
        "audio_tower",
        "multi_modal_projector",
    ]
    # Parakeet trains its subsampling front end in full and adapts its conformer layers.
    assert audio_full_train_modules(by_name["parakeet_ctc"]) == ["encoder.subsampling"]


def test_qwen2_audio_placeholder_formula_matches_the_processor() -> None:
    """The copied formula: frames -> stride-2 conv -> stride-2 pool (transformers 5.5)."""
    from foundationscale.train.audio import _MASK_LENGTH_TOKEN_FORMULAS, _qwen2_audio_tokens

    assert _MASK_LENGTH_TOKEN_FORMULAS["Qwen2AudioProcessor"] is _qwen2_audio_tokens
    # 3000 frames (a full 30 s window) -> 1500 after the conv -> 750 tokens.
    assert _qwen2_audio_tokens(3000) == 750
    assert _qwen2_audio_tokens(1001) == 250  # 1001 -> 501 -> 250
    assert _qwen2_audio_tokens(1) == 0  # (1 - 2) // 2 floors to -1
