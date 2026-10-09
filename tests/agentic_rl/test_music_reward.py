"""Tests for the music reward port (slice 4c): no network, no real abc2midi.

Every MIDI byte sequence here is built in-process by the tiny writer below;
every ``abc2midi`` invocation is a fake, executable shell script written to
``tmp_path`` that copies a pre-built MIDI file to the ``-o`` path (or emits
"Error" lines, produces no output, or sleeps past the timeout) -- never the
real binary.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.rewards import music as rewards_music
from foundationscale.agentic_rl.rewards.music import (
    MusicReward,
    MusicRewardRefusal,
    MusicScore,
)
from foundationscale.agentic_rl.rewards.music import core as music_core
from foundationscale.agentic_rl.rewards.music import feats as music_feats
from foundationscale.agentic_rl.rewards.music import pipeline as music_pipeline
from foundationscale.agentic_rl.rewards.music.baseline import REF_FULL4K

# `foundationscale.agentic_rl.rewards.music`'s __init__ re-exports the FUNCTION
# named ``score`` from the submodule of the same name (``from .score import
# score``), which rebinds ``score`` as a package ATTRIBUTE -- and
# ``import a.b.c as x`` walks that same attribute chain, so it is just as
# shadowed. ``sys.modules`` is keyed by the dotted string instead and is not
# affected by the rebind, so that is how the submodule is reached here.
music_score = sys.modules["foundationscale.agentic_rl.rewards.music.score"]

# ---------------------------------------------------------------------------
# A tiny standard-MIDI-file writer, built in-test (no third-party MIDI lib).
# ---------------------------------------------------------------------------


def _vlq(n: int) -> bytes:
    """Encode ``n`` as a MIDI variable-length quantity."""
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _track(events: list[tuple[int, bytes]]) -> bytes:
    """Build one MTrk chunk from (delta_time, raw_event_bytes) pairs."""
    data = b"".join(_vlq(dt) + ev for dt, ev in events)
    data += b"\x00\xff\x2f\x00"  # end of track
    return b"MTrk" + struct.pack(">I", len(data)) + data


def _midi(tracks: list[bytes], division: int = 480, fmt: int = 1) -> bytes:
    """Build a full Standard MIDI File from a list of pre-built MTrk chunks."""
    header = b"MThd" + struct.pack(">IHHH", 6, fmt, len(tracks), division)
    return header + b"".join(tracks)


def _note_on(ch: int, pitch: int, vel: int = 90) -> bytes:
    return bytes([0x90 | ch, pitch, vel])


def _note_off(ch: int, pitch: int) -> bytes:
    return bytes([0x80 | ch, pitch, 0])


def _tempo_event(usec_per_qtr: int) -> bytes:
    return bytes([0xFF, 0x51, 0x03]) + usec_per_qtr.to_bytes(3, "big")


def _rich_midi_bytes() -> bytes:
    """A melody (ch0) + harmony (ch1) long enough to exercise every
    length-gated branch in ``feats.analyze`` (len(mel) > 24, len(ioi) > 12,
    len(strain) > 8), plus a drum channel, a program change, a control
    change, a channel-pressure event, a sysex block, a non-tempo meta event,
    and one note-on reached purely through MIDI running status (no repeated
    status byte) -- so ``core.parse_midi`` exercises every event-type branch
    in the same fixture used for the feature-level assertions.
    """
    events: list[tuple[int, bytes]] = [
        (0, bytes([0xFF, 0x03, 4]) + b"Test"),  # track name meta (non-tempo)
        (0, _tempo_event(500_000)),  # 120 bpm
        (0, bytes([0xC0, 5])),  # program change, ch0 -> program 5
        (0, bytes([0xB0, 7, 100])),  # control change, ch0
        (0, bytes([0xD0, 64])),  # channel pressure, ch0
        (0, bytes([0xF0]) + _vlq(3) + b"\x7e\x00\xf7"),  # sysex block
        (0, bytes([0xE0, 0, 64])),  # pitch bend, ch0 (unmapped -> else: j+=2)
    ]
    dur = 240
    pitches = [60, 62, 64, 65, 67, 69, 71, 72] * 5  # 40 notes, C major x5
    for i, p in enumerate(pitches):
        events.append((0 if i == 0 else dur, _note_on(0, p, 90)))
        if i % 3 == 0:
            events.append((0, _note_on(1, (p + 7) % 128, 70)))  # harmony
        if i == 7:
            # running status: a second ch0 note-on with NO status byte,
            # immediately after the one above (same status => omittable).
            events.append((dur, bytes([p + 2, 95])))
            events.append((dur, _note_off(0, p + 2)))
        events.append((dur, _note_off(0, p)))
        if i % 3 == 0:
            events.append((0, _note_off(1, (p + 7) % 128)))
    # a couple of large leaps for large_leap_rate / leap_resolution coverage
    events.append((dur, _note_on(0, 40, 100)))
    events.append((dur, _note_off(0, 40)))
    events.append((0, _note_on(0, 80, 100)))
    events.append((dur, _note_off(0, 80)))
    # one drum note (channel 9), excluded from pitched analysis
    events.append((0, _note_on(9, 36, 100)))
    events.append((dur, _note_off(9, 36)))
    return _midi([_track(events)])


def _single_note_midi_bytes() -> bytes:
    """Exactly one pitched note: exercises every empty-list branch in
    ``feats.analyze`` (``aiv``, ``ioi``, ``mom``, ``diam`` all empty).
    """
    events = [
        (0, _tempo_event(500_000)),
        (0, _note_on(0, 60, 90)),
        (240, _note_off(0, 60)),
    ]
    return _midi([_track(events)])


def _no_notes_midi_bytes() -> bytes:
    events = [(0, _tempo_event(500_000))]
    return _midi([_track(events)])


def _drums_only_midi_bytes() -> bytes:
    events = [
        (0, _tempo_event(500_000)),
        (0, _note_on(9, 36, 100)),
        (240, _note_off(9, 36)),
    ]
    return _midi([_track(events)])


def _zero_tempo_midi_bytes() -> bytes:
    """A valid-looking MIDI whose tempo meta event is 0 usec/quarter: valid
    enough for ``core.parse_midi`` (which never inspects tempo) but makes
    ``feats.analyze`` divide by zero -- the "unexpected exception inside
    feature extraction" case :func:`~foundationscale.agentic_rl.rewards.music.pipeline.do`
    catches into ``scorer_skip``.
    """
    events = [
        (0, _tempo_event(0)),
        (0, _note_on(0, 60, 90)),
        (240, _note_off(0, 60)),
    ]
    return _midi([_track(events)])


def _bad_header_bytes() -> bytes:
    return b"not a midi file at all"


def _smpte_division_midi_bytes() -> bytes:
    """A division with the SMPTE flag (bit 15) set -- ``core.parse_midi``
    refuses it (``div & 0x8000``) even though it is a nonzero value.
    """
    return _midi([_track([(0, _tempo_event(500_000))])], division=0x8080)


def _truncated_ntrk_midi_bytes() -> bytes:
    """Header declares two tracks but only one MTrk chunk follows -- the
    parser's ``if data[i : i + 4] != b"MTrk": break`` path.
    """
    header = b"MThd" + struct.pack(">IHHH", 6, 1, 2, 480)
    one_track = _track(
        [(0, _tempo_event(500_000)), (0, _note_on(0, 60, 90)), (240, _note_off(0, 60))]
    )
    return header + one_track


# ---------------------------------------------------------------------------
# core.parse_midi
# ---------------------------------------------------------------------------


def test_parse_midi_rejects_non_midi_bytes() -> None:
    assert music_core.parse_midi(_bad_header_bytes()) is None


def test_parse_midi_rejects_zero_division() -> None:
    data = b"MThd" + struct.pack(">IHHH", 6, 1, 1, 0) + _track([(0, _tempo_event(500_000))])
    assert music_core.parse_midi(data) is None


def test_parse_midi_rejects_smpte_division() -> None:
    assert music_core.parse_midi(_smpte_division_midi_bytes()) is None


def test_parse_midi_stops_at_missing_track_chunk() -> None:
    m = music_core.parse_midi(_truncated_ntrk_midi_bytes())
    assert m is not None
    assert len(m["notes"]) == 1


def test_parse_midi_parses_every_event_type_and_running_status() -> None:
    m = music_core.parse_midi(_rich_midi_bytes())
    assert m is not None
    assert m["div"] == 480
    assert m["tempo"] == 500_000
    assert m["progs"][0] == 5  # program change captured
    # 40 melody notes (+1 running-status note), 14 harmony notes, 2 leap
    # notes, 1 drum note.
    assert len(m["notes"]) == 40 + 1 + 14 + 2 + 1


def test_parse_midi_treats_zero_velocity_note_on_as_note_off() -> None:
    events = [
        (0, _tempo_event(500_000)),
        (0, _note_on(0, 60, 90)),
        (240, bytes([0x90, 60, 0])),  # note-on velocity 0 == note-off
    ]
    m = music_core.parse_midi(_midi([_track(events)]))
    assert m is not None
    assert m["notes"] == [(0, 240, 0, 60, 90)]


# ---------------------------------------------------------------------------
# core.py helper functions not fully reached through analyze()
# ---------------------------------------------------------------------------


def test_f0_a440() -> None:
    assert music_core.f0(69) == pytest.approx(440.0)


def test_roughness_requires_at_least_two_pitches() -> None:
    assert music_core.roughness([60]) == 0.0
    assert music_core.roughness([]) == 0.0


def test_roughness_without_velocities_uses_unit_amplitude() -> None:
    assert music_core.roughness([60, 61]) > 0.0


def test_harmonicity_degenerate_cases() -> None:
    assert music_core.harmonicity([]) == 0.0
    assert music_core.harmonicity([60]) == 1.0
    assert 0.0 <= music_core.harmonicity([60, 64, 67]) <= 1.0


def test_ce_empty_pitches_is_none() -> None:
    assert music_core._ce([]) is None
    coord = music_core._ce([60], [1.0])
    assert coord is not None
    assert len(coord) == 3


def test_dist_with_a_missing_endpoint_is_zero() -> None:
    assert music_core._dist(None, (0.0, 0.0, 0.0)) == 0.0
    assert music_core._dist((0.0, 0.0, 0.0), None) == 0.0
    assert music_core._dist((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == pytest.approx(math.sqrt(3))


def test_krumhansl_key_handles_an_empty_histogram() -> None:
    tonic, minor, r1, r2 = music_core.krumhansl_key([0.0] * 12)
    assert isinstance(tonic, int)
    assert minor in (0, 1)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_krumhansl_key_prefers_c_major_for_a_c_major_histogram() -> None:
    hist = [0.0] * 12
    for pc in (0, 2, 4, 5, 7, 9, 11):
        hist[pc] = 1.0
    hist[0] = 5.0  # strong tonic weight
    tonic, minor, _r1, _r2 = music_core.krumhansl_key(hist)
    assert tonic == 0
    assert minor == 0


def test_ngram_contexts_of_every_length() -> None:
    seq = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    ng = music_core.NGram(seq, 3)
    p0 = ng.p((), 0)
    p1 = ng.p((1,), 2)
    p2 = ng.p((2, 1), 0)
    assert 0.0 < p0 <= 1.0
    assert 0.0 < p1 <= 1.0
    assert 0.0 < p2 <= 1.0
    ic, ent, ics = ng.ic_entropy(seq)
    assert isinstance(ic, float)
    assert isinstance(ent, float)
    assert len(ics) == len(seq)


def test_ngram_ic_entropy_of_an_empty_sequence_is_zero() -> None:
    ng = music_core.NGram([0, 1, 2], 3)
    ic, ent, ics = ng.ic_entropy([])
    assert ic == 0.0
    assert ent == 0.0
    assert ics == []


# ---------------------------------------------------------------------------
# feats.analyze
# ---------------------------------------------------------------------------


def test_analyze_skips_a_non_midi_file(tmp_path: Path) -> None:
    p = tmp_path / "a.mid"
    p.write_bytes(_bad_header_bytes())
    assert music_feats.analyze(p) == {"skip": "bad_midi"}


def test_analyze_skips_a_file_with_no_notes(tmp_path: Path) -> None:
    p = tmp_path / "a.mid"
    p.write_bytes(_no_notes_midi_bytes())
    assert music_feats.analyze(p) == {"skip": "no_notes"}


def test_analyze_skips_a_drums_only_file(tmp_path: Path) -> None:
    p = tmp_path / "a.mid"
    p.write_bytes(_drums_only_midi_bytes())
    assert music_feats.analyze(p) == {"skip": "drums_only"}


def test_analyze_raises_on_a_zero_tempo_file(tmp_path: Path) -> None:
    p = tmp_path / "a.mid"
    p.write_bytes(_zero_tempo_midi_bytes())
    with pytest.raises(ZeroDivisionError):
        music_feats.analyze(p)


def test_analyze_single_note_hits_every_empty_list_branch(tmp_path: Path) -> None:
    p = tmp_path / "a.mid"
    p.write_bytes(_single_note_midi_bytes())
    feat = music_feats.analyze(p)
    assert "skip" not in feat
    assert feat["n_note"] == 1
    assert feat["mel_interval_mean"] == 0.0
    assert feat["large_leap_rate"] == 0.0
    assert feat["leap_resolution"] == 1.0
    assert feat["ioi_mean"] == 0.0
    assert feat["cloud_momentum_mean"] == 0.0
    assert feat["cloud_diameter_mean"] == 0.0
    # short-sequence branches (len(mel) <= 12 and <= 24)
    assert feat["surprisal_mean"] == 0.0
    assert feat["motif_recurrence"] == 0.0


def test_analyze_rich_midi_computes_a_full_feature_vector(tmp_path: Path) -> None:
    p = tmp_path / "a.mid"
    p.write_bytes(_rich_midi_bytes())
    feat = music_feats.analyze(p)
    assert "skip" not in feat
    assert feat["n_chan"] == 2
    assert feat["n_drum"] == 1
    assert feat["polyphony_rate"] > 0.0
    assert feat["roughness_mean"] > 0.0
    assert feat["harmonicity_mean"] > 0.0
    # long-sequence branches (len(mel) > 12, > 24; len(ioi) > 12; strain > 8)
    assert feat["surprisal_mean"] > 0.0
    assert feat["motif_recurrence"] >= 0.0
    assert feat["tension_peaks"] >= 0
    for key in ("_pc_hist", "_iv_hist", "_dur_hist"):
        assert key in feat
    assert len(feat["_pc_hist"]) == 12
    assert len(feat["_iv_hist"]) == 13
    assert len(feat["_dur_hist"]) == 8


# ---------------------------------------------------------------------------
# score.py
# ---------------------------------------------------------------------------


def test_score_returns_none_for_a_skipped_feature_dict() -> None:
    assert music_score.score({"skip": "bad_midi"}, REF_FULL4K) is None


def _ideal_feat() -> dict[str, Any]:
    """A feature dict scoring 1.0 on every SPEC entry: ``p50`` for a "band"
    feature (any point in [p10, p90] does), ``p25`` for "low" (``_low``
    scores 1.0 for any x in [p05, p25]), ``p75`` for "high" (``_high`` scores
    1.0 for any x >= p75).
    """
    feat: dict[str, Any] = {}
    for name, kind, _group in music_score.SPEC:
        r = REF_FULL4K[name]
        feat[name] = {"band": r["p50"], "low": r["p25"], "high": r["p75"]}[kind]
    hist = REF_FULL4K["_hist"]
    feat["_pc_hist"] = list(hist["_pc_hist"])
    feat["_iv_hist"] = list(hist["_iv_hist"])
    feat["_dur_hist"] = list(hist["_dur_hist"])
    return feat


def test_score_of_the_ideal_human_feature_vector_is_one_hundred() -> None:
    result = music_score.score(_ideal_feat(), REF_FULL4K)
    assert result is not None
    assert result["total"] == pytest.approx(100.0)
    assert all(v == pytest.approx(1.0) for v in result["per_feature"].values())
    assert all(v == pytest.approx(100.0) for v in result["groups"].values())


def test_score_ignores_a_feature_absent_from_the_dict_and_defaults_missing_groups() -> None:
    feat = _ideal_feat()
    del feat["tension_peaks"]
    result = music_score.score(feat, REF_FULL4K)
    assert result is not None
    assert "tension_peaks" not in result["per_feature"]


def test_score_without_any_histogram_falls_back_to_the_default_distance_penalty() -> None:
    feat = _ideal_feat()
    for key in ("_pc_hist", "_iv_hist", "_dur_hist"):
        del feat[key]
    result = music_score.score(feat, REF_FULL4K)
    assert result is not None
    assert result["js_mean"] == pytest.approx(0.3)


def test_score_of_an_extreme_feature_vector_is_low() -> None:
    feat: dict[str, Any] = {}
    for name, _kind, _group in music_score.SPEC:
        r = REF_FULL4K[name]
        feat[name] = r["lo"] - abs(r["lo"]) - 1000.0
    result = music_score.score(feat, REF_FULL4K)
    assert result is not None
    assert result["total"] < 50.0


def test_band_in_range_and_both_out_of_range_tails() -> None:
    r = REF_FULL4K["note_density"]
    assert music_score._band(r["p50"], r) == 1.0
    assert music_score._band(r["lo"], r) < 1.0
    assert music_score._band(r["hi"], r) < 1.0


def test_low_scorer_full_branch_matrix() -> None:
    r = REF_FULL4K["polyphony_rate"]
    assert music_score._low(r["p05"], r) == 1.0  # <= p25 and >= p05
    assert music_score._low(r["lo"] - 1.0, r) < 1.0  # <= p25 and < p05
    assert music_score._low(r["p95"], r) < 1.0  # > p25, fractional span


def test_high_scorer_both_branches() -> None:
    r = REF_FULL4K["key_certainty"]
    assert music_score._high(r["p90"], r) == 1.0
    assert music_score._high(r["lo"], r) < 1.0


def test_js_of_identical_distributions_is_near_zero() -> None:
    p = [0.25, 0.25, 0.25, 0.25]
    assert music_score.js(p, list(p)) == pytest.approx(0.0, abs=1e-9)


def test_js_of_disjoint_distributions_is_positive() -> None:
    p = [1.0, 0.0, 0.0, 0.0]
    q = [0.0, 0.0, 0.0, 1.0]
    assert music_score.js(p, q) > 0.5


def test_build_ref_filters_by_group_and_skip(tmp_path: Path) -> None:
    rows = []
    for i in range(20):
        row: dict[str, Any] = {"group": "human", "skip": None}
        for name, _kind, _group in music_score.SPEC:
            row[name] = float(i)
        row["_pc_hist"] = [1.0 / 12] * 12
        row["_iv_hist"] = [1.0 / 13] * 13
        row["_dur_hist"] = [1.0 / 8] * 8
        rows.append(row)
    rows.append({**rows[0], "group": "model"})  # filtered out by group
    rows.append({**rows[0], "skip": "bad"})  # filtered out by skip
    dist_path = tmp_path / "dist.json"
    dist_path.write_text(__import__("json").dumps(rows), encoding="utf-8")

    ref = music_score.build_ref(dist_path, group="human")
    assert set(name for name, _k, _g in music_score.SPEC) <= set(ref)
    assert ref["note_density"]["lo"] == 0.0
    assert ref["note_density"]["hi"] == 19.0
    assert len(ref["_hist"]["_pc_hist"]) == 12


# ---------------------------------------------------------------------------
# pipeline.extract_abc
# ---------------------------------------------------------------------------


def test_extract_abc_from_a_fenced_block() -> None:
    text = "Here is a tune:\n```abc\nX:1\nK:C\nCDEF|\n```\nHope you like it."
    assert music_pipeline.extract_abc(text) == "X:1\nK:C\nCDEF|"


def test_extract_abc_from_bare_text() -> None:
    text = "sure, here it is:\nX:1\nK:C\nCDEF|"
    assert music_pipeline.extract_abc(text) == "X:1\nK:C\nCDEF|"


def test_extract_abc_returns_none_when_nothing_matches() -> None:
    assert music_pipeline.extract_abc("just chatting, no music here") is None
    assert music_pipeline.extract_abc("") is None
    assert music_pipeline.extract_abc(None) is None


def test_extract_abc_falls_back_past_a_fence_with_no_abc_marker() -> None:
    text = "```text\nnot abc\n```\nbut X:1\nK:C\nCDEF|"
    assert music_pipeline.extract_abc(text) == "X:1\nK:C\nCDEF|"


def test_extract_abc_picks_the_last_bare_match() -> None:
    text = "X:1\nK:C\nC|\nactually scratch that, X:2\nK:D\nD|"
    assert music_pipeline.extract_abc(text) == "X:2\nK:D\nD|"


# ---------------------------------------------------------------------------
# pipeline._midi_notes_progs
# ---------------------------------------------------------------------------


def test_midi_notes_progs_rejects_non_midi_bytes() -> None:
    assert music_pipeline._midi_notes_progs(_bad_header_bytes()) is None


def test_midi_notes_progs_rejects_zero_division() -> None:
    data = b"MThd" + struct.pack(">IHHH", 6, 1, 1, 0) + _track([(0, _tempo_event(500_000))])
    assert music_pipeline._midi_notes_progs(data) is None


def test_midi_notes_progs_reads_notes_and_program_changes() -> None:
    notes, progs = music_pipeline._midi_notes_progs(_rich_midi_bytes())
    assert len(notes) > 0
    assert (0, 5) in progs


# ---------------------------------------------------------------------------
# A fake abc2midi, written as an executable shell script in tmp_path.
# ---------------------------------------------------------------------------


def _write_fake_abc2midi(tmp_path: Path, body: str, name: str = "abc2midi") -> str:
    script = tmp_path / name
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def _write_prebuilt_midi(tmp_path: Path, data: bytes, name: str = "prebuilt.mid") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _copies_prebuilt_script(tmp_path: Path, midi_path: Path, extra: str = "") -> str:
    return _write_fake_abc2midi(tmp_path, f'cp "{midi_path}" "$3"\n{extra}')


# ---------------------------------------------------------------------------
# pipeline.do
# ---------------------------------------------------------------------------


def test_do_scores_a_successful_render(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _rich_midi_bytes())
    script = _copies_prebuilt_script(tmp_path, midi_path)
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    r = music_pipeline.do(rec, script, 10.0)
    assert r["reject"] == 0
    assert "total" in r
    assert 0.0 <= r["total"] <= 100.0


def test_do_rejects_on_error_lines(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _rich_midi_bytes())
    script = _copies_prebuilt_script(tmp_path, midi_path, extra='echo "Error: bad note"')
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    r = music_pipeline.do(rec, script, 10.0)
    assert r["err"] >= 1
    assert r["reject"] == 1


def test_do_rejects_on_too_many_bar_errors(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _rich_midi_bytes())
    bar_lines = "\n".join(f'echo "Bar {i} has a problem"' for i in range(10))
    script = _copies_prebuilt_script(tmp_path, midi_path, extra=bar_lines)
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    r = music_pipeline.do(rec, script, 10.0)
    assert r["bar"] == 10
    assert r["reject"] == 1


def test_do_rejects_on_a_blank_line_in_the_abc(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _rich_midi_bytes())
    script = _copies_prebuilt_script(tmp_path, midi_path)
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\n\nCDEF|"}
    r = music_pipeline.do(rec, script, 10.0)
    assert r["blank"] == 1
    assert r["reject"] == 1


def test_do_rejects_on_channel_program_conflict(tmp_path: Path) -> None:
    events = [
        (0, _tempo_event(500_000)),
        (0, bytes([0xC0, 1])),  # ch0 -> program 1
        (0, _note_on(0, 60, 90)),
        (240, _note_off(0, 60)),
        (0, bytes([0xC0, 2])),  # ch0 -> program 2 (conflict)
        (0, _note_on(0, 62, 90)),
        (240, _note_off(0, 62)),
    ]
    conflict_midi = _midi([_track(events)])
    midi_path = _write_prebuilt_midi(tmp_path, conflict_midi)
    script = _copies_prebuilt_script(tmp_path, midi_path)
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    r = music_pipeline.do(rec, script, 10.0)
    assert r["ch_conflict"] == 1
    assert r["reject"] == 1


def test_do_skips_empty_abc(tmp_path: Path) -> None:
    script = _write_fake_abc2midi(tmp_path, "exit 0")
    r = music_pipeline.do({"key": "k", "id": 0, "rep": 0, "abc": "   \n  "}, script, 10.0)
    assert r["skip"] == "empty_abc"


def test_do_skips_when_abc2midi_produces_no_output(tmp_path: Path) -> None:
    script = _write_fake_abc2midi(tmp_path, "exit 0")
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    r = music_pipeline.do(rec, script, 10.0)
    assert r["skip"] == "no_midi"


def test_do_skips_on_an_unscoreable_feature_extraction(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _zero_tempo_midi_bytes())
    script = _copies_prebuilt_script(tmp_path, midi_path)
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    r = music_pipeline.do(rec, script, 10.0)
    assert r["scorer_skip"].startswith("analyze:")
    assert "total" not in r


def test_do_raises_abc2midi_missing_for_a_nonexistent_binary(tmp_path: Path) -> None:
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    with pytest.raises(music_pipeline.Abc2MidiMissing):
        music_pipeline.do(rec, str(tmp_path / "does-not-exist"), 10.0)


def test_do_raises_abc2midi_error_for_an_os_error(tmp_path: Path) -> None:
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    with pytest.raises(music_pipeline.Abc2MidiError) as excinfo:
        music_pipeline.do(rec, str(tmp_path), 10.0)  # a directory, not a binary
    assert excinfo.value.cause_name in ("PermissionError", "IsADirectoryError")


def test_do_raises_abc2midi_error_on_timeout(tmp_path: Path) -> None:
    script = _write_fake_abc2midi(tmp_path, "sleep 5")
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    with pytest.raises(music_pipeline.Abc2MidiError) as excinfo:
        music_pipeline.do(rec, script, 0.2)
    assert excinfo.value.cause_name == "TimeoutExpired"


def test_do_falls_back_to_a_generic_skip_for_a_non_subprocess_error() -> None:
    rec = {"key": "k", "id": 0, "rep": 0, "abc": "X:1\nK:C\nCDEF|"}
    r = music_pipeline.do(rec, 12345, 10.0)  # type: ignore[arg-type]
    assert r["skip"].startswith("abc2midi:")


# ---------------------------------------------------------------------------
# pipeline.compute_score (upstream API parity, kept for reference)
# ---------------------------------------------------------------------------


def test_compute_score_returns_zero_for_no_abc() -> None:
    assert music_pipeline.compute_score(None, "no music here") == 0.0


def test_compute_score_scores_a_successful_render(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _rich_midi_bytes())
    script = _copies_prebuilt_script(tmp_path, midi_path)
    value = music_pipeline.compute_score(
        None, "X:1\nK:C\nCDEF|", abc2midi_bin=script, timeout_s=10.0
    )
    assert 0.0 <= value <= 1.0


def test_compute_score_reraises_abc2midi_missing(tmp_path: Path) -> None:
    with pytest.raises(music_pipeline.Abc2MidiMissing):
        music_pipeline.compute_score(None, "X:1\nK:C\nCDEF|", abc2midi_bin=str(tmp_path / "nope"))


def test_compute_score_swallows_an_infra_timeout_as_zero(tmp_path: Path) -> None:
    script = _write_fake_abc2midi(tmp_path, "sleep 5")
    value = music_pipeline.compute_score(
        None, "X:1\nK:C\nCDEF|", abc2midi_bin=script, timeout_s=0.2
    )
    assert value == 0.0


# ---------------------------------------------------------------------------
# MusicScore / MusicReward construction refusals
# ---------------------------------------------------------------------------


def test_music_reward_refuses_an_empty_binary_path() -> None:
    with pytest.raises(MusicRewardRefusal, match="abc2midi_bin"):
        MusicReward(abc2midi_bin="")


def test_music_reward_refuses_a_non_str_binary_path() -> None:
    with pytest.raises(MusicRewardRefusal, match="abc2midi_bin"):
        MusicReward(abc2midi_bin=123)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan"), float("inf"), True])
def test_music_reward_refuses_a_bad_timeout(timeout_s: float) -> None:
    with pytest.raises(MusicRewardRefusal, match="timeout_s"):
        MusicReward(abc2midi_bin="abc2midi", timeout_s=timeout_s)


def test_music_reward_accepts_a_valid_configuration() -> None:
    reward = MusicReward(abc2midi_bin="abc2midi")
    assert reward.timeout_s == 60.0


def test_music_score_refuses_a_bool_value() -> None:
    with pytest.raises(MusicRewardRefusal, match="value"):
        MusicScore(True, None, {})  # type: ignore[arg-type]


def test_music_score_refuses_a_nan_value() -> None:
    with pytest.raises(MusicRewardRefusal, match="value"):
        MusicScore(float("nan"), None, {})


def test_music_score_refuses_an_empty_reason() -> None:
    with pytest.raises(MusicRewardRefusal, match="reason"):
        MusicScore(0.5, "", {})


def test_music_score_refuses_none_value_without_an_infra_reason() -> None:
    with pytest.raises(MusicRewardRefusal, match="value"):
        MusicScore(None, None, {})
    with pytest.raises(MusicRewardRefusal, match="value"):
        MusicScore(None, "rejected", {})


def test_music_score_refuses_non_mapping_details() -> None:
    with pytest.raises(MusicRewardRefusal, match="details"):
        MusicScore(0.5, None, [1, 2, 3])  # type: ignore[arg-type]


def test_music_score_accepts_every_legitimate_shape() -> None:
    MusicScore(0.73, None, {"total": 73})
    MusicScore(0.0, "no_abc", {})
    MusicScore(None, "infra:abc2midi_missing", {})


# ---------------------------------------------------------------------------
# MusicReward.score end to end
# ---------------------------------------------------------------------------


def test_music_reward_score_no_abc_in_response() -> None:
    reward = MusicReward(abc2midi_bin="abc2midi")
    result = reward.score("just chatting, no music here")
    assert result == MusicScore(0.0, "no_abc", {})


def test_music_reward_score_successful_render(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _rich_midi_bytes())
    script = _copies_prebuilt_script(tmp_path, midi_path)
    reward = MusicReward(abc2midi_bin=script, timeout_s=10.0)
    result = reward.score("```abc\nX:1\nK:C\nCDEF|\n```")
    assert result.reason is None
    assert result.value is not None
    assert 0.0 <= result.value <= 1.0
    assert result.details["reject"] == 0


def test_music_reward_score_rejected_render(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _rich_midi_bytes())
    script = _copies_prebuilt_script(tmp_path, midi_path, extra='echo "Error: bad note"')
    reward = MusicReward(abc2midi_bin=script, timeout_s=10.0)
    result = reward.score("X:1\nK:C\nCDEF|")
    assert result.value == 0.0
    assert result.reason == "rejected"


def test_music_reward_score_unscored_no_output(tmp_path: Path) -> None:
    script = _write_fake_abc2midi(tmp_path, "exit 0")
    reward = MusicReward(abc2midi_bin=script, timeout_s=10.0)
    result = reward.score("X:1\nK:C\nCDEF|")
    assert result.value == 0.0
    assert result.reason == "unscored:no_midi"


def test_music_reward_score_unscored_analyze_failure(tmp_path: Path) -> None:
    midi_path = _write_prebuilt_midi(tmp_path, _zero_tempo_midi_bytes())
    script = _copies_prebuilt_script(tmp_path, midi_path)
    reward = MusicReward(abc2midi_bin=script, timeout_s=10.0)
    result = reward.score("X:1\nK:C\nCDEF|")
    assert result.value == 0.0
    assert result.reason is not None
    assert result.reason.startswith("unscored:analyze:")


def test_music_reward_score_infra_missing_binary(tmp_path: Path) -> None:
    reward = MusicReward(abc2midi_bin=str(tmp_path / "nope"), timeout_s=10.0)
    result = reward.score("X:1\nK:C\nCDEF|")
    assert result == MusicScore(None, "infra:abc2midi_missing", {})


def test_music_reward_score_infra_os_error(tmp_path: Path) -> None:
    reward = MusicReward(abc2midi_bin=str(tmp_path), timeout_s=10.0)
    result = reward.score("X:1\nK:C\nCDEF|")
    assert result.value is None
    assert result.reason is not None
    assert result.reason.startswith("infra:abc2midi_error:")


def test_music_reward_score_infra_timeout(tmp_path: Path) -> None:
    script = _write_fake_abc2midi(tmp_path, "sleep 5")
    reward = MusicReward(abc2midi_bin=script, timeout_s=0.2)
    result = reward.score("X:1\nK:C\nCDEF|")
    assert result == MusicScore(None, "infra:abc2midi_error:TimeoutExpired", {})


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_rewards_music_reexports_the_scorer_internals() -> None:
    assert rewards_music.SPEC is music_score.SPEC
    assert rewards_music.W is music_score.W
    assert rewards_music.do is music_pipeline.do
    assert rewards_music.extract_abc is music_pipeline.extract_abc
    assert rewards_music.analyze is music_feats.analyze
    assert rewards_music.parse_midi is music_core.parse_midi
    assert rewards_music.REF_FULL4K is REF_FULL4K


def test_rewards_package_reexports_the_adapter() -> None:
    from foundationscale.agentic_rl import rewards

    assert rewards.MusicReward is MusicReward
    assert rewards.MusicScore is MusicScore
