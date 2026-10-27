"""speech_longform pause_cut_merge.py and build_concat.py, on synthetic audio only.

Cuts land in pauses and never exceed the cap; cut pieces reassemble the recording; merge orders
pieces numerically and accounts for a recording nothing was decoded for; concatenation planning
respects its limits. No model, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_DIR = Path(__file__).resolve().parents[2] / "validation_campaigns" / "speech_longform"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pcm = _load("pause_cut_merge")
bc = _load("build_concat")
SR = 16000


def _tone(seconds: float, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_pause_bounds_cut_inside_the_silences():
    x = np.concatenate([_tone(10), _silence(2), _tone(10), _silence(2), _tone(10)])
    bounds = pcm.pause_bounds(x, SR, max_s=15.0)
    assert bounds[0] == 0 and bounds[-1] == len(x)
    assert all(b - a <= 15 * SR for a, b in zip(bounds, bounds[1:], strict=False)), (
        "MUST_PASS: every piece <= max_s"
    )
    assert bounds == sorted(set(bounds)), "MUST_PASS: strictly increasing"
    silences = [(10 * SR, 12 * SR), (22 * SR, 24 * SR)]
    for cut in bounds[1:-1]:
        assert any(lo <= cut <= hi for lo, hi in silences), (
            f"MUST_FIRE: cut at {cut / SR:.2f}s is in speech"
        )


def test_pause_bounds_fall_back_to_hard_cuts_without_pauses():
    x = _tone(70)
    bounds = pcm.pause_bounds(x, SR, max_s=28.0)
    assert bounds == [0, 28 * SR, 56 * SR, len(x)], "MUST_PASS: no pause -> cuts at max_s"


def test_short_trailing_piece_joins_the_previous_one():
    x = np.concatenate([_tone(20), _silence(1), _tone(7.5)])  # 28.5 s, cap 28.4 s
    bounds = pcm.pause_bounds(x, SR, max_s=28.4, min_s=1.0)
    assert len(bounds) == 3 and bounds[-1] == len(x)
    tail = _tone(28.3)
    tail_bounds = pcm.pause_bounds(np.concatenate([tail, _tone(0.5)]), SR, max_s=28.4, min_s=1.0)
    # no pause: hard cut at 28.4 s; the 0.4 s tail stays apart (merging would exceed 28.4 s)
    assert tail_bounds == [0, int(28.4 * SR), int(28.8 * SR)], (
        "MUST_PASS: merging never breaks max_s"
    )
    # a 0.5 s tail after a hard cut is merged only when the merged piece still fits
    y = np.concatenate([_tone(28.0), _tone(0.5)])
    yb = pcm.pause_bounds(y, SR, max_s=28.0, min_s=1.0)
    assert all(b - a <= 28 * SR for a, b in zip(yb, yb[1:], strict=False)), (
        "MUST_PASS: merging never breaks max_s"
    )


def test_bounds_refuse_bad_limits():
    with pytest.raises(ValueError):
        pcm.pause_bounds(_tone(1), SR, max_s=0)
    with pytest.raises(ValueError):
        pcm.fixed_bounds(10, SR, chunk_s=0)


def test_fixed_bounds_cover_the_signal():
    assert pcm.fixed_bounds(65 * SR, SR, 30.0) == [0, 30 * SR, 60 * SR, 65 * SR]
    assert pcm.fixed_bounds(0, SR, 30.0) == [0, 0]


def _write(path: Path, x: np.ndarray, sr: int = SR) -> str:
    sf.write(path, x, sr, format="FLAC", subtype="PCM_16")
    return str(path)


def test_cut_rows_pieces_reassemble_the_recording(tmp_path):
    x = np.concatenate([_tone(12), _silence(1), _tone(12), _silence(1), _tone(12)])
    audio = _write(tmp_path / "rec.flac", x)
    fs_rows = [{"id": "call1", "audio": audio, "answer": "a b c"}]
    nemo_rows = [{"audio_filepath": audio, "duration": len(x) / SR, "text": "a b c"}]
    for rows, rec in ((fs_rows, "call1"), (nemo_rows, "rec")):
        pieces = pcm.cut_rows(rows, str(tmp_path / f"out_{rec}"), mode="pause", max_s=20.0)
        assert [p["id"] for p in pieces] == [f"{rec}::{k:04d}" for k in range(len(pieces))]
        joined = np.concatenate([sf.read(p["audio"], dtype="float32")[0] for p in pieces])
        original = sf.read(audio, dtype="float32")[0]
        assert joined.shape == original.shape and np.allclose(joined, original, atol=1e-4), (
            "MUST_PASS: the pieces are the recording, cut and nothing else"
        )
        assert all(p["duration"] <= 20.0 for p in pieces)
    fixed = pcm.cut_rows(fs_rows, str(tmp_path / "fixed"), mode="fixed", chunk_s=10.0)
    assert len(fixed) == 4


def test_cut_rows_refuses_non_16k_audio_and_bad_mode(tmp_path):
    audio = _write(tmp_path / "rec24.flac", _tone(3)[: 3 * 24000 // 2], sr=24000)
    with pytest.raises(ValueError, match="24000 Hz"):
        pcm.cut_rows([{"id": "r", "audio": audio}], str(tmp_path / "o"))
    with pytest.raises(ValueError, match="mode"):
        pcm.cut_rows([], str(tmp_path / "o"), mode="vad")
    with pytest.raises(ValueError, match="audio"):
        pcm.cut_rows([{"id": "r"}], str(tmp_path / "o"))


def test_merge_orders_pieces_numerically_and_counts_missing():
    seg_eval = {"all": [{"id": f"call1::{k:04d}", "hypothesis": f"w{k}"} for k in (10, 2, 1)]}
    rows = [
        {"id": "call1", "audio": "x.flac", "answer": "ref one"},
        {"audio_filepath": "/d/call2.flac", "text": "ref two", "duration": 9.0},
    ]
    eval_rows, nemo_rows, missing = pcm.merge(seg_eval, rows)
    assert eval_rows[0] == {"id": "call1", "reference": "ref one", "hypothesis": "w1 w2 w10"}
    assert eval_rows[1] == {"id": "call2", "reference": "ref two", "hypothesis": ""}
    assert missing == 1, "MUST_FIRE: a recording with no decoded piece is counted, not dropped"
    assert nemo_rows[1]["duration"] == 9.0 and nemo_rows[1]["pred_text"] == ""
    with pytest.raises(ValueError, match="recording"):
        pcm.merge({"all": [{"id": "no-separator", "hypothesis": "x"}]}, rows)


def test_cli_cut_and_merge(tmp_path):
    x = np.concatenate([_tone(15), _silence(1), _tone(15)])
    audio = _write(tmp_path / "rec.flac", x)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"id": "r1", "audio": audio, "answer": "hello"}) + "\n")
    assert pcm.main(["cut", "--manifest", str(manifest), "--out-dir", str(tmp_path / "cut")]) == 0
    segs = [
        json.loads(line) for line in (tmp_path / "cut" / "segments.jsonl").read_text().splitlines()
    ]
    seg_eval = tmp_path / "seg_eval.json"
    seg_eval.write_text(json.dumps({"all": [{"id": s["id"], "hypothesis": "hello"} for s in segs]}))
    out, nemo = tmp_path / "eval.json", tmp_path / "pred.jsonl"
    argv = [
        "merge",
        "--segments-eval",
        str(seg_eval),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--nemo-out",
        str(nemo),
    ]
    assert pcm.main(argv) == 0
    assert json.loads(out.read_text())["missing"] == 0
    assert json.loads(nemo.read_text().splitlines()[0])["pred_text"] == " ".join(
        ["hello"] * len(segs)
    )
    seg_eval.write_text(json.dumps({"all": []}))
    assert pcm.main(argv) == 5, "MUST_FIRE: merge exits 5 when a recording has no decoded piece"


def test_parse_call_order_both_shapes_and_refusal():
    assert bc.parse_call_order({"id": "e_1", "source_id": "4450488/272.wav"}) == ("4450488", 272)
    assert bc.parse_call_order({"id": "train_4485206_458"}) == ("4485206", 458)
    assert bc.parse_call_order({"id": "train_2020-03-02-Call_11_29"}) == ("2020-03-02-Call_11", 29)
    with pytest.raises(ValueError, match="neither"):
        bc.parse_call_order({"id": "plainid"})


def test_plan_recordings_respects_limits_and_drops_short():
    rows = [{"id": str(i), "duration": 50.0} for i in range(7)]
    plans, dropped = bc.plan_recordings(rows, min_s=60.0, max_s=160.0, gap_s=1.0)
    assert [len(p) for p in plans] == [3, 3] and dropped == 1, (
        "MUST_PASS: 3 x 50 s + 2 gaps fit 160 s"
    )
    plans, dropped = bc.plan_recordings([{"id": "a", "duration": 20.0}], min_s=60.0)
    assert plans == [] and dropped == 1


def test_build_concat_cli(tmp_path):
    clips = []
    for k in range(6):
        path = _write(tmp_path / f"c{k}.flac", _tone(25.0, freq=200 + 10 * k))
        clips.append(
            {
                "id": f"train_callA_{k}",
                "audio": path,
                "answer": f"word{k}",
                "duration": 25.0,
                "text": "Transcribe this audio exactly.",
            }
        )
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text("".join(json.dumps(c) + "\n" for c in reversed(clips)))
    out = tmp_path / "concat.jsonl"
    argv = [
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--audio-dir",
        str(tmp_path / "a"),
        "--min-s",
        "60",
        "--max-s",
        "80",
        "--gap-s",
        "0.5",
    ]
    assert bc.main(argv) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["answer"] for r in rows] == ["word0 word1 word2", "word3 word4 word5"]
    assert rows[0]["clips"] == ["train_callA_0", "train_callA_1", "train_callA_2"]
    assert rows[0]["duration"] == pytest.approx(76.0, abs=0.01)
    bad = tmp_path / "c24.flac"
    _write(bad, _tone(1.0)[:24000], sr=24000)
    manifest.write_text(
        json.dumps({"id": "t_x_0", "audio": str(bad), "answer": "a", "duration": 70.0}) + "\n"
    )
    with pytest.raises(ValueError, match="16 kHz"):
        bc.main(
            ["--manifest", str(manifest), "--out", str(out), "--audio-dir", str(tmp_path / "b")]
        )
