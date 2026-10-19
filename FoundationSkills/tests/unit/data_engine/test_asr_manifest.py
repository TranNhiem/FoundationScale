"""A4.3: target_format "asr" - FS speech manifests validated by FS, audio sample rates MEASURED (16 kHz kept)."""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import wave
from pathlib import Path

import foundationskills.skills.data_engine.ops  # noqa: F401 - register real ops
from foundationskills.core.artifacts import read_artifact
from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills.data_engine.skill import DataEngineSkill

HAS_SOUNDFILE = importlib.util.find_spec("soundfile") is not None


def _wav(path: Path, rate: int, seconds: float = 1.0) -> str:
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i / 20))) for i in range(frames)))
    return str(path)


def _manifest(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _run(tmp_path: Path, manifest: str):
    request = {"target_format": "asr", "sources": [{"uri": manifest, "kind": "speech_manifest"}],
               "out_dir": str(tmp_path / "out")}
    return DataEngineSkill().execute(request, SkillContext(workdir=tmp_path))


def _readiness(result) -> dict:
    return next(read_artifact(r.path).payload for r in result.artifacts if r.type == "readiness_report")


def test_16khz_rows_are_kept_and_others_dropped_as_measured(tmp_path) -> None:
    good = _wav(tmp_path / "a.wav", 16000)
    off = _wav(tmp_path / "b.wav", 22050)
    manifest = _manifest(tmp_path, [
        {"id": "a", "audio": good, "answer": "hello there", "duration": 1.0},
        {"id": "b", "audio": off, "answer": "general kenobi", "duration": 1.0},
    ])
    result = _run(tmp_path, manifest)
    if not HAS_SOUNDFILE:  # the op cannot measure here: a named refusal, never an assumed rate
        assert result.status is Status.REFUSED and "DE-IN-008" in [f.rule_id for f in result.findings]
        return
    assert result.status is Status.PASS, [f.message for f in result.findings]
    readiness = _readiness(result)
    checks = {c["rule_id"]: c for c in readiness["checks"]}
    assert checks["DE-RDY-012"]["passed"] is True and "sample_rate_not_22050" in checks["DE-RDY-012"]["detail"]
    assert checks["DE-RDY-013"]["passed"] is True
    dataset = next(read_artifact(r.path).payload for r in result.artifacts if r.type == "dataset")
    assert dataset["num_records"] == 1 and dataset["format"] == "asr"


def test_a_duplicate_id_makes_the_fs_manifest_report_not_ok(tmp_path) -> None:
    audio = _wav(tmp_path / "a.wav", 16000)
    other = _wav(tmp_path / "c.wav", 16000)
    manifest = _manifest(tmp_path, [
        {"id": "a", "audio": audio, "answer": "one", "duration": 1.0},
        {"id": "a", "audio": other, "answer": "two", "duration": 1.0},
    ])
    result = _run(tmp_path, manifest)
    if not HAS_SOUNDFILE:
        assert result.status is Status.REFUSED
        return
    checks = {c["rule_id"]: c for c in _readiness(result)["checks"]}
    assert checks["DE-RDY-013"]["passed"] is False and "duplicate ids 1" in checks["DE-RDY-013"]["detail"]
    assert result.status is Status.RED


def test_asr_without_a_speech_manifest_source_is_refused(tmp_path) -> None:
    (tmp_path / "rows.jsonl").write_text('{"id": "a", "text": "x"}\n')
    result = DataEngineSkill().execute({"target_format": "asr", "sources": [{"uri": str(tmp_path / "rows.jsonl"),
                                                                           "kind": "jsonl"}]},
                                       SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and "DE-IN-009" in [f.rule_id for f in result.findings]
