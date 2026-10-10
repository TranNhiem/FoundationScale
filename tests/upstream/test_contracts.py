"""Tests for the owned speech contracts.

Run under ``FS_FORBID_SKIPS=1``: no skip, no importorskip, no xfail. Nothing
here imports torch, NeMo or transformers and nothing opens an audio file -- the
contracts are data, and data is testable as data. The JSONL files are written
into ``tmp_path``, so no fixture corpus is required and every problem code in
``manifest_row_problems``'s vocabulary is exercised by name.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from foundationscale.upstream.contracts import (
    SPEECH_MANIFEST_SCHEMA_VERSION,
    ManifestReport,
    SpeechRunConfig,
    load_run_config,
    manifest_row_problems,
    read_speech_manifest,
    to_argv,
    to_nemo_aed_row,
)


def _row(**overrides: object) -> dict[str, object]:
    """A valid row, with overrides applied on top. The base row is the shape the
    campaigns actually write: id, audio, answer, duration plus two optionals."""
    row: dict[str, object] = {
        "id": "utt-1",
        "audio": "/data/utt-1.wav",
        "answer": "HELLO WORLD",
        "duration": 3.5,
    }
    row.update(overrides)
    return row


def _write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- row problems


def test_valid_row_with_optionals_has_no_problems() -> None:
    row = _row(text="transcribe", source_id="src-9", conversion="canary-probe", segments=3)
    assert manifest_row_problems(row) == []


def test_valid_row_without_optionals_has_no_problems() -> None:
    assert manifest_row_problems(_row()) == []


def test_missing_every_required_key_is_reported_separately() -> None:
    assert manifest_row_problems({}) == [
        "missing:id",
        "missing:audio",
        "missing:answer",
        "missing:duration",
    ]


def test_missing_one_required_key_is_reported() -> None:
    row = _row()
    del row["answer"]
    assert manifest_row_problems(row) == ["missing:answer"]


def test_bad_type_id_and_audio_refused() -> None:
    assert manifest_row_problems(_row(id=7)) == ["bad_type:id"]
    assert manifest_row_problems(_row(id="")) == ["bad_type:id"]
    assert manifest_row_problems(_row(id="   ")) == ["bad_type:id"]
    assert manifest_row_problems(_row(audio=["/data/a.wav"])) == ["bad_type:audio"]
    assert manifest_row_problems(_row(audio="")) == ["bad_type:audio"]


def test_bad_type_answer_refused() -> None:
    assert manifest_row_problems(_row(answer=12)) == ["bad_type:answer"]
    assert manifest_row_problems(_row(answer=None)) == ["bad_type:answer"]


def test_empty_answer_is_its_own_problem() -> None:
    assert manifest_row_problems(_row(answer="")) == ["empty_answer"]
    assert manifest_row_problems(_row(answer="   ")) == ["empty_answer"]


def test_bad_type_duration_including_bool() -> None:
    # bool is an int in Python: True must not become a 1-second clip.
    assert manifest_row_problems(_row(duration=True)) == ["bad_type:duration"]
    assert manifest_row_problems(_row(duration="3.5")) == ["bad_type:duration"]
    assert manifest_row_problems(_row(duration=None)) == ["bad_type:duration"]


def test_bad_duration_for_unusable_numbers() -> None:
    assert manifest_row_problems(_row(duration=0)) == ["bad_duration"]
    assert manifest_row_problems(_row(duration=0.0)) == ["bad_duration"]
    assert manifest_row_problems(_row(duration=-1.5)) == ["bad_duration"]
    assert manifest_row_problems(_row(duration=float("nan"))) == ["bad_duration"]
    assert manifest_row_problems(_row(duration=float("inf"))) == ["bad_duration"]


def test_optional_fields_are_type_checked_when_present() -> None:
    assert manifest_row_problems(_row(text=["x"])) == ["bad_type:text"]
    assert manifest_row_problems(_row(source_id=4)) == ["bad_type:source_id"]
    assert manifest_row_problems(_row(conversion=[1])) == ["bad_type:conversion"]
    assert manifest_row_problems(_row(conversion=None)) == []
    assert manifest_row_problems(_row(segments=True)) == ["bad_type:segments"]
    assert manifest_row_problems(_row(segments="2")) == ["bad_type:segments"]
    assert manifest_row_problems(_row(segments=0)) == ["bad_type:segments"]
    assert manifest_row_problems(_row(segments=2)) == []


def test_unknown_key_is_a_problem_and_is_named() -> None:
    assert manifest_row_problems(_row(duraton=1)) == ["unknown_key:duraton"]
    # A misspelled required key must surface as BOTH a miss and a typo, sorted
    # deterministically so two runs' problem lists compare equal.
    assert manifest_row_problems(_row(audio=None, audi="/data/a.wav", answer="")) == [
        "unknown_key:audi",
        "bad_type:audio",
        "empty_answer",
    ]


def test_unknown_keys_are_sorted_and_positive_int_duration_is_valid() -> None:
    assert manifest_row_problems(_row(zz=1, aa=2)) == ["unknown_key:aa", "unknown_key:zz"]
    assert manifest_row_problems(_row(duration=3)) == []


def test_non_mapping_row_reports_bad_type_row() -> None:
    assert manifest_row_problems(["not", "a", "row"]) == ["bad_type:row"]  # type: ignore[arg-type]


# ---------------------------------------------------------------------- reader


def test_reader_reads_only_valid_rows_and_counts_problems(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            # one recording per row: this test is about problem codes, not duplicates
            json.dumps(_row(id="a", audio="/data/a.wav")),
            json.dumps(_row(id="b", audio="/data/b.wav", answer="")),
            "{not json",
            json.dumps(_row(id="c", audio="/data/c.wav", duraton=1.0)),
            "",
            "   ",
            json.dumps(_row(id="d", audio="/data/d.wav", duration=2.0)),
        ],
    )
    rows, report = read_speech_manifest(path)
    assert [r["id"] for r in rows] == ["a", "d"]
    assert report.rows_read == 5  # blank lines are not rows
    assert report.rows_valid == 2
    assert report.problems == {
        "empty_answer": 1,
        "invalid_json": 1,
        "unknown_key:duraton": 1,
    }
    assert report.duplicate_ids == 0
    assert report.duplicate_audio_paths == 0
    assert not report.ok


def test_reader_counts_duplicate_ids_and_returns_all_three(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            json.dumps(_row(id="a")),
            json.dumps(_row(id="a", audio="/data/x.wav")),
            json.dumps(_row(id="b", audio="/data/y.wav")),
        ],
    )
    rows, report = read_speech_manifest(path)
    assert len(rows) == 3
    assert report.duplicate_ids == 1
    assert report.duplicate_audio_paths == 0
    assert not report.ok


def test_reader_counts_duplicate_audio_paths_three_rows_two_files(tmp_path: Path) -> None:
    # The measured shape of the builder bug, at 3 rows: two transcripts share one
    # recording. Row coverage says 3/3; only this counter says otherwise.
    path = _write(
        tmp_path,
        [
            json.dumps(_row(id="a", audio="/data/p1.wav")),
            json.dumps(_row(id="b", audio="/data/p1.wav")),
            json.dumps(_row(id="c", audio="/data/p2.wav")),
        ],
    )
    rows, report = read_speech_manifest(path)
    assert len(rows) == 3
    assert report.rows_valid == 3
    assert report.problems == {}
    assert report.duplicate_ids == 0
    assert report.duplicate_audio_paths == 1
    assert not report.ok  # duplicates fail `ok` even with zero problems


def test_reader_counts_a_collision_even_on_an_invalid_row(tmp_path: Path) -> None:
    # The refused row still shares an audio path with a kept one: the collision
    # must not hide behind the row's other problems.
    path = _write(
        tmp_path,
        [
            json.dumps(_row(id="a", audio="/data/p1.wav")),
            json.dumps(_row(id="b", audio="/data/p1.wav", answer="")),
        ],
    )
    rows, report = read_speech_manifest(path)
    assert [r["id"] for r in rows] == ["a"]
    assert report.duplicate_audio_paths == 1


def test_reader_counts_a_non_object_json_line(tmp_path: Path) -> None:
    path = _write(tmp_path, [json.dumps(_row(id="a")), "[1, 2, 3]"])
    rows, report = read_speech_manifest(path)
    assert len(rows) == 1
    assert report.problems == {"bad_type:row": 1}


def test_reader_on_a_clean_manifest_is_ok(tmp_path: Path) -> None:
    path = _write(tmp_path, [json.dumps(_row(id="a")), json.dumps(_row(id="b", audio="/b.wav"))])
    rows, report = read_speech_manifest(path)
    assert len(rows) == 2
    assert report.rows_read == 2 and report.rows_valid == 2
    assert report.ok


def test_reader_raises_file_not_found_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_speech_manifest(tmp_path / "absent.jsonl")


def test_report_as_manifest_is_sorted_and_carries_the_schema() -> None:
    report = ManifestReport(
        rows_read=3,
        rows_valid=1,
        problems={"unknown_key:z": 1, "empty_answer": 1},
        duplicate_ids=0,
        duplicate_audio_paths=0,
    )
    manifest = report.as_manifest()
    assert manifest["schema_version"] == SPEECH_MANIFEST_SCHEMA_VERSION
    assert manifest["rows_read"] == 3
    assert manifest["rows_valid"] == 1
    assert list(manifest["problems"]) == ["empty_answer", "unknown_key:z"]  # sorted
    assert manifest["ok"] is False


def test_report_refuses_a_count_that_cannot_cover_the_refused_rows() -> None:
    with pytest.raises(ValueError):
        ManifestReport(
            rows_read=3,
            rows_valid=1,
            problems={},
            duplicate_ids=0,
            duplicate_audio_paths=0,
        )


# ------------------------------------------------------------------- nemo rows


def test_to_nemo_aed_row_fields_and_purity() -> None:
    nemo = to_nemo_aed_row(_row(duration=3), pnc="yes", source_lang="en", target_lang="de")
    assert nemo == {
        "audio_filepath": "/data/utt-1.wav",
        "duration": 3,
        "text": "HELLO WORLD",
        "source_lang": "en",
        "target_lang": "de",
        "taskname": "asr",
        "pnc": "yes",
    }
    # Pure: the row's own duration travels through; nothing was opened to
    # re-measure it. Calling twice must agree.
    assert to_nemo_aed_row(_row(duration=3), pnc="yes") == to_nemo_aed_row(
        _row(duration=3), pnc="yes"
    )


def test_to_nemo_aed_row_defaults_to_english() -> None:
    nemo = to_nemo_aed_row(_row(), pnc="no")
    assert nemo["source_lang"] == "en"
    assert nemo["target_lang"] == "en"
    assert nemo["pnc"] == "no"


def test_to_nemo_aed_row_refuses_anything_but_yes_or_no() -> None:
    with pytest.raises(ValueError) as excinfo:
        to_nemo_aed_row(_row(), pnc="maybe")
    assert "pnc" in str(excinfo.value)
    assert "'yes'" in str(excinfo.value) and "'no'" in str(excinfo.value)
    with pytest.raises(ValueError):
        to_nemo_aed_row(_row(), pnc=True)  # type: ignore[arg-type]


def test_to_nemo_aed_row_refuses_an_invalid_row_by_name() -> None:
    with pytest.raises(ValueError) as excinfo:
        to_nemo_aed_row(_row(answer=""), pnc="no")
    assert "empty_answer" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        to_nemo_aed_row(_row(zz=1), pnc="no")
    assert "unknown_key:zz" in str(excinfo.value)


def test_to_nemo_aed_row_refuses_a_bad_language_name() -> None:
    with pytest.raises(ValueError) as excinfo:
        to_nemo_aed_row(_row(), pnc="no", source_lang="")
    assert "source_lang" in str(excinfo.value)


# ------------------------------------------------------------------ run config


def _full_mapping() -> dict[str, object]:
    return {
        "model": "nvidia/canary-1b-flash",
        "train_manifest": "/data/train.jsonl",
        "out_dir": "/out/run-7",
        "max_steps": 1500,
        "batch_size": 4,
        "lr": 2e-5,
        "warmup": 100,
        "pnc": "yes",
        "max_duration": 40.0,
        "seed": 7,
        "save_every": 250,
        "freeze": ["transf_decoder", "encoder.0"],
    }


def test_load_run_config_fills_defaults() -> None:
    cfg = load_run_config({"model": "m", "train_manifest": "t", "out_dir": "o"})
    assert cfg == SpeechRunConfig(model="m", train_manifest="t", out_dir="o")
    assert cfg.max_steps == 500
    assert cfg.batch_size == 8
    assert cfg.lr == 1e-5
    assert cfg.warmup == 50
    assert cfg.pnc == "no"
    assert cfg.max_duration == 25.0
    assert cfg.seed is None
    assert cfg.save_every == 0
    assert cfg.freeze == ()


def test_load_run_config_reads_every_field_and_normalises_freeze() -> None:
    cfg = load_run_config(_full_mapping())
    assert cfg.model == "nvidia/canary-1b-flash"
    assert cfg.train_manifest == "/data/train.jsonl"
    assert cfg.out_dir == "/out/run-7"
    assert cfg.max_steps == 1500
    assert cfg.batch_size == 4
    assert cfg.lr == 2e-5
    assert cfg.warmup == 100
    assert cfg.pnc == "yes"
    assert cfg.max_duration == 40.0
    assert cfg.seed == 7
    assert cfg.save_every == 250
    assert cfg.freeze == ("transf_decoder", "encoder.0")  # list in, tuple stored


def test_load_run_config_refuses_an_unknown_key_by_name() -> None:
    mapping = _full_mapping()
    mapping["max_duration_seconds"] = 30.0
    with pytest.raises(ValueError) as excinfo:
        load_run_config(mapping)
    assert "max_duration_seconds" in str(excinfo.value)


def test_load_run_config_refuses_a_missing_required_key_by_name() -> None:
    with pytest.raises(ValueError) as excinfo:
        load_run_config({"train_manifest": "t", "out_dir": "o"})
    assert "model" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        load_run_config({"model": "m", "train_manifest": "t", "out_dir": "o", "tags": []})
    assert "tags" in str(excinfo.value)


def test_load_run_config_refuses_bool_for_an_int_field() -> None:
    mapping = _full_mapping()
    mapping["max_steps"] = True
    with pytest.raises(ValueError) as excinfo:
        load_run_config(mapping)
    assert "max_steps" in str(excinfo.value)


def test_load_run_config_refuses_non_positive_and_negative_values_by_name() -> None:
    cases = (
        ("max_steps", 0),
        ("batch_size", -1),
        ("lr", 0),
        ("max_duration", -25.0),
        ("lr", float("nan")),
        ("warmup", -1),
        ("save_every", -5),
    )
    for key, value in cases:
        mapping = _full_mapping()
        mapping[key] = value
        with pytest.raises(ValueError) as excinfo:
            load_run_config(mapping)
        assert key in str(excinfo.value), f"{key}={value!r} refusal did not name the key"


def test_load_run_config_refuses_a_non_yes_no_pnc_by_name() -> None:
    mapping = _full_mapping()
    mapping["pnc"] = "maybe"
    with pytest.raises(ValueError) as excinfo:
        load_run_config(mapping)
    assert "pnc" in str(excinfo.value)
    assert "yes" in str(excinfo.value) and "no" in str(excinfo.value)


def test_load_run_config_refuses_a_bare_string_freeze_by_name() -> None:
    mapping = _full_mapping()
    mapping["freeze"] = "transf_decoder"
    with pytest.raises(ValueError) as excinfo:
        load_run_config(mapping)
    assert "freeze" in str(excinfo.value)


def test_load_run_config_refuses_a_non_mapping() -> None:
    with pytest.raises(ValueError):
        load_run_config(["model", "m"])  # type: ignore[arg-type]


def test_run_config_refuses_invalid_values_at_construction() -> None:
    with pytest.raises(ValueError) as excinfo:
        SpeechRunConfig(model="m", train_manifest="t", out_dir="o", pnc="True")
    assert "pnc" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        SpeechRunConfig(model="m", train_manifest="t", out_dir="o", batch_size=False)  # type: ignore[arg-type]
    assert "batch_size" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        SpeechRunConfig(model="m", train_manifest="t", out_dir="o", freeze=("",))  # type: ignore[arg-type]
    assert "freeze" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        SpeechRunConfig(model="m", train_manifest="t", out_dir="o", freeze="abc")  # type: ignore[arg-type]
    assert "freeze" in str(excinfo.value)


def test_run_config_is_frozen_and_hashable() -> None:
    cfg = load_run_config({"model": "m", "train_manifest": "t", "out_dir": "o"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model = "other"  # type: ignore[misc]
    assert cfg in {cfg}


# ---------------------------------------------------------------------- to argv


def test_to_argv_full_config_is_the_exact_flag_list() -> None:
    cfg = load_run_config(_full_mapping())
    assert to_argv(cfg) == [
        "--model",
        "nvidia/canary-1b-flash",
        "--train",
        "/data/train.jsonl",
        "--out-dir",
        "/out/run-7",
        "--max-steps",
        "1500",
        "--batch-size",
        "4",
        "--lr",
        "2e-05",
        "--warmup",
        "100",
        "--pnc",
        "yes",
        "--max-duration",
        "40.0",
        "--seed",
        "7",
        "--save-every",
        "250",
        "--freeze",
        "transf_decoder",
        "--freeze",
        "encoder.0",
    ]


def test_to_argv_omits_seed_save_every_and_freeze_when_unset() -> None:
    cfg = load_run_config({"model": "m", "train_manifest": "t", "out_dir": "o"})
    argv = to_argv(cfg)
    assert argv == [
        "--model",
        "m",
        "--train",
        "t",
        "--out-dir",
        "o",
        "--max-steps",
        "500",
        "--batch-size",
        "8",
        "--lr",
        "1e-05",
        "--warmup",
        "50",
        "--pnc",
        "no",
        "--max-duration",
        "25.0",
    ]
    assert "--seed" not in argv
    assert "--save-every" not in argv
    assert "--freeze" not in argv


def test_to_argv_emits_one_freeze_flag_per_prefix_in_order() -> None:
    cfg = load_run_config(
        {"model": "m", "train_manifest": "t", "out_dir": "o", "freeze": ["a", "b", "c"]}
    )
    argv = to_argv(cfg)
    assert [argv[i + 1] for i, flag in enumerate(argv) if flag == "--freeze"] == ["a", "b", "c"]


def test_to_argv_omits_zero_save_every_but_keeps_a_seed_of_zero() -> None:
    cfg = load_run_config(
        {"model": "m", "train_manifest": "t", "out_dir": "o", "seed": 0, "save_every": 0}
    )
    argv = to_argv(cfg)
    assert argv[-2:] == ["--seed", "0"]
    assert "--save-every" not in argv


def test_to_argv_round_trip_is_deterministic() -> None:
    first = to_argv(load_run_config(_full_mapping()))
    second = to_argv(load_run_config(_full_mapping()))
    assert first == second
    assert first is not second


def test_to_argv_refuses_a_non_config() -> None:
    with pytest.raises(ValueError):
        to_argv({"model": "m"})  # type: ignore[arg-type]
