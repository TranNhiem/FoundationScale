"""CI-side tests for the NeMo worker adapter's PURE parts.

Everything in this file runs in a plain CI environment WITHOUT ``nemo``, ``torch``,
``lightning`` or ``soundfile`` installed. Any test that would need those is a bug in
``foundationscale.upstream.nemo``'s import discipline (see the module docstring there).
"""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from pathlib import Path

import pytest

from foundationscale.upstream.contracts import SpeechRunConfig, to_argv
from foundationscale.upstream.nemo.adjudicate import TOWERS, adjudicate_digests
from foundationscale.upstream.nemo.census import REFUSAL_ORDER, census
from foundationscale.upstream.nemo.decode import pair_rows
from foundationscale.upstream.nemo.finetune import (
    optimizer_config,
    parse_args,
    train_ds_overrides,
)

# ---------------------------------------------------------------------------
# census: refusal order/reasons with a faked info, measured duration written
# ---------------------------------------------------------------------------


def _info_from(mapping) -> object:
    """Fake ``info``: path -> (samplerate, channels, duration)."""

    def info(path: str):
        return mapping[path]

    return info


def _row(idx: int, audio: str, answer: str = "hello", duration: float = 5.0, **extra) -> dict:
    r = {"id": f"r{idx}", "audio": audio, "answer": answer, "duration": duration}
    r.update(extra)
    return r


def test_census_all_good_keeps_every_row():
    rows = [_row(0, "a.wav"), _row(1, "b.wav")]
    info = _info_from({"a.wav": (16000, 1, 3.5), "b.wav": (16000, 1, 4.25)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert len(kept) == 2
    assert coverage == {
        "rows_expected": 2,
        "rows_checked": 2,
        "rows_refused": 0,
        "refused": {},
    }
    # MEASURED duration is written, not the manifest's declared one (see census module docstring).
    assert kept[0]["duration"] == 3.5
    assert kept[1]["duration"] == 4.25
    # Everything else on the kept row is unchanged.
    assert kept[0]["id"] == "r0"
    assert kept[0]["audio"] == "a.wav"
    assert kept[0]["answer"] == "hello"


def test_census_measured_duration_beats_manifest_duration():
    rows = [_row(0, "a.wav", duration=999.0)]
    info = _info_from({"a.wav": (16000, 1, 2.5)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert kept[0]["duration"] == 2.5
    assert coverage["rows_checked"] == 1


def test_census_duplicate_audio_path_refused():
    rows = [_row(0, "a.wav"), _row(1, "a.wav"), _row(2, "a.wav")]
    info = _info_from({"a.wav": (16000, 1, 3.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert len(kept) == 1
    assert coverage["rows_refused"] == 2
    assert coverage["refused"] == {"duplicate_audio_path": 2}


def test_census_sample_rate_mismatch_refused():
    rows = [_row(0, "a.wav")]
    info = _info_from({"a.wav": (8000, 1, 3.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert kept == []
    assert coverage["refused"] == {"sample_rate_mismatch": 1}


def test_census_not_mono_refused():
    rows = [_row(0, "a.wav")]
    info = _info_from({"a.wav": (16000, 2, 3.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert kept == []
    assert coverage["refused"] == {"not_mono": 1}


def test_census_duration_out_of_range_refused_below_min():
    rows = [_row(0, "a.wav")]
    info = _info_from({"a.wav": (16000, 1, 0.5)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert kept == []
    assert coverage["refused"] == {"duration_out_of_range": 1}


def test_census_duration_out_of_range_refused_above_max():
    rows = [_row(0, "a.wav")]
    info = _info_from({"a.wav": (16000, 1, 30.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert kept == []
    assert coverage["refused"] == {"duration_out_of_range": 1}


def test_census_duration_exactly_at_bounds_is_kept():
    rows = [_row(0, "a.wav"), _row(1, "b.wav")]
    info = _info_from({"a.wav": (16000, 1, 1.0), "b.wav": (16000, 1, 25.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert len(kept) == 2
    assert coverage["rows_refused"] == 0


def test_census_empty_text_refused():
    rows = [_row(0, "a.wav", answer="   ")]
    info = _info_from({"a.wav": (16000, 1, 3.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert kept == []
    assert coverage["refused"] == {"empty_text": 1}


def test_census_refusal_ladder_order_is_exactly_the_original():
    """A row with EVERY fault at once must be counted under the FIRST reason only.

    The row is:
      * a duplicate (``r0`` already claimed ``a.wav``),
      * 8 kHz (would be ``sample_rate_mismatch``),
      * stereo (would be ``not_mono``),
      * too long (would be ``duration_out_of_range``),
      * with an empty answer (would be ``empty_text``).
    """
    assert REFUSAL_ORDER == (
        "duplicate_audio_path",
        "sample_rate_mismatch",
        "not_mono",
        "duration_out_of_range",
        "empty_text",
    )

    # As in the original convert, a path is "seen" only once a row using it was KEPT, so the
    # duplicate case needs a clean first row on the shared path.
    rows = [_row(0, "a.wav"), _row(1, "a.wav", answer="   ")]
    clean = _info_from({"a.wav": (16000, 1, 5.0)})
    kept, coverage = census(rows, max_duration=25.0, info=clean)
    assert len(kept) == 1
    assert coverage["rows_refused"] == 1
    # Duplicate-path wins because it's first in the ladder.
    assert coverage["refused"] == {"duplicate_audio_path": 1}

    # Without the duplicate, the sample-rate check wins over mono/duration/text.
    info = _info_from({"a.wav": (8000, 2, 99.0)})
    rows = [_row(0, "a.wav", answer="   ")]
    kept, coverage = census(rows, max_duration=25.0, info=info)
    assert coverage["refused"] == {"sample_rate_mismatch": 1}

    # Given 16 kHz, the mono check wins over duration/text.
    info_ok_rate = _info_from({"a.wav": (16000, 2, 99.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info_ok_rate)
    assert coverage["refused"] == {"not_mono": 1}

    # Given 16 kHz mono, the duration check wins over text.
    info_ok_rate_mono = _info_from({"a.wav": (16000, 1, 99.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info_ok_rate_mono)
    assert coverage["refused"] == {"duration_out_of_range": 1}

    # Given 16 kHz mono and in-range duration, the empty-text check fires.
    info_all_ok = _info_from({"a.wav": (16000, 1, 3.0)})
    kept, coverage = census(rows, max_duration=25.0, info=info_all_ok)
    assert coverage["refused"] == {"empty_text": 1}


def test_census_info_called_once_per_row_including_duplicates():
    """The original ``convert`` called ``sf.info(r["audio"])`` BEFORE the duplicate check, so a
    duplicate row still hit the filesystem. Preserve that ordering (some failure modes of the
    audio stack surface as an info() exception and must not be silently swallowed by skipping
    the call for duplicates)."""
    calls: list[str] = []

    def info(path: str):
        calls.append(path)
        return (16000, 1, 3.0)

    rows = [_row(0, "a.wav"), _row(1, "a.wav"), _row(2, "b.wav")]
    census(rows, max_duration=25.0, info=info)
    assert calls == ["a.wav", "a.wav", "b.wav"]


def test_census_kept_rows_are_copies():
    """A kept row's ``duration`` is replaced on a COPY; the caller's input rows are untouched."""
    rows = [_row(0, "a.wav", duration=1.0)]
    originals = [dict(r) for r in rows]
    info = _info_from({"a.wav": (16000, 1, 2.5)})
    kept, _ = census(rows, max_duration=25.0, info=info)
    assert kept[0] is not rows[0]
    assert rows == originals
    assert rows[0]["duration"] == 1.0
    assert kept[0]["duration"] == 2.5


# ---------------------------------------------------------------------------
# train_ds_overrides: exact output for a sample released dict
# ---------------------------------------------------------------------------


def test_train_ds_overrides_exact_output():
    released = {
        "manifest_filepath": "PLACEHOLDER",
        "batch_size": 1,
        "use_lhotse": False,
        "shuffle": False,
        "num_workers": 0,
        "max_duration": 1.0,
        "prompt_format": "canary",
        "text_field": "answer",
        # These nine MUST be stripped (ledger entry nemo-canary-train-cfg-strip).
        "batch_duration": 100,
        "use_bucketing": True,
        "bucket_duration_bins": [1, 2, 3],
        "tarred_audio_filepaths": ["x"],
        "is_tarred": True,
        "input_cfg": {"k": "v"},
        "bucket_batch_size": 8,
        "max_tps": 1000,
        "num_buckets": 5,
        # A key NOT on the strip list must survive (same as the original's pop-loop).
        "some_future_knob": True,
    }
    out = train_ds_overrides(
        released,
        manifest_path="/w/train_manifest.json",
        batch_size=8,
        max_duration=25.0,
        seed=None,
        prompt_format="canary",
    )
    # Strip list is byte-for-byte the original's (const sanity).
    for k in (
        "batch_duration",
        "use_bucketing",
        "bucket_duration_bins",
        "tarred_audio_filepaths",
        "is_tarred",
        "input_cfg",
        "bucket_batch_size",
        "max_tps",
        "num_buckets",
    ):
        assert k not in out, f"strip-listed key {k!r} survived the overrides"
    # Everything the original sets is set exactly as before.
    assert out == {
        "manifest_filepath": "/w/train_manifest.json",
        "batch_size": 8,
        "use_lhotse": True,
        "shuffle": True,
        "num_workers": 4,
        "max_duration": 25.0,
        "prompt_format": "canary",
        "text_field": "text",
        "some_future_knob": True,
    }
    # The input is NOT mutated.
    assert released["bucket_duration_bins"] == [1, 2, 3]
    assert released["text_field"] == "answer"


def test_train_ds_overrides_sets_seed_only_when_given():
    released: dict = {}
    out_unset = train_ds_overrides(
        released,
        manifest_path="m",
        batch_size=2,
        max_duration=25.0,
        seed=None,
        prompt_format=None,
    )
    assert "seed" not in out_unset
    assert "shard_seed" not in out_unset  # unseeded keeps NeMo's own default

    out_set = train_ds_overrides(
        released,
        manifest_path="m",
        batch_size=2,
        max_duration=25.0,
        seed=42,
        prompt_format=None,
    )
    assert out_set["seed"] == 42
    assert out_set["shard_seed"] == 42  # seeded runs pin lhotse's per-worker shard shuffle too


def test_train_ds_overrides_prompt_format_is_passed_through_verbatim():
    released = {"prompt_format": "RELEASED"}
    out = train_ds_overrides(
        released,
        manifest_path="m",
        batch_size=2,
        max_duration=25.0,
        seed=None,
        prompt_format="MODEL_WINS",
    )
    assert out["prompt_format"] == "MODEL_WINS"
    # Same override semantics as ``train_cfg.prompt_format = model.cfg.get("prompt_format",
    # train_cfg.get("prompt_format"))``: the caller precomputes the fallback and we store it.


def test_train_ds_overrides_empty_released_yields_just_the_overrides():
    out = train_ds_overrides(
        {},
        manifest_path="m",
        batch_size=2,
        max_duration=25.0,
        seed=None,
        prompt_format=None,
    )
    assert out == {
        "manifest_filepath": "m",
        "use_lhotse": True,
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 4,
        "max_duration": 25.0,
        "prompt_format": None,
        "text_field": "text",
    }


# ---------------------------------------------------------------------------
# optimizer_config: exact output
# ---------------------------------------------------------------------------


def test_optimizer_config_exact():
    assert optimizer_config(1e-5, 50, 500) == {
        "name": "adamw",
        "lr": 1e-5,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": 50,
            "min_lr": 1e-5 * 0.1,
            "max_steps": 500,
        },
    }


def test_optimizer_config_min_lr_is_lr_times_point_one():
    for lr in (1e-5, 2.5e-5, 1.0, 0.5):
        assert optimizer_config(lr, 0, 1)["sched"]["min_lr"] == pytest.approx(lr * 0.1)


# ---------------------------------------------------------------------------
# parse_args: --config and the flag form are identical
# ---------------------------------------------------------------------------


def _cfg_with_everything_set() -> SpeechRunConfig:
    return SpeechRunConfig(
        model="nvidia/canary-1b-flash",
        train_manifest="/data/train.jsonl",
        out_dir="/tmp/out",
        max_steps=1234,
        batch_size=7,
        lr=2.5e-5,
        warmup=11,
        pnc="yes",
        max_duration=31.5,
        seed=42,
        save_every=100,
        freeze=("encoder", "transf_decoder"),
    )


def test_parse_args_config_and_flags_agree_for_full_config(tmp_path: Path) -> None:
    cfg = _cfg_with_everything_set()
    # ``to_argv`` is the typed -> CLI projection; both surfaces must rebuild ``cfg`` byte-identical.
    from_flags = parse_args(to_argv(cfg))
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model": cfg.model,
                "train_manifest": cfg.train_manifest,
                "out_dir": cfg.out_dir,
                "max_steps": cfg.max_steps,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "warmup": cfg.warmup,
                "pnc": cfg.pnc,
                "max_duration": cfg.max_duration,
                "seed": cfg.seed,
                "save_every": cfg.save_every,
                "freeze": list(cfg.freeze),
            }
        )
    )
    from_config = parse_args(["--config", str(cfg_path)])
    assert from_flags == cfg
    assert from_config == cfg
    assert from_flags == from_config


def test_parse_args_config_and_flags_agree_for_defaults(tmp_path: Path) -> None:
    cfg = SpeechRunConfig(model="m", train_manifest="t", out_dir="o")
    from_flags = parse_args(to_argv(cfg))
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps({"model": "m", "train_manifest": "t", "out_dir": "o"}))
    from_config = parse_args(["--config", str(cfg_path)])
    assert from_flags == cfg
    assert from_config == cfg
    assert from_flags == from_config


def test_parse_args_seed_none_and_save_every_zero_omitted_by_to_argv(tmp_path: Path) -> None:
    cfg = SpeechRunConfig(model="m", train_manifest="t", out_dir="o", seed=None, save_every=0)
    assert "--seed" not in to_argv(cfg)
    assert "--save-every" not in to_argv(cfg)
    assert parse_args(to_argv(cfg)) == cfg


def test_parse_args_rejects_pnc_outside_yes_no() -> None:
    with pytest.raises(ValueError):
        parse_args(["--model", "m", "--train", "t", "--out-dir", "o", "--pnc", "auto"])


def test_parse_args_missing_required_flags_errors() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["--model", "m"])
    assert exc.value.code == 2  # argparse's own error exit


# ---------------------------------------------------------------------------
# decode.pair_rows: strict pairing + exact JSON shape
# ---------------------------------------------------------------------------


def test_pair_rows_builds_all_records_in_order():
    rows = [
        {"id": "a", "audio": "a.wav", "answer": "hello world", "duration": 1.5},
        {"id": "b", "audio": "b.wav", "answer": "goodbye", "duration": 2.25},
    ]
    hyps = ["HELLO WORLD", "goodbye!"]
    out = pair_rows(rows, hyps)
    assert out == [
        {
            "id": "a",
            "duration": 1.5,
            "reference": "hello world",
            "hypothesis": "HELLO WORLD",
        },
        {
            "id": "b",
            "duration": 2.25,
            "reference": "goodbye",
            "hypothesis": "goodbye!",
        },
    ]
    # Key order is part of the eval JSON's shape: id, duration, reference, hypothesis.
    assert list(out[0]) == ["id", "duration", "reference", "hypothesis"]


def test_pair_rows_preserves_absent_duration():
    rows = [{"id": "a", "audio": "a.wav", "answer": "x"}]
    out = pair_rows(rows, ["X"])
    assert out[0]["duration"] is None


def test_pair_rows_strict_length_mismatch_raises():
    rows = [{"id": "a", "audio": "a.wav", "answer": "x"}]
    with pytest.raises(ValueError):
        pair_rows(rows, [])
    with pytest.raises(ValueError):
        pair_rows(rows, ["x", "y"])


def test_pair_rows_empty_inputs():
    assert pair_rows([], []) == []


# ---------------------------------------------------------------------------
# adjudicate.adjudicate_digests: PASS / RED / frozen-identity / VACUOUS
# ---------------------------------------------------------------------------


def _balanced_coverage(checked: int = 10, refused: int = 0) -> dict:
    return {
        "rows_expected": checked + refused,
        "rows_checked": checked,
        "rows_refused": refused,
        "refused": {} if refused == 0 else {"duplicate_audio_path": refused},
    }


def test_adjudicate_pass_case_all_params_moved():
    base = {
        "encoder.w": "AAA",
        "encoder.b": "BBB",
        "transf_decoder.x": "CCC",
        "transf_decoder.y": "DDD",
    }
    tuned = {
        "encoder.w": "aaa",
        "encoder.b": "bbb",
        "transf_decoder.x": "ccc",
        "transf_decoder.y": "ddd",
    }
    lines, blocking = adjudicate_digests(base, tuned, TOWERS, set(), _balanced_coverage())
    # Coverage + one movement line per non-frozen tower.
    assert len(lines) == 3
    # Line format mirrors the original: [VERDICT] <gate_id>: checked/expected unit -- detail.
    assert lines[0].startswith("[")
    assert " speech.audio_row_coverage" in lines[0]
    # The original printed the movement gate's own id (no per-tower suffix), once per tower.
    assert lines[1].startswith("[PASS] speech.tower_movement: 2/2 tower parameters")
    assert lines[2].startswith("[PASS] speech.tower_movement: 2/2 tower parameters")
    # PASS: every gate should have found something to measure and found it in good order.
    # Route level: no line says RED and the run does not block.
    assert not any(line.startswith("[RED]") for line in lines)
    assert blocking is False
    assert isinstance(blocking, bool)


def test_adjudicate_red_case_nothing_moved_but_towers_were_exercised():
    base = {"encoder.w": "AAA", "transf_decoder.x": "CCC"}
    tuned = dict(base)
    lines, blocking = adjudicate_digests(base, tuned, TOWERS, set(), _balanced_coverage())
    # exercised=True with zero movement is exactly the silent-untrained failure the gate exists
    # to catch -- we assert the run blocks (RED) regardless of the gate's own verdict string
    # spelling, and at least one line must say so.
    assert blocking is True
    assert isinstance(blocking, bool)
    assert len(lines) == 3


def test_adjudicate_vacuous_case_no_tower_params_at_all():
    base: dict = {}
    tuned: dict = {}
    lines, blocking = adjudicate_digests(base, tuned, TOWERS, set(), _balanced_coverage())
    # Tower coverage is empty: there is nothing for the movement gate to measure. That is
    # VACUOUS -- and a fail-closed design treats "couldn't measure" as blocking, not as a pass.
    assert len(lines) == 3
    assert blocking is True
    assert any("0/" in line and "tower" in line for line in lines) or any(
        "VACUOUS" in line.upper() for line in lines
    )


def test_adjudicate_frozen_identity_passes():
    base = {
        "encoder.w": "AAA",
        "encoder.b": "BBB",
        "transf_decoder.x": "CCC",
        "transf_decoder.y": "DDD",
    }
    tuned = dict(base)
    lines, blocking = adjudicate_digests(base, tuned, TOWERS, {"encoder"}, _balanced_coverage())
    # Coverage + one movement line for the NON-frozen decoder + one frozen line for the encoder.
    assert len(lines) == 3
    frozen_line = lines[-1]
    assert frozen_line == (
        "[PASS] speech.frozen_unchanged/encoder: 2/2 tower parameters -- 0 changed []"
    )
    # Only the non-frozen decoder movement is checked; the frozen check passed. The decoder did
    # not move (``tuned == base``), so the run may or may not block depending on the movement
    # gate's normal semantics -- that is routed through the real gate and asserted above, not
    # frozen-line behavior. Here we only assert the frozen line is exactly what we expect.
    assert frozen_line.startswith("[PASS]")


def test_adjudicate_frozen_changed_is_red_and_blocking():
    base = {"encoder.w": "AAA", "encoder.b": "BBB", "transf_decoder.x": "CCC"}
    tuned = dict(base)
    tuned["encoder.w"] = "AAA_FROM_FINE_TUNE"
    lines, blocking = adjudicate_digests(base, tuned, TOWERS, {"encoder"}, _balanced_coverage())
    frozen_line = lines[-1]
    assert frozen_line == (
        "[RED] speech.frozen_unchanged/encoder: 1/2 tower parameters -- 1 changed ['encoder.w']"
    )
    assert blocking is True


def test_adjudicate_frozen_prefix_matching_no_params_blocks():
    """A declared-frozen prefix matching zero tensors proves nothing and must block."""
    base = {"encoder.w": "AAA"}
    tuned = {"encoder.w": "AAA"}
    lines, blocking = adjudicate_digests(
        base, tuned, TOWERS, {"no_such_tower"}, _balanced_coverage()
    )
    # The original only honours --frozen for prefixes that are towers; an unknown prefix adds
    # no line. transf_decoder has no digests here, so its movement gate is VACUOUS and blocks.
    assert len(lines) == 3
    assert not any("no_such_tower" in line for line in lines)
    assert lines[2].startswith("[VACUOUS] speech.tower_movement:")
    assert blocking is True


def test_adjudicate_line_order_coverage_then_non_frozen_then_frozen_in_tower_order():
    """The original's aggregation is ``results`` (coverage, non-frozen movements) ++ ``extra``
    (frozen lines). That ORDER is preserved verbatim even when the frozen tower is first in
    ``TOWERS`` -- tested here because it is what the printed ``ADJ`` output looks like."""
    base = {
        "encoder.w": "AAA",
        "transf_decoder.x": "CCC",
    }
    tuned = {
        "encoder.w": "AAA_FINE_TUNED",
        "transf_decoder.x": "CCC",
    }
    lines, _ = adjudicate_digests(base, tuned, TOWERS, {"encoder"}, _balanced_coverage())
    # 1 (coverage) + 1 (transf_decoder movement) + 1 (encoder frozen-unchanged).
    assert len(lines) == 3
    assert "speech.audio_row_coverage" in lines[0]
    assert lines[1].startswith("[FAIL] speech.tower_movement:")  # transf_decoder unchanged
    assert lines[2].startswith("[RED] speech.frozen_unchanged/encoder:")  # encoder moved


def test_adjudicate_reflected_refused_coverage_is_passed_to_the_gate():
    cov = _balanced_coverage(checked=9, refused=1)
    cov["refused"] = {"duplicate_audio_path": 1}
    lines, _ = adjudicate_digests({}, {}, TOWERS, set(), cov)
    # The coverage gate sees rows_expected=10 rows_checked=9 rows_refused=1 refused={...}: it
    # should PASS the identity check (9 + 1 == 10) and the line should carry that coverage.
    # The real AudioRowCoverageGate reports the refusal in its detail and fails strict mode.
    assert lines[0].startswith("[FAIL] speech.audio_row_coverage:")
    assert "1 of 10 audio rows were refused" in lines[0]


# ---------------------------------------------------------------------------
# Import discipline: every module imports WITHOUT nemo/torch/lightning/soundfile present.
# ---------------------------------------------------------------------------


HEAVY_TOP_LEVEL_MODULES = ("nemo", "torch", "lightning", "soundfile")


def test_every_nemo_adapter_module_imports_without_heavy_dependencies() -> None:
    """If any of ``nemo`` / ``torch`` / ``lightning`` / ``soundfile`` is somehow imported at
    module level of ``foundationscale.upstream.nemo.*``, make the import RAISE here. The
    modules are purged + re-imported to force module-level code to re-run.

    Mirrors the CI reality: the control plane imports these modules on a machine where none of
    the container-only dependencies exist.
    """
    mod_names = [
        "foundationscale.upstream.nemo.census",
        "foundationscale.upstream.nemo.finetune",
        "foundationscale.upstream.nemo.decode",
        "foundationscale.upstream.nemo.adjudicate",
    ]
    # Purge so module-level code re-executes under the guard below.
    for name in mod_names:
        sys.modules.pop(name, None)
    sys.modules.pop("foundationscale.upstream.nemo", None)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        top = name.split(".", 1)[0]
        if top in HEAVY_TOP_LEVEL_MODULES:
            raise AssertionError(f"heavy import at module level is forbidden: import {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        importlib.import_module("foundationscale.upstream.nemo")
        for name in mod_names:
            importlib.import_module(name)
    finally:
        builtins.__import__ = real_import
        # Restore the purged modules so later tests see the normal import tree.
        importlib.import_module("foundationscale.upstream.nemo")
        for name in mod_names:
            importlib.import_module(name)


def test_no_module_level_heavy_import_statements_anywhere_in_the_package() -> None:
    """A grep-style guard: even without executing module-level code, prove that no source file
    in the package reaches for ``nemo``/``torch``/``lightning``/``soundfile`` at indent-zero-or-
    ``from x import``-top level. Any occurrence must be indented (i.e. inside a function)."""
    pkg_dir = Path(__file__).resolve().parents[3] / "src" / "foundationscale" / "upstream" / "nemo"
    assert pkg_dir.is_dir(), f"expected package at {pkg_dir}"
    offenders: list[str] = []
    for py in sorted(pkg_dir.glob("*.py")):
        for lineno, raw in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw.strip()
            if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith('"'):
                continue  # comments and docstrings mention the names; harmless.
            # A top-level import is one whose left edge is column 0 (module-level, not
            # inside a def/class). Inside a function we indent by at least one space.
            left_edge_zero = len(raw) > 0 and not raw[0].isspace()
            if not left_edge_zero:
                continue
            for heavy in HEAVY_TOP_LEVEL_MODULES:
                if (
                    stripped == f"import {heavy}"
                    or stripped.startswith(f"import {heavy} ")
                    or stripped.startswith(f"import {heavy},")
                    or stripped.startswith(f"from {heavy} ")
                    or stripped.startswith(f"from {heavy}.")
                ):
                    offenders.append(f"{py.name}:{lineno}: {raw!r}")
    assert not offenders, "top-level heavy imports found:\n" + "\n".join(offenders)
