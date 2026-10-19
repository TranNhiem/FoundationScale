#!/usr/bin/env python3
"""build_cpt_pool.py -- CPT pre-training pool from the Earnings-22 'robust' parquet shards (SP-1).

ROADMAP SP-1 is speech continued pre-training: we want audio for building a speech corpus WHERE
THE LABELS OF THE TUNING SET MUST NOT LEAK IN. So this script decodes EVERY train segment of
earnings22_robust that is not:
  * in manifests/earnings_train6000.jsonl or manifests/earnings_validation1000.jsonl (each id
    counted separately -- those 7000 segments are the tuning and held-out sets, and a CPT run that
    saw their transcripts before SFT would teach the SOP with the exam in the training data), or
  * a row whose source_id names a test call -- the set of the FIRST path components of the
    source_id values in manifests/ood_earnings223000_conv.jsonl ('4432298/0.wav' -> '4432298') is
    exactly the family of the out-of-distribution test calls, and every segment of one of those
    calls is Test rather than Train as a fact about the corpus.

Rows are EXCLUDED BY REASON and every exclusion is COUNTED: rows_seen = excluded + written
(+ a counted audio failure), always, and the record refuses to hide a row that left the run
without being counted.

Audio is written as 16 kHz mono FLAC under <out-dir>/flac/<id>.flac. The 'robust' shards hold the
SAME audio as the manifests (the manifests were decanted from these shards at 16 kHz), so 16 000 Hz
is the corpus rate here and a shard at any other rate is RESAMPLED (soundfile + scipy
resample_poly) and counted -- resampling at corpus build is about writing one consistent file
format, not about changing what an ASR model hears at eval time (eval still refuses to resample).

The REAL transcript ('sentence') IS carried in the pool row's "answer" field so decode.sh/eval on
the pool (a teacher that is EVALUATED here) sees the manifest schema it expects and its eval.json
writes the record of true sentence vs hypothesis that make_cpt_manifest consumes. The CPT TRAINING
manifest is built by make_cpt_manifest and takes the TEACHER HYPOTHESIS as the label, never
"sentence": sentence is written ONLY to be read by eval (diagnostics) and is replaced -- so it is
kept only as the word of the person who will run decode.sh. The "pool": true flag says the row is
a pool row and not an SFT training row.

The id is "train_<source_id>_<segment_id>", matching the SFT manifests' id convention so that two
manifests can be joined on id and the exclusion lists can be applied to the pool by id
consistently.

Exit codes: 0 ok / 95 coverage short (rows left the run with neither a file nor a counted reason)
/ 96 refusal (nothing written if directory/manifest/columns check fails before we begin).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy
import scipy.signal
import soundfile

EXIT_OK = 0
EXIT_COVERAGE_SHORT = 95
EXIT_REFUSAL = 96

TARGET_RATE = 16000
"""Corpus rate: every SFT manifest row is 16 kHz FLAC, so pool FLAC matches at 16 kHz mono."""

REQUIRED_PARQUET_COLUMNS: tuple[str, ...] = (
    "source_id",
    "audio",
    "segment_id",
    "sentence",
    "start_ts",
    "end_ts",
)

EXCLUDE_MANIFESTS: tuple[tuple[str, str], ...] = (
    ("earnings_train6000", "earnings_train6000.jsonl"),
    ("earnings_validation1000", "earnings_validation1000.jsonl"),
)

TEST_CALL_MANIFEST = "ood_earnings223000_conv.jsonl"

SAMPLE_MANIFEST_ROWS = (0, 1, 2, 3, 4)


class Refusal(Exception):
    """A precondition this script refuses before anything is written: exit 96."""


def _env_record() -> dict[str, Any]:
    """The versions a second run must match to reproduce these files (step_env, inlined)."""
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
    }
    for name in ("soundfile", "scipy", "pyarrow", "pandas"):
        try:
            import importlib.metadata as metadata

            record[name] = metadata.version(name)
        except Exception:
            record[name] = None
    return record


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Write the SP-1 CPT pool (16 kHz mono FLAC + pool.jsonl) from Earnings-22."
    )
    ap.add_argument(
        "--data-dir",
        required=True,
        help="dir of the 'robust' parquet shards (train-*.parquet)",
    )
    ap.add_argument(
        "--manifests-dir",
        required=True,
        help="dir with the train6000 / validation1000 / ood test manifests",
    )
    ap.add_argument("--out-dir", required=True, help="output dir: flac/ + pool.jsonl + record")
    ap.add_argument(
        "--limit", type=int, default=None, help="write at most --limit rows (dry runs / smoke)"
    )
    return ap.parse_args()


def read_manifest_ids(path: Path, keys: tuple[str, ...] = ("id",)) -> set[str]:
    """Set of 'id' values in a jsonl manifest (Refusal if the manifest has no id-like key)."""
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Refusal(f"{path.name} line {lineno} is not valid JSON: {exc}") from exc
            for key in keys:
                if key in row:
                    ids.add(str(row[key]))
                    break
            else:
                raise Refusal(
                    f"{path.name} line {lineno} has no id key ({', '.join(keys)}): "
                    "the exclusion set says which rows are held out and it refuses to guess"
                )
    return ids


def read_source_ids(path: Path) -> set[str]:
    """First path component of each row's source_id, e.g. '4432298/0.wav' -> '4432298'.

    Every segment of one of these calls is a TEST call (Test, not Train): the component names the
    call in every segment id of that call's shards.
    """
    source_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Refusal(f"{path.name} line {lineno} is not valid JSON: {exc}") from exc
            value = row.get("source_id")
            if not isinstance(value, str) or not value.strip():
                raise Refusal(
                    f"{path.name} line {lineno} has no source_id: cannot determine which call "
                    "rows are test calls"
                )
            component = value.replace("\\", "/").split("/")[0]
            if component:
                source_ids.add(component)
    return source_ids


def read_shards(data_dir: Path) -> list[Path]:
    """Sorted 'train-*.parquet' shards in data_dir (a directory with none is a refusal)."""
    import pyarrow.parquet as pq

    shards = sorted(glob_paths(data_dir / "train-*.parquet"))
    if not shards:
        raise Refusal(f"no train-*.parquet shards in {data_dir}")
    for shard in shards:
        schema = pq.ParquetFile(shard).schema_arrow
        missing_names = [name for name in REQUIRED_PARQUET_COLUMNS if name not in schema.names]
        if missing_names:
            raise Refusal(
                f"{shard.name} misses columns {missing_names} (found {schema.names}): "
                "the pool is built from source_id/segment_id/sentence and refuses a guessed column"
            )
    return shards


def glob_paths(pattern: Path) -> list[Path]:
    return sorted(Path(pattern).parent.glob(Path(pattern).name))


def normalise_wave(data: Any) -> Any:
    """Mono float32 mixdown: one channel if many (average, as speech_metrics' loaders)."""
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr.astype("float32", copy=False)


def _resample(data: Any, src_rate: int) -> Any:
    from math import gcd

    g = gcd(int(src_rate), TARGET_RATE)
    up, down = TARGET_RATE // g, int(src_rate) // g
    return scipy.signal.resample_poly(data, up, down).astype("float32")


def write_flac(path: Path, data: Any, rate: int) -> None:
    soundfile.write(str(path), data, rate, format="FLAC", subtype="PCM_16")


def run(args: argparse.Namespace, results: dict[str, Any], out_dir: Path) -> tuple[int, str]:
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_dir():
        raise Refusal(f"--data-dir not a directory: {data_dir}")
    manifests_dir = Path(args.manifests_dir).expanduser()
    if not manifests_dir.is_dir():
        raise Refusal(f"--manifests-dir not a directory: {manifests_dir}")

    exclude_ids: dict[str, set[str]] = {}
    all_excluded_ids: set[str] = set()
    for label, filename in EXCLUDE_MANIFESTS:
        manifest = manifests_dir / filename
        if not manifest.is_file():
            raise Refusal(f"exclusion manifest not found: {manifest}")
        ids = read_manifest_ids(manifest)
        exclude_ids[label] = ids
        all_excluded_ids.update(ids)
        results["exclusion_ids"][label] = len(ids)

    test_manifest = manifests_dir / TEST_CALL_MANIFEST
    if not test_manifest.is_file():
        raise Refusal(
            f"test-call manifest not found: {test_manifest} (it names the OOD test calls, "
            "without which we cannot exclude them from the CPT pool)"
        )
    test_calls = read_source_ids(test_manifest)
    results["test_call_source_ids"] = len(test_calls)

    shards = read_shards(data_dir)
    results["shards"] = [str(shard) for shard in shards]

    import pyarrow.parquet as pq

    flac_dir = out_dir / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "pool.jsonl"

    rows_seen = 0
    rows_written = 0
    rows_resampled = 0
    stereo_merged = 0
    rows_failed = 0
    written_duration_s = 0.0

    with pool_path.open("w", encoding="utf-8") as pool_f:
        for shard in shards:
            table = pq.read_table(shard)
            columns = table.to_pydict()
            source_ids = columns["source_id"]
            audio_structs = columns["audio"]
            segment_ids = columns["segment_id"]
            sentences = columns["sentence"]
            start_ts_values = columns["start_ts"]
            end_ts_values = columns["end_ts"]
            for row_index in range(len(source_ids)):
                rows_seen += 1
                if args.limit is not None and rows_written >= args.limit:
                    results["limit_reason_not_written"] = rows_seen - rows_written
                    rows_seen = rows_written + 0
                    break
                source_id = str(source_ids[row_index]) if source_ids[row_index] is not None else ""
                segment_id = (
                    str(segment_ids[row_index]) if segment_ids[row_index] is not None else ""
                )
                rid = f"train_{source_id}_{segment_id}"
                if rid in all_excluded_ids:
                    if rid in exclude_ids["earnings_train6000"]:
                        results["skipped_by_reason"]["excluded_train6000"] += 1
                    elif rid in exclude_ids["earnings_validation1000"]:
                        results["skipped_by_reason"]["excluded_validation1000"] += 1
                    else:
                        results["skipped_by_reason"]["excluded_manifest_id"] += 1
                    continue
                call_key = source_id.replace("\\", "/").split("/")[0]
                if call_key in test_calls:
                    results["skipped_by_reason"]["excluded_test_call"] += 1
                    continue

                blob = audio_structs[row_index]
                if not isinstance(blob, dict) or "bytes" not in blob or blob["bytes"] is None:
                    results["skipped_by_reason"]["missing_audio_bytes"] += 1
                    continue
                try:
                    with __import__("io").BytesIO(blob["bytes"]) as handle:
                        data, rate = soundfile.read(handle, dtype="float32", always_2d=False)
                except Exception:
                    results["skipped_by_reason"]["audio_unreadable"] += 1
                    rows_failed += 1
                    continue
                data = normalise_wave(data)
                if getattr(data, "ndim", 1) == 2:
                    stereo_merged += 1
                if int(rate) != TARGET_RATE:
                    try:
                        data = _resample(data, int(rate))
                        rows_resampled += 1
                        rate = TARGET_RATE
                    except Exception:
                        results["skipped_by_reason"]["resample_failed"] += 1
                        rows_failed += 1
                        continue

                sentence = sentences[row_index]
                if not isinstance(sentence, str):
                    results["skipped_by_reason"]["missing_sentence"] += 1
                    continue

                if len(data) < TARGET_RATE // 10:
                    # a segment with (almost) no samples writes a 0-byte FLAC without raising;
                    # MEASURED: train_4446796_0 did, and the teacher decode died on it
                    results["skipped_by_reason"]["audio_too_short"] += 1
                    continue
                out_file = flac_dir / f"{rid}.flac"
                try:
                    write_flac(out_file, data, TARGET_RATE)
                    if out_file.stat().st_size == 0:
                        raise OSError("0-byte FLAC")
                except Exception:
                    out_file.unlink(missing_ok=True)
                    results["skipped_by_reason"]["flac_write_failed"] += 1
                    rows_failed += 1
                    continue

                duration = float(len(data)) / float(TARGET_RATE)
                written_duration_s += duration
                rows_written += 1
                pool_f.write(
                    json.dumps(
                        {
                            "id": rid,
                            "source_id": source_id,
                            "text": "Transcribe this audio exactly.",
                            "answer": sentence,
                            "audio": str(out_file.resolve()),
                            "duration": round(duration, 6),
                            "start_ts": start_ts_values[row_index],
                            "end_ts": end_ts_values[row_index],
                            "segment_id": segment_id,
                            "pool": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    counted = rows_written + sum(results["skipped_by_reason"].values())
    results["rows_seen"] = rows_seen
    results["rows_written"] = rows_written
    results["audio_failures"] = rows_failed
    results["rows_resampled"] = rows_resampled
    results["rows_mixed_to_mono"] = stereo_merged
    results["hours"] = round(written_duration_s / 3600.0, 4)
    results["rows_counted"] = counted
    results["rows_unaccounted"] = rows_seen - counted
    results["pool"] = str(pool_path.resolve())
    if rows_written == 0:
        return EXIT_COVERAGE_SHORT, f"coverage short: no row written from {rows_seen} shards rows"
    if results["rows_unaccounted"] not in (0, results.get("limit_reason_not_written", 0)):
        return (
            EXIT_COVERAGE_SHORT,
            f"coverage short: {results['rows_unaccounted']} rows left the run uncounted",
        )
    return EXIT_OK, "ok"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "build_cpt_pool.record.json"
    results: dict[str, Any] = {
        "kind": "speech_cpt_build_pool",
        "env": _env_record(),
        "args": vars(args),
        "target_rate": TARGET_RATE,
        "shards": [],
        "exclusion_ids": {},
        "test_call_source_ids": 0,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_counted": 0,
        "rows_unaccounted": 0,
        "audio_failures": 0,
        "rows_resampled": 0,
        "rows_mixed_to_mono": 0,
        "hours": 0.0,
        "skipped_by_reason": {
            "excluded_train6000": 0,
            "excluded_validation1000": 0,
            "excluded_manifest_id": 0,
            "excluded_test_call": 0,
            "missing_audio_bytes": 0,
            "audio_unreadable": 0,
            "resample_failed": 0,
            "missing_sentence": 0,
            "flac_write_failed": 0,
            "audio_too_short": 0,
        },
        "limit_reason_not_written": 0,
    }
    code, reason = EXIT_OK, "ok"
    t0 = time.time()
    try:
        code, reason = run(args, results, out_dir)
    except Refusal as exc:
        code, reason = EXIT_REFUSAL, f"refusal: {exc}"
        results["refusal"] = str(exc)
        sys.stderr.write(f"REFUSE {exc}\n")
    except SystemExit:
        code, reason = EXIT_REFUSAL, "refusal: exited before the pool was written"
        raise
    except Exception:
        results["fatal"] = traceback.format_exc()
        code, reason = 1, "fatal"
        sys.stderr.write(results["fatal"])
    finally:
        results["elapsed_s"] = time.time() - t0
        results["exit_code"] = code
        results["exit_reason"] = reason
        record_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"kind={results['kind']}")
    print(
        f"rows_seen={results['rows_seen']} written={results['rows_written']} "
        f"hours={results['hours']} resampled={results['rows_resampled']}"
    )
    for skip_reason, count in results["skipped_by_reason"].items():
        print(f"skip.{skip_reason}={count}")
    print(f"unaccounted={results['rows_unaccounted']}")
    print(f"pool={results.get('pool')}")
    print(f"record={record_path}")
    print(f"exit={code} {reason}")
    print(f"seconds={results['elapsed_s']:.1f}")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as se:
        sys.exit(se.code if se.code is not None else 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
