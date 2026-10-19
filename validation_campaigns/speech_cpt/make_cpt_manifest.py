#!/usr/bin/env python3
"""make_cpt_manifest.py -- CPT training labels from the teacher decode of the SP-1 pool.

ROADMAP SP-1 part B: the SFT manifest rows carry the REAL transcript as "answer", and SFT trains a
model to reproducely write THE SENTENCE that is the target of its own fine-tuning -- but that is the
label we are NOT allowed to use for a CPT corpus: the pool's transcrips ARE the tuning wording
(earnings22_robust, by construction) and training CPT against them would rehearse SFT's exam before
SFT is asked to write it.

So: we decode the pool with a TEACHER (eval_qwen3_asr.py's greedy transcript, wrapped by
decode.sh's eval.json record of {id, reference, hypothesis}), and take the TEACHER HYPOTHESIS as
"answer": CPT teaches the model to WRITE what its own speech hears (a weak label), unsupervised
from the transcript column. The pool's "answer" (the real sentence) is consumed here ONLY as the
reference the teacher was scored against -- corpus WER on the pool is a DIAGNOSTIC in record.json
(it says whether the weak labels are sound ASR), never a filter: text that is hard is still text a
model should learn to write, and we do not curate CPT into an easy subset.

Rows are dropped ONLY where the weak label is structurally useless as a training target, counted per
reason (no silent drops -- rows_seen = written + dropped):
  * missing_no_hypothesis -- eval.json has no hypothesis row for this id (the decode truncated,
    or failed its row) -- a row with no target teaches nothing,
  * empty_hypothesis -- normalize_transcript(hypothesis) is empty (a label of "" is not "write
    ASR", it is "write nothing"),
  * loop_hypothesis -- loop_words(hypothesis) > 0 (the teacher degenerated into a word loop;
    training on that is how a fine-tune learns to run away with itself),
  * speaking_rate_sanity -- word count of the hypothesis exceeds 4x the clip duration in seconds +
    10 (a 5 s clip cannot hold 200 words -- the teacher ignored the BEAM of what speech can say and
    hallucinated; the formula is in the row check below and is never used to reject normal dense
    speech, only runaway emissions).

is_runaway is NOT used: is_runaway(hypothesis, hypothesis) is self-referential (a hypothesis
against itself can only agree with itself), so the
loop detector and the speaking-rate bound stand in its place.

Output rows are exactly FoundationScale SFT manifest rows (train columns for speech audio SFT with
audio column "audio"): {"id", "text", "answer": <hypothesis>, "audio", "duration"} (+ optional
"source_id" for provenance).

Exit: 0 ok / 95 coverage short (no row, or rows dropped with no recorded reason) / 96 refusal (a
column or file is missing before anything is written).
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

from foundationscale.train.speech_metrics import (
    TRANSCRIPT_NORMALIZER_ID,
    loop_words,
    normalize_transcript,
    word_errors,
)

EXIT_OK = 0
EXIT_COVERAGE_SHORT = 95
EXIT_REFUSAL = 96

DROP_REASONS: tuple[str, ...] = (
    "missing_no_hypothesis",
    "empty_hypothesis",
    "loop_hypothesis",
    "speaking_rate_sanity",
)

WORD_LK_PER_SECOND = 4.0
WORD_LK_INTERCEPT = 10.0


class Refusal(Exception):
    """A precondition this script refuses before anything is written: exit 96."""


def _env_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "normalizer": TRANSCRIPT_NORMALIZER_ID,
    }
    import importlib.metadata as metadata

    for name in ("torch", "transformers", "soundfile"):
        try:
            record[name] = metadata.version(name)
        except Exception:
            record[name] = None
    return record


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build the SP-1 CPT training manifest from the teacher's decode of the pool."
    )
    ap.add_argument("--pool", required=True, help="pool.jsonl from build_cpt_pool.py")
    ap.add_argument(
        "--pseudo", required=True, help="eval.json from decode.sh on the pool (teacher decode)"
    )
    ap.add_argument("--out", required=True, help="CPT training manifest jsonl (SFT row format)")
    ap.add_argument("--record", required=None, help="record path (default: <out stem>.record.json)")
    ap.add_argument(
        "--limit", type=int, default=None, help="emit at most --limit rows (smoke runs)"
    )
    return ap.parse_args()


def read_pool(path: Path) -> list[dict[str, Any]]:
    """Every non-empty line is one pool row; a non-JSON line is REFUSED, not lost silently."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Refusal(f"use {path.name} line {lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise Refusal(
                    f"pool {path.name} line {lineno} is not a JSON object: {str(row)[:60]}"
                )
            for key in ("id", "answer", "audio", "duration"):
                if key not in row:
                    raise Refusal(
                        f"pool {path.name} line {lineno} misses field {key!r}: "
                        "a CPT row takes id/answer/audio/duration from it"
                    )
            rows.append(row)
    return rows


def read_teacher(path: Path) -> dict[str, dict[str, str]]:
    """id -> {"reference", "hypothesis"} from eval.json (eval_qwen3_asr.py's format)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Refusal(f"--pseudo {path} is not valid JSON: {exc}") from exc
    items = payload.get("all")
    if not isinstance(items, list):
        raise Refusal(f"--pseudo {path} has no 'all' list of {{id, reference, hypothesis}}")
    mapping: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            continue
        rid = str(item["id"])
        mapping[rid] = {
            "reference": str(item.get("reference", "")),
            "hypothesis": str(item.get("hypothesis", "")),
        }
    return mapping


def speaking_rate_ok(hypothesis: str, clip_duration_s: float) -> bool:
    """A teacher that hallucinates runs past what humans can say in -- 4 words/sec + 10 slack.

    This can only catch runaway emissions where the word count exceeds what sound can hold; it is a
    cheap bound that never rejects real (fast) dense speech: one word per 250 ms sustained plus 10
    words of interjection, punctuation and the like on top of it.
    """
    words = len(hypothesis) if isinstance(hypothesis, list) else len(hypothesis.split())
    max_words = WORD_LK_PER_SECOND * float(clip_duration_s) + WORD_LK_INTERCEPT
    return words <= max_words


def run(args: argparse.Namespace, results: dict[str, Any], out_path: Path) -> tuple[int, str]:
    pool_path = Path(args.pool).expanduser()
    if not pool_path.is_file():
        raise Refusal(f"--pool not found: {pool_path}")
    pseudo_path = Path(args.pseudo).expanduser()
    if not pseudo_path.is_file():
        raise Refusal(f"--pseudo not found: {pseudo_path}")

    pool_rows = read_pool(pool_path)
    results["pool_rows"] = len(pool_rows)
    if not pool_rows:
        raise Refusal(f"pool {pool_path} holds no rows")

    teacher = read_teacher(pseudo_path)
    results["teacher_records"] = len(teacher)
    meta = json.loads(pseudo_path.read_text(encoding="utf-8"))
    if meta.get("fatal") or len(teacher) < len(pool_rows):
        # a partial teacher decode is not a weak label for the missing rows: refuse, never train
        # on the subset that happened to finish
        raise Refusal(
            f"teacher decode incomplete: {len(teacher)} of {len(pool_rows)} rows"
            f"{' (fatal recorded)' if meta.get('fatal') else ''}; rerun the decode"
        )

    written = 0
    wer_sum = 0.0
    wer_compared = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in pool_rows:
            if args.limit is not None and written >= args.limit:
                break
            rid = str(row["id"])
            ref = row["answer"]
            record = teacher.get(rid)
            if record is None:
                results["dropped_by_reason"]["missing_no_hypothesis"] += 1
                continue

            hypothesis = record["hypothesis"]
            norm_hyp = normalize_transcript(hypothesis)
            if not norm_hyp:
                results["dropped_by_reason"]["empty_hypothesis"] += 1
                continue
            if loop_words(norm_hyp) > 0:
                results["dropped_by_reason"]["loop_hypothesis"] += 1
                continue
            if not speaking_rate_ok(norm_hyp, float(row["duration"])):
                results["dropped_by_reason"]["speaking_rate_sanity"] += 1
                continue

            if normalize_transcript(ref):
                diagnostic_wer = word_errors(ref, hypothesis).rate()
                wer_sum += float(diagnostic_wer)
                wer_compared += 1
                results.setdefault("diagnostic_samples", [])
                if len(results["diagnostic_samples"]) < 10:
                    results["diagnostic_samples"].append(
                        {
                            "id": rid,
                            "reference": ref,
                            "hypothesis": hypothesis,
                            "wer": float(diagnostic_wer),
                        }
                    )

            out_f.write(
                json.dumps(
                    {
                        "id": rid,
                        "text": "Transcribe this audio exactly.",
                        "answer": hypothesis,
                        "audio": row["audio"],
                        "duration": row["duration"],
                        "source_id": row.get("source_id"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    results["rows_written"] = written
    results["rows_seen"] = (
        len(pool_rows)
        if args.limit is None
        else written + sum(results["dropped_by_reason"].values())
    )
    results["rows_unaccounted"] = (
        results["rows_seen"] - written - sum(results["dropped_by_reason"].values())
    )
    results["corpus_error_rate_diagnostic"] = (
        round(wer_sum / wer_compared, 6) if wer_compared else None
    )
    results["diagnostic_compared_rows"] = wer_compared
    results["out"] = str(out_path.resolve())

    if results["rows_unaccounted"] != 0:
        return EXIT_COVERAGE_SHORT, (
            f"coverage short: {results['rows_unaccounted']} rows left the run "
            "with neither a training row nor a counted drop"
        )
    if written == 0:
        return (
            EXIT_COVERAGE_SHORT,
            f"coverage short: 0 training rows from {results['rows_seen']} pool rows",
        )
    return EXIT_OK, "ok"


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record_path = (
        Path(args.record).expanduser()
        if args.record
        else out_path.parent / (out_path.stem + ".record.json")
    )
    results: dict[str, Any] = {
        "kind": "speech_cpt_manifest",
        "env": _env_record(),
        "args": vars(args),
        "normalizer": TRANSCRIPT_NORMALIZER_ID,
        "pool_rows": 0,
        "teacher_records": 0,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_unaccounted": 0,
        "corpus_error_rate_diagnostic": None,
        "diagnostic_compared_rows": 0,
        "diagnostic_samples": [],
        "dropped_by_reason": {reason: 0 for reason in DROP_REASONS},
    }
    code, reason = EXIT_OK, "ok"
    t0 = time.time()
    try:
        code, reason = run(args, results, out_path)
    except Refusal as exc:
        code, reason = EXIT_REFUSAL, f"refusal: {exc}"
        results["refusal"] = str(exc)
        sys.stderr.write(f"REFUSE {exc}\n")
    except SystemExit:
        code, reason = EXIT_REFUSAL, "refusal: exited before the manifest was written"
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
        f"unaccounted={results['rows_unaccounted']}"
    )
    for drop_reason, count in results["dropped_by_reason"].items():
        print(f"drop.{drop_reason}={count}")
    print(
        f"corpus_error_rate_diagnostic={results['corpus_error_rate_diagnostic']} "
        f"compared={results['diagnostic_compared_rows']}"
    )
    print(f"out={results['out']}")
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
