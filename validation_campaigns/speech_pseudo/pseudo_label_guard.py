"""Teacher pseudo-labels for untranscribed ASR audio, and the refusal that keeps evaluation speech out of them.

purpose
-------
An unlabeled public corpus can contain the same broadcasts or recordings an evaluation set was cut
from. Matching whole sentences misses it: the corpus is segmented differently from the evaluation
set, so a sentence the evaluation set holds whole is split across several corpus clips and never
matches verbatim. Overlap is therefore measured on WORD 8-GRAMS of the TEACHER transcripts against
the evaluation REFERENCES, and a recording that clearly overlaps is excluded WHOLE -- its
neighbouring segments are the same recording, and keeping the segments that happened to fall
outside an 8-gram is exactly how the same broadcast re-enters the training pool.

Measured once, on two pools:
  a 966 h public unlabeled medical corpus held clips sharing >= 3 rare 8-grams with an evaluation
  set in 647 of its 672 recordings -- REFUSED, because a pool whose recordings are almost all
  present in the evaluation material is that material, not an unlabeled corpus beside it;
  a 4,331 h radio/podcast pool flagged 10.0% of recordings and 0.47% of clips -- ACCEPTED with the
  flagged parts excluded, because a tenth of the recordings carrying a shared phrase is a phrase,
  not a source.

Template phrases (dates, greetings, station idents) make ANY-8-gram matches common on broadcast
audio -- 12.9% of clips there carry at least one -- so the verdict is driven by the STRONG signal
(several shared 8-grams that are RARE across the pool) and never by the bare match count.

inputs
------
``candidates``   manifest rows {"id", "audio", "duration", "rec"?} -- ``rec`` is the recording /
                 source-group id and defaults to the clip id (a clip with no siblings).
``teacher``      the teacher's decode of those clips, as an evaluator output
                 {"all": [{"id", "hypothesis", ...}]} -- every candidate MUST be decoded.
``student``      optional, the same shape: a kept row must then also agree with the teacher.
``eval_refs``    JSON lines holding the evaluation references in "answer" or "text".

outputs
-------
``PSEUDO.jsonl`` the candidate rows with "answer" = the teacher text (and "cer_agree" when a
                 student decode was given), written atomically.
``REPORT.json``  the accounting: candidates and hours, decoded, strong clips with their share,
                 recordings and recordings dropped with their share, the clip-level drops
                 (8-gram, short, disagree), kept rows and hours, the top shared n-grams with
                 counts, every threshold, and the verdict with its reasons.

usage
-----
  python validation_campaigns/speech_pseudo/pseudo_label_guard.py --candidates CANDS.jsonl \
      --teacher TEACHER_EVAL.json [--student STUDENT_EVAL.json] \
      --eval-refs REF1.jsonl [--eval-refs REF2.jsonl ...] \
      --out PSEUDO.jsonl --report REPORT.json \
      [--n 8] [--rare-max 2] [--strong-min 3] [--max-cer 0.03] \
      [--max-rec-drop 0.30] [--max-strong 0.01] [--min-hours 50]

Exit 0 on an OK verdict, 5 on a RED_FLAG (the pseudo file is still written, so the flagged pool can
be inspected), 2 on refused input.

WHAT IS NOT CLAIMED
-------------------
  the kept pseudo-labels are   they are the TEACHER's transcripts of audio nobody transcribed, and
  not verified transcripts     nothing here checks them against the audio. Agreement with a student
                               decode is two models sharing an error, not a correction of one.
  an OK verdict is not         the 8-gram test sees VERBATIM overlap only. A re-read, a paraphrase,
  "this pool is clean"         a translation, or the same story in other words shares no 8-gram and
                               passes. The verdict is about shared wording and nothing else.
  the thresholds generalize    3 rare 8-grams, 30% of recordings, 1% strong clips and 50 h are the
                               arms that separated the two pools above. Another corpus, another
                               segmentation or another language may need other arms, and this
                               module will not notice that it does.
  a gain is claimed            no WER, no training run and no downstream score is measured here.
                               This file decides what may ENTER a training pool and says nothing
                               about what training on it does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence

from foundationscale.train.speech_metrics import char_errors, normalize_transcript

__all__ = ["agree", "eval_ngrams", "ngrams", "overlap", "verdict", "words"]


def _refuse_number(
    name: str, value: object, *, minimum: float, maximum: float | None = None
) -> float:
    """``value`` as a float inside its declared range, or a refusal naming the parameter.

    A threshold outside its range is not a threshold that wants tuning, it is a typo that silently
    re-arms the detector (``--rare-max -1`` would make every shared 8-gram "rare" and refuse the
    whole pool), so it is refused here rather than absorbed into a report nobody can re-derive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        span = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {span}, got {value!r}")
    return number


def _share(part: int, whole: int) -> float:
    """``part`` over ``whole`` -- 0.0 where there is nothing to take a share OF.

    Not a 0.0 of measurement: zero recordings cannot have dropped a share of themselves, and that
    is a fact about the pool rather than a claim that it is clean.
    """
    return part / whole if whole else 0.0


def words(text: str) -> list[str]:
    """``text`` as the words the 8-gram overlap test measures.

    This IS ``speech_metrics.normalize_transcript`` under a local name: case, punctuation and token
    edges fold here, so a teacher transcript and an evaluation reference can only ever meet as
    n-grams of the SAME normalized words. Refuses a non-string -- a manifest row that carried a
    number where a transcript belongs is a defect to name, not a transcript to normalize.
    """
    if not isinstance(text, str):
        raise ValueError(f"transcript must be a str, got {type(text).__name__}")
    return normalize_transcript(text)


def _token_list(tokens: Sequence[str]) -> list[str]:
    """``tokens`` as a list of words, or a refusal naming the offending position."""
    if isinstance(tokens, (str, bytes)):
        raise ValueError("tokens must be a sequence of words, not one string: run words() first")
    checked: list[str] = []
    for position, token in enumerate(tokens):
        if not isinstance(token, str):
            raise ValueError(f"token {position} must be a str, got {type(token).__name__}")
        checked.append(token)
    return checked


def ngrams(tokens: Sequence[str], n: int = 8) -> set[tuple[str, ...]]:
    """The word ``n``-grams of ``tokens``, as tuples -- the unit the overlap test compares.

    Fewer than ``n`` words is an EMPTY set and not an error: a short clip has no 8-gram to share,
    and padding one into existence would let a seven-word greeting match a reference. Refuses
    ``n < 1`` (there is no 0-gram to measure) and a token that is not a word.
    """
    order = int(_refuse_number("n", n, minimum=1))
    checked = _token_list(tokens)
    return {tuple(checked[i : i + order]) for i in range(len(checked) - order + 1)}


def eval_ngrams(reference_texts: Sequence[str], n: int = 8) -> set[tuple[str, ...]]:
    """Every word ``n``-gram of every evaluation reference -- the set a candidate must stay out of.

    The union over ALL reference texts: an evaluation set is many utterances and a candidate
    overlaps it by matching any one of them. Refuses a reference that is not a transcript string,
    because a silently skipped reference is an evaluation set the guard never looked at.
    """
    order = int(_refuse_number("n", n, minimum=1))
    grams: set[tuple[str, ...]] = set()
    for position, text in enumerate(reference_texts):
        if not isinstance(text, str):
            raise ValueError(f"reference {position} must be a str, got {type(text).__name__}")
        grams |= ngrams(words(text), order)
    return grams


def _candidate_fields(row: Mapping[str, object], position: int) -> tuple[str, str, str]:
    """``(id, rec, teacher)`` from one overlap candidate, or a refusal naming the row."""
    for key in ("id", "rec", "teacher"):
        if key not in row:
            raise ValueError(
                f"candidate {position} is missing {key!r}: overlap needs the clip id, its recording "
                "id and the teacher transcript"
            )
        if not isinstance(row[key], str):
            raise ValueError(f"candidate {position} has a non-string {key!r} ({row[key]!r})")
    clip_id, rec, teacher = str(row["id"]), str(row["rec"]), str(row["teacher"])
    if not clip_id:
        raise ValueError(
            f"candidate {position} has an empty 'id': an unnamed clip cannot be accounted for"
        )
    return clip_id, rec, teacher


def overlap(
    candidates: Sequence[Mapping[str, object]],
    eval_grams: set[tuple[str, ...]],
    *,
    n: int = 8,
    rare_max: int = 2,
    strong_min: int = 3,
) -> dict[str, object]:
    """Teacher transcripts against the evaluation n-grams: what is shared, what is STRONG, what stays.

    Per candidate, ``shared`` are its n-grams that occur in ``eval_grams``; ``freq`` counts, for
    each shared n-gram, how many CANDIDATES share it. That frequency is the whole distinction this
    guard rests on: a date or a greeting appears in many pool clips and in the evaluation set both,
    so one shared 8-gram proves nothing (12.9% of broadcast clips carry one), while an 8-gram that
    is RARE across the pool and still shared is wording that only a common source produces. A
    candidate is ``strong`` when it shares >= ``strong_min`` n-grams of frequency <= ``rare_max``,
    and its whole recording joins ``bad_recs`` -- the neighbouring segments are the same recording.

    ``keep`` is the survivors, in candidate order: in no bad recording, sharing NO n-gram at all
    (the clip-level drop is the same refusal at a smaller unit), and carrying >= 2 normalized words
    (a one-word "transcript" is not a pseudo-label, it is noise with an answer field).

    Refuses a candidate row that is not the documented triple, a duplicated id (two rows claiming
    one clip would double-count its n-grams and corrupt ``freq``), and thresholds outside their
    ranges.
    """
    order = int(_refuse_number("n", n, minimum=1))
    rare_ceiling = int(_refuse_number("rare_max", rare_max, minimum=0))
    strong_floor = int(_refuse_number("strong_min", strong_min, minimum=1))
    if not isinstance(eval_grams, set):
        raise ValueError(
            f"eval_grams must be the set eval_ngrams() returns, got {type(eval_grams).__name__}"
        )

    shared_by_id: dict[str, set[tuple[str, ...]]] = {}
    rec_of: dict[str, str] = {}
    word_count: dict[str, int] = {}
    for position, row in enumerate(candidates):
        clip_id, rec, teacher = _candidate_fields(row, position)
        if clip_id in shared_by_id:
            raise ValueError(
                f"duplicate candidate id {clip_id!r}: one clip may only be counted once"
            )
        tokens = words(teacher)
        shared_by_id[clip_id] = ngrams(tokens, order) & eval_grams
        rec_of[clip_id] = rec
        word_count[clip_id] = len(tokens)

    freq: dict[tuple[str, ...], int] = {}
    for shared in shared_by_id.values():
        for gram in shared:
            freq[gram] = freq.get(gram, 0) + 1

    strong = {
        clip_id
        for clip_id, shared in shared_by_id.items()
        if sum(1 for gram in shared if freq[gram] <= rare_ceiling) >= strong_floor
    }
    bad_recs = {rec_of[clip_id] for clip_id in strong}
    clip_shared = {clip_id for clip_id, shared in shared_by_id.items() if shared}
    keep = [
        clip_id
        for clip_id, shared in shared_by_id.items()
        if rec_of[clip_id] not in bad_recs and not shared and word_count[clip_id] >= 2
    ]
    return {
        "clip_shared": clip_shared,
        "strong": strong,
        "bad_recs": bad_recs,
        "freq": freq,
        "keep": keep,
    }


def agree(teacher: str, student: str, max_cer: float = 0.03) -> tuple[bool, float]:
    """Does the student decode the clip the way the teacher did -- close enough to trust the text?

    The character error rate of the NORMALIZED student transcript against the NORMALIZED teacher
    transcript (``speech_metrics.char_errors``), kept when ``cer <= max_cer``. The teacher is the
    reference here because its text is what would become the answer: a student that says something
    else is either the teacher hallucinating or the student failing, and this guard cannot tell
    which, so it keeps neither.

    An empty normalized teacher is DISAGREEMENT with ``cer`` 1.0 rather than an exception: the rate
    is genuinely undefined over a zero-length reference (``char_errors`` would refuse to publish
    one), and "undefined" must not be laundered into a pass -- a pseudo-label with no words in it
    is the one row that can never be right.
    """
    ceiling = _refuse_number("max_cer", max_cer, minimum=0.0)
    if not isinstance(student, str):
        raise ValueError(f"student transcript must be a str, got {type(student).__name__}")
    if not words(teacher):
        return False, 1.0
    cer = char_errors(teacher, student).rate()
    return cer <= ceiling, cer


def _report_value(report: Mapping[str, object], key: str) -> object:
    """The value ``report`` must carry, or a refusal naming the missing key."""
    if key not in report:
        raise ValueError(
            f"report is missing {key!r}: verdict() reads the counts the report publishes"
        )
    return report[key]


def _report_count(report: Mapping[str, object], key: str) -> int:
    """A non-negative integer count out of ``report``, or a refusal naming the key."""
    value = _report_value(report, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"report {key} must be a non-negative int, got {value!r}")
    return value


def verdict(
    report: Mapping[str, object],
    *,
    max_rec_drop: float = 0.30,
    max_strong: float = 0.01,
    min_hours: float = 50.0,
) -> tuple[str, list[str]]:
    """``report`` as one verdict and the reasons that produced it -- the campaign's go / no-go.

    Three arms, each a refusal with a reason attached rather than a silent shrink of the pool:

      recording drop share   ``recordings_dropped / recordings`` above ``max_rec_drop``. This is the
                             arm that caught the 966 h medical corpus (647 of 672 recordings) and
                             waved the 4,331 h radio/podcast pool through at 10.0%: a pool cut from
                             the evaluation material loses most of its recordings, a pool beside it
                             loses a phrase.

      strong-clip share      ``strong_clips / candidates`` above ``max_strong``. Several RARE shared
                             8-grams in one clip is source overlap; the bare match count is not,
                             which is why 12.9% of broadcast clips carrying a template 8-gram is not
                             this arm.

      kept hours             below ``min_hours``. A refusal that empties the pool has not saved the
                             evaluation set, it has only destroyed the campaign, and that is a
                             RED_FLAG too -- the operator must see it rather than ship 4 h.

    ``"OK"`` with an empty reason list, or ``"RED_FLAG"`` with every reason that fired (all of them,
    not the first: an operator fixing one arm at a time needs the whole list). Refuses a report
    missing a count it must read and a threshold outside its range.
    """
    rec_ceiling = _refuse_number("max_rec_drop", max_rec_drop, minimum=0.0, maximum=1.0)
    strong_ceiling = _refuse_number("max_strong", max_strong, minimum=0.0, maximum=1.0)
    hours_floor = _refuse_number("min_hours", min_hours, minimum=0.0)

    candidates = _report_count(report, "candidates")
    strong_clips = _report_count(report, "strong_clips")
    recordings = _report_count(report, "recordings")
    recordings_dropped = _report_count(report, "recordings_dropped")
    kept_hours = _refuse_number("kept_hours", _report_value(report, "kept_hours"), minimum=0.0)

    reasons: list[str] = []
    rec_share = _share(recordings_dropped, recordings)
    if rec_share > rec_ceiling:
        reasons.append(
            f"recordings dropped: {recordings_dropped}/{recordings} ({rec_share:.4f}) is above "
            f"max_rec_drop {rec_ceiling:.4f} -- a recording that clearly overlaps an evaluation set is "
            "excluded whole, and this pool is cut from that material"
        )
    strong_share = _share(strong_clips, candidates)
    if strong_share > strong_ceiling:
        reasons.append(
            f"strong clips: {strong_clips}/{candidates} ({strong_share:.4f}) is above max_strong "
            f"{strong_ceiling:.4f} -- several rare shared 8-grams is source overlap, not a template phrase"
        )
    if kept_hours < hours_floor:
        reasons.append(
            f"kept hours: {kept_hours:.1f} is below min_hours {hours_floor:.1f} -- what survives the "
            "refusal is too small to be the pool this campaign promised"
        )
    return ("RED_FLAG", reasons) if reasons else ("OK", [])


def read_jsonl(path: str) -> list[dict[str, object]]:
    """Every JSON object in ``path``, one per line, or a refusal naming file and line."""
    rows: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object, got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def candidate_id(row: Mapping[str, object]) -> str:
    """The clip id of a manifest row."""
    value = row.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate row has no usable 'id' ({value!r})")
    return value


def candidate_rec(row: Mapping[str, object]) -> str:
    """The recording / source-group id of a manifest row -- the clip id when it has no siblings."""
    value = row.get("rec", row.get("id"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate row has no usable 'rec' ({value!r})")
    return value


def candidate_duration(row: Mapping[str, object]) -> float:
    """The clip duration in seconds -- the unit the hours in a report are summed in."""
    value = row.get("duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate {row.get('id')!r} has a non-numeric 'duration' ({value!r})")
    seconds = float(value)
    if seconds < 0.0:
        raise ValueError(f"candidate {row.get('id')!r} has a negative duration ({value!r})")
    return seconds


def read_candidates(path: str) -> list[dict[str, object]]:
    """The manifest rows of ``path``, validated and in file order.

    Refuses a row without ``id`` / ``audio`` / ``duration``, a duplicated id (two rows claiming one
    clip would double-count it in every share the report publishes) and a duration that is not a
    non-negative number of seconds.
    """
    rows = read_jsonl(path)
    seen: set[str] = set()
    for position, row in enumerate(rows, start=1):
        for key in ("id", "audio", "duration"):
            if key not in row:
                raise ValueError(f"{path}:{position}: candidate row is missing {key!r}")
        clip_id = candidate_id(row)
        if clip_id in seen:
            raise ValueError(f"{path}:{position}: duplicate candidate id {clip_id!r}")
        seen.add(clip_id)
        candidate_rec(row)
        candidate_duration(row)
    return rows


def read_hypotheses(path: str, role: str) -> dict[str, str]:
    """``{id: hypothesis}`` from an evaluator output file, or a refusal naming the shape it wanted."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("all"), list):
        raise ValueError(f"{path}: {role} eval must be an object holding an 'all' list of rows")
    hypotheses: dict[str, str] = {}
    for position, item in enumerate(payload["all"], start=1):
        if not isinstance(item, dict) or "id" not in item or "hypothesis" not in item:
            raise ValueError(f"{path}: 'all'[{position}] must carry 'id' and 'hypothesis'")
        clip_id = str(item["id"])
        if clip_id in hypotheses:
            raise ValueError(f"{path}: duplicate {role} hypothesis for id {clip_id!r}")
        hypotheses[clip_id] = str(item["hypothesis"])
    return hypotheses


def read_references(paths: Sequence[str]) -> list[str]:
    """Every evaluation reference in ``paths``, from "answer" or "text".

    Refuses a row with NEITHER field, naming file and line: a reference file that silently yields
    fewer references than it holds is an evaluation set the overlap test only half looked at, and
    the missing half is exactly where a shared recording would hide.
    """
    texts: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: reference row must be a JSON object")
                if "answer" in row:
                    text = row["answer"]
                elif "text" in row:
                    text = row["text"]
                else:
                    raise ValueError(
                        f"{path}:{line_number}: reference row has neither 'answer' nor 'text' -- a row "
                        "with no reference cannot anchor the overlap test"
                    )
                if not isinstance(text, str):
                    raise ValueError(
                        f"{path}:{line_number}: reference text must be a str, got {type(text).__name__}"
                    )
                texts.append(text)
    return texts


def pseudo_row(
    row: Mapping[str, object], teacher_text: str, cer: float | None
) -> dict[str, object]:
    """The manifest row as a pseudo-labeled row: the teacher's text becomes the answer."""
    out = dict(row)
    out["answer"] = teacher_text.strip()
    if cer is not None:
        out["cer_agree"] = round(cer, 4)
    return out


def build_report(
    *,
    candidates: int,
    candidate_hours: float,
    decoded: int,
    strong_clips: int,
    recordings: int,
    recordings_dropped: int,
    drops: Mapping[str, int],
    kept_rows: int,
    kept_hours: float,
    top_shared_ngrams: Sequence[tuple[str, int]],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    """Every number this campaign measured, in one JSON-ready accounting.

    The drop counters are a LEDGER: ``recording_clip_drops`` + ``clip_8gram_drops`` + ``short_drops``
    + ``disagree_drops`` + ``kept_rows`` is ``candidates``, so a reader can see that every clip was
    either kept or refused for a stated reason. A clip refused for two reasons is counted once, by
    the first reason in that order -- the recording refusal first, because it is the strongest claim.
    """
    return {
        "candidates": candidates,
        "candidate_hours": round(candidate_hours, 1),
        "decoded": decoded,
        "strong_clips": strong_clips,
        "strong_share": round(_share(strong_clips, candidates), 6),
        "recordings": recordings,
        "recordings_dropped": recordings_dropped,
        "recordings_dropped_share": round(_share(recordings_dropped, recordings), 6),
        "recording_clip_drops": drops.get("recording_clip_drops", 0),
        "clip_8gram_drops": drops.get("clip_8gram_drops", 0),
        "short_drops": drops.get("short_drops", 0),
        "disagree_drops": drops.get("disagree_drops", 0),
        "kept_rows": kept_rows,
        "kept_hours": round(kept_hours, 1),
        "top_shared_ngrams": [[gram, count] for gram, count in top_shared_ngrams],
        "thresholds": dict(thresholds),
    }


def write_jsonl_atomic(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    """``rows`` as JSON lines at ``path``, through a temp file and ``os.replace``.

    Atomic because a half-written pseudo file is indistinguishable from a complete one once it
    lands in a training pipeline: the reader sees every row or no rows, never the first 40%.
    """
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json_atomic(path: str, payload: Mapping[str, object]) -> None:
    """``payload`` as one JSON object at ``path``, atomically -- see ``write_jsonl_atomic``."""
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The command line as a namespace, with every ``--eval-refs`` group flattened to one list."""
    parser = argparse.ArgumentParser(
        prog="pseudo_label_guard.py",
        description="Teacher pseudo-labels for untranscribed audio, with the evaluation-overlap refusal.",
        usage=(
            "pseudo_label_guard.py --candidates CANDS.jsonl --teacher TEACHER_EVAL.json "
            "[--student STUDENT_EVAL.json] --eval-refs REF1.jsonl [--eval-refs REF2.jsonl ...] "
            "--out PSEUDO.jsonl --report REPORT.json [--n 8] [--rare-max 2] [--strong-min 3] "
            "[--max-cer 0.03] [--max-rec-drop 0.30] [--max-strong 0.01] [--min-hours 50]"
        ),
    )
    parser.add_argument(
        "--candidates", required=True, help="FS manifest rows: id, audio, duration, rec?"
    )
    parser.add_argument(
        "--teacher", required=True, help="teacher decode, evaluator output {'all': [...]}"
    )
    parser.add_argument("--student", help="optional student decode, same shape as --teacher")
    parser.add_argument(
        "--eval-refs",
        required=True,
        nargs="+",
        action="append",
        metavar="REF.jsonl",
        help="evaluation references, JSON lines with 'answer' or 'text' (repeatable)",
    )
    parser.add_argument("--out", required=True, help="pseudo-labeled JSONL to write")
    parser.add_argument("--report", required=True, help="report JSON to write")
    parser.add_argument("--n", type=int, default=8, help="word n-gram order (default 8)")
    parser.add_argument(
        "--rare-max", type=int, default=2, help="an n-gram at most this common is rare"
    )
    parser.add_argument(
        "--strong-min", type=int, default=3, help="rare shared n-grams that make a clip strong"
    )
    parser.add_argument("--max-cer", type=float, default=0.03, help="teacher/student CER ceiling")
    parser.add_argument(
        "--max-rec-drop", type=float, default=0.30, help="recording drop share ceiling"
    )
    parser.add_argument("--max-strong", type=float, default=0.01, help="strong-clip share ceiling")
    parser.add_argument("--min-hours", type=float, default=50.0, help="hours that must survive")
    args = parser.parse_args(argv)
    args.eval_refs = [path for group in args.eval_refs for path in group]
    return args


def run(args: argparse.Namespace) -> dict[str, object]:
    """Read, refuse, filter, write -- and the report that accounts for every clip."""
    n = int(_refuse_number("--n", args.n, minimum=1))
    rare_max = int(_refuse_number("--rare-max", args.rare_max, minimum=0))
    strong_min = int(_refuse_number("--strong-min", args.strong_min, minimum=1))
    max_cer = _refuse_number("--max-cer", args.max_cer, minimum=0.0)
    max_rec_drop = _refuse_number("--max-rec-drop", args.max_rec_drop, minimum=0.0, maximum=1.0)
    max_strong = _refuse_number("--max-strong", args.max_strong, minimum=0.0, maximum=1.0)
    min_hours = _refuse_number("--min-hours", args.min_hours, minimum=0.0)

    candidates = read_candidates(args.candidates)
    teacher = read_hypotheses(args.teacher, "teacher")
    student = read_hypotheses(args.student, "student") if args.student else {}
    references = read_references(args.eval_refs)

    ids = [candidate_id(row) for row in candidates]
    missing = sorted({clip_id for clip_id in ids if clip_id not in teacher})
    if missing:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        raise ValueError(
            f"{len(missing)} of {len(candidates)} candidate ids have no teacher hypothesis ({shown}): "
            "every pseudo-label IS the teacher's transcript, so an undecoded clip is a refusal and "
            "not a row to fill in later"
        )

    eval_grams = eval_ngrams(references, n=n)
    measured = [
        {"id": candidate_id(row), "rec": candidate_rec(row), "teacher": teacher[candidate_id(row)]}
        for row in candidates
    ]
    hits = overlap(measured, eval_grams, n=n, rare_max=rare_max, strong_min=strong_min)
    bad_recs = hits["bad_recs"]
    clip_shared = hits["clip_shared"]

    drops: dict[str, int] = {}
    kept: list[dict[str, object]] = []
    for row in candidates:
        clip_id = candidate_id(row)
        text = teacher[clip_id]
        if candidate_rec(row) in bad_recs:
            drops["recording_clip_drops"] = drops.get("recording_clip_drops", 0) + 1
            continue
        if clip_id in clip_shared:
            drops["clip_8gram_drops"] = drops.get("clip_8gram_drops", 0) + 1
            continue
        if len(words(text)) < 2:
            drops["short_drops"] = drops.get("short_drops", 0) + 1
            continue
        cer: float | None = None
        if args.student:
            # A clip the student never decoded scores as a full deletion and is dropped here: it
            # cannot pass agree(), and keeping it would ship a text nothing corroborated.
            passed, cer = agree(text, student.get(clip_id, ""), max_cer=max_cer)
            if not passed:
                drops["disagree_drops"] = drops.get("disagree_drops", 0) + 1
                continue
        kept.append(pseudo_row(row, text, cer))

    write_jsonl_atomic(args.out, kept)

    freq = hits["freq"]
    top = sorted(freq.items(), key=lambda item: (-item[1], " ".join(item[0])))[:10]
    report = build_report(
        candidates=len(candidates),
        candidate_hours=sum(candidate_duration(row) for row in candidates) / 3600.0,
        decoded=len(teacher),
        strong_clips=len(hits["strong"]),
        recordings=len({candidate_rec(row) for row in candidates}),
        recordings_dropped=len(bad_recs),
        drops=drops,
        kept_rows=len(kept),
        kept_hours=sum(candidate_duration(row) for row in kept) / 3600.0,
        top_shared_ngrams=[(" ".join(gram), count) for gram, count in top],
        thresholds={
            "n": n,
            "rare_max": rare_max,
            "strong_min": strong_min,
            "max_cer": max_cer,
            "max_rec_drop": max_rec_drop,
            "max_strong": max_strong,
            "min_hours": min_hours,
        },
    )
    label, reasons = verdict(
        report, max_rec_drop=max_rec_drop, max_strong=max_strong, min_hours=min_hours
    )
    report["verdict"] = label
    report["reasons"] = reasons
    write_json_atomic(args.report, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI: 0 on OK, 5 on RED_FLAG (pseudo file still written), 2 on refused input."""
    args = parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "reasons": report["reasons"],
                "kept_rows": report["kept_rows"],
                "kept_hours": report["kept_hours"],
            },
            ensure_ascii=False,
        )
    )
    return 5 if report["verdict"] == "RED_FLAG" else 0


if __name__ == "__main__":
    sys.exit(main())
