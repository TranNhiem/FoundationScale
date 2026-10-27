"""Known-case, contract and end-to-end tests for ``pseudo_label_guard``.

Every test is a MUST_PASS / MUST_FIRE pair: the claim the accepted path must publish, and the
refusal or count that must fire beside it. Nothing here is skipped, xfailed or parametrized away
-- these assertions ARE the measured claim, and a pseudo-label pool that quietly absorbs the
evaluation set is a benchmark that was never measured in the first place. Synthetic text only: no
GPU, no network, no audio, and the script is loaded by path because it lives outside the package.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "validation_campaigns" / "speech_pseudo" / "pseudo_label_guard.py"

_spec = importlib.util.spec_from_file_location("pseudo_label_guard", MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load pseudo_label_guard from {MODULE_PATH}")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

EVAL_SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank today"
TEMPLATE = "good morning and welcome to the program today"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _write_eval(path: Path, hypotheses: dict[str, str]) -> None:
    payload = {"all": [{"id": clip_id, "hypothesis": text} for clip_id, text in hypotheses.items()]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _clips() -> list[dict[str, object]]:
    """Six synthetic clips over three recordings, 10 h each -- 60 h, above the default min_hours."""
    return [
        {
            "id": f"c{index + 1}",
            "audio": f"audio/c{index + 1}.wav",
            "duration": 36000.0,
            "rec": f"rec-{'abc'[index // 2]}",
        }
        for index in range(6)
    ]


def _teacher(rows: list[dict[str, object]]) -> dict[str, str]:
    """Clean teacher text per clip: >= 2 words and no 8-gram of either evaluation reference."""
    return {
        str(row["id"]): f"clip {row['id']} covers the morning traffic on the ring road"
        for row in rows
    }


def _argv(
    candidates: Path,
    teacher: Path,
    refs: list[Path],
    out: Path,
    report: Path,
    **extra: str,
) -> list[str]:
    argv = ["--candidates", str(candidates), "--teacher", str(teacher)]
    for ref in refs:
        argv += ["--eval-refs", str(ref)]
    argv += ["--out", str(out), "--report", str(report)]
    for name, value in extra.items():
        argv += ["--" + name.replace("_", "-"), str(value)]
    return argv


def _report(**overrides: object) -> dict[str, object]:
    """A complete report at values that pass every arm -- the baseline each RED_FLAG perturbs."""
    report: dict[str, object] = {
        "candidates": 100,
        "candidate_hours": 100.0,
        "decoded": 100,
        "strong_clips": 0,
        "strong_share": 0.0,
        "recordings": 10,
        "recordings_dropped": 1,
        "recordings_dropped_share": 0.1,
        "recording_clip_drops": 4,
        "clip_8gram_drops": 2,
        "short_drops": 1,
        "disagree_drops": 0,
        "kept_rows": 93,
        "kept_hours": 60.0,
        "top_shared_ngrams": [],
        "thresholds": {},
        "verdict": "",
        "reasons": [],
    }
    report.update(overrides)
    return report


def test_ngrams_and_eval_ngrams_short_and_long() -> None:
    """MUST_PASS: eight words is exactly one word 8-gram and fourteen words is seven of them.

    MUST_FIRE: fewer than n words is an EMPTY set -- a seven-word greeting carries no 8-gram to
    share, and padding one into existence would let it match an evaluation reference.
    """
    seven = guard.words("one two three four five six seven")
    assert len(seven) == 7, "MUST_PASS: the short input really is seven normalized words"
    assert guard.ngrams(seven) == set(), (
        "MUST_FIRE: seven words carry no word 8-gram -- fewer than n words must be the empty set"
    )

    exactly_eight = guard.words(TEMPLATE)
    assert len(exactly_eight) == 8, "MUST_PASS: the template phrase is exactly eight words"
    assert guard.ngrams(exactly_eight) == {tuple(exactly_eight)}, (
        "MUST_PASS: eight words is exactly one 8-gram, the whole phrase as one tuple"
    )

    long_tokens = guard.words(EVAL_SENTENCE)
    long_grams = guard.ngrams(long_tokens)
    assert len(long_tokens) == 14 and len(long_grams) == 7, (
        "MUST_FIRE: fourteen words is seven sliding 8-grams -- not one whole-sentence match and "
        "not fourteen"
    )
    assert long_grams == {tuple(long_tokens[i : i + 8]) for i in range(7)}, (
        "MUST_PASS: the 8-grams are the sliding word windows of the normalized text"
    )

    assert guard.eval_ngrams(["one two three", TEMPLATE]) == {tuple(exactly_eight)}, (
        "MUST_PASS: eval_ngrams is the union over every reference, and short references contribute "
        "nothing"
    )
    assert guard.eval_ngrams([]) == set(), (
        "MUST_FIRE: no references is no 8-grams -- an empty evaluation set matches nothing"
    )

    # MUST_FIRE: an n-gram order below 1 is refused loudly, never silently widened
    with pytest.raises(ValueError, match="n must be >= 1"):
        guard.ngrams(["a", "b"], 0)


def test_overlap_drops_the_whole_recording_and_keeps_the_unrelated() -> None:
    """MUST_PASS: a clip copying an evaluation sentence is strong and its WHOLE recording goes.

    MUST_FIRE: the clean sibling of that recording is dropped too (its neighbours are the same
    recording), while a template 8-gram shared by three candidates is too common to make anything
    strong and still drops those clips at clip level, and a one-word teacher text is never kept.
    """
    candidates = [
        {"id": "copy-1", "rec": "rec-copy", "teacher": EVAL_SENTENCE},
        {
            "id": "copy-2",
            "rec": "rec-copy",
            "teacher": "a clean sibling clip of the same recording here",
        },
        {"id": "tmpl-1", "rec": "rec-t1", "teacher": TEMPLATE},
        {"id": "tmpl-2", "rec": "rec-t2", "teacher": TEMPLATE},
        {"id": "tmpl-3", "rec": "rec-t3", "teacher": TEMPLATE},
        {
            "id": "clean-1",
            "rec": "rec-clean",
            "teacher": "an unrelated clip about weather and traffic",
        },
        {"id": "short-1", "rec": "rec-short", "teacher": "hello"},
    ]
    eval_grams = guard.eval_ngrams([EVAL_SENTENCE, TEMPLATE])
    report = guard.overlap(candidates, eval_grams)

    assert "copy-1" in report["strong"], (
        "MUST_PASS: a clip copying an evaluation sentence shares seven 8-grams and is strong"
    )
    assert report["bad_recs"] == {"rec-copy"}, (
        "MUST_FIRE: the strong clip refuses its WHOLE recording -- one source, one refusal"
    )
    assert "copy-2" not in report["keep"] and "copy-2" not in report["clip_shared"], (
        "MUST_FIRE: the clean sibling is dropped by its recording and not by its own text -- it "
        "shares no 8-gram and would have been kept if the recording were not refused whole"
    )

    template_ids = {"tmpl-1", "tmpl-2", "tmpl-3"}
    assert report["strong"].isdisjoint(template_ids), (
        "MUST_FIRE: a template 8-gram shared by three candidates has freq 3 > rare_max 2 and can "
        "never make a clip strong"
    )
    assert template_ids <= report["clip_shared"], (
        "MUST_PASS: the template clips do share an 8-gram and are counted as sharing it"
    )
    assert template_ids.isdisjoint(set(report["keep"])), (
        "MUST_FIRE: too common to be strong is not too common to drop -- the shared 8-gram still "
        "drops those clips at clip level"
    )
    assert report["freq"][tuple(guard.words(TEMPLATE))] == 3, (
        "MUST_FIRE: freq counts the candidates sharing the n-gram -- three template clips is 3, "
        "not 1 and not 13"
    )

    assert report["keep"] == ["clean-1"], (
        "MUST_PASS: the unrelated clip is the only survivor, in candidate order"
    )
    assert "short-1" not in report["keep"], (
        "MUST_FIRE: a one-word teacher text is not a pseudo-label and is not kept"
    )


def test_agree_keeps_identical_and_near_identical_text_only() -> None:
    """MUST_PASS: identical text agrees at CER 0.0 and one changed character in a long text passes.

    MUST_FIRE: a different sentence fails the ceiling, and an empty teacher is DISAGREEMENT at CER
    1.0 -- the rate is undefined over zero reference characters and must never read as a pass.
    """
    long_text = EVAL_SENTENCE
    assert guard.agree(long_text, long_text) == (True, 0.0), (
        "MUST_PASS: identical normalized text is (True, 0.0) -- the one case where 0.0 is measured"
    )

    one_character = long_text.replace("fox", "fix")
    passed, cer = guard.agree(long_text, one_character, max_cer=0.03)
    assert passed and 0.0 < cer <= 0.03, (
        "MUST_PASS: one changed character in 56 measured characters is about 0.018 and inside the "
        "0.03 ceiling"
    )

    other_passed, other_cer = guard.agree(
        long_text, "a completely different sentence about traffic"
    )
    assert not other_passed and other_cer > 0.03, (
        "MUST_FIRE: a different sentence is disagreement -- the ceiling is 0.03 and not a courtesy"
    )

    empty_passed, empty_cer = guard.agree("", long_text)
    assert (empty_passed, empty_cer) == (False, 1.0), (
        "MUST_FIRE: an empty normalized teacher is disagreement at cer 1.0, never a 0.0 that reads "
        "as perfection over zero evidence"
    )


def test_verdict_ok_and_each_red_flag_reason_alone() -> None:
    """MUST_PASS: a pool inside every arm publishes OK with no reasons.

    MUST_FIRE: each arm alone publishes RED_FLAG with EXACTLY ONE reason -- a recording that is
    mostly evaluation material, a strong-clip share above the ceiling, and a pool too small to
    train on are three different refusals and must not blur into one.
    """
    ok_label, ok_reasons = guard.verdict(_report())
    assert (ok_label, ok_reasons) == ("OK", []), (
        "MUST_PASS: 1/10 recordings, 0 strong clips and 60 kept hours is OK with no reasons"
    )

    rec_label, rec_reasons = guard.verdict(_report(recordings_dropped=4))
    assert rec_label == "RED_FLAG" and len(rec_reasons) == 1, (
        "MUST_FIRE: 4/10 recordings dropped is above max_rec_drop 0.30 and is one reason on its own"
    )
    assert "recordings dropped" in rec_reasons[0], (
        "MUST_FIRE: the recording arm names the share it refused (4/10 = 0.4000)"
    )

    strong_label, strong_reasons = guard.verdict(_report(strong_clips=2))
    assert strong_label == "RED_FLAG" and len(strong_reasons) == 1, (
        "MUST_FIRE: 2/100 strong clips is above max_strong 0.01 and is one reason on its own"
    )
    assert "strong clips" in strong_reasons[0], (
        "MUST_FIRE: the strong-clip arm names the share it refused (2/100 = 0.0200)"
    )

    hours_label, hours_reasons = guard.verdict(_report(kept_hours=12.0))
    assert hours_label == "RED_FLAG" and len(hours_reasons) == 1, (
        "MUST_FIRE: 12 kept hours is below min_hours 50 and is one reason on its own"
    )
    assert "kept hours" in hours_reasons[0], (
        "MUST_FIRE: the hours arm names the floor it refused (12.0 below 50.0)"
    )


def test_cli_clean_campaign_exits_zero_and_writes_pseudo_and_report(tmp_path: Path) -> None:
    """MUST_PASS: a clean pool writes its pseudo rows and an OK report and exits 0.

    MUST_FIRE: every clip is accounted for -- 6 kept, 0 dropped in any lane -- because a report
    whose drops do not add up is a pool nobody can audit.
    """
    rows = _clips()
    candidates = tmp_path / "cands.jsonl"
    teacher = tmp_path / "teacher_eval.json"
    ref_answer = tmp_path / "refs_answer.jsonl"
    ref_text = tmp_path / "refs_text.jsonl"
    out = tmp_path / "pseudo.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(candidates, rows)
    _write_eval(teacher, _teacher(rows))
    _write_jsonl(ref_answer, [{"answer": EVAL_SENTENCE}])
    _write_jsonl(ref_text, [{"text": TEMPLATE}])

    code = guard.main(_argv(candidates, teacher, [ref_answer, ref_text], out, report_path))

    assert code == 0, "MUST_PASS: a clean campaign exits 0 -- the pool may be used"
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 6 and all("answer" in row for row in written), (
        "MUST_PASS: every kept row is the candidate row with the teacher text as 'answer'"
    )
    assert all("cer_agree" not in row for row in written), (
        "MUST_FIRE: no student decode was given, so no cer_agree is invented for the rows"
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "OK" and report["reasons"] == [], (
        "MUST_PASS: 0/3 recordings dropped, 0 strong clips and 60.0 kept hours is an OK verdict"
    )
    assert (
        report["kept_rows"],
        report["kept_hours"],
        report["clip_8gram_drops"],
        report["short_drops"],
        report["disagree_drops"],
        report["recording_clip_drops"],
    ) == (6, 60.0, 0, 0, 0, 0), (
        "MUST_FIRE: the ledger closes -- 6 kept + 0 dropped is the 6 candidates handed in"
    )
    assert report["top_shared_ngrams"] == [] and report["recordings"] == 3, (
        "MUST_FIRE: a clean pool shares no n-gram at all and still names its three recordings"
    )


def test_cli_overlapping_source_exits_5_and_drops_the_whole_recording(tmp_path: Path) -> None:
    """MUST_PASS: a source overlapping an evaluation set exits 5; its pseudo file is still written.

    MUST_FIRE: the WHOLE recording leaves the pool -- the overlapping clip and its clean sibling
    both -- so the same broadcast cannot re-enter through the segment that fell outside an 8-gram.
    """
    rows = _clips()
    candidates = tmp_path / "cands.jsonl"
    teacher = tmp_path / "teacher_eval.json"
    refs = tmp_path / "refs.jsonl"
    out = tmp_path / "pseudo.jsonl"
    report_path = tmp_path / "report.json"
    hypotheses = _teacher(rows)
    hypotheses["c1"] = EVAL_SENTENCE
    _write_jsonl(candidates, rows)
    _write_eval(teacher, hypotheses)
    _write_jsonl(refs, [{"answer": EVAL_SENTENCE}, {"text": TEMPLATE}])

    code = guard.main(_argv(candidates, teacher, [refs], out, report_path))

    assert code == 5, "MUST_FIRE: an overlapping source is a RED_FLAG and exits 5, not 0"
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    kept_ids = {row["id"] for row in written}
    assert out.exists() and len(written) == 4, (
        "MUST_PASS: the pseudo file is still written on a RED_FLAG so the flagged pool "
        "can be inspected"
    )
    assert kept_ids == {"c3", "c4", "c5", "c6"}, (
        "MUST_FIRE: rec-a is refused WHOLE -- the overlapping c1 and its clean sibling c2 are both "
        "gone, and only the untouched recordings survive"
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "RED_FLAG" and report["recordings_dropped"] == 1, (
        "MUST_FIRE: one of three recordings is dropped and the verdict says RED_FLAG"
    )
    assert report["recordings"] == 3 and report["recording_clip_drops"] == 2, (
        "MUST_FIRE: the whole source costs 2 clips, not just the 1 clip that matched"
    )
    assert any("recordings dropped" in reason for reason in report["reasons"]), (
        "MUST_FIRE: the reason names the recording drop share (1/3 = 0.3333 above 0.30)"
    )


def test_cli_refuses_a_candidate_without_a_teacher_hypothesis(tmp_path: Path) -> None:
    """MUST_PASS: an undecoded candidate refuses the whole run before anything is written.

    MUST_FIRE: the refusal COUNTS the missing ids -- a pseudo-label IS the teacher's transcript, so
    a clip the teacher never decoded is a refusal and not a row to fill in later.
    """
    rows = _clips()
    candidates = tmp_path / "cands.jsonl"
    teacher = tmp_path / "teacher_eval.json"
    refs = tmp_path / "refs.jsonl"
    out = tmp_path / "pseudo.jsonl"
    report_path = tmp_path / "report.json"
    hypotheses = _teacher(rows)
    del hypotheses["c3"]
    _write_jsonl(candidates, rows)
    _write_eval(teacher, hypotheses)
    _write_jsonl(refs, [{"answer": EVAL_SENTENCE}])

    code = guard.main(_argv(candidates, teacher, [refs], out, report_path))

    assert code == 2, "MUST_FIRE: refused input exits 2 -- not 0 and not 5"
    assert not out.exists() and not report_path.exists(), (
        "MUST_PASS: a refused run writes neither pseudo rows nor a report -- nothing to mistake "
        "for a complete pool"
    )
