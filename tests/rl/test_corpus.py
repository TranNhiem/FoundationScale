"""Tests for the ShareGPT corpus loader.

WHAT IS CLAIMED: well-shaped .json and .jsonl corpora -- single files or
directories holding both -- load into Samples with the declared roles,
responses, images and video; MCQ gold is returned only when a single
unambiguous letter survives think-block stripping; structural violations
refuse with messages naming the failing record, turn, line or suffix.

WHAT IS NOT CLAIMED: no tokenisation or media decoding is exercised;
the abstention of prose answers is asserted as declared behaviour, not
treated as a defect to be repaired.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundationscale.rl.corpus import Sample, extract_mcq_gold, load_sharegpt
from foundationscale.rl.interfaces import BatchRefusal

_MCQ_QUESTION = "Which planet is red?\nA. Venus\nB. Mars\nAnswer with a single letter."


def _record(
    sample_id: str = "rec-1",
    human: str = "Hello there.",
    gpt: str = "General Kenobi.",
    **extras: object,
) -> dict[str, object]:
    base: dict[str, object] = {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": gpt},
        ],
    }
    base.update(extras)
    return base


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _write_json(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_jsonl_file_loads_into_samples_with_role_mapping(tmp_path: Path) -> None:
    corpus = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [_record(sample_id="a"), _record(sample_id="b", gpt="Second reply.")],
    )

    samples = load_sharegpt(corpus)

    assert len(samples) == 2
    assert isinstance(samples[0], Sample)
    assert samples[0].sample_id == "a"
    assert samples[0].prompt_turns == (("user", "Hello there."),)
    assert samples[0].response == "General Kenobi."
    assert samples[1].response == "Second reply."


def test_json_array_file_loads_with_same_sample_shape(tmp_path: Path) -> None:
    corpus = _write_json(tmp_path / "corpus.json", [_record(sample_id="a")])

    (sample,) = load_sharegpt(corpus)

    assert sample.sample_id == "a"
    assert sample.prompt_turns == (("user", "Hello there."),)
    assert sample.response == "General Kenobi."


def test_directory_loads_records_from_both_suffixes(tmp_path: Path) -> None:
    _write_json(tmp_path / "part.json", [_record(sample_id="from-json")])
    _write_jsonl(tmp_path / "part.jsonl", [_record(sample_id="from-jsonl")])

    samples = load_sharegpt(tmp_path)

    assert {s.sample_id for s in samples} == {"from-json", "from-jsonl"}


def test_directory_without_supported_suffixes_refuses(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not a corpus", encoding="utf-8")

    with pytest.raises(BatchRefusal, match=r"\.json or \.jsonl"):
        load_sharegpt(tmp_path)


def test_blank_lines_in_jsonl_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "sparse.jsonl"
    path.write_text(
        json.dumps(_record(sample_id="a")) + "\n\n   \n" + json.dumps(_record("b")) + "\n",
        encoding="utf-8",
    )

    samples = load_sharegpt(path)

    assert [s.sample_id for s in samples] == ["a", "b"]


def test_malformed_jsonl_line_refuses_naming_line_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(_record()) + "\n{not-json\n" + json.dumps(_record("c")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BatchRefusal, match="line 2"):
        load_sharegpt(path)


def test_json_top_level_object_refuses(tmp_path: Path) -> None:
    path = tmp_path / "object.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")

    with pytest.raises(BatchRefusal, match="top-level dict"):
        load_sharegpt(path)


def test_missing_path_refuses(tmp_path: Path) -> None:
    with pytest.raises(BatchRefusal, match="does not exist"):
        load_sharegpt(tmp_path / "absent.jsonl")


def test_empty_corpus_refuses_as_vacuous(tmp_path: Path) -> None:
    corpus = _write_jsonl(tmp_path / "empty.jsonl", [])

    with pytest.raises(BatchRefusal, match="yielded 0"):
        load_sharegpt(corpus)


def test_limit_caps_samples_and_zero_limit_refuses(tmp_path: Path) -> None:
    corpus = _write_jsonl(tmp_path / "many.jsonl", [_record(sample_id=str(i)) for i in range(5)])

    capped = load_sharegpt(corpus, limit=2)
    assert [s.sample_id for s in capped] == ["0", "1"]

    with pytest.raises(BatchRefusal, match="below 1"):
        load_sharegpt(corpus, limit=0)


def test_mcq_gold_accepts_bare_letter_and_strips_think_block() -> None:
    assert extract_mcq_gold(_MCQ_QUESTION, "B") == "B"
    # Reasoning rehearses discarded candidates; only the surviving letter counts.
    assert extract_mcq_gold(_MCQ_QUESTION, "<think>maybe A or C</think>B") == "B"


def test_mcq_gold_abstains_on_prose_answer_with_many_letters() -> None:
    assert extract_mcq_gold(_MCQ_QUESTION, "The answer is B.") is None


def test_mcq_gold_abstains_when_question_lacks_marker() -> None:
    # A bare letter under a non-MCQ question has no verifiable target.
    assert extract_mcq_gold("What is your favourite letter?", "B") is None


def test_record_ending_on_human_turn_refuses(tmp_path: Path) -> None:
    record = {
        "id": "tail-human",
        "conversations": [
            {"from": "gpt", "value": "Speaking first."},
            {"from": "human", "value": "Replying last."},
        ],
    }
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [record])

    with pytest.raises(BatchRefusal, match="FINAL turn must"):
        load_sharegpt(corpus)


def test_record_with_unknown_speaker_refuses_naming_speaker(tmp_path: Path) -> None:
    record = {
        "id": "odd-speaker",
        "conversations": [
            {"from": "robot", "value": "Beep."},
            {"from": "gpt", "value": "Boop."},
        ],
    }
    corpus = _write_jsonl(tmp_path / "corpus.jsonl", [record])

    with pytest.raises(BatchRefusal, match="'robot'"):
        load_sharegpt(corpus)


def test_system_key_becomes_first_prompt_turn(tmp_path: Path) -> None:
    corpus = _write_jsonl(
        tmp_path / "corpus.jsonl",
        [_record(system="You are concise.")],
    )

    (sample,) = load_sharegpt(corpus)

    assert sample.prompt_turns[0] == ("system", "You are concise.")
    assert sample.prompt_turns[1][0] == "user"


def test_image_str_and_list_populate_images_and_bad_video_refuses(tmp_path: Path) -> None:
    corpus = _write_jsonl(
        tmp_path / "media.jsonl",
        [
            _record(sample_id="single", image="a.png"),
            _record(sample_id="multi", image=["a.png", "b.png"]),
        ],
    )

    single, multi = load_sharegpt(corpus)
    assert single.images == ("a.png",)
    assert multi.images == ("a.png", "b.png")

    bad = _write_jsonl(tmp_path / "bad-video.jsonl", [_record(video=12)])
    with pytest.raises(BatchRefusal, match="non-string video key"):
        load_sharegpt(bad)
