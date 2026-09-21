"""Tests for the measured-corpus converter.

Every test asserts against the coverage arithmetic (``inspected``/``dropped``)
as well as the payload, because the round-1 refusal tests were satisfiable by a
stub that never inspected anything and always refused. These are written so
that stub -- and a vacuous-success stub that always returns one row -- both
fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from foundationscale.families.corpus import (
    CorpusResult,
    CorpusRow,
    ThinkPolicy,
    convert_records,
)
from foundationscale.families.registry import FamilySpec


class _FamilyStub:
    """Duck-typed FamilySpec stand-in.

    corpus.py reads only ``.name`` off the family; the real spec's construction
    requirements are the registry's contract, not this module's, so pulling a
    real FamilySpec in here would couple these tests to fields the converter
    never touches.
    """

    def __init__(self, name: str) -> None:
        self.name = name


_FAMILY = cast(FamilySpec, _FamilyStub("qwen_test_family"))


def _turn(role: str, value: Any) -> dict[str, Any]:
    return {"from": role, "value": value}


def _good_conv() -> list[dict[str, Any]]:
    return [_turn("human", "What is 2+2?"), _turn("gpt", "It is 4.")]


def _record(record_id: Any, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"id": record_id, "conversations": _good_conv()}
    record.update(extra)
    return record


def _convert(records: list[dict[str, Any]], root: Path, **kwargs: Any) -> CorpusResult:
    kwargs.setdefault("think_policy", ThinkPolicy.KEEP)
    return convert_records(records, corpus_root=str(root), family=_FAMILY, **kwargs)


def _corpus_distinct_drops() -> list[dict[str, Any]]:
    """5 convertible records plus 3 unconvertible ones, each in its OWN bucket,
    listed last so first-seen order makes the expected ``dropped`` tuple exact.
    """
    records = [_record(i) for i in range(5)]
    records.append(_record("no-conv", conversations=None))
    records.append(
        _record("bad-role", conversations=[_turn("human", "hi"), _turn("wizard", "boo")])
    )
    records.append(
        _record(
            "bad-placeholder",
            conversations=[_turn("human", "<image> what is shown?"), _turn("gpt", "a cat")],
        )
    )
    return records


def _corpus_dominant_drops() -> list[dict[str, Any]]:
    """5 convertible plus 3 unconvertible with one clearly DOMINANT bucket, so
    the refusal text has an unambiguous top bucket to name.
    """
    records = [_record(i) for i in range(5)]
    records.append(_record("no-conv-1", conversations=None))
    records.append(_record("no-conv-2", conversations=None))
    records.append(
        _record(
            "bad-placeholder",
            conversations=[_turn("human", "<image> what is shown?"), _turn("gpt", "a cat")],
        )
    )
    return records


def test_all_three_value_shapes_convert(tmp_path: Path) -> None:
    """If deleted: the {"answer", "reasoning"} CoT dict shape can be refused
    again -- round 1's exact bug, which refused every CoT row in the corpus --
    and think-block KEEP/STRIP handling regresses unobserved."""
    plain = _record("plain")
    with_think = _record(
        "think",
        conversations=[
            _turn("human", "Why is the sky blue?"),
            _turn("gpt", "<think>rayleigh scattering</think>\nRayleigh scattering."),
        ],
    )
    cot_dict = _record(
        "cot",
        conversations=[
            _turn("human", "What is 2+2?"),
            _turn("gpt", {"answer": "4", "reasoning": "adding two and two"}),
        ],
    )
    kept = _convert([plain, with_think, cot_dict], tmp_path, think_policy=ThinkPolicy.KEEP)
    assert kept.refusal is None
    assert isinstance(kept, CorpusResult)
    assert all(isinstance(row, CorpusRow) for row in kept.rows)
    assert [row.record_id for row in kept.rows] == ["plain", "think", "cot"]
    assert kept.inspected == 3

    cot_text = kept.rows[2].text
    # The dict converts to a think block plus its ANSWER field -- never a repr.
    assert "adding two and two" in cot_text
    assert "<think>" in cot_text
    assert "\n4" in cot_text
    assert "{'answer'" not in cot_text

    think_text = kept.rows[1].text
    assert "rayleigh scattering" in think_text  # KEEP preserves the block

    stripped = _convert([with_think], tmp_path, think_policy=ThinkPolicy.STRIP)
    assert stripped.refusal is None
    assert "<think>" not in stripped.rows[0].text
    assert "rayleigh scattering" not in stripped.rows[0].text
    assert "Rayleigh scattering." in stripped.rows[0].text


def test_mixed_corpus_tallies_drops_without_refusing(tmp_path: Path) -> None:
    """If deleted: drop bookkeeping can regress to a bare count (or nothing),
    hiding WHICH key set was lost -- and a partial corpus stops being an
    announced, caller-visible choice."""
    corpus = _corpus_distinct_drops()
    result = _convert(corpus, tmp_path)
    assert result.refusal is None
    assert len(result.rows) == 5
    assert result.inspected == 8
    assert result.dropped == (
        ("missing/empty 'conversations'", 1),
        ("unmapped role", 1),
        ("placeholder/payload mismatch", 1),
    )


def test_exceeded_max_drop_fraction_refuses_and_names_buckets(tmp_path: Path) -> None:
    """If deleted: a 3/8-dropped corpus trains on the residue at rc 0 -- the
    exact vacuous-success shape the house rules exist to forbid."""
    corpus = _corpus_dominant_drops()
    result = _convert(corpus, tmp_path, max_drop_fraction=0.1)
    assert result.refusal is not None
    assert result.rows == ()
    assert "0.375" in result.refusal
    assert "missing/empty 'conversations'" in result.refusal
    assert result.inspected == len(corpus)
    assert result.dropped == (
        ("missing/empty 'conversations'", 2),
        ("placeholder/payload mismatch", 1),
    )


def test_zero_convertible_records_refuses_even_without_limit(tmp_path: Path) -> None:
    """If deleted: an all-dropped run can slide to GREEN by returning [] --
    a converter that inspected everything and kept nothing did not convert."""
    corpus = [_record("x", conversations=None), _record("y", conversations=None)]
    result = _convert(corpus, tmp_path, max_drop_fraction=None)
    assert result.refusal is not None
    assert result.rows == ()
    assert "zero rows" in result.refusal
    assert result.inspected == 2
    assert result.dropped == (("missing/empty 'conversations'", 2),)


def test_inspected_counts_input_records(tmp_path: Path) -> None:
    """If deleted: a never-inspect/always-refuse stub passes every refusal
    test again -- the specific hole in the round-1 suite this function
    exists to close, on success AND refusal AND empty-input paths."""
    good = _corpus_distinct_drops()
    success = _convert(good, tmp_path)
    assert success.refusal is None
    assert success.inspected == len(good)

    bad = _corpus_dominant_drops()
    refused = _convert(bad, tmp_path, max_drop_fraction=0.0)
    assert refused.refusal is not None
    assert refused.inspected == len(bad)

    empty = _convert([], tmp_path)
    assert empty.refusal is not None
    assert empty.inspected == 0


def test_existence_only_media_notice_names_the_count(tmp_path: Path) -> None:
    """If deleted: the 21.5%-of-existing-files-fail-decode gap goes back to
    being silently deferred to the loader, and an exists-only count can once
    again be read as 'loadable'."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    # Deliberately NOT valid images: this module announces EXISTENCE ONLY and
    # must import without PIL, so valid payloads would test the wrong thing.
    (image_dir / "a.bin").write_bytes(b"not a decodable image")
    (image_dir / "b.bin").write_bytes(b"also not decodable")
    record = _record(
        "media",
        image=["images/a.bin", "images/b.bin"],
        conversations=[
            _turn("human", "<image><image> which is brighter?"),
            _turn("gpt", "the first"),
        ],
    )
    result = _convert([record], tmp_path)
    assert result.refusal is None
    note = next((a for a in result.announcements if "loader's gate" in a), None)
    assert note is not None, "media notice absent -- the deferral is unannounced"
    assert "2/2" in note
    assert "NOT established" in note
    assert result.rows[0].image_paths == (
        str(image_dir / "a.bin"),
        str(image_dir / "b.bin"),
    )


def test_integer_id_survives_with_type_intact(tmp_path: Path) -> None:
    """If deleted: ids can be silently str()-coerced again, re-enabling the
    cross-key-set join collision that motivated finding 8."""
    result = _convert([_record(7)], tmp_path)
    assert result.refusal is None
    assert type(result.rows[0].record_id) is int
    assert result.rows[0].record_id == 7


def test_a_missing_media_path_lowers_the_existence_count(tmp_path: Path) -> None:
    """If deleted: nothing distinguishes a real existence check from a constant
    True. Measured against the suite as generated -- mutating
    ``os.path.exists(path)`` to ``True`` left all seven tests passing, because
    every fixture referenced files that were all present, so the count could
    only ever come back full."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "present.bin").write_bytes(b"exists but is not decodable")
    record = _record(
        "half-missing",
        image=["images/present.bin", "images/absent.bin"],
        conversations=[
            _turn("human", "<image><image> compare these"),
            _turn("gpt", "only one of them is here"),
        ],
    )
    result = _convert([record], tmp_path)

    # A missing referenced file is NOT a refusal: the loader owns that gate.
    # What this module owes is an honest count, and a full count would be a lie.
    assert result.refusal is None
    note = next((a for a in result.announcements if "loader's gate" in a), None)
    assert note is not None, "media notice absent -- the deferral is unannounced"
    assert "1/2" in note, f"existence count did not fall for a missing path: {note}"
    assert "2/2" not in note
    # The path is still resolved and handed on; resolution and existence are
    # separate facts and collapsing them is what the notice exists to prevent.
    assert result.rows[0].image_paths == (
        str(image_dir / "present.bin"),
        str(image_dir / "absent.bin"),
    )
