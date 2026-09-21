"""Additional coverage tests for the measured-corpus converter.

Every test here pins one branch of corpus.py that the main conversion suite
(test_corpus_conversion.py) does not reach: the exact refusal text for each
unrecognisable ``conversations[].value`` shape, the media-field type guards,
the caller-error ValueError for an out-of-range ``max_drop_fraction``, the
whole-corpus refusal on a non-mapping record, and the named drop buckets for
unsupported ids and assistant-free records. Assertions are on refusal TEXT,
bucket names, and coverage arithmetic -- never on "it did not raise".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from foundationscale.families.corpus import (
    CorpusResult,
    ThinkPolicy,
    convert_records,
)
from foundationscale.families.registry import FamilySpec


class _FamilyStub:
    """Duck-typed FamilySpec stand-in.

    corpus.py reads only ``.name`` off the family; the real spec's construction
    requirements are the registry's contract, not this module's.
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


def _convert(records: list[Any], root: Path, **kwargs: Any) -> CorpusResult:
    kwargs.setdefault("think_policy", ThinkPolicy.KEEP)
    return convert_records(records, corpus_root=str(root), family=_FAMILY, **kwargs)


def test_assistant_dict_without_reasoning_returns_bare_candidate(tmp_path: Path) -> None:
    """Pins corpus.py line 130: an assistant dict with a usable answer key but
    NO reasoning field must yield the candidate itself, not a think block and
    not a refusal.

    If deleted: the no-reasoning branch can regress to refusing (dropping
    every non-CoT dict row) or to synthesising an empty ``<think></think>``
    block that supervises noise as reasoning.
    """
    record = _record(
        "bare-answer",
        conversations=[
            _turn("human", "What is 2+2?"),
            _turn("gpt", {"answer": "  the answer is four  "}),
        ],
    )
    result = _convert([record], tmp_path)
    assert result.refusal is None
    assert len(result.rows) == 1
    assert "the answer is four" in result.rows[0].text
    assert "<think>" not in result.rows[0].text
    assert "{'answer'" not in result.rows[0].text  # never the dict repr
    assert result.inspected == 1


def test_assistant_dict_with_no_usable_answer_key_refuses_by_name(tmp_path: Path) -> None:
    """Pins corpus.py lines 131-135 and the 384-385 exception boundary: an
    assistant dict carrying none of the _ANSWER_KEYS is refused with the
    OBSERVED keys named, and the refusal arrives as a string on CorpusResult.

    If deleted: the converter can guess a field as the answer (supervising a
    metadata blob as the reply) or raise past the public boundary, moving the
    refusal into a traceback the gate cannot quote.
    """
    record = _record(
        "mystery-dict",
        conversations=[
            _turn("human", "hello?"),
            _turn("gpt", {"summary": "hi", "notes": "there"}),
        ],
    )
    result = _convert([record], tmp_path)
    assert result.rows == ()
    assert isinstance(result.refusal, str)
    assert "no usable answer key" in result.refusal
    assert "observed keys: ['notes', 'summary']" in result.refusal
    assert result.inspected == 1


def test_non_assistant_role_dict_value_refuses_and_names_the_role(tmp_path: Path) -> None:
    """Pins corpus.py lines 136-140: a dict ``value`` on a NON-assistant turn
    is refused with the mapped role named, because coercing it would re-encode
    structure into repr-shaped training text.

    If deleted: a user-side list-of-parts dict once again renders as the
    Python repr ``[{'type': 'text', ...}]`` and trains through unnoticed --
    the exact silent-repr defect this refusal exists against.
    """
    record = _record(
        "user-dict",
        conversations=[
            _turn("human", {"text": "hello"}),
            _turn("gpt", "hi"),
        ],
    )
    result = _convert([record], tmp_path)
    assert result.rows == ()
    assert isinstance(result.refusal, str)
    assert "user 'value' is a dict ['text']" in result.refusal
    assert "only assistant" in result.refusal


def test_unrecognised_value_type_refuses_and_names_the_type(tmp_path: Path) -> None:
    """Pins corpus.py lines 141-146: a ``value`` that is neither str nor dict
    is refused with the observed type name, never str()-repr'd into the text
    column.

    If deleted: ``str(3.5)`` lands in the 'text' column as silent training
    text, re-creating the measured repr-leak this branch was written to stop.
    """
    record = _record(
        "float-value",
        conversations=[
            _turn("human", 3.5),
            _turn("gpt", "a number?"),
        ],
    )
    result = _convert([record], tmp_path)
    assert result.rows == ()
    assert isinstance(result.refusal, str)
    assert "has unrecognised type 'float'" in result.refusal


def test_single_string_media_value_resolves_as_one_path(tmp_path: Path) -> None:
    """Pins corpus.py line 176: a bare STRING media field is treated as one
    path, not iterated character by character.

    If deleted: a plain ``"images/frame.bin"`` value can be re-split into
    characters (Sequence iteration) and each character 'resolved' -- an
    honest-looking row whose image paths are pure garbage.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "frame.bin").write_bytes(b"exists but is not decodable")
    record = _record(
        "one-image",
        image="images/frame.bin",
        conversations=[
            _turn("human", "<image> what is shown?"),
            _turn("gpt", "a frame"),
        ],
    )
    result = _convert([record], tmp_path)
    assert result.refusal is None
    assert result.rows[0].image_paths == (str(image_dir / "frame.bin"),)
    note = next((a for a in result.announcements if "loader's gate" in a), None)
    assert note is not None, "media notice absent -- the existence count is unannounced"
    assert "1/1" in note


def test_scalar_media_field_refuses_with_the_type_named(tmp_path: Path) -> None:
    """Pins corpus.py lines 189-192: a media field that is neither str, bytes,
    nor a Sequence of paths is refused with its observed type.

    If deleted: an int media field falls into a generic per-entry failure that
    names the wrong problem (or crashes the conversion instead of refusing).
    """
    record = _record("int-image", image=42)
    result = _convert([record], tmp_path)
    assert result.rows == ()
    assert isinstance(result.refusal, str)
    assert "field 'image' has type 'int'" in result.refusal
    assert "expected a path string or a list of path strings" in result.refusal


def test_non_path_entry_inside_a_media_list_refuses(tmp_path: Path) -> None:
    """Pins corpus.py line 197: a list-shaped media field whose ENTRY is not a
    usable path string is refused, naming the offending entry.

    If deleted: a ``7`` or ``""`` inside the path list is either resolved into
    a nonsense path (silently corrupt media payload) or reported as a
    field-level type error when the real problem is one entry.
    """
    record = _record("bad-entry", image=["chart.png", 7])
    result = _convert([record], tmp_path)
    assert result.rows == ()
    assert isinstance(result.refusal, str)
    assert "field 'image' contains non-path entry 7" in result.refusal


def test_out_of_range_max_drop_fraction_raises_caller_error(tmp_path: Path) -> None:
    """Pins corpus.py line 266: a ``max_drop_fraction`` outside [0.0, 1.0] is
    a caller WIRING bug, raised as ValueError -- it must never surface as a
    data refusal the gate prints beside rc 96.

    If deleted: ``max_drop_fraction=1.5`` passes silently and the drop
    arithmetic below compares against a nonsense bound.
    """
    with pytest.raises(ValueError, match="max_drop_fraction must lie in") as excinfo:
        _convert([], tmp_path, max_drop_fraction=1.5)
    assert "got 1.5" in str(excinfo.value)


def test_non_mapping_record_refuses_the_whole_corpus(tmp_path: Path) -> None:
    """Pins corpus.py lines 300-304 and 384-385: a non-mapping ITEM in the
    record stream refuses the corpus as a whole (rows == ()), with the item's
    index and type named, rather than coercing a partial stream.

    If deleted: a partially-coerced stream that looks complete slides out at
    rc 0 -- strictly worse than no output, per the refusal's own text.
    """
    records: list[Any] = [_record(0), 42]
    result = _convert(records, tmp_path)
    assert result.rows == ()
    assert isinstance(result.refusal, str)
    assert "item 1 is 'int', not a mapping" in result.refusal
    assert "refusing the corpus as a whole" in result.refusal
    assert result.inspected == 2


def test_unsupported_record_id_type_drops_into_a_named_bucket(tmp_path: Path) -> None:
    """Pins corpus.py lines 308 and 314: a bool id is NOT str()-coerced; the
    record is dropped with bucket 'unsupported record id type' and conversion
    of the other records still proceeds.

    If deleted: round 1's silent str() retype returns, retyping IDs and
    inviting collisions once distinct key sets are joined on id -- the exact
    motivation for keeping the corpus's OWN id type.
    """
    records: list[Any] = [_record("good"), _record(True)]
    result = _convert(records, tmp_path)
    assert result.refusal is None
    assert len(result.rows) == 1
    assert result.rows[0].record_id == "good"
    assert result.inspected == 2
    assert result.dropped == (("unsupported record id type", 1),)
    assert any("id is 'bool'; expected str or int" in a for a in result.announcements)


@pytest.mark.parametrize("system_key", ["system", "system_prompt", "system_promt"])
def test_system_prompt_variant_prepends_a_system_message(tmp_path: Path, system_key: str) -> None:
    """Pins corpus.py line 342 across all three measured key spellings
    (including the estate's ``system_promt`` typo): a non-empty system string
    becomes a stripped ``system`` message at the head of the rendered text.

    If deleted: any one spelling silently loses the system prompt, and the
    model trains on conversations missing their behavioural directive.
    """
    record = _record("sys", **{system_key: "  Be terse.  "})
    result = _convert([record], tmp_path)
    assert result.refusal is None
    assert len(result.rows) == 1
    assert "<|im_start|>system\nBe terse.<|im_end|>" in result.rows[0].text


def test_record_without_assistant_turn_drops_into_a_named_bucket(tmp_path: Path) -> None:
    """Pins corpus.py lines 371 and 375: a record whose messages contain no
    assistant turn is dropped with bucket 'no assistant turn' -- there is
    nothing to supervise.

    If deleted: assistant-free records can produce supervised rows whose only
    plausible target is the USER turn, silently inverting the training signal.
    """
    records: list[Any] = [
        _record("good"),
        _record("quiet", conversations=[_turn("human", "anyone there?")]),
    ]
    result = _convert(records, tmp_path)
    assert result.refusal is None
    assert len(result.rows) == 1
    assert result.inspected == 2
    assert result.dropped == (("no assistant turn", 1),)
    assert any("nothing to supervise" in a for a in result.announcements)
