"""Dangling modality placeholders in text-only supervision (#506).

``_dropped_column_notice`` announces the columns the text-only arm discards,
which is what T1-22 asked for. A corpus that KEEPS a media column still trips it
-- that is T1-22's own undeclared control. It cannot see this defect, though:
the thin path REQUIRES a ``text`` column, and the conversion that satisfies that
requirement folds the media reference in, after which there is no column left to
name and the notice falls silent on the shape conversion produces.

Measured on the estate's 210-row assembly-video corpus: 210/210 rows retained a
``<video>`` marker, that checkpoint's own tokenizer split it into three ordinary
tokens with no special id, training ran 20 steps, the save gates passed and the
process exited 0 without one line naming the marker. The model-side twin already
speaks -- ``modality.dormant_towers`` reports the towers as carried and not
trained -- so the run disclosed that those towers went unexercised here while
staying silent about the data assuming otherwise.

The notice is NOT a refusal, and the negative controls below are the point:
folding the reference into ``text`` is legitimate, and a ``<video>`` can be
ordinary prose. A guard that refused those would refuse the conversion this
plane's own refusal text recommends ("Unset {var} to train text-only on the
same corpus").
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path

from foundationscale.train.loop import (
    _text_column_or_none,
    _unresolved_placeholder_notice,
)

_VIDEO_ROW = "Q: <video>\nWhich SOP step is the operator performing?\nA: Step 3."
_CLEAN_ROW = "Q: What year was the cave opened?\nA: 1974."


def test_every_row_dangling_is_named_with_its_count() -> None:
    """The shape actually measured on the estate corpus: 210/210 rows."""
    notice = _unresolved_placeholder_notice([_VIDEO_ROW] * 210, image_declared=False)

    assert notice is not None
    assert "<video> in 210/210 rows" in notice
    assert "video" in notice
    assert "#506" in notice
    # It must say what the silence costs, not merely that a marker was seen.
    assert "never received" in notice


def test_partial_contamination_reports_the_true_fraction() -> None:
    """A count, not a boolean: three dangling rows in eight is not "a corpus"."""
    rows = [_VIDEO_ROW] * 3 + [_CLEAN_ROW] * 5
    notice = _unresolved_placeholder_notice(rows, image_declared=False)

    assert notice is not None
    assert "<video> in 3/8 rows" in notice


def test_clean_corpus_is_silent() -> None:
    """NEGATIVE CONTROL. A notice that fires on clean text is noise, not a guard.

    If this ever fails, every text-only run in CI grows a line claiming its
    supervision references material it never saw -- which would be false, and
    would train readers to ignore the true one.
    """
    assert _unresolved_placeholder_notice([_CLEAN_ROW] * 40, image_declared=False) is None


def test_declared_image_resolves_its_own_placeholder() -> None:
    """NEGATIVE CONTROL for the declaration axis.

    ``<image>`` is dangling only while no image column is declared. When one is,
    the image arm loads pixels for it and the marker is resolved -- reporting it
    would contradict the arm standing right beside it. Video stays dangling in
    the same breath, because no declaration on this plane can resolve it.
    """
    rows = ["Q: <image>\nWhat is shown?\nA: A cave."] * 4

    assert _unresolved_placeholder_notice(rows, image_declared=True) is None

    undeclared = _unresolved_placeholder_notice(rows, image_declared=False)
    assert undeclared is not None
    assert "<image> in 4/4 rows" in undeclared

    both = _unresolved_placeholder_notice(["Q: <image> <video>\nA: x."] * 2, image_declared=True)
    assert both is not None
    assert "<video> in 2/2 rows" in both
    assert "<image>" not in both


def test_empty_corpus_does_not_divide_by_zero() -> None:
    assert _unresolved_placeholder_notice([], image_declared=False) is None


class _Unindexable:
    """A streaming split: map() and column_names, but no column access.

    Not a convenience fixture -- this is what ``IterableDataset`` does, and the
    thin path accepts one. The first cut of #506 indexed the column directly and
    took sixteen tests down with a RED exit before any of them reached a
    manifest, which is what a real streaming corpus would have done to a real
    run.
    """

    column_names = ("text",)

    def map(self, *a: object, **k: object) -> _Unindexable:
        return self


class _Lazy:
    """A split whose column access returns something that is not a sequence."""

    def __getitem__(self, key: str) -> object:
        return iter(["Q: <video>"])


def test_unindexable_split_reads_as_unmeasured_not_as_clean() -> None:
    """The distinction the whole repository is about: absent != clean."""
    assert _text_column_or_none(_Unindexable()) is None


def test_lazy_column_is_refused_rather_than_counted() -> None:
    """A generator has no len(); counting it would print a confident 0/0."""
    assert _text_column_or_none(_Lazy()) is None


def test_a_real_sequence_column_is_returned() -> None:
    """POSITIVE CONTROL: the two guards above must not reject the real shape."""
    rows = {"text": [_VIDEO_ROW, _CLEAN_ROW]}
    assert _text_column_or_none(rows) == [_VIDEO_ROW, _CLEAN_ROW]


class _ColumnLike(Sequence[str]):
    """What ``datasets`` actually hands back -- NOT a list.

    Measured against the real library on the estate corpus: ``Dataset["text"]``
    returns a ``datasets.Column``, a lazy view that registers as ``Sequence``
    and answers ``len()`` -- 210 rows, notice fired at 210/210. The dict-of-list
    fixture above is therefore KINDER than production, which is the shape of
    bug this repository keeps finding in other people's tests. This control
    exists so that narrowing the accessor's check to ``isinstance(column, list)``
    -- the obvious "tighten it up" edit -- fails here instead of silently
    turning every real run's scan into an UNMEASURED.
    """

    def __init__(self, rows: list[str]) -> None:
        self._rows = rows

    def __getitem__(self, index: int) -> str:  # type: ignore[override]
        return self._rows[index]

    def __len__(self) -> int:
        return len(self._rows)


def test_the_production_column_type_is_accepted() -> None:
    column = _ColumnLike([_VIDEO_ROW] * 3 + [_CLEAN_ROW])
    assert not isinstance(column, list)

    got = _text_column_or_none({"text": column})
    assert got is column, "a Sequence-registered lazy column is the production shape"

    notice = _unresolved_placeholder_notice(got, image_declared=False)
    assert notice is not None
    assert "<video> in 3/4 rows" in notice


def test_the_unmeasured_branch_announces() -> None:
    """An abstention nobody prints is the defect, not the cure.

    ``_text_column_or_none`` returning None is a real loss of coverage, so the
    call site owes the operator a line saying the scan did not happen. Pin the
    wording in the source: a silent ``else`` here would restore exactly the
    silence #506 exists to remove, and no unit test of the helper can see it.
    """
    source = Path(inspect.getsourcefile(_unresolved_placeholder_notice)).read_text()
    assert "_texts = _text_column_or_none(" in source
    unmeasured = source[source.index("if _texts is None:") :][:1200]
    assert "UNMEASURED" in unmeasured
    assert "_mark(" in unmeasured
    assert "#506" in unmeasured


def test_the_image_arm_scans_with_image_resolved() -> None:
    """Declaring images must not exempt the arm from the video disclosure.

    An omni corpus can carry pixels AND a folded '<video>'. The image arm passes
    image_declared=True, which resolves '<image>' (the collator loads pixels for
    it) while leaving video and audio reportable.
    """
    source = Path(inspect.getsourcefile(_unresolved_placeholder_notice)).read_text()
    assert "_unresolved_placeholder_notice(_img_texts, image_declared=True)" in source


def test_the_notice_is_wired_into_the_text_only_arm() -> None:
    """A helper nobody calls is a vacuous guard.

    Every assertion above exercises the function directly, so all of them stay
    green if the call site is deleted -- which is precisely how this defect
    survived: the true statement existed nowhere the operator could read it.
    Pin the wiring in the source, next to the sibling notice it completes.
    """
    source = Path(inspect.getsourcefile(_unresolved_placeholder_notice)).read_text()
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_unresolved_placeholder_notice" in called, (
        "the #506 notice is defined but never called; the text-only arm is silent again"
    )
    assert "_dropped_column_notice" in called
    # Anchor on the CALL, not the name: the definition sorts earlier than both
    # call sites, so matching the bare name here would compare the wrong pair
    # and pass while the wiring sat anywhere at all.
    assert source.index("_dropped_column_notice(columns)") < source.index(
        "_dangling = _unresolved_placeholder_notice("
    ), "the two notices must stay adjacent; the second explains the first's blind spot"
