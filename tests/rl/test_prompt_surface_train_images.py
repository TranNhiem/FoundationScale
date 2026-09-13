"""The TRAIN plane's pixel carrier: the collator and the drop probe (#410).

WHY THIS MODULE EXISTS. Finding #410: the train plane could not carry a
pixel. encode_prompts serves the GENERATE path, but training turns dataset
rows into the batch the forward actually receives through a COLLATOR, and
that collator had no image path at all -- and nothing checked that the batch
coming out of a collator still contained the pixel column the dataset
declared. A collator that drops pixel_values trains text-only under a
multimodal label with every emitted signal healthy: #371's silent drop, one
layer up.

The two functions that close it shipped with ZERO coverage, and this
module's floor is 100%: it was RED at 79.4%, missing lines 295-297 and
316-348 -- both bodies entire.

WHY EACH LEG IS SHAPED THIS WAY. Every leg is written to FAIL on the broken
shape, not to touch lines. The stub processor emits pixel_values ONLY when
it is handed a non-empty images list, and records how many images it
actually received -- a stub that always emits the key would pass on a
collator that drops the image, which is exactly the defect under test. The
loader is monkeypatched to a sentinel everywhere EXCEPT the missing-path
leg, which uses a real nonexistent path: that refusal fires on the
Path.exists() check BEFORE PIL is ever imported, so it is genuinely
exercised on a torch-free, PIL-free host.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.rl import prompt_surface
from foundationscale.rl.prompt_surface import (
    PIXEL_KEY,
    PromptSurface,
    refuse_if_pixel_column_dropped,
    train_image_collator_or_refuse,
)


class _FakeIds:
    """The smallest object that satisfies the collator's labels injection.

    The collator does ``batch["labels"] = batch["input_ids"].clone()``, so
    input_ids needs ``.clone()`` and nothing else. This -- not a torch stub
    -- is what keeps the module runnable on the torch-free CI legs.
    """

    def __init__(self, rows: list[str]) -> None:
        self.rows = list(rows)
        self.clone_calls = 0

    def clone(self) -> _FakeIds:
        self.clone_calls += 1
        return _FakeIds(self.rows)


class _SentinelImage:
    # What the monkeypatched loader returns: proof that a path was loaded,
    # carrying the path so load ORDER can be asserted, not just load count.
    def __init__(self, path: str) -> None:
        self.path = path


class _RecordingProcessor:
    """Processor-shaped stub with the falsifiability contract built in.

    pixel_values is emitted ONLY when __call__ is handed a non-empty images
    list. A stub that always emits the key makes every positive leg pass on
    a collator that drops the image -- the exact defect under test -- so the
    emission is conditional here and the received count is recorded for the
    legs to assert on.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.images_received: list[int] = []
        self.include_labels = False
        self.labels_sentinel = _FakeIds(["gold"])

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        images = kwargs.get("images") or []
        self.images_received.append(len(images))
        batch: dict[str, Any] = {"input_ids": _FakeIds(kwargs["text"])}
        if images:
            batch["pixel_values"] = ["px"] * len(images)
        if self.include_labels:
            batch["labels"] = self.labels_sentinel
        return batch


@pytest.fixture
def processor() -> tuple[PromptSurface, _RecordingProcessor]:
    rec = _RecordingProcessor()
    surface = PromptSurface(
        kind="processor", surface=rec, reason="test fixture", supports_images=True
    )
    return surface, rec


@pytest.fixture
def image_loader(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Swap _load_image_or_refuse for a recorder that returns sentinels.

    Keeps PIL out of the process for the happy-path legs; the missing-path
    leg below uses the REAL loader, which refuses before PIL is imported.
    """
    loaded: list[tuple[str, str]] = []

    def fake_load(sample_id: str, path: str) -> _SentinelImage:
        loaded.append((sample_id, path))
        return _SentinelImage(path)

    monkeypatch.setattr(prompt_surface, "_load_image_or_refuse", fake_load)
    return loaded


def test_a_batch_that_keeps_the_pixel_column_is_not_refused() -> None:
    # The complementary leg: without it the probe could refuse EVERY batch
    # and still pass -- the vacuous-gate shape this repo keeps finding.
    result = refuse_if_pixel_column_dropped(
        {"input_ids": [1], "attention_mask": [1], PIXEL_KEY: ["px"]},
        image_column="image",
    )
    assert result is None


def test_a_batch_that_dropped_the_pixel_column_is_refused_and_names_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: a collator whose output has text keys only. The refusal
    # must name BOTH the declared image column and the dropped pixel column,
    # or the operator cannot tell which axis vanished.
    with pytest.raises(SystemExit) as excinfo:
        refuse_if_pixel_column_dropped(
            {"input_ids": [1], "attention_mask": [1]},
            image_column="image",
        )
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "'image'" in err
    assert PIXEL_KEY in err
    assert "DROPPED" in err


def test_non_string_keys_are_normalised_before_the_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: batch keys that are not all strings. The refusal message
    # does sorted(keys); without the str() normalisation that sort raises
    # TypeError on mixed types and the probe crashes INSTEAD of refusing --
    # the drop it exists to report would be masked by its own report.
    mixed_present = {1: "a", PIXEL_KEY: "b"}
    assert refuse_if_pixel_column_dropped(mixed_present.keys(), image_column="image") is None
    with pytest.raises(SystemExit) as excinfo:
        refuse_if_pixel_column_dropped({1: "a", "input_ids": "b"}.keys(), image_column="image")
    assert excinfo.value.code == 96
    assert PIXEL_KEY in capsys.readouterr().err


def test_the_pixel_key_override_is_honoured_in_both_directions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An override that is accepted but ignored would refuse (or pass) against
    # the wrong spelling, so both directions are pinned: the custom key
    # satisfies the probe, and the DEFAULT key no longer does.
    result = refuse_if_pixel_column_dropped(
        {"pixels": [1]}, image_column="image", pixel_key="pixels"
    )
    assert result is None
    with pytest.raises(SystemExit) as excinfo:
        refuse_if_pixel_column_dropped({PIXEL_KEY: [1]}, image_column="image", pixel_key="pixels")
    assert excinfo.value.code == 96
    assert "'pixels'" in capsys.readouterr().err


def test_a_tokenizer_surface_is_refused_before_any_collator_is_built(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: a train-time image collator requested on a text-only
    # surface. The refusal fires at CONSTRUCTION, before any row is seen --
    # encoding images through a tokenizer is the silent drop one layer up.
    surface = PromptSurface(
        kind="tokenizer", surface=object(), reason="test fixture", supports_images=False
    )
    with pytest.raises(SystemExit) as excinfo:
        train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    assert excinfo.value.code == 96
    assert "'tokenizer'" in capsys.readouterr().err


def test_a_processor_without_image_support_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: kind says processor but the surface cannot take images.
    # The two surface refusals differ ONLY in the kind they name, so each
    # leg asserts its own kind string -- one cannot pass the other's test.
    surface = PromptSurface(
        kind="processor", surface=object(), reason="test fixture", supports_images=False
    )
    with pytest.raises(SystemExit) as excinfo:
        train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    assert excinfo.value.code == 96
    assert "'processor'" in capsys.readouterr().err


def test_two_rows_with_one_image_each_forward_exactly_two_images(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # THE LOAD-BEARING HAPPY PATH. If the collator dropped the image cells,
    # the stub would receive ZERO images and emit NO pixel_values, and both
    # assertions below would fail -- this leg cannot pass on the #410 shape.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate(
        [
            {"text": "first", "image": ["/corpus/a.png"]},
            {"text": "second", "image": ["/corpus/b.png"]},
        ]
    )
    assert image_loader == [("train-row[0]", "/corpus/a.png"), ("train-row[1]", "/corpus/b.png")]
    assert rec.images_received == [2]
    assert len(batch["pixel_values"]) == 2


def test_a_bare_string_image_cell_is_coerced_to_one_image(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # A corpus row whose image cell is a plain string, not a list. Iterating
    # the string instead of wrapping it would load one "image" PER CHARACTER
    # -- the coercion is what this leg pins.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "only", "image": "/corpus/only.png"}])
    assert image_loader == [("train-row[0]", "/corpus/only.png")]
    assert rec.images_received == [1]
    assert len(batch["pixel_values"]) == 1


def test_a_pathlib_path_image_cell_is_coerced_likewise(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "p", "image": Path("/corpus/p.png")}])
    assert image_loader == [("train-row[0]", "/corpus/p.png")]
    assert rec.images_received == [1]
    assert len(batch["pixel_values"]) == 1


def test_rows_without_images_contribute_zero_images_and_do_not_crash(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # A mixed batch: a missing cell, an explicit None, and a non-dict row
    # (the vars() branch). None must become ZERO images -- not one attempt
    # to load the path "None" -- and the stub emits no pixel column for an
    # empty images list, which is the honest shape of a text-only batch.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate(
        [
            {"text": "no-cell"},
            {"text": "none-cell", "image": None},
            SimpleNamespace(text="namespace-row", image=None),
        ]
    )
    assert image_loader == []
    assert rec.images_received == [0]
    assert "pixel_values" not in batch
    assert rec.calls[0]["text"] == ["no-cell", "none-cell", "namespace-row"]


def test_multi_image_rows_flatten_in_row_order(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # FAILING INPUT: rows carrying several images each. Any flattening order
    # other than row-major-then-per-row pairs one row's pixels with another
    # row's text -- wrong supervision, zero error signal. The loader's call
    # record IS the order contract.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate(
        [
            {"text": "a", "image": ["/corpus/a1.png", "/corpus/a2.png"]},
            {"text": "b", "image": ["/corpus/b1.png", "/corpus/b2.png", "/corpus/b3.png"]},
        ]
    )
    assert [path for _row, path in image_loader] == [
        "/corpus/a1.png",
        "/corpus/a2.png",
        "/corpus/b1.png",
        "/corpus/b2.png",
        "/corpus/b3.png",
    ]
    assert rec.images_received == [5]
    assert len(batch["pixel_values"]) == 5


def test_labels_are_injected_as_a_clone_of_input_ids_when_absent(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # The injection must be a CLONE: aliasing input_ids would let a later
    # in-place shift for causal LM corrupt the inputs themselves.
    surface, _rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "a", "image": "/corpus/a.png"}])
    ids = batch["input_ids"]
    assert ids.clone_calls == 1
    assert batch["labels"] is not ids
    assert batch["labels"].rows == ids.rows


def test_labels_already_present_are_not_overwritten(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # A processor that emits its own labels (e.g. with the prompt masked)
    # must keep them; overwriting would silently retrain on the prompt too.
    surface, rec = processor
    rec.include_labels = True
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "a", "image": "/corpus/a.png"}])
    assert batch["labels"] is rec.labels_sentinel
    assert batch["input_ids"].clone_calls == 0


def test_text_column_default_and_override_read_the_right_cell(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    surface, rec = processor
    default_collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    default_collate([{"text": "default-cell", "image": "/corpus/a.png"}])
    override_collate = train_image_collator_or_refuse(
        surface, image_column="image", max_length=8, text_column="prompt"
    )
    override_collate([{"prompt": "override-cell", "text": "ignored", "image": "/corpus/b.png"}])
    assert rec.calls[0]["text"] == ["default-cell"]
    assert rec.calls[1]["text"] == ["override-cell"]


def test_max_length_is_forwarded_to_the_surface_unchanged(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # A collator that truncated at a hard-coded length would silently change
    # the supervision width the run was priced against.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=1234)
    collate([{"text": "a", "image": "/corpus/a.png"}])
    assert rec.calls[0]["max_length"] == 1234
    assert rec.calls[0]["truncation"] is True


def test_a_declared_image_path_that_does_not_exist_is_refused(
    processor: tuple[PromptSurface, _RecordingProcessor],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: a corpus row pointing at an image deleted after
    # curation. NO monkeypatch here on purpose: _load_image_or_refuse
    # refuses on the Path.exists() check BEFORE importing PIL, so this leg
    # exercises the real loader on a torch-free, PIL-free host. A silent
    # drop would recreate #371 one layer down.
    surface, _rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    with pytest.raises(SystemExit) as excinfo:
        collate([{"text": "a", "image": "/no/such/dir/ghost-410.png"}])
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "train-row[0]" in err
    assert "/no/such/dir/ghost-410.png" in err
