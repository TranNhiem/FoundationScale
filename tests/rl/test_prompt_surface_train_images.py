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

    pixel_values is emitted ONLY when __call__ is handed an images argument
    containing at least one real image. A stub that always emits the key
    makes every positive leg pass on a collator that drops the image -- the
    exact defect under test -- so the emission is conditional here and the
    received images are recorded for the legs to assert on. The images are
    now recorded VERBATIM, because the measured contract nests them one
    sub-list per feature, and the nesting itself (an empty sub-list for an
    image-less row) is what keeps the image and text lists aligned -- a
    count-only record could not see a collator that slipped the boundaries.

    apply_chat_template RENDERS rather than echoes: each image block becomes
    "<image>" and each text block its own text. A stub that returned the
    cell text unchanged would pass on a collator that skipped the template
    entirely, which is the second defect the new contract closes.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.images_received: list[Any] = []
        self.template_calls: list[dict[str, Any]] = []
        self.include_labels = False
        self.labels_sentinel = _FakeIds(["gold"])

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.template_calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        rendered: list[str] = []
        for block in conversation[0]["content"]:
            rendered.append(block["text"] if block["type"] == "text" else "<image>")
        return "".join(rendered)

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        images = kwargs.get("images")
        self.images_received.append(images)
        total_images = sum(len(cell) for cell in (images or []))
        batch: dict[str, Any] = {"input_ids": _FakeIds(kwargs["text"])}
        if total_images:
            batch["pixel_values"] = ["px"] * total_images
        if self.include_labels:
            batch["labels"] = self.labels_sentinel
        return batch


def _nested_paths(images: Any) -> list[list[str]]:
    """Read back the per-feature sub-list structure the collator built.

    The shape IS the contract: one sub-list per row, empty where the row
    carried no image. Flattening here would hide exactly the misalignment
    the measured fix exists to prevent.
    """
    return [[cell.path for cell in cell_list] for cell_list in (images or [])]


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
    # WHAT CHANGED under the measured fix: the two images must arrive NESTED
    # one sub-list per feature, [[a],[b]]. Measured on both real processor
    # families, a flat [a, b] against two templated texts reads 1 image-batch
    # against 2 texts and raises on one family while silently misaligning on
    # the other -- so the total AND the nesting are both pinned.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate(
        [
            {"text": "first", "image": ["/corpus/a.png"]},
            {"text": "second", "image": ["/corpus/b.png"]},
        ]
    )
    assert image_loader == [("train-row[0]", "/corpus/a.png"), ("train-row[1]", "/corpus/b.png")]
    assert _nested_paths(rec.images_received[0]) == [["/corpus/a.png"], ["/corpus/b.png"]]
    assert len(batch["pixel_values"]) == 2
    # The template ran, for the TRAIN plane: tokenize off, and NO generation
    # prompt -- labels are input_ids here, so an assistant-turn opener would
    # be trained on.
    assert len(rec.template_calls) == 2
    assert all(call["tokenize"] is False for call in rec.template_calls)
    assert all(call["add_generation_prompt"] is False for call in rec.template_calls)
    assert rec.calls[0]["text"] == ["<image>first", "<image>second"]


def test_a_bare_string_image_cell_is_coerced_to_one_image(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # A corpus row whose image cell is a plain string, not a list. Iterating
    # the string instead of wrapping it would load one "image" PER CHARACTER
    # -- the coercion is what this leg pins. Under the new contract the one
    # coerced image still arrives nested inside its own per-feature sub-list.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "only", "image": "/corpus/only.png"}])
    assert image_loader == [("train-row[0]", "/corpus/only.png")]
    assert _nested_paths(rec.images_received[0]) == [["/corpus/only.png"]]
    assert len(batch["pixel_values"]) == 1


def test_a_pathlib_path_image_cell_is_coerced_likewise(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "p", "image": Path("/corpus/p.png")}])
    assert image_loader == [("train-row[0]", "/corpus/p.png")]
    assert _nested_paths(rec.images_received[0]) == [["/corpus/p.png"]]
    assert len(batch["pixel_values"]) == 1


def test_rows_without_images_contribute_zero_images_and_do_not_crash(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # A mixed batch: a missing cell, an explicit None, and a non-dict row
    # (the vars() branch), plus one row that DOES carry an image. None must
    # become ZERO images -- not one attempt to load the path "None".
    #
    # WHAT CHANGED under the measured fix: an image-less row in an imaged
    # batch no longer contributes NOTHING. It contributes an EMPTY SUB-LIST,
    # because the processor now receives images nested one sub-list per
    # feature -- measured, dropping the empty sub-list makes the outer list
    # shorter than the text list and reproduces the desynchronisation the
    # fix closed. The pins below assert that exact structure.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate(
        [
            {"text": "no-cell"},
            {"text": "none-cell", "image": None},
            {"text": "lead", "image": "/corpus/lead.png"},
            SimpleNamespace(text="namespace-row", image=None),
        ]
    )
    assert image_loader == [("train-row[2]", "/corpus/lead.png")]
    assert _nested_paths(rec.images_received[0]) == [
        [],
        [],
        ["/corpus/lead.png"],
        [],
    ]
    assert len(batch["pixel_values"]) == 1
    assert rec.calls[0]["text"] == ["no-cell", "none-cell", "<image>lead", "namespace-row"]

    # The complementary shape: NO row in the batch carries any image at all.
    # That batch takes the TEXT path, which passes no images kwarg
    # whatsoever -- passing even a nested list of empty sub-lists there
    # would be an image-shaped call on a text-shaped batch.
    text_batch = collate([{"text": "only-text"}])
    assert "images" not in rec.calls[-1]
    assert "pixel_values" not in text_batch


def test_multi_image_rows_nest_in_row_order(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # FAILING INPUT: rows carrying several images each. Renamed from
    # "flatten" to "nest": under the measured fix the images are no longer
    # one flat list across all features but one sub-list PER FEATURE, and it
    # is precisely NOT flatt
    # ened at the hand-off. Within that structure the order contract is
    # unchanged: row-major across rows, cell order within a row. Any other
    # pairing joins one row's pixels to another row's text -- wrong
    # supervision, zero error signal. The loader's call record covers order;
    # the nesting pin covers the sub-list boundaries.
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
    assert _nested_paths(rec.images_received[0]) == [
        ["/corpus/a1.png", "/corpus/a2.png"],
        ["/corpus/b1.png", "/corpus/b2.png", "/corpus/b3.png"],
    ]
    assert rec.calls[0]["text"] == ["<image><image>a", "<image><image><image>b"]
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
    # The cell text now travels through apply_chat_template, so the stub's
    # transcript holds the RENDERED string -- "<image>" plus the cell. That
    # makes these pins stronger than before: they assert BOTH that the right
    # cell was read AND that the template ran, which an echoing assertion on
    # the raw cell text could not see.
    surface, rec = processor
    default_collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    default_collate([{"text": "default-cell", "image": "/corpus/a.png"}])
    override_collate = train_image_collator_or_refuse(
        surface, image_column="image", max_length=8, text_column="prompt"
    )
    override_collate([{"prompt": "override-cell", "text": "ignored", "image": "/corpus/b.png"}])
    assert rec.calls[0]["text"] == ["<image>default-cell"]
    assert rec.calls[1]["text"] == ["<image>override-cell"]


def test_max_length_is_forwarded_to_the_surface_unchanged(
    processor: tuple[PromptSurface, _RecordingProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # RE-SCOPED to the text path. max_length keeps its original meaning --
    # forwarded unchanged, truncation on -- but ONLY for a batch with no
    # images. The image path no longer carries it at all, and deliberately:
    # no constant can be right there. Measured, the SAME image expands to
    # 2337 tokens through one family's processor and ~258 through another's,
    # and truncating at any fixed length desynchronises the template's
    # placeholders from the pixels. A batch with no image-bearing row has no
    # placeholders to desynchronise, so truncation is safe and this leg uses
    # exactly that shape.
    surface, rec = processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=1234)
    collate([{"text": "a"}, {"text": "b"}])
    assert rec.calls[0]["max_length"] == 1234
    assert rec.calls[0]["truncation"] is True
    assert "images" not in rec.calls[0]


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

class _ShapedIds(_FakeIds):
    """_FakeIds plus the one attribute the width guard reads.

    The guard measures the batch with ``int(input_ids.shape[-1])``, so the
    stub must carry a shape whose last dim is the encoded width the leg
    wants. A width-less stub would make the guard crash mid-measurement,
    masking the very refusal the window legs exist to observe.
    """

    def __init__(self, rows: list[str], width: int) -> None:
        super().__init__(rows)
        self.shape = (len(rows), width)

    def clone(self) -> _ShapedIds:
        self.clone_calls += 1
        return _ShapedIds(self.rows, self.shape[-1])


class _TemplatedProcessor:
    """Processor-shaped stub for the templated, nested-image collator.

    Two falsifiability contracts, neither removable:

    * pixel_values is emitted ONLY when the nested images argument flattens
      to something non-empty. A stub that always emits the key passes on a
      collator that drops the pixels -- the defect under test.
    * apply_chat_template RENDERS the blocks it is handed instead of
      echoing the cell text, so the legs can count placeholders in the
      string the processor actually received. An echoing stub passes on a
      collator that skips the chat template entirely.

    encoded_width is the width the next call will report, so the window
    legs steer the guard without a real tokenizer, and tokenizer starts as
    None so the no-declared-window shape is the fixture's own default.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.template_calls: list[dict[str, Any]] = []
        self.encoded_width = 4
        self.tokenizer: Any = None
        self.include_labels = False
        self.labels_sentinel = _ShapedIds(["gold"], 1)

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.template_calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        parts: list[str] = []
        for message in conversation:
            for block in message["content"]:
                if block["type"] == "image":
                    parts.append("<image>")
                else:
                    parts.append(str(block["text"]))
        return "".join(parts)

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        texts = list(kwargs["text"])
        batch: dict[str, Any] = {"input_ids": _ShapedIds(texts, self.encoded_width)}
        images = kwargs.get("images")
        if images:
            flat = [image for sublist in images for image in sublist]
            if flat:
                batch["pixel_values"] = flat
        if self.include_labels:
            batch["labels"] = self.labels_sentinel
        return batch


@pytest.fixture
def templated_processor() -> tuple[PromptSurface, _TemplatedProcessor]:
    rec = _TemplatedProcessor()
    surface = PromptSurface(
        kind="processor", surface=rec, reason="test fixture", supports_images=True
    )
    return surface, rec


def test_images_reach_the_processor_nested_one_sublist_per_row(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # THE NESTING CONTRACT. One outer element PER FEATURE -- including the
    # row with no image -- so the images list and the texts list stay the
    # same length. Row-index stability is pinned too: the loader id is the
    # feature index (train-row[2]), not the ordinal among image rows.
    surface, rec = templated_processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate(
        [
            {"text": "row-a", "image": "/corpus/a.png"},
            {"text": "row-b"},
            {"text": "row-c", "image": ["/corpus/c1.png", "/corpus/c2.png"]},
        ]
    )
    images = rec.calls[0]["images"]
    assert len(images) == 3
    assert [len(sublist) for sublist in images] == [1, 0, 2]
    assert images[0][0].path == "/corpus/a.png"
    assert [img.path for img in images[2]] == ["/corpus/c1.png", "/corpus/c2.png"]
    assert image_loader == [
        ("train-row[0]", "/corpus/a.png"),
        ("train-row[2]", "/corpus/c1.png"),
        ("train-row[2]", "/corpus/c2.png"),
    ]
    assert len(rec.calls[0]["text"]) == 3
    assert len(batch["pixel_values"]) == 3
    assert len(rec.template_calls) == 3


def test_a_flat_image_list_is_not_what_the_processor_receives(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # POSITIVE CONTROL, inverted. On the #410 shape the images argument is
    # the flat [img, img], which is also non-empty and would satisfy a leg
    # asserting only on count. The contract is structural: every top-level
    # element is a LIST, so a bare sentinel at any position fails here.
    surface, rec = templated_processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    collate(
        [
            {"text": "a", "image": "/corpus/a.png"},
            {"text": "b", "image": "/corpus/b.png"},
        ]
    )
    images = rec.calls[0]["images"]
    assert len(images) == 2
    assert all(isinstance(sublist, list) for sublist in images)
    assert not any(isinstance(element, _SentinelImage) for element in images)


def test_an_image_less_row_contributes_an_empty_sublist_not_an_omission(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # Measured on Gemma-4: omitting the empty sub-list lets the processor
    # normalise the short outer list to ONE image-batch and then read N
    # images against M texts -- the original defect, one layer of wrapping
    # down. The empty list IS the alignment and must be present, and the
    # loader must not be consulted for the row that has nothing to load.
    surface, rec = templated_processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate(
        [
            {"text": "no-image"},
            {"text": "has-image", "image": "/corpus/b.png"},
        ]
    )
    images = rec.calls[0]["images"]
    assert len(images) == 2
    assert images[0] == []
    assert isinstance(images[0], list)
    assert image_loader == [("train-row[1]", "/corpus/b.png")]
    assert len(batch["pixel_values"]) == 1


def test_text_is_built_from_the_processors_own_chat_template(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # One {"type": "image"} placeholder block per loaded image, in front of
    # the text block, generation prompt OFF: this is the train plane and
    # labels are input_ids, so an assistant-turn opener would be trained
    # on. Positive control: the stub RENDERS the blocks, so the string the
    # processor received must carry the placeholders -- a collator passing
    # the raw cell text untemplated cannot produce them.
    surface, rec = templated_processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    collate([{"text": "row-text", "image": ["/corpus/a.png", "/corpus/b.png"]}])
    template_call = rec.template_calls[0]
    conversation = template_call["conversation"]
    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"] == [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": "row-text"},
    ]
    assert template_call["tokenize"] is False
    assert template_call["add_generation_prompt"] is False
    processor_text = rec.calls[0]["text"]
    assert processor_text == ["<image><image>row-text"]
    assert processor_text != ["row-text"]


def test_the_chat_template_runs_for_image_less_rows_too(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # A row with zero images still gets templated -- with a content list
    # that is a TEXT BLOCK ONLY. Templating only the image rows would make
    # the rendered shapes inconsistent inside one batch.
    surface, rec = templated_processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    collate(
        [
            {"text": "plain"},
            {"text": "pictured", "image": "/corpus/p.png"},
        ]
    )
    assert rec.template_calls[0]["conversation"][0]["content"] == [
        {"type": "text", "text": "plain"}
    ]
    assert rec.template_calls[1]["conversation"][0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "pictured"},
    ]
    assert rec.calls[0]["text"] == ["plain", "<image>pictured"]


def test_the_image_path_carries_no_truncation_and_no_max_length(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # Measured: the same image expands to ~2337 tokens through one family
    # and ~258 through another, so no literal max_length can be right on
    # this path, and truncation=True desynchronises the placeholders from
    # the pixels. ABSENCE of both keys -- not any particular value -- is
    # the contract.
    surface, rec = templated_processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    collate([{"text": "a", "image": "/corpus/a.png"}])
    call = rec.calls[0]
    assert "truncation" not in call
    assert "max_length" not in call
    assert call["padding"] is True
    assert call["add_special_tokens"] is False
    assert call["return_tensors"] == "pt"


def test_an_all_text_batch_takes_the_truncating_text_path(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # With zero pixels across every row there are no placeholders to
    # desynchronise, so truncation is safe and max_length keeps its
    # meaning. Positive control: the same call must carry NO images key at
    # all -- an all-empty nested list would be the untruncated image path
    # reaching a batch that has nothing to protect.
    surface, rec = templated_processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=64)
    batch = collate([{"text": "alpha"}, {"text": "beta", "image": None}])
    call = rec.calls[0]
    assert call["truncation"] is True
    assert call["max_length"] == 64
    assert "images" not in call
    assert image_loader == []
    assert "pixel_values" not in batch
    assert len(rec.template_calls) == 2


def test_a_batch_wider_than_the_declared_window_is_refused_naming_both_numbers(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # FAILING INPUT: encoded width 9 against a declared window of 8. The
    # image path refuses to truncate BY DESIGN, so an overrun has exactly
    # one honest exit: refuse, naming the width AND the bound, or the
    # operator cannot tell how far over the row went.
    surface, rec = templated_processor
    rec.tokenizer = SimpleNamespace(model_max_length=8)
    rec.encoded_width = 9
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    with pytest.raises(SystemExit) as excinfo:
        collate([{"text": "a", "image": "/corpus/a.png"}])
    assert excinfo.value.code == 96
    err = capsys.readouterr().err
    assert "'image'" in err
    assert "9" in err
    assert "8" in err


def test_a_batch_exactly_at_the_declared_window_is_not_refused(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # THE NEAR MISS. width == declared is the widest batch the model
    # explicitly prices as legal; a '>=' comparison would refuse it and
    # shrink every family's usable window by one token.
    surface, rec = templated_processor
    rec.tokenizer = SimpleNamespace(model_max_length=128)
    rec.encoded_width = 128
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "a", "image": "/corpus/a.png"}])
    assert len(batch["pixel_values"]) == 1


def test_the_transformers_unset_sentinel_is_not_treated_as_a_window(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # transformers writes int(1e30) -- measured exactly
    # 1000000000000000019884624838656 on gemma-4-E4B-it -- to mean "this
    # family never declared a bound". The guard must not refuse against it,
    # and must not pretend a check ran when it did not.
    surface, rec = templated_processor
    rec.tokenizer = SimpleNamespace(model_max_length=1000000000000000019884624838656)
    rec.encoded_width = 1_000_000
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "a", "image": "/corpus/a.png"}])
    assert len(batch["pixel_values"]) == 1


def test_a_processor_with_no_tokenizer_declares_no_window_and_is_not_refused(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # No tokenizer attribute means the width is UNMEASURABLE against a
    # bound that does not exist. The guard returns rather than inventing a
    # threshold -- even at an absurd encoded width.
    surface, rec = templated_processor
    assert rec.tokenizer is None
    rec.encoded_width = 1_000_000
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "a", "image": "/corpus/a.png"}])
    assert len(batch["pixel_values"]) == 1


def test_a_batch_carrying_no_input_ids_has_no_width_to_measure(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
) -> None:
    # The OTHER unmeasurable arm: the family declares a real window, but the
    # batch carries nothing to measure against it -- a processor that names
    # its ids differently, or a mapping already consumed. Refusing there would
    # be refusing on an absence, so the guard returns. Driven directly because
    # no reachable collate path produces a batch without input_ids, and the
    # branch would otherwise be code that only production ever executes.
    surface, rec = templated_processor
    rec.tokenizer = SimpleNamespace(model_max_length=8)
    prompt_surface._refuse_if_image_batch_exceeds_declared_window(surface, {}, "image")
    # CONTROL: same surface, same declared window of 8, a batch that DOES
    # carry a width -- and it refuses. Without this the no-raise above would
    # equally describe a guard that never fires at all.
    with pytest.raises(SystemExit) as excinfo:
        prompt_surface._refuse_if_image_batch_exceeds_declared_window(
            surface, {"input_ids": SimpleNamespace(shape=(1, 9))}, "image"
        )
    assert excinfo.value.code == 96


def test_labels_are_cloned_from_input_ids_on_the_templated_path(
    templated_processor: tuple[PromptSurface, _TemplatedProcessor],
    image_loader: list[tuple[str, str]],
) -> None:
    # The injection survives the rewrite: a CLONE, carrying the templated
    # rows and the measured width, never an alias an in-place shift could
    # use to corrupt the inputs.
    surface, _rec = templated_processor
    collate = train_image_collator_or_refuse(surface, image_column="image", max_length=8)
    batch = collate([{"text": "a", "image": "/corpus/a.png"}])
    ids = batch["input_ids"]
    assert ids.clone_calls == 1
    assert batch["labels"] is not ids
    assert batch["labels"].rows == ids.rows
    assert batch["labels"].shape == ids.shape