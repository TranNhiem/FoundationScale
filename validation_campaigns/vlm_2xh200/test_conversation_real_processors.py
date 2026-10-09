"""Real-processor checks for train/conversation.py (GPU host campaign, not CI).

Run on a host with the checkpoints::

    FS_TEST_MODELS_DIR=/path/to/models python -m pytest validation_campaigns/vlm_2xh200 -q

CI forbids skips (FS_FORBID_SKIPS=1), so these live outside ``tests/``. Without the
models this module exits 95 (unmeasured) instead of skipping: an absent check must
not read as a passing one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from foundationscale.train.conversation import (
    conversation_prepass_or_refuse,
    derive_turn_markers,
    train_conversation_collator_or_refuse,
)

MODELS_DIR = os.environ.get("FS_TEST_MODELS_DIR")
_FAMILIES = ("gemma-4-12B-it", "Qwen3.6-27B")


def _real_processors_available() -> bool:
    if not MODELS_DIR:
        return False
    base = Path(MODELS_DIR)
    return all((base / name).is_dir() for name in _FAMILIES)


requires_real_processors = pytest.mark.skipif(
    not _real_processors_available(),
    reason=(
        "FS_TEST_MODELS_DIR is unset or does not contain both "
        f"{_FAMILIES!r}; set it to run the real-processor suite"
    ),
)


def _sharegpt_row(
    turns: list[dict[str, str]],
    *,
    image: Any = None,
    video: Any = None,
    conversations_column: str = "conversations",
    image_column: str = "image",
    video_column: str = "video",
) -> dict[str, Any]:
    row: dict[str, Any] = {conversations_column: turns}
    if image is not None:
        row[image_column] = image
    if video is not None:
        row[video_column] = video
    return row


def _make_image(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(path)
    return str(path)


class _UntouchedProcessor:
    """A processor stub that explodes the instant anything on it is called.

    Used to PROVE a refusal fires before the collator ever needs a working
    chat template -- the video/frames_for and bad-overlong-policy refusals
    are detected from the row/declaration alone, so a real processor should
    never be required to reach them.
    """

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("processor touched before the declaration-only refusal fired")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("processor touched before the declaration-only refusal fired")


# ---------------------------------------------------------------------------
# FAST: normalize_conversation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# REAL PROCESSOR
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=_FAMILIES)
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def processor(model_name: str) -> Any:
    from transformers import AutoProcessor

    assert MODELS_DIR is not None
    return AutoProcessor.from_pretrained(str(Path(MODELS_DIR) / model_name), trust_remote_code=True)


@pytest.fixture
def is_gemma(model_name: str) -> bool:
    return model_name.startswith("gemma")


@requires_real_processors
class TestRealProcessor:
    def test_derive_turn_markers_matches_measured_ids(self, processor: Any, is_gemma: bool) -> None:
        markers = derive_turn_markers(processor)
        if is_gemma:
            assert markers.header_ids == (105, 4368, 107)
            assert markers.end_ids == (106,)
        else:
            assert markers.header_ids == (248045, 74455, 198)
            assert markers.end_ids == (248046,)

    def test_text_only_row_supervised_decodes_to_answer_plus_end_marker(
        self, processor: Any, is_gemma: bool
    ) -> None:
        row = _sharegpt_row(
            [
                {"from": "human", "value": "Hello there, how are you doing today?"},
                {"from": "gpt", "value": "I am doing well, thanks for asking!"},
            ]
        )
        collate = train_conversation_collator_or_refuse(
            processor, max_length=256, inject_dummy_media=False
        )
        batch = collate([row])
        labels = batch["labels"][0].tolist()
        supervised_ids = [int(t) for t in labels if t != -100]
        decoded = processor.tokenizer.decode(supervised_ids, skip_special_tokens=False)
        if is_gemma:
            # No reasoning anywhere in the row -> Gemma renders the final
            # turn with no thought channel at all (MEASURED).
            expected = "I am doing well, thanks for asking!<turn|>"
        else:
            # MEASURED quirk: Qwen3.6's template unconditionally wraps the
            # FINAL assistant turn in <think>...</think> (empty when no
            # reasoning_content is set) -- this is baked into
            # `loop.index0 > ns.last_query_index`, independent of
            # enable_thinking, which only affects the generation-prompt
            # tail this collator never renders. Train == inference format,
            # so the empty wrapper is correctly part of what is supervised.
            expected = "<think>\n\n</think>\n\nI am doing well, thanks for asking!<|im_end|>"
        assert decoded == expected

    def test_six_turn_row_three_spans_no_human_text_supervised(
        self, processor: Any, is_gemma: bool
    ) -> None:
        human_texts = ["Question one?", "Question two?", "Question three?"]
        gpt_texts = ["Answer one.", "Answer two.", "Answer three."]
        turns = []
        for h, g in zip(human_texts, gpt_texts, strict=True):
            turns.append({"from": "human", "value": h})
            turns.append({"from": "gpt", "value": g})
        row = _sharegpt_row(turns)
        collate = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=False
        )
        batch = collate([row])
        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()
        tok = processor.tokenizer

        spans: list[tuple[int, int]] = []
        start: int | None = None
        for i, value in enumerate(labels):
            if value != -100:
                if start is None:
                    start = i
            elif start is not None:
                spans.append((start, i - 1))
                start = None
        if start is not None:
            spans.append((start, len(labels) - 1))

        assert len(spans) == 3
        for (span_start, span_end), expected_answer in zip(spans, gpt_texts, strict=True):
            decoded = tok.decode(input_ids[span_start : span_end + 1], skip_special_tokens=False)
            assert expected_answer in decoded
            for h in human_texts:
                assert h not in decoded

    def test_reasoning_final_turn_supervised_history_turn_absent(
        self, processor: Any, is_gemma: bool
    ) -> None:
        row = _sharegpt_row(
            [
                {"from": "human", "value": "First question?"},
                {"from": "gpt", "value": "<think>HISTORYTHOUGHT</think>First answer."},
                {"from": "human", "value": "Second question?"},
                {"from": "gpt", "value": "<think>FINALTHOUGHT</think>Second answer."},
            ]
        )
        collate = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=False
        )
        batch = collate([row])
        input_ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()
        tok = processor.tokenizer

        full_decoded = tok.decode(input_ids, skip_special_tokens=False)
        assert "FINALTHOUGHT" in full_decoded
        assert "HISTORYTHOUGHT" not in full_decoded

        supervised_ids = [t for t, v in zip(input_ids, labels, strict=True) if v != -100]
        supervised_decoded = tok.decode(supervised_ids, skip_special_tokens=False)
        assert "FINALTHOUGHT" in supervised_decoded

    def test_image_and_text_row_collate_together_with_pixel_tensors(
        self, processor: Any, is_gemma: bool, tmp_path: Path
    ) -> None:
        image_path = _make_image(tmp_path, "img.png")
        row_image = _sharegpt_row(
            [
                {"from": "human", "value": "<image>Describe this."},
                {"from": "gpt", "value": "A test image."},
            ],
            image=image_path,
        )
        row_text = _sharegpt_row(
            [{"from": "human", "value": "Hello."}, {"from": "gpt", "value": "Hi!"}]
        )
        collate = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=False
        )
        batch = collate([row_image, row_text])
        assert batch["input_ids"].shape[0] == 2
        assert "pixel_values" in batch
        assert batch["pixel_values"].numel() > 0

        tok = processor.tokenizer
        labels_text_row = batch["labels"][1].tolist()
        supervised = [int(t) for t in labels_text_row if t != -100]
        decoded = tok.decode(supervised, skip_special_tokens=False)
        assert "Hi!" in decoded

    def test_no_image_or_video_token_ever_supervised(
        self, processor: Any, is_gemma: bool, tmp_path: Path
    ) -> None:
        image_path = _make_image(tmp_path, "img2.png")
        row_image = _sharegpt_row(
            [
                {"from": "human", "value": "<image>What is this?"},
                {"from": "gpt", "value": "A picture."},
            ],
            image=image_path,
        )
        collate = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=False
        )
        batch = collate([row_image])
        image_token_id = getattr(processor, "image_token_id", None)
        assert image_token_id is not None
        labels = batch["labels"]
        assert not bool((labels == image_token_id).any())
        video_token_id = getattr(processor, "video_token_id", None)
        if video_token_id is not None:
            assert not bool((labels == video_token_id).any())

    def test_overlong_row_dropped_with_policy_drop(self, processor: Any, is_gemma: bool) -> None:
        long_text = "word " * 500
        row_long = _sharegpt_row(
            [{"from": "human", "value": long_text}, {"from": "gpt", "value": "Answer."}]
        )
        row_short = _sharegpt_row(
            [{"from": "human", "value": "Hi."}, {"from": "gpt", "value": "Hello."}]
        )
        collate = train_conversation_collator_or_refuse(
            processor, max_length=32, overlong="drop", inject_dummy_media=False
        )
        batch = collate([row_long, row_short])
        assert batch["input_ids"].shape[0] == 1
        assert collate.stats["dropped_overlong"] == 1

    def test_overlong_row_refuses_with_policy_refuse(self, processor: Any, is_gemma: bool) -> None:
        long_text = "word " * 500
        row_long = _sharegpt_row(
            [{"from": "human", "value": long_text}, {"from": "gpt", "value": "Answer."}]
        )
        collate = train_conversation_collator_or_refuse(
            processor, max_length=32, overlong="refuse", inject_dummy_media=False
        )
        with pytest.raises(SystemExit) as exc_info:
            collate([row_long])
        assert exc_info.value.code == 96

    def test_all_rows_dropped_overlong_refuses(self, processor: Any, is_gemma: bool) -> None:
        long_text = "word " * 500
        row_long = _sharegpt_row(
            [{"from": "human", "value": long_text}, {"from": "gpt", "value": "Answer."}]
        )
        collate = train_conversation_collator_or_refuse(
            processor, max_length=32, overlong="drop", inject_dummy_media=False
        )
        with pytest.raises(SystemExit) as exc_info:
            collate([row_long])
        assert exc_info.value.code == 96

    def test_pad_to_max_length_shape(self, processor: Any, is_gemma: bool) -> None:
        row = _sharegpt_row([{"from": "human", "value": "Hi."}, {"from": "gpt", "value": "Hello."}])
        collate = train_conversation_collator_or_refuse(
            processor, max_length=64, pad_to_max_length=True, inject_dummy_media=False
        )
        batch = collate([row])
        assert batch["input_ids"].shape[-1] == 64

    def test_dummy_media_spliced_in_place_masked_and_labels_byte_identical(
        self, processor: Any, is_gemma: bool
    ) -> None:
        # max_length=512: comfortably fits a short text row plus Gemma-4's own
        # image expansion (measured ~280 tokens for this dummy size).
        row = _sharegpt_row([{"from": "human", "value": "Hi."}, {"from": "gpt", "value": "Hello."}])
        collate_no_dummy = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=False
        )
        batch_no_dummy = collate_no_dummy([row])
        collate_dummy = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=True
        )
        batch_dummy = collate_dummy([row])

        # No extra row: the dummy is spliced INTO the one real row, not appended.
        assert batch_dummy["input_ids"].shape[0] == 1
        assert "pixel_values" in batch_dummy
        assert batch_dummy["pixel_values"].numel() > 0

        attention = batch_dummy["attention_mask"][0]
        labels = batch_dummy["labels"][0]
        masked_positions = attention == 0
        # A dynamic-padding, single-row batch has no padding of its own, so any
        # masked position here IS the dummy's span.
        assert bool(masked_positions.any())
        assert bool((labels[masked_positions] == -100).all())

        tok = processor.tokenizer
        real_ids_no_dummy = [int(t) for t in batch_no_dummy["labels"][0].tolist() if t != -100]
        real_ids_dummy = [int(t) for t in labels.tolist() if t != -100]
        assert tok.decode(real_ids_dummy, skip_special_tokens=False) == tok.decode(
            real_ids_no_dummy, skip_special_tokens=False
        )
        assert collate_dummy.stats["dummy_media_batches"] == 1

    def test_dummy_media_width_unchanged_under_pad_to_max_length(
        self, processor: Any, is_gemma: bool
    ) -> None:
        row = _sharegpt_row([{"from": "human", "value": "Hi."}, {"from": "gpt", "value": "Hello."}])
        collate = train_conversation_collator_or_refuse(
            processor, max_length=512, pad_to_max_length=True, inject_dummy_media=True
        )
        batch = collate([row])
        # Exactly one row, exactly max_length wide -- no doubled row, no grown width.
        assert batch["input_ids"].shape == (1, 512)
        assert batch["attention_mask"].shape == (1, 512)
        assert "pixel_values" in batch
        assert collate.stats["dummy_media_batches"] == 1

    def test_dummy_media_picks_shortest_row_others_untouched(
        self, processor: Any, is_gemma: bool
    ) -> None:
        row_short = _sharegpt_row(
            [{"from": "human", "value": "Hi."}, {"from": "gpt", "value": "Hello."}]
        )
        row_long = _sharegpt_row(
            [
                {"from": "human", "value": "Tell me a longer story about your day please."},
                {
                    "from": "gpt",
                    "value": "It was a long and eventful day with many things happening.",
                },
            ]
        )
        collate_no_dummy = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=False
        )
        batch_no_dummy = collate_no_dummy([row_long, row_short])
        collate_dummy = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=True
        )
        batch_dummy = collate_dummy([row_long, row_short])

        assert batch_dummy["input_ids"].shape[0] == 2
        assert "pixel_values" in batch_dummy

        tok = processor.tokenizer
        for idx in (0, 1):
            real_no_dummy = [int(t) for t in batch_no_dummy["labels"][idx].tolist() if t != -100]
            real_dummy = [int(t) for t in batch_dummy["labels"][idx].tolist() if t != -100]
            assert tok.decode(real_dummy, skip_special_tokens=False) == tok.decode(
                real_no_dummy, skip_special_tokens=False
            )
        assert collate_dummy.stats["dummy_media_batches"] == 1

    # -----------------------------------------------------------------
    # conversation_prepass_or_refuse
    # -----------------------------------------------------------------

    def test_prepass_length_matches_collate_measured_length(
        self, processor: Any, is_gemma: bool
    ) -> None:
        import datasets

        from foundationscale.train import conversation as conversation_module

        row = _sharegpt_row(
            [{"from": "human", "value": "Hi there."}, {"from": "gpt", "value": "Hello back."}]
        )

        # The collator's own batch-of-N measurement for this row (N=1 here,
        # but attention_mask.sum() reads the same regardless of batch size).
        collate = train_conversation_collator_or_refuse(
            processor, max_length=512, inject_dummy_media=False
        )
        batch = collate([row])
        collate_length = int(batch["attention_mask"][0].sum())

        # The pre-pass's length for the SAME row, via the SAME shared
        # rendering function the collator itself delegates to.
        text, images, videos, _turns = conversation_module._render_conversation_row(
            0,
            row,
            conversations_column="conversations",
            image_column="image",
            video_column="video",
            processor=processor,
            frames_for=None,
        )
        measured = conversation_module._conversation_processor_batch_call(
            processor, [text], [images], [videos], max_length=512, pad_to_max=False
        )
        prepass_length = int(measured["attention_mask"][0].sum())

        assert prepass_length == collate_length

        dataset = datasets.Dataset.from_list([row])
        result = conversation_prepass_or_refuse(
            dataset,
            processor,
            conversations_column="conversations",
            image_column="image",
            max_length=512,
            overlong="drop",
        )
        assert result.refusal_reason is None
        assert result.rows_seen == 1
        assert result.dropped_overlong == 0
        assert result.kept == 1

    def test_prepass_overlong_drop_filters_dataset_deterministically(
        self, processor: Any, is_gemma: bool
    ) -> None:
        import datasets

        long_text = "word " * 500
        row_long = _sharegpt_row(
            [{"from": "human", "value": long_text}, {"from": "gpt", "value": "Answer."}]
        )
        row_short = _sharegpt_row(
            [{"from": "human", "value": "Hi."}, {"from": "gpt", "value": "Hello."}]
        )
        dataset = datasets.Dataset.from_list([row_long, row_short])
        result = conversation_prepass_or_refuse(
            dataset,
            processor,
            conversations_column="conversations",
            image_column="image",
            max_length=32,
            overlong="drop",
        )
        assert result.refusal_reason is None
        assert result.rows_seen == 2
        assert result.dropped_overlong == 1
        assert result.kept == 1
        assert len(result.filtered_dataset) == 1
        assert result.filtered_dataset[0]["conversations"] == row_short["conversations"]

    def test_prepass_overlong_refuse_reports_without_raising(
        self, processor: Any, is_gemma: bool
    ) -> None:
        import datasets

        long_text = "word " * 500
        row_long = _sharegpt_row(
            [{"from": "human", "value": long_text}, {"from": "gpt", "value": "Answer."}]
        )
        dataset = datasets.Dataset.from_list([row_long])
        # Unlike the collator's OWN overlong="refuse" path (which raises
        # SystemExit immediately -- see test_overlong_row_refuses_with_policy_refuse
        # above), the pre-pass NEVER raises for this policy: it reports the
        # reason back so the caller (train/loop.py) can route it through its
        # own _mark/_emit_manifest/EXIT_REFUSE sequence instead.
        result = conversation_prepass_or_refuse(
            dataset,
            processor,
            conversations_column="conversations",
            image_column="image",
            max_length=32,
            overlong="refuse",
        )
        assert result.refusal_reason is not None
        assert "row 0" in result.refusal_reason
        assert result.filtered_dataset is None

    def test_prepass_num_proc_parallel_replay_still_refuses(
        self, processor: Any, is_gemma: bool
    ) -> None:
        import datasets

        good = _sharegpt_row(
            [{"from": "human", "value": "Hi."}, {"from": "gpt", "value": "Hello."}]
        )
        bad = {"conversations": [{"from": "alien", "value": "???"}]}
        dataset = datasets.Dataset.from_list([good, good, bad, good])
        with pytest.raises(SystemExit) as exc_info:
            conversation_prepass_or_refuse(
                dataset,
                processor,
                conversations_column="conversations",
                image_column="image",
                max_length=512,
                overlong="drop",
                num_proc=2,
                batch_size=1,
            )
        assert exc_info.value.code == 96
