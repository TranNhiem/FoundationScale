"""Tests for foundationscale.train.conversation.

Two tiers, split by whether a real chat-template processor is available:

  FAST (always run, no model files): normalize_conversation's ShareGPT ->
  content-block mapping, every refusal path reachable without a processor,
  and assistant_label_mask on hand-built id sequences. These are what CI
  runs on every PR.

  REAL-PROCESSOR (skipped unless FS_TEST_MODELS_DIR is set and contains both
  ``gemma-4-12B-it`` and ``Qwen3.6-27B``): header/end-marker derivation,
  full collation, reasoning-turn behaviour and batching quirks that only a
  real ``Gemma4UnifiedProcessor``/``Qwen3VLProcessor`` chat template can
  prove. Run with::

      FS_TEST_MODELS_DIR=/workspace/fs-vlm/models \
          pytest tests/train/test_conversation_collator.py -q

Processor-only checkpoints are required (tokenizer/processor config), not
full model weights -- loading is CPU-only and fast.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from foundationscale.train.conversation import (
    TurnMarkers,
    assistant_label_mask,
    conversation_prepass_or_refuse,
    derive_turn_markers,
    normalize_conversation,
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


def test_normalize_text_only_two_turn() -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "Hi there."}, {"from": "gpt", "value": "Hello!"}]
    )
    messages = normalize_conversation(
        row, conversations_column="conversations", image_column="image", video_column="video"
    )
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hi there."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
    ]


def test_normalize_messages_alias_role_content() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello!"},
        ]
    }
    messages = normalize_conversation(
        row, conversations_column="messages", image_column="image", video_column="video"
    )
    assert messages[0] == {"role": "user", "content": [{"type": "text", "text": "Hi."}]}
    assert messages[1] == {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}


def test_normalize_system_turn() -> None:
    row = _sharegpt_row(
        [
            {"from": "system", "value": "Be nice."},
            {"from": "human", "value": "Hi."},
            {"from": "gpt", "value": "Hello!"},
        ]
    )
    messages = normalize_conversation(
        row, conversations_column="conversations", image_column="image", video_column="video"
    )
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": "Be nice."}]}


def test_normalize_image_marker_single() -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "<image>What is this?"}, {"from": "gpt", "value": "A cat."}],
        image="/tmp/fake.png",
    )
    messages = normalize_conversation(
        row, conversations_column="conversations", image_column="image", video_column="video"
    )
    assert messages[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "What is this?"},
    ]


def test_normalize_image_marker_multiple() -> None:
    row = _sharegpt_row(
        [
            {"from": "human", "value": "Compare <image> and <image>."},
            {"from": "gpt", "value": "They differ."},
        ],
        image=["/tmp/a.png", "/tmp/b.png"],
    )
    messages = normalize_conversation(
        row, conversations_column="conversations", image_column="image", video_column="video"
    )
    assert messages[0]["content"] == [
        {"type": "text", "text": "Compare "},
        {"type": "image"},
        {"type": "text", "text": " and "},
        {"type": "image"},
        {"type": "text", "text": "."},
    ]


def test_normalize_video_marker() -> None:
    row = _sharegpt_row(
        [
            {"from": "human", "value": "<video>Describe the clip."},
            {"from": "gpt", "value": "A dog runs."},
        ],
        video="/tmp/clip.mp4",
    )
    messages = normalize_conversation(
        row, conversations_column="conversations", image_column="image", video_column="video"
    )
    assert messages[0]["content"] == [
        {"type": "video"},
        {"type": "text", "text": "Describe the clip."},
    ]


def test_normalize_think_split() -> None:
    row = _sharegpt_row(
        [
            {"from": "human", "value": "Explain."},
            {"from": "gpt", "value": "<think>because reasons</think>The answer is 42."},
        ]
    )
    messages = normalize_conversation(
        row, conversations_column="conversations", image_column="image", video_column="video"
    )
    assistant_msg = messages[1]
    assert assistant_msg["reasoning_content"] == "because reasons"
    assert assistant_msg["content"] == [{"type": "text", "text": "The answer is 42."}]


def test_normalize_assistant_without_think_tag_has_no_reasoning_key() -> None:
    row = _sharegpt_row([{"from": "human", "value": "Hi."}, {"from": "gpt", "value": "Hello!"}])
    messages = normalize_conversation(
        row, conversations_column="conversations", image_column="image", video_column="video"
    )
    assert "reasoning_content" not in messages[1]
    assert messages[1]["content"] == [{"type": "text", "text": "Hello!"}]


def test_normalize_image_marker_count_mismatch_refuses() -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "<image><image>Two?"}, {"from": "gpt", "value": "Yes."}],
        image="/tmp/only_one.png",
    )
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_media_without_markers_refuses() -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "No marker here."}, {"from": "gpt", "value": "OK."}],
        image="/tmp/orphan.png",
    )
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_markers_without_media_refuses() -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "<image>What is this?"}, {"from": "gpt", "value": "Unclear."}]
    )
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_multiple_video_markers_refuses() -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "<video><video>Two clips?"}, {"from": "gpt", "value": "No."}],
        video="/tmp/clip.mp4",
    )
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_unknown_role_refuses() -> None:
    row = {"conversations": [{"from": "alien", "value": "???"}]}
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_missing_value_field_refuses() -> None:
    row = {"conversations": [{"from": "human"}]}
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_none_value_refuses_without_coercion(capsys: pytest.CaptureFixture[str]) -> None:
    row = {"conversations": [{"from": "human", "value": None}]}
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96
    # Never trained as the literal string "None".
    assert "'None'" not in capsys.readouterr().err


def test_normalize_non_string_value_refuses() -> None:
    row = {"conversations": [{"from": "human", "value": 42}]}
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_unterminated_think_tag_refuses() -> None:
    row = _sharegpt_row(
        [
            {"from": "human", "value": "Explain."},
            {"from": "gpt", "value": "<think>reasoning with no close"},
        ]
    )
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_stray_close_think_tag_refuses() -> None:
    row = _sharegpt_row(
        [
            {"from": "human", "value": "Explain."},
            {"from": "gpt", "value": "no open tag</think>The answer."},
        ]
    )
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column="image", video_column="video"
        )
    assert exc_info.value.code == 96


def test_normalize_image_column_none_text_only_is_clean() -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "No image here."}, {"from": "gpt", "value": "OK."}]
    )
    messages = normalize_conversation(
        row, conversations_column="conversations", image_column=None, video_column="video"
    )
    assert messages[0]["content"] == [{"type": "text", "text": "No image here."}]


def test_normalize_image_column_none_with_marker_refuses_naming_declaration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "<image>What is this?"}, {"from": "gpt", "value": "Unclear."}]
    )
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row, conversations_column="conversations", image_column=None, video_column="video"
        )
    assert exc_info.value.code == 96
    assert "FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN" in capsys.readouterr().err


def test_normalize_refusal_names_row_index(capsys: pytest.CaptureFixture[str]) -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "<image>Look."}, {"from": "gpt", "value": "OK."}]
    )
    with pytest.raises(SystemExit) as exc_info:
        normalize_conversation(
            row,
            conversations_column="conversations",
            image_column="image",
            video_column="video",
            row_index=3,
        )
    assert exc_info.value.code == 96
    captured = capsys.readouterr()
    assert "row 3" in captured.err


# ---------------------------------------------------------------------------
# FAST: assistant_label_mask
# ---------------------------------------------------------------------------


def test_assistant_label_mask_single_span() -> None:
    markers = TurnMarkers(header_ids=(9, 9), end_ids=(8,))
    input_ids = torch.tensor([[1, 2, 9, 9, 5, 6, 7, 8, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    labels = assistant_label_mask(input_ids, attention_mask, markers)
    expected = torch.tensor([[-100, -100, -100, -100, 5, 6, 7, 8, -100, -100]])
    assert torch.equal(labels, expected)


def test_assistant_label_mask_multi_span() -> None:
    markers = TurnMarkers(header_ids=(9,), end_ids=(8,))
    input_ids = torch.tensor([[9, 1, 2, 8, 3, 9, 4, 5, 8]])
    attention_mask = torch.ones_like(input_ids)
    labels = assistant_label_mask(input_ids, attention_mask, markers)
    expected = torch.tensor([[-100, 1, 2, 8, -100, -100, 4, 5, 8]])
    assert torch.equal(labels, expected)


def test_assistant_label_mask_expected_spans_match_passes() -> None:
    markers = TurnMarkers(header_ids=(9,), end_ids=(8,))
    input_ids = torch.tensor([[9, 1, 2, 8, 3, 9, 4, 5, 8]])
    attention_mask = torch.ones_like(input_ids)
    labels = assistant_label_mask(input_ids, attention_mask, markers, expected_spans=[2])
    expected = torch.tensor([[-100, 1, 2, 8, -100, -100, 4, 5, 8]])
    assert torch.equal(labels, expected)


def test_assistant_label_mask_expected_spans_mismatch_refuses() -> None:
    # Two spans are found, but the row's own conversation structure only
    # declared one assistant turn -- as if a user/system turn's text
    # happened to quote the literal header/end-marker ids.
    markers = TurnMarkers(header_ids=(9,), end_ids=(8,))
    input_ids = torch.tensor([[9, 1, 2, 8, 3, 9, 4, 5, 8]])
    attention_mask = torch.ones_like(input_ids)
    with pytest.raises(SystemExit) as exc_info:
        assistant_label_mask(input_ids, attention_mask, markers, expected_spans=[1])
    assert exc_info.value.code == 96


def test_assistant_label_mask_no_expected_spans_is_unaffected() -> None:
    # Default (None) keeps every existing direct caller, including the
    # fixtures above, unaffected by the new check.
    markers = TurnMarkers(header_ids=(9, 9), end_ids=(8,))
    input_ids = torch.tensor([[1, 2, 9, 9, 5, 6, 7, 8, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    labels = assistant_label_mask(input_ids, attention_mask, markers, expected_spans=None)
    expected = torch.tensor([[-100, -100, -100, -100, 5, 6, 7, 8, -100, -100]])
    assert torch.equal(labels, expected)


def test_assistant_label_mask_header_absent_refuses() -> None:
    markers = TurnMarkers(header_ids=(9, 9), end_ids=(8,))
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    with pytest.raises(SystemExit) as exc_info:
        assistant_label_mask(input_ids, attention_mask, markers)
    assert exc_info.value.code == 96


def test_assistant_label_mask_padding_masked_via_attention_mask() -> None:
    markers = TurnMarkers(header_ids=(9,), end_ids=(8,))
    # right-padded: header, content, end, then two pad positions.
    input_ids = torch.tensor([[9, 5, 6, 8, 3, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    labels = assistant_label_mask(input_ids, attention_mask, markers)
    expected = torch.tensor([[-100, 5, 6, 8, -100, -100]])
    assert torch.equal(labels, expected)


def test_assistant_label_mask_pad_id_zero_inside_span_stays_supervised() -> None:
    markers = TurnMarkers(header_ids=(9,), end_ids=(8,))
    # left-padded with pad id 0; the span's own content legitimately contains
    # a literal 0 token too -- masking must come from attention_mask alone,
    # never from an id==0 comparison (gemma's real pad id is 0).
    input_ids = torch.tensor([[0, 0, 9, 0, 7, 8]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    labels = assistant_label_mask(input_ids, attention_mask, markers)
    expected = torch.tensor([[-100, -100, -100, 0, 7, 8]])
    assert torch.equal(labels, expected)


# ---------------------------------------------------------------------------
# FAST: declaration-only collator refusals (processor never touched)
# ---------------------------------------------------------------------------


def test_collator_video_without_frames_for_refuses(tmp_path: Path) -> None:
    row = _sharegpt_row(
        [
            {"from": "human", "value": "<video>Describe this clip."},
            {"from": "gpt", "value": "A clip."},
        ],
        video=str(tmp_path / "clip.mp4"),
    )
    collate = train_conversation_collator_or_refuse(_UntouchedProcessor(), max_length=128)
    with pytest.raises(SystemExit) as exc_info:
        collate([row])
    assert exc_info.value.code == 96


def test_collator_invalid_overlong_policy_refuses() -> None:
    with pytest.raises(SystemExit) as exc_info:
        train_conversation_collator_or_refuse(
            _UntouchedProcessor(), max_length=128, overlong="bogus"
        )
    assert exc_info.value.code == 96


# ---------------------------------------------------------------------------
# FAST: conversation_prepass_or_refuse -- validation-only paths (no real
# processor needed: every case here fails before apply_chat_template is
# ever reached, same guarantee _UntouchedProcessor already proves for the
# collator's own declaration-only refusals).
# ---------------------------------------------------------------------------


class _FakeMappedDataset:
    """Just enough of datasets.Dataset's post-.map() surface to test with."""

    def __init__(self, rows: list[dict[str, Any]], columns: dict[str, list[Any]]) -> None:
        self._rows = rows
        self._columns = columns

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._columns[key]
        return self._rows[key]

    def select(self, indices: list[int]) -> _FakeDataset:
        return _FakeDataset([self._rows[i] for i in indices])


class _FakeDataset:
    """A minimal stand-in for datasets.Dataset -- only what the pre-pass uses.

    num_proc is accepted and ignored (always runs sequentially): every FAST
    test here fails during row VALIDATION, which raises authentically
    regardless of num_proc (the parallel-only catch-as-data path only
    matters once a worker process is actually involved).
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def map(
        self,
        fn: Any,
        *,
        batched: bool,
        batch_size: int,
        with_indices: bool,
        num_proc: int | None,
        desc: str,
    ) -> _FakeMappedDataset:
        columns: dict[str, list[Any]] = {}
        for start in range(0, len(self._rows), batch_size):
            chunk = self._rows[start : start + batch_size]
            indices = list(range(start, start + len(chunk)))
            batch: dict[str, list[Any]] = {}
            for row in chunk:
                for key, value in row.items():
                    batch.setdefault(key, []).append(value)
            result = fn(batch, indices)
            for key, value in result.items():
                columns.setdefault(key, []).extend(value)
        return _FakeMappedDataset(self._rows, columns)


def test_prepass_refuses_on_first_bad_row(capsys: pytest.CaptureFixture[str]) -> None:
    # Both rows fail validation (so a real processor is never required by
    # EITHER), but with two DIFFERENT reasons -- proving the row-0 message is
    # what surfaces, not row-1's, since a sequential pre-pass must raise on
    # the first failure it reaches and never touch row 1 at all.
    bad_first = {"conversations": [{"from": "alien", "value": "???"}]}
    bad_second = {"conversations": [{"from": "human", "value": None}]}
    dataset = _FakeDataset([bad_first, bad_second])
    with pytest.raises(SystemExit) as exc_info:
        conversation_prepass_or_refuse(
            dataset,
            _UntouchedProcessor(),
            conversations_column="conversations",
            image_column="image",
            max_length=128,
            overlong="drop",
        )
    assert exc_info.value.code == 96
    assert "unrecognized conversation role" in capsys.readouterr().err


def test_prepass_image_column_none_with_marker_refuses() -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "<image>What is this?"}, {"from": "gpt", "value": "Unclear."}]
    )
    dataset = _FakeDataset([row])
    with pytest.raises(SystemExit) as exc_info:
        conversation_prepass_or_refuse(
            dataset,
            _UntouchedProcessor(),
            conversations_column="conversations",
            image_column=None,
            max_length=128,
            overlong="drop",
        )
    assert exc_info.value.code == 96


def test_prepass_missing_image_file_refuses(tmp_path: Path) -> None:
    row = _sharegpt_row(
        [{"from": "human", "value": "<image>Look."}, {"from": "gpt", "value": "OK."}],
        image=str(tmp_path / "does-not-exist.png"),
    )
    dataset = _FakeDataset([row])
    with pytest.raises(SystemExit) as exc_info:
        conversation_prepass_or_refuse(
            dataset,
            _UntouchedProcessor(),
            conversations_column="conversations",
            image_column="image",
            max_length=128,
            overlong="drop",
        )
    assert exc_info.value.code == 96


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
