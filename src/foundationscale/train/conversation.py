"""The conversation-aware mixed-modality SFT collator.

Generalizes ``train/audio.py::train_audio_collator_or_refuse`` (assistant-only
span, per-row placeholder verification, exit-96 refusals naming the row) from
one text cell plus one modality to multi-turn ShareGPT rows that each carry
0..N images, 0..1 video, or nothing at all, mixed freely in one batch and one
run.

Four pieces, in the order a row passes through them:

  normalize_conversation    ShareGPT ``{from,value}`` (or the ``{role,content}``
                             alias) becomes HF content-block messages.
                             ``<image>``/``<video>`` markers in human turns
                             become ``{"type": "image"}``/``{"type": "video"}``
                             blocks in place; a marker count that disagrees
                             with the declared media is a refusal -- this
                             module never guesses placement.

  derive_turn_markers       The assistant header and end-of-turn token
                             sequences, derived at RUNTIME from the
                             processor's own chat template rather than
                             hard-coded, because a literal copied from one
                             checkpoint silently stops being true the moment
                             the template changes. MEASURED on transformers
                             5.18.0: neither Gemma4UnifiedProcessor's nor
                             Qwen3VLProcessor's template carries
                             ``{% generation %}``, so
                             ``return_assistant_tokens_mask`` is unavailable
                             and this module derives the boundary itself.

  assistant_label_mask      Every token from a header occurrence through the
                             matching end-of-turn marker (inclusive) is
                             supervised; everything else is -100, including
                             padding -- masked via ``attention_mask``, never
                             by comparing against a pad id, because Gemma's
                             pad id is 0 and 0 is a legal content id.

  train_conversation_collator_or_refuse
                             Wires the three together into one
                             ``features -> batch`` callable: per-row
                             rendering, ONE processor call for the batch
                             (images nested per row, images omitted entirely
                             when the whole batch is text-only -- see the
                             docstring below for the two MEASURED shapes a
                             real processor accepts and the two it does
                             not), drop/refuse on overlong rows measured
                             AFTER media expansion, and one dummy image
                             SPLICED INTO an existing row (never an extra
                             batch row) for FSDP collective safety when a
                             batch carries no media at all.

Refusals follow ``rl.prompt_surface._refuse_exit_96``'s mechanism exactly
(stderr line prefixed "REFUSAL (exit 96):", then ``SystemExit(96)``) but are
re-implemented locally rather than imported, mirroring
``train/audio.py``'s ``_audio_refuse_exit_96``: it keeps rl/prompt_surface.py
untouched and keeps this module's own import contract simple. Image loading
DOES import ``rl.prompt_surface._load_image_or_refuse`` directly (function
locally) -- that loader's missing/unreadable-file refusal is exactly what
this module needs and duplicating it would be the only module with its own,
divergent copy of the #371 defense.

torch, PIL and transformers are unreachable from here at import time (the
train package must import under a bare interpreter); they enter function-
locally, inside the functions that actually tokenize, load pixels, or build
tensors.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:  # never executed at runtime: the import contract above forbids it
    import torch

__all__ = [
    "ConversationPrepassResult",
    "TurnMarkers",
    "assistant_label_mask",
    "conversation_prepass_or_refuse",
    "derive_turn_markers",
    "normalize_conversation",
    "train_conversation_collator_or_refuse",
]


def _refuse_exit_96(message: str) -> NoReturn:
    """Loud refusal, exit 96. Never 1, never a silent fallback.

    Mirrors ``rl.prompt_surface._refuse_exit_96`` / ``train.audio._audio_refuse_exit_96``:
    every refusal names the row and the missing/mismatched input so the
    failure is actionable, then exits 96 rather than degrading quietly.

    The message is ALSO attached to the raised exception
    (``exc.fs_refusal_message``), alongside the stderr print -- never in
    place of it. ``SystemExit(96)`` leaves ``.code == 96`` exactly as before
    (every existing ``exc.code == 96`` test keeps working unchanged); the
    attribute exists so a caller that catches this exception on purpose --
    :func:`conversation_prepass_or_refuse`'s single-row replay is the one
    today -- can recover the human message without re-parsing stderr.
    """
    print(f"REFUSAL (exit 96): {message}", file=sys.stderr)
    exc = SystemExit(96)
    exc.fs_refusal_message = message  # type: ignore[attr-defined]
    raise exc


def _as_dict(row: Any) -> Mapping[str, Any]:
    """Accept a plain mapping or a namespace-like row, like the other collators do."""
    return row if isinstance(row, Mapping) else vars(row)


# ---------------------------------------------------------------------------
# normalize_conversation
# ---------------------------------------------------------------------------

# ShareGPT's "from" spellings (and the handful of synonyms real corpora use)
# mapped onto the three roles a chat template understands. An unrecognized
# spelling is a refusal, never a guess -- doctrine: ambiguity is an
# abstention.
_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
}

_IMAGE_MARKER = "<image>"
_VIDEO_MARKER = "<video>"
_MARKER_SPLIT_RE = re.compile(r"(<image>|<video>)")
# Assistant value "<think>X</think>Y" -> reasoning_content=X, content=Y.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"^<think>(.*?)</think>(.*)$", re.DOTALL)


def _think_tag_balance_issue(text: str) -> str | None:
    """None if `text`'s <think>/</think> tags balance (including zero of each).

    An unbalanced count is refused rather than silently treated as plain
    content: an opened reasoning block with no close, or a close with no
    open, means the text was truncated or corrupted upstream, and guessing
    where the missing tag belongs would be exactly the wrong kind of guess
    this module exists to refuse.
    """
    open_count = text.count(_THINK_OPEN)
    close_count = text.count(_THINK_CLOSE)
    if open_count > close_count:
        return (
            f"{open_count} {_THINK_OPEN!r} tag(s) but only {close_count} "
            f"{_THINK_CLOSE!r} tag(s) -- an opened reasoning block with no close"
        )
    if close_count > open_count:
        return (
            f"{close_count} {_THINK_CLOSE!r} tag(s) but only {open_count} "
            f"{_THINK_OPEN!r} tag(s) -- a closing tag with no opening one"
        )
    return None


def _role_and_text(turn: Mapping[str, Any], row_index: int) -> tuple[str, str]:
    if "from" in turn:
        raw_role = turn["from"]
        key = "value"
    elif "role" in turn:
        raw_role = turn["role"]
        key = "content"
    else:
        _refuse_exit_96(
            f"row {row_index}: conversation turn {turn!r} has neither a ShareGPT "
            "'from'/'value' pair nor a 'role'/'content' pair; cannot tell who is "
            "speaking"
        )
    # A missing/None/non-string value is a refusal, never `str(None)` -> "None"
    # silently trained as if it were real text, and never a missing key quietly
    # defaulted to "" -- both would train on content nobody wrote.
    if key not in turn:
        _refuse_exit_96(
            f"row {row_index}: conversation turn {turn!r} has no {key!r} field; "
            "refusing rather than treating an absent value as empty text"
        )
    raw_text = turn[key]
    if raw_text is None or not isinstance(raw_text, str):
        _refuse_exit_96(
            f"row {row_index}: conversation turn {turn!r}'s {key!r} is "
            f"{raw_text!r} ({type(raw_text).__name__}), not a string; refusing "
            "rather than coercing it with str()"
        )
    role = _ROLE_MAP.get(str(raw_role).lower())
    if role is None:
        _refuse_exit_96(
            f"row {row_index}: unrecognized conversation role {raw_role!r}; expected "
            f"one of {sorted(set(_ROLE_MAP))}. Guessing a mapping for an unknown role "
            "name risks training 'human' turns as 'assistant' targets, so this refuses "
            "rather than guessing"
        )
    return role, raw_text


def _split_markers(text: str) -> tuple[list[dict[str, Any]], int, int]:
    """Split a human turn's text around ``<image>``/``<video>`` markers, in place.

    Returns the content blocks plus how many of each marker were found, so
    the caller can check that count against the row's declared media without
    a second pass over the text.
    """
    blocks: list[dict[str, Any]] = []
    image_count = 0
    video_count = 0
    for part in _MARKER_SPLIT_RE.split(text):
        if part == _IMAGE_MARKER:
            blocks.append({"type": "image"})
            image_count += 1
        elif part == _VIDEO_MARKER:
            blocks.append({"type": "video"})
            video_count += 1
        elif part:
            blocks.append({"type": "text", "text": part})
    if not blocks:
        # An empty human turn: keep one empty text block rather than an empty
        # content list, which some chat templates treat as a missing turn.
        blocks = [{"type": "text", "text": ""}]
    return blocks, image_count, video_count


def _media_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _declared_images(data: Mapping[str, Any], image_column: str | None) -> list[Any]:
    """``_media_list(data.get(image_column))``, but safe for ``image_column=None``.

    ``image_column=None`` means no image column is declared at all: there is
    no key to look up (``data.get(None)`` would always miss anyway, since no
    row has a literal ``None`` key), so this returns ``[]`` directly rather
    than reaching into ``data`` -- the same outcome, stated without passing a
    non-``str`` key to a ``Mapping[str, Any]``.
    """
    if image_column is None:
        return []
    return _media_list(data.get(image_column))


def normalize_conversation(
    row: Any,
    *,
    conversations_column: str,
    image_column: str | None,
    video_column: str,
    row_index: int = 0,
) -> list[dict[str, Any]]:
    """ShareGPT (or ``{role,content}``) turns -> HF content-block messages.

    ``row_index`` is caller-supplied context for refusal messages (the
    collator passes the row's real position in the batch; direct callers,
    including these tests, may leave it at its default). The row may be a
    plain mapping or a namespace-like object (``vars()`` is used in the
    latter case), matching the other train-plane collators.

    ``image_column=None`` means no image column is declared at all (the
    operator never ran ``FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN``), NOT "guess
    the column is named 'image'": ``data.get(None)`` always misses, so
    ``images`` is always empty, and a row that still carries an ``<image>``
    marker refuses below rather than silently training text-only under a
    conversations label that implied otherwise.

    Refuses (96) when:
      * a turn names a role this module does not recognize;
      * a turn's value/content is missing, None, or not a string -- never
        coerced with ``str()``;
      * an assistant turn's ``<think>``/``</think>`` tags are unbalanced;
      * the ``<image>``/``<video>`` marker count in the human turns disagrees
        with the media actually declared on the row, in EITHER direction --
        media with no markers would place pixels with no declared position,
        and markers with no media would place a block the processor can
        never fill. Never guessed, in either case.
    """
    data = _as_dict(row)
    turns = data.get(conversations_column)
    if not turns:
        _refuse_exit_96(
            f"row {row_index}: column {conversations_column!r} is missing or empty; "
            "there is no conversation to train on"
        )

    images = _declared_images(data, image_column)
    video_value = data.get(video_column)
    has_video = bool(video_value)

    messages: list[dict[str, Any]] = []
    image_marker_count = 0
    video_marker_count = 0
    for turn in turns:
        role, text = _role_and_text(turn, row_index)
        if role == "user":
            blocks, img_count, vid_count = _split_markers(text)
            image_marker_count += img_count
            video_marker_count += vid_count
            messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            think_issue = _think_tag_balance_issue(text)
            if think_issue is not None:
                _refuse_exit_96(
                    f"row {row_index}: assistant turn has {think_issue}; refusing "
                    "rather than guessing where the reasoning block begins or ends"
                )
            match = _THINK_RE.match(text)
            if match:
                reasoning = match.group(1).strip()
                content_text = match.group(2).strip()
                messages.append(
                    {
                        "role": "assistant",
                        "reasoning_content": reasoning,
                        "content": [{"type": "text", "text": content_text}],
                    }
                )
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        else:  # role == "system"
            messages.append({"role": "system", "content": [{"type": "text", "text": text}]})

    if image_marker_count != len(images):
        if image_column is None and image_marker_count > 0:
            _refuse_exit_96(
                f"row {row_index}: {image_marker_count} <image> marker(s) in the human "
                "turns, but no image column is declared "
                "(FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN is unset): declare the image column "
                "rather than training this row text-only under a conversations label "
                "that implies otherwise"
            )
        _refuse_exit_96(
            f"row {row_index}: {image_marker_count} <image> marker(s) across the human "
            f"turns but {len(images)} image(s) declared in column {image_column!r}; "
            "counts must match exactly -- this never guesses where a marker-less image "
            "or a media-less marker belongs"
        )
    expected_video_markers = 1 if has_video else 0
    if video_marker_count != expected_video_markers:
        _refuse_exit_96(
            f"row {row_index}: {video_marker_count} <video> marker(s) across the human "
            f"turns but {'a video is' if has_video else 'no video is'} declared in "
            f"column {video_column!r}; at most one <video> marker is supported per row "
            "and it must match the declaration exactly"
        )
    return messages


# ---------------------------------------------------------------------------
# derive_turn_markers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnMarkers:
    """The assistant turn's opening header and closing end-of-turn ids.

    Both are derived once per processor by :func:`derive_turn_markers` --
    never hard-coded -- because they are properties of THIS checkpoint's
    chat template, not a constant of the family name.
    """

    header_ids: tuple[int, ...]
    end_ids: tuple[int, ...]


def _render_ids(processor: Any, messages: list[dict[str, Any]]) -> list[int]:
    """Tokenize one dummy conversation; always returns a flat id list.

    MEASURED (transformers 5.18.0, both families): for a single (non-nested)
    conversation, ``apply_chat_template(..., tokenize=True, return_dict=True)``
    returns ``input_ids`` as a flat list directly, NOT batch-wrapped -- but
    this is not asserted of every processor version, so the batch-wrapped
    shape is unwrapped defensively rather than assumed absent.
    """
    out = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, add_generation_prompt=False
    )
    ids = out["input_ids"]
    if ids and isinstance(ids[0], list):
        return list(ids[0])
    return list(ids)


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    for i in range(limit):
        if left[i] != right[i]:
            return i
    return limit


def _common_suffix(left: list[int], right: list[int]) -> list[int]:
    limit = min(len(left), len(right))
    out: list[int] = []
    for i in range(1, limit + 1):
        if left[-i] != right[-i]:
            break
        out.append(left[-i])
    out.reverse()
    return out


def derive_turn_markers(processor: Any) -> TurnMarkers:
    """Derive the assistant header and end-of-turn ids from the LIVE template.

    Two dummy renders, compared, rather than one render diffed against a
    generation prompt: MEASURED (transformers 5.18.0), a generation prompt's
    tail is not a clean header -- Gemma appends an empty thought channel
    (``<|channel>thought\\n<channel|>``) and Qwen appends ``<think>\\n`` after
    the role header, because that tail is written for an in-progress
    completion, not a rendered message. Comparing two renders of an ACTUAL
    assistant message sidesteps that:

      HEADER: render ``[user1, assistant(X), user2]`` for two different
      assistant contents X. The assistant turn is deliberately NOT the last
      message, because MEASURED, Qwen3.6's template wraps only the turn
      after the last user query in ``<think>...</think>`` (``loop.index0 >
      ns.last_query_index``) -- making the assistant non-final renders it as
      plain ``<|im_start|>assistant\\n{content}``, the same shape Gemma
      already uses for a non-final turn. The common PREFIX of the two
      renders, after the identical ``user1``-only prefix, is then exactly
      the role header: everything up to where the two contents diverge is
      content-INDEPENDENT, and the two renders are built to diverge at the
      first content token.

      END: render ``[user1, assistant(X)]`` (the assistant now IS final, so
      a family that unconditionally wraps the final turn is free to do so --
      it does not matter here) for the same two contents. The common SUFFIX
      of the two renders is "whatever closes a turn regardless of content",
      which includes the template's own trailing newline after the
      end-of-turn tag. That trailing separator is dropped by keeping only
      the LEADING RUN of tokens in that suffix that the tokenizer itself
      marks special (``tokenizer.all_special_ids``) -- the plain newline
      that follows is not one, so the run stops exactly at the end-of-turn
      tag. MEASURED to equal exactly Gemma-4's ``<turn|>`` (one token) and
      Qwen3.6's ``<|im_end|>`` (one token).

    Refuses (96) if either derivation comes up empty: a family whose
    template renders a user turn differently depending on what follows it
    (breaking the header assumption) or that inserts no special token after
    assistant content (breaking the end assumption) cannot be driven by this
    module's generic method, and training against a boundary this module
    could not verify would be a silent guess.
    """
    tokenizer = processor.tokenizer
    special_ids = set(tokenizer.all_special_ids)

    user1 = {"role": "user", "content": [{"type": "text", "text": "Dummy prompt text."}]}
    user2 = {"role": "user", "content": [{"type": "text", "text": "Second prompt text."}]}
    content_a = {"role": "assistant", "content": [{"type": "text", "text": "Alpha one two three"}]}
    content_b = {"role": "assistant", "content": [{"type": "text", "text": "Beta"}]}

    ids_user1_only = _render_ids(processor, [user1])
    ids_header_a = _render_ids(processor, [user1, content_a, user2])
    ids_header_b = _render_ids(processor, [user1, content_b, user2])
    if ids_header_a[: len(ids_user1_only)] != ids_user1_only:
        _refuse_exit_96(
            f"could not derive an assistant header for processor {type(processor).__name__}: "
            "the user turn's own rendering changes depending on what follows it, so the "
            "boundary between the user turn and the assistant header cannot be located"
        )
    prefix_len = _common_prefix_len(ids_header_a, ids_header_b)
    header_ids = tuple(ids_header_a[len(ids_user1_only) : prefix_len])
    if not header_ids:
        _refuse_exit_96(
            f"could not derive a non-empty assistant header for processor "
            f"{type(processor).__name__}: two differently-worded assistant turns rendered "
            "identically up to their content, which means no role-boundary token "
            "separates the user turn from the assistant turn in this template"
        )

    ids_end_a = _render_ids(processor, [user1, content_a])
    ids_end_b = _render_ids(processor, [user1, content_b])
    suffix = _common_suffix(ids_end_a, ids_end_b)
    end_ids: list[int] = []
    for token_id in suffix:
        if token_id not in special_ids:
            break
        end_ids.append(token_id)
    if not end_ids:
        _refuse_exit_96(
            f"could not derive a non-empty end-of-turn marker for processor "
            f"{type(processor).__name__}: no special token follows the assistant content "
            "in two differently-worded renders, so no end-of-turn boundary could be found"
        )
    return TurnMarkers(header_ids=header_ids, end_ids=tuple(end_ids))


# ---------------------------------------------------------------------------
# assistant_label_mask
# ---------------------------------------------------------------------------


def _find_all(ids_row: list[int], pattern: list[int]) -> list[int]:
    n, m = len(ids_row), len(pattern)
    if m == 0 or m > n:
        return []
    return [i for i in range(n - m + 1) if ids_row[i : i + m] == pattern]


def assistant_label_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    markers: TurnMarkers,
    expected_spans: Sequence[int] | None = None,
) -> torch.Tensor:
    """Labels = assistant content only; -100 everywhere else, including padding.

    For each row, every occurrence of ``markers.header_ids`` is paired with
    the NEXT occurrence of ``markers.end_ids`` strictly after it (a
    two-pointer walk over the two occurrence lists, so a multi-turn row
    pairs each header with its own end marker rather than the last one in
    the row). The span from just after the header through the end marker,
    INCLUSIVE, is supervised; everything else starts at -100 and stays
    there, including:

      * the header and the human/system turns around it;
      * padding, masked via ``attention_mask == 0`` -- NEVER via an id
        comparison against a pad token, because Gemma's own pad id is 0 and
        0 is also a legal content token id (the #450 lesson, carried over
        from ``train.audio``'s identical rule).

    Refuses (96) when a row contains no occurrence of the header at all:
    nothing in that row could be supervised, and returning an all -100 row
    silently would be the defect this module exists to refuse rather than
    produce.

    ``expected_spans``, when given, is one int per row -- the number of
    assistant TURNS the caller's own conversation structure declared for
    that row (not read from the tokens). A mismatch against the number of
    header/end spans actually found refuses: a row whose human or system
    turn happens to quote the literal header/end-marker text (e.g. pasted
    chat-template syntax) can make this function find MORE spans than the
    row actually has assistant turns, silently supervising text nobody
    meant to supervise. Optional and defaulted to None so direct callers
    (including existing tests) that only have hand-built ids, with no
    conversation structure to compare against, are unaffected.
    """
    import torch  # noqa: PLC0415

    header = list(markers.header_ids)
    end = list(markers.end_ids)
    batch_size, _width = input_ids.shape
    labels = torch.full_like(input_ids, -100)

    for row in range(batch_size):
        ids_row = input_ids[row].tolist()
        header_starts = _find_all(ids_row, header)
        if not header_starts:
            _refuse_exit_96(
                f"row {row}: no occurrence of the assistant header {tuple(header)} found "
                "in this row's tokens; nothing in it can be supervised. This usually means "
                "the row's rendered text never reached an assistant turn, or this "
                "processor's template does not match the header derived for it"
            )
        end_starts = _find_all(ids_row, end)
        end_index = 0
        spans_found = 0
        for header_start in header_starts:
            content_start = header_start + len(header)
            while end_index < len(end_starts) and end_starts[end_index] < content_start:
                end_index += 1
            if end_index >= len(end_starts):
                break
            span_end = end_starts[end_index] + len(end)
            labels[row, content_start:span_end] = input_ids[row, content_start:span_end]
            end_index += 1
            spans_found += 1
        if spans_found == 0:
            _refuse_exit_96(
                f"row {row}: the assistant header {tuple(header)} occurs, but no "
                f"end-of-turn marker {tuple(end)} follows any occurrence; no span could "
                "be closed and nothing in this row can be supervised"
            )
        if expected_spans is not None and spans_found != expected_spans[row]:
            _refuse_exit_96(
                f"row {row}: found {spans_found} supervised span(s) but this row's "
                f"conversation declares {expected_spans[row]} assistant turn(s); a "
                "mismatch means a human or system turn's text likely quotes the "
                "literal header/end-of-turn marker, which would otherwise supervise "
                "text nobody meant to supervise. Refusing rather than training on it"
            )

    labels[attention_mask == 0] = -100
    return labels


# ---------------------------------------------------------------------------
# Row rendering, shared by the collator AND the distributed-safety pre-pass
# ---------------------------------------------------------------------------


def _render_conversation_row(
    index: int,
    row: Any,
    *,
    conversations_column: str,
    image_column: str | None,
    video_column: str,
    processor: Any,
    frames_for: Callable[[Mapping[str, Any]], dict[str, Any]] | None,
) -> tuple[str, list[Any], list[Any], int]:
    """Render one row to ``(text, loaded_images, video_frames, assistant_turn_count)``.

    The ONE rendering path both :func:`train_conversation_collator_or_refuse`
    and :func:`conversation_prepass_or_refuse` call -- a row's prepass-measured
    length is the row's collate-time length because they are the same
    computation, not two implementations kept in sync by hand.
    """
    data = _as_dict(row)
    messages = normalize_conversation(
        row,
        conversations_column=conversations_column,
        image_column=image_column,
        video_column=video_column,
        row_index=index,
    )
    from foundationscale.rl.prompt_surface import _load_image_or_refuse  # noqa: PLC0415

    image_paths = _declared_images(data, image_column)
    loaded_images = [_load_image_or_refuse(f"row[{index}]", str(p)) for p in image_paths]

    video_value = data.get(video_column)
    video_frames: list[Any] = []
    if video_value:
        if frames_for is None:
            _refuse_exit_96(
                f"row {index}: carries a video in column {video_column!r} but no frame "
                "budget was declared (frames_for=None); declare a frame budget or drop "
                "the video column. This never guesses how many frames to decode"
            )
        frame_info = frames_for(data)
        video_frames = list(frame_info["frames"])

    assistant_turn_count = sum(1 for m in messages if m["role"] == "assistant")
    has_reasoning = any("reasoning_content" in m for m in messages if m["role"] == "assistant")
    text = str(
        processor.apply_chat_template(messages, tokenize=False, enable_thinking=has_reasoning)
    )
    return text, loaded_images, video_frames, assistant_turn_count


def _conversation_processor_batch_call(
    processor: Any,
    texts: list[str],
    images_by_row: list[list[Any]],
    videos_by_row: list[list[Any]],
    *,
    max_length: int,
    pad_to_max: bool,
) -> Any:
    """ONE ``processor(...)`` call for a whole batch of rendered rows.

    See :func:`train_conversation_collator_or_refuse`'s docstring for the two
    MEASURED media-kwarg shapes this depends on (nested per-row lists when
    any row carries media, the kwarg OMITTED entirely when none do).
    """
    kwargs: dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding_side": "right"}
    if pad_to_max:
        kwargs["padding"] = "max_length"
        kwargs["max_length"] = max_length
    else:
        kwargs["padding"] = True
    if any(images_by_row):
        kwargs["images"] = images_by_row
    if any(videos_by_row):
        kwargs["videos"] = videos_by_row
        kwargs["do_sample_frames"] = False
    return processor(**kwargs)


# ---------------------------------------------------------------------------
# train_conversation_collator_or_refuse
# ---------------------------------------------------------------------------


def train_conversation_collator_or_refuse(
    processor: Any,
    *,
    conversations_column: str = "conversations",
    image_column: str | None = "image",
    video_column: str = "video",
    max_length: int,
    overlong: str = "drop",
    pad_to_max_length: bool = False,
    frames_for: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    inject_dummy_media: bool = True,
) -> Any:
    """Build the conversation TRAIN collator; refuse (96) at construction on a bad declaration.

    Per row: :func:`normalize_conversation`, then
    ``processor.apply_chat_template(messages, tokenize=False,
    enable_thinking=<True iff this row carries reasoning>)`` -- per-row, not
    per-batch, because whether the system turn declares thinking support is
    a property of what THIS row's assistant turns actually contain. The
    whole batch is then encoded in ONE ``processor(...)`` call (never one
    call per row -- a second call per row is exactly the silent-drop risk
    this family of collator exists to refuse), with two MEASURED shapes for
    the media kwargs (transformers 5.18.0, both families):

      * at least one row in the batch carries an image: ``images=`` is a
        list with ONE SUB-LIST PER ROW, empty for a row with none (never a
        flat list -- a flat list desyncs the image count from the row count
        the moment one row has more than one image, or when any row has
        zero, exactly the failure ``rl.prompt_surface.encode_prompts``
        documents for the sibling image collator).
      * NO row in the batch carries an image: the ``images`` kwarg is
        OMITTED entirely. MEASURED: passing ``images=[[], []]`` for an
        all-text batch raises (``stack expects a non-empty TensorList`` on
        Gemma-4, ``IndexError`` on Qwen3.6) on both families -- there is no
        nested-empty-lists shape a text-only batch can pass. Videos follow
        the identical shape, with ``do_sample_frames=False`` added whenever
        any row carries one (MEASURED: without it, Gemma-4 tries to
        re-sample a hard-coded frame count and raises when the supplied clip
        is shorter).

    Length is measured and padding applied via TWO processor calls at most,
    never more: a dynamic-padding "measure" pass over every row's own
    encoded length (``attention_mask.sum()`` per row, valid for any padding
    side) decides which rows are overlong; a second pass with only the
    surviving rows (and, if requested, a forced ``padding="max_length"``) is
    skipped entirely when nothing needed to change -- the common case costs
    one call. An overlong row (measured AFTER media expansion, since a video
    or several images can turn one marker into thousands of ids) is dropped
    (counted in ``collate.stats["dropped_overlong"]``) or refused per
    ``overlong``; this path NEVER truncates, because truncating a row that
    carries media placeholders can cut mid-image or drop a turn's own end
    marker. Padding is forced to the RIGHT (``padding_side="right"``) at
    call time regardless of the tokenizer's own default -- Gemma-4's
    tokenizer defaults to left padding, which the right-padded labelling and
    assistant-turn search this module performs assumes is not in force.

    This overlong handling is this function's OWN assertion backstop, not
    the primary enforcement point: train/loop.py calls
    :func:`conversation_prepass_or_refuse` on the whole dataset BEFORE the
    Trainer (and therefore this collator) ever runs, so every row this
    collator sees under a correctly-wired caller is already measured and
    already decided. A per-rank, per-batch decision here -- one rank
    dropping or refusing a row while its FSDP peers do not -- is exactly the
    collective-hang and wrong-exit-code failure the pre-pass exists to move
    earlier and make identical across ranks; this code path staying in place
    is what catches a caller that skipped the pre-pass, not what this plane
    relies on in normal operation.

    When a batch has NO media at all and ``inject_dummy_media`` is true, one
    minimal dummy image is spliced INTO an existing row rather than appended
    as an extra batch row -- LLaMA-Factory's ``fake_input_ids`` pattern
    (``MultiModalDataCollatorForSeq2Seq.__call__``), adapted to this
    module's one-coherent-processor-call design: LLaMA-Factory tokenizes
    its fake turn standalone and splices the raw ids onto ``features[0]``
    before its own padding collator runs; this module instead appends one
    ``{"type": "image"}`` block to the END of the chosen row's OWN LAST USER
    TURN and re-renders that one row through the SAME batched
    ``processor(...)`` call used for the rest of the batch, so the
    pixel/grid/token-type tensors for the dummy come from the real
    processor and are automatically consistent with the rest of the batch
    -- no hand-merged tensors. Appending within the existing user turn
    (never as a NEW trailing turn) is deliberate: a new turn would change
    which assistant turn is "final" under Qwen3.6's reasoning-stripping
    rule (``last_query_index``), silently stripping a real row's own
    supervised reasoning. The candidate row is the SHORTEST surviving row
    first (least padding burden); if the dummy's own expansion would push
    it past ``max_length``, the next-shortest row is tried, and the batch
    is refused (96) only if no row has room. The exact inserted span is
    located per row by diffing the row's rendered ids against the SAME
    row's own dummy-free baseline (common prefix/suffix, the identical
    technique :func:`derive_turn_markers` uses) rather than computed from a
    constant, so it is exact regardless of where in the row's own text the
    insertion landed. The span's ``attention_mask`` is forced to zero
    before labelling, so :func:`assistant_label_mask`'s own padding rule
    masks it to -100 with no separate code path, and because it sits inside
    a USER turn it was never going to be supervised anyway -- masking
    attention only stops it being ATTENDED to by the row's own real
    content. Because nothing about the row's real turns, roles or counts
    changes, every row's real supervised labels are byte-identical to a
    no-dummy collation of the same batch, INCLUDING the row that hosts the
    dummy. This keeps the vision tower in every rank's forward pass even on
    a rank whose batch happens to be all-text -- without doubling the
    row count, or (under ``pad_to_max_length``) the compute/activation
    cost of a full extra ``max_length``-wide row.

    A row that declares a video but no ``frames_for`` is refused (96) the
    moment that row is seen, BEFORE the processor is touched at all -- the
    declaration alone is enough to know the row cannot be collated, and this
    module does not guess a frame budget. When ``frames_for`` IS given, its
    ``{"frames": [...], "fps": ..., "frames_indices": [...], "duration":
    ...}`` result is passed as a native ``videos=`` input (frames only; the
    real decoder, timestamp math and ``video_metadata`` plumbing are owned
    by a separate piece of work. This is a basic path, not the full video
    contract).
    """
    if overlong not in ("drop", "refuse"):
        _refuse_exit_96(
            f"overlong={overlong!r} is not one of 'drop'/'refuse'; refusing rather than "
            "silently treating an unknown policy as one of the two"
        )

    markers_cache: list[TurnMarkers | None] = [None]

    def get_markers() -> TurnMarkers:
        cached = markers_cache[0]
        if cached is None:
            cached = derive_turn_markers(processor)
            markers_cache[0] = cached
        return cached

    stats: dict[str, int] = {"rows_seen": 0, "dropped_overlong": 0, "dummy_media_batches": 0}

    def _render_row(index: int, row: Any) -> tuple[str, list[Any], list[Any], int]:
        return _render_conversation_row(
            index,
            row,
            conversations_column=conversations_column,
            image_column=image_column,
            video_column=video_column,
            processor=processor,
            frames_for=frames_for,
        )

    def _row_with_dummy_image(index: int, row: Any) -> str:
        """Re-render one row's text with one dummy image block appended to its LAST user turn.

        Only the marker matters here, not pixel data -- ``apply_chat_template``
        renders ``{"type": "image"}`` into the family's placeholder text from
        the message structure alone; the actual dummy ``PIL.Image`` is only
        needed later, when the modified text is run back through the real
        ``processor(...)`` call that expands the placeholder and produces
        pixel/grid tensors.

        Appending to an EXISTING user turn's own content -- rather than a
        new trailing turn -- changes no turn's role, position or count, so
        every other token in the row (including a real reasoning span in
        its own final assistant turn) renders exactly as it would with no
        dummy at all. Refuses (96) only in the pathological case of a row
        with no user turn at all to attach to -- normalize_conversation
        does not produce such rows from a valid conversation.
        """
        messages = normalize_conversation(
            row,
            conversations_column=conversations_column,
            image_column=image_column,
            video_column=video_column,
            row_index=index,
        )
        for message in reversed(messages):
            if message["role"] == "user":
                message["content"] = [*message["content"], {"type": "image"}]
                break
        else:
            _refuse_exit_96(
                f"row {index}: inject_dummy_media=True but this row's conversation has no "
                "user turn to attach a dummy image to"
            )
        has_reasoning = any("reasoning_content" in m for m in messages if m["role"] == "assistant")
        return str(
            processor.apply_chat_template(messages, tokenize=False, enable_thinking=has_reasoning)
        )

    def _inject_dummy_media(
        rows: list[Any],
        survivors: list[int],
        texts: list[str],
        final_texts: list[str],
        final_images: list[list[Any]],
        final_videos: list[list[Any]],
        lengths: list[float],
    ) -> tuple[Any, int, tuple[int, int]]:
        """Splice one dummy image into whichever surviving row has room.

        Tries rows shortest-first; each trial re-renders ONLY that row and
        re-runs the full batched processor call (cheap: this only triggers
        on an all-text batch, and almost always succeeds on the first,
        shortest candidate). The dummy's exact span in the chosen row is
        found by diffing that row's own dummy-free baseline ids against its
        post-insertion ids (common prefix + common suffix): the two renders
        are identical everywhere except the inserted block, so what is left
        after removing the shared prefix and suffix IS the span, regardless
        of where in the row's own text the image landed.
        """
        from PIL import Image  # noqa: PLC0415

        dummy_image = Image.new("RGB", (32, 32), color=(0, 0, 0))
        order = sorted(range(len(survivors)), key=lambda local_idx: lengths[survivors[local_idx]])
        for local_idx in order:
            original_index = survivors[local_idx]
            modified_text = _row_with_dummy_image(original_index, rows[original_index])
            trial_texts = list(final_texts)
            trial_images = list(final_images)
            trial_texts[local_idx] = modified_text
            trial_images[local_idx] = [dummy_image]
            trial = _batch_call(
                trial_texts, trial_images, final_videos, pad_to_max=pad_to_max_length
            )
            real_length = int(trial["attention_mask"][local_idx].sum())
            if real_length > max_length:
                continue
            baseline_ids = [
                int(t)
                for t in processor.tokenizer(texts[original_index], add_special_tokens=False)[
                    "input_ids"
                ]
            ]
            modified_ids = [int(t) for t in trial["input_ids"][local_idx][:real_length].tolist()]
            prefix_len = _common_prefix_len(modified_ids, baseline_ids)
            suffix = _common_suffix(modified_ids, baseline_ids)
            span = (prefix_len, real_length - len(suffix))
            return trial, local_idx, span
        _refuse_exit_96(
            f"inject_dummy_media=True but no row in this batch has room for the dummy media "
            f"within max_length={max_length}; every candidate row would exceed it"
        )

    def _batch_call(
        texts: list[str],
        images_by_row: list[list[Any]],
        videos_by_row: list[list[Any]],
        *,
        pad_to_max: bool,
    ) -> Any:
        return _conversation_processor_batch_call(
            processor,
            texts,
            images_by_row,
            videos_by_row,
            max_length=max_length,
            pad_to_max=pad_to_max,
        )

    def collate(features: Sequence[Any]) -> dict[str, Any]:
        rows = list(features)
        stats["rows_seen"] += len(rows)
        if not rows:
            _refuse_exit_96("the collator received an empty batch of rows; nothing to train on")

        texts: list[str] = []
        images_by_row: list[list[Any]] = []
        videos_by_row: list[list[Any]] = []
        assistant_turn_counts: list[int] = []
        for index, row in enumerate(rows):
            text, images, videos, assistant_turns = _render_row(index, row)
            texts.append(text)
            images_by_row.append(images)
            videos_by_row.append(videos)
            assistant_turn_counts.append(assistant_turns)

        # Pass 1: measure every row's own expanded length (dynamic padding;
        # the padded WIDTH here is irrelevant, only the per-row
        # attention_mask sum is read).
        measured = _batch_call(texts, images_by_row, videos_by_row, pad_to_max=False)
        lengths = measured["attention_mask"].sum(dim=-1).tolist()
        overlong_indices = [i for i, length in enumerate(lengths) if length > max_length]

        if overlong_indices:
            if overlong == "refuse":
                first = overlong_indices[0]
                _refuse_exit_96(
                    f"{len(overlong_indices)} row(s) exceed max_length={max_length} after "
                    f"media expansion (first: row {first}, {int(lengths[first])} tokens); this "
                    "path never truncates (truncation could cut media tokens or an "
                    "end-of-turn marker), so the run refuses rather than corrupts"
                )
            survivors = [i for i in range(len(rows)) if i not in set(overlong_indices)]
            stats["dropped_overlong"] += len(overlong_indices)
            if not survivors:
                _refuse_exit_96(
                    f"all {len(rows)} row(s) in this batch exceeded max_length={max_length} "
                    "after media expansion under overlong='drop'; nothing survives to train on"
                )
        else:
            survivors = list(range(len(rows)))

        final_texts = [texts[i] for i in survivors]
        final_images = [images_by_row[i] for i in survivors]
        final_videos = [videos_by_row[i] for i in survivors]
        needs_second_pass = bool(overlong_indices) or pad_to_max_length

        dummy_row_index: int | None = None
        dummy_span: tuple[int, int] | None = None
        if inject_dummy_media and not any(final_images) and not any(final_videos):
            final, dummy_row_index, dummy_span = _inject_dummy_media(
                rows, survivors, texts, final_texts, final_images, final_videos, lengths
            )
            stats["dummy_media_batches"] += 1
        elif needs_second_pass:
            final = _batch_call(
                final_texts, final_images, final_videos, pad_to_max=pad_to_max_length
            )
        else:
            final = measured

        input_ids = final["input_ids"]
        attention_mask = final["attention_mask"]
        if dummy_row_index is not None and dummy_span is not None:
            start, end = dummy_span
            attention_mask[dummy_row_index, start:end] = 0

        markers = get_markers()
        final_expected_spans = [assistant_turn_counts[i] for i in survivors]
        labels = assistant_label_mask(
            input_ids, attention_mask, markers, expected_spans=final_expected_spans
        )

        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        video_token_id = getattr(processor, "video_token_id", None)
        if video_token_id is not None:
            labels[labels == video_token_id] = -100

        out = dict(final)
        out["attention_mask"] = attention_mask
        out["labels"] = labels
        return out

    collate.stats = stats  # type: ignore[attr-defined]
    return collate


# ---------------------------------------------------------------------------
# conversation_prepass_or_refuse -- distributed-safety pre-pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationPrepassResult:
    """Outcome of :func:`conversation_prepass_or_refuse`.

    ``refusal_reason`` is None on success, naming the overlong row and its
    measured length when ``overlong="refuse"`` found one. It is the ONLY
    failure mode this function reports rather than raising: every row-level
    validation failure (bad conversation structure, a missing image file)
    raises ``SystemExit(96)`` directly, the same mechanism every other
    declaration-time refusal in this plane already uses. ``overlong="refuse"``
    is reported back instead because train/loop.py is the one that knows how
    to turn it into its own manifest-aware refusal (``_mark``/
    ``_emit_manifest``/``EXIT_REFUSE``), which this module -- deliberately
    ignorant of loop.py and the manifest -- cannot do itself.

    ``filtered_dataset`` is the caller's dataset with every overlong row
    removed under ``overlong="drop"`` (identical to the input when nothing
    was dropped), or None when ``refusal_reason`` is set (there is nothing
    valid to train on).
    """

    refusal_reason: str | None
    filtered_dataset: Any | None
    rows_seen: int
    dropped_overlong: int
    kept: int
    seconds: float


def conversation_prepass_or_refuse(
    dataset: Any,
    processor: Any,
    *,
    conversations_column: str,
    image_column: str | None,
    video_column: str = "video",
    max_length: int,
    overlong: str,
    frames_for: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    num_proc: int | None = None,
    batch_size: int = 16,
) -> ConversationPrepassResult:
    """Validate and measure EVERY row before a single training step is paid for.

    This exists because the collator's own checks run PER RANK, AT COLLATE
    TIME, inside ``trainer.train()``: one rank can raise or drop a row while
    its peers do not, which under FSDP's per-step collectives is a hang
    (the healthy ranks block in an all-reduce the refusing rank never
    reaches) rather than a clean, diagnosable exit -- and a refusal raised
    from inside the Trainer's own call stack was measured to surface as exit
    1, not 96, losing the exit-contract's own vocabulary for "declaration
    rejected". Calling this BEFORE ``Trainer`` is constructed moves every one
    of those failures to the same place and the same exit code every other
    declaration-time refusal in this plane already uses.

    For every row: :func:`_render_conversation_row` (which runs
    :func:`normalize_conversation` -- structure, role, think-tag and
    marker/media-count checks -- and loads every declared image file,
    refusing on a missing one), then ONE single-row ``processor(...)`` call
    to measure the row's fully-expanded token length -- the SAME rendering
    and the SAME processor call shape :func:`train_conversation_collator_or_refuse`
    uses, so a length measured here is the length collation would measure,
    never an approximation of it.

    DISTRIBUTED SAFETY: this is a pure function of row content and the
    declared knobs (conversations/image/video column, max_length, overlong,
    frames_for) -- no randomness anywhere in it -- so calling it
    independently on every rank, with NO collective, yields the identical
    kept/dropped decision on every one of them. That is also what makes it
    safe to call where this plane calls it, in train/loop.py's dataset-
    construction step, BEFORE ``TrainingArguments`` builds accelerate's
    process group: there is no process group yet to synchronize a
    rank-0-decides-first scheme over, and this needs none.

    ``num_proc`` trades a narrower failure-message guarantee for speed on a
    large corpus. A worker process raising ``SystemExit`` is not a reliable
    way to fail ``datasets.map`` (the pool machinery was not built to treat a
    worker's clean exit as "the whole call failed"), so with ``num_proc > 1``
    each worker catches its OWN row failures as data instead of letting them
    escape, and -- only if at least one row failed -- the FIRST offending row
    is re-rendered, UNPROTECTED, back on the calling process: the exact same
    validation runs again, un-caught, producing the identical message and
    ``SystemExit(96)`` a single-process pre-pass would have raised directly,
    at the cost of repeating that one row's work. ``num_proc=None`` (the
    default) runs single-process, where the first failure already raises
    authentically with no replay needed.
    """
    import time  # noqa: PLC0415

    start = time.monotonic()
    is_parallel = num_proc is not None and num_proc > 1

    def _measure_batch(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        n = len(indices)
        lengths = [0] * n
        failed = [False] * n
        columns = list(batch)
        for i in range(n):
            row = {key: batch[key][i] for key in columns}
            idx = indices[i]
            try:
                text, images, videos, _turns = _render_conversation_row(
                    idx,
                    row,
                    conversations_column=conversations_column,
                    image_column=image_column,
                    video_column=video_column,
                    processor=processor,
                    frames_for=frames_for,
                )
            except SystemExit:
                if is_parallel:
                    failed[i] = True
                    continue
                raise  # single-process: let the authentic refusal propagate
            result = _conversation_processor_batch_call(
                processor,
                [text],
                [images],
                [videos],
                max_length=max_length,
                pad_to_max=False,
            )
            lengths[i] = int(result["attention_mask"][0].sum())
        return {"__fs_prepass_length": lengths, "__fs_prepass_failed": failed}

    mapped = dataset.map(
        _measure_batch,
        batched=True,
        batch_size=batch_size,
        with_indices=True,
        num_proc=num_proc,
        desc="conversation pre-pass (validate + measure length)",
    )

    if is_parallel and any(mapped["__fs_prepass_failed"]):
        first_failed = mapped["__fs_prepass_failed"].index(True)
        # Unprotected replay on the calling process: always raises
        # SystemExit(96) for real, with the authentic message, for a row
        # this pass already confirmed fails validation.
        _render_conversation_row(
            first_failed,
            dataset[first_failed],
            conversations_column=conversations_column,
            image_column=image_column,
            video_column=video_column,
            processor=processor,
            frames_for=frames_for,
        )
        raise AssertionError(  # pragma: no cover -- the replay above always raises
            f"conversation_prepass_or_refuse: row {first_failed} was flagged as failing "
            "during the parallel pass but its unprotected replay did not raise"
        )

    lengths: list[int] = mapped["__fs_prepass_length"]
    rows_seen = len(lengths)
    overlong_indices = [i for i, length in enumerate(lengths) if length > max_length]
    seconds = time.monotonic() - start

    if overlong_indices and overlong == "refuse":
        first = overlong_indices[0]
        return ConversationPrepassResult(
            refusal_reason=(
                f"{len(overlong_indices)} row(s) exceed max_length={max_length} after "
                f"media expansion (first: row {first}, {lengths[first]} tokens), measured "
                "in the pre-pass BEFORE a single training step was paid for; this path "
                "never truncates (truncation could cut media tokens or an end-of-turn "
                "marker), so the run refuses rather than corrupts"
            ),
            filtered_dataset=None,
            rows_seen=rows_seen,
            dropped_overlong=len(overlong_indices),
            kept=0,
            seconds=seconds,
        )

    overlong_set = set(overlong_indices)
    kept_indices = [i for i in range(rows_seen) if i not in overlong_set]
    filtered = dataset.select(kept_indices) if overlong_indices else dataset
    return ConversationPrepassResult(
        refusal_reason=None,
        filtered_dataset=filtered,
        rows_seen=rows_seen,
        dropped_overlong=len(overlong_indices),
        kept=len(kept_indices),
        seconds=seconds,
    )
