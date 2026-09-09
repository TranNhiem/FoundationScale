"""ShareGPT corpus loading: one schema in, one typed :class:`Sample` plane out.

The RL trainer consumes a uniform sequence of prompt/response pairs; the
user's corpus arrives as nineteen ShareGPT-shaped JSON datasets whose records
share a single schema:

    {id, conversations: [{from: "human"|"gpt", value: str}], ...}

with optional ``system``, ``image`` (str or list of str), and ``video`` keys.
Text-only, image-text, multi-image, and video records differ only in which
extra keys they carry; the conversational core is identical, and that core is
all this module reads. Keys a record carries that the training plane does not
consume (``license``, ``width``, ``height_list``, ``_meta``, ``category``,
``open_ended_qa``, ``close_ended_qa`` and friends) are ignored, not modelled.

Multiple-choice-question datasets put the verifiable answer in the final
assistant turn: an optional ``<think>...</think>`` block followed by a single
letter A--Z. :func:`extract_mcq_gold` recovers it.

WHAT IS CLAIMED: a record shaped as declared above is parsed into a Sample;
the MCQ gold letter is returned only when it is unambiguous.

WHAT IS NOT CLAIMED: no tokenisation, no image/video decoding, no schema
violations beyond missing required keys are diagnosed -- malformed values fail
closed at the point they are structurally wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from foundationscale.rl.interfaces import BatchRefusal

__all__ = (
    "Sample",
    "extract_mcq_gold",
    "load_sharegpt",
)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LETTER_RE = re.compile(r"[A-Z]")
_ROLE_MAP = {"human": "user", "gpt": "assistant"}


def extract_mcq_gold(question: str, answer: str) -> str | None:
    """Extract the single-letter MCQ gold from an assistant turn, or abstain.

    MCQ questions in this corpus end "Answer with a single letter." and the
    assistant turn holds an optional ``<think>...</think>`` block followed by
    the letter. The think block is stripped FIRST: reasoning rehearses
    candidate letters, and counting letters inside it would see ghosts the
    writer discarded. Outside the block the answer is scanned for standalone
    letters A--Z.

    Return the letter ONLY when exactly one candidate is found. Any other
    count -- zero candidates, or two or more -- is an ambiguity, and an
    ambiguous gold is an ABSTENTION, never a guess. A guessed gold trains
    the model backwards against a wrong target while reporting a clean
    accuracy; an abstained gold costs one row of supervision and trains
    nothing at all. Silence is recoverable; silent wrongness is not.

    The ``question`` is accepted so the verifier can refuse MCQ-shaped gold
    extraction from a non-MCQ record: a record whose question does not ask
    for a single letter has no verifiable letter to extract, and claims none.

    returns None on ANY ambiguity.
    """
    if not question.rstrip().endswith("Answer with a single letter."):
        return None
    stripped = _THINK_BLOCK_RE.sub(" ", answer)
    candidates = tuple(_LETTER_RE.findall(stripped.upper()))
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) != 1:
        return None
    return unique[0]


@dataclass(frozen=True, slots=True)
class Sample:
    """One parsed corpus record: prompt turns, the final reply, an optional gold.

    ``prompt_turns`` is ``(role, text)`` with role in
    ``{"system", "user", "assistant"}``; a ``system`` key becomes the first
    turn. ``response`` is the FINAL assistant turn -- in one-turn records the
    reply to the (only) human turn; in multi-turn records the final reply,
    with earlier turns retained in ``prompt_turns`` as context.

    ``gold`` is the extracted MCQ letter, else ``None``. Absence is the
    PERMANENT state of open-ended records, not a missing value to repair.
    """

    sample_id: str
    prompt_turns: tuple[tuple[str, str], ...]
    response: str
    gold: str | None
    images: tuple[str, ...] = ()
    video: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_turns", tuple(self.prompt_turns))
        object.__setattr__(self, "images", tuple(self.images))
        for role, _text in self.prompt_turns:
            if role not in ("system", "user", "assistant"):
                raise BatchRefusal(
                    f"sample {self.sample_id!r} carries prompt role {role!r}; "
                    f'only ("system", "user", "assistant") are admitted'
                )


def _parse_record(record: object, index: int) -> Sample:
    if not isinstance(record, dict):
        raise BatchRefusal(
            f"record {index} is {type(record).__name__}; every corpus record "
            f'must be a JSON object with "id" and "conversations"'
        )
    sample_id = record.get("id", f"record-{index}")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        offered = len(conversations) if isinstance(conversations, list) else 0
        raise BatchRefusal(
            f"record {index} ({sample_id!r}) carries {offered} of at least 1 required "
            f"conversation turn(s); a record with no turns supervises nothing"
        )
    turns: list[tuple[str, str]] = []
    system = record.get("system")
    if system is not None:
        if not isinstance(system, str):
            raise BatchRefusal(
                f"record {index} ({sample_id!r}) has a non-string system key "
                f"({type(system).__name__}); the system turn must be text"
            )
        turns.append(("system", system))
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            raise BatchRefusal(
                f"record {index} ({sample_id!r}) turn {turn_index} is "
                f"{type(turn).__name__}; every turn must be an object with "
                f'"from" and "value"'
            )
        speaker = turn.get("from")
        if speaker not in _ROLE_MAP:
            raise BatchRefusal(
                f"record {index} ({sample_id!r}) turn {turn_index} has speaker "
                f'{speaker!r}; only ("human", "gpt") appear in this corpus'
            )
        value = turn.get("value")
        if not isinstance(value, str):
            raise BatchRefusal(
                f"record {index} ({sample_id!r}) turn {turn_index} has a "
                f"non-string value ({type(value).__name__}); a turn's value "
                f"must be text"
            )
        turns.append((_ROLE_MAP[speaker], value))
    if not turns or turns[-1][0] != "assistant":
        raise BatchRefusal(
            f"record {index} ({sample_id!r}) ends on a "
            f"{turns[-1][0] if turns else 'absent'} turn; the FINAL turn must "
            f"be the assistant reply the policy is trained toward"
        )
    *prompt_turns, (_role, response) = turns
    question = ""
    for role, text in prompt_turns:
        if role == "user":
            question = text
    images: tuple[str, ...] = ()
    image = record.get("image")
    if isinstance(image, str):
        images = (image,)
    elif isinstance(image, list):
        images = tuple(item for item in image if isinstance(item, str))
    video = record.get("video")
    if video is not None and not isinstance(video, str):
        raise BatchRefusal(
            f"record {index} ({sample_id!r}) has a non-string video key "
            f"({type(video).__name__}); a video reference must be text"
        )
    return Sample(
        sample_id=str(sample_id),
        prompt_turns=tuple(prompt_turns),
        response=response,
        gold=extract_mcq_gold(question, response),
        images=images,
        video=video,
    )


def _read_records(file_path: Path) -> list[object]:
    """Read one corpus file as either a JSON array or JSON Lines.

    BOTH shapes are real. The suffix chooses the parser rather than a
    content sniff, because a sniff cannot tell a single-line JSON array
    from a one-record JSONL file, and guessing wrong on a large file is
    an expensive way to learn that.
    """
    if file_path.suffix == ".jsonl":
        records: list[object] = []
        with file_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise BatchRefusal(
                        f"corpus file {str(file_path)!r} line {line_number} is "
                        f"not valid JSON ({exc.msg}); a JSON Lines corpus "
                        f"carries exactly one record per non-empty line"
                    ) from exc
        return records
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise BatchRefusal(
            f"corpus file {str(file_path)!r} holds a top-level "
            f"{type(payload).__name__}; a .json corpus is a list of records "
            f"(line-delimited records belong in a .jsonl file)"
        )
    return list(payload)


def load_sharegpt(path: str | Path, *, limit: int | None = None) -> tuple[Sample, ...]:
    """Load a ShareGPT-shaped JSON corpus into typed Samples.

    ``path`` may name a single ``.json`` file holding a list of records, a
    single ``.jsonl`` file holding one record per line, or a directory of
    such files read in sorted order for determinism (``.json`` first, then
    ``.jsonl``). ``limit`` caps the number of samples returned, taking the
    first ``limit`` in corpus order.

    BOTH suffixes are supported because the measured corpus is JSON Lines:
    a directory scan that admitted only ``.json`` found zero files in every
    dataset directory and refused a corpus that was present and readable.

    Refuses (BatchRefusal) on: a missing path, a top-level shape that is
    neither list nor directory of lists, or any malformed record -- the
    refusing message names which record and which term failed. An EMPTY
    loaded set is refused as vacuous: a trainer with zero prompts measures
    nothing, and ``0`` samples is not a degenerate run but a broken path.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise BatchRefusal(f"corpus path {str(path)!r} does not exist")
    records: list[object] = []
    files: tuple[Path, ...]
    if resolved.is_dir():
        files = tuple(sorted(resolved.glob("*.json"))) + tuple(sorted(resolved.glob("*.jsonl")))
        if not files:
            raise BatchRefusal(
                f"corpus directory {str(path)!r} holds 0 of at least 1 required "
                f".json or .jsonl file(s); no records can be read"
            )
    else:
        files = (resolved,)
    for file_path in files:
        records.extend(_read_records(file_path))
    if not records:
        raise BatchRefusal(
            f"corpus {str(path)!r} yielded 0 of at least 1 required record(s); "
            f"an empty corpus supervises nothing"
        )
    samples = tuple(_parse_record(record, i) for i, record in enumerate(records))
    if limit is not None:
        if limit < 1:
            raise BatchRefusal(
                f"limit {limit} is below 1 of at least 1 required sample(s); "
                f"a zero limit requests a vacuous load"
            )
        samples = samples[:limit]
    return samples
