"""Measured-corpus conversion: estate ``conversations`` records into loop rows.

The corpus contract is measured, not assumed: eight distinct key sets, three
legitimate ``conversations[].value`` shapes, and a media layer where "exists on
disk" and "decodes" are different measured facts. This module converts what it
recognises, drops what it cannot under a NAMED bucket, and refuses -- as text,
never as an exception -- whenever the coverage arithmetic would make rc 0 a
lie. A partial corpus is a legitimate caller decision; an INVISIBLE partial
corpus is not, so every result carries what was inspected and what was dropped.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from foundationscale.families.registry import FamilySpec

__all__ = [
    "CorpusResult",
    "CorpusRow",
    "ThinkPolicy",
    "convert_records",
]


class ThinkPolicy(Enum):
    """The declared fate of ``<think>...</think>`` content in assistant turns.

    This is an enum rather than a bool because the estate has already paid for
    the accidental version: the FoxBrain SFT path stripped reasoning
    unconditionally, so CoT and non-CoT records became byte-identical at the
    model input and the difference was discovered by reading the template, not
    by any announcement. Whatever this converter does to reasoning, it does on
    purpose, and the policy is named in every result.
    """

    STRIP = "strip"
    KEEP = "keep"


@dataclass(frozen=True)
class CorpusRow:
    """One converted record: the loop's required ``text`` plus its media payload.

    ``record_id`` keeps the corpus's OWN type: the estate's eight key sets carry
    both int and str ids, and round 1's unconditional ``str()`` retype is exactly
    the kind of silent coercion that makes two distinct key sets collide once a
    downstream stage joins on id. Media stays OUT of ``text`` as resolved
    absolute paths; folding pixel bytes into a string column is what finding #521
    blocked, and re-encoding a path into the text would recreate the T1-22 shape
    -- a column nobody drops, referenced by a marker nobody resolves.
    """

    record_id: str | int
    text: str
    image_paths: tuple[str, ...]
    video_paths: tuple[str, ...]


@dataclass(frozen=True)
class CorpusResult:
    """Payload, coverage arithmetic, and the reason it stopped -- one object.

    ``refusal`` is a STRING (printed next to rc 96 upstream), never an exception
    and never None-on-error: a converter that raises moves the refusal into a
    traceback the gate cannot quote. ``inspected`` and ``dropped`` are the
    anti-vacuous-success fields: "at least one row survived" once let an
    8-key-set corpus train on an unannounced subset at rc 0, so coverage is now
    stated in numbers a caller can compare against expectations instead of
    inferred from silence. ``dropped`` keeps first-seen bucket order so
    equal-count tallies still render deterministically from one input stream.
    """

    rows: tuple[CorpusRow, ...]
    announcements: tuple[str, ...]
    refusal: str | None
    family: str
    think_policy: ThinkPolicy
    inspected: int
    dropped: tuple[tuple[str, int], ...]


class _Refusal(Exception):
    """Internal control flow only. Caught at the public boundary and rephrased
    as ``CorpusResult.refusal`` so no exception ever escapes this module."""


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_ROLE_MAP: Mapping[str, str] = {"human": "user", "gpt": "assistant"}

# Keys tried, in order, when a gpt "value" arrives as a dict. The estate's
# measured CoT shape is {"reasoning", "answer"}; "response"/"output" are the
# aliases seen when the same rows pass through the non-omni formatter. A dict
# carrying none of them is not coerced -- guessing which field was the answer
# is how a reasoning block ends up supervised as the reply.
_ANSWER_KEYS: tuple[str, ...] = ("answer", "response", "output")


def _record_label(record: Mapping[str, Any], index: int) -> str:
    rid = record.get("id", index)
    return f"record id={rid!r}"


def _coerce_value(value: Any, *, role: str, label: str) -> str:
    """Total handling of the polymorphic ``conversations[].value``.

    The estate measured the failure this exists against: the same rows carry a
    ``str`` in their *-omni-format.jsonl and a ``dict`` in the sibling file,
    and a converter that assumed ``str`` once rendered a system turn as the
    Python repr ``[{'type': 'text', ...}]`` and nobody noticed. So the rule
    here is: recognise a shape and normalise it, or refuse and name the
    observed type. ``str()`` is never called on a non-str.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if role == "assistant":
            for key in _ANSWER_KEYS:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    reasoning = value.get("reasoning")
                    if isinstance(reasoning, str) and reasoning.strip():
                        return f"<think>{reasoning.strip()}</think>\n{candidate.strip()}"
                    return candidate
            raise _Refusal(
                f"{label}: assistant 'value' is a dict with no usable answer key "
                f"(tried {_ANSWER_KEYS}; observed keys: {sorted(value)!r}). Refusing "
                "rather than guessing which field is the answer"
            )
        raise _Refusal(
            f"{label}: {role} 'value' is a dict {sorted(value)!r}; only assistant "
            "turns carry dict values on this corpus, and coercing anything else "
            "would re-encode structure into repr-shaped training text"
        )
    observed = type(value).__name__
    raise _Refusal(
        f"{label}: {role} 'value' has unrecognised type {observed!r} "
        f"(sample: {value!r:.120}). Only str and assistant-side dict shapes are "
        "recognised; anything else is refused and named, never str()-repr'd into "
        "the 'text' column -- that exact silent repr was measured once on this "
        "estate and trained through unnoticed"
    )


def _apply_think_policy(text: str, *, role: str, policy: ThinkPolicy) -> str:
    if role != "assistant":
        # Directives like /no_think in USER text are template input, not CoT;
        # stripping them here would lie about what the user asked.
        return text
    if policy is ThinkPolicy.STRIP:
        return _THINK_RE.sub("", text).strip()
    return text


def _resolve_media(raw: Any, *, key: str, root: str, label: str) -> tuple[tuple[str, ...], int]:
    """Resolve the image/video field to absolute paths plus an existence count.

    Resolution is per-entry because the corpus mixes relative and absolute values
    in the SAME file. Existence is COUNTED (and announced by the caller) while
    decodability is deliberately NOT established: 120 of 559 measured estate
    images pass ``Path.exists`` and fail PIL decode, and this module must
    import without PIL. Returning the count is how the exists-vs-decodes gap gets
    named on this module's watch -- the round-1 defect was the silence about the
    deferral, not the deferral itself.
    """
    if raw is None:
        return (), 0
    entries: Iterable[Any]
    if isinstance(raw, str):
        entries = (raw,)
    elif isinstance(raw, (bytes, bytearray)):
        # bytes satisfies isinstance(Sequence) but is never a path LIST here;
        # without this guard a blob field iterates into ints, and each int dies
        # with a per-entry message that names the wrong problem.
        raise _Refusal(
            f"{label}: field {key!r} is {type(raw).__name__!r}; expected a path "
            "string or a list of path strings, not a byte payload -- media bytes "
            "stay out of this module by design"
        )
    elif isinstance(raw, Sequence):
        entries = raw
    else:
        raise _Refusal(
            f"{label}: field {key!r} has type {type(raw).__name__!r}; expected "
            "a path string or a list of path strings"
        )
    resolved: list[str] = []
    existing = 0
    for entry in entries:
        if isinstance(entry, bool) or not isinstance(entry, str) or not entry.strip():
            raise _Refusal(f"{label}: field {key!r} contains non-path entry {entry!r:.80}")
        candidate = Path(entry.strip())
        candidate = candidate if candidate.is_absolute() else Path(root) / candidate
        resolved.append(str(candidate))
        if candidate.exists():
            existing += 1
    return tuple(resolved), existing


def _check_media_placeholders(text: str, n_images: int, n_videos: int, label: str) -> None:
    """Validate placeholder counts against the payload; the text is NOT touched.

    Renamed from ``_split_user_media``: that name promised a transformation and
    delivered a validation, and names are how a reader decides whether a call
    site still needs the payload wired elsewhere. The check itself stays because
    Job 202 showed an unvalidated mismatch surfacing as silent StopIteration
    dataloader exhaustion on one rank; a mismatch here becomes a BUCKETED drop,
    not a red run and not a quiet pass.
    """
    found_images = text.count("<image>")
    found_videos = text.count("<video>")
    if found_images != n_images or found_videos != n_videos:
        raise ValueError(
            f"{label}: placeholder/payload mismatch -- {found_images} '<image>' "
            f"marker(s) for {n_images} image path(s), {found_videos} '<video>' "
            f"marker(s) for {n_videos} video path(s)"
        )


def _canonical_render(messages: Sequence[Mapping[str, str]]) -> str:
    """Deterministic im_start/im_end rendering when no chat template is injected.

    Kept minimal on purpose: the converter owns STRUCTURE, the model's own
    chat_template.jinja owns rendering. When a template callable is injected it
    wins; tests pin this fallback so the module is checkable with no model and
    no GPU -- but its use is always announced, because training against the
    fallback while believing the model template rendered is exactly the
    silent-abstain shape the house rules ban.
    """
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    return "".join(parts)


def convert_records(
    records: Iterable[Mapping[str, Any]],
    *,
    corpus_root: str,
    family: FamilySpec,
    think_policy: ThinkPolicy,
    render_chat: Callable[[Sequence[Mapping[str, str]]], str] | None = None,
    max_drop_fraction: float | None = None,
) -> CorpusResult:
    """Turn estate ``conversations`` records into (text, media) rows.

    Driven by ``family`` so a third family is a registry entry, not a branch in
    this file. ``max_drop_fraction`` encodes a decision the CALLER must own:
    None means "convert what converts and tally the rest" -- legitimate when a
    partial corpus is chosen on purpose, because the tally is still announced.
    A number means the function itself refuses past that fraction, because a run
    that silently trains on the convertible residue of an 8-key-set corpus
    reports numbers about a corpus nobody approved. Zero produced rows refuses
    under EITHER setting: a converter that inspected everything and kept nothing
    did not succeed.
    """
    if max_drop_fraction is not None and not 0.0 <= max_drop_fraction <= 1.0:
        # A caller CONFIG error is raised, not refused-as-data: refusals describe
        # the corpus, and a number outside [0, 1] is a bug in the caller's wiring.
        raise ValueError(f"max_drop_fraction must lie in [0.0, 1.0], got {max_drop_fraction!r}")
    render = render_chat if render_chat is not None else _canonical_render
    announcements: list[str] = [
        f"corpus conversion under family {family.name!r}, "
        f"think_policy={think_policy.value!r}, corpus_root={corpus_root!r}, "
        f"max_drop_fraction={max_drop_fraction!r}"
    ]
    if render_chat is None:
        announcements.append(
            "no chat template injected; using the module's canonical im_start "
            "renderer. This is explicit and announced -- verify the rendered "
            "text matches the target model's chat_template before treating any "
            "quality number from this run as model evidence"
        )

    rows: list[CorpusRow] = []
    # Drops are bucketed BY REASON. An unattributed count ("3 records dropped")
    # is what let the FoxBrain GB200 audit leave 500-row Class-A mismatches as
    # an exercise for the reader; here the reason is part of the number, and the
    # first observed detail is kept so a bucket is diagnosable without a re-run.
    drop_counts: dict[str, int] = {}
    drop_first: dict[str, str] = {}
    inspected = 0
    media_total = 0
    media_exists = 0

    def notice(bucket: str, detail: str) -> None:
        drop_counts[bucket] = drop_counts.get(bucket, 0) + 1
        drop_first.setdefault(bucket, detail)

    try:
        for index, record in enumerate(records):
            inspected += 1
            if not isinstance(record, Mapping):
                raise _Refusal(
                    f"item {index} is {type(record).__name__!r}, not a mapping; "
                    "refusing the corpus as a whole -- a partially-coerced stream "
                    "that looks complete is worse than no output"
                )
            label = _record_label(record, index)
            rid = record.get("id", index)
            if isinstance(rid, bool) or not isinstance(rid, (str, int)):
                notice(
                    "unsupported record id type",
                    f"{label}: id is {type(rid).__name__!r}; expected str or int. "
                    "Round 1 silently str()-coerced ids, which retyped values and "
                    "invited collisions once distinct key sets were joined on id",
                )
                continue

            turns_raw = record.get("conversations")
            if (
                not isinstance(turns_raw, Sequence)
                or isinstance(turns_raw, (str, bytes, bytearray))
                or not turns_raw
            ):
                notice(
                    "missing/empty 'conversations'",
                    f"{label}: 'conversations' is missing, not a list, or empty",
                )
                continue

            images, n_img = _resolve_media(
                record.get("image"), key="image", root=corpus_root, label=label
            )
            videos, n_vid = _resolve_media(
                record.get("video"), key="video", root=corpus_root, label=label
            )
            media_total += len(images) + len(videos)
            media_exists += n_img + n_vid

            messages: list[dict[str, str]] = []
            system = (
                record.get("system") or record.get("system_prompt") or record.get("system_promt")
            )
            if isinstance(system, str) and system.strip():
                messages.append({"role": "system", "content": system.strip()})

            bad: tuple[str, str] | None = None
            for turn in turns_raw:
                if not isinstance(turn, Mapping):
                    raise _Refusal(
                        f"{label}: a conversations turn is {type(turn).__name__!r}, not a mapping"
                    )
                raw_role = turn.get("from")
                role = _ROLE_MAP.get(str(raw_role))
                if role is None:
                    bad = ("unmapped role", f"{label}: unmapped role {raw_role!r}")
                    break
                text = _apply_think_policy(
                    _coerce_value(turn.get("value"), role=role, label=label),
                    role=role,
                    policy=think_policy,
                )
                if role == "user":
                    try:
                        _check_media_placeholders(text, len(images), len(videos), label)
                    except ValueError as exc:
                        bad = ("placeholder/payload mismatch", str(exc))
                        break
                messages.append({"role": role, "content": text})
            if bad is not None:
                notice(bad[0], bad[1])
                continue
            if not any(m["role"] == "assistant" for m in messages):
                notice(
                    "no assistant turn",
                    f"{label}: no assistant turn, nothing to supervise",
                )
                continue
            rows.append(
                CorpusRow(
                    record_id=rid,
                    text=render(messages),
                    image_paths=images,
                    video_paths=videos,
                )
            )
    except _Refusal as refusal:
        return CorpusResult(
            rows=(),
            announcements=tuple(announcements),
            refusal=str(refusal),
            family=family.name,
            think_policy=think_policy,
            inspected=inspected,
            dropped=tuple(drop_counts.items()),
        )

    # Existence was counted per path during resolution precisely so this notice
    # can name the gap: N of M paths exist, decodability unknown. The estate
    # measured 120/559 existing files failing decode, so an exists-only count
    # read as "loadable" is a measured 21.5% lie.
    if media_total:
        media_note = (
            f"{media_exists}/{media_total} resolved media path(s) pass EXISTENCE "
            "checks only; decodability is the loader's gate and is NOT established "
            "here (the estate measured 120/559 existing files failing decode), so "
            "do not read this count as 'loadable'"
        )
        missing = media_total - media_exists
        if missing:
            media_note += (
                f"; {missing} resolved path(s) are absent on disk outright and "
                "will fail at open time"
            )
    else:
        media_note = "corpus carries no media references; nothing was resolved"
    announcements.append(media_note)

    for bucket, count in drop_counts.items():
        announcements.append(
            f"dropped {count} record(s) [{bucket}]; first observed: {drop_first[bucket]}"
        )
    dropped_total = sum(drop_counts.values())

    if not rows:
        # A component that inspected nothing did not pass -- and one that
        # inspected everything and kept nothing did not convert. Either shape
        # returning rc 0 with [] is how a 0/0 run exits GREEN.
        if inspected == 0:
            detail = (
                "the record stream was empty -- nothing was inspected, and a "
                "converter that inspected nothing did not succeed"
            )
        else:
            detail = (
                f"all {inspected} inspected record(s) were dropped (buckets "
                "above); an empty conversion is refused, never returned as "
                "success, regardless of max_drop_fraction"
            )
        return CorpusResult(
            rows=(),
            announcements=tuple(announcements),
            refusal=f"conversion yielded zero rows: {detail}",
            family=family.name,
            think_policy=think_policy,
            inspected=inspected,
            dropped=tuple(drop_counts.items()),
        )

    if max_drop_fraction is not None and dropped_total > 0:
        fraction = dropped_total / inspected
        if fraction > max_drop_fraction:
            top = sorted(drop_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            bucket_text = ", ".join(f"{bucket} ({count})" for bucket, count in top)
            return CorpusResult(
                rows=(),
                announcements=tuple(announcements),
                refusal=(
                    f"drop fraction {fraction:.3f} ({dropped_total}/{inspected} "
                    f"record(s)) exceeds max_drop_fraction={max_drop_fraction}; "
                    f"top buckets: {bucket_text}. Refusing the whole conversion: "
                    "returning the survivors at rc 0 would be the unannounced "
                    "partial-corpus success this tally exists to prevent"
                ),
                family=family.name,
                think_policy=think_policy,
                inspected=inspected,
                dropped=tuple(drop_counts.items()),
            )

    announcements.insert(
        1,
        f"converted {len(rows)}/{inspected} record(s) into 'text' rows; "
        f"{dropped_total} dropped across {len(drop_counts)} reason bucket(s); "
        "media payloads are carried as resolved paths on each row",
    )
    return CorpusResult(
        rows=tuple(rows),
        announcements=tuple(announcements),
        refusal=None,
        family=family.name,
        think_policy=think_policy,
        inspected=inspected,
        dropped=tuple(drop_counts.items()),
    )
