"""The audio leg's rollout: one clip in, group_size transcripts out.

This module is exactly the slice of the RL loop whose code changes when the
prompt surface is a WAVEFORM instead of prompt turns. It owns the clip + gold
manifest reader (:func:`load_audio_manifest`), the transcription-request prompt
builder (train/audio.py's own ``apply_transcription_request`` surface -- the
call its SFT collator wraps a target transcript with; here the target is
GENERATED, never supplied), the row-by-row transcript decode
(:func:`_decode_completion`), and the one ``generate()`` per step. From the
decoded rows on, the step is ``RLTrainer._price_generated`` -- the SAME
scoring, abstention accounting and ``_priced_tail`` pricing the built-in text
leg runs -- so the two legs cannot drift in what they do with a scored row.

The clip's audio feature tensors (``input_features`` and its frame mask) come
straight out of the request call and travel as that surface's own modality
keys into that seam, whose existing plumbing
(``_expand_modality_kwargs_for_group`` / ``_narrow_modality_kwargs_by_row``,
reached through ``_priced_tail``) forwards them to EVERY forward -- old,
current and frozen-reference log-probabilities alike. No parallel feature path
is invented here: the keys below are one row per CLIP, exactly the
per-training-row alignment that plumbing already assumes.

WHAT THIS MODULE CLAIMS: every manifest record is read under its own line
number, and a blank line, a non-object record, or a missing/blank ``audio`` /
``text`` field is refused naming that line -- never skipped; one prompt per
chunk record is built from ITS clip and the run's declared language (the
record's gold never enters the prompt); exactly one transcript per generated
row is decoded, scored against that row's own record, and priced conditioned on
the same clip the completion was generated from.

WHAT THIS MODULE DOES NOT CLAIM: that any clip decodes (the request API owns
that refusal -- one bad clip refuses the run rather than shortening the batch);
that the decode's tokenization agrees with the gold's word segmentation (that
is what the WER scorer's NORMALIZER decides, see speech_metrics); that the
penalty magnitudes of any declared reward are tuned; or that ppo or the
online-preference family run over audio -- ``run()`` refuses those together
with audio mode, because their step functions price entirely different rows
from the ones this rollout produces.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime

    from foundationscale.rl.algorithm import StepReport

from foundationscale.rl.trainer import (
    TrainerRefusal,
    _generation_mode,
)

__all__ = (
    "AudioRow",
    "asr_rollout_step",
    "load_audio_manifest",
)

# The keys the modality path does NOT carry: every other request-API output
# (``input_features``, its frame mask) is conditioning and is forwarded
# verbatim, the same split _price_generated applies to a processor's keys.
_TEXT_KEYS = ("input_ids", "attention_mask")


@dataclass(frozen=True, slots=True)
class AudioRow:
    """One audio manifest record: a clip path and its VERIFIED gold transcript.

    ``line`` is the manifest line the record was read from and is carried so
    every later refusal can NAME its source row rather than some batch index.
    ``gold`` is ``text`` under the name the backbone's own draw loop reads:
    ``RLTrainer.run`` filters and shards ``sample.gold`` and must see the same
    field on every leg. Nothing here compares the transcript to the clip --
    matching the two is the WER reward's measurement, and a manifest may
    legitimately record what the audio intends to say.
    """

    line: int
    audio: str
    text: str

    @property
    def gold(self) -> str:
        """The record's verified transcript, in the corpus loader's own vocabulary."""
        return self.text


def load_audio_manifest(path: str | None) -> tuple[AudioRow, ...]:
    """Load and validate one jsonl audio manifest (per-record keys: ``audio``, ``text``).

    WHAT IS CLAIMED on success: every returned row names a non-blank clip and a
    non-blank gold transcript, each still tagged with its manifest line for
    later diagnostics, and the returned denominator is every usable row of the
    file -- the reader never drops a record in silence.

    WHAT IS NOT CLAIMED: that any named clip exists or decodes (that refusal
    belongs to the request API, mid-rollout and named), that the transcript
    matches the clip, or that the transcript normalizes to a non-empty token
    list (a gold that normalizes to nothing ABSTAINS in WERReward and is
    counted out by the step's ``used < offered`` accounting rather than scored
    against nothing).
    """
    if not isinstance(path, str) or not path.strip():
        raise TrainerRefusal(
            f"field dataset={path!r}: 1 of 1 dataset paths must be a non-empty "
            f"string; audio mode reads the clip+transcript manifest through "
            f"this path"
        )
    rows: list[AudioRow] = []
    try:
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise TrainerRefusal(
                        f"manifest line {line_number}: the line is blank; 1 of 1 "
                        f"jsonl records must name a clip and its transcript "
                        f"rather than being silently skipped"
                    )
                try:
                    loaded = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrainerRefusal(
                        f"manifest line {line_number}: invalid JSON at column "
                        f"{exc.colno}: {exc.msg}"
                    ) from exc
                if not isinstance(loaded, dict):
                    raise TrainerRefusal(
                        f"manifest line {line_number}: decoded JSON is "
                        f"{type(loaded).__name__}, not object; 1 of 1 manifest "
                        f"records must be a JSON object with 'audio' and 'text' "
                        f"fields"
                    )
                fields: dict[str, str] = {}
                for field_name in ("audio", "text"):
                    value = loaded.get(field_name)
                    if not isinstance(value, str) or not value.strip():
                        raise TrainerRefusal(
                            f"manifest line {line_number}, field {field_name!r}: "
                            f"value {value!r} is not a non-empty string; a "
                            f"manifest record must name its clip (audio) and its "
                            f"gold transcript (text) -- a blank one would drop "
                            f"the declared modality or the verifiable answer "
                            f"between the file and the forward"
                        )
                    fields[field_name] = value
                rows.append(AudioRow(line=line_number, audio=fields["audio"], text=fields["text"]))
    except OSError as exc:
        raise TrainerRefusal(
            f"dataset={path!r}: the jsonl audio manifest could not be read "
            f"({exc}); 0 records are available for the run"
        ) from exc
    except UnicodeDecodeError as exc:
        raise TrainerRefusal(
            f"dataset={path!r}: the jsonl audio manifest is not UTF-8 ({exc}); "
            f"transcript supervision cannot be attributed to bytes whose "
            f"decoding is unknown"
        ) from exc
    if not rows:
        raise TrainerRefusal(
            f"dataset={path!r}: 0 of 0 lines supplied a usable audio record; an "
            f"empty run measures nothing and is refused"
        )
    return tuple(rows)


def _decode_completion(processor: Any, token_ids: Any) -> str:
    """One generated id row decoded to its transcript -- alone, never batched.

    ``Qwen3ASRProcessor.batch_decode`` IGNORES its ``return_format`` argument:
    whatever is asked for, a call answers one structured result per sequence --
    which is precisely how a batch of transcripts gets read as a batch of
    objects and scored against the wrong thing. Each row is therefore decoded
    on its own and its text read explicitly (a bare string, or a record/attribute
    carrying ``text``), and any other answer shape is REFUSED with its type
    named: this string IS the model's response, and a wrong read here would
    silently rescore every row against something that is not a transcript.
    """
    decoded = processor.batch_decode(
        [[int(token) for token in token_ids]], skip_special_tokens=True
    )
    if not decoded:
        raise TrainerRefusal(
            "0 of 1 decoded rows were answered for one generated completion; a "
            "completion with no decode has no scoreable text and is refused "
            "rather than scored as empty"
        )
    row = decoded[0]
    if isinstance(row, str):
        return row
    text = row.get("text") if isinstance(row, Mapping) else getattr(row, "text", None)
    if isinstance(text, str):
        return text
    raise TrainerRefusal(
        f"batch_decode answered {type(row).__name__} ({row!r}) for one generated "
        f"row; a row's transcript is read as a string, or as a record/attribute "
        f"carrying text, and any other shape is refused rather than scored"
    )


def _left_pad_rows(values: Sequence[Any], *, fill: int) -> Any:
    """1-D per-clip rows LEFT-padded into one rectangular ``(rows, width)`` tensor.

    Left, never right: the pads must PRECEDE each request, not land between it
    and the continuation ``generate()`` appends to it (the #546 generation
    invariant ``run`` states for the tokenizer's own padding side). The rows
    keep their right edges aligned, so the request is always continued at the
    same column for every clip in the batch.
    """
    import torch  # function-local: see rl/trainer.py's module docstring

    widths = [int(value.numel()) for value in values]
    width = max(widths)
    out = torch.full((len(values), width), fill, dtype=values[0].dtype)
    for index, (value, row_width) in enumerate(zip(values, widths, strict=True)):
        out[index, width - row_width :] = value
    return out


def _stack_feature_rows(values: Sequence[Any], *, key: str) -> Any:
    """One feature key's per-clip rows stacked along batch, padded at the frame axis.

    The rows carry one clip each and need not agree on how long that clip is.
    The pad axis is the ONE axis they disagree on -- read from the shapes
    themselves, never from the key's name -- because the answer for this family
    is ``input_features`` (mel by frames) and its frame mask (frames), and a
    second guess at orientation is exactly the kind of silent corruption this
    repository refuses to ship. Rows disagreeing on MORE than that one axis are
    refused naming the key and the shapes. Padded entries carry 0 -- the value
    every mask in this plane already writes for "not there", and zero features
    sitting under a 0 mask -- and a batch whose rows agreed completely is a
    plain ``cat``.
    """
    import torch  # function-local: see rl/trainer.py's module docstring

    ranks = {int(value.dim()) for value in values}
    if len(ranks) != 1:
        raise TrainerRefusal(
            f"key {key!r}: {len(ranks)} distinct ranks across {len(values)} "
            f"clip row(s) (shapes {[tuple(int(d) for d in v.shape) for v in values]}); "
            f"one clip's features are one structured row here, and rows that "
            f"disagree about their own dimensionality cannot be padded safely"
        )
    reference = [int(d) for d in values[0].shape]
    varying = {
        axis
        for axis in range(1, len(reference))
        if any(int(value.shape[axis]) != reference[axis] for value in values)
    }
    if len(varying) > 1:
        raise TrainerRefusal(
            f"key {key!r}: {len(varying)} axes vary across {len(values)} clip "
            f"row(s) (shapes {[tuple(int(d) for d in v.shape) for v in values]}); "
            f"only one axis may be the audio frame axis, and padding a second "
            f"unannounced axis would corrupt the conditioning it names"
        )
    if not varying:
        return torch.cat(list(values), dim=0)
    axis = varying.pop()
    width = max(int(value.shape[axis]) for value in values)
    padded: list[Any] = []
    for value in values:
        missing = width - int(value.shape[axis])
        if missing:
            shape = [int(d) for d in value.shape]  # every other axis as the clip's own
            shape[axis] = missing
            value = torch.cat([value, value.new_zeros(shape)], dim=axis)
        padded.append(value)
    return torch.cat(padded, dim=0)


def _encode_audio_prompts(
    *,
    processor: Any,
    chunk: Sequence[AudioRow],
    language: str,
    pad_token_id: int,
    device: str,
) -> dict[str, Any]:
    """One transcription request per clip, collated into one left-padded batch.

    Each row is exactly what ``processor.apply_transcription_request(audio=...,
    language=...)`` answers -- the typed-prefix request train/audio.py's own
    collator wraps its TARGET with; here the request stays unwrapped, because
    the transcript is generated, never supplied, and the record's gold never
    enters it. Values are normalised to batched tensors here (the request is
    made without ``return_tensors``, so a build answering python lists must not
    reach ``index_select`` as one) and one clip is 1 batch row by contract: a
    request answering more rows for one clip is refused with its shapes, since
    cutting or re-attributing one would do both silently.

    ``input_ids`` are LEFT-padded to the batch's longest request (see
    ``_left_pad_rows``) with an ``attention_mask`` derived from the per-row
    widths. Every other key -- ``input_features`` and its frame mask on this
    family -- is one row per clip, so it stacks through
    :func:`_stack_feature_rows` with the frame axis read off the row shapes
    themselves and is then forwarded verbatim through the modality path.

    Refuses a request answering no mapping, no ``input_ids``, non-tensor-shaped
    values, more than one sequence row, or rows whose key sets differ (a key
    missing from one row cannot be padded in without naming a conditioning it
    may not carry).
    """
    import torch  # function-local: see rl/trainer.py's module docstring

    rows: list[dict[str, Any]] = []
    for record in chunk:
        produced = processor.apply_transcription_request(audio=record.audio, language=language)
        if not isinstance(produced, Mapping):
            raise TrainerRefusal(
                f"manifest line {record.line}, clip {record.audio!r}: "
                f"apply_transcription_request answered "
                f"{type(produced).__name__}, not a mapping of prompt tensors; "
                f"1 of 1 prompt surfaces must yield input_ids plus its feature "
                f"keys"
            )
        row: dict[str, Any] = {}
        for key, value in produced.items():
            try:
                tensor = torch.as_tensor(value)
            # torch raises RuntimeError ("Could not infer dtype of object") for an
            # arbitrary object, not only TypeError/ValueError.
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TrainerRefusal(
                    f"manifest line {record.line}, key {key!r}: the request "
                    f"answered {type(value).__name__}, which no tensor can be "
                    f"read from; the prompt surface's own values are required "
                    f"here"
                ) from exc
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if int(tensor.shape[0]) != 1:
                raise TrainerRefusal(
                    f"manifest line {record.line}, key {key!r}: the request "
                    f"answered {tuple(int(d) for d in tensor.shape)} for ONE "
                    f"clip; one clip is 1 batch row here, and a wider answer "
                    f"would have to be cut or re-attributed silently"
                )
            row[key] = tensor
        if "input_ids" not in row:
            raise TrainerRefusal(
                f"manifest line {record.line}, clip {record.audio!r}: the "
                f"transcription request answered keys {sorted(row)} and 0 of "
                f"them carry input_ids; generate() has no prompt to continue"
            )
        ids = row["input_ids"].to(dtype=torch.long).reshape(-1)
        row["input_ids"] = ids
        row["attention_mask"] = (
            row["attention_mask"].to(dtype=torch.long).reshape(-1)
            if "attention_mask" in row
            else torch.ones_like(ids)
        )
        rows.append(row)

    key_sets = {frozenset(row) for row in rows}
    if len(key_sets) != 1:
        raise TrainerRefusal(
            f"{len(key_sets)} distinct key set(s) across {len(rows)} "
            f"transcription request row(s): "
            f"{sorted(tuple(sorted(keys)) for keys in key_sets)}; a batch whose "
            f"rows disagree about what the prompt carries cannot be stacked, and "
            f"dropping a key would drop it from the conditioning"
        )

    batched: dict[str, Any] = {
        "input_ids": _left_pad_rows([row["input_ids"] for row in rows], fill=int(pad_token_id)),
        "attention_mask": _left_pad_rows([row["attention_mask"] for row in rows], fill=0),
    }
    for key in rows[0]:
        if key in _TEXT_KEYS:
            continue
        batched[key] = _stack_feature_rows([row[key] for row in rows], key=key)
    return {key: value.to(device) for key, value in batched.items()}


def asr_rollout_step(
    *,
    trainer: Any,
    step: int,
    chunk: Sequence[AudioRow],
    model: Any,
    processor: Any,
    reward: Any,
    objective: Any,
    loss_fn: Any,
    optimizer: Any,
    ref_model: Any | None,
    device: str,
    pad_token_id: int,
    language: str,
    ctx: Any,
) -> StepReport | None:
    """One generate-score-price step over ``chunk``'s clips: the audio ``_one_step``.

    WHAT IS CLAIMED: one transcription request per record (its clip and the
    declared language ONLY -- its gold is never in the prompt, so scoring stays
    a measurement and not a rehearsal), one ``generate()`` under the SAME
    declared sampling settings every text rollout uses with
    ``group_size`` completions per clip, one decoded transcript per generated
    row (row by row -- see ``_decode_completion``), and then
    ``trainer._price_generated``: the shared seam, scoring completion row i
    against ``golds[i // group_size]`` and pricing the kept rows with the clip's
    feature tensors riding the modality-kwargs path to every old / current /
    reference forward.

    WHAT IS NOT CLAIMED: that any particular step produces a report (a step
    whose transcripts all abstain is UNMEASURED, in the seam's own voice), or
    that a clip decodes -- the transcription request owns that refusal, named
    per manifest line.
    """
    import torch  # function-local: see rl/trainer.py's module docstring

    from foundationscale.rl.distributed import generate_kwargs_for

    prompt_ids = _encode_audio_prompts(
        processor=processor,
        chunk=chunk,
        language=language,
        pad_token_id=pad_token_id,
        device=device,
    )
    # The audio tower does not cast its inputs: float32 log-mel features meet
    # bf16 conv weights and the forward raises. Floating features take the
    # policy's dtype; token ids and integer masks are left alone.
    feature_dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
    prompt_ids = {
        key: (value.to(feature_dtype) if value.is_floating_point() else value)
        for key, value in prompt_ids.items()
    }
    with torch.no_grad(), _generation_mode(model) as gen_model:
        generated = gen_model.generate(
            **prompt_ids,
            max_new_tokens=trainer.config.max_new_tokens,
            num_return_sequences=trainer.config.group_size,
            do_sample=True,
            temperature=trainer.config.temperature,
            top_p=trainer.config.top_p,
            # top_k=0 disables the cutoff; generate() wants the sentinel, and
            # passing None would silently restore the checkpoint's value -- the
            # very inheritance the declared sampling surface removed.
            top_k=trainer.config.top_k,
            pad_token_id=pad_token_id,
            # synced_gpus under fsdp: generate must keep stepping while ANY
            # rank is still generating, or an early-EOS rank deadlocks the next
            # all-gather -- see _one_step's own note at its generate call.
            **generate_kwargs_for(ctx, trainer.config.sharding),
        )
    prompt_width = int(prompt_ids["input_ids"].shape[1])
    completions = [
        _decode_completion(processor, generated[row, prompt_width:])
        for row in range(int(generated.shape[0]))
    ]
    golds: list[str | None] = [record.text for record in chunk]
    return trainer._price_generated(
        step=step,
        generated=generated,
        prompt_ids=prompt_ids,
        prompt_width=prompt_width,
        completions=completions,
        golds=golds,
        reward=reward,
        model=model,
        optimizer=optimizer,
        ref_model=ref_model,
        device=device,
        pad_token_id=pad_token_id,
        objective=objective,
        loss_fn=loss_fn,
        ctx=ctx,
    )
