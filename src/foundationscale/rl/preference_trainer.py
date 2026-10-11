"""Offline preference training over the differentiable tensor plane.

This trainer closes the offline counterpart of the RL loop: it reads one
JSONL preference file, supervises the supplied completions, prices one of
the six declared preference objectives through
``preference_torch.TensorPreferenceLoss``, and takes one optimiser step per
``StepReport``. The float objective implementations in ``losses.py`` and
``preference_objectives.py`` remain the audited oracles; this module does
not restate their margin arithmetic.

torch and transformers are imported lazily in ``run()``, the same idiom
``trainer.py`` uses, so importing this module on a torch-free host always
succeeds. A host that cannot supply either dependency is refused under the
repository's exit-96 contract with the missing dependency named. The
preference loop is text-only: a record carrying image or video data refuses
rather than training on its prompt alone.

WHAT THIS MODULE CLAIMS: references are produced in-step from a second,
frozen model exactly when the bound objective declares itself
reference-anchored; reference-free objectives never load that model; every
step backpropagates the same policy log-probability readings that were
priced; over-long completions are truncated, counted and printed, while a
prompt that exceeds the context budget is a refusal; and each report is
mirrored in ``history`` with the loss, the paired metrics when they are
defined, and the measured L2 movement of one fixed probe parameter.

WHAT THIS MODULE DOES NOT CLAIM: convergence, benchmark results, or a
correspondence claim beyond the tensor/oracle parity tests; ownership of
the objective arithmetic, which remains in the oracle classes and their
tensor recipe; video or image processing; or that a caller-chosen
``max_length`` is appropriate for the model's true positional limit. The
latter is configuration evidence supplied by the caller, not a quantity
this module can measure before the first forward.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    import torch

from foundationscale.gates.objective_gates import MetricObservation
from foundationscale.rl.algorithm import StepReport
from foundationscale.rl.distributed import (
    DistContext,
    agree_all,
    agree_max,
    all_reduce_mean,
    all_reduce_sum,
    destroy,
    init_distributed,
    is_main,
    save_checkpoint,
    shard_indices,
    wrap_ddp,
    wrap_fsdp2,
)
from foundationscale.rl.interfaces import BatchRefusal, LossOutput
from foundationscale.rl.losses import DPOLoss
from foundationscale.rl.preference import PreferenceObjective
from foundationscale.rl.preference_objectives import (
    CPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)
from foundationscale.rl.preference_torch import (
    TensorPreferenceLoss,
    is_paired,
    needs_reference,
    preference_metrics,
)
from foundationscale.rl.prompt_surface import resolve_prompt_surface
from foundationscale.rl.registry import lookup_algorithm
from foundationscale.rl.trainer import (
    _ADAPTERS,
    _FLATTENED_PATCH_PAIRS,
    MasterWeightOptimizer,
    TrainerRefusal,
    _AdapterDisabledReference,
    _apply_lora_adapter,
    _load_causal_lm,
    _loss_components,
    _micro_batched_backward,
    _narrow_modality_kwargs_by_row,
    _padding_backwards,
    _reference_plan,
    _refuse_exit_96,
    _refuse_vacuous_run,
    _token_logprobs,
    _trainable_parameters,
)
from foundationscale.train.conversation import (
    TurnMarkers,
    _conversation_processor_batch_call,
    _render_conversation_row,
    assistant_label_mask,
    derive_turn_markers,
)
from foundationscale.video import FrameBudget, frames_for_row


def _classify_modality_key(
    key: str,
    value: Any,
    *,
    row_count: int,
    total_images: int | None,
    total_videos: int | None,
) -> str:
    """'generic' | 'image' | 'video' for one ``modality_kwargs`` entry.

    MEASURED (gemma-4-12B-it processor, 2026-10-11): a hardcoded key-name
    list (``pixel_values``/``image_grid_thw``, mirroring
    ``rl.trainer._FLATTENED_PATCH_PAIRS``) is NOT enough. Gemma's processor
    also returns ``image_position_ids``/``video_position_ids`` -- ONE ROW
    PER IMAGE/VIDEO, exactly like ``pixel_values``/``pixel_values_videos``
    -- under neither of those two hardcoded names, so they fell through to
    the training-row-range "generic" branch and were narrowed by the WRONG
    axis the moment a batch mixed modalities (``RuntimeError: start (2) +
    length (1) exceeds dimension size (2)`` the first time a text-only row
    shared a step with two image rows). Classifying by SHAPE instead of a
    name list generalises across families without enumerating every
    per-family key: a key whose row-0 width equals the batch's training row
    count is training-row-aligned ("generic", including every PER-TOKEN key
    this plane has ever measured -- checked FIRST, so a width that
    coincidentally also equals a media total is still treated as the safe,
    historical per-row case); otherwise a width matching the total image or
    video count is that modality's own per-media-item axis. ``"video"`` in
    the key name is the ONLY disambiguator needed when a batch's image and
    video totals coincide (both ``_FLATTENED_PATCH_PAIRS`` entries, and
    gemma's ``*_position_ids``, name their video variant with that
    substring; MEASURED true of every key observed on gemma and Qwen3.6).
    """
    width0 = int(value.shape[0]) if hasattr(value, "shape") and value.dim() >= 1 else None
    if width0 is None or width0 == row_count:
        return "generic"
    if "video" in key and total_videos is not None and width0 == total_videos:
        return "video"
    if total_images is not None and width0 == total_images:
        return "image"
    if total_videos is not None and width0 == total_videos:
        return "video"
    return "generic"


__all__ = (
    "PreferenceTrainConfig",
    "PreferenceTrainer",
    "load_pairs_jsonl",
)


# Keys used internally while validating a JSONL record. They are deliberately
# prefixed ``_`` so a caller's ordinary record fields are not shadowed.
_LINE_KEY = "_line"

# The preference trainer has no processor path: an associated pixel or clip
# must be named and refused rather than omitted from the prompt. The two
# singular/plural spellings are the corpus surface used elsewhere in the RL
# plane; a null value names a permissible absent field rather than a carrier.
_MODALITY_KEYS: tuple[str, ...] = ("image", "images", "video", "videos")


@dataclass(frozen=True, slots=True)
class _EncodedCompletion:
    """One prompt/completion pair encoded as an unshifted id row.

    ``inputs`` includes the prompt and completion. ``target_mask`` is already
    shifted with the model's targets: prompt-prediction positions are 0 and
    completion-prediction positions are 1. Storing the shifted mask with the
    encoding prevents the batch builder from re-deriving each row's prompt
    boundary after right padding has been introduced.
    """

    inputs: tuple[int, ...]
    target_mask: tuple[int, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class _EncodedBatch:
    """The tensors one preference step prices and supervises.

    The tensors share the row order in ``policy_logprobs``: interleaved
    chosen/rejected rows for a paired objective, or ordinary completion rows
    for KTO. ``desirable`` stays outside the padded matrices because it is a
    per-row label, not a token-level reading.

    ``modality_kwargs`` is ``None`` for every text-only run (the historical
    shape, unchanged) and a dict of the processor's OWN extra keys (
    ``pixel_values``/``image_grid_thw``/``pixel_values_videos``/
    ``video_grid_thw``/``mm_token_type_ids``/... -- whatever the loaded
    family's processor returned beyond ``input_ids``/``attention_mask``) when
    the run declares ``image_column``/``video_column``.

    MEASURED (Qwen3.6-27B, a mixed image+video+text pair file): the raw
    processor output's ``image_grid_thw``/``video_grid_thw`` carry ONE ROW
    PER IMAGE/VIDEO THE BATCH ACTUALLY CARRIES, never one row per training
    row -- a batch where row 2 is the only video-bearing row still gets a
    1-row ``video_grid_thw``, not an 8-row one padded with zeros for the
    other 7. :func:`foundationscale.rl.trainer._narrow_modality_kwargs_by_row`
    (built for the online RL trainer, where "ONE image per prompt is this
    plane's only declared shape" per its own sibling
    ``_expand_modality_kwargs_for_group``'s docstring) assumes those two
    row-spaces already coincide, and applying it directly to a HETEROGENEOUS
    batch narrows the wrong media block or indexes out of range -- MEASURED:
    ``ValueError: Video features and video tokens do not match, tokens: 0,
    features: 2400`` the first time a video-row and an image/text-row shared
    one micro-batch slice. ``image_counts``/``video_counts`` (one int per
    TRAINING row, 0 when that row carries none) are what
    :meth:`PreferenceTrainer._forward_logprobs` uses to translate a
    batch-row range into the matching MEDIA-row range before calling the
    reused helper -- the helper itself is unchanged; only the row range
    handed to it is corrected per modality. ``None`` iff the whole batch
    carries neither image nor video keys (reruns exactly as a no-media call
    would).
    """

    target_ids: torch.Tensor
    completion_mask: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    desirable: torch.Tensor | None
    modality_kwargs: dict[str, torch.Tensor] | None = None
    image_counts: tuple[int, ...] | None = None
    video_counts: tuple[int, ...] | None = None


@dataclass
class PreferenceTrainConfig:
    """Configuration for one offline preference run.

    ``model`` and ``dataset`` are never defaulted: a default model would
    smuggle an untested loading surface into omitted configuration, and a
    default preference file would silently redefine the supervision.

    Optional objective knobs are ``None`` until specified. They map to the
    objective field of the same semantic name only: ``beta`` is not IPO's
    ``tau``, and DPO's auxiliary SFT weight is not implicit SFT control for
    ORPO or CPO, whose SFT components are always declared. ``library``
    defaults exactly as the registered objective factories do.

    WHAT IS CLAIMED: these are all the knobs this trainer reads, and an
    unsupported knob for a resolved algorithm refuses rather than being
    ignored.

    WHAT IS NOT CLAIMED: that any default is tuned for a particular model or
    corpus.
    """

    model: str
    dataset: str
    algorithm: str = "dpo"
    beta: float | None = None
    tau: float | None = None
    ipo_length_normalise: bool | None = None
    gamma: float | None = None
    lambda_: float | None = None
    sft_weight: float = 0.0
    kto_reference_point: float | None = None
    learning_rate: float = 1e-6
    master_weights: bool | None = None
    pairs_per_step: int = 4
    max_steps: int = 10
    max_length: int = 2048
    logprob_micro_batch: int = 0
    seed: int = 0
    device: str | None = None
    # Multi-GPU data parallel surface. "none" (default) reproduces the
    # original single-process behaviour exactly; "ddp" replicates the model;
    # "fsdp" shards fp32 master parameters via FSDP2 with a bf16 compute
    # policy (which replaces MasterWeightOptimizer, see distributed.py).
    sharding: str = "none"
    save_dir: str | None = None
    save_every: int = 0  # 0 = only a final save when save_dir is set
    gradient_checkpointing: bool = False
    # Declared adapter mode. None means FULL FINE-TUNE -- the SAME convention
    # train/loop.py's SFT plane and rl/trainer.py's RLTrainConfig use. Every
    # adapter_* knob is None by default, and a partial specification is
    # refused in PreferenceTrainer.__init__.
    adapter: str | None = None
    adapter_rank: int | None = None
    adapter_alpha: float | None = None
    adapter_targets: tuple[str, ...] | None = None
    adapter_dropout: float | None = None
    # Media declaration surface, mirroring train/loop.py's SFT plane and
    # video.py's own env names (FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN /
    # _VIDEO_COLUMN / _VIDEO_FRAMES / _VIDEO_SAMPLING / _VIDEO_MAX_SIDE /
    # _VIDEO_CACHE_DIR) as explicit config fields instead of environment
    # reads: this trainer is constructed directly, not through the SFT CLI's
    # environment surface. ``image_column``/``video_column`` are None by
    # default -- EXACTLY today's text-only trainer, byte-identical, with
    # every _MODALITY_KEYS carrier refused (see ``run``). Declaring one opts
    # that one modality's named field into the processor-rendered path;
    # every ``video_*`` knob below is validated as a partial declaration
    # (refused) when ``video_column`` is None, the same doctrine
    # ``adapter``/``adapter_*`` already enforces above.
    image_column: str | None = None
    video_column: str | None = None
    video_frames: int = 0
    video_sampling: str = "uniform"
    video_max_side: int | None = None
    video_cache_dir: str | None = None


def _record_kind(record: Mapping[str, Any]) -> tuple[bool, bool]:
    """Classify which coordination schema a JSON object claims.

    The two booleans are returned, rather than collapsing to a single kind,
    because the diagnostic needs to distinguish an absent schema from the
    stronger mixed-schema fact in which both are present.
    """
    has_paired = "chosen" in record or "rejected" in record
    has_kto = "completion" in record or "label" in record
    return has_paired, has_kto


def _record_text_field(record: Mapping[str, Any], name: str, line: int) -> str:
    """Read one mandatory, non-empty text field from a record.

    A JSON null or empty completion is not treated as an empty prefix or an
    unlabelled side: the objective would lose one unit of supervision while
    the denominator still claimed a row, so construction refuses here.
    """
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise TrainerRefusal(
            f"dataset line {line}, field {name!r}: value {value!r} is not a "
            f"non-empty string; the preference record's prompt and "
            f"supervised text must both be explicit"
        )
    return value


def load_pairs_jsonl(path: str, *, paired: bool) -> list[dict[str, Any]]:
    """Load and validate one JSONL preference dataset.

    A paired objective consumes records with ``prompt``, ``chosen`` and
    ``rejected`` fields. KTO consumes records with ``prompt``, ``completion``
    and a boolean (or 0/1) ``label``. A record carrying both families' fields
    is a mixed schema and refuses even when each family could individually
    select enough fields from it: silently choosing one family would change
    the supervision file's denominator.

    WHAT IS CLAIMED on success: every returned record carries the required
    fields, each text field is a non-empty string, each KTO label resolves to
    one bool, and the reserved ``_line`` entry names the source line for
    later diagnostics.

    WHAT IS NOT CLAIMED: that prompts render under any particular chat
    template, that any model can fit a record, or that media payloads are
    supported. Media are detected and refused by ``PreferenceTrainer.run``.
    """
    if not isinstance(path, str) or not path.strip():
        raise TrainerRefusal(
            f"field dataset={path!r}: 1 of 1 dataset paths must be a non-empty string"
        )
    records: list[dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise TrainerRefusal(
                        f"dataset line {line_number}: the line is blank; "
                        f"1 of 1 JSONL records must carry an explicit "
                        f"supervision schema rather than being silently "
                        f"skipped"
                    )
                try:
                    loaded = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrainerRefusal(
                        f"dataset line {line_number}: invalid JSON at column {exc.colno}: {exc.msg}"
                    ) from exc
                if not isinstance(loaded, dict):
                    raise TrainerRefusal(
                        f"dataset line {line_number}: decoded JSON is "
                        f"{type(loaded).__name__}, not object; 1 of 1 "
                        f"records must be a JSON object with named "
                        f"supervision fields"
                    )
                has_paired, has_kto = _record_kind(loaded)
                if has_paired and has_kto:
                    paired_fields = sorted(k for k in ("chosen", "rejected") if k in loaded)
                    kto_fields = sorted(k for k in ("completion", "label") if k in loaded)
                    raise TrainerRefusal(
                        f"dataset line {line_number}: 2 of 2 supervision "
                        f"schemas are present ({paired_fields} and {kto_fields}); "
                        f"a mixed record cannot name one row denominator"
                    )
                chosen: str | None = None
                rejected: str | None = None
                completion: str | None = None
                desirable: bool | None = None
                if paired:
                    if has_kto or not has_paired:
                        expected = ("prompt", "chosen", "rejected")
                        missing = tuple(name for name in expected if name not in loaded)
                        raise TrainerRefusal(
                            f"dataset line {line_number}: a paired "
                            f"objective requires fields {expected}; got "
                            f"{sorted(loaded)} with {len(missing)} of 3 "
                            f"required fields absent {missing!r}"
                        )
                    chosen = _record_text_field(loaded, "chosen", line_number)
                    rejected = _record_text_field(loaded, "rejected", line_number)
                else:
                    if has_paired or not has_kto:
                        expected = ("prompt", "completion", "label")
                        missing = tuple(name for name in expected if name not in loaded)
                        raise TrainerRefusal(
                            f"dataset line {line_number}: a KTO objective "
                            f"requires fields {expected}; got "
                            f"{sorted(loaded)} with {len(missing)} of 3 "
                            f"required fields absent {missing!r}"
                        )
                    completion = _record_text_field(loaded, "completion", line_number)
                    label = loaded.get("label")
                    if isinstance(label, bool):
                        desirable = label
                    elif isinstance(label, (int, float)) and label in (0, 1):
                        desirable = bool(label)
                    else:
                        raise TrainerRefusal(
                            f"dataset line {line_number}, field 'label': "
                            f"value {label!r} is not bool, 0 or 1; a KTO "
                            f"record must name one desirable row family"
                        )
                prompt = _record_text_field(loaded, "prompt", line_number)
                record = dict(loaded)
                record[_LINE_KEY] = line_number
                record["prompt"] = prompt
                if paired:
                    record["chosen"] = chosen
                    record["rejected"] = rejected
                else:
                    record["completion"] = completion
                    record["label"] = desirable
                records.append(record)
    except OSError as exc:
        raise TrainerRefusal(
            f"dataset={path!r}: the JSONL preference file could not be "
            f"read ({exc}); 0 records are available for the run"
        ) from exc
    except UnicodeDecodeError as exc:
        raise TrainerRefusal(
            f"dataset={path!r}: the JSONL preference file is not UTF-8 "
            f"({exc}); text supervision cannot be attributed to bytes "
            f"whose decoding is unknown"
        ) from exc
    if not records:
        raise TrainerRefusal(
            f"dataset={path!r}: 0 of 0 lines supplied a usable preference "
            f"record; an empty run measures nothing and is refused"
        )
    return records


class PreferenceTrainer:
    """One loop: read preference rows, encode, price and step.

    WHAT IS CLAIMED: ``pairs_per_step`` records are consumed in one
    seed-shuffled repeated order; paired records are encoded as interleaved
    chosen/rejected rows; references are recomputed, never copied from a
    generation pass; micro-batched policy scoring delivers gradients through
    the existing detached-leaf replay; and a step's metrics are read from
    the same detached tensors that priced the loss.

    WHAT IS NOT CLAIMED: generation, a rollout source, a reward, or an
    advantage estimator. Offline preference supervision supplies its own
    records, so the trainer's ``reward_stats`` abstains as ``None``.
    """

    def __init__(self, config: PreferenceTrainConfig) -> None:
        """Validate the denominator-sized configuration before any model load.

        Refusing here keeps an absent or malformed run declaration separate
        from a later environment failure: the message names the config field
        that made the run incoherent, not the first model call that happened
        to depend on it.
        """
        for field_name, value in (("model", config.model), ("dataset", config.dataset)):
            if not isinstance(value, str) or not value.strip():
                raise TrainerRefusal(
                    f"field {field_name}={value!r}: a non-empty string is "
                    f"required; absence of a model or dataset name does not "
                    f"select a default"
                )
        if isinstance(config.learning_rate, bool) or not isinstance(
            config.learning_rate, (int, float)
        ):
            raise TrainerRefusal(
                f"field learning_rate={config.learning_rate!r}: a finite "
                f"positive number is required"
            )
        if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
            raise TrainerRefusal(
                f"field learning_rate={config.learning_rate!r}: a "
                f"non-positive or non-finite rate does not name a descent "
                f"direction"
            )
        # The values are typed Any on purpose: these are RUNTIME validations
        # of constructor input, and a statically int-typed loop variable
        # would make the isinstance legs read as unreachable code instead of
        # as the refusal surface they are.
        integer_fields: tuple[tuple[str, Any, int], ...] = (
            ("pairs_per_step", config.pairs_per_step, 1),
            ("max_steps", config.max_steps, 1),
            ("max_length", config.max_length, 2),
            ("seed", config.seed, 0),
            ("logprob_micro_batch", config.logprob_micro_batch, 0),
        )
        # Distinct loop names from the model/dataset loop above: reusing
        # them would fuse two unrelated declared types into one variable and
        # turn these legs into statically impossible checks.
        for int_field_name, int_value, minimum in integer_fields:
            if isinstance(int_value, bool) or not isinstance(int_value, int):
                raise TrainerRefusal(
                    f"field {int_field_name}={int_value!r}: an integer is required"
                )
            if int_value < minimum:
                raise TrainerRefusal(
                    f"field {int_field_name}={int_value}: {minimum} is the minimum "
                    f"valid value; the run denominator would otherwise be "
                    f"vacuous"
                )
        if config.master_weights is not None and not isinstance(config.master_weights, bool):
            raise TrainerRefusal(
                f"field master_weights={config.master_weights!r}: expected "
                f"None, True or False; this switch selects the optimiser "
                f"weight plane"
            )
        if config.sharding not in ("none", "ddp", "fsdp"):
            raise TrainerRefusal(
                f"field sharding={config.sharding!r}: expected one of "
                f"'none', 'ddp' or 'fsdp'; an unknown sharding value would "
                f"silently replicate or mis-shard the policy"
            )
        if isinstance(config.save_every, bool) or not isinstance(config.save_every, int):
            raise TrainerRefusal(f"field save_every={config.save_every!r}: an integer is required")
        if config.save_every < 0:
            raise TrainerRefusal(
                f"field save_every={config.save_every}: 0 is the minimum "
                f"valid value; a negative cadence names no checkpoint step"
            )
        if config.save_dir is not None and (
            not isinstance(config.save_dir, str) or not config.save_dir.strip()
        ):
            raise TrainerRefusal(
                f"field save_dir={config.save_dir!r}: None or a non-empty "
                f"string is required; an empty path names no checkpoint root"
            )
        if not isinstance(config.gradient_checkpointing, bool):
            raise TrainerRefusal(
                f"field gradient_checkpointing={config.gradient_checkpointing!r}: "
                f"a bool is required"
            )
        if config.kto_reference_point is not None and (
            isinstance(config.kto_reference_point, bool)
            or not isinstance(config.kto_reference_point, (int, float))
            or not math.isfinite(config.kto_reference_point)
            or config.kto_reference_point < 0.0
        ):
            raise TrainerRefusal(
                f"field kto_reference_point="
                f"{config.kto_reference_point!r}: a finite non-negative "
                f"reference point is required; a KL divergence cannot be "
                f"negative or non-finite"
            )
        if config.adapter is not None and config.adapter not in _ADAPTERS:
            raise TrainerRefusal(f"field adapter={config.adapter!r} is not one of {_ADAPTERS}")
        if config.adapter is None:
            # A partial specification is a refusal, not a hint: every
            # adapter_* field with adapter unset is a statement about nothing.
            for field_name in (
                "adapter_rank",
                "adapter_alpha",
                "adapter_targets",
                "adapter_dropout",
            ):
                value = getattr(config, field_name)
                if value is not None:
                    raise TrainerRefusal(
                        f"field {field_name}={value!r} is set while adapter is "
                        f"None: a partial adapter specification is refused. Set "
                        f"adapter to one of {_ADAPTERS}, or clear {field_name}"
                    )
        elif (
            config.adapter_rank is None
            or isinstance(config.adapter_rank, bool)
            or (not isinstance(config.adapter_rank, int) or int(config.adapter_rank) < 1)
        ):
            raise TrainerRefusal(
                f"field adapter={config.adapter!r} requires adapter_rank to be a "
                f"positive int; got adapter_rank={config.adapter_rank!r}. A "
                "missing or non-positive rank silently defines the adapter's "
                "capacity, which is exactly the unrecorded-config failure"
            )
        if config.adapter_targets is not None:
            config.adapter_targets = tuple(config.adapter_targets)
            if not config.adapter_targets:
                raise TrainerRefusal(
                    "field adapter_targets=() is refused as vacuous: name at "
                    "least one target, or pass None to use peft's per-model "
                    "defaults"
                )
        for column_field in ("image_column", "video_column"):
            column_value = getattr(config, column_field)
            if column_value is not None and (
                not isinstance(column_value, str) or not column_value.strip()
            ):
                raise TrainerRefusal(
                    f"field {column_field}={column_value!r}: None or a "
                    f"non-empty string is required; an empty column name "
                    f"names no field"
                )
        if config.video_column is None:
            # Same "partial declaration is refused" doctrine as adapter/
            # adapter_*: a video_* knob set with no video_column is a
            # statement about nothing, not a hint to infer one.
            for video_field, default in (
                ("video_frames", 0),
                ("video_sampling", "uniform"),
                ("video_max_side", None),
                ("video_cache_dir", None),
            ):
                value = getattr(config, video_field)
                if value != default:
                    raise TrainerRefusal(
                        f"field {video_field}={value!r} is set while "
                        f"video_column is None: a partial video "
                        f"specification is refused. Declare video_column, "
                        f"or clear {video_field}"
                    )
            video_budget: FrameBudget | None = None
        else:
            if isinstance(config.video_frames, bool) or not isinstance(config.video_frames, int):
                raise TrainerRefusal(
                    f"field video_frames={config.video_frames!r}: an integer is required"
                )
            if config.video_max_side is not None and (
                isinstance(config.video_max_side, bool)
                or not isinstance(config.video_max_side, int)
            ):
                raise TrainerRefusal(
                    f"field video_max_side={config.video_max_side!r}: None or "
                    f"an integer is required"
                )
            try:
                video_budget = FrameBudget(
                    frames=config.video_frames,
                    sampling=config.video_sampling,
                    max_side=config.video_max_side,
                )
            except (ValueError, TypeError) as exc:
                raise TrainerRefusal(
                    f"field video_column={config.video_column!r} is declared "
                    f"but its frame budget is invalid: {exc}"
                ) from exc
        self.config = config
        self._needs_media = config.image_column is not None or config.video_column is not None
        self._video_budget = video_budget
        self._processor: Any | None = None
        self._turn_markers: TurnMarkers | None = None
        self._objective = self._resolve_objective()
        self._paired = is_paired(self._objective)
        self._reference_required = needs_reference(self._objective)
        self.history: list[dict[str, Any]] = []
        self.truncated_completion_count = 0
        self._kto_reference_point = (
            float(config.kto_reference_point) if config.kto_reference_point is not None else None
        )
        self._probe_parameter: Any | None = None
        self._probe_initial: Any | None = None

    def _unsupported_knobs(self, supported: tuple[str, ...]) -> tuple[str, ...]:
        """Name configured knobs this objective does not consume.

        This is a policy check, not a convenience: accepting an unrelated
        positive knob would make the printed objective configuration and the
        priced objective disagree, which is exactly the silent fallback the
        preference plane refuses.
        """
        configured: list[str] = []
        knob_values: tuple[tuple[str, Any], ...] = (
            ("beta", self.config.beta),
            ("tau", self.config.tau),
            ("ipo_length_normalise", self.config.ipo_length_normalise),
            ("gamma", self.config.gamma),
            ("lambda_", self.config.lambda_),
            ("sft_weight", self.config.sft_weight),
            ("kto_reference_point", self.config.kto_reference_point),
        )
        for name, value in knob_values:
            if name not in supported and value not in (None, 0.0):
                configured.append(name)
        return tuple(configured)

    def _replace_objective(self, objective: Any, **changes: Any) -> PreferenceObjective:
        """Rebuild a frozen oracle objective with exactly the supplied knobs.

        ``dataclasses.replace`` preserves every name field a caller selected
        on a custom-constructed registry objective, avoiding a trainer-side
        restatement of the objective's name surface.
        """
        replaced = replace(objective, **changes)
        if not isinstance(replaced, PreferenceObjective):  # pragma: no cover
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r}: rebuilding "
                f"{type(objective).__name__} did not yield a "
                f"PreferenceObjective"
            )
        return replaced

    def _resolve_objective(self) -> PreferenceObjective:
        """Resolve and configure the registry objective for this family.

        WHAT IS CLAIMED: the registry entry is a ``PreferenceObjective``;
        the value returned carries this config's supported knobs and no
        unsupported configured knob.

        WHAT IS NOT CLAIMED: that a group-relative or online registry entry
        can be adapted by guessing columns. Such an entry is a family
        mismatch and is refused by name.
        """
        algorithm = lookup_algorithm(self.config.algorithm)
        objective = getattr(algorithm, "_objective", None)
        if objective is None:
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r}: 0 of 1 required "
                f"objective instances are exposed by the registry binding "
                f"({type(algorithm).__name__}); this trainer prices a "
                f"declared preference objective, not an inference about "
                f"the algorithm's class"
            )
        if not isinstance(objective, PreferenceObjective):
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r}: family mismatch; "
                f"its objective {type(objective).__name__} is not a "
                f"PreferenceObjective. The preference family is dpo, ipo, "
                f"kto, orpo, simpo or cpo; group-relative objectives belong "
                f"to RLTrainer"
            )
        if isinstance(objective, DPOLoss):
            unsupported = self._unsupported_knobs(("beta", "sft_weight"))
            changes: dict[str, Any] = {}
            if self.config.beta is not None:
                changes["beta"] = self.config.beta
            changes["sft_weight"] = self.config.sft_weight
        elif isinstance(objective, IPOLoss):
            unsupported = self._unsupported_knobs(("tau", "ipo_length_normalise"))
            changes = {}
            if self.config.tau is not None:
                changes["tau"] = self.config.tau
            if self.config.ipo_length_normalise is not None:
                changes["length_normalise"] = self.config.ipo_length_normalise
        elif isinstance(objective, KTOLoss):
            unsupported = self._unsupported_knobs(("beta", "kto_reference_point"))
            changes = {}
            if self.config.beta is not None:
                changes["beta"] = self.config.beta
        elif isinstance(objective, ORPOLoss):
            unsupported = self._unsupported_knobs(("lambda_",))
            changes = {}
            if self.config.lambda_ is not None:
                changes["lambda_"] = self.config.lambda_
        elif isinstance(objective, SimPOLoss):
            unsupported = self._unsupported_knobs(("beta", "gamma"))
            changes = {}
            if self.config.beta is not None:
                changes["beta"] = self.config.beta
            if self.config.gamma is not None:
                changes["gamma"] = self.config.gamma
        elif isinstance(objective, CPOLoss):
            unsupported = self._unsupported_knobs(("beta", "lambda_"))
            changes = {}
            if self.config.beta is not None:
                changes["beta"] = self.config.beta
            if self.config.lambda_ is not None:
                changes["lambda_"] = self.config.lambda_
        else:
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r}: objective "
                f"{type(objective).__name__} satisfies the preference "
                f"protocol but is not one of the six oracle classes this "
                f"tensor recipe can price"
            )
        if unsupported:
            raise TrainerRefusal(
                f"algorithm {self.config.algorithm!r}: "
                f"{len(unsupported)} configured knob(s) are unsupported by "
                f"{type(objective).__name__}: {unsupported}; ignoring them "
                f"would make the declared and measured objectives differ"
            )
        return self._replace_objective(objective, **changes)

    def _encode_completion(
        self,
        *,
        tokenizer: Any,
        record: Mapping[str, Any],
        prompt_text: str,
        completion_text: str,
        side: str,
    ) -> _EncodedCompletion:
        """Encode one completion and derive its shifted supervision mask.

        The prompt alone is never truncated: changing the prompt changes the
        conditional the preference measured, not merely the amount of
        supervision. Only the completion tail is truncated, and each such
        record increments the public truncated-completion count.
        """
        line = int(record[_LINE_KEY])
        try:
            encoded_prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            encoded_completion = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        except Exception as exc:  # noqa: BLE001 -- preserve the failing surface in the refusal
            raise TrainerRefusal(
                f"dataset line {line}, side {side!r}: tokenizer encoding "
                f"failed ({exc}); a row whose tokenization cannot be "
                f"measured cannot enter the preference denominator"
            ) from exc
        prompt_ids = tuple(int(value) for value in encoded_prompt)
        completion_ids = tuple(int(value) for value in encoded_completion)
        if not prompt_ids:
            raise TrainerRefusal(
                f"dataset line {line}, side {side!r}: the templated prompt "
                f"encoded to 0 tokens; no conditional position exists from "
                f"which a completion token can be supervised"
            )
        if not completion_ids:
            raise TrainerRefusal(
                f"dataset line {line}, side {side!r}: the completion "
                f"encoded to 0 tokens; an unsupervised side cannot be "
                f"reported as a measured preference row"
            )
        if len(prompt_ids) > self.config.max_length:
            raise TrainerRefusal(
                f"dataset line {line}, side {side!r}: the templated prompt "
                f"is {len(prompt_ids)} tokens but max_length="
                f"{self.config.max_length}; truncating a prompt changes "
                f"the measured conditional and is refused"
            )
        room = self.config.max_length - len(prompt_ids)
        if room <= 0:
            raise TrainerRefusal(
                f"dataset line {line}, side {side!r}: the prompt consumes "
                f"all {self.config.max_length} model positions, leaving "
                f"0 for a supervised completion token"
            )
        truncated = len(completion_ids) > room
        if truncated:
            completion_ids = completion_ids[:room]
            self.truncated_completion_count += 1
        inputs = prompt_ids + completion_ids
        # The first target position predicts prompt token 1. The one before
        # the completion predicts the first completion token and is therefore
        # supervised. Position boundaries, not decoded strings, define this
        # mask because BPE merges need not align with textual whitespace.
        target_mask = (0,) * (len(prompt_ids) - 1) + (1,) * len(completion_ids)
        return _EncodedCompletion(
            inputs=inputs,
            target_mask=target_mask,
            truncated=truncated,
        )

    def _encode_record_completion(
        self,
        *,
        tokenizer: Any,
        record: Mapping[str, Any],
        completion: str,
        side: str,
    ) -> _EncodedCompletion:
        """Apply the chat template and encode one side of a preference row."""
        line = int(record[_LINE_KEY])
        messages = [{"role": "user", "content": record["prompt"]}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as exc:  # noqa: BLE001 -- template failures are data-surface failures
            raise TrainerRefusal(
                f"dataset line {line}: tokenizer.apply_chat_template "
                f"failed ({exc}); silently joining strings would change "
                f"the measured prompt distribution"
            ) from exc
        if not isinstance(prompt_text, str):
            raise TrainerRefusal(
                f"dataset line {line}: apply_chat_template returned "
                f"{type(prompt_text).__name__}, not str; the trainer "
                f"requires an explicit rendered prompt"
            )
        return self._encode_completion(
            tokenizer=tokenizer,
            record=record,
            prompt_text=prompt_text,
            completion_text=completion,
            side=side,
        )

    def _encode_batch(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        pad_token_id: int,
        device: str,
    ) -> _EncodedBatch:
        """Build one right-padded tensor batch in the kernel's row layout.

        Right padding is sufficient because this trainer never calls
        ``generate``: no continuation is appended after the supplied ids, so
        the causally visible tokens cannot be crossed by pads. The invariant
        that justifies left padding in the online trainer is deliberately not
        restated as a requirement here.

        Dispatches to :meth:`_encode_batch_media` when the run declares
        ``image_column``/``video_column`` -- a distinct method rather than a
        branch threaded through every line below, because the media path
        renders through the SFT processor pipeline
        (``_render_conversation_row``/``_conversation_processor_batch_call``/
        ``assistant_label_mask``) instead of a bare tokenizer, and the two
        pipelines share no line of tensor-building code. ``tokenizer`` and
        ``pad_token_id`` are unused on that path (the processor renders and
        pads its own batch); they stay required parameters because the sole
        caller, ``_one_step``, passes them unconditionally for both paths.
        """
        if not records:  # pragma: no cover -- run() always passes a non-empty chunk
            raise TrainerRefusal(
                "0 records were supplied for a configured step; an empty "
                "batch has no supervision denominator"
            )
        if self._needs_media:
            return self._encode_batch_media(records=records, device=device)
        encoded_rows: list[_EncodedCompletion] = []
        desirable_values: list[bool] = []
        for record in records:
            if self._paired:
                encoded_rows.append(
                    self._encode_record_completion(
                        tokenizer=tokenizer,
                        record=record,
                        completion=str(record["chosen"]),
                        side="chosen",
                    )
                )
                encoded_rows.append(
                    self._encode_record_completion(
                        tokenizer=tokenizer,
                        record=record,
                        completion=str(record["rejected"]),
                        side="rejected",
                    )
                )
            else:
                encoded_rows.append(
                    self._encode_record_completion(
                        tokenizer=tokenizer,
                        record=record,
                        completion=str(record["completion"]),
                        side="completion",
                    )
                )
                label = record["label"]
                if not isinstance(label, bool):
                    raise TrainerRefusal(
                        f"dataset line {record[_LINE_KEY]}: KTO label "
                        f"{label!r} was not a bool after validation"
                    )
                desirable_values.append(label)

        import torch

        row_count = len(encoded_rows)
        sequence_width = max(len(row.inputs) for row in encoded_rows)
        target_width = sequence_width - 1
        input_ids = torch.full(
            (row_count, sequence_width),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (row_count, sequence_width),
            dtype=torch.long,
            device=device,
        )
        completion_mask = torch.zeros(
            (row_count, target_width),
            dtype=torch.float32,
            device=device,
        )
        for row_index, row in enumerate(encoded_rows):
            width = len(row.inputs)
            input_ids[row_index, :width] = torch.tensor(
                row.inputs,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row_index, :width] = 1
            completion_mask[row_index, : len(row.target_mask)] = torch.tensor(
                row.target_mask,
                dtype=torch.float32,
                device=device,
            )
        target_ids = input_ids[:, 1:]
        desirable = (
            torch.tensor(desirable_values, dtype=torch.bool, device=device)
            if desirable_values
            else None
        )
        return _EncodedBatch(
            target_ids=target_ids,
            completion_mask=completion_mask,
            attention_mask=attention_mask,
            input_ids=input_ids,
            desirable=desirable,
        )

    def _encode_batch_media(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        device: str,
    ) -> _EncodedBatch:
        """Build one processor-rendered batch for an image/video-declared run.

        Each OUTPUT row (chosen, rejected, or one KTO completion) is rendered
        INDEPENDENTLY through the exact SFT pipeline
        (:func:`foundationscale.train.conversation._render_conversation_row`,
        which itself calls ``normalize_conversation`` for marker validation
        and image loading / ``frames_for_row`` for video) as a synthetic
        two-turn conversation ``[user: prompt, assistant: this side's
        completion]``, then the whole step's rows are encoded in ONE
        :func:`_conversation_processor_batch_call`, byte-identical to how
        the SFT collator calls it. A paired record's chosen and rejected
        rows therefore each reload and re-render the SAME declared image(s)
        /video segment rather than sharing one upstream processor call: this
        pays one extra PIL decode (video frames are cache-hit after the
        first side, see ``video.video_frame_paths``) per pair instead of
        re-deriving ``rl.trainer._expand_modality_kwargs_for_group``'s
        GROUP/``kept_indices`` semantics for a shape -- two differently
        truncated completions sharing one prompt -- that function was never
        built for (it expands ONE prompt into K ROLLOUT rows, not two
        independently-worded sides). ``_narrow_modality_kwargs_by_row`` IS
        reused unchanged below, in ``_forward_logprobs``, for micro-batch
        row-range slicing -- that helper's block-vs-row distinction is
        exactly the one this batch still needs.

        Labels are derived with ``assistant_label_mask`` off the SAME
        ``derive_turn_markers`` boundary the SFT plane uses
        (``self._turn_markers``, computed once in ``run()``), then shifted
        into this trainer's ``completion_mask`` convention
        (``labels[:, 1:] != -100``) rather than re-deriving a prompt/
        completion token boundary by hand: the processor expands
        ``<image>``/``<video>`` markers into a family-specific placeholder
        token COUNT this trainer has no other way to predict, so only the
        rendered token stream itself can say where the assistant span starts.

        Refuses (never truncates) a row whose rendered length exceeds
        ``max_length``: unlike the text-only ``_encode_completion`` path
        (which may truncate a completion's tail), truncating here could cut
        mid-image placeholder or drop a turn's own end-of-turn marker, so
        the only safe response to an overlong media row is to refuse and
        name it.
        """
        import torch

        rows: list[dict[str, Any]] = []
        desirable_values: list[bool] = []
        for record in records:
            sides: tuple[tuple[str, str], ...]
            if self._paired:
                sides = (
                    ("chosen", str(record["chosen"])),
                    ("rejected", str(record["rejected"])),
                )
            else:
                sides = (("completion", str(record["completion"])),)
            for _side, completion_text in sides:
                row = dict(record)
                row["conversations"] = [
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": completion_text},
                ]
                rows.append(row)
                if not self._paired:
                    label = record["label"]
                    if not isinstance(label, bool):
                        raise TrainerRefusal(
                            f"dataset line {record[_LINE_KEY]}: KTO label "
                            f"{label!r} was not a bool after validation"
                        )
                    desirable_values.append(label)

        frames_for: Any = None
        if self.config.video_column is not None:
            video_column: str = self.config.video_column
            if self._video_budget is None:  # pragma: no cover -- __init__ always builds one
                raise TrainerRefusal(
                    "video_column is declared but 0 frame budgets were built at construction time"
                )
            video_budget: FrameBudget = self._video_budget
            base_dir = Path(self.config.dataset).resolve().parent
            cache_dir = self.config.video_cache_dir or str(base_dir / ".fs_video_frames")

            def frames_for(data: Mapping[str, Any]) -> dict[str, Any]:
                return frames_for_row(
                    data,
                    video_column=video_column,
                    budget=video_budget,
                    cache_dir=cache_dir,
                    base_dir=base_dir,
                )

        texts: list[str] = []
        images_by_row: list[list[Any]] = []
        videos_by_row: list[list[Any]] = []
        video_metadata_by_row: list[list[dict[str, Any]]] = []
        for row_index, row in enumerate(rows):
            (
                text,
                loaded_images,
                video_frames,
                video_metadata,
                assistant_turn_count,
            ) = _render_conversation_row(
                row_index,
                row,
                conversations_column="conversations",
                image_column=self.config.image_column,
                video_column=self.config.video_column,
                processor=self._processor,
                frames_for=frames_for,
            )
            if assistant_turn_count != 1:  # pragma: no cover -- this method builds exactly one
                raise TrainerRefusal(
                    f"row {row_index}: rendered {assistant_turn_count} "
                    f"assistant turn(s), expected exactly 1 (the completion "
                    f"side being priced)"
                )
            texts.append(text)
            images_by_row.append(loaded_images)
            videos_by_row.append(video_frames)
            video_metadata_by_row.append(video_metadata)

        batch_feature = _conversation_processor_batch_call(
            self._processor,
            texts,
            images_by_row,
            videos_by_row,
            video_metadata_by_row,
            max_length=self.config.max_length,
            pad_to_max=False,
        )
        input_ids = batch_feature["input_ids"].to(device)
        attention_mask = batch_feature["attention_mask"].to(device)
        row_count = int(input_ids.shape[0])

        overlong = tuple(
            row_index
            for row_index in range(row_count)
            if int(attention_mask[row_index].sum()) > self.config.max_length
        )
        if overlong:
            shown = ", ".join(str(index) for index in overlong[:5])
            more = f" and {len(overlong) - 5} more" if len(overlong) > 5 else ""
            raise TrainerRefusal(
                f"{len(overlong)} rendered row(s) exceed max_length="
                f"{self.config.max_length} after media expansion (row "
                f"index(es) {shown}{more}); media rows are never truncated "
                f"-- truncation could cut a media placeholder or drop a "
                f"turn's end marker -- raise max_length or drop the record"
            )

        if self._turn_markers is None:  # pragma: no cover -- run() derives it first
            raise TrainerRefusal(
                "0 of 1 required turn markers are derived; a media-declared "
                "run cannot locate the supervised span without them"
            )
        labels = assistant_label_mask(
            input_ids,
            attention_mask,
            self._turn_markers,
            expected_spans=[1] * row_count,
        )
        completion_mask = (labels[:, 1:] != -100).to(torch.float32)
        target_ids = input_ids[:, 1:]

        modality_kwargs: dict[str, torch.Tensor] = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in batch_feature.items()
            if key not in ("input_ids", "attention_mask")
        }
        desirable = (
            torch.tensor(desirable_values, dtype=torch.bool, device=device)
            if desirable_values
            else None
        )
        # One entry per TRAINING row (0 for a row with none of that
        # modality) -- the row-order match to images_by_row/videos_by_row
        # is exact because both were built in the SAME enumerate(rows) loop
        # above that fed _conversation_processor_batch_call. See
        # _EncodedBatch's docstring for why _forward_logprobs needs these:
        # the processor's own image_grid_thw/video_grid_thw count MEDIA,
        # not training rows, and the two counts only coincide when every
        # row carries exactly one of that modality.
        image_counts = tuple(len(images) for images in images_by_row)
        video_counts = tuple(len(videos) for videos in videos_by_row)
        return _EncodedBatch(
            target_ids=target_ids,
            completion_mask=completion_mask,
            attention_mask=attention_mask,
            input_ids=input_ids,
            desirable=desirable,
            modality_kwargs=modality_kwargs or None,
            image_counts=image_counts if any(image_counts) else None,
            video_counts=video_counts if any(video_counts) else None,
        )

    def _forward_logprobs(
        self,
        *,
        model: Any,
        batch: _EncodedBatch,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Return target log-probabilities for one half-open row slice.

        Every key in ``batch.modality_kwargs`` is classified by
        :func:`_classify_modality_key` -- shape against the batch's training
        row count and total image/video count, not a hardcoded key-name list
        (see that function's docstring for the measured gemma defect this
        closes: ``image_position_ids``/``video_position_ids`` are per-media
        axes under neither of ``rl.trainer``'s two flattened-patch names).
        ``_narrow_modality_kwargs_by_row`` is reused UNCHANGED once per
        bucket, with that bucket's OWN row range: the training-row range for
        'generic' keys, the MEDIA-row range (translated via
        ``image_counts``/``video_counts``) for 'image'/'video' keys -- never
        the training-row range directly for the latter, whenever the batch
        can be heterogeneous (some rows image, some video, some neither; see
        ``_EncodedBatch``'s docstring for the measured defect this closes).

        A modality absent from this particular row range (``media_start ==
        media_end``, e.g. ``logprob_micro_batch=1`` landing on a text-only or
        other-modality row) OMITS that modality's keys entirely rather than
        passing a 0-row tensor. MEASURED (Qwen3.6-27B): passing an EMPTY
        ``pixel_values_videos``/``video_grid_thw`` pair (rather than leaving
        both keys out) still routes into the vision tower's forward, which
        crashes on the degenerate input (``RuntimeError: repeats must have
        the same size as input along dim, but got repeats.size(0) = 0 and
        input.size(0) = 1`` inside transformers' own
        ``get_vision_interpolation_indices_and_weights``) -- the same
        "OMIT, never pass an empty collection" rule
        ``_conversation_processor_batch_call``'s own docstring already
        states for the whole-batch ``images=``/``videos=`` kwargs, recurring
        here at the per-micro-batch-slice level.
        """
        width = end - start
        extra_kwargs: dict[str, Any] = {}
        if batch.modality_kwargs:
            row_count = int(batch.input_ids.shape[0])
            total_images = sum(batch.image_counts) if batch.image_counts is not None else None
            total_videos = sum(batch.video_counts) if batch.video_counts is not None else None
            buckets: dict[str, dict[str, Any]] = {"generic": {}, "image": {}, "video": {}}
            key_bucket: dict[str, str] = {}
            for key, value in batch.modality_kwargs.items():
                bucket = _classify_modality_key(
                    key,
                    value,
                    row_count=row_count,
                    total_images=total_images,
                    total_videos=total_videos,
                )
                buckets[bucket][key] = value
                key_bucket[key] = bucket
            # Reconciliation pass: a FLATTENED-PATCH value (Qwen3.6's
            # pixel_values/pixel_values_videos) has row-0 width equal to the
            # TOTAL PATCH COUNT, not total_images/total_videos, so the shape
            # check above cannot classify it directly -- it is reclassified
            # into whichever bucket its own paired COUNT key
            # (image_grid_thw/video_grid_thw, always one row per media item,
            # so the shape check DOES classify it) landed in. A pair whose
            # count key is absent (gemma has none) or was not classified as
            # image/video (an ordinary text-only batch) leaves the value key
            # exactly where the shape check already put it.
            for value_key, count_key in _FLATTENED_PATCH_PAIRS:
                count_bucket = key_bucket.get(count_key)
                if (
                    value_key in batch.modality_kwargs
                    and count_bucket is not None
                    and count_bucket != "generic"
                    and key_bucket.get(value_key) != count_bucket
                ):
                    buckets[key_bucket[value_key]].pop(value_key, None)
                    buckets[count_bucket][value_key] = batch.modality_kwargs[value_key]
                    key_bucket[value_key] = count_bucket
            if buckets["generic"]:
                extra_kwargs.update(
                    _narrow_modality_kwargs_by_row(buckets["generic"], start=start, end=end)
                )
            for name, counts in (("image", batch.image_counts), ("video", batch.video_counts)):
                media_kwargs = buckets[name]
                if not media_kwargs:
                    continue
                # counts is None means "uniform modality" (every row carries
                # exactly one, the RL trainer's own precondition): the
                # media-row range IS the training-row range, unchanged.
                # Heterogeneous batches carry real per-row counts (0 for a
                # row with none of this modality) and need the translation.
                if counts is None:
                    media_start, media_end = start, end
                else:
                    media_start = sum(counts[:start])
                    media_end = sum(counts[:end])
                if media_end == media_start:
                    continue
                extra_kwargs.update(
                    _narrow_modality_kwargs_by_row(media_kwargs, start=media_start, end=media_end)
                )
            # Diagnostic, same shape as rl.trainer's own "forwarding modality
            # keys to the scorer" line: per-rank (plain print() carries no
            # rank prefix under torchrun, unlike the NCCL error dumps), so a
            # hang's last line on each rank's own stream shows exactly which
            # modality that rank's micro-batch was about to force the model
            # to touch -- the evidence that confirms or refutes a per-rank
            # modality mismatch at the point of a collective timeout.
            print(
                f"[preference_trainer] rank={os.environ.get('RANK', '0')} "
                f"micro-batch rows [{start}, {end}) modality keys: "
                f"{', '.join(sorted(extra_kwargs)) or '(none)'}",
                file=sys.stderr,
            )
        logits = model(
            input_ids=batch.input_ids.narrow(0, start, width),
            attention_mask=batch.attention_mask.narrow(0, start, width),
            **extra_kwargs,
        ).logits
        return _token_logprobs(logits, batch.target_ids.narrow(0, start, width))

    def _measure_weight_delta(self) -> float:
        """Measure the fixed probe parameter's L2 movement in fp32.

        The probe is fixed at model load and its initial value is stored on
        the host, so the reading compares parameter value against parameter
        value rather than confusing wrapper-master arithmetic with model
        arithmetic.
        """
        import torch

        if self._probe_parameter is None or self._probe_initial is None:
            raise TrainerRefusal(
                "0 of 1 required probe parameters are bound; a completed "
                "step cannot report weight movement it never measured"
            )
        current = self._probe_parameter.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        return float(torch.linalg.vector_norm(current - self._probe_initial))

    def _estimate_kto_reference_point(
        self,
        *,
        policy_logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> float:
        """Estimate KTO's z0 once from the first measured batch.

        The estimate is the masked mean ``policy - reference`` over
        supervised tokens. It happens once, after the batch has been
        shaped, in no-grad tensors, and is recorded on every KTO report, so
        the reference point remains a measured input rather than changing
        with later batch composition.
        """
        import torch

        with torch.no_grad():
            differences = (policy_logprobs.detach() - reference_logprobs.detach()) * completion_mask
            denominator = completion_mask.sum()
            if not bool(denominator > 0):  # pragma: no cover -- encoding guarantees supervision
                raise TrainerRefusal(
                    "0 supervised tokens were available for the first KTO "
                    "batch; z0 would be a mean over an empty denominator"
                )
            return float(differences.sum() / denominator)

    def _one_step(
        self,
        *,
        step: int,
        records: Sequence[Mapping[str, Any]],
        policy_model: Any,
        reference_model: Any | None,
        tokenizer: Any,
        optimizer: Any,
        device: str,
        pad_token_id: int,
        loss_fn: TensorPreferenceLoss,
        ctx: DistContext | None = None,
    ) -> StepReport | None:
        """Perform one measured preference step, or return None if it is UNMEASURED.

        The reference pass and the micro-batched policy pass both price the
        same padded row layout. Metrics are computed before the optimiser
        moves, and the weight delta is computed after it moves; collapsing
        either pair would report evidence from a different parameter state.
        """
        import torch

        batch = self._encode_batch(
            records=records,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            device=device,
        )
        row_count = int(batch.input_ids.shape[0])
        micro_batch = self.config.logprob_micro_batch
        use_micro_batching = 0 < micro_batch < row_count
        row_slices: tuple[tuple[int, int], ...] = (
            tuple(
                (start, min(start + micro_batch, row_count))
                for start in range(0, row_count, micro_batch)
            )
            if use_micro_batching
            else ((0, row_count),)
        )
        # The micro-slice COUNT, and whether a rank slices AT ALL, must agree
        # across ranks: every narrowed forward() re-enters the model and so
        # re-fires one round of FSDP2's per-layer unshard all-gathers. A rank
        # that issues fewer forward() calls than a peer (either because its
        # row_count produced fewer ceil-divided slices, or because its
        # row_count fell at/under logprob_micro_batch while a peer's did not)
        # leaves that peer's later all-gathers short a participant, which
        # waits forever rather than erroring. MEASURED on 2x H200: a
        # rank0/rank1 prompt-level shard of 5/4 preference rows produced a
        # 3-slice rank beside a 2-slice rank and the run hung past a 90
        # minute collective timeout with rank0 still issuing all-gathers no
        # peer would answer; rl/trainer.py hit the identical class of bug for
        # GRPO and fixed it with agree_max plus a shared zero-weight pad
        # slice (_padding_backwards) -- this mirrors that fix rather than
        # re-deriving it.
        real_slices = len(row_slices)
        extra_slices = 0
        if ctx is not None and ctx.is_distributed:
            extra_slices = max(0, agree_max(real_slices, ctx) - real_slices)
            use_micro_batching = not agree_all(not use_micro_batching, ctx)
        pad_slice = (0, min(1, row_count))
        padded_slices = row_slices + (pad_slice,) * extra_slices

        if use_micro_batching:
            # One detached leaf lets the declared scalar see the complete
            # batch. Its backward is then delivered to parameter graphs one
            # fresh slice at a time by _micro_batched_backward. Padding
            # slices are forwarded too -- so every rank issues the same
            # number of all-gathers -- then dropped before concatenation.
            with torch.no_grad():
                policy_logprobs = (
                    torch.cat(
                        [
                            self._forward_logprobs(
                                model=policy_model,
                                batch=batch,
                                start=start,
                                end=end,
                            )
                            for start, end in padded_slices
                        ][:real_slices],
                        dim=0,
                    )
                    .detach()
                    .requires_grad_(True)
                )
        else:
            policy_logprobs = self._forward_logprobs(
                model=policy_model,
                batch=batch,
                start=0,
                end=row_count,
            ).requires_grad_(True)

        reference_logprobs: torch.Tensor | None = None
        if self._reference_required:
            if reference_model is None:  # pragma: no cover -- run() refuses earlier
                raise TrainerRefusal(
                    f"objective {type(self._objective).__name__} requires "
                    f"a frozen reference model but 0 were loaded"
                )
            with torch.no_grad():
                reference_logprobs = torch.cat(
                    [
                        self._forward_logprobs(
                            model=reference_model,
                            batch=batch,
                            start=start,
                            end=end,
                        )
                        for start, end in padded_slices
                    ][:real_slices],
                    dim=0,
                ).detach()

        kl_reference_point: float | None
        if isinstance(self._objective, KTOLoss):
            if reference_logprobs is None:  # pragma: no cover -- KTO requires a reference
                raise TrainerRefusal(
                    "KTO requires a measured reference plane before z0 can "
                    "be compared to beta(pi - ref)"
                )
            if self._kto_reference_point is None:
                self._kto_reference_point = self._estimate_kto_reference_point(
                    policy_logprobs=policy_logprobs,
                    reference_logprobs=reference_logprobs,
                    completion_mask=batch.completion_mask,
                )
            kl_reference_point = self._kto_reference_point
        else:
            kl_reference_point = None

        # A loss that refuses its batch (ORPO at p == 1, a non-finite
        # reading) is decided locally and acted on collectively: a rank-local
        # raise under FSDP kills or deadlocks every peer (#455).
        refusal: BatchRefusal | None = None
        try:
            loss_tensor = loss_fn(
                policy_logprobs=policy_logprobs,
                completion_mask=batch.completion_mask,
                reference_logprobs=reference_logprobs,
                kl_reference_point=kl_reference_point,
                desirable=batch.desirable,
            )
            paired_metrics = (
                preference_metrics(
                    self._objective,
                    policy_logprobs,
                    batch.completion_mask,
                    reference_logprobs=reference_logprobs,
                )
                if self._paired
                else None
            )
        except BatchRefusal as exc:
            refusal = exc
        if ctx is not None and not agree_all(refusal is None, ctx):
            where = f"rank {ctx.rank}: {refusal}" if refusal is not None else "a peer rank"
            print(
                f"[preference_trainer] step {step} UNMEASURED: the loss refused "
                f"its batch on {where}; every rank skips the update",
                file=sys.stderr,
            )
            return None
        if refusal is not None:
            raise refusal

        metric_values: dict[str, float]
        if paired_metrics is not None:
            metric_values = paired_metrics
            if set(metric_values) != {"accuracy", "margin"}:
                raise TrainerRefusal(
                    "preference_metrics did not return exactly the paired "
                    "accuracy and margin observations; an absent paired "
                    "metric is measured missing, never replaced with 0.0"
                )
        else:
            metric_values = preference_metrics(
                self._objective,
                policy_logprobs,
                batch.completion_mask,
                reference_logprobs=reference_logprobs,
            )
            if metric_values:
                raise TrainerRefusal(
                    f"preference_metrics returned paired metrics "
                    f"{sorted(metric_values)} for unpaired objective "
                    f"{type(self._objective).__name__}; SUP=KTO rows cannot "
                    f"support pair accuracy"
                )

        policy_forward_slice = lambda start, end: self._forward_logprobs(  # noqa: E731
            model=policy_model,
            batch=batch,
            start=start,
            end=end,
        )
        optimizer.zero_grad()
        if use_micro_batching:
            _micro_batched_backward(
                loss_tensor=loss_tensor,
                current_logprobs=policy_logprobs,
                row_slices=row_slices,
                forward_slice=policy_forward_slice,
            )
        else:
            loss_tensor.backward()
        # Padding passes: one collective-matching forward+backward per extra
        # slice, gradient exactly zero, so a rank with fewer real slices than
        # its busiest peer still issues the same number of all-gather /
        # reduce-scatter rounds. A no-op when extra_slices == 0 (every rank
        # agreed, including the single-pass path above).
        _padding_backwards(
            forward_slice=policy_forward_slice,
            extra_slices=extra_slices,
            n_rows=row_count,
        )
        optimizer.step()

        measured_loss = float(loss_tensor.detach())
        metric_observations: list[Any] = [
            MetricObservation(name=name, value=value) for name, value in metric_values.items()
        ]
        if isinstance(self._objective, KTOLoss):
            if batch.desirable is None:  # pragma: no cover -- KTO batches always build it
                raise TrainerRefusal(
                    "0 of 1 required KTO desirable tensors were present after encoding"
                )
            measured_z0 = self._kto_reference_point
            if measured_z0 is None:  # pragma: no cover -- pricing above measures it first
                raise TrainerRefusal(
                    "0 of 1 required KTO reference points are measured; "
                    "the pricing step above refuses before a report can "
                    "be built without one"
                )
            desirable_count = int(batch.desirable.sum())
            undesirable_count = int(batch.desirable.numel()) - desirable_count
            metric_observations.extend(
                (
                    MetricObservation(
                        name=self._objective.desirable_count_metric_name,
                        value=float(desirable_count),
                    ),
                    MetricObservation(
                        name=self._objective.undesirable_count_metric_name,
                        value=float(undesirable_count),
                    ),
                    MetricObservation(
                        name="kl_reference_point",
                        value=float(measured_z0),
                    ),
                )
            )
        loss_output = LossOutput(
            loss=measured_loss,
            components=_loss_components(objective=self._objective, total=measured_loss),
            metrics=tuple(metric_observations),
        )
        weight_delta = self._measure_weight_delta()
        self.history.append(
            {
                "step": step,
                "loss": measured_loss,
                "accuracy": metric_values.get("accuracy"),
                "margin": metric_values.get("margin"),
                "weight_delta_l2": weight_delta,
            }
        )
        return StepReport(
            step=step,
            loss=loss_output,
            rows=len(records),
            reward_stats=None,
            sync=None,
        )

    def run(self) -> list[StepReport]:
        """Run the offline preference loop and return measured step reports.

        WHAT IS CLAIMED: environment failures exit 96 with their dependency
        or surface named; every model is loaded only for a role the objective
        declares; records cycle in one seed-shuffled order; and vacuous runs
        refuse after the attempted/measured denominator is known.

        WHAT IS NOT CLAIMED: that the random or shuffled draw is sampled
        without replacement per epoch beyond the stated cycle, or that a
        media-carrying record whose column is NOT declared can be
        downgraded to text.
        """
        records = load_pairs_jsonl(self.config.dataset, paired=self._paired)
        declared_columns = {
            column
            for column in (self.config.image_column, self.config.video_column)
            if column is not None
        }
        carriers = tuple(
            (record[_LINE_KEY], key)
            for record in records
            for key in _MODALITY_KEYS
            if key not in declared_columns and record.get(key) is not None
        )
        if carriers:
            shown = ", ".join(f"line {line} ({key})" for line, key in carriers[:5])
            more = f" and {len(carriers) - 5} more" if len(carriers) > 5 else ""
            _refuse_exit_96(
                f"{len(carriers)} preference record field(s) carry image or "
                f"video data in an undeclared column: {shown}{more}. "
                f"The offline preference trainer is text-only for any "
                f"undeclared field; declare image_column/video_column to "
                f"train on them, or drop the field(s) -- never guessed, and "
                f"training on the prompt alone would silently drop the "
                f"modality"
            )
        try:
            import torch
        except ImportError:
            _refuse_exit_96(
                "1 of 2 required dependencies absent: torch; the tensor "
                "preference plane needs it and no pure-python weight "
                "update fallback exists"
            )
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            _refuse_exit_96(
                "1 of 2 required dependencies absent: transformers; models "
                "are loaded through AutoModelForCausalLM and templates "
                "through apply_chat_template"
            )

        ctx = init_distributed(self.config.sharding)
        if self.config.pairs_per_step < ctx.world_size:
            # shard_indices would hand every rank ZERO pairs; refuse
            # identically on every rank rather than train on nothing.
            destroy(ctx)
            raise TrainerRefusal(
                f"pairs_per_step={self.config.pairs_per_step} < world_size="
                f"{ctx.world_size}: sharding is over pairs, so some rank would "
                "hold none; raise pairs_per_step to at least the world size"
            )
        device: str | None
        if ctx.is_distributed:
            device = str(ctx.device)
        else:
            device = self.config.device
            if device is None:
                device = (
                    "mps"
                    if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
        if ctx.is_distributed and self.config.pairs_per_step < ctx.world_size:
            raise TrainerRefusal(
                f"field pairs_per_step={self.config.pairs_per_step}: fewer "
                f"records than the {ctx.world_size} data-parallel ranks "
                f"would leave a rank with 0 rows; the preference step "
                f"denominator must cover every rank"
            )
        torch.manual_seed(self.config.seed)

        try:
            prompt_surface = resolve_prompt_surface(self.config.model, self._needs_media)
            tokenizer = (
                prompt_surface.surface
                if prompt_surface.kind == "tokenizer"
                else prompt_surface.surface.tokenizer
            )
        except Exception as exc:  # noqa: BLE001 -- load-surface failure is a refusal
            _refuse_exit_96(f"tokenizer load failed for {self.config.model!r}: {exc}")
        if self._needs_media:
            # resolve_prompt_surface(..., True) REFUSES (96) rather than
            # returning kind='tokenizer' (see its own docstring), so this is
            # a type narrowing, not a runtime branch: self._processor is
            # always the loaded AutoProcessor here.
            self._processor = prompt_surface.surface
            self._turn_markers = derive_turn_markers(self._processor)
        if not getattr(tokenizer, "chat_template", None):
            _refuse_exit_96(
                f"tokenizer for {self.config.model!r} has no chat "
                f"template: prompt construction refuses to concatenate "
                f"strings silently"
            )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                _refuse_exit_96(
                    f"tokenizer for {self.config.model!r} has neither a "
                    f"pad token nor an eos token to record as padding"
                )
            tokenizer.pad_token = tokenizer.eos_token
            print(
                f"[preference_trainer] tokenizer pad_token was absent for "
                f"{self.config.model!r}; using eos_token as the explicit "
                f"pad token",
                file=sys.stderr,
            )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            _refuse_exit_96(
                f"tokenizer for {self.config.model!r} still exposes no "
                f"pad_token_id after the recorded eos fallback"
            )
        # No generate() call exists here, so the generation invariant behind
        # the online trainer's left padding does not apply. Right padding is
        # stated to keep pads after each supplied completion and outside the
        # shifted supervision mask.
        tokenizer.padding_side = "right"
        inner_tokenizer = getattr(prompt_surface.surface, "tokenizer", None)
        if inner_tokenizer is not None:
            inner_tokenizer.padding_side = "right"

        policy_model: Any
        if self._needs_media:
            # _load_causal_lm picks AutoModelForImageTextToText
            # deterministically BEFORE from_pretrained and PROVES the loaded
            # instance can consume pixels (exit 96, naming the model type,
            # otherwise) -- the same defect-closing helper rl/trainer.py's
            # RLTrainer uses, reused unchanged rather than re-trying
            # AutoModelForCausalLM first the way the text-only branch below
            # still does for byte-identical historical behaviour.
            policy_model = _load_causal_lm(self.config.model, needs_images=True)
        else:
            try:
                # Annotated Any: transformers 5.x wraps ``from_pretrained``
                # in a decorator whose type does not survive inference, so
                # an unannotated binding makes the subsequent ``.to(device)``
                # read the device string as the wrapper's self argument.
                policy_model = AutoModelForCausalLM.from_pretrained(self.config.model)
            except Exception as exc:  # noqa: BLE001 -- named model-load refusal
                _refuse_exit_96(f"policy model load failed for {self.config.model!r}: {exc}")
        # adapter_notes is a dict a manifest-keeping caller could consume;
        # this trainer has no manifest of its own -- _apply_lora_adapter
        # already PRINTS the same facts to stderr, this module's existing
        # "record the config" surface (see the optimizer= print below).
        policy_model, _adapter_notes = _apply_lora_adapter(
            policy_model,
            adapter=self.config.adapter,
            adapter_rank=self.config.adapter_rank,
            adapter_alpha=self.config.adapter_alpha,
            adapter_targets=self.config.adapter_targets,
            adapter_dropout=self.config.adapter_dropout,
            log_prefix="[preference_trainer]",
        )
        if self.config.gradient_checkpointing:
            policy_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if self.config.adapter is not None:
                # peft freezes every base-model parameter, so the first
                # (embedding) activation in the checkpointed chain carries
                # requires_grad=False and torch.utils.checkpoint has nothing
                # to build a backward graph through. This hooks the input
                # embedding's output to require grad regardless -- the
                # standard peft + gradient-checkpointing pairing.
                policy_model.enable_input_require_grads()
        model_config = getattr(policy_model, "config", None)
        if model_config is not None:
            model_config.use_cache = False
        if self.config.sharding == "fsdp":
            # fp32 cast BEFORE sharding so the sharded parameters are fp32
            # masters; MixedPrecisionPolicy then casts to bf16 for compute.
            # This deliberately replaces MasterWeightOptimizer under fsdp.
            policy_model = policy_model.float()
            # shard_towers=False for a media-declared run: see wrap_fsdp2's
            # own docstring for the measured FSDP all-gather deadlock this
            # avoids when per-row modality is heterogeneous (declared
            # image_column/video_column rows mixed with text-only ones),
            # unlike the online RL trainer's uniform-one-image-per-prompt
            # rollouts, which keep the default.
            policy_model = wrap_fsdp2(policy_model, ctx, shard_towers=not self._needs_media)
        elif self.config.sharding == "ddp":
            policy_model.to(device)
            policy_model = wrap_ddp(policy_model, ctx)
        else:
            policy_model.to(device)
        policy_model.train()

        probe = next(
            (parameter for parameter in policy_model.parameters() if parameter.requires_grad),
            None,
        )
        if probe is None:
            raise TrainerRefusal(
                f"model {self.config.model!r}: 0 of its parameters require "
                f"grad; a frozen policy leaves no measured weight to train"
            )
        self._probe_parameter = probe
        self._probe_initial = probe.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        )

        reference_model: Any | None = None
        if self._reference_required:
            # This trainer has no distinct reference_model concept (unlike
            # RLTrainConfig): the reference is always self.config.model's own
            # initial weights, and there is no iterative refresh cadence
            # either, so _reference_plan is called with both held at values
            # that make it depend on adapter alone -- "disable_adapter" when
            # an adapter is declared, "second_copy" (the historical path)
            # when it is not. TrainerRefusal cannot actually fire here (that
            # needs refresh_every > 0, fixed at 0), but calling through the
            # shared function keeps the two trainers' dispatch provably the
            # same decision rather than two hand-copies of it.
            plan = _reference_plan(
                adapter=self.config.adapter,
                reference_model=None,
                model=self.config.model,
                refresh_every=0,
            )
            if plan == "disable_adapter":
                # adapter='lora': the frozen reference IS the base model
                # peft already wraps (see _AdapterDisabledReference's
                # docstring in rl/trainer.py).
                reference_model = _AdapterDisabledReference(policy_model)
                print(
                    f"[preference_trainer] reference: adapter='lora' -- "
                    f"reusing the policy with the adapter disabled instead "
                    f"of loading a second model copy for "
                    f"{type(self._objective).__name__}",
                    file=sys.stderr,
                )
            else:
                loaded_reference: Any
                if self._needs_media:
                    # Same defect-closing loader as the policy load above:
                    # a second AutoModelForCausalLM copy of a media-declared
                    # qwen3_5-family checkpoint would load without error and
                    # silently absorb pixel_values through **kwargs.
                    loaded_reference = _load_causal_lm(self.config.model, needs_images=True)
                else:
                    try:
                        # Same Any binding as the policy load, for the same
                        # wrapped-loader reason; the two-step build keeps the
                        # Optional plane free of the loader's inferred type.
                        loaded_reference = AutoModelForCausalLM.from_pretrained(self.config.model)
                    except Exception as exc:  # noqa: BLE001
                        _refuse_exit_96(
                            f"reference model load failed for "
                            f"{self.config.model!r}: {exc}; objective "
                            f"{type(self._objective).__name__} is "
                            f"reference-anchored, so substituting the policy "
                            f"readings would change its margin"
                        )
                loaded_reference.eval()
                loaded_reference.requires_grad_(False)
                if self.config.sharding == "fsdp":
                    # The reference is sharded too; its no-grad forwards still
                    # all-gather, so every rank must issue the same calls.
                    # shard_towers=False for the SAME reason as the policy
                    # load above -- the reference scores the identical
                    # heterogeneous rows the policy does.
                    loaded_reference = loaded_reference.float()
                    loaded_reference = wrap_fsdp2(
                        loaded_reference, ctx, shard_towers=not self._needs_media
                    )
                else:
                    # ddp: the reference is a plain replica, never DDP-wrapped.
                    loaded_reference.to(device)
                reference_model = loaded_reference
                print(
                    f"[preference_trainer] reference model loaded for "
                    f"{type(self._objective).__name__}; reference scores are "
                    f"recomputed under torch.no_grad each step",
                    file=sys.stderr,
                )
        else:
            print(
                f"[preference_trainer] objective "
                f"{type(self._objective).__name__} is reference-free; no "
                f"second model was loaded",
                file=sys.stderr,
            )

        if self.config.sharding == "fsdp":
            # MasterWeightOptimizer keeps host fp32 copies of parameters;
            # under FSDP2 the parameters ARE fp32 sharded DTensors, so host
            # copies would break the DTensor layout. Plain AdamW is used.
            use_masters = False
            master_reason = "fsdp: sharded fp32 DTensor params make AdamW the master-weight path"
        elif self.config.master_weights is not None:
            use_masters = self.config.master_weights
            master_reason = f"forced by config master_weights={self.config.master_weights}"
        else:
            use_masters = probe.dtype != torch.float32
            master_reason = (
                f"auto: params are {probe.dtype}, whose ulp exceeds the update at this lr"
                if use_masters
                else "auto: params already fp32, no mastering needed"
            )
        optimizer: Any
        if use_masters:
            optimizer = MasterWeightOptimizer(
                _trainable_parameters(policy_model),
                lr=self.config.learning_rate,
            )
        else:
            optimizer = torch.optim.AdamW(
                _trainable_parameters(policy_model),
                lr=self.config.learning_rate,
            )
        print(
            "[preference_trainer] optimizer="
            + ("MasterWeightOptimizer(host-fp32)" if use_masters else "AdamW(direct)")
            + f" param_dtype={probe.dtype} lr={self.config.learning_rate}"
            + f" reason={master_reason}",
            file=sys.stderr,
        )

        shuffled = list(records)
        random.Random(self.config.seed).shuffle(shuffled)
        loss_fn = TensorPreferenceLoss(objective=self._objective)
        reports: list[StepReport] = []
        cursor = 0
        # Prompt indices are sharded, never completion rows: every rank
        # receives the same count, the remainder is dropped deterministically,
        # and the global ordering agrees because every rank used the same seed.
        local_offsets = shard_indices(self.config.pairs_per_step, ctx)
        # Compare the GLOBAL kept count: the per-rank count is always below
        # pairs_per_step on >1 rank, which made this warn on every even split.
        if len(local_offsets) * ctx.world_size < self.config.pairs_per_step and is_main(ctx):
            print(
                f"[preference_trainer] pairs_per_step={self.config.pairs_per_step} "
                f"is not divisible by world_size={ctx.world_size}; "
                f"{self.config.pairs_per_step - len(local_offsets) * ctx.world_size} "
                f"record(s) per step are dropped deterministically",
                file=sys.stderr,
            )
        for step in range(self.config.max_steps):
            chunk = tuple(shuffled[(cursor + offset) % len(shuffled)] for offset in local_offsets)
            cursor += self.config.pairs_per_step
            report = self._one_step(
                step=step,
                records=chunk,
                policy_model=policy_model,
                reference_model=reference_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                device=device,
                pad_token_id=int(pad_token_id),
                loss_fn=loss_fn,
                ctx=ctx,
            )
            if report is not None:
                # An UNMEASURED step (None) was agreed on every rank, so all
                # ranks skip these collectives together.
                reports.append(report)
            if report is not None and ctx.is_distributed:
                # Means are averaged across ranks, counts are summed: every
                # rank ends the step holding the same global history entry.
                history_entry = self.history[-1]
                history_entry["loss"] = all_reduce_mean(float(report.loss.loss), ctx)
                history_entry["rows"] = int(all_reduce_sum(float(len(chunk)), ctx))
                if history_entry.get("accuracy") is not None:
                    history_entry["accuracy"] = all_reduce_mean(
                        float(history_entry["accuracy"]), ctx
                    )
                if history_entry.get("margin") is not None:
                    history_entry["margin"] = all_reduce_mean(float(history_entry["margin"]), ctx)
            if (
                self.config.save_dir is not None
                and self.config.save_every > 0
                and (step + 1) % self.config.save_every == 0
            ):
                # Collective under fsdp: every rank reaches this call in the
                # same step because the loop trip count is config-fixed.
                save_checkpoint(
                    policy_model,
                    prompt_surface.surface,
                    f"{self.config.save_dir}/step_{step + 1}",
                    ctx,
                    sharding=self.config.sharding,
                    step=step + 1,
                )
        if self.truncated_completion_count:
            print(
                f"[preference_trainer] truncated "
                f"{self.truncated_completion_count} completion side(s) to "
                f"max_length={self.config.max_length}",
                file=sys.stderr,
            )
        _refuse_vacuous_run(
            attempted=self.config.max_steps,
            measured=len(reports),
        )
        if self.config.save_dir is not None:
            save_checkpoint(
                policy_model,
                prompt_surface.surface,
                f"{self.config.save_dir}/final",
                ctx,
                sharding=self.config.sharding,
                step=self.config.max_steps,
            )
        # destroy is a no-op unless init_distributed above created a group.
        destroy(ctx)
        return reports
