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
from foundationscale.rl.interfaces import LossOutput
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
    MasterWeightOptimizer,
    TrainerRefusal,
    _loss_components,
    _micro_batched_backward,
    _refuse_exit_96,
    _refuse_vacuous_run,
    _token_logprobs,
)

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
    """

    target_ids: torch.Tensor
    completion_mask: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    desirable: torch.Tensor | None


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
        self.config = config
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
        """
        if not records:  # pragma: no cover -- run() always passes a non-empty chunk
            raise TrainerRefusal(
                "0 records were supplied for a configured step; an empty "
                "batch has no supervision denominator"
            )
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

    def _forward_logprobs(
        self,
        *,
        model: Any,
        batch: _EncodedBatch,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Return target log-probabilities for one half-open row slice."""
        width = end - start
        logits = model(
            input_ids=batch.input_ids.narrow(0, start, width),
            attention_mask=batch.attention_mask.narrow(0, start, width),
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
    ) -> StepReport:
        """Perform one measured preference step.

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

        if use_micro_batching:
            # One detached leaf lets the declared scalar see the complete
            # batch. Its backward is then delivered to parameter graphs one
            # fresh slice at a time by _micro_batched_backward.
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
                            for start, end in row_slices
                        ],
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
            reference_slices: tuple[tuple[int, int], ...]
            if 0 < micro_batch < row_count:
                reference_slices = row_slices
            elif micro_batch > 0:
                reference_slices = ((0, row_count),)
            else:
                reference_slices = ((0, row_count),)
            with torch.no_grad():
                reference_logprobs = torch.cat(
                    [
                        self._forward_logprobs(
                            model=reference_model,
                            batch=batch,
                            start=start,
                            end=end,
                        )
                        for start, end in reference_slices
                    ],
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

        loss_tensor = loss_fn(
            policy_logprobs=policy_logprobs,
            completion_mask=batch.completion_mask,
            reference_logprobs=reference_logprobs,
            kl_reference_point=kl_reference_point,
            desirable=batch.desirable,
        )

        metric_values: dict[str, float]
        if self._paired:
            metric_values = preference_metrics(
                self._objective,
                policy_logprobs,
                batch.completion_mask,
                reference_logprobs=reference_logprobs,
            )
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

        optimizer.zero_grad()
        if use_micro_batching:
            _micro_batched_backward(
                loss_tensor=loss_tensor,
                current_logprobs=policy_logprobs,
                row_slices=row_slices,
                forward_slice=lambda start, end: self._forward_logprobs(
                    model=policy_model,
                    batch=batch,
                    start=start,
                    end=end,
                ),
            )
        else:
            loss_tensor.backward()
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
            components=_loss_components(self._objective, measured_loss),
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
        media-carrying record can be downgraded to text.
        """
        records = load_pairs_jsonl(self.config.dataset, paired=self._paired)
        carriers = tuple(
            (record[_LINE_KEY], key)
            for record in records
            for key in _MODALITY_KEYS
            if record.get(key) is not None
        )
        if carriers:
            shown = ", ".join(f"line {line} ({key})" for line, key in carriers[:5])
            more = f" and {len(carriers) - 5} more" if len(carriers) > 5 else ""
            _refuse_exit_96(
                f"{len(carriers)} preference record field(s) carry image or "
                f"video data: {shown}{more}. The offline preference "
                f"trainer is text-only; training on the prompt alone "
                f"would silently drop the modality"
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

        device = self.config.device
        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        torch.manual_seed(self.config.seed)

        try:
            prompt_surface = resolve_prompt_surface(self.config.model, False)
            tokenizer = (
                prompt_surface.surface
                if prompt_surface.kind == "tokenizer"
                else prompt_surface.surface.tokenizer
            )
        except Exception as exc:  # noqa: BLE001 -- load-surface failure is a refusal
            _refuse_exit_96(f"tokenizer load failed for {self.config.model!r}: {exc}")
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

        try:
            # Annotated Any: transformers 5.x wraps ``from_pretrained`` in a
            # decorator whose type does not survive inference, so an
            # unannotated binding makes the subsequent ``.to(device)`` read
            # the device string as the wrapper's self argument.
            policy_model: Any = AutoModelForCausalLM.from_pretrained(self.config.model)
        except Exception as exc:  # noqa: BLE001 -- named model-load refusal
            _refuse_exit_96(f"policy model load failed for {self.config.model!r}: {exc}")
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
            try:
                # Same Any binding as the policy load, for the same
                # wrapped-loader reason; the two-step build keeps the
                # Optional plane free of the loader's inferred type.
                loaded_reference: Any = AutoModelForCausalLM.from_pretrained(self.config.model)
            except Exception as exc:  # noqa: BLE001
                _refuse_exit_96(
                    f"reference model load failed for "
                    f"{self.config.model!r}: {exc}; objective "
                    f"{type(self._objective).__name__} is "
                    f"reference-anchored, so substituting the policy "
                    f"readings would change its margin"
                )
            loaded_reference.to(device)
            loaded_reference.eval()
            loaded_reference.requires_grad_(False)
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

        if self.config.master_weights is not None:
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
                policy_model.parameters(),
                lr=self.config.learning_rate,
            )
        else:
            optimizer = torch.optim.AdamW(
                policy_model.parameters(),
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
        for step in range(self.config.max_steps):
            chunk = tuple(
                shuffled[(cursor + offset) % len(shuffled)]
                for offset in range(self.config.pairs_per_step)
            )
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
            )
            reports.append(report)
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
        return reports
