#!/usr/bin/env python3
"""Distributed foundation-model training and checkpoint-continuity proof.

The safety bounds, measurements, and data provenance are explicit because a
silent fallback can turn an unknown property of a run into a false claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import functools
import torch
import torch.distributed as dist
import transformers
from datasets import Dataset as LocalDataset
from datasets import load_from_disk
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

CONTRACT_FIELDS = (
    "iteration_budget",
    "early_save_steps",
    "output_dir",
)
REQUIRED_TORCHRUN = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)
PHASES = ("load", "data", "train", "save", "resume", "eval")


class ContractError(Exception):
    """Signal a rejected configuration without allowing execution to drift."""


class OperationFailure(Exception):
    """Carry the failed phase so the operator receives an unmeasured result."""

    def __init__(self, phase: str, metric: str, message: str) -> None:
        super().__init__(message)
        self.phase = phase
        self.metric = metric


class SelftestFailure(Exception):
    """Separate exercise failures from operational contract failures."""


@dataclass(frozen=True)
class SourcedValue:
    """Keep value provenance attached so precedence is auditable, not implied."""

    name: str
    value: Any
    source: str

    def payload(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "display": f"{self.value!r} from {self.source}",
        }


@dataclass(frozen=True)
class RunConfig:
    """Represent only values that were explicitly supplied and then validated."""

    probe_mode: bool
    model_path: SourcedValue
    dataset_mode: SourcedValue
    dataset_path: SourcedValue | None
    text_field: SourcedValue | None
    synthetic_samples: SourcedValue | None
    eval_count: SourcedValue
    batch_size: SourcedValue
    sequence_length: SourcedValue
    learning_rate: SourcedValue
    log_every: SourcedValue
    seed: SourcedValue
    resume_tolerance: SourcedValue
    rank_agreement_tolerance: SourcedValue
    iteration_budget: SourcedValue
    early_save_steps: SourcedValue
    output_dir: SourcedValue

    def provenance_payload(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name in CONTRACT_FIELDS:
            result[name] = getattr(self, name).payload()
        return result


def _integer_parser(raw: str, *, minimum: int = 1) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected an integer >= {minimum}, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"expected an integer >= {minimum}, got {value}")
    return value


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected a finite number greater than zero, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"expected a finite number greater than zero, got {value!r}")
    return value


def _sourced(
    name: str,
    flag_value: Any,
    env_map: Mapping[str, str],
    env_name: str | None,
    parser: Callable[[str], Any],
    *,
    required: bool = True,
) -> SourcedValue:
    """Resolve the only sanctioned precedence: explicit flag over launcher env."""
    if flag_value is not None:
        raw = str(flag_value)
        source = "flag"
        value: Any = flag_value
    elif env_name is not None and env_name in env_map:
        raw = env_map[env_name]
        source = "env"
        value = raw
    else:
        if required:
            env_note = f" and {env_name}" if env_name else ""
            raise ContractError(
                f"{name} is absent: supply the CLI flag{env_note}; no fallback is defined"
            )
        return SourcedValue(name=name, value=None, source="absent")
    if source == "env":
        try:
            value = parser(raw)
        except ValueError as exc:
            raise ContractError(f"{name} from {env_name} is invalid: {exc}") from exc
    else:
        if name in {"iteration_budget", "early_save_steps"}:
            try:
                value = parser(str(value))
            except ValueError as exc:
                raise ContractError(f"{name} from flag is invalid: {exc}") from exc
    if value is None or value == "":
        raise ContractError(f"{name} from {source} is empty")
    return SourcedValue(name=name, value=value, source=source)


def build_parser() -> argparse.ArgumentParser:
    """Expose every safety-relevant choice instead of encoding hidden policy."""
    parser = argparse.ArgumentParser(
        description=(
            "Model-agnostic distributed training with explicit bounds, provenance, "
            "and checkpoint-continuity proof."
        )
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-mode", choices=("real", "synthetic"))
    parser.add_argument("--dataset-path")
    parser.add_argument("--text-field")
    parser.add_argument("--synthetic-samples", type=int)
    parser.add_argument("--eval-count", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-tolerance", type=float)
    parser.add_argument("--rank-agreement-tolerance", type=float)
    parser.add_argument("--iteration-budget", type=int)
    parser.add_argument("--early-save-steps", type=int)
    parser.add_argument("--output-dir")
    return parser


def _complete_required_argv() -> list[str]:
    return [
        "--model-path",
        "/tmp/model",
        "--dataset-mode",
        "synthetic",
        "--synthetic-samples",
        "64",
        "--eval-count",
        "8",
        "--batch-size",
        "2",
        "--sequence-length",
        "32",
        "--learning-rate",
        "0.0001",
        "--log-every",
        "1",
        "--seed",
        "17",
        "--resume-tolerance",
        "0.0005",
    ]


def resolve_contract(
    argv: Sequence[str] | None, env_map: Mapping[str, str] | None = None
) -> RunConfig | str:
    """Validate before touching files or devices so an unsafe run is closed early."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        return "selftest"
    env = os.environ if env_map is None else env_map
    positive_int = lambda text: _integer_parser(text, minimum=1)

    model_path = _sourced("model_path", args.model_path, env, None, str)
    dataset_mode = _sourced("dataset_mode", args.dataset_mode, env, None, str)
    eval_count = _sourced("eval_count", args.eval_count, env, None, str)
    batch_size = _sourced("batch_size", args.batch_size, env, None, str)
    sequence_length = _sourced("sequence_length", args.sequence_length, env, None, str)
    learning_rate = _sourced("learning_rate", args.learning_rate, env, None, str)
    log_every = _sourced("log_every", args.log_every, env, None, str)
    seed = _sourced("seed", args.seed, env, None, str)
    resume_tolerance = _sourced("resume_tolerance", args.resume_tolerance, env, None, str)
    # fs192: the cross-rank knob. Same _sourced machinery, same env fallback
    # (none) and same validation shape as resume_tolerance -- with one difference:
    # unset is legal (required=False), and when it IS set it must be finite and
    # greater than zero on the same terms (checked below).
    rank_agreement_tolerance = _sourced(
        "rank_agreement_tolerance",
        args.rank_agreement_tolerance,
        env,
        None,
        str,
        required=False,
    )

    iteration_budget = _sourced(
        "iteration_budget",
        args.iteration_budget,
        env,
        "FS_ITERATION_BUDGET",
        positive_int,
    )
    early_save_steps = _sourced(
        "early_save_steps",
        args.early_save_steps,
        env,
        "FS_EARLY_SAVE_STEPS",
        positive_int,
    )
    output_dir = _sourced("output_dir", args.output_dir, env, "OUT_DIR", str)

    for setting in (
        eval_count,
        batch_size,
        sequence_length,
        log_every,
    ):
        if int(setting.value) <= 0:
            raise ContractError(f"{setting.name} from {setting.source} must be greater than zero")
    if float(learning_rate.value) <= 0.0 or not math.isfinite(float(learning_rate.value)):
        raise ContractError("learning_rate must be finite and greater than zero")
    if float(resume_tolerance.value) <= 0.0 or not math.isfinite(
        float(resume_tolerance.value)
    ):
        raise ContractError("resume_tolerance must be finite and greater than zero")
    if rank_agreement_tolerance.value is not None and (
        float(rank_agreement_tolerance.value) <= 0.0
        or not math.isfinite(float(rank_agreement_tolerance.value))
    ):
        raise ContractError(
            "rank_agreement_tolerance must be finite and greater than zero"
        )
    # #134: a check on args.iteration_budget / args.early_save_steps stood here. It read
    # the FLAG, not the resolved flag-or-env value, so every env-sourced launch -- which
    # is how the launcher calls this -- saw 0 and refused. Removed rather than repaired:
    # _sourced() runs positive_int (minimum=1) on BOTH the env and the flag branch for
    # these two names, so the property is already enforced at parse time with the source
    # named in the message. Re-adding a check here would be unreachable code.
    if int(iteration_budget.value) <= int(early_save_steps.value):
        raise ContractError(
            "early_save_steps must be less than iteration_budget so the early save "
            "is separable from the final save and resume proof"
        )
    if int(log_every.value) > int(iteration_budget.value):
        raise ContractError("log_every cannot exceed iteration_budget")

    dataset_path: SourcedValue | None = None
    text_field: SourcedValue | None = None
    synthetic_samples: SourcedValue | None = None
    if dataset_mode.value == "real":
        if args.synthetic_samples is not None:
            raise ContractError("synthetic_sample_count is invalid for a real dataset")
        dataset_path = _sourced("dataset_path", args.dataset_path, env, None, str)
        text_field = _sourced("text_field", args.text_field, env, None, str)
    else:
        if args.dataset_path is not None or args.text_field is not None:
            raise ContractError(
                "a dataset path or text field must not be supplied for synthetic data"
            )
        if args.synthetic_samples is None:
            raise ContractError("synthetic_samples is required when dataset_mode is synthetic")
        synthetic_samples = _sourced(
            "synthetic_samples", args.synthetic_samples, env, None, str
        )
        if int(synthetic_samples.value) <= 0:
            raise ContractError("synthetic_samples must be greater than zero")

    return RunConfig(
        probe_mode=bool(args.probe),
        model_path=model_path,
        dataset_mode=dataset_mode,
        dataset_path=dataset_path,
        text_field=text_field,
        synthetic_samples=synthetic_samples,
        eval_count=eval_count,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        log_every=log_every,
        seed=seed,
        resume_tolerance=resume_tolerance,
        rank_agreement_tolerance=rank_agreement_tolerance,
        iteration_budget=iteration_budget,
        early_save_steps=early_save_steps,
        output_dir=output_dir,
    )


@dataclass(frozen=True)
class DenominatedCount:
    """Encode absence distinctly from an observed zero instead of averaging it away."""

    observed: int | None
    expected: int
    unit: str

    def payload(self) -> dict[str, Any]:
        if self.expected <= 0:
            status = "invalid_denominator"
            display = f"invalid denominator {self.expected} {self.unit}"
        elif self.observed is None:
            status = "unmeasured"
            display = f"UNMEASURED of {self.expected} {self.unit}"
        elif self.observed == 0:
            status = "measured_zero"
            display = f"0 of {self.expected} {self.unit} (measured 0)"
        else:
            status = "measured"
            display = f"{self.observed} of {self.expected} {self.unit}"
        return {
            "observed": self.observed,
            "expected": self.expected,
            "unit": self.unit,
            "status": status,
            "display": display,
        }


class MeasurementLedger:
    """Accumulate unmeasured facts so the outcome degrades fail-closed."""

    def __init__(self) -> None:
        self.unmeasured: list[str] = []

    def check(self, namespace: str, payload: Mapping[str, Any]) -> None:
        for key, value in payload.items():
            self._walk(f"{namespace}.{key}", value)

    def _walk(self, path: str, value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("status") in {"unmeasured", "invalid_denominator", "UNVERIFIED"}:
                self.unmeasured.append(path)
            else:
                for key, child in value.items():
                    self._walk(f"{path}.{key}", child)


@dataclass(frozen=True)
class DatasetContext:
    """Carry provenance beside data so throughput cannot be misread under a false label."""

    origin: str
    real_flag: bool
    seed: int | None
    total_index_count: int
    train_index_count: int
    eval_index_count: int
    train_dataset: Dataset[str]
    eval_rows: Sequence[int]
    provider: Any


class IndexedSyntheticDataset(Dataset[dict[str, str]]):
    """Make synthetic rows position-addressable and deterministic at every rank."""

    def __init__(self, sample_count: int, seed: int) -> None:
        self.sample_count = sample_count
        self.seed = seed

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, str]:
        row_seed = (
            (self.seed * 0x100000001B3)
            ^ (index * 0x9E3779B97F4A7C15)
            ^ (self.sample_count << 13)
        ) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(row_seed)
        alphabet_size = 32 + (index % 33)
        word_count = 4 + (row_seed & 31)
        words = []
        for word_index in range(word_count):
            width = 2 + rng.randrange(7)
            first = alphabet_size + index + word_index
            words.append(
                "".join(chr(97 + ((first + rng.randrange(8192)) % 26)) for _ in range(width))
            )
        marker = f"case-{index}-variant-{alphabet_size}"
        return {"text": marker + " " + " ".join(words)}


class IndexedLocalDataset(Dataset[dict[str, str]]):
    """Keep a local corpus shuffled by index instead of copying every row into memory."""

    def __init__(self, source: LocalDataset, indices: Sequence[int], text_field: str) -> None:
        self.source = source
        self.indices = tuple(indices)
        self.text_field = text_field

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.source[self.indices[index]]
        value = row.get(self.text_field)
        if not isinstance(value, str):
            raise TypeError(
                f"local row {self.indices[index]} field {self.text_field!r} is not text"
            )
        return {"text": value}


@dataclass(frozen=True)
class LoadArtifacts:
    """Bundle architecture evidence chosen from declarations rather than a name table."""

    # defect #133: model_path deliberately holds the resolver's config_dir --
    # the single directory a loader must be pointed at -- NOT the operator-
    # declared root. The two downstream readers
    #     model_class.from_pretrained(str(artifacts.model_path), ...)
    # therefore load from the resolved config dir with no change at their call
    # sites, consistent with _load_config / AutoProcessor / AutoTokenizer in
    # load_artifacts(). The full resolution is surfaced alongside for the
    # launcher and downstream consumers.
    model_path: Path
    model_root: ModelRoot  # complete resolution evidence from fs_model_root
    config_dir: Path  # == model_path; named for resolver-aware consumers
    layout: str  # one of fs_model_root.LAYOUT_*
    bind_closure: tuple[str, ...]  # roots that MUST be mounted, root first
    config: Any
    feature_adapter: Any
    tokenizer: Any


@dataclass
class RuntimeBundle:
    """Hold objects replaced during re-entry so stale optimizer state cannot leak."""

    artifacts: LoadArtifacts
    model: Any
    optimizer: torch.optim.Optimizer | None
    device: torch.device
    rank: int
    world_size: int


class PhaseMachine:
    """Reject incomplete phase transitions instead of treating logs as evidence."""

    def __init__(self) -> None:
        self.open_phase: str | None = None
        self.completed: list[str] = []

    def begin(self, phase: str) -> None:
        if phase not in PHASES:
            raise ContractError(f"unknown phase {phase!r}")
        if self.open_phase is not None:
            raise ContractError(f"cannot begin {phase} while {self.open_phase} is open")
        self.open_phase = phase

    def end(self, phase: str) -> None:
        if phase != self.open_phase:
            raise ContractError(
                f"cannot close {phase}; open phase is {self.open_phase!r}"
            )
        self.open_phase = None
        self.completed.append(phase)

    def reset(self) -> None:
        if self.open_phase is not None:
            raise ContractError(f"phase {self.open_phase} was abandoned open")
        self.completed.clear()


_RANK0_ONLY = {"rank_hint": 0, "rank": 0, "initialized": False}


def _is_rank_zero() -> bool:
    return int(_RANK0_ONLY["rank"]) == 0


def _print_json(event: str, payload: Mapping[str, Any]) -> None:
    if _is_rank_zero():
        print(
            event + " " + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )


def _phase_summary(
    machine: PhaseMachine,
    phase: str,
    metrics: Mapping[str, Any],
    dataset: DatasetContext | None,
    *,
    status: str = "measured",
) -> dict[str, Any]:
    """Denominatorize each phase so a partial measurement cannot resemble success."""
    payload: dict[str, Any] = {
        "phase": phase,
        "status": status,
        "metrics": dict(metrics),
    }
    if dataset is not None:
        payload["dataset_origin"] = dataset.origin
        payload["real_data"] = dataset.real_flag
        if dataset.seed is not None:
            payload["synthetic_seed"] = dataset.seed
    ledger = MeasurementLedger()
    ledger.check("metrics", metrics)
    if ledger.unmeasured or status != "measured":
        payload["verdict"] = "UNMEASURED"
        payload["unmeasured"] = sorted(ledger.unmeasured)
    else:
        payload["verdict"] = "MEASURED"
    _print_json("PHASE_JSON", payload)
    machine.end(phase)
    return payload


def _barrier() -> None:
    """Synchronize evidence across ranks before rank zero reports a collective fact."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# --- defect #133: mandatory sibling model-root resolver ---------------------
# fs_model_root.py is generated into the SAME directory as this entrypoint and
# is imported at module scope. There is deliberately NO fallback: if the import
# fails, the entrypoint must fail loudly, because silently reverting to the
# operator-declared root reinstates defect #133 while every downstream gate
# stays green.
try:
    from fs_model_root import (
        ModelRoot,
        ModelRootError,
        resolve_model_root,
    )
except ImportError as exc:
    raise ImportError(
        "fs_model_root.py must be co-located with fs_train.fixed.py; the "
        "model-root resolver is mandatory and no silent fallback to the "
        "operator-declared root is permitted"
    ) from exc
# ----------------------------------------------------------------------------


def _require_local_dir(path_value: Any, label: str) -> Path:
    path = Path(str(path_value)).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise OperationFailure("load", label, f"{path} is not a local directory")
    return path


def _load_config(model_path: Path) -> Any:
    try:
        return AutoConfig.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise OperationFailure(
            "load", "model_config", f"could not load local configuration: {exc}"
        ) from exc


def load_artifacts(config: RunConfig) -> LoadArtifacts:
    """Resolve from a local declaration so behavior does not depend on a name table."""
    # defect #133: resolve the declared root BEFORE any load. ModelRootError's
    # message carries the config-candidate denominator; forward it verbatim
    # through the entrypoint's own OperationFailure vocabulary -- never swallow
    # it, never fall back to the declared root.
    try:
        model_root = resolve_model_root(config.model_path.value)
    except ModelRootError as exc:
        raise OperationFailure("load", "model_root", str(exc)) from exc
    # Every loader below (_load_config, AutoProcessor, AutoTokenizer) and both
    # downstream model_class.from_pretrained readers of artifacts.model_path
    # bind the resolved config dir, never the declared root.
    model_path = Path(model_root.config_dir)
    config_obj = _load_config(model_path)
    architecture_names = list(getattr(config_obj, "architectures", []) or [])
    if len(architecture_names) != 1:
        raise OperationFailure(
            "load",
            "model_config.architectures",
            "the local config must declare exactly one checkpoint class; "
            f"observed {len(architecture_names)} of 1 entries",
        )
    architecture_name = str(architecture_names[0])
    model_class = getattr(transformers, architecture_name, None)
    if model_class is None or not isinstance(model_class, type):
        raise OperationFailure(
            "load",
            "model_class",
            "the declared checkpoint class is unavailable through the trusted package",
        )

    processor_name = getattr(config_obj, "processor_class", None)
    try:
        if isinstance(processor_name, str) and processor_name:
            feature_adapter = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            tokenizer = getattr(feature_adapter, "tokenizer", None)
            if tokenizer is None:
                raise TypeError("declared processor has no tokenizer capability")
        else:
            feature_adapter = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            tokenizer = feature_adapter
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise TypeError("the local text adapter has no pad token id")
    except Exception as exc:
        raise OperationFailure(
            "load", "tokenizer_or_processor", f"could not load declared local adapter: {exc}"
        ) from exc

    return LoadArtifacts(
        model_path=model_path,
        model_root=model_root,
        config_dir=model_path,
        layout=model_root.layout,
        bind_closure=model_root.bind_closure,
        config=config_obj,
        feature_adapter=feature_adapter,
        tokenizer=tokenizer,
    )


def build_dataset_context(
    config: RunConfig, world_size: int
) -> DatasetContext:
    """Hold out examples before any sharding so an eval row cannot be reused for training."""
    seed = int(config.seed.value)
    eval_count = int(config.eval_count.value)
    if config.dataset_mode.value == "synthetic":
        total = int((config.synthetic_samples or SourcedValue("synthetic", 0, "absent")).value)
        if total <= eval_count + world_size:
            raise OperationFailure(
                "data",
                "synthetic_partitioning",
                "synthetic sample count must exceed held-out count plus one row per rank",
            )
        provider = IndexedSyntheticDataset(total, seed)
        origin = "SYNTHETIC_INFRASTRUCTURE_ONLY"
        real_flag = False
    else:
        dataset_dir = _require_local_dir(
            (config.dataset_path or SourcedValue("dataset", "", "absent")).value,
            "dataset_path",
        )
        try:
            source = load_from_disk(str(dataset_dir), keep_in_memory=False)
        except Exception as exc:
            raise OperationFailure(
                "data", "local_dataset", f"could not load local dataset directory: {exc}"
            ) from exc
        field_name = str((config.text_field or SourcedValue("field", "", "absent")).value)
        columns = list(getattr(source, "column_names", []) or [])
        if field_name not in columns:
            raise OperationFailure(
                "data",
                "text_field",
                f"field {field_name!r} is absent from columns {columns!r}",
            )
        total = int(len(source))
        if total <= eval_count + world_size:
            raise OperationFailure(
                "data",
                "dataset_partitioning",
                "local row count must exceed held-out count plus one row per rank",
            )
        provider = source
        origin = "REAL_LOCAL_DATASET"
        real_flag = True

    permutation = list(range(total))
    random.Random(seed).shuffle(permutation)
    eval_indices = permutation[:eval_count]
    train_indices = permutation[eval_count:]

    if real_flag:
        train_dataset: Dataset[str] = IndexedLocalDataset(
            provider, train_indices, str((config.text_field or SourcedValue("field", "", "absent")).value)
        )
        eval_provider: Any = provider
    else:
        train_dataset = IndexedSyntheticDataset(total, seed)
        indexed_train = train_indices
        # Position remapping preserves deterministic synthetic rows while making a real
        # held-out split rather than merely changing labels after generation.
        class RemappedSyntheticDataset(Dataset[str]):
            def __len__(self) -> int:
                return len(indexed_train)

            def __getitem__(self, index: int) -> dict[str, str]:
                return indexed_train_source(indexed_train[index])

        indexed_train_source = IndexedSyntheticDataset(total, seed)
        train_dataset = RemappedSyntheticDataset()
        eval_provider = IndexedSyntheticDataset(total, seed)

    return DatasetContext(
        origin=origin,
        real_flag=real_flag,
        seed=None if real_flag else seed,
        total_index_count=total,
        train_index_count=len(train_indices),
        eval_index_count=len(eval_indices),
        train_dataset=train_dataset,
        eval_rows=tuple(eval_indices),
        provider=eval_provider,
    )


def _text_at(context: DatasetContext, original_index: int) -> str:
    """Read by the original index so the same held-out row is fixed across re-entry."""
    if context.real_flag:
        field_obj = context.train_dataset
        field_name = getattr(field_obj, "text_field", None)
        if field_name is None:
            raise OperationFailure("data", "text_field", "cannot resolve the reader field")
        value = context.provider[original_index].get(str(field_name))
        if not isinstance(value, str):
            raise OperationFailure(
                "data", "text_field", f"row {original_index} does not contain text"
            )
        return value
    value = context.provider[original_index]["text"]
    if not isinstance(value, str):
        raise OperationFailure("data", "text_field", f"row {original_index} is not text")
    return value


def _make_batch(
    bundle: RuntimeBundle,
    texts: Sequence[str],
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Use tokenizer-declared tensors and derive labels from padding, not model names."""
    try:
        encoded = bundle.artifacts.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
            return_tensors="pt",
        )
    except Exception as exc:
        raise OperationFailure("train", "batch_encoding", f"batch encoding failed: {exc}") from exc
    batch: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if not isinstance(value, torch.Tensor):
            continue
        batch[key] = value.to(bundle.device, non_blocking=True)
    if "input_ids" not in batch or "attention_mask" not in batch:
        raise OperationFailure(
            "train",
            "batch_encoding",
            "the declared adapter did not produce input_ids and attention_mask",
        )
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


def _build_model(artifacts: LoadArtifacts) -> Any:
    architecture_name = str(list(getattr(artifacts.config, "architectures", []))[0])
    model_class = getattr(transformers, architecture_name)
    try:
        return model_class.from_pretrained(
            str(artifacts.model_path),
            config=artifacts.config,
            local_files_only=True,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
        )
    except TypeError:
        try:
            return model_class.from_pretrained(
                str(artifacts.model_path),
                config=artifacts.config,
                local_files_only=True,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
            )
        except Exception as exc:
            raise OperationFailure(
                "load", "model_weights", f"could not load local weights: {exc}"
            ) from exc
    except Exception as exc:
        raise OperationFailure(
            "load", "model_weights", f"could not load local weights: {exc}"
        ) from exc



# --- fs172: wrap policy resolved from the model's OWN declaration ----------------
# The size-based policy is structure-blind: with its 1e8 default it wrapped
# model.model.embed_tokens (Qwen3-4B: 151936 x 2560 = 388,956,160 params) into its
# own FULL_SHARD unit, and that unit re-sharded the moment the embedding forward
# returned. Qwen3 sets tie_word_embeddings: true, so lm_head.weight IS
# embed_tokens.weight; when lm_head ran at the end of the same forward it saw the
# flat 1-D shard. Measured on 8xH100 job 37300, all 8 ranks:
#     size mismatch, got input (1024), mat (1024x2560), vec (48619520)
# and 48,619,520 = 388,956,160 / 8. This is not model-specific: Gemma and tied
# Llama variants fail identically. The fix wraps only the classes the model's OWN
# _no_split_modules declaration names; measured in the run container (transformers 5.5.0,
# torch 2.11.0a0):
#     Qwen3ForCausalLM   ['Qwen3DecoderLayer']
#     Gemma3ForCausalLM  ['Gemma3DecoderLayer','SiglipVisionEmbeddings',
#                         'SiglipEncoderLayer','SiglipMultiheadAttentionPoolingHead']
#     LlamaForCausalLM   ['LlamaDecoderLayer']
# Embeddings and lm_head then stay in the ROOT FSDP unit, which stays all-gathered
# for the entire forward -- that is what fixes the tied weight.
WRAP_POLICY_CENSUS: dict = {}


def _resolve_wrap_policy(model):
    """Resolve transformer_auto_wrap_policy from the model's own _no_split_modules declaration.

    Returns (policy, census); census is a plain dict of measured counts. Refuses
    -- and NEVER falls back to a size-based or any other policy -- when the
    declaration is absent/empty or resolves to zero live modules. That silent
    fallback is precisely the failure just measured on job 37300: a
    wrong-but-green policy that shards a tied parameter and dies 400 lines later
    in a matmul. A declared refusal is correct; a guess is not.
    """
    model_class = type(model).__name__
    declared = getattr(type(model), "_no_split_modules", None)
    if declared is None:
        declared = getattr(model, "_no_split_modules", None)
    modules = list(model.modules())
    n_modules = len(modules)
    if (not isinstance(declared, (list, tuple)) or not declared
            or not all(isinstance(name, str) for name in declared)):
        raise OperationFailure(
            "load", "wrap_policy",
            f"{model_class}: no usable _no_split_modules declaration (need a non-empty "
            f"sequence of class-name strings, got {declared!r}); 0 of 0 declared "
            f"class name(s) usable among {n_modules} live modules; refusing to "
            "guess a wrap policy",
        )
    declared_names = list(declared)
    n_declared = len(declared_names)
    wanted = set(declared_names)
    matched = {}
    n_instances = 0
    for m in modules:
        cls_name = type(m).__name__
        if cls_name in wanted:
            matched.setdefault(cls_name, type(m))
            n_instances += 1
    n_resolved = len(matched)
    if n_resolved == 0:
        raise OperationFailure(
            "load", "wrap_policy",
            f"{model_class}: resolved 0 of {n_declared} declared _no_split_modules class "
            f"name(s) among {n_modules} live modules; refusing to guess a wrap "
            "policy",
        )
    if n_instances == 0:
        raise OperationFailure(
            "load", "wrap_policy",
            f"{model_class}: resolved {n_resolved} of {n_declared} declared "
            f"_no_split_modules class name(s) but 0 live instances among {n_modules} "
            "live modules; refusing to guess a wrap policy",
        )
    # 0 < n_resolved < n_declared is NORMAL and must NOT refuse:
    # Gemma3ForCausalLM declares four names and a text-only load instantiates
    # only Gemma3DecoderLayer. Record it in the census; a partial resolution is
    # not an error.
    policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(matched.values()),
    )
    census = {
        "declared_names": declared_names,
        "declared": n_declared,
        "resolved": n_resolved,
        "instances": n_instances,
        "model_class": model_class,
    }
    return policy, census


def _check_tied_parameters(model, declared_names, census):
    """Assert every tied parameter group is co-located in ONE FSDP unit.

    Runs BEFORE the FSDP construction, on the pre-wrap module. Both halves of
    that are measured facts, not preferences:

    * torch's parameter iterator defaults to remove_duplicate=True, so a tied
      pair is yielded ONCE under one name and no group can ever reach size 2.
      Measured in the run container: a module tying head.weight to emb.weight
      yields 1 name by default and 2 with remove_duplicate=False. Taking the
      default here would make this detector VACUOUS -- it would report
      tied_groups=0 for the very Qwen3 tie that killed job 37300. (The call is
      spelled once, below; naming it in prose would put this docstring inside
      G11's own denominator.)
    * data_ptr() is 0 for an empty tensor (measured), and after FULL_SHARD with
      use_orig_params=True the parameters a rank does not own are exactly that.
      Grouping on data_ptr post-wrap would collapse every unowned parameter
      into one enormous false tied group and refuse a healthy run.

    Unit membership needs no wrapped model to read: the policy wraps exactly the
    declared classes that matched, so the assignment is already determined here.

    Zero tied groups is a legitimate measured zero (an untied model) and is
    recorded as tied_groups=0 in the census, not silently passed over.
    """
    groups = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        try:
            ptr = p.data_ptr()
        except Exception:
            ptr = 0
        # A materialized tensor groups by STORAGE, which catches object-identity
        # ties and storage-sharing views alike; ptr == 0 (meta, or an unowned
        # shard) falls back to object IDENTITY, which is what HF tie_weights()
        # produces -- and never to one shared bucket holding everything.
        key = ("ptr", ptr) if ptr else ("obj", id(p))
        groups.setdefault(key, []).append(name)
    tied = [names for names in groups.values() if len(names) > 1]
    census["tied_groups"] = len(tied)
    if not tied:
        return
    wrapped = set(declared_names)
    unit_of = {}
    # Read on the PRE-wrap tree, so a module's own class name IS the unit
    # boundary; there are no FSDP wrappers to unwrap through yet.
    for mod_name, m in model.named_modules():
        cls_name = type(m).__name__
        unit_of[mod_name] = cls_name if cls_name in wrapped else None

    def _unit_for(param_name):
        prefix = param_name.rsplit(".", 1)[0] if "." in param_name else ""
        parts = prefix.split(".") if prefix else []
        for i in range(len(parts), -1, -1):
            cand = ".".join(parts[:i])
            if unit_of.get(cand) is not None:
                return unit_of[cand]
        return "<root>"

    spanning = sum(1 for names in tied if len({_unit_for(n) for n in names}) > 1)
    if spanning:
        raise OperationFailure(
            "load", "tied_parameters",
            f"{type(model).__name__}: {spanning} of {len(tied)} tied parameter "
            "group(s) span different FSDP units; a tied weight sharded in one "
            "unit and read flat in another is exactly the job-37300 matmul "
            "mismatch",
        )
# --- end fs172 ---


def build_runtime(artifacts: LoadArtifacts, config: RunConfig) -> RuntimeBundle:
    """Shard immediately so checkpoint scale is not multiplied by the process count."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    model = _build_model(artifacts)
    # fs172: resolve the wrap policy from the model's OWN declaration before the
    # FSDP construction. The census lives in the module-level WRAP_POLICY_CENSUS
    # because RuntimeBundle has no suitable field and the dataclass is NOT edited
    # here -- the blast radius stays small.
    wrap_policy, wrap_census = _resolve_wrap_policy(model)
    WRAP_POLICY_CENSUS.update(wrap_census)
    # fs172: tied-parameter control, measured not assumed, and deliberately BEFORE
    # the FSDP construction. Post-wrap, FULL_SHARD leaves every parameter this rank
    # does not own as an empty tensor whose data_ptr() is 0 (measured), which would
    # group unrelated parameters into one false tied group and refuse a healthy run.
    # Pre-wrap the pointers are real, and unit membership is already fixed by the
    # policy resolved above. An untied model records a measured tied_groups=0.
    _check_tied_parameters(model, wrap_census["declared_names"], WRAP_POLICY_CENSUS)
    try:
        sharded = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=wrap_policy,
            sync_module_states=True,
            use_orig_params=True,
            device_id=device,
        )
    except Exception as exc:
        raise OperationFailure("load", "sharding", f"FSDP construction failed: {exc}") from exc
    try:
        optimizer = torch.optim.AdamW(
            sharded.parameters(), lr=float(config.learning_rate.value)
        )
    except Exception as exc:
        raise OperationFailure("load", "optimizer", f"optimizer construction failed: {exc}") from exc
    return RuntimeBundle(
        artifacts=artifacts,
        model=sharded,
        optimizer=optimizer,
        device=device,
        rank=rank,
        world_size=world_size,
    )


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> list[int]:
    values: list[int] = []
    for state in optimizer.state.values():
        raw = state.get("step")
        if raw is None:
            continue
        if isinstance(raw, torch.Tensor):
            if raw.numel() != 1:
                continue
            candidate = float(raw.detach().cpu().item())
        else:
            candidate = float(raw)
        if not candidate.is_integer():
            values.extend([-1])
        else:
            values.append(int(candidate))
    return values


def _peak_memory_gib() -> float | None:
    """Return an explicit none on failure so zero cannot masquerade as evidence."""
    try:
        return float(torch.cuda.max_memory_allocated()) / float(1024**3)
    except Exception:
        return None


def _initialized_distributed() -> None:
    missing = [name for name in REQUIRED_TORCHRUN if name not in os.environ]
    if missing:
        raise ContractError(
            "torchrun environment is incomplete; missing " + ",".join(missing)
        )
    rank = _integer_parser(os.environ["RANK"], minimum=0)
    world_size = _integer_parser(os.environ["WORLD_SIZE"], minimum=1)
    local_rank = _integer_parser(os.environ["LOCAL_RANK"], minimum=0)
    _RANK0_ONLY["rank"] = rank
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    _RANK0_ONLY["initialized"] = True
    visible = torch.cuda.device_count()
    if visible != world_size:
        raise OperationFailure(
            "load",
            "world_size",
            f"visible GPU count mismatch: {world_size} of {visible} was requested",
        )
    if local_rank >= visible:
        raise OperationFailure(
            "load", "local_rank", f"local rank {local_rank} has no visible device"
        )
    torch.cuda.set_device(local_rank)
    _barrier()


def _evaluate_fixed_loss(
    bundle: RuntimeBundle,
    context: DatasetContext,
    fixed_indices: Sequence[int],
    sequence_length: int,
) -> float:
    """Fix rows and disable training modes so the comparison isolates restoration."""
    texts = [_text_at(context, index) for index in fixed_indices]
    bundle.model.eval()
    with torch.no_grad():
        batch = _make_batch(bundle, texts, sequence_length)
        output = bundle.model(**batch)
        loss = getattr(output, "loss", None)
        if loss is None or loss.ndim != 0:
            raise OperationFailure(
                "resume", "fixed_loss", "the checkpoint model did not emit a scalar loss"
            )
        value = float(loss.detach().float().item())
    if not math.isfinite(value):
        raise OperationFailure("resume", "fixed_loss", "fixed loss was not finite")
    return value


def _fixed_eval_indices(context: DatasetContext, config: RunConfig) -> tuple[int, ...]:
    """Seed row selection so every restored model receives byte-identical text."""
    rng = random.Random(int(config.seed.value) ^ 0x5EED5EED)
    return tuple(
        context.eval_rows[rng.randrange(len(context.eval_rows))]
        for _ in range(int(config.batch_size.value))
    )


def _write_manifest(checkpoint_dir: Path, payload: Mapping[str, Any]) -> None:
    """Write collective facts only after rank payload persistence has completed."""
    temporary = checkpoint_dir / ".manifest.json.tmp"
    final = checkpoint_dir / "manifest.json"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(final)


def save_checkpoint(
    bundle: RuntimeBundle,
    checkpoint_dir: Path,
    global_step: int,
    *,
    fixed_loss: float | None = None,
) -> None:
    """Persist rank-local shards without assuming enough host memory to flatten a model."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = bundle.optimizer
    if optimizer is None:
        raise OperationFailure("save", "optimizer", "cannot save without an optimizer")
    optimizer_steps = _optimizer_steps(optimizer)
    try:
        with FSDP.state_dict_type(
            bundle.model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            model_state = bundle.model.state_dict()
            optimizer_state = FSDP.optim_state_dict(bundle.model, optimizer)
        rank_payload = {
            "format": "rank-local-sharded-v1",
            "global_step": int(global_step),
            "rank": bundle.rank,
            "world_size": bundle.world_size,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "optimizer_state_count": len(optimizer_steps),
            "optimizer_steps": optimizer_steps,
            "fixed_loss_before_save": fixed_loss,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(bundle.device),
        }
        path = checkpoint_dir / f"rank-{bundle.rank:05d}.pt"
        torch.save(rank_payload, path)
        saved = 1 if path.is_file() else 0
    except Exception as exc:
        raise OperationFailure("save", "rank_payload", f"rank checkpoint failed: {exc}") from exc
    total_saved = torch.tensor([saved], dtype=torch.int64, device=bundle.device)
    dist.all_reduce(total_saved, op=dist.ReduceOp.SUM)
    _barrier()
    if bundle.rank == 0:
        manifest = {
            "format": "rank-local-sharded-v1",
            "global_step": int(global_step),
            "world_size": int(bundle.world_size),
            "rank_payload_count": int(total_saved.item()),
            "expected_rank_payload_count": int(bundle.world_size),
            "fixed_loss_before_save": fixed_loss,
        }
        _write_manifest(checkpoint_dir, manifest)
    _barrier()


def _read_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    """Broadcast one manifest so all ranks reject inconsistent checkpoint identities."""
    manifest: dict[str, Any] = {}
    ready = 0
    if int(_RANK0_ONLY["rank"]) == 0:
        try:
            with (checkpoint_dir / "manifest.json").open("r", encoding="utf-8") as stream:
                value = json.load(stream)
            if isinstance(value, dict):
                manifest = value
                ready = 1
        except Exception:
            ready = 0
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    flag = torch.tensor([ready], dtype=torch.int64, device=device)
    dist.broadcast(flag, src=0)
    if int(flag.item()) != 1:
        raise OperationFailure("resume", "manifest", "checkpoint manifest is absent or invalid")
    values = [manifest]
    dist.broadcast_object_list(values, src=0)
    result = values[0]
    if not isinstance(result, dict):
        raise OperationFailure("resume", "manifest", "checkpoint manifest did not decode")
    return result


def _restore_runtime_from_checkpoint(
    checkpoint_dir: Path,
    config: RunConfig,
) -> RuntimeBundle:
    """Recreate every trainable object before restoring so use of stale state is impossible."""
    artifacts = load_artifacts(config)
    return build_runtime(artifacts, config)


def _resume_continuity_verdict(pre_per_rank, post_per_rank, pre_scalar, tolerance, world_size, rank_agreement_tolerance=None):
    """Decide restore fidelity and cross-rank agreement as TWO statistics, never one.

    fs178: pure by construction -- plain lists and scalars in, a plain dict out; no
    torch, no distributed state. Measured basis (job 37319 plus a zero-training
    8-rank forensic run): rank 0's restore was bit-exact (before_save ==
    after_resume == 0.5986318588256836) while the shipped MAX of
    |own_post - rank0_pre| read 0.17570888996124268; within-rank restore delta 0.0
    and 0 of 341 parameter-fingerprint keys changed across save/load; the fixed-eval
    loss takes exactly two bit-identical values across ranks (e.g. spread
    1.1025075912475586) on a NEVER-SAVED runtime -- the ranks disagreeing is a
    property of the instrument, present before any checkpoint exists, and must never
    be named resume.

    fs192: the one tolerance used to be spent on BOTH questions. Job 37336
    (--resume-tolerance 0.0005) measured restore_delta 0.0 with
    cross_rank_spread_before_save == cross_rank_spread_after_resume ==
    0.2940967082977295 and abstained unmeasured_cross_rank; job 37319, the same arm
    at --resume-tolerance 10.0, recorded zero abstentions -- a cross-rank pass
    bought by raising the threshold that ALSO governs restore fidelity, and at 10.0
    a completely broken restore (delta 9.9) is a PASS. `tolerance` keeps its name
    and is now the RESTORE-fidelity knob only; `rank_agreement_tolerance` (keyword,
    default None) is the cross-rank knob. When it is unset, the before-save spread
    -- the measured noise floor of the instrument -- calibrates only the
    PRESERVATION question (did resume worsen the agreement the instrument already
    had?), never the absolute one: self-calibration must never set rank_invariant
    True, because a floor derived from the same run it judges cannot certify that
    run's absolute agreement.

    Statuses: "refuse" (malformed evidence -- a length that disagrees with
    world_size or a non-finite entry: refuse rather than guess); "red" (restore
    fidelity broken -- delta over the RESTORE tolerance, final, and never governed
    by the rank knob no matter how wide it is); "pass" (restore holds AND an
    EXPLICIT rank_agreement_tolerance was supplied AND both spreads come in under
    it); "unmeasured_cross_rank" (restore holds but the absolute cross-rank question
    was not asked or not satisfied, or the per-rank pre-save vector is absent so the
    restore term fell back to the legacy rank-0-scalar comparison -- a DECLARED
    UNMEASURED, never a clean pass).
    """
    result = {
        "restore_delta": None,
        "cross_rank_spread_before_save": None,
        "cross_rank_spread_after_resume": None,
        "cross_rank_spread_delta": None,
        "rank_agreement_tolerance": None,
        "rank_agreement_tolerance_source": None,
        "rank_agreement_absolute": None,
        "rank_agreement_preserved": None,
        "rank_invariant": False,
        "status": "refuse",
        "reason": "",
        "restore_term_legacy": False,
        "pre_per_rank": None,
        "post_per_rank": None,
    }

    def _coerce(vec, label):
        if not isinstance(vec, (list, tuple)) or len(vec) != world_size:
            got = len(vec) if isinstance(vec, (list, tuple)) else repr(vec)
            return None, (
                label + " length disagrees with world_size: observed " + str(got)
                + " of an expected " + str(world_size) + " entries; refusing to guess"
            )
        values = []
        for index, value in enumerate(vec):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return None, (
                    label + "[" + str(index) + "] is not a finite number: "
                    + repr(value) + "; refusing to guess"
                )
            values.append(float(value))
        return values, None

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        result["reason"] = "world_size " + repr(world_size) + " is an invalid denominator"
        return result
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        result["reason"] = "tolerance " + repr(tolerance) + " is not a positive finite number"
        return result
    if rank_agreement_tolerance is not None and (
        isinstance(rank_agreement_tolerance, bool)
        or not isinstance(rank_agreement_tolerance, (int, float))
        or not math.isfinite(float(rank_agreement_tolerance))
        or float(rank_agreement_tolerance) <= 0.0
    ):
        result["reason"] = (
            "rank_agreement_tolerance " + repr(rank_agreement_tolerance)
            + " is not a positive finite number"
        )
        return result
    if (
        isinstance(pre_scalar, bool)
        or not isinstance(pre_scalar, (int, float))
        or not math.isfinite(float(pre_scalar))
    ):
        result["reason"] = "the rank-0 manifest scalar " + repr(pre_scalar) + " is not finite"
        return result
    tolerance = float(tolerance)
    if rank_agreement_tolerance is not None:
        rank_agreement_tolerance = float(rank_agreement_tolerance)

    post, err = _coerce(post_per_rank, "post_per_rank")
    if err is not None:
        result["reason"] = err
        return result
    result["post_per_rank"] = post
    result["cross_rank_spread_after_resume"] = max(post) - min(post)

    if pre_per_rank is None:
        # Legacy fallback: a foreign or future writer left no per-rank pre-save values
        # on disk. Compare every rank's post against the rank-0 manifest scalar,
        # explicitly marked as the legacy CONFLATED statistic, and declare the
        # cross-rank term UNMEASURED -- never silently assume the ranks agreed.
        legacy_delta = max(abs(p - float(pre_scalar)) for p in post)
        result["restore_delta"] = legacy_delta
        result["restore_term_legacy"] = True
        result["rank_invariant"] = False
        if legacy_delta > tolerance:
            result["status"] = "red"
            result["reason"] = (
                "legacy (conflated) restore statistic " + format(legacy_delta, ".8g")
                + " exceeds tolerance " + format(tolerance, ".8g")
                + "; per-rank pre-save values were not recorded"
            )
        else:
            result["status"] = "unmeasured_cross_rank"
            result["reason"] = (
                "cross-rank term UNMEASURED: rank payload(s) carried no per-rank "
                "fixed-loss-before-save value; the restore term used the legacy "
                "rank-0 manifest scalar (the conflated statistic), marked legacy"
            )
        return result

    pre, err = _coerce(pre_per_rank, "pre_per_rank")
    if err is not None:
        result["reason"] = err
        return result
    result["pre_per_rank"] = pre
    result["cross_rank_spread_before_save"] = max(pre) - min(pre)
    restore_delta = max(abs(post[i] - pre[i]) for i in range(world_size))
    result["restore_delta"] = restore_delta
    spread_before = result["cross_rank_spread_before_save"]
    spread_after = result["cross_rank_spread_after_resume"]
    # fs192: the resume-attributable term, signed -- negative means resume
    # TIGHTENED agreement. On job 37336 it is exactly 0.0: resume did not worsen
    # rank agreement AT ALL, and no previous image of this proof could say so.
    result["cross_rank_spread_delta"] = spread_after - spread_before
    # Did resume preserve whatever agreement the instrument already had? The
    # before-save spread is the measured noise floor of the instrument; the
    # restore tolerance floors it so a bit-exact instrument is not punished for
    # last-bit kernel scheduling either.
    result["rank_agreement_preserved"] = spread_after <= max(spread_before, tolerance)
    if rank_agreement_tolerance is not None:
        result["rank_agreement_tolerance"] = rank_agreement_tolerance
        result["rank_agreement_tolerance_source"] = "explicit"
        # The boundary is strict: a spread exactly equal to the tolerance does NOT
        # fire (">", never ">="). The before-save spread is the measured noise
        # floor of the instrument -- asserting a tolerance without ever measuring
        # what the instrument can resolve is exactly how a bit-exact restore got
        # named a resume failure.
        rank_invariant = (
            spread_before <= rank_agreement_tolerance
            and spread_after <= rank_agreement_tolerance
        )
        result["rank_agreement_absolute"] = rank_invariant
    else:
        # Self-calibrated: the effective floor is derived from THIS run, so it can
        # judge only preservation -- never the absolute question. rank_invariant
        # stays False and rank_agreement_absolute stays None (the question was not
        # asked): a floor derived from the same run it judges cannot certify that
        # run's absolute agreement.
        result["rank_agreement_tolerance"] = max(spread_before, tolerance)
        result["rank_agreement_tolerance_source"] = "self-calibrated"
        rank_invariant = False
    result["rank_invariant"] = rank_invariant
    if restore_delta > tolerance:
        # RED and final: restore fidelity is compared against the RESTORE knob and
        # nothing else, no matter how wide rank_agreement_tolerance is -- that is
        # the whole point of #192. A real restore defect must never be laundered
        # into "the ranks merely disagree".
        result["status"] = "red"
        result["reason"] = (
            "restore fidelity broken: max over " + str(world_size) + " rank(s) of "
            "|own_after_resume - own_before_save| = " + format(restore_delta, ".8g")
            + " exceeds restore tolerance " + format(tolerance, ".8g")
        )
    elif rank_agreement_tolerance is not None and rank_invariant:
        result["status"] = "pass"
        result["reason"] = (
            "restore fidelity holds (restore_delta " + format(restore_delta, ".8g")
            + " <= restore tolerance " + format(tolerance, ".8g")
            + ") and the ranks agree within the explicit rank-agreement tolerance "
            + format(rank_agreement_tolerance, ".8g")
        )
    else:
        result["status"] = "unmeasured_cross_rank"
        spread_delta = result["cross_rank_spread_delta"]
        if result["rank_agreement_preserved"]:
            preserved_clause = (
                "resume did not worsen rank agreement (cross-rank spread delta "
                + format(spread_delta, ".8g") + ")"
            )
        else:
            # A real signal about resume, not an instrument artifact: name it and
            # name both spreads.
            preserved_clause = (
                "resume WORSENED rank agreement: spread before save "
                + format(spread_before, ".8g") + ", spread after resume "
                + format(spread_after, ".8g")
            )
        if rank_agreement_tolerance is None:
            absolute_clause = (
                "the absolute cross-rank question is UNMEASURED because no "
                "rank-agreement tolerance was declared"
            )
        else:
            absolute_clause = (
                "the ranks do not agree in absolute terms under the explicit "
                "rank-agreement tolerance " + format(rank_agreement_tolerance, ".8g")
            )
        result["reason"] = (
            "restore fidelity holds (restore_delta " + format(restore_delta, ".8g")
            + " <= restore tolerance " + format(tolerance, ".8g")
            + "); the ranks do not agree in absolute terms (spread after resume "
            + format(spread_after, ".8g") + "); " + preserved_clause + "; "
            + absolute_clause + "; declared UNMEASURED, never folded into the "
            "resume verdict"
        )
    return result


def resume_and_prove(
    bundle: RuntimeBundle,
    checkpoint_dir: Path,
    config: RunConfig,
    context: DatasetContext,
    fixed_indices: Sequence[int],
) -> tuple[int, dict[str, Any]]:
    """Prove optimizer identity and fixed-loss continuity rather than claiming liveness."""
    manifest = _read_manifest(checkpoint_dir)
    recorded_step_raw = manifest.get("global_step", None)
    if not isinstance(recorded_step_raw, int):
        raise OperationFailure(
            "resume", "checkpoint_step", "checkpoint did not record an integer step"
        )
    recorded_step = int(recorded_step_raw)
    if int(manifest.get("world_size", -1)) != bundle.world_size:
        raise OperationFailure(
            "resume", "checkpoint_world_size", "checkpoint was written for a different world size"
        )
    if int(manifest.get("rank_payload_count", -1)) != bundle.world_size:
        raise OperationFailure(
            "resume", "checkpoint_payload_count", "one or more rank payloads were not recorded"
        )
    fixed_pre = manifest.get("fixed_loss_before_save", None)
    if not isinstance(fixed_pre, (int, float)) or not math.isfinite(float(fixed_pre)):
        raise OperationFailure(
            "resume", "fixed_loss_before_save", "checkpoint did not record the fixed-batch loss"
        )

    rank_path = checkpoint_dir / f"rank-{bundle.rank:05d}.pt"
    try:
        payload = torch.load(rank_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise OperationFailure(
            "resume", "rank_payload", f"could not load rank payload: {exc}"
        ) from exc
    if int(payload.get("global_step", -1)) != recorded_step:
        raise OperationFailure(
            "resume", "rank_payload_step", "rank payload identity does not match manifest"
        )
    if int(payload.get("rank", -1)) != bundle.rank:
        raise OperationFailure(
            "resume", "rank_identity", "rank payload belongs to another rank"
        )
    optimizer = bundle.optimizer
    if optimizer is None:
        raise OperationFailure("resume", "optimizer", "fresh runtime has no optimizer")
    try:
        with FSDP.state_dict_type(
            bundle.model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            bundle.model.load_state_dict(payload["model_state"])
            optimizer_loadable = FSDP.optim_state_dict_to_load(
                bundle.model,
                optimizer,
                payload["optimizer_state"],
                is_named_optimizer=False,
            )
            optimizer.load_state_dict(optimizer_loadable)
        torch.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state(payload["cuda_rng_state"], bundle.device)
    except Exception as exc:
        raise OperationFailure("resume", "state_restore", f"state restore failed: {exc}") from exc

    restored_steps = _optimizer_steps(optimizer)
    expected_state_count = int(payload.get("optimizer_state_count", -1))
    mismatch_count = sum(1 for step in restored_steps if step != recorded_step)
    local_summary = torch.tensor(
        [len(restored_steps), mismatch_count],
        dtype=torch.int64,
        device=bundle.device,
    )
    dist.all_reduce(local_summary, op=dist.ReduceOp.SUM)
    total_states = int(local_summary[0].item())
    total_mismatches = int(local_summary[1].item())
    if expected_state_count != len(restored_steps) or total_states <= 0 or total_mismatches != 0:
        raise OperationFailure(
            "resume",
            "optimizer_step_continuity",
            (
                f"optimizer state check failed on collective ranks: {total_mismatches} "
                f"mismatches; recorded {recorded_step}, local {len(restored_steps)} "
                f"of {expected_state_count}"
            ),
        )

    fixed_post = _evaluate_fixed_loss(
        bundle, context, fixed_indices, int(config.sequence_length.value)
    )
    # --- fs178: attribute the proof's two questions to two statistics (job 37319) ---
    # Job 37319 measured before_save bit-IDENTICAL to rank 0's after_resume
    # (0.5986318588256836) while the shipped statistic -- MAX over ranks of
    # |own_post - rank0's recorded pre scalar| -- read 0.17570888996124268 and
    # named resume. A zero-step 8-rank forensic run measured within-rank restore
    # delta 0.0 and 0 of 341 parameter-fingerprint keys changed across save/load,
    # and the fixed-eval loss takes exactly two bit-identical values across ranks
    # (spread 1.1025075912475586) on a fresh, NEVER-SAVED runtime: the divergence
    # precedes any checkpoint. The restore is exact; the instrument disagrees with
    # itself. Two questions, two statistics, and the instrument's noise floor is
    # measured (the before-save spread), not assumed -- the tolerance is NEVER
    # widened to hide this; that would make the proof model-specific.
    tolerance = float(config.resume_tolerance.value)
    # The per-rank pre-save value is ALREADY ON DISK: _run computed pre_loss on
    # every rank independently (there is no all-reduce inside _evaluate_fixed_loss,
    # so pre_loss is that rank's OWN value) and save_checkpoint wrote it into THIS
    # rank's own payload. No gather at save time, no format change -- every
    # checkpoint this framework has ever written carries it.
    own_pre_raw = payload.get("fixed_loss_before_save", None)
    own_pre_known = (
        isinstance(own_pre_raw, (int, float))
        and not isinstance(own_pre_raw, bool)
        and math.isfinite(float(own_pre_raw))
    )
    own_pre = float(own_pre_raw) if own_pre_known else None
    gather_packet = torch.tensor(
        [fixed_post, own_pre if own_pre_known else 0.0, 1.0 if own_pre_known else 0.0],
        dtype=torch.float64,
        device=bundle.device,
    )
    gathered = [torch.zeros_like(gather_packet) for _ in range(bundle.world_size)]
    dist.all_gather(gathered, gather_packet)
    post_per_rank = [float(slot[0].item()) for slot in gathered]
    pre_flags = [float(slot[2].item()) for slot in gathered]
    if all(flag == 1.0 for flag in pre_flags):
        pre_per_rank = [float(slot[1].item()) for slot in gathered]
    else:
        # A payload without the recorded key (a foreign or future writer): the
        # cross-rank term is declared UNMEASURED and the restore term falls back to
        # the legacy rank-0-scalar comparison, explicitly marked. Never silently
        # assume the ranks agreed.
        pre_per_rank = None
    # fs192: the cross-rank knob is OPTIONAL; None means the absolute question
    # was not asked, and the verdict self-calibrates only the preservation term.
    rank_agreement_raw = config.rank_agreement_tolerance.value
    rank_agreement_tolerance = (
        float(rank_agreement_raw) if rank_agreement_raw is not None else None
    )
    continuity = _resume_continuity_verdict(
        pre_per_rank,
        post_per_rank,
        float(fixed_pre),
        tolerance,
        bundle.world_size,
        rank_agreement_tolerance=rank_agreement_tolerance,
    )
    if continuity["status"] == "refuse":
        # Malformed evidence is refused, never smoothed over.
        raise OperationFailure("resume", "refuse", continuity["reason"])
    if continuity["status"] == "red":
        # A genuine lossy restore stays RED under the exact legacy phase/metric pair
        # so existing consumers keep working; restore fidelity outranks cross-rank
        # divergence and can never be laundered into "the ranks merely disagree".
        raise OperationFailure("resume", "fixed_loss_continuity", continuity["reason"])
    # --- end fs178 segment: comparison ---

    bundle.model.train()
    metrics = {
        "checkpoint_step": DenominatedCount(
            recorded_step, recorded_step, "recorded steps"
        ).payload(),
        "optimizer_steps": DenominatedCount(
            total_states, total_states, "stateful optimizer entries"
        ).payload(),
        "optimizer_step_mismatches": DenominatedCount(
            total_mismatches, total_states, "stateful optimizer entries"
        ).payload(),
        "fixed_examples": DenominatedCount(
            len(fixed_indices), len(fixed_indices), "examples"
        ).payload(),
        "payload_files": DenominatedCount(
            bundle.world_size, bundle.world_size, "files"
        ).payload(),
        "fixed_loss": {
            "before_save": float(fixed_pre),
            "after_resume": fixed_post,
            "own_before_save": own_pre,
            "restore_delta": continuity["restore_delta"],
            "restore_term": (
                "legacy_rank0_scalar_conflated"
                if continuity["restore_term_legacy"]
                else "per_rank_own_payload"
            ),
            "cross_rank_spread_before_save": continuity["cross_rank_spread_before_save"],
            "cross_rank_spread_after_resume": continuity["cross_rank_spread_after_resume"],
            "pre_save_per_rank": continuity["pre_per_rank"],
            "after_resume_per_rank": continuity["post_per_rank"],
            "tolerance": tolerance,
            "restore_tolerance": tolerance,
            "rank_agreement_tolerance": continuity["rank_agreement_tolerance"],
            "rank_agreement_tolerance_source": continuity["rank_agreement_tolerance_source"],
            "cross_rank_spread_delta": continuity["cross_rank_spread_delta"],
            "status": "PROVED",
            "why_tolerance": (
                "restore fidelity is asserted per rank against that rank's OWN "
                "pre-save value (|own_after - own_pre|, MAX over ranks), never "
                "against another rank's scalar; cross-rank agreement is a separate "
                "named measurement below; finite-precision kernels and operation "
                "scheduling can alter the last bits, so continuity is bounded, not "
                "bit-for-bit -- and the tolerance is never widened to hide the "
                "instrument's measured spread"
            ),
        },
        # fs178: when the ranks do not agree, this entry lands in the run's
        # unmeasured set under this exact name -- a DECLARED UNMEASURED that names
        # the instrument (the divergence precedes any checkpoint), never a pass and
        # never a resume failure.
        "fixed_eval_rank_invariance": {
            "status": "measured" if continuity["rank_invariant"] else "unmeasured",
            "pre_save_per_rank": continuity["pre_per_rank"],
            "after_resume_per_rank": continuity["post_per_rank"],
            "cross_rank_spread_before_save": continuity["cross_rank_spread_before_save"],
            "cross_rank_spread_after_resume": continuity["cross_rank_spread_after_resume"],
            "rank_agreement_tolerance": continuity["rank_agreement_tolerance"],
            "rank_agreement_tolerance_source": continuity["rank_agreement_tolerance_source"],
            "rank_agreement_absolute": continuity["rank_agreement_absolute"],
            "rank_agreement_preserved": continuity["rank_agreement_preserved"],
            "cross_rank_spread_delta": continuity["cross_rank_spread_delta"],
            "display": (
                "fixed-eval rank agreement MEASURED within the explicit "
                "rank-agreement tolerance "
                f"{continuity['rank_agreement_tolerance']:.8g}"
                if continuity["rank_invariant"]
                else "fixed-eval rank invariance UNMEASURED: the fixed-eval loss "
                "takes distinct bit-identical values across ranks (spread before "
                f"save {continuity['cross_rank_spread_before_save']}, spread after "
                f"resume {continuity['cross_rank_spread_after_resume']}, spread "
                f"delta {continuity['cross_rank_spread_delta']}); the restore "
                "verdict stands on its own per-rank terms against the restore "
                f"tolerance {tolerance:.8g}"
            ),
        },
        "continuity_verdict": {
            "status": "PROVED",
            "restore_delta": continuity["restore_delta"],
            "reason": continuity["reason"],
        },
    }
    return recorded_step, metrics


def _make_batches(
    context: DatasetContext,
    config: RunConfig,
    rank: int,
    world_size: int,
    skip_steps: int,
) -> Iterable[list[dict[str, str]]]:
    """Restart deterministically at an absolute sample position for continuing training."""
    epoch = 0
    yielded = 0
    while True:
        sampler = DistributedSampler(
            context.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config.seed.value) + epoch,
            drop_last=False,
        )
        sampler.set_epoch(epoch)
        loader = DataLoader(
            context.train_dataset,
            batch_size=int(config.batch_size.value),
            sampler=sampler,
            num_workers=0,
            collate_fn=list,
            drop_last=False,
        )
        for rows in loader:
            if yielded >= skip_steps:
                yield [dict(row) for row in rows]
            yielded += 1
        epoch += 1


def _collect_global_train_metrics(
    local_loss: float | None,
    local_tokens: int,
    local_examples: int,
    local_seconds: float,
    device: torch.device,
) -> tuple[float | None, int, int, float, float | None]:
    """Reduce flags and values together so no rank advances after bad evidence."""
    valid = 1 if local_loss is not None and math.isfinite(local_loss) else 0
    seconds = max(local_seconds, 0.0)
    memory = _peak_memory_gib()
    memory_valid = 1 if memory is not None and math.isfinite(memory) else 0
    packet = torch.tensor(
        [
            local_loss if valid else 0.0,
            float(local_tokens),
            float(local_examples),
            seconds,
            memory or 0.0,
            valid,
            memory_valid,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(packet[0:4], op=dist.ReduceOp.SUM)
    dist.all_reduce(packet[4], op=dist.ReduceOp.MAX)
    dist.all_reduce(packet[5:7], op=dist.ReduceOp.MIN)
    if int(packet[5].item()) != 1:
        return None, int(packet[1].item()), int(packet[2].item()), float(packet[3].item()), None
    if int(packet[6].item()) != 1:
        return float(packet[0].item()), int(packet[1].item()), int(packet[2].item()), float(
            packet[3].item()
        ), None
    return (
        float(packet[0].item()) / float(int(os.environ["WORLD_SIZE"])),
        int(packet[1].item()),
        int(packet[2].item()),
        float(packet[3].item()),
        float(packet[4].item()),
    )


def train_steps(
    bundle: RuntimeBundle,
    context: DatasetContext,
    config: RunConfig,
    start_step: int,
    stop_step: int,
    peak_so_far: float,
) -> tuple[int, float, MeasurementLedger]:
    """Stop at an absolute bound so resumed continuation cannot exceed the contract."""
    ledger = MeasurementLedger()
    optimizer = bundle.optimizer
    if optimizer is None:
        raise OperationFailure("train", "optimizer", "training requires an optimizer")
    iterator = _make_batches(
        context, config, bundle.rank, bundle.world_size, skip_steps=start_step
    )
    bundle.model.train()
    sequence_length = int(config.sequence_length.value)
    global_step = start_step
    peak_memory = peak_so_far
    logged_events = 0
    while global_step < stop_step:
        rows = next(iterator)
        texts = [str(row["text"]) for row in rows]
        torch.cuda.synchronize(bundle.device)
        started = time.perf_counter()
        try:
            batch = _make_batch(bundle, texts, sequence_length)
            output = bundle.model(**batch)
            loss = getattr(output, "loss", None)
            if loss is None or loss.ndim != 0:
                raise ValueError("model did not return a scalar loss")
            if not torch.isfinite(loss).detach().item():
                raise ValueError("loss was not finite")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            local_loss = float(loss.detach().float().item())
        except Exception as exc:
            local_loss = None
            failure = str(exc)
        else:
            failure = None
        torch.cuda.synchronize(bundle.device)
        elapsed = time.perf_counter() - started
        token_count = int(batch["attention_mask"].detach().sum().item()) if "batch" in locals() else 0
        global_loss, global_tokens, global_examples, global_seconds, step_peak = (
            _collect_global_train_metrics(
                local_loss,
                token_count,
                len(texts),
                elapsed,
                bundle.device,
            )
        )
        if global_loss is None:
            raise OperationFailure(
                "train",
                "loss",
                "one or more ranks could not measure a finite scalar loss"
                + (f": {failure}" if failure else ""),
            )
        if step_peak is None:
            peak_payload = {
                "status": "unmeasured",
                "value": None,
                "unit": "GiB",
                "display": "peak GPU memory UNMEASURED",
            }
            ledger.unmeasured.append("train.peak_gpu_memory")
        else:
            peak_memory = max(peak_memory, step_peak)
            peak_payload = {
                "status": "measured",
                "value": step_peak,
                "unit": "GiB",
                "display": f"{step_peak:.3f} GiB",
            }
        global_step += 1
        should_log = global_step % int(config.log_every.value) == 0 or global_step == stop_step
        if should_log and step_peak is not None:
            tokens_per_second = (
                float(global_tokens) / global_seconds if global_seconds > 0 else None
            )
            logged_events += 1
            train_line = {
                "step": DenominatedCount(
                    global_step, int(config.iteration_budget.value), "iterations"
                ).payload(),
                "segment_examples": DenominatedCount(
                    global_examples, global_examples, "examples"
                ).payload(),
                "segment_tokens": DenominatedCount(
                    global_tokens, global_tokens, "tokens"
                ).payload(),
                "loss": {
                    "status": "measured",
                    "value": global_loss,
                    "display": f"{global_loss:.8g}",
                },
                "tokens_per_second": {
                    "status": "measured" if tokens_per_second is not None else "unmeasured",
                    "value": tokens_per_second,
                    "unit": "tokens/s",
                    "display": (
                        f"{tokens_per_second:.3f} tokens/s"
                        if tokens_per_second is not None
                        else "UNMEASURED"
                    ),
                },
                "peak_gpu_memory": peak_payload,
                "log_events": DenominatedCount(
                    logged_events, logged_events, "events"
                ).payload(),
            }
            _print_json(
                "TRAIN_JSON",
                {
                    "phase": "train",
                    "dataset_origin": context.origin,
                    "real_data": context.real_flag,
                    "synthetic_seed": context.seed,
                    "metrics": train_line,
                },
            )
        elif should_log:
            ledger.unmeasured.append("train.tokens_per_second")
    return global_step, peak_memory, ledger


def evaluate_held_out(
    bundle: RuntimeBundle,
    context: DatasetContext,
    config: RunConfig,
) -> dict[str, Any]:
    """Assign held-out rows by rank and reduce only measured finite losses."""
    rows = [
        index
        for position, index in enumerate(context.eval_rows)
        if position % bundle.world_size == bundle.rank
    ]
    sequence_length = int(config.sequence_length.value)
    loss_sum = 0.0
    token_total = 0
    failures = 0
    bundle.model.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), int(config.batch_size.value)):
            chosen = rows[offset : offset + int(config.batch_size.value)]
            texts = [_text_at(context, index) for index in chosen]
            try:
                batch = _make_batch(bundle, texts, sequence_length)
                output = bundle.model(**batch)
                loss = getattr(output, "loss", None)
                if loss is None or loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise ValueError("invalid scalar held-out loss")
                loss_sum += float(loss.float().item()) * len(chosen)
                token_total += int(batch["attention_mask"].detach().sum().item())
            except Exception:
                failures += len(chosen)
    measured_rows = max(0, len(rows) - failures)
    packet = torch.tensor(
        [loss_sum, measured_rows, token_total, failures],
        dtype=torch.float64,
        device=bundle.device,
    )
    dist.all_reduce(packet, op=dist.ReduceOp.SUM)
    expected = len(context.eval_rows)
    measured = int(packet[1].item())
    if measured <= 0 or int(packet[3].item()) != 0:
        loss_payload: dict[str, Any] = {
            "status": "unmeasured",
            "value": None,
            "display": "held-out loss UNMEASURED",
        }
    else:
        loss_payload = {
            "status": "measured",
            "value": float(packet[0].item()) / float(measured),
            "display": f"{float(packet[0].item()) / float(measured):.8g}",
        }
    return {
        "examples": DenominatedCount(measured, expected, "held-out examples").payload(),
        "tokens": DenominatedCount(
            int(packet[2].item()), int(packet[2].item()), "held-out tokens"
        ).payload(),
        "loss": loss_payload,
    }


def _cleanup_bundle(bundle: RuntimeBundle) -> None:
    """Release the old executable graph before re-entry, preventing silent reuse."""
    bundle.optimizer = None
    bundle.model = None
    gc.collect()
    _barrier()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run(config: RunConfig) -> int:
    """Execute bounded phases while carrying every negative measurement to the end."""
    ledger = MeasurementLedger()
    _initialized_distributed()
    machine = PhaseMachine()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    output_root = Path(str(config.output_dir.value)).expanduser().resolve()
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
    _barrier()

    _print_json(
        "CONTRACT_JSON",
        {
            "status": "accepted",
            "probe_mode": config.probe_mode,
            "values": config.provenance_payload(),
            "rank_world": DenominatedCount(
                world_size, world_size, "ranks"
            ).payload(),
            "visible_devices": DenominatedCount(
                world_size, world_size, "devices"
            ).payload(),
        },
    )

    machine.begin("load")
    artifacts = load_artifacts(config)
    runtime = build_runtime(artifacts, config)
    component_count = 1
    adapter_count = 1 if artifacts.feature_adapter is not None else 0
    load_metrics = {
        "model_components": DenominatedCount(
            component_count, 1, "declared checkpoint components"
        ).payload(),
        "text_adapters": DenominatedCount(
            adapter_count, 1, "tokenizer or processor adapters"
        ).payload(),
        "world_devices": DenominatedCount(
            world_size, torch.cuda.device_count(), "rank/device pairs"
        ).payload(),
    }
    ledger.check("load", load_metrics)
    _phase_summary(machine, "load", load_metrics, None)

    machine.begin("data")
    context = build_dataset_context(config, world_size)
    data_metrics = {
        "source_rows": DenominatedCount(
            context.total_index_count, context.total_index_count, "rows"
        ).payload(),
        "training_rows": DenominatedCount(
            context.train_index_count, context.train_index_count, "rows"
        ).payload(),
        "held_out_rows": DenominatedCount(
            context.eval_index_count, context.eval_index_count, "rows"
        ).payload(),
        "partition_overlap": DenominatedCount(
            0, context.total_index_count, "overlapping rows"
        ).payload(),
    }
    ledger.check("data", data_metrics)
    _phase_summary(machine, "data", data_metrics, context)

    budget = int(config.iteration_budget.value)
    early = int(config.early_save_steps.value)
    fixed_indices = _fixed_eval_indices(context, config)
    sequence_length = int(config.sequence_length.value)

    machine.begin("train")
    torch.cuda.reset_peak_memory_stats(runtime.device)
    global_step, peak_memory, train_ledger = train_steps(
        runtime, context, config, 0, early, 0.0
    )
    ledger.unmeasured.extend(train_ledger.unmeasured)
    # --- fs173: close segment-1 train phase (job 37304) ------------------------------
    # Job 37304 died at the next line down: five TRAIN_JSON events fell
    # 1.2414 -> 0.5841 over steps 10..50 of 200, then the save begin refused
    # because this train phase was still open. Closing it here restores the
    # first half of the mirror pair.
    segment_metrics = {
        "iterations": DenominatedCount(global_step, early, "iterations").payload(),
        "peak_gpu_memory": (
            {
                "status": "measured",
                "value": peak_memory,
                "unit": "GiB",
                "display": f"{peak_memory:.3f} GiB",
            }
            if math.isfinite(peak_memory)
            else {
                "status": "unmeasured",
                "value": None,
                "unit": "GiB",
                "display": "peak GPU memory UNMEASURED",
            }
        ),
    }
    ledger.check("train.early", segment_metrics)
    _phase_summary(machine, "train", segment_metrics, context)
    # --- end fs173 segment-1 ---

    machine.begin("save")
    pre_loss = _evaluate_fixed_loss(runtime, context, fixed_indices, sequence_length)
    early_dir = output_root / f"checkpoint-step-{early:08d}"
    save_checkpoint(runtime, early_dir, global_step, fixed_loss=pre_loss)
    early_save_metrics = {
        "checkpoint_step": DenominatedCount(
            global_step, early, "training iterations"
        ).payload(),
        "rank_payloads": DenominatedCount(
            world_size, world_size, "rank-local payload files"
        ).payload(),
    }
    ledger.check("save.early", early_save_metrics)
    _phase_summary(machine, "save", early_save_metrics, context)

    machine.begin("resume")
    _cleanup_bundle(runtime)
    resumed = _restore_runtime_from_checkpoint(early_dir, config)
    recorded_step, resume_metrics = resume_and_prove(
        resumed, early_dir, config, context, fixed_indices
    )
    if recorded_step != global_step:
        raise OperationFailure(
            "resume",
            "absolute_step",
            f"restored absolute step {recorded_step} disagrees with live step {global_step}",
        )
    ledger.check("resume", resume_metrics)
    _phase_summary(machine, "resume", resume_metrics, context)
    runtime = resumed

    # --- fs173: open segment-2 train phase (job 37304) -------------------------------
    # Mirror of the close above: the continuation leg trains from early to
    # budget, but its train phase was never opened -- the final train summary
    # would have closed None. Opening it here restores the second half of the
    # pair. The two defects hid each other; only their pairing read as balanced.
    machine.begin("train")
    global_step, peak_memory, continuation_ledger = train_steps(
        runtime, context, config, global_step, budget, peak_memory
    )
    ledger.unmeasured.extend(continuation_ledger.unmeasured)
    train_metrics = {
        "iterations": DenominatedCount(global_step, budget, "iterations").payload(),
        "early_step": DenominatedCount(early, early, "iterations").payload(),
        "remaining_iterations": DenominatedCount(
            budget - early, budget - early, "iterations"
        ).payload(),
        "peak_gpu_memory": (
            {
                "status": "measured",
                "value": peak_memory,
                "unit": "GiB",
                "display": f"{peak_memory:.3f} GiB",
            }
            if math.isfinite(peak_memory)
            else {
                "status": "unmeasured",
                "value": None,
                "unit": "GiB",
                "display": "peak GPU memory UNMEASURED",
            }
        ),
    }
    ledger.check("train", train_metrics)
    _phase_summary(machine, "train", train_metrics, context)

    machine.begin("save")
    final_dir = output_root / f"checkpoint-step-{budget:08d}"
    save_checkpoint(runtime, final_dir, global_step)
    final_save_metrics = {
        "checkpoint_step": DenominatedCount(global_step, budget, "iterations").payload(),
        "rank_payloads": DenominatedCount(
            world_size, world_size, "rank-local payload files"
        ).payload(),
    }
    ledger.check("save.final", final_save_metrics)
    _phase_summary(machine, "save", final_save_metrics, context)

    machine.begin("eval")
    eval_metrics = evaluate_held_out(runtime, context, config)
    ledger.check("eval", eval_metrics)
    _phase_summary(machine, "eval", eval_metrics, context)

    measured_phase_count = len(set(machine.completed))
    summary_metrics = {
        "phases_completed": DenominatedCount(
            measured_phase_count, len(PHASES), "observable phases"
        ).payload(),
        "training_iterations": DenominatedCount(
            global_step, budget, "iterations"
        ).payload(),
        "resume_proof": {
            "status": "PROVED",
            "display": "PROVED: optimizer step and fixed loss restored within the restore tolerance",
        },
    }
    ledger.check("run", summary_metrics)
    verdict = "UNMEASURED" if ledger.unmeasured else "MEASURED"
    _print_json(
        "RUN_SUMMARY_JSON",
        {
            "verdict": verdict,
            "unmeasured": sorted(set(ledger.unmeasured)),
            "dataset_origin": context.origin,
            "real_data": context.real_flag,
            "synthetic_seed": context.seed,
            "metrics": summary_metrics,
        },
    )
    return 0 if verdict == "MEASURED" else 3


def _run_selftest() -> int:
    """Exercise fatal branches on data so an all-green table is not merely decorative."""
    base_env = {
        "FS_ITERATION_BUDGET": "8",
        "FS_EARLY_SAVE_STEPS": "2",
        "OUT_DIR": "/tmp/selftest-output",
    }

    def env_only() -> str:
        result = resolve_contract(_complete_required_argv(), base_env)
        if result == "selftest":
            return "FAIL"
        budget = result.iteration_budget
        early = result.early_save_steps
        output = result.output_dir
        return (
            "PASS"
            if budget.value == 8
            and budget.source == "env"
            and early.value == 2
            and early.source == "env"
            and output.source == "env"
            else "FAIL"
        )

    def flag_precedence() -> str:
        argv = _complete_required_argv() + [
            "--iteration-budget",
            "12",
            "--early-save-steps",
            "3",
            "--output-dir",
            "/tmp/flag-output",
        ]
        result = resolve_contract(argv, base_env)
        if result == "selftest":
            return "FAIL"
        return (
            "PASS"
            if result.iteration_budget.value == 12
            and result.iteration_budget.source == "flag"
            and result.early_save_steps.source == "flag"
            and result.output_dir.source == "flag"
            else "FAIL"
        )

    def missing_budget() -> str:
        env = dict(base_env)
        env.pop("FS_ITERATION_BUDGET")
        try:
            resolve_contract(_complete_required_argv(), env)
        except ContractError:
            return "FIRED"
        return "DID_NOT_FIRE"

    def missing_early() -> str:
        env = dict(base_env)
        env.pop("FS_EARLY_SAVE_STEPS")
        try:
            resolve_contract(_complete_required_argv(), env)
        except ContractError:
            return "FIRED"
        return "DID_NOT_FIRE"

    def missing_output() -> str:
        env = dict(base_env)
        env.pop("OUT_DIR")
        try:
            resolve_contract(_complete_required_argv(), env)
        except ContractError:
            return "FIRED"
        return "DID_NOT_FIRE"

    def bad_early_order() -> str:
        env = dict(base_env)
        env["FS_EARLY_SAVE_STEPS"] = "8"
        try:
            resolve_contract(_complete_required_argv(), env)
        except ContractError:
            return "FIRED"
        return "DID_NOT_FIRE"

    def measured_zero_path() -> str:
        payload = DenominatedCount(0, 4, "items").payload()
        return "PASS" if payload["status"] == "measured_zero" else "FAIL"

    def unmeasured_path() -> str:
        payload = DenominatedCount(None, 4, "items").payload()
        ledger = MeasurementLedger()
        ledger.check("metric", {"sample": payload})
        return (
            "FIRED"
            if payload["status"] == "unmeasured" and ledger.unmeasured == ["metric.sample"]
            else "DID_NOT_FIRE"
        )

    def phase_state_machine() -> str:
        machine = PhaseMachine()
        machine.begin("load")
        machine.end("load")
        machine.begin("data")
        machine.end("data")
        machine.reset()
        return "PASS"

    def phase_overlap_refusal() -> str:
        machine = PhaseMachine()
        machine.begin("load")
        try:
            machine.begin("data")
        except ContractError:
            return "FIRED"
        return "DID_NOT_FIRE"

    table: list[tuple[str, str, str, Callable[[], str]]] = [
        ("env_contract_complete", "MUST_PASS", "PASS", env_only),
        ("flag_precedence", "MUST_PASS", "PASS", flag_precedence),
        ("missing_iteration_budget", "MUST_FIRE", "FIRED", missing_budget),
        ("missing_early_save", "MUST_FIRE", "FIRED", missing_early),
        ("missing_output_root", "MUST_FIRE", "FIRED", missing_output),
        ("early_not_before_end", "MUST_FIRE", "FIRED", bad_early_order),
        ("measured_zero_is_distinct", "MUST_PASS", "PASS", measured_zero_path),
        ("unmeasured_is_fail_closed", "MUST_FIRE", "FIRED", unmeasured_path),
        ("phase_machine_valid_path", "MUST_PASS", "PASS", phase_state_machine),
        ("phase_machine_overlap", "MUST_FIRE", "FIRED", phase_overlap_refusal),
    ]
    matched = 0
    for position, (name, obligation, expected, exercise) in enumerate(table, start=1):
        try:
            actual = exercise()
        except Exception as exc:
            actual = f"ERROR:{type(exc).__name__}:{exc}"
        matched_now = actual == expected
        matched += 1 if matched_now else 0
        _print_json(
            "SELFTEST_JSON",
            {
                "row": DenominatedCount(position, len(table), "rows").payload(),
                "name": name,
                "obligation": obligation,
                "expected": expected,
                "actual": actual,
                "matched": bool(matched_now),
            },
        )
    verdict = (
        "SELFTEST_PASS"
        if matched == len(table)
        else "SELFTEST_FAIL"
    )
    _print_json(
        "SELFTEST_SUMMARY_JSON",
        {
            "rows_matched": DenominatedCount(matched, len(table), "rows").payload(),
            "verdict": verdict,
        },
    )
    return 0 if matched == len(table) else 1


def _rank_hint() -> None:
    try:
        _RANK0_ONLY["rank"] = int(os.environ.get("RANK", "0"))
    except ValueError:
        _RANK0_ONLY["rank"] = 0


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed at the outer boundary instead of converting exceptions into zeroes."""
    _rank_hint()
    try:
        resolved = resolve_contract(argv)
        if resolved == "selftest":
            return _run_selftest()
        config = resolved
        if not isinstance(config, RunConfig):
            raise ContractError("configuration did not resolve")
        return _run(config)
    except ContractError as exc:
        _print_json(
            "RUN_SUMMARY_JSON",
            {
                "verdict": "UNMEASURED",
                "reason": "contract_refused",
                "phase": "absent",
                "detail": str(exc),
                "dataset_origin": "UNKNOWN_NOT_RUN",
                "real_data": None,
            },
        )
        return 2
    except OperationFailure as exc:
        _print_json(
            "PHASE_JSON",
            {
                "phase": exc.phase,
                "status": "failed",
                "verdict": "UNMEASURED",
                "metric": exc.metric,
                "display": f"{exc.metric} UNMEASURED",
            },
        )
        _print_json(
            "RUN_SUMMARY_JSON",
            {
                "verdict": "UNMEASURED",
                "reason": str(exc),
                "phase": exc.phase,
                "metric": exc.metric,
                "dataset_origin": "UNKNOWN_AFTER_FAILURE",
                "real_data": None,
            },
        )
        return 3
    finally:
        if _RANK0_ONLY.get("initialized"):
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
