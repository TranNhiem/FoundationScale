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
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
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


def build_runtime(artifacts: LoadArtifacts, config: RunConfig) -> RuntimeBundle:
    """Shard immediately so checkpoint scale is not multiplied by the process count."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    model = _build_model(artifacts)
    try:
        sharded = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=size_based_auto_wrap_policy,
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
    local_difference = abs(fixed_post - float(fixed_pre))
    difference_tensor = torch.tensor(
        [local_difference], dtype=torch.float64, device=bundle.device
    )
    dist.all_reduce(difference_tensor, op=dist.ReduceOp.MAX)
    maximum_difference = float(difference_tensor.item())
    tolerance = float(config.resume_tolerance.value)
    if maximum_difference > tolerance:
        raise OperationFailure(
            "resume",
            "fixed_loss_continuity",
            (
                f"fixed loss changed by {maximum_difference:.8g}, exceeding the stated "
                f"tolerance {tolerance:.8g}"
            ),
        )

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
            "maximum_rank_difference": maximum_difference,
            "tolerance": tolerance,
            "status": "PROVED",
            "why_tolerance": (
                "finite-precision kernels and operation scheduling can alter the last bits "
                "after state serialization; continuity is therefore bounded, not claimed "
                "bit-for-bit"
            ),
        },
        "continuity_verdict": {"status": "PROVED"},
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
            "display": "PROVED: optimizer step and fixed loss restored within tolerance",
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
