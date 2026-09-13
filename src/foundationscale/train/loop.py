"""The thinnest REAL training path through FoundationScale.

The trainer is ``transformers.Trainer`` -- deliberately. FoundationScale's
contribution is the verification plane AROUND the mature loop, not a new
training loop:

    Topology -> ClusterProfile -> consistency findings -> BLOCK before a
    single GPU is touched -> model/data/Trainer -> save-gate callback ->
    train -> final save -> adjudicate -> exit code.

Exit-code contract (house doctrine): 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE.
``import foundationscale.train.loop`` is torch-free; torch/transformers/
datasets are imported INSIDE :func:`train`, and their absence is a REFUSE
(96) naming the extra -- never a bare ImportError traceback.
"""

from __future__ import annotations

import errno
import inspect
import json
import os
import struct
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foundationscale.gates.core import (
    REGISTRY,
    GateRegistry,
    GateReport,
    Lifecycle,
    run_event,
)
from foundationscale.topology import (
    ClusterProfile,
    Finding,
    Severity,
    Topology,
    blocking,
    declared_vs_effective,
    partition_consistency,
    profile_by_name,
    render_findings,
)

# transformers is an optional extra; absent in CI and on login nodes.
#
# The type-checking branch is UNCONDITIONAL on purpose. Guarding this with a
# bare `# type: ignore` made the typecheck result depend on whether the extra
# happened to be installed -- clean without transformers, two errors with it,
# on identical source. That is the same class as the unpinned-formatter defect
# (#111): a gate whose verdict moves without a code change. Pinning the checker
# to the `object` base makes `mypy src` deterministic in both environments,
# while runtime still binds the real TrainerCallback whenever it is importable.
if TYPE_CHECKING:
    _CallbackBase = object
else:
    try:
        from transformers import TrainerCallback as _CallbackBase
    except Exception:  # noqa: BLE001 -- ANY failure degrades to a plain base class
        _CallbackBase = object

EXIT_PASS = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

# --- Environment failures are not this plane's claim to answer (#409) -------
#
# A construction failure has two populations and they carry OPPOSITE verdicts.
# A bad model id, an incompatible architecture or a missing weight file is a
# real RED: the operator asked for something this plane cannot build, and the
# claim is false. An ENOLCK from a home directory that serves no file locking,
# a full disk, a read-only mount or an unreachable hub is not a statement about
# the plane at all -- construction never got far enough to measure anything, so
# the honest verdict is 96 CANNOT-MEASURE. This is the rule the kernel-selection
# arm further down already states for itself; it was never applied to the outer
# boundary, which is where every other construction failure lands.
#
# Collapsing the two is how a framework comes to accuse itself. On an NFS home
# -- the default on most HPC estates -- the HuggingFace filelock raises
# OSError(37, "No locks available") before a single parameter is read, and a
# bare `except Exception -> RED` reports that as a broken training plane.
# MEASURED on a GB200 estate: four telemetry tests failed their own
# `rc in (EXIT_PASS, EXIT_UNMEASURED)` precondition on rc=5 for exactly this
# reason, and the identical suite passed once HF_HOME pointed at node-local
# scratch. Nothing about the framework differed between those two runs.
#
# The table is keyed on ERRNO, not on message text, because an errno is a stable
# kernel-level fact while the message is libc- and locale-dependent -- keying on
# text would make the verdict vary by machine, which is the axis #83 refuses.
_ENVIRONMENT_ERRNOS: dict[int, str] = {
    errno.ENOLCK: "the filesystem serves no file locking (typical of an NFS home)",
    errno.ENOSPC: "the filesystem is full",
    errno.EDQUOT: "the disk quota is exhausted",
    errno.EROFS: "the filesystem is read-only",
    errno.EACCES: "the path is not readable by this user",
    errno.EPERM: "the operation is not permitted for this user",
    errno.ECONNREFUSED: "the remote host refused the connection",
    errno.ENETUNREACH: "the network is unreachable",
    errno.ETIMEDOUT: "the connection timed out",
    errno.EMFILE: "the per-process open-file limit is exhausted",
    errno.ENFILE: "the system-wide open-file limit is exhausted",
}


def _environment_failure_reason(exc: BaseException) -> str | None:
    """Name the environment fault behind `exc`, or None if it is a genuine RED.

    Walks the ``__cause__``/``__context__`` chain rather than judging only the
    outermost exception, because the libraries this plane calls wrap an OSError
    in their own exception type far more often than they let it through: a check
    on the top frame alone would classify every wrapped ENOLCK as RED and leave
    the split doing nothing at all. The walk is cycle-guarded because an
    exception raised while handling itself can close the chain into a loop.
    """
    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        if isinstance(cursor, OSError) and cursor.errno in _ENVIRONMENT_ERRNOS:
            detail = _ENVIRONMENT_ERRNOS[cursor.errno]
            where = f" at {cursor.filename}" if cursor.filename else ""
            return f"{detail}{where} [errno {cursor.errno}]"
        cursor = cursor.__cause__ or cursor.__context__
    return None


EXTRA = "foundationscale[train]"
EXTRA_HINT = f"pip install '{EXTRA}'"

TOKENIZE_MAX_LENGTH = 128
# NOT a name of our choosing. checkpoint.dcp_meta.load_manifest -- the reader
# every checkpoint gate goes through -- searches a fixed tuple of basenames, and
# this must be one of them or the gates see no manifest at all. It was
# "foundationscale.run.json" for one release: a name with the project in it,
# which reads deliberate and shares not one character with anything the reader
# looks for, so every save-gate verdict came back VACUOUS on every real run
# (#225). Same class as #150 -- a producer and a consumer each internally
# consistent and never introduced.
#
# Of the three accepted names this is the RESERVED one, chosen for its strictness:
# a file here that is malformed RAISES CheckpointFormatError instead of being
# skipped as absent. Under either of the other two, corruption is indistinguishable
# from a run that wrote nothing. That trade is only safe because what we write is a
# validated RunManifest -- see _build_run_manifest.
MANIFEST_NAME = "run_manifest.json"

# The precision names a run may DECLARE. Declarable is not executable: nvfp4
# is in the accepted set so a manifest can say "nvfp4 was asked for", and
# train() then refuses it -- see the Step.START refusal -- rather than mapping
# it onto bf16, which is the silent-fallback defect of finding #342. Sorted,
# because refusal messages interpolate it verbatim and a sorted set is the
# house contract for naming accepted values.
PRECISIONS: tuple[str, ...] = ("bf16", "fp16", "fp32", "nvfp4")
# The adapter modes the package plane wires itself. "lora" is the only one;
# TrainConfig.adapter=None means full fine-tune, and it means it explicitly.
ADAPTERS: tuple[str, ...] = ("lora",)
# The sharding declarations the execution plane can actually HONOUR. "ddp" is
# the only one: transformers.Trainer as wired here provides data parallelism
# only, and every sharded alternative (FSDP/ZeRO/DeepSpeed) is adjudicated and
# recorded in gates/ and provenance/ -- measured, named, and never built. The
# tuple exists so the refusal message can name the accepted set, the same
# reason PRECISIONS is a tuple; sorted for the verbatim-interpolation contract.
SHARDING_STRATEGIES: tuple[str, ...] = ("ddp",)


class Step:
    """Declared launch markers. Every line the entry emits starts with one."""

    START = "fs:train:start"
    TOPOLOGY = "fs:train:topology"
    PROFILE = "fs:train:profile"
    CONSISTENCY = "fs:train:consistency"
    PARTITION = "fs:train:partition"
    VALIDATED = "fs:train:validated"
    BLOCKED = "fs:train:blocked"
    REFUSE = "fs:train:refuse"
    DEPS = "fs:train:deps"
    DATA = "fs:train:data"
    TRAINER = "fs:train:trainer"
    # Marker, not absence (same doctrine as UNMEASURED below): adapter wiring
    # and the vacuous-attach refusal both report here, so a log shows the step
    # ran rather than implying an adapter run is indistinguishable from
    # full fine-tune. MARKERS is derived from this class, so adding the member
    # is also adding it to the denominator.
    ADAPTER = "fs:train:adapter"
    RUN = "fs:train:run"
    SAVED = "fs:train:saved"
    SAVE_GATE = "fs:train:save_gate"
    OBJECTIVE_GATE = "fs:train:objective_gate"
    MANIFEST = "fs:train:manifest"
    ADJUDICATE = "fs:train:adjudicate"
    DONE = "fs:train:done"
    RED = "fs:train:red"
    # UNMEASURED is a marker, not the absence of one (doctrine 5). Before this
    # existed, the two UNMEASURED returns in train() borrowed ADJUDICATE, so a
    # log reader could see the exit code 95 but not which step abstained --
    # and a step that abstains silently is indistinguishable from one that ran.
    UNMEASURED = "fs:train:unmeasured"


# MARKERS is the marker denominator: every consumer that asks "which steps exist"
# reads this. It used to be a hand-written tuple listing Step's members, and it
# drifted -- Step.PARTITION was declared at :98 and emitted twice from _partition
# scanning, yet named nowhere in the tuple, so 19 of 20 markers were in the
# denominator and the partition step was invisible to anything counting steps
# (#312). Derive it from Step instead, so a member cannot be added without
# entering the denominator. Class bodies preserve declaration order, so the
# derived order is the source order; dunders are excluded by the underscore test,
# which is also what keeps the class docstring out.
MARKERS: tuple[str, ...] = tuple(
    value
    for name, value in vars(Step).items()
    if not name.startswith("_") and isinstance(value, str)
)


def _mark(step: str, msg: str = "") -> None:
    print(f"[{step}]".ljust(24) + f" {msg}".rstrip(), flush=True)


def fs_version() -> str:
    """The installed package version, or ``0+unknown`` when not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("foundationscale")
        except PackageNotFoundError:
            return "0+unknown"
    except Exception:  # noqa: BLE001
        return "0+unknown"


def _tf_version() -> str:
    """The installed transformers version, for refusal messages only.

    Named in a refusal, a version is the first thing an operator checks, so it
    is read from the module actually imported rather than from the pin that was
    requested. Falls back to a stated unknown -- never to silence.
    """
    try:
        import transformers

        return str(getattr(transformers, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """Everything the thin path needs, as data.

    Fail closed (doctrine 4): fields whose value is one machine's fact
    -- ``nodes``, ``gpus_per_node``, the cluster profile -- have NO default.
    Harmless knobs (seed, lr, batch size) carry defaults.
    """

    model: str  # HF model id or local path. Model-agnostic: this is data, not code.
    dataset: str  # HF dataset id, or a .json/.jsonl file, or a dir of them.
    output_dir: Path
    # Machine facts -- no defaults, on purpose.
    nodes: int
    gpus_per_node: int
    # Cluster profile: exactly one of the three must be provided.
    profile: ClusterProfile | None = None
    profile_name: str | None = None
    profile_path: Path | None = None
    # What the run set out to optimise (e.g. "sft"). Default None means the run
    # declared NOTHING, which is not a neutral state: the objective gate refuses
    # an undeclared run at its first observed step. That refusal is the intended
    # fail-closed behaviour -- a run whose objective is unstated cannot have its
    # loss components, reward scale or hyperparameter drift checked against
    # anything -- so the value is left absent here rather than defaulted to the
    # plausible "sft", which would make the gate compare the loop against itself.
    objective: str | None = None
    # Declared training precision. None means NOT DECLARED and is never coerced:
    # a run that says nothing trains at whatever dtype the model loader picks,
    # the manifest records None, and the observed-vs-declared check at the
    # first checkpoint abstains rather than passing. Coercing None to "bf16"
    # would launder an absent statement into a measured claim (#342). "nvfp4"
    # is declarable here but refused by train() -- declarable is not executable.
    precision: str | None = None
    # Declared adapter mode. None means FULL FINE-TUNE, stated as data rather
    # than implied by the absence of peft wiring. Every adapter_* knob is None
    # by default, and a partial specification is refused in __post_init__ --
    # adapter_rank set with adapter unset is a config the operator did not mean.
    adapter: str | None = None
    adapter_rank: int | None = None
    adapter_alpha: float | None = None
    adapter_targets: tuple[str, ...] | None = None
    adapter_dropout: float | None = None
    # --- The nine declaration axes -------------------------------------------
    # Every one of these defaults to None, and None means exactly what
    # precision's None means: not declared -- claim nothing, apply nothing,
    # record the absence. Each axis has a live HF Trainer default behind it
    # (AdamW, accumulation 1, max_grad_norm 1.0, no recompute, linear LR with
    # zero warmup, model-config attention), and coercing None to that default
    # here would launder an absent statement into a measured claim (#342) while
    # making a manifest-less runtime default look like a declaration. The
    # defaults therefore stay transformers' problem, unrecorded and unclaimed;
    # a declared value is wired in train() and resolved through the manifest's
    # ConfigResolver path like everything else.
    optimizer: str | None = None
    gradient_accumulation_steps: int | None = None
    max_grad_norm: float | None = None
    gradient_checkpointing: bool | None = None
    # attn_implementation is NOT a TrainingArguments knob: it binds at MODEL
    # construction (AutoModelForCausalLM.from_pretrained), and train() refuses
    # (96) when this transformers release does not name the parameter, because
    # older releases route it through opaque **kwargs where a misspelt or
    # unsupported name is silently ignored. Never passed when None.
    attn_implementation: str | None = None
    lr_scheduler_type: str | None = None
    warmup_steps: int | None = None
    # Declarable only -- there is NO wiring behind these two. sharding_strategy
    # of None or "ddp" proceeds ("ddp" is what the backend actually executes);
    # anything else is REFUSED (96) at START rather than run as plain DDP under
    # a declaration that says otherwise (#375's defect class). True
    # cpu_optimizer_offload is REFUSED (96) for the same reason: no offload
    # backend exists in this plane.
    sharding_strategy: str | None = None
    cpu_optimizer_offload: bool | None = None
    # logging_steps sits OUTSIDE the nine axes above on purpose: it is the one
    # knob whose absence still binds. None means not declared, and the wiring
    # site in _train then falls back to max(1, min(10, max_steps)) -- the
    # historical unconditional binding -- so an undeclared run behaves exactly
    # as it did before the knob existed. The nine axes' None applies NOTHING;
    # this one's None applies the fallback, and the manifest records both the
    # declaration (config section) and the value actually bound (telemetry
    # section's logging_steps_effective).
    logging_steps: int | None = None
    # Harmless knobs.
    max_steps: int = 20
    per_device_batch_size: int = 1
    learning_rate: float = 5e-5
    save_interval: int = 50
    seed: int = 42
    dp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    dry_run: bool = False
    # A directory of LAUNCHER scripts (sbatch/srun/shell) for the partition
    # spelling scan. Optional and default-None because the driver legitimately
    # has none: `train()` is invoked inside an allocation, not as the thing
    # that requests one. When it is None the scan is reported UNMEASURED, never
    # silently skipped and never faked clean.
    launch_corpus: Path | None = None
    # NOT a declaration axis -- provenance ABOUT the other fields. It carries
    # the field names the command line actually supplied, so the manifest can
    # say `source="cli"` of a value the operator typed and `source="default"`
    # of one nobody did. Without it every key was stamped "cli", which is a
    # claim about where a value came from that the loop had not measured, and
    # it was false for every omitted axis.
    #
    # None is a third state and it is load-bearing: it means this config was
    # not built by the parser at all (a test, an embedding caller, a future
    # config-file path), and the manifest then records "config" rather than
    # inventing a command line that never ran. An empty tuple means the parser
    # DID run and supplied nothing -- observably different, and only the
    # sentinel-parse in cli.py can tell them apart.
    cli_declared: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.profile_path is not None:
            object.__setattr__(self, "profile_path", Path(self.profile_path))
        provided = sum(
            1 for v in (self.profile, self.profile_path, self.profile_name) if v is not None
        )
        if provided != 1:
            raise ValueError(
                "exactly one of profile / profile_path / profile_name is "
                "required (fail closed: a cluster profile is a machine fact "
                "and has no default)"
            )
        if not self.model or not self.dataset:
            raise ValueError("model and dataset must be non-empty")
        for field_name in (
            "nodes",
            "gpus_per_node",
            "max_steps",
            "per_device_batch_size",
            "save_interval",
            "dp",
            "tp",
            "pp",
            "ep",
            "cp",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be >= 1")
        if self.precision is not None and self.precision not in PRECISIONS:
            raise ValueError(f"precision={self.precision!r} is not one of {PRECISIONS}")
        # Range checks on the declared optional axes mirror the ones
        # TrainingArguments performs -- but performed HERE, at statement time,
        # so an out-of-range declaration is a config error with a named field
        # rather than a rejection buried inside a constructor three steps later.
        # Values these checks cannot see (unknown optimizer names, unschedulable
        # schedulers) are refused at Trainer-construction time instead, where
        # transformers is the authority on its own accepted vocabulary.
        accum = self.gradient_accumulation_steps
        if accum is not None and int(accum) < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.warmup_steps is not None and int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.logging_steps is not None and int(self.logging_steps) < 1:
            raise ValueError(
                "logging_steps must be >= 1; 0 would emit no training log at "
                "all, and a run that logs nothing is UNMEASURED by construction"
            )
        if self.max_grad_norm is not None and float(self.max_grad_norm) <= 0:
            raise ValueError("max_grad_norm must be > 0; 0 would zero every step")
        if self.adapter is not None and self.adapter not in ADAPTERS:
            raise ValueError(f"adapter={self.adapter!r} is not one of {ADAPTERS}")
        if self.adapter is None:
            # A partial specification is a refusal, not a hint: every adapter_*
            # field with adapter unset is a statement about nothing.
            for field_name in (
                "adapter_rank",
                "adapter_alpha",
                "adapter_targets",
                "adapter_dropout",
            ):
                value = getattr(self, field_name)
                if value is not None:
                    raise ValueError(
                        f"{field_name}={value!r} is set while adapter is None: "
                        f"a partial adapter specification is refused. Set "
                        f"adapter to one of {ADAPTERS}, or clear {field_name}"
                    )
        elif self.adapter_rank is None or int(self.adapter_rank) < 1:
            raise ValueError(
                f"adapter={self.adapter!r} requires adapter_rank to be a "
                f"positive int; got adapter_rank={self.adapter_rank!r}. A "
                "missing or non-positive rank silently defines the adapter's "
                "capacity, which is exactly the unrecorded-config failure"
            )
        if self.adapter_targets is not None:
            object.__setattr__(self, "adapter_targets", tuple(self.adapter_targets))
            if not self.adapter_targets:
                # all([]) is True and that is the founding defect of this
                # codebase: an adapter that targets nothing trains nothing
                # while looking like it trained.
                raise ValueError(
                    "adapter_targets=() is refused as vacuous: an adapter that "
                    "targets no module trains nothing while looking like it "
                    "trained. Name at least one target, or pass None to use "
                    "peft's per-model defaults"
                )


def _loudest() -> Severity:
    """The most blocking severity the enum defines, without guessing its name."""
    return list(Severity)[-1]


def _resolve_profile(cfg: TrainConfig) -> ClusterProfile:
    if cfg.profile is not None:
        return cfg.profile
    if cfg.profile_path is not None:
        return ClusterProfile.from_json(cfg.profile_path)
    assert cfg.profile_name is not None  # guaranteed by __post_init__
    return profile_by_name(cfg.profile_name)


def _effective_topology(cfg: TrainConfig) -> Topology | Finding | None:  # noqa: ARG001
    """The topology the runtime actually built, derived from torchrun env.

    Every field comes from runtime evidence, never from ``cfg`` -- the
    previous version sourced tp/pp/ep/cp AND gpus_per_node from ``cfg``, so
    5 of the 7 fields declared_vs_effective compares were equal by
    construction and could never differ:

    * ``dp`` is ``WORLD_SIZE``. Measured, not assumed: the
      transformers.Trainer kwargs dict built in :func:`train` carries no
      tensor/pipeline/expert/context-parallel key of any kind, so every rank
      the launcher started is a data-parallel replica.
    * ``tp``/``pp``/``ep``/``cp`` are pinned to 1 for the same reason: no
      model-parallel degree is wired anywhere in this plane, so 1 is the
      degree the runtime executes. :func:`train` refuses any declared degree
      > 1 before a model is loaded, which is what makes the pin honest
      rather than a second silent default.
    * ``gpus_per_node`` is ``LOCAL_WORLD_SIZE``, the launcher's own
      statement of how many ranks landed on each node.
    * ``nodes`` is ``WORLD_SIZE // LOCAL_WORLD_SIZE``.

    ``cfg`` is retained in the signature so a test can hand this function a
    config whose axes DISAGREE with the runtime and prove the return value
    does not echo it -- a control that dropping the parameter would delete.
    It is deliberately unread. Returns ``None`` on the driver process (no
    WORLD_SIZE), a :class:`Finding` when the runtime evidence cannot form a
    topology, else the effective one.
    """
    raw = os.environ.get("WORLD_SIZE")
    if raw is None:
        return None
    try:
        world = int(raw)
    except ValueError:
        return Finding(
            code="train.world_size",
            severity=_loudest(),
            message=f"WORLD_SIZE={raw!r} is not an integer",
        )
    local_raw = os.environ.get("LOCAL_WORLD_SIZE")
    if local_raw is None:
        # An axis with no runtime evidence is UNMEASURED, not a default:
        # reading gpus_per_node from cfg here was the equal-by-construction
        # defect, and it recorded a 4-rank run as an 8-GPU topology.
        return Finding(
            code="train.local_world_size_unset",
            severity=_loudest(),
            message=(
                f"WORLD_SIZE={world} but LOCAL_WORLD_SIZE is unset: "
                "gpus_per_node has no runtime evidence, and an axis with no "
                "runtime evidence is UNMEASURED -- it is never defaulted "
                "from cfg"
            ),
        )
    try:
        gpn = int(local_raw)
    except ValueError:
        return Finding(
            code="train.local_world_size",
            severity=_loudest(),
            message=f"LOCAL_WORLD_SIZE={local_raw!r} is not an integer",
        )
    if gpn < 1 or gpn > world:
        return Finding(
            code="train.local_world_size",
            severity=_loudest(),
            message=(
                f"LOCAL_WORLD_SIZE={gpn} lies outside [1, WORLD_SIZE={world}]: "
                "the launcher's own ranks-per-node statement is impossible, "
                "so no topology can be derived from it"
            ),
        )
    if world % gpn != 0:
        return Finding(
            code="train.ragged_world",
            severity=_loudest(),
            message=(
                f"WORLD_SIZE={world} is not divisible by "
                f"LOCAL_WORLD_SIZE={gpn}: the last node carries fewer ranks. "
                "An uneven last node is real runtime evidence, and it is "
                "not a Topology"
            ),
        )
    try:
        # tp=pp=ep=cp=1 is measured, not assumed: the transformers.Trainer
        # kwargs dict in train() carries no tensor/pipeline/expert/context-
        # parallel key of any kind, so the plane executes pure data
        # parallelism and dp is the whole world. There is no max(..., 1)
        # clamp: a degree that cannot be derived is a Finding above, never
        # a fabricated 1.
        return Topology(
            dp=world,
            tp=1,
            pp=1,
            ep=1,
            cp=1,
            nodes=world // gpn,
            gpus_per_node=gpn,
        )
    except Exception as exc:  # noqa: BLE001 -- malformed runtime evidence is a finding
        return Finding(
            code="train.effective_topology",
            severity=_loudest(),
            message=f"WORLD_SIZE={world} cannot form a valid topology: {exc}",
        )


def _default_context_builder(ckpt_dir: Path | str) -> Any:
    """Torch-free by contract: checkpoint_gates parses metadata with stdlib only."""
    from foundationscale.gates.checkpoint_gates import CheckpointGateContext

    return CheckpointGateContext.from_path(ckpt_dir)


ContextBuilder = Callable[[Path | str], Any]


def _run_save_gates(
    registry: GateRegistry,
    ckpt_dir: Path,
    *,
    context_builder: ContextBuilder | None = None,
    event: Lifecycle = Lifecycle.SAVE,
) -> tuple[GateReport | None, Exception | None]:
    builder = context_builder or _default_context_builder
    try:
        ctx = builder(ckpt_dir)
    except Exception as exc:  # noqa: BLE001 -- reported as UNMEASURED by callers
        return None, exc
    # Typed dispatch for the same reason FoundationScaleSaveGate.on_save uses it: a
    # broadcast hands gates from other context families a context they cannot read,
    # and the resulting AttributeError is scored as a blocking ERROR (#250). This is
    # the second of the two call sites, and it is the one the end-of-run adjudicator
    # reaches -- fixing only on_save left the run stopping at exactly the same place,
    # one instrument later, which is why both are stated here rather than shared.
    return run_event(registry, event, ctx, missing_ctx="report-skip"), None


# The safetensors dtypes accepted as agreeing with each declared precision.
# bf16/fp16 tolerate F32: under transformers autocast the MASTER weights stay
# fp32 and save_pretrained serializes those, so demanding purity would turn
# every healthy autocast run RED. The detector limit this tolerance creates --
# an honest fp32 run and a bf16 run with fp32 masters serialize identically --
# is stated as WHAT IS NOT CLAIMED on check_precision_agreement.
_PRECISION_ACCEPTED_DTYPES: dict[str, tuple[str, ...]] = {
    "bf16": ("BF16", "F32"),
    "fp16": ("F16", "F32"),
    "fp32": ("F32",),
}

# The precisions this plane can EXECUTE, as torch attribute names resolved at the
# call site (a module-scope torch attribute would put an import in a module the
# torch-free gates scope, #325).
#
# nvfp4 is deliberately absent. It is DECLARABLE -- PRECISIONS admits it, so an
# operator can name it and the refusal is about the backend rather than the
# spelling -- and it has no executor, so train() refuses 96 well before the load.
# The gap this table closes is that fp32 and fp16 were in exactly the same
# position and did NOT refuse: #342 gave TrainConfig a precision field, and
# nothing ever bound it to the load. `model_kwargs` carried attn_implementation
# and nothing else, so `--precision fp32` loaded a bf16 checkpoint as bf16 and
# trained it, and the first save reported
#   declared precision='fp32' but the saved tensors disagree: accepted dtypes
#   ('F32',), observed {'BF16': 338}
# -- a RED on a scientific claim the plane had never attempted. One declared axis
# must not have two honesty contracts: either execute the declaration, or refuse
# it by name. bf16/fp16/fp32 are executable, so they are executed.
_PRECISION_TORCH_DTYPES: dict[str, str] = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


def _dtype_histogram(ckpt_dir: Path) -> dict[str, int] | None:
    """Count saved tensors by safetensors dtype, stdlib only (no torch).

    Reads the 8-byte header length and the JSON header of every
    ``*.safetensors`` shard, where each tensor entry carries its ``dtype``
    string. Returns None when there is nothing to look at -- zero shards, or
    zero tensors across them -- so the caller REFUSES the comparison as
    vacuous. It never returns ``{}``: an empty dict from here would be a
    measured zero over an unmeasured set. A malformed shard RAISES; the caller
    treats unreadable headers the same as absent tensors.
    """
    counts: dict[str, int] = {}
    total = 0
    for shard in sorted(ckpt_dir.glob("*.safetensors")):
        with shard.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                raise ValueError(f"{shard} is too short to be a safetensors file")
            (header_len,) = struct.unpack("<Q", raw)
            header = json.loads(handle.read(header_len))
        if not isinstance(header, dict):
            raise ValueError(f"{shard} has a non-object safetensors header")
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta.get("dtype") if isinstance(meta, dict) else None
            if not isinstance(dtype, str):
                raise ValueError(f"{shard}:{name} carries no usable dtype in its header")
            counts[dtype] = counts.get(dtype, 0) + 1
            total += 1
    if total == 0:
        return None
    return counts


@dataclass(frozen=True)
class PrecisionAgreement:
    """The outcome of one observed-vs-declared precision comparison.

    ``status`` is a word, not an exit code: "pass" / "red" / "abstain" /
    "refuse". ``observed`` is None whenever there was nothing to look at --
    it is never ``{}`` for the unmeasured case (doctrine: abstention is a
    value, and an empty histogram is not one).
    """

    status: str
    declared: str | None
    observed: dict[str, int] | None
    message: str


def check_precision_agreement(
    declared: str | None,
    observed: Mapping[str, int] | None,
) -> PrecisionAgreement:
    """Compare the declared precision against the observed saved dtypes.

    WHAT IS CLAIMED: agreement means every dtype in the observed histogram is
    accepted for the declared precision (``_PRECISION_ACCEPTED_DTYPES``);
    disagreement reports status "red" and its message names the declaration,
    the full observed histogram, the total tensor count, and the per-dtype
    counts that were rejected. Nothing declared (None) abstains -- UNMEASURED,
    never PASS. Nothing to look at (None or empty) refuses as vacuous -- a
    comparison over zero tensors is the ``all([]) is True`` failure and must
    never pass. A declared precision with no accepted-dtype mapping (nvfp4:
    declarable, no backend) refuses rather than guessing.

    WHAT IS NOT CLAIMED: this reads serialized dtypes, not compute. An honest
    fp32 run and a bf16-autocast run with fp32 master weights serialize the
    same dtypes, so dtype inspection cannot separate them; separating compute
    precision from master-weight storage requires in-step telemetry that the
    artifact does not carry. This check contracts to serialize-time truth only.
    """
    # ORDER IS LOAD-BEARING. The absence of a DECLARATION is adjudicated before
    # the absence of TENSORS, and the two are different states. A refusal says
    # "a claim was made and could not be checked"; with nothing declared there is
    # no claim, so refusing would sink every run that simply did not pass
    # --precision. The vacuity rule (all([]) is True) applies to the denominator
    # of a claim, and an empty denominator under no claim is UNMEASURED.
    if declared is None:
        # Bind the narrowed histogram ONCE rather than testing an `empty` flag
        # twice: a bool carries no narrowing, so `observed.items()` under
        # `if not empty` is only provably safe to a reader, not to the type
        # checker -- and the same two calls could drift apart under edit.
        seen = None if observed is None or not observed else dict(sorted(observed.items()))
        return PrecisionAgreement(
            status="abstain",
            declared=None,
            observed=seen,
            message=(
                "declared precision=None: nothing was declared, so there is no "
                "claim to check "
                + (
                    "against, and 0 saved tensors were available to inspect (both sides absent)"
                    if seen is None
                    else f"the observed histogram {seen} against"
                )
                + " -- UNMEASURED, abstaining, never PASS"
            ),
        )
    if observed is None or not observed:
        return PrecisionAgreement(
            status="refuse",
            declared=declared,
            observed=None,
            message=(
                f"precision check refused as vacuous: precision={declared!r} was "
                f"declared but 0 saved tensors were available to inspect; a "
                "comparison over nothing must not pass"
            ),
        )
    accepted = _PRECISION_ACCEPTED_DTYPES.get(declared)
    if accepted is None:
        return PrecisionAgreement(
            status="refuse",
            declared=declared,
            observed=dict(sorted(observed.items())),
            message=(
                f"declared precision={declared!r} is declarable but has no "
                f"accepted-dtype mapping in the package plane (no backend); "
                "refusing to guess one rather than passing by the rule of "
                "another precision"
            ),
        )
    rejected = {d: n for d, n in observed.items() if d not in accepted}
    if rejected:
        return PrecisionAgreement(
            status="red",
            declared=declared,
            observed=dict(sorted(observed.items())),
            message=(
                f"declared precision={declared!r} but the saved tensors "
                f"disagree: accepted dtypes {accepted}, observed "
                f"{dict(sorted(observed.items()))} over "
                f"{sum(observed.values())} tensor(s); rejected dtypes "
                f"{dict(sorted(rejected.items()))}"
            ),
        )
    return PrecisionAgreement(
        status="pass",
        declared=declared,
        observed=dict(sorted(observed.items())),
        message=(
            f"declared precision={declared!r} agrees with the saved tensors: "
            f"{dict(sorted(observed.items()))} all within accepted dtypes "
            f"{accepted}"
        ),
    )


class FoundationScaleSaveGate(_CallbackBase):
    """``TrainerCallback`` wiring the registered checkpoint gates into ``on_save``.

    On every save it runs the gate sweep for the lifecycle event over the
    just-written checkpoint directory. A blocking verdict BOTH sets
    ``control.should_training_stop = True`` AND is recorded on the instance
    (``.blocked`` / ``.reports`` / ``.records``). A gate that fires and lets
    the run continue is a check that cannot fail; this callback fails closed.

    The FIRST save additionally runs the observed-vs-declared precision check
    (DELIVERABLE 1c): the dtype histogram of the just-written shards is read
    with stdlib only and compared against ``declared_precision`` by
    :func:`check_precision_agreement`. It rides THIS callback -- the same
    ``.blocked`` / ``.records`` / ``should_training_stop`` machinery as the
    registered gates -- rather than a parallel reporter, and it does not
    depend on the checkpoint context, so it still runs on a host where the
    context family cannot be built. It is deliberately NOT a registry gate:
    a ``Gate`` on ``Lifecycle.FIRST_SAVE`` receives only a context built from
    the checkpoint directory, and the declared precision lives on
    ``TrainConfig``, not beside the checkpoint; faking that handoff through
    ``checkpoint.save_complete``'s manifest read would make the declaration's
    provenance depend on the artifact family being gated. What promoting it
    would require is named in RISKS.

    Importable without transformers/torch: with the extra absent the base
    degrades to ``object`` and the class is driven directly in tests.
    """

    def __init__(
        self,
        registry: GateRegistry | None = None,
        context_builder: ContextBuilder | None = None,
        declared_precision: str | None = None,
    ) -> None:
        self.registry = registry if registry is not None else REGISTRY
        self.context_builder = context_builder or _default_context_builder
        # None is carried to the comparator AS None: it means nothing was
        # declared and the check abstains. Defaulting it here to, say, "bf16"
        # would silently upgrade every direct construction of this callback
        # into a declared-precision run -- the coercion #342 forbids.
        self.declared_precision = declared_precision
        self.precision_agreement: PrecisionAgreement | None = None
        self.reports: list[GateReport] = []
        self.records: list[dict[str, Any]] = []
        self.blocked = False
        self._saves = 0

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        step = getattr(state, "global_step", self._saves)
        event = Lifecycle.FIRST_SAVE if self._saves == 0 else Lifecycle.SAVE
        self._saves += 1
        ckpt_dir = Path(getattr(args, "output_dir", ".")) / f"checkpoint-{step}"
        # Observed-vs-declared precision at the FIRST save, through this
        # callback's blocked/records/should_training_stop state -- the existing
        # save-gate machinery, not a side channel. It runs BEFORE the context
        # build and independently of it: the histogram needs only the shard
        # headers, so an unbuildable checkpoint context degrades the registered
        # gates to UNMEASURED without silencing this check. RED and a vacuous
        # REFUSE both stop the run; an abstention (nothing declared) does not
        # move the verdict -- it is recorded, and the manifest reports it.
        if event is Lifecycle.FIRST_SAVE:
            try:
                histogram = _dtype_histogram(ckpt_dir)
            except Exception as exc:  # noqa: BLE001 -- unreadable shard headers
                histogram = None
                _mark(
                    Step.SAVE_GATE,
                    f"precision: could not read safetensors headers under {ckpt_dir} "
                    f"({exc!r}); the comparison will refuse as vacuous",
                )
            agreement = check_precision_agreement(self.declared_precision, histogram)
            self.precision_agreement = agreement
            self.records.append(
                {
                    "event": f"{event.value}.precision",
                    "checkpoint": str(ckpt_dir),
                    "verdicts": {"precision.observed_vs_declared": agreement.status},
                    "declared": agreement.declared,
                    "observed": agreement.observed,
                }
            )
            _mark(
                Step.SAVE_GATE,
                f"precision {agreement.status.upper()}: {agreement.message}",
            )
            if agreement.status in ("red", "refuse"):
                self.blocked = True
                control.should_training_stop = True
        try:
            ctx = self.context_builder(ckpt_dir)
        except Exception as exc:  # noqa: BLE001 -- expected on undecodable saves
            _mark(
                Step.SAVE_GATE,
                f"UNMEASURED 0/0 gates: cannot build checkpoint context for {ckpt_dir}: {exc}",
            )
            return control
        # Typed dispatch, not GateRegistry.run. `run` broadcasts one context to every
        # gate registered for the event, so a gate from another context family
        # (parity, objective) is handed a CheckpointGateContext and dies inside
        # check() as a raw AttributeError one frame down -- which the sweep counts as
        # a blocking ERROR and the loop turns into should_training_stop. Whether a run
        # trained at all then depended on whether anything in the process had imported
        # foundationscale.verify.parity, because registration is an import side effect
        # (#250). run_event consults the declared Gate.context_type instead; its own
        # docstring names this broadcast failure as the reason it exists.
        #
        # The context is passed BARE, not as {type(ctx): ctx}. A typed map is the
        # stronger form, but run_event refuses to hand a mapping to a gate declaring
        # no context_type -- choosing an entry for it would be a guess -- so a map
        # here turns every legacy gate a caller registered through the `registry`
        # argument into an ERROR. The bare object is the documented migration shape:
        # it reaches a declaring gate by isinstance, and a legacy gate unchanged.
        #
        # missing_ctx="report-skip" declares the abstention rather than blocking on it.
        # This backend writes one source per save, so there is no second checkpoint for
        # a parity gate to compare against: the context is absent because the
        # comparison does not exist here, not because the wiring was forgotten.
        # Blocking would be fail-closed against the wrong proposition. The gates stay
        # in the printed denominator and surface as SKIP with a detail naming them --
        # an abstention that is visible, and never a PASS.
        report = run_event(self.registry, event, ctx, missing_ctx="report-skip")
        self.reports.append(report)
        self.records.append(
            {
                "event": event.value,
                "checkpoint": str(ckpt_dir),
                "verdicts": {r.gate_id: r.verdict.value for r in report.results},
            }
        )
        registered = report.registered if report.registered is not None else len(report.results)
        denominator = f"{len(report.results)}/{registered} gates"
        blockers = report.blocking
        if blockers:
            self.blocked = True
            control.should_training_stop = True
            _mark(
                Step.SAVE_GATE,
                f"RED {denominator}, {len(blockers)} blocking "
                f"({', '.join(g.gate_id for g in blockers)}); "
                "should_training_stop=True -- the run stops NOW",
            )
        else:
            _mark(Step.SAVE_GATE, f"PASS {denominator} over {ckpt_dir}")
        return control


# The gates whose blocking verdict on the BACKSTOP arm means "the loss was never
# readable", not "the objective is wrong". Named as a set rather than tested
# inline so the distinction is one declared thing: adding a gate that reads the
# loss component means adding it here, and forgetting to is visible.
_UNREAD_ON_BACKSTOP = frozenset({"objective.loss_components"})


class FoundationScaleObjectiveGate(_CallbackBase):
    """``TrainerCallback`` wiring the registered objective gates into the run.

    Dispatches ``Lifecycle.STEP_ZERO`` exactly once, carrying the declared
    objective, the live objective hyperparameters, the step-0 record snapshotted
    at ``on_train_begin``, and the observed loss. A blocking verdict BOTH sets
    ``control.should_training_stop = True`` AND is recorded on the instance
    (``.blocked`` / ``.unmeasured`` / ``.reports``), exactly as the save gate
    does: a gate that fires and lets the run continue cannot fail.

    WHEN it fires is the load-bearing decision, and it was measured rather than
    assumed. The dispatch happens at the first step for which a loss has been
    OBSERVED -- the first training log carrying ``"loss"`` -- not at
    ``global_step == 1``. The trainer's logging cadence, not the gate, decides
    when a loss exists: this loop bound ``logging_steps: 10`` unconditionally,
    so at step 1 there was no loss, and a step-1 dispatch over the four
    registered gates measured ``report.ok=False,
    blocking=['objective.loss_components']`` on a perfectly healthy run. That
    verdict is a property of the cadence knob, not of the objective, and it
    would have stopped every real run at step 1.

    Firing on the first observable step cannot silently never happen: if no loss
    is ever logged, ``on_train_end`` dispatches the same sweep with the component
    unobserved. That arm is measured too -- it blocks on
    ``objective.loss_components`` -- but it is recorded as ``.unmeasured``
    rather than ``.blocked``, because a component the instrument never got to
    read is UNMEASURED (95) and not RED (5). A gate that can quietly not run is
    the vacuous pass this codebase refuses; a backstop that lies about which
    state it is in is the same defect.

    The backstop is now hard to reach, which is the point: the cadence is the
    declared ``logging_steps`` when the operator set one, and is otherwise
    bound to ``max(1, min(10, max_steps))``, so every run emits at least one
    training log. Reaching it means the trainer logged nothing at all, which
    is worth hearing about rather than papering over.

    Importable without transformers/torch: with the extra absent the base
    degrades to ``object`` and the class is driven directly in tests.
    """

    def __init__(
        self,
        objective: str | None,
        learning_rate: float,
        per_device_batch_size: int,
        max_steps: int,
        registry: GateRegistry | None = None,
    ) -> None:
        # The scalars the gate reads are passed individually, not as the whole
        # TrainConfig. A callback holding the config can read a field it was
        # never meant to see, and -- worse for a fingerprint -- any future
        # config field would silently change what the step-0 snapshot covers
        # without this signature changing. Naming them here makes the snapshot's
        # contents a property of this class, not of whatever TrainConfig grows.
        self.objective = objective
        self.learning_rate = learning_rate
        self.per_device_batch_size = per_device_batch_size
        self.max_steps = max_steps
        self.registry = registry if registry is not None else REGISTRY
        self.reports: list[GateReport] = []
        self.blocked = False
        self.unmeasured = False
        self._fired = False
        self._last_loss: float | None = None
        self._step0_hparams: Mapping[str, Any] | None = None
        self._step0_fingerprint: str | None = None

    def _hparams(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "per_device_batch_size": self.per_device_batch_size,
            "max_steps": self.max_steps,
        }

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        # Local import, the same idiom as _default_context_builder: this module
        # must stay importable on a torch-free host, so nothing gate-adjacent is
        # imported at module scope.
        from types import MappingProxyType

        from foundationscale.gates.objective_gates import fingerprint_hparams

        # The STEP-0 RECORD: taken ONCE, here, and never recomputed. It is what
        # hparam_drift compares the live mapping against at dispatch; deriving it
        # lazily at dispatch would compare the run against itself and drift would
        # be vacuously PASS. Wrapped immutable so nothing between here and the
        # dispatch can edit the record it claims to be checking against.
        snapshot = MappingProxyType(self._hparams())
        self._step0_hparams = snapshot
        self._step0_fingerprint = fingerprint_hparams(snapshot)
        return control

    def on_log(
        self,
        args: Any,  # noqa: ARG002
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Any:
        # Training logs carry "loss"; eval-only logs do not. When the key is
        # absent nothing is recorded rather than invented: a fabricated 0.0 would
        # be scored by the coverage gate as a real, perfect measurement.
        if logs is None or "loss" not in logs:
            return control
        self._last_loss = float(logs["loss"])
        if self._fired or getattr(state, "global_step", 0) < 1:
            return control
        return self._dispatch(control, observed=True)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        if self._fired:
            return control
        return self._dispatch(control, observed=False)

    def _dispatch(self, control: Any, *, observed: bool) -> Any:
        from foundationscale.gates.objective_gates import (
            LossComponent,
            ObjectiveGateContext,
            ValueProvenance,
        )
        from foundationscale.rl import LossOutput, build_objective_gate_context

        # Fire EXACTLY ONCE. The boolean guard -- not a step comparison -- is
        # what makes "once" hold: a trainer that resumes mid-run or emits an
        # extra log cannot re-fire STEP_ZERO and append a second, contradictory
        # record to self.reports.
        self._fired = True

        objective_prov: ValueProvenance | None = None
        if self.objective is not None:
            # source="config" because the value came from a TrainConfig field.
            # The gate refuses any source outside {"cli", "config", "default",
            # "env"}, so the source is stated, not invented.
            objective_prov = ValueProvenance(
                name="objective", value=self.objective, source="config", recorded=True
            )

        if observed and self._last_loss is not None:
            scalar = self._last_loss
            component = LossComponent(
                name="sft_loss", weight=1.0, observed=True, contribution=scalar
            )
        else:
            # Absent is not zero: the component goes in unmeasured with
            # contribution=None so the gate sees the hole. LossOutput.loss is a
            # required float and the bridge reads only `.components`, so the
            # scalar is NaN rather than a plausible number -- if a future reader
            # does consume it, NaN propagates instead of passing for a
            # measurement. A measured 0.0 is a real observation; this is not one.
            scalar = float("nan")
            component = LossComponent(
                name="sft_loss", weight=1.0, observed=False, contribution=None
            )

        ctx = build_objective_gate_context(
            LossOutput(loss=scalar, components=(component,)),
            objective=objective_prov,
            declared_components=("sft_loss",),
            # Live mapping, rebuilt here -- NOT the step-0 snapshot. hparam_drift
            # exists to compare the two; passing the snapshot would make the
            # comparison reflexive and the gate vacuously PASS.
            current_hparams=self._hparams(),
            # Keyword-required with no default: the context builder refuses to
            # let the step-0 record be forgotten.
            step0_fingerprint=self._step0_fingerprint,
            step0_hparams=self._step0_hparams,
            origin="<train-loop>",
        )
        # Typed dispatch for the same reason FoundationScaleSaveGate.on_save uses
        # it: GateRegistry.run broadcasts one context to every gate on the event,
        # and a gate from another context family dies on a context it cannot
        # read, which the sweep counts as a blocking ERROR (#250).
        # The MAPPING form, not the bare context the two save-gate call sites
        # pass. run_event implements both (it branches on isinstance(contexts,
        # Mapping)) but declares only `Mapping[type, Any]`, so the bare form
        # typechecks there solely because those contexts are typed Any -- an
        # annotation narrower than the implementation, holding only where the
        # caller has no types. Naming ObjectiveGateContext here also states which
        # family this context belongs to at the call site instead of leaving it
        # to isinstance discovery. Measured identical to the bare form: same
        # gates, same verdicts, same report.
        report = run_event(
            self.registry,
            Lifecycle.STEP_ZERO,
            {ObjectiveGateContext: ctx},
            missing_ctx="report-skip",
        )
        self.reports.append(report)
        denominator = f"{len(report.results)}/{report.registered} gates"
        blockers = report.blocking
        if not blockers:
            _mark(Step.OBJECTIVE_GATE, f"PASS {denominator} at the first observed step")
            return control
        # On the backstop arm the unread component is reporting that it COULD NOT
        # measure, which is 95 and not 5. A blocker from any other gate is a real
        # refusal even here, so the two are separated by gate id rather than by
        # which arm we happen to be on.
        abstained = [] if observed else [g for g in blockers if g.gate_id in _UNREAD_ON_BACKSTOP]
        refusals = [g for g in blockers if g not in abstained]
        if not refusals:
            self.unmeasured = True
            _mark(
                Step.OBJECTIVE_GATE,
                f"UNMEASURED {denominator}, {len(abstained)} abstaining "
                f"({', '.join(g.gate_id for g in abstained)}): no training log "
                "carried a loss before the run ended, so the gate had nothing to "
                "read -- the objective claim is unfalsifiable, not refuted",
            )
            return control
        self.blocked = True
        control.should_training_stop = True
        # COUNT and NAMES come from the same list. They did not, once: the count
        # was taken over every blocker and the names over the refusals alone, so
        # a backstop dispatch printed "2 blocking (objective.declared)" -- a
        # denominator and a numerator from different sets, in the one line an
        # operator reads to find out what stopped the run.
        note = (
            ""
            if not abstained
            else f", plus {len(abstained)} abstaining ({', '.join(g.gate_id for g in abstained)})"
        )
        _mark(
            Step.OBJECTIVE_GATE,
            f"RED {denominator}, {len(refusals)} blocking "
            f"({', '.join(g.gate_id for g in refusals)}){note}; "
            "should_training_stop=True -- the run stops NOW",
        )
        return control


def _load_raw_dataset(hf_datasets: Any, ref: str) -> Any:
    """Load an HF dataset id, a .json/.jsonl file, or a directory of them."""
    p = Path(ref)
    if p.suffix in {".json", ".jsonl"}:
        return hf_datasets.load_dataset("json", data_files=str(p))
    if p.is_dir():
        files = sorted(str(f) for f in p.glob("*.json*"))
        if files:
            return hf_datasets.load_dataset("json", data_files=files)
    return hf_datasets.load_dataset(ref)


def _manifest_payload(
    cfg: TrainConfig,
    *,
    stage: str,
    extra: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "foundationscale.run-manifest/v1",
        "stage": stage,
        "argv": list(sys.argv),  # the composed launch command survives HERE
        "python": sys.version.split()[0],
        "foundationscale": fs_version(),
        "exit_contract": {
            "PASS": EXIT_PASS,
            "RED": EXIT_RED,
            "UNMEASURED": EXIT_UNMEASURED,
            "REFUSE": EXIT_REFUSE,
        },
        "config": {
            "model": cfg.model,
            "dataset": cfg.dataset,
            "output_dir": str(cfg.output_dir),
            "max_steps": cfg.max_steps,
            "per_device_batch_size": cfg.per_device_batch_size,
            "learning_rate": cfg.learning_rate,
            "save_interval": cfg.save_interval,
            "seed": cfg.seed,
            "topology": {
                "dp": cfg.dp,
                "tp": cfg.tp,
                "pp": cfg.pp,
                "ep": cfg.ep,
                "cp": cfg.cp,
            },
            "nodes": cfg.nodes,
            "gpus_per_node": cfg.gpus_per_node,
            "profile_name": cfg.profile_name,
            "profile_path": (str(cfg.profile_path) if cfg.profile_path is not None else None),
            "dry_run": cfg.dry_run,
            # Declared, never coerced: None here means the run did not say,
            # and the manifest carries that absence rather than a plausible
            # dtype (finding #342). The OBSERVED dtype histogram is a separate
            # key, written at done-time from the first-save measurement.
            "precision": cfg.precision,
            "adapter": cfg.adapter,
            "adapter_rank": cfg.adapter_rank,
            "adapter_alpha": cfg.adapter_alpha,
            "adapter_targets": (
                list(cfg.adapter_targets) if cfg.adapter_targets is not None else None
            ),
            "adapter_dropout": cfg.adapter_dropout,
            # The nine declaration axes are recorded UNCONDITIONALLY -- None and
            # all. That is the precision rule (#342) applied to every axis: a
            # key present carrying None says "the operator abstained and the
            # engine default applied, unclaimed"; a missing key would say
            # "this version of the loop never populated the field", and a
            # reader must be able to tell those apart. Omitting None keys here
            # is exactly the defect provenance exists to catch: an axis the
            # manifest cannot distinguish from unpopulated.
            "optimizer": cfg.optimizer,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_grad_norm": cfg.max_grad_norm,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "attn_implementation": cfg.attn_implementation,
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "warmup_steps": cfg.warmup_steps,
            "logging_steps": cfg.logging_steps,
            "sharding_strategy": cfg.sharding_strategy,
            "cpu_optimizer_offload": cfg.cpu_optimizer_offload,
        },
        "extra": extra or {},
        # Outcome telemetry is its own top-level section, never folded into
        # config: config is what the run DECLARED, telemetry is what the run
        # MEASURED, and a reader diffing two runs' configs must not have to
        # separate one from the other. Empty before training completes -- an
        # empty section at stage="train" is a true statement, not a missing
        # one. The schema string stays v1: the key is additive, and the
        # validated manifest's schema_version is owned by the provenance
        # package, not by this payload.
        "telemetry": telemetry or {},
    }


def _run_id(cfg: TrainConfig) -> str:
    """A run identifier that is stable within a run and distinct across runs.

    ``FS_RUN_ID`` wins when set, so a launcher that already owns run identity
    can impose it and have the manifest agree with the scheduler's records.
    Otherwise it is derived from the output directory and the argv digest:
    re-invoking the SAME command into the SAME directory is a resumption of one
    run and gets one id, while changing either is a different run. Deriving it
    rather than randomising keeps a re-run reproducible.
    """
    override = os.environ.get("FS_RUN_ID")
    if override:
        return override
    digest = sha256(" ".join(sys.argv).encode("utf-8")).hexdigest()[:8]
    return f"{Path(cfg.output_dir).name or 'run'}-{digest}"


# Config keys that state HOW MANY routed experts a layer has, across the model
# families this path has to serve without knowing any of them by name. Listed as
# data because the alternative -- `if "mixtral" in model_type` -- is how a
# framework becomes a framework for two models.
#
# `num_experts_per_tok` is ABSENT on purpose and its absence is load-bearing: it
# is the router's top-k, not the expert count, and every MoE config carries it.
# Reading it as a count would declare an 8-expert denominator for a 128-expert
# layer and the byte gate would then confirm a checkpoint 94% short.
_EXPERT_COUNT_KEYS: tuple[str, ...] = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_experts_per_layer",
)


def _tied_aliases(model: Any, names: set[str]) -> set[str]:
    """Names present in ``state_dict`` that the saved artifact will NOT contain.

    A tied weight is one tensor under two names. ``save_pretrained`` writes the
    source and drops the alias, so declaring the alias asserts the checkpoint
    should hold something the format never stores -- and the completeness gate
    would report a missing tensor on a perfectly healthy save. MEASURED on
    tiny-random-gpt2: 65 state_dict keys, 64 artifact tensors, the difference
    being ``lm_head.weight`` tied to ``transformer.wte.weight``. Most causal LMs
    tie by default, so the naive declaration is not an edge case -- it is the
    common case, and it fails in the dangerous direction of a false RED.

    ``_tied_weights_keys`` is a list in transformers 4.x and a dict (alias ->
    source) in 5.x, so it is read by shape rather than by version, the same way
    the TrainingArguments knob is bound by introspection below. Gated on the
    config flag: the attribute names what WOULD be tied, and a model configured
    with tying off saves those tensors for real.
    """
    if not getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        return set()
    keys = getattr(type(model), "_tied_weights_keys", None) or getattr(
        model, "_tied_weights_keys", None
    )
    if isinstance(keys, dict):
        candidates = set(keys)
    elif isinstance(keys, (list, tuple, set)):
        candidates = {str(k) for k in keys}
    else:
        return set()
    return candidates & names


def _declare_checkpoint(model: Any) -> tuple[Any, dict[str, str]]:
    """Build the gates' denominator from the model IN MEMORY, before any save.

    The checkpoint gates cannot say "64 tensors were declared and 61 arrived"
    without a declared 64, and the doctrine is explicit that the denominator has
    to be produced independently of the artifact. Reading it back off the
    safetensors we just wrote would be a tautology -- the file would agree with
    itself and the gate would pass on a checkpoint that dropped half the model.
    ``state_dict()`` is the honest source: it is what the trainer holds and
    therefore what the save is obliged to persist.

    Dense-vs-MoE is decided from TWO sources that must agree, and disagreement
    is reported rather than resolved:

      * the config's expert-count key, if it has one;
      * whether any parameter name mentions experts, using the gates' own
        vocabulary (:func:`~foundationscale.gates.checkpoint_gates.mentions_expert`).

    ``num_experts=0`` is a POSITIVE dense declaration, not a default, and the
    schema only accepts it with ``moe_layer_basis`` recorded -- an unexplained
    denominator is an unaccountable one. Where the two sources disagree, or a
    count is present that this path cannot price, ``num_experts`` is left None:
    the gates then read UNKNOWN and fail closed, which is the correct outcome
    for a run whose shape we could not establish. #54 is the finding that says
    absence-of-key must never mint a zero.

    DELIVERABLE 2e -- correctness under peft. What I found, by inspection of
    the peft save path (the live experiment is named in RISKS): the
    declaration path BREAKS under a peft-wrapped model. ``Trainer.save_model``
    on a PeftModel persists ADAPTER tensors only (``adapter_model.safetensors``),
    while ``state_dict()`` still reports every frozen base weight, now under a
    ``base_model.model.*`` prefix. Declaring the full state-dict would assert
    FQNs the artifact never contains, and the completeness gate would report
    the entire base model missing on every healthy LoRA run -- a false RED,
    the same shape as the tied-alias defect corrected above. The declaration
    is therefore scoped to the adapter tensors when a peft wrapper is
    detected (``model.peft_config`` present), and ``declaration.adapter_scope``
    records which arm ran. The vacuous-adapter refusal upstream guarantees
    that scope is non-empty when adapter mode was declared; an undetected
    wrapper is the residual risk and is named in RISKS.

    Returns the declaration and a dict of audit notes for the manifest's config
    block, so the basis of every number here survives into the artifact.
    """
    from foundationscale.gates.checkpoint_gates import matches_expert_family, mentions_expert
    from foundationscale.provenance.manifest import DeclaredCheckpoint

    state = model.state_dict()
    names = set(state)
    tied = _tied_aliases(model, names)
    if getattr(model, "peft_config", None) is not None:
        # peft persists adapters only (see 2e in the docstring): the honest
        # denominator is the adapter subset of the state dict, not the frozen
        # base weights the artifact will never contain.
        declared = {n for n in names if ".lora_" in n} - tied
        adapter_scope = (
            f"peft-wrapped model: declared {len(declared)} adapter tensor(s) "
            "only; the saved artifact carries adapter weights, not base weights"
        )
    else:
        declared = names - tied
        adapter_scope = "full model (no peft wrapper detected on model.peft_config)"
    notes = {
        "declaration.source": "model.state_dict() in memory, before the first save",
        "declaration.state_dict_keys": str(len(names)),
        "declaration.tied_excluded": ",".join(sorted(tied)) or "(none)",
        "declaration.adapter_scope": adapter_scope,
    }

    config = getattr(model, "config", None)
    found = {
        key: int(getattr(config, key))
        for key in _EXPERT_COUNT_KEYS
        if isinstance(getattr(config, key, None), int)
    }
    mentioned = sorted(n for n in declared if mentions_expert(n))
    notes["declaration.config_expert_keys"] = (
        ", ".join(f"{k}={v}" for k, v in sorted(found.items())) or "(none present)"
    )
    notes["declaration.expert_named_tensors"] = str(len(mentioned))

    counts = {v for v in found.values() if v > 0}
    if not counts and not mentioned:
        # Two independent sources agree on dense. This is the two-source contract
        # of #59: neither "the config has no expert key" nor "no tensor is named
        # like an expert" is sufficient alone, because the first is satisfied by
        # a config we failed to parse and the second by a naming scheme we do
        # not recognise. Together they are a measurement.
        basis = (
            f"dense: config declares none of {list(_EXPERT_COUNT_KEYS)}, and 0 of "
            f"{len(declared)} declared tensors carry an expert path segment"
        )
        notes["declaration.basis"] = basis
        return (
            DeclaredCheckpoint(
                num_experts=0,
                num_moe_layers=None,
                expected_expert_bytes=None,
                declared_fqns=tuple(sorted(declared)),
                moe_layer_basis=basis,
            ),
            notes,
        )

    # From here the model is MoE, or the two sources disagree. Price only the
    # tensors whose layout the gates can actually verify -- counting a name we
    # cannot parse would inflate the byte denominator and turn the byte gate
    # into a check that always fails.
    priced = [n for n in declared if matches_expert_family(n)]
    expert_bytes: int | None = None
    if priced:
        try:
            expert_bytes = sum(int(state[n].numel()) * int(state[n].element_size()) for n in priced)
        except Exception:  # noqa: BLE001 -- a tensor that cannot be sized is not priced
            expert_bytes = None
    num_experts = next(iter(counts)) if len(counts) == 1 else None
    if num_experts is None:
        basis = (
            f"UNKNOWN: expert-count keys {found or '{}'} do not agree on a single "
            f"value while {len(mentioned)} tensor(s) are expert-named; the gates "
            "must fail closed rather than adopt one of them"
        )
    elif not mentioned:
        basis = (
            f"UNKNOWN: config declares {num_experts} experts but 0 of "
            f"{len(declared)} tensors are expert-named -- the two sources "
            "disagree, so neither is adopted"
        )
        num_experts = None
    else:
        basis = (
            f"MoE: config key(s) {found} declare {num_experts} experts; "
            f"{len(mentioned)} expert-named tensor(s), {len(priced)} in a layout "
            "the gates can price"
        )
    notes["declaration.basis"] = basis
    notes["declaration.priced_expert_tensors"] = str(len(priced))
    return (
        DeclaredCheckpoint(
            num_experts=num_experts,
            num_moe_layers=None,
            expected_expert_bytes=expert_bytes,
            declared_fqns=tuple(sorted(declared)),
            moe_layer_basis=basis,
        ),
        notes,
    )


# Units of the outcome metrics the loop records, keyed by metric name. The
# unit describes the METRIC, not the outcome, so an entry that could not be
# measured keeps the unit of the thing it failed to measure. A key absent
# from this table is unitless (None) -- train_loss and every scalar like it.
_TELEMETRY_UNITS: dict[str, str] = {
    "train_runtime_s": "s",
    "samples_per_second": "samples/s",
    "steps_per_second": "steps/s",
    "peak_memory_allocated_bytes": "bytes",
    "peak_memory_reserved_bytes": "bytes",
    "logging_steps_effective": "steps",
}
# total_flos is deliberately ABSENT. It was declared here with no producer: this
# loop counts no FLOPs, so the key could never be recorded and the table stated a
# unit for a quantity that does not exist. A units table whose key set is wider
# than the emitted key set is the drift the telemetry unit control exists to
# catch, and it caught this one. Adding FLOP accounting later means adding the
# emission and this row together, never this row alone.


def _cuda_availability(torch_module: Any) -> tuple[bool, str]:
    """Decide whether a CUDA peak-memory counter exists, and say why when it does not.

    Args:
        torch_module: The imported ``torch``. Passed IN rather than read from
            module state so this decision is a unit with inputs. That is not
            decoration: deleting ``torch.cuda`` from the real module to reach the
            no-module branch breaks transformers long before control arrives
            here, so through ``train()`` alone that branch is unreachable, and an
            unreachable branch is untested by construction however it is written.

    Returns:
        ``(available, reason)``. ``reason`` is the UNMEASURED text to record when
        ``available`` is False, and is DISTINCT per branch: "this build exposes no
        torch.cuda" and "torch.cuda.is_available() is False" are different facts
        about the machine, and a reader of the manifest must be able to tell which
        one held. When ``available`` is True there is a counter to read and the
        reason is empty.
    """
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return False, (
            "UNMEASURED: this torch build exposes no torch.cuda module, so this "
            "run has no CUDA peak-memory counter to read"
        )
    if not bool(cuda.is_available()):
        return False, (
            "UNMEASURED: torch.cuda.is_available() is False, so this run has "
            "no CUDA peak-memory counter to read"
        )
    return True, ""


def _build_run_manifest(
    cfg: TrainConfig,
    *,
    stage: str,
    extra: dict[str, Any] | None,
    declared: Any = None,
    notes: dict[str, str] | None = None,
    telemetry: dict[str, tuple[Any, str]] | None = None,
) -> Any:
    """Construct the real :class:`~foundationscale.provenance.RunManifest`.

    This used to be a hand-rolled dict, and that is the whole of finding #225.
    The package ships a structured manifest with capture helpers, and
    ``dcp_meta.load_manifest`` -- the reader every checkpoint gate goes
    through -- validates against THAT schema. A bespoke payload beside a real
    reader is not provenance; it is a file. Three things were wrong at once and
    any one of them was sufficient to make every gate abstain:

      * it was written only at ``done``, after the last save that could read it;
      * under a name (``foundationscale.run.json``) the reader never searches;
      * without the four keys the reader requires at top level.

    The supplementary run detail (argv, stage, the exit contract) is carried in
    ``config`` as :class:`EffectiveValue` entries rather than as extra top-level
    keys, because that is the field for it and because unknown top-level keys
    are preserved and resurfaced as findings -- correct behaviour that would
    make every well-formed run report one.

    Returns ``None`` if provenance is unavailable, so the caller can degrade to
    the plain writer and SAY so instead of losing the manifest.
    """
    try:
        from foundationscale.provenance import (
            EffectiveValue,
            RunManifest,
            TelemetryEntry,
            capture_code_provenance,
            capture_environment,
        )
        from foundationscale.provenance import (
            Topology as ProvTopology,
        )
    except Exception:  # noqa: BLE001 -- provenance must never lose the manifest
        return None

    detail = _manifest_payload(cfg, stage=stage, extra=extra)
    config: dict[str, Any] = {}

    def _put(key: str, value: Any, source: str) -> None:
        config[key] = EffectiveValue(key=key, value=str(value), source=source)

    def _config_source(field: str) -> str:
        """Where this config field's value came from -- recorded, not guessed.

        Every key used to be stamped ``"cli"``, which said the operator typed
        it. For an omitted axis that is simply false, and it is the worst kind
        of false: the field exists BECAUSE the 24-run split happened when the
        decisive value came from somewhere nothing recorded, so a source that
        is filled in by construction reports full provenance while carrying
        none.

        The answer is not inferred from the value. ``value is None`` would work
        for the nine axes, whose only absence marker is None, and would be
        wrong for ``max_steps=20``, which is ambiguous between a typed flag and
        the field default. cli.py measures the distinction with a sentinel
        parse and hands the answer over in ``cli_declared``.

        A config the parser never touched gets ``"config"`` -- the vocabulary's
        word for a value set programmatically, and already what the objective
        provenance uses for a TrainConfig-sourced value. Guessing ``"default"``
        there would be the same defect one layer along: a caller that passed
        ``optimizer="adamw_torch"`` in Python did not accept a default.
        """
        if cfg.cli_declared is None:
            return "config"
        return "cli" if field in cfg.cli_declared else "default"

    for key, value in detail["config"].items():
        if isinstance(value, dict):
            # Flatten rather than str() a dict. `topology.tp = 2` is greppable
            # and diffable against another run's manifest; "{'dp': 1, 'tp': 2}"
            # is one opaque string whose equality depends on key insertion order.
            #
            # The SOURCE is looked up on the leaf name, not the dotted key:
            # `topology.tp` is a presentation choice of this emitter, while the
            # thing the operator did or did not type is `--tp`, whose field is
            # `tp`.
            for sub, subvalue in value.items():
                _put(f"{key}.{sub}", subvalue, _config_source(sub))
        else:
            # None reaches here for every undeclared axis and is written as
            # "None" -- an EXPLICIT record of absence. The ConfigResolver's
            # contract is {key, value, source, ...} for every field, and the
            # key is never what carries the abstention: key present + value
            # None = abstained; key absent = this loop never populated the
            # field. All nine declaration axes go through this same path.
            _put(key, value, _config_source(key))
    # argv is the composed launch command; without it a run is not reproducible
    # from its own output (#180). stage says WHICH point in the lifecycle wrote
    # this file, so a manifest found beside a checkpoint is self-dating.
    _put("argv", " ".join(detail["argv"]), "process")
    _put("stage", stage, "derived")
    _put("exit_contract", json.dumps(detail["exit_contract"]), "declared")
    # Which interpreter ran the gates is part of the verdict, not trivia: #83
    # made the gate reports carry it for exactly this reason, and a CLEAR that
    # does not say which Python produced it is unattributable.
    _put("python", detail["python"], "process")
    _put("foundationscale", detail["foundationscale"], "package")
    for key, value in (extra or {}).items():
        _put(f"extra.{key}", value, "derived")
    # How the declaration below was arrived at -- which config keys were read,
    # how many tensors were excluded as tied aliases, what decided dense-vs-MoE.
    # The declaration is the denominator every checkpoint verdict is computed
    # against, so a manifest that carries the number without its basis makes the
    # verdict unauditable: nobody reading it later can tell a measured 64 from a
    # guessed one.
    for key, value in (notes or {}).items():
        _put(key, value, "measured")

    # provenance.Topology and topology.Topology share field NAMES for different
    # quantities (#222), so the mapping is written out rather than splatted.
    prov_topology = ProvTopology(
        nodes=cfg.nodes,
        gpus_per_node=cfg.gpus_per_node,
        tensor_parallel=cfg.tp,
        pipeline_parallel=cfg.pp,
        data_parallel=cfg.dp,
        expert_parallel=cfg.ep,
        context_parallel=cfg.cp,
    )
    # Both capture helpers report a DECLARED status rather than raising when
    # there is nothing to capture (NOT_A_REPOSITORY outside a git tree), so an
    # unversioned working directory yields an honest manifest, not a missing one.
    # Outcome telemetry is ONE top-level section of TelemetryEntry, with the
    # value passed THROUGH unstringified -- a number surviving as a number is
    # the whole point of the type. No introspection probe remains here because
    # there is nothing to probe: foundationscale.provenance lives in THIS
    # repository and is versioned with this file, so a constructor keyword it
    # does not accept is a build error that must be loud, never a silent
    # degradation into config -- the section reserved for what the run
    # DECLARED.
    telemetry_section = {
        key: TelemetryEntry(
            key=key,
            value=value,
            source=source,
            unit=_TELEMETRY_UNITS.get(key),
        )
        for key, (value, source) in (telemetry or {}).items()
    }
    return RunManifest(
        run_id=_run_id(cfg),
        attempt=int(os.environ.get("FS_ATTEMPT", "1")),
        code=capture_code_provenance(Path.cwd(), entrypoint=sys.argv[0] or None),
        config=config,
        environment=capture_environment(),
        topology=prov_topology,
        artifact_paths={"output_dir": str(cfg.output_dir)},
        # None before a model exists (blocked, dry_run) -- and that is the right
        # value, not a placeholder: the gates read a missing declaration as
        # UNKNOWN and fail closed, which is exactly correct for a run that never
        # built the model it would have been declaring.
        declared=declared,
        telemetry=telemetry_section,
    )


def _emit_manifest(
    cfg: TrainConfig,
    *,
    stage: str,
    extra: dict[str, Any] | None = None,
    declared: Any = None,
    notes: dict[str, str] | None = None,
    telemetry: dict[str, tuple[Any, str]] | None = None,
) -> Path:
    """Write the run manifest where the checkpoint gates will look for it.

    ``MANIFEST_NAME`` is the reserved strict basename, chosen deliberately: a
    file under that name which is unreadable or incomplete RAISES rather than
    counting as absent, so a corrupt manifest cannot masquerade as a run that
    never wrote one. That is only a safe choice because what we write here is a
    validated ``RunManifest`` and not a bespoke dict.

    ``declared`` carries the checkpoint denominator when a model exists to
    derive it from; the gates read its absence as UNKNOWN and fail closed.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    manifest = _build_run_manifest(
        cfg, stage=stage, extra=extra, declared=declared, notes=notes, telemetry=telemetry
    )
    if manifest is None:
        # Degrade loudly. The previous implementation probed provenance for four
        # writer names it does not export and fell through here silently on
        # every single run, which read like integration and was none.
        path.write_text(
            json.dumps(
                # The degraded writer cannot use EffectiveValue -- that
                # import is exactly what failed above -- so the provenance
                # shape is spelled out field-for-field to match it. An
                # unmeasured entry keeps its "unmeasured" source and its
                # stated reason here too.
                _manifest_payload(
                    cfg,
                    stage=stage,
                    extra=extra,
                    telemetry={
                        key: {"value": str(value), "source": source, "findings": []}
                        for key, (value, source) in (telemetry or {}).items()
                    },
                ),
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        _mark(
            Step.MANIFEST,
            f"DEGRADED run manifest ({stage}) -> {path}: foundationscale.provenance "
            "is unavailable, so this file does NOT satisfy the gate reader's schema "
            "and every checkpoint gate will abstain",
        )
        return path
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    _mark(Step.MANIFEST, f"run manifest ({stage}) -> {path}")
    return path


def train(cfg: TrainConfig) -> int:
    """Run the thin path and adjudicate it, totally.

    Returns 0/5/95/96 per the exit-code contract. Never raises for an
    expected condition; unexpected trainer exceptions adjudicate as RED.

    That second sentence used to be a claim rather than an implementation.
    ``_train`` guards the trainer construction, the ``trainer.train()`` call
    and the final save individually -- but an AST census of its body found 65
    top-level statements standing outside any ``try`` whose handlers return an
    ``EXIT_`` constant, among them ``_effective_topology``, the gate-callback
    constructors, ``_emit_manifest`` and ``_run_save_gates``. The last one
    parses safetensors headers off disk and is the most realistic raiser of
    the set. An exception from any of them left ``_train``, left ``main``
    (``cli.py`` deliberately keeps its ``except ValueError`` off the call, so
    that a training failure is never reported as a refusal), and reached the
    interpreter, which prints a traceback and exits **1** -- the one code the
    0/5/95/96 contract forbids, and the code #171 already showed a launcher
    cannot interpret.

    Guarding those 65 sites one by one is the version of this fix that is
    wrong the day someone adds a 66th. The contract is a property of THIS
    function's boundary, so the boundary is where it is enforced: ``train``
    is a total wrapper over ``_train``. Inner handlers still run first and
    keep their specific verdicts -- REFUSE for an unhonourable declaration,
    RED for a trainer that failed -- and this catch only ever sees what none
    of them claimed. RED (not UNMEASURED) is the verdict, matching the
    existing ``except Exception`` arms around trainer construction and the
    save: an unclassified crash means the run did not establish its claim,
    and the safe direction is the blocking one. ``BaseException`` is
    deliberately not caught, so Ctrl-C and ``SystemExit`` pass through.

    ``precision='nvfp4'`` is REFUSED (96) before anything runs. That refusal
    is the choice, made deliberately over a seam: transformers has no
    TrainingArguments flag for nvfp4 and this loop wires no quantization
    seam of its own, so the only executable interpretation today would be
    "train at the loader's default dtype and claim nvfp4" -- the silent
    fallback that is finding #342's defect class and the worst possible
    outcome. An explicitly-named nvfp4 seam would name something that does
    not exist yet; a refusal names exactly what is true. The {full, LoRA} x
    {bf16, nvfp4} matrix keeps its nvfp4 cells empty until a real backend
    lands.

    The same refuse-over-claim rule now covers three more declarations,
    all at START, before a GPU is touched (the #375 batch): any of
    tp/pp/ep/cp > 1 (the backend executes data parallelism only);
    sharding_strategy outside {"ddp"}; and cpu_optimizer_offload=True
    (no offload backend exists in this plane). A declaration the runtime
    cannot honour is a manifest entry waiting to lie.
    """
    try:
        return _train(cfg)
    except Exception as exc:  # noqa: BLE001 -- the contract has no code for "crashed"
        # The traceback is the only diagnosis of an unclassified crash, so it
        # goes to stderr exactly where the interpreter would have put it. What
        # changes is the exit code, not the operator's evidence.
        traceback.print_exc(file=sys.stderr)
        # #409: an environment fault that escapes the thin path is still an
        # environment fault. Classify BEFORE adjudicating, or a full disk is
        # published as a crashed framework. This is the widest of the three
        # sites -- anything unguarded anywhere in _train lands here.
        environment = _environment_failure_reason(exc)
        if environment is None:
            step, verdict = Step.RED, EXIT_RED
            detail = (
                f"unhandled {type(exc).__name__} escaped the thin path: {exc}. "
                "Adjudicated RED (5) at the train() boundary rather than "
                "allowed to exit 1, which sits outside the 0/5/95/96 "
                "contract. Full traceback on stderr"
            )
        else:
            step, verdict = Step.REFUSE, EXIT_REFUSE
            detail = (
                f"unhandled {type(exc).__name__} escaped the thin path, and it is "
                f"an environment fault: {environment}. Adjudicated REFUSE (96), not "
                "RED (5): the machine failed, not the training plane, so no claim "
                "about this plane was measured. Full traceback on stderr"
            )
        try:
            _mark(step, detail)
            _emit_manifest(
                cfg,
                stage="crashed",
                extra={"exit": verdict, "unhandled_exception": type(exc).__name__},
            )
        except Exception:  # noqa: BLE001 -- reporting must not replace the verdict
            pass
        # Return the DECLARED constant on each branch rather than the name
        # bound above. Identical behaviour -- `verdict` is EXIT_RED exactly
        # when `environment is None` -- but checks/exit_contract_scope.py
        # resolves a returned constant and cannot resolve a name assigned in
        # two branches. Returning `verdict` put this entry point, main(), and
        # both `raise SystemExit(main())` sites into UNRESOLVED, which is
        # neither in-contract nor out: six sites in no axis at all (#381).
        if environment is None:
            return EXIT_RED
        return EXIT_REFUSE


def _train(cfg: TrainConfig) -> int:
    """Body of :func:`train`; see there for the contract this is wrapped in."""
    _mark(
        Step.START,
        f"model={cfg.model} dataset={cfg.dataset} "
        f"output_dir={cfg.output_dir} dry_run={cfg.dry_run}",
    )

    if cfg.precision == "nvfp4":
        _mark(
            Step.REFUSE,
            "precision='nvfp4' is declared, but the package plane has no nvfp4 "
            "backend yet: no TrainingArguments flag exists and no quantization "
            "seam is wired in this loop. Refusing rather than silently "
            "training at another dtype -- a silent bf16/fp32 fallback is "
            "finding #342's defect class",
        )
        _emit_manifest(
            cfg,
            stage="refused",
            extra={"exit": EXIT_REFUSE, "precision": "nvfp4"},
        )
        return EXIT_REFUSE

    # Measured, not assumed: cfg.tp/pp/ep/cp have ZERO execution consumers in
    # this plane. Every occurrence is a record or validate site, and the
    # Trainer kwargs dict built in step 6 carries no tensor/pipeline/expert/
    # context-parallel key of any kind -- a run declaring tp=8 would train
    # 8-way DDP while the manifest records tp=8. That is the unwired-knob
    # class, and the house idiom for it is refusal (nvfp4 above), never a
    # silent 1. Each degree is named independently: a guard on the product
    # would pass tp=2, pp=1 combinations it must catch.
    unwired = [name for name in ("tp", "pp", "ep", "cp") if getattr(cfg, name) > 1]
    if unwired:
        _mark(
            Step.REFUSE,
            f"declared {', '.join(f'{name}={getattr(cfg, name)}' for name in unwired)} "
            "cannot be executed: the plane builds a transformers.Trainer "
            "whose kwargs carry no tensor/pipeline/expert/context-parallel "
            "key, so the degree would be recorded but never executed "
            "(finding #375). "
            "Refusing rather than training pure DDP under a parallel label "
            "the run does not have",
        )
        _emit_manifest(
            cfg,
            stage="refused",
            extra={"exit": EXIT_REFUSE, "unwired_degrees": ",".join(unwired)},
        )
        return EXIT_REFUSE

    # Declaring a sharded execution this plane does not have is #375 written on
    # a second axis: None and "ddp" are what transformers.Trainer actually
    # executes here, everything else is REFUSED rather than silently run as
    # plain DDP underneath a manifest that says otherwise.
    if cfg.sharding_strategy not in (None, *SHARDING_STRATEGIES):
        _mark(
            Step.REFUSE,
            f"sharding_strategy={cfg.sharding_strategy!r} is declared, but no "
            f"sharded backend is wired in this plane: the execution path is "
            f"transformers.Trainer, which provides data parallelism only, and "
            f"the only honourable declarations are None and {SHARDING_STRATEGIES}. "
            "FSDP / ZeRO / DeepSpeed were measured, adjudicated and recorded in "
            "gates/ and provenance/ -- and never built. Refusing rather than "
            "running plain DDP while the manifest claims something else "
            "happened (#375's defect class: declaration without execution)",
        )
        _emit_manifest(
            cfg,
            stage="refused",
            extra={"exit": EXIT_REFUSE, "sharding_strategy": cfg.sharding_strategy},
        )
        return EXIT_REFUSE

    if cfg.cpu_optimizer_offload is True:
        _mark(
            Step.REFUSE,
            "cpu_optimizer_offload=True is declared, but this plane has no "
            "optimizer-offload backend: the only routes transformers offers for "
            "it are the DeepSpeed/FSDP integrations, and those executors are "
            "adjudicated and recorded in gates/ and provenance/, never built. "
            "Refusing rather than training with optimizer states resident on "
            "device while the manifest records an offload that never happened "
            "(--cpu-optimizer-offload false or omit the flag to declare the "
            "execution this backend actually performs)",
        )
        _emit_manifest(
            cfg,
            stage="refused",
            extra={"exit": EXIT_REFUSE, "cpu_optimizer_offload": True},
        )
        return EXIT_REFUSE

    # --- 1. Topology (validated on construction) --------------------------
    try:
        declared = Topology(
            dp=cfg.dp,
            tp=cfg.tp,
            pp=cfg.pp,
            ep=cfg.ep,
            cp=cfg.cp,
            nodes=cfg.nodes,
            gpus_per_node=cfg.gpus_per_node,
        )
    except Exception as exc:  # noqa: BLE001 -- a precondition, refused before any GPU
        _mark(Step.REFUSE, f"topology is not constructible (nothing touched): {exc}")
        return EXIT_REFUSE
    _mark(Step.TOPOLOGY, declared.describe())

    # --- 2. ClusterProfile ------------------------------------------------
    try:
        profile = _resolve_profile(cfg)
    except Exception as exc:  # noqa: BLE001 -- missing/refused profile input
        _mark(Step.REFUSE, f"cluster profile refused: {exc}")
        return EXIT_REFUSE
    _mark(
        Step.PROFILE,
        f"{profile.name}: scheduler={profile.scheduler} gpus_per_node={profile.gpus_per_node}",
    )

    # --- 3. Consistency findings, BEFORE a single GPU is touched ----------
    findings: list[Finding] = list(declared.validate_against(profile))
    effective = _effective_topology(cfg)
    if effective is None:
        _mark(
            Step.CONSISTENCY,
            "no torchrun runtime (WORLD_SIZE unset); declared-vs-effective "
            "comparison skipped on the driver",
        )
    elif isinstance(effective, Finding):
        findings.append(effective)
    else:
        findings.extend(declared_vs_effective(declared, effective))
        _mark(
            Step.CONSISTENCY,
            f"WORLD_SIZE={os.environ['WORLD_SIZE']}: declared-vs-effective compared",
        )
    # partition_consistency compares partition SPELLINGS across launcher
    # scripts. Pointing it at the cluster-profile JSON is a category error: a
    # profile declares `partitions` as a field, never as an sbatch line, so the
    # scan finds zero declarations and blocks -- correctly, but for a reason
    # that has nothing to do with the run. Scan a real launcher corpus or
    # declare that none was supplied.
    if cfg.launch_corpus is not None:
        root = Path(cfg.launch_corpus)
        try:
            corpus = {
                str(p): p.read_text(encoding="utf-8", errors="replace")
                for p in sorted(root.rglob("*"))
                if p.is_file() and p.suffix in (".sh", ".sbatch", ".slurm", "")
            }
        except OSError as exc:
            _mark(Step.REFUSE, f"cannot read launch corpus {root}: {exc}")
            return EXIT_REFUSE
        _mark(Step.PARTITION, f"scanning {len(corpus)} launcher file(s) under {root}")
        findings.append(partition_consistency(corpus))
    else:
        _mark(
            Step.PARTITION,
            "UNMEASURED: no launch corpus supplied (--launch-corpus), so partition "
            "spelling was not compared. The driver runs INSIDE an allocation and "
            "owns no sbatch scripts; this is an absent measurement, not a clean one",
        )

    _mark(Step.VALIDATED, render_findings(findings))
    blockers = blocking(findings)
    if blockers:
        _mark(
            Step.BLOCKED,
            f"{len(blockers)}/{len(findings)} finding(s) block; stopping BEFORE "
            "an allocation is burned -- no GPU touched, no torch imported",
        )
        _emit_manifest(cfg, stage="blocked", extra={"blocking": [f.code for f in blockers]})
        return EXIT_RED

    # --- Dry-run: the ENTIRE prologue above, zero GPUs. -------------------
    if cfg.dry_run:
        _emit_manifest(cfg, stage="dry_run")
        _mark(Step.DONE, "dry-run PASS: full validation prologue ran, 0 GPUs touched")
        return EXIT_PASS

    # --- 4. Optional deps. Absence is REFUSE, never a traceback. ----------
    _mark(Step.DEPS, f"importing torch/transformers/datasets (optional extra '{EXTRA}')")
    try:
        import datasets as hf_datasets
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        _mark(
            Step.REFUSE,
            f"missing optional dependency {missing!r}; install with {EXTRA_HINT}",
        )
        return EXIT_REFUSE

    # --- 5. Model + tokenizer + dataset -----------------------------------
    torch.manual_seed(cfg.seed)
    # attn_implementation binds at MODEL CONSTRUCTION, not on TrainingArguments
    # -- no such knob exists there, so it rides from_pretrained. The contract is
    # the same one the TrainingArguments introspection below enforces (#342: a
    # silent drop is worse than an honest refusal), but the INSTRUMENT is not:
    # acceptance is MEASURED after the load, never inferred from a signature.
    #
    # It used to be inferred, and the inference was false in every release.
    # ``AutoModelForCausalLM.from_pretrained`` is a dispatcher whose signature
    # is ``(*model_args, **kwargs)``; it has never named ``attn_implementation``
    # and neither does ``PreTrainedModel.from_pretrained`` on transformers 5.x.
    # A guard reading "absent from the signature => absent from the release"
    # therefore refused EVERY declared value, on a stack that measurably accepts
    # it -- the axis was 100% dead while its test stayed green by asserting the
    # refusal. The oracle below is the one transformers itself publishes:
    # ``config._attn_implementation``, the loader's own record of what it
    # selected. It is strictly stronger than introspection, because it observes
    # the outcome rather than predicting it from the calling convention.
    #
    # Three refusals, all of them measurements: the load rejected the value
    # (ValueError from from_pretrained), the loaded model exposes no reading at
    # all (acceptance unprovable), or the reading disagrees with the declaration
    # (accepted then overridden). When None was declared, NOTHING is passed and
    # the model-config default applies, unclaimed (#342's rule) -- and nothing
    # is verified, because there is no declaration to verify.
    model_kwargs: dict[str, Any] = {}
    if cfg.attn_implementation is not None:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    # Bind the declared precision at construction. The spelling is `dtype`, and
    # that is MEASURED rather than read off a changelog: on transformers 5.13.0
    # both `dtype` and `torch_dtype` produce torch.float32 and the latter warns
    # "`torch_dtype` is deprecated! Use `dtype` instead!", while declaring
    # nothing loads the checkpoint's own torch.bfloat16 -- so the knob is live
    # and the negative control fires. Same rule as attn_implementation: when
    # None is declared, NOTHING is passed and the checkpoint's dtype applies,
    # unclaimed (#342), which keeps every existing run bit-identical.
    if cfg.precision in _PRECISION_TORCH_DTYPES:
        model_kwargs["dtype"] = getattr(torch, _PRECISION_TORCH_DTYPES[cfg.precision])
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)
        except (ValueError, TypeError, ImportError) as exc:
            if cfg.attn_implementation is None and cfg.precision is None:
                raise
            # REFUSE (96), not RED (5). The operator stated a kernel and the
            # load did not complete; no training was attempted, so there is no
            # run to score. Scoring it RED would report a failed run where
            # there was a rejected request -- the same conflation the
            # TrainingArguments arm refuses one layer down.
            #
            # ImportError is in the tuple because that is what an UNAVAILABLE
            # backend raises: transformers 5.16.1 answers
            # attn_implementation="flash_attention_2" on a build without
            # flash-attn with ImportError, and an unsupported NAME with
            # ValueError. Both are the same event to the operator -- "this
            # build cannot give you the kernel you asked for" -- and both must
            # land on 96 rather than one on 96 and one on 5. The exception is
            # quoted rather than paraphrased: the cause is whatever the loader
            # said it was, and this marker does not claim to have diagnosed it.
            _mark(
                Step.REFUSE,
                f"model load did not complete while attn_implementation="
                f"{cfg.attn_implementation!r} and precision={cfg.precision!r} were "
                f"declared, on transformers "
                f"{_tf_version()}: {type(exc).__name__}: {exc}",
            )
            _emit_manifest(
                cfg,
                stage="refused",
                extra={
                    "exit": EXIT_REFUSE,
                    "attn_implementation": cfg.attn_implementation,
                    "precision": cfg.precision,
                },
            )
            return EXIT_REFUSE
        if cfg.attn_implementation is not None:
            selected = getattr(model.config, "_attn_implementation", None)
            if selected != cfg.attn_implementation:
                _mark(
                    Step.REFUSE,
                    f"attn_implementation={cfg.attn_implementation!r} is declared, but "
                    f"the loaded model reports {selected!r}: the value was accepted "
                    "by the call and then dropped or overridden on the way in "
                    "(or this release publishes no _attn_implementation reading, "
                    "in which case acceptance cannot be proved). Refusing rather "
                    "than training under an unverified kernel declaration",
                )
                _emit_manifest(
                    cfg,
                    stage="refused",
                    extra={
                        "exit": EXIT_REFUSE,
                        "attn_implementation": cfg.attn_implementation,
                        "attn_implementation_selected": selected,
                    },
                )
                return EXIT_REFUSE
        if cfg.precision in _PRECISION_TORCH_DTYPES:
            # Observe the outcome, never infer acceptance from the call. The
            # kwarg could be renamed, deprecated into a no-op, or overridden by
            # a config field, and every one of those failures is silent at the
            # call site and loud here. This is the reading that makes the
            # save-gate comparison meaningful: without it, "declared fp32" is a
            # string in a manifest rather than a property of the weights.
            wanted = getattr(torch, _PRECISION_TORCH_DTYPES[cfg.precision])
            observed = getattr(model, "dtype", None)
            if observed != wanted:
                _mark(
                    Step.REFUSE,
                    f"precision={cfg.precision!r} is declared, so the model was "
                    f"loaded with dtype={wanted}, but the loaded model reports "
                    f"{observed!r}: the dtype was accepted by the call and then "
                    "dropped or overridden on the way in (or this release "
                    "publishes no model.dtype reading, in which case acceptance "
                    "cannot be proved). Refusing rather than training under an "
                    "unverified precision declaration -- a run that trains bf16 "
                    "under an fp32 label is the failure this axis exists to stop",
                )
                _emit_manifest(
                    cfg,
                    stage="refused",
                    extra={
                        "exit": EXIT_REFUSE,
                        "precision": cfg.precision,
                        "precision_dtype_observed": str(observed),
                    },
                )
                return EXIT_REFUSE
            # CONSISTENCY, not a new marker: this is a declared-vs-effective
            # reading, the same channel that reports the torchrun comparison.
            _mark(
                Step.CONSISTENCY,
                f"precision={cfg.precision!r} bound at load and verified: the "
                f"loaded model reports dtype={observed}",
            )
        raw = _load_raw_dataset(hf_datasets, cfg.dataset)
        split = "train" if "train" in raw else next(iter(raw))
        columns = raw[split].column_names
        if "text" not in columns:
            _mark(
                Step.REFUSE,
                f"dataset {cfg.dataset!r} split {split!r} has columns {columns}; "
                "the thin path requires a 'text' column",
            )
            return EXIT_REFUSE
        tokenized = raw[split].map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=TOKENIZE_MAX_LENGTH),
            batched=True,
            remove_columns=columns,
        )
    except Exception as exc:  # noqa: BLE001 -- classified into RED vs REFUSE below
        environment = _environment_failure_reason(exc)
        if environment is not None:
            # 96, not 5. Construction never began, so no claim about this plane
            # was measured -- see _ENVIRONMENT_ERRNOS for why the two populations
            # must not share a verdict.
            _mark(
                Step.REFUSE,
                f"model/dataset construction could not be ATTEMPTED: {environment}. "
                "This is a property of the machine, not of the training plane, so "
                "there is no run to score. Point HF_HOME (and HF_DATASETS_CACHE) at "
                f"a filesystem without this fault and re-run. Underlying: {exc!r}",
            )
            return EXIT_REFUSE
        _mark(Step.RED, f"model/dataset construction failed: {exc!r}")
        return EXIT_RED
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _mark(
        Step.DATA,
        f"{len(tokenized)} examples tokenized (split={split}, max_length={TOKENIZE_MAX_LENGTH})",
    )

    # --- Adapters (LoRA), wrapped HERE -- upstream of the declaration -------
    #
    # Wrapping happens before _declare_checkpoint because under an adapter the
    # saved artifact holds ADAPTER tensors, and the declaration is the
    # denominator the save gates measure the artifact against (see 2e in the
    # _declare_checkpoint docstring).
    adapter_notes: dict[str, str] = {}
    if cfg.adapter is not None:
        # __post_init__ validates adapter against ADAPTERS, so "lora" is the
        # only value reachable -- but the branch is explicit, so a future
        # accepted mode cannot silently reuse the LoRA wiring.
        if cfg.adapter == "lora":
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                # REFUSE, never fall through: a run the operator believes is
                # LoRA but is secretly full-FT is a catastrophic silent failure.
                _mark(
                    Step.REFUSE,
                    f"adapter='lora' is declared but the optional dependency "
                    f"'peft' is not installed; install with {EXTRA_HINT}. "
                    "Refusing rather than silently running a full fine-tune",
                )
                return EXIT_REFUSE
            # Unset knobs are OMITTED from the LoraConfig, not defaulted here:
            # peft's own defaults then apply, and this config invents no value
            # the operator did not state.
            lora_config: dict[str, Any] = {"r": cfg.adapter_rank}
            if cfg.adapter_alpha is not None:
                lora_config["lora_alpha"] = cfg.adapter_alpha
            if cfg.adapter_targets is not None:
                lora_config["target_modules"] = list(cfg.adapter_targets)
            if cfg.adapter_dropout is not None:
                lora_config["lora_dropout"] = cfg.adapter_dropout
            try:
                model = get_peft_model(model, LoraConfig(**lora_config))
            except Exception as exc:  # noqa: BLE001 -- construction failure is RED
                # Deliberately NOT run through _environment_failure_reason (#409):
                # get_peft_model() rewrites an already-constructed model in memory
                # and opens no file and no socket, so there is no environment errno
                # for the classifier to find. A branch here would be unfireable by
                # any test, which is a worse defect than the asymmetry it removes.
                _mark(Step.RED, f"peft wrapping of adapter='lora' failed: {exc!r}")
                return EXIT_RED
            # MEASURE what the adapter attached to. LoRA parameters are named
            # <module>.lora_A.<adapter>.weight / <module>.lora_B.<adapter>.weight,
            # so the attached module set is the distinct prefixes. A target
            # pattern matching zero modules leaves it EMPTY while every gate
            # downstream stayed green -- this codebase has shipped exactly that
            # LoRA defect once. Empty is REFUSE (96), and it is a real measured
            # zero, not an abstention: the set was enumerated to get it.
            lora_param_names = [name for name, _ in model.named_parameters() if ".lora_" in name]
            attached_modules = sorted({name.split(".lora_")[0] for name in lora_param_names})
            if not attached_modules:
                _mark(
                    Step.ADAPTER,
                    f"adapter='lora' attached to 0 modules (targets="
                    f"{list(cfg.adapter_targets) if cfg.adapter_targets is not None else None!r}); "
                    "refusing as vacuous -- an adapter that targets nothing "
                    "trains nothing while looking like it trained",
                )
                _emit_manifest(
                    cfg,
                    stage="refused",
                    extra={
                        "exit": EXIT_REFUSE,
                        "adapter": "lora",
                        "attached_modules": 0,
                    },
                )
                return EXIT_REFUSE
            trainable = sum(int(p.numel()) for _, p in model.named_parameters() if p.requires_grad)
            total_params = sum(int(p.numel()) for _, p in model.named_parameters())
            adapter_notes = {
                "adapter.mode": "lora",
                "adapter.rank": str(cfg.adapter_rank),
                "adapter.alpha": str(cfg.adapter_alpha),
                "adapter.targets_declared": (
                    ",".join(cfg.adapter_targets)
                    if cfg.adapter_targets is not None
                    else "(peft defaults)"
                ),
                "adapter.attached_modules": str(len(attached_modules)),
                "adapter.resolved_targets": ",".join(attached_modules),
                "adapter.trainable_params": str(trainable),
                "adapter.total_params": str(total_params),
            }
            _mark(
                Step.ADAPTER,
                f"lora attached to {len(attached_modules)} module(s); "
                f"{trainable}/{total_params} parameters trainable; undeclared "
                "knobs left to peft defaults",
            )
        else:  # pragma: no cover -- __post_init__ refuses every other value
            _mark(Step.REFUSE, f"adapter={cfg.adapter!r} has no wiring in the thin path")
            return EXIT_REFUSE

    # Derive the checkpoint denominator HERE -- from the model in memory, once,
    # before a single tensor has been written. Doing it after a save would read
    # the denominator off the artifact it is supposed to adjudicate, and the file
    # would agree with itself no matter what it had dropped (#70). Doing it once
    # rather than per-save also means every checkpoint in the run is measured
    # against the same declaration, so a mid-run divergence is visible instead of
    # being absorbed by a denominator that moved with it.
    #
    # Named `declared_ckpt`, not `declared`: `declared` is already bound in this
    # function to the declared TOPOLOGY. Two different declarations, one word --
    # #222's shape, caught here by the typechecker rather than by a wrong number
    # in a report.
    try:
        declared_ckpt, decl_notes = _declare_checkpoint(model)
    except Exception as exc:  # noqa: BLE001 -- undeclarable is a STATE, not a crash
        declared_ckpt, decl_notes = None, {"declaration.error": repr(exc)}
        _mark(
            Step.MANIFEST,
            f"could not derive a checkpoint declaration ({exc!r}); the manifest "
            "will carry none and every checkpoint gate will fail closed on UNKNOWN",
        )
    else:
        _mark(Step.MANIFEST, f"declared checkpoint: {decl_notes['declaration.basis']}")
    # Adapter measurements travel to the manifest through the SAME notes
    # channel as the declaration basis: they are measured facts about the run
    # (EffectiveValue source="measured"), alongside the declaration they
    # rescoped under peft.
    manifest_notes = {**decl_notes, **adapter_notes}

    # --- 6. Trainer + save-gate callback -----------------------------------
    #
    # `save_safetensors=True` was a TrainingArguments knob in transformers 4.x
    # and is GONE in 5.x, where safetensors is the only serialization path.
    # Passing it unconditionally is a TypeError on 5.x -- which is exactly how
    # this line was found: the declared extra says `transformers>=4.40`, and the
    # first real execution of train() died at this call on 5.16.1 (#225). But
    # dropping it unconditionally is not the fix either, because on 4.x it is
    # load-bearing: without it a 4.x Trainer can emit pytorch_model.bin, and the
    # save gate downstream reads safetensors.
    #
    # So the knob is bound by introspection rather than by assumption, and --
    # because an accepted keyword is a claim about behaviour, not a proof of it
    # -- the artifact format is ASSERTED after the save in step 8 instead of
    # being trusted here. Introspection alone would carry the same defect one
    # level up: it proves the argument was tolerated, not that it took effect.
    # logging_steps is the one knob whose absence still binds, so the
    # effective cadence is resolved ONCE here: the declared value when the
    # operator set one, the historical run-length-bounded expression when
    # they did not. Computing it at the wiring site and again at the record
    # site would be two expressions that can drift; the telemetry section
    # below records this binding as logging_steps_effective.
    logging_steps_effective = (
        cfg.logging_steps if cfg.logging_steps is not None else max(1, min(10, cfg.max_steps))
    )
    kwargs: dict[str, object] = {
        "output_dir": str(cfg.output_dir),
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.per_device_batch_size,
        "learning_rate": cfg.learning_rate,
        "save_strategy": "steps",
        "save_steps": cfg.save_interval,
        "seed": cfg.seed,
        # Bounded by the run's own length, not a bare 10. A constant cadence
        # means any run SHORTER than the cadence emits no training log at all --
        # no loss is ever reported, for the whole run. That was invisible while
        # nothing read the loss; the objective gate reads it, and a 3-step run
        # was landing on the gate's backstop arm (UNMEASURED) purely because the
        # cadence outran the run. The floor of 1 keeps max_steps=1 observable,
        # and any run of 10 steps or more keeps the previous cadence exactly.
        # A DECLARED logging_steps wins over that fallback -- that is the whole
        # point of the knob -- and the effective value is resolved once above
        # so the telemetry section records the cadence actually bound.
        "logging_steps": logging_steps_effective,
        "report_to": [],
        "ddp_find_unused_parameters": False,
    }
    # The declared precision is wired into the flags EXPLICITLY (1b). fp32 sets
    # both off rather than omitting them: an environment-leaning default behind
    # TrainingArguments must not move the run off its declaration. None adds
    # NOTHING -- no flag, no claim, no coercion -- and the manifest records the
    # absence. nvfp4 never reaches here (refused at START). If a future
    # transformers drops bf16/fp16, the `dropped` refusal below catches it
    # rather than silently training off-declaration.
    if cfg.precision == "bf16":
        kwargs["bf16"] = True
    elif cfg.precision == "fp16":
        kwargs["fp16"] = True
    elif cfg.precision == "fp32":
        kwargs["bf16"] = False
        kwargs["fp16"] = False
    # The remaining declaration axes, each wired ONLY when declared -- the
    # precision rule applied verbatim: None adds NOTHING, no flag, no claim, no
    # coercion, because every one of these has a live TrainingArguments default
    # behind it (AdamW, accumulation 1, max_grad_norm 1.0, no recompute, linear
    # LR with zero warmup) and passing that default undeclared would convert an
    # abstention into the appearance of a statement (#342). Two of the nine are
    # deliberately NOT in this dict: attn_implementation binds at model
    # construction (introspected and possibly refused at its own site above),
    # and sharding_strategy / cpu_optimizer_offload have no TrainingArguments
    # wiring at all -- they are refused at START rather than silently dropped.
    # Every key added here flows through the `accepted` introspection check
    # below, so an older transformers REFUSES on a knob it does not know
    # instead of silently training without it.
    if cfg.optimizer is not None:
        kwargs["optim"] = cfg.optimizer
    if cfg.gradient_accumulation_steps is not None:
        kwargs["gradient_accumulation_steps"] = cfg.gradient_accumulation_steps
    if cfg.max_grad_norm is not None:
        kwargs["max_grad_norm"] = cfg.max_grad_norm
    if cfg.gradient_checkpointing is not None:
        kwargs["gradient_checkpointing"] = cfg.gradient_checkpointing
    if cfg.lr_scheduler_type is not None:
        kwargs["lr_scheduler_type"] = cfg.lr_scheduler_type
    if cfg.warmup_steps is not None:
        kwargs["warmup_steps"] = cfg.warmup_steps
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "save_safetensors" in accepted:
        kwargs["save_safetensors"] = True
    # Every other key above is required for the thin path to mean anything. If a
    # future release drops one, refuse loudly rather than train something that
    # is not what was asked for -- a silently ignored max_steps is a run whose
    # cost is unbounded, and a silently ignored save_steps is a run with no
    # checkpoints for the gate to read.
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        _mark(
            Step.REFUSE,
            f"transformers {_tf_version()} TrainingArguments does not accept "
            f"{dropped}; the thin path cannot honour the requested config. "
            f"Pin a supported version ({EXTRA_HINT})",
        )
        return EXIT_REFUSE
    # Step 4's guarded import is NOT sufficient to establish that the training
    # dependencies are present. transformers imports fine without accelerate and
    # then raises ImportError from TrainingArguments.__post_init__, which is
    # here -- past the refusal path, so it escaped as a traceback and exit 1.
    # Exit 1 is in none of the four declared states (0/5/95/96), and #171 is the
    # same defect one plane up: a code outside the namespace is a verdict the
    # caller cannot interpret. A missing dependency is a REFUSE wherever it is
    # discovered, so the discovery site is wrapped rather than trusted.
    # declared_precision goes in as-is: None stays None and the first-save
    # precision check abstains instead of passing a run that said nothing.
    gate_callback = FoundationScaleSaveGate(declared_precision=cfg.precision)
    # `list[Any]`, not `[gate_callback]` inline. The TYPE_CHECKING block at the
    # top of this module pins FoundationScaleSaveGate's base to `object` so the
    # typecheck does not depend on whether the optional extra is installed --
    # but that only settled the CLASS. The CALL still read TrainingArguments and
    # Trainer from whichever stubs were present, and Trainer.__init__ declares
    # `callbacks: list[TrainerCallback] | None`, so with transformers installed
    # mypy rejected an object-based callback that is correct at runtime: clean
    # in CI, one error locally, identical source. Exactly the divergence that
    # block claims to have removed, surviving one level down because the fix was
    # applied to the declaration and the symptom lives at the use.
    objective_callback = FoundationScaleObjectiveGate(
        objective=cfg.objective,
        learning_rate=cfg.learning_rate,
        per_device_batch_size=cfg.per_device_batch_size,
        max_steps=cfg.max_steps,
    )
    callbacks: list[Any] = [gate_callback, objective_callback]
    # And the third time, on the same axis. `# type: ignore[arg-type]` here was
    # NEEDED with transformers installed (**kwargs is dict[str, Any] against a
    # long typed signature) and UNUSED without it (TrainingArguments resolves to
    # Any under the module's ignore_missing_imports override), so with
    # warn_unused_ignores the identical source was clean locally and one error on
    # all three CI Pythons. A silencer whose own necessity depends on the
    # environment is not a narrower verdict than the error it suppresses -- it is
    # the same defect as #111/#229 written in one comment.
    #
    # Binding the constructor through an explicitly Any-typed local makes the
    # CALL environment-independent, which is what the two notes above were
    # reaching for: neither environment produces an error, so neither needs a
    # silencer. Runtime is untouched -- this is the same object under a second
    # name. Verified in both directions, not one: mypy is clean here with
    # transformers importable, and clean again with transformers forced to Any.
    _TrainingArguments: Any = TrainingArguments
    try:
        args = _TrainingArguments(**kwargs)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=callbacks,
        )
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        _mark(
            Step.REFUSE,
            f"transformers {_tf_version()} needs {missing!r} to build a Trainer, "
            f"and it is absent; install with {EXTRA_HINT}",
        )
        return EXIT_REFUSE
    except (ValueError, TypeError) as exc:
        # A ValueError/TypeError here is a REJECTED DECLARATION, not a training
        # failure: TrainingArguments.__post_init__ is where the newly declarable
        # axes (an optimizer name this release does not know, an unschedulable
        # lr_scheduler_type, a warmup_steps its version rejects) are adjudicated
        # by the authority on its own vocabulary. Uncaught, that escape is a
        # traceback and exit 1 -- a code in none of the four declared states
        # (#171's rule reaches this plane too). Catching it here is the same
        # contract the ImportError arm above already honours: the run cannot
        # honour the requested config, so it REFUSES, naming the rejection.
        # Transformers' message is interpolated verbatim -- it names the knob
        # and its accepted values better than a paraphrase would.
        _mark(
            Step.REFUSE,
            f"transformers {_tf_version()} rejected the declared config at "
            f"Trainer/TrainingArguments construction: {exc}. Refusing (96) "
            f"rather than retrying with a guessed value -- the operator "
            f"declared it, so the operator corrects it",
        )
        _emit_manifest(
            cfg,
            stage="refused",
            extra={"exit": EXIT_REFUSE, "trainer_construction": str(exc)},
        )
        return EXIT_REFUSE
    _mark(
        Step.TRAINER,
        "transformers.Trainer constructed (single-node DDP is automatic under "
        "torchrun); FoundationScaleSaveGate attached",
    )

    # --- 7. Train -----------------------------------------------------------
    #
    # The manifest is written BEFORE the first step, not after the last one.
    # It used to be emitted only at `blocked`, `dry_run` and `done` -- three
    # stages that share the property of being past every save the callback can
    # gate. So `checkpoint.save_complete` looked beside each checkpoint, found
    # nothing, and correctly abstained; `checkpoint.first_save` then failed
    # closed on the abstention and stopped the run; and the manifest that would
    # have satisfied the gate was written afterwards, by the failure path. A
    # producer that runs after its consumer supplies nothing, however correct
    # the bytes it eventually writes.
    #
    # Ordering is therefore part of the contract, not an implementation detail:
    # anything a save gate reads has to exist before a save can happen. The
    # `done` emission below still runs and overwrites this one with the final
    # exit -- that is intended, since the run's outcome is only knowable then.
    _emit_manifest(cfg, stage="train", declared=declared_ckpt, notes=manifest_notes)
    _mark(Step.RUN, "training starts")
    # Peak-memory instrumentation starts HERE, before the first step: the peak
    # counter is reset so the figure read after the last step is the TRAINING
    # peak, not a high-water mark carried over from model construction or
    # tokenization. torch was imported locally in step 4, so this adds no
    # module-scope torch import (#354). Where CUDA is absent there is no
    # counter to reset and nothing is faked: the telemetry entries are
    # recorded UNMEASURED with the reason, never 0 and never absent.
    # Probe for the module before the function: torch.cuda is absent on a
    # CPU-only build and on the fake torch a test double installs, and reaching
    # through it unguarded raised AttributeError, which train() adjudicated RED.
    # An instrument that is not there is UNMEASURED, never a failed run -- and
    # the two reasons are distinct, so the record says which one applied.
    cuda_available, no_peak_reason = _cuda_availability(torch)
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
    try:
        # TrainOutput is RETAINED, not discarded: its .metrics mapping is the
        # trainer's own statement of train_runtime, train_samples_per_second,
        # train_steps_per_second and train_loss, and it is the only measured
        # source the telemetry section has. It used to be dropped on the
        # floor, so those numbers existed transiently on stdout and nowhere
        # else.
        train_output = trainer.train()
    except Exception as exc:  # noqa: BLE001 -- classified into RED vs REFUSE below
        environment = _environment_failure_reason(exc)
        if environment is not None:
            _mark(
                Step.REFUSE,
                f"Trainer.train() could not run to completion: {environment}. This "
                "is a property of the machine, not of the training plane, so the "
                f"run is unscored rather than failed. Underlying: {exc!r}",
            )
            return EXIT_REFUSE
        _mark(Step.RED, f"Trainer.train() raised: {exc!r}")
        return EXIT_RED
    if objective_callback.blocked:
        _mark(
            Step.RED,
            "an objective gate fired at the first observed step and the run was "
            "stopped (should_training_stop=True) -- adjudicating as RED",
        )
        return EXIT_RED
    if gate_callback.blocked:
        _mark(
            Step.RED,
            "a save gate fired during training and the run was stopped "
            "(should_training_stop=True) -- adjudicating as RED",
        )
        return EXIT_RED

    # --- Outcome telemetry, read from the instruments that measured it ------
    #
    # Every entry is a (value, source) pair bound for the manifest's telemetry
    # section. source="measured" means the number came from the trainer's own
    # TrainOutput.metrics or from torch's CUDA peak-memory counters;
    # source="derived" marks the one entry computed from the config (the
    # effective logging cadence); an entry that could not be measured is
    # present with source="unmeasured" and a value STATING the reason --
    # never 0, never absent-meaning-fine. The peak is read after the last
    # training step and before the final save, so it prices training rather
    # than serialization.
    if cuda_available:
        peak_allocated: tuple[Any, str] = (torch.cuda.max_memory_allocated(), "measured")
        peak_reserved: tuple[Any, str] = (torch.cuda.max_memory_reserved(), "measured")
    else:
        peak_allocated = (no_peak_reason, "unmeasured")
        peak_reserved = (no_peak_reason, "unmeasured")
    train_metrics = getattr(train_output, "metrics", None)
    if not isinstance(train_metrics, dict):
        # A TrainOutput without a metrics mapping is not a measured zero; the
        # entries below say which instrument was unread rather than minting
        # numbers the trainer never reported.
        train_metrics = None

    def _metric(key: str) -> tuple[Any, str]:
        if train_metrics is None:
            return (
                "UNMEASURED: Trainer.train() returned no metrics mapping, so "
                f"{key} was never reported by the trainer",
                "unmeasured",
            )
        value = train_metrics.get(key)
        if value is None:
            return (
                f"UNMEASURED: TrainOutput.metrics carries no {key!r} key on "
                f"transformers {_tf_version()}",
                "unmeasured",
            )
        return (value, "measured")

    def _loss_curve() -> tuple[Any, str]:
        """Read the per-step loss series the trainer already logged.

        train_loss is the trainer's aggregate mean over the whole run: one
        scalar. Adjudicators compare CURVES -- did the loss move when a knob
        moved, and do two accumulation settings trace the same path -- and a
        one-point mean cannot answer either, so the series is read from
        trainer.state.log_history at whatever cadence logging_steps bound.
        """
        try:
            state = getattr(trainer, "state", None)
            if state is None:
                return (
                    "UNMEASURED: trainer exposes no state attribute, so the "
                    "per-step loss history was never readable",
                    "unmeasured",
                )
            history = getattr(state, "log_history", None)
            if not isinstance(history, list):
                return (
                    "UNMEASURED: trainer.state exposes no log_history list, so "
                    "the per-step loss history was never readable",
                    "unmeasured",
                )
            by_step: dict[int, float] = {}
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                loss = entry.get("loss")
                step = entry.get("step")
                # bool is an int subclass; a True/False here is a flag, not a
                # measurement, and the codebase excludes it wherever numbers
                # are read. The train-summary row has no "loss" key and is
                # dropped by this same filter, never by position.
                if isinstance(loss, bool) or isinstance(step, bool):
                    continue
                if not isinstance(loss, (int, float)) or not isinstance(step, (int, float)):
                    continue
                if isinstance(step, float) and not step.is_integer():
                    continue
                # Assignment, not append: a resumed run can log a step twice,
                # and the later record is the one the run ended believing.
                by_step[int(step)] = float(loss)
        except Exception as exc:  # noqa: BLE001 -- telemetry must never turn a run RED
            return (
                f"UNMEASURED: reading trainer.state.log_history raised {type(exc).__name__}: {exc}",
                "unmeasured",
            )
        if not by_step:
            return (
                "UNMEASURED: trainer.state.log_history carried no entry with "
                "both a numeric loss and a numeric step; with "
                f"logging_steps_effective={logging_steps_effective} and "
                f"cfg.max_steps={cfg.max_steps} the cadence and run length are "
                "the pair an operator changes to make a loss observable",
                "unmeasured",
            )
        return ([[step, by_step[step]] for step in sorted(by_step)], "measured")

    telemetry: dict[str, tuple[Any, str]] = {
        "train_runtime_s": _metric("train_runtime"),
        "samples_per_second": _metric("train_samples_per_second"),
        "steps_per_second": _metric("train_steps_per_second"),
        "train_loss": _metric("train_loss"),
        # The aggregate above stays: it is the trainer's own statement. It is
        # not a curve -- one mean cannot show divergence or equivalence -- so
        # the per-step series is carried beside it rather than replacing it.
        "train_loss_curve": _loss_curve(),
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "logging_steps_effective": (logging_steps_effective, "derived"),
    }

    # --- 8. Final save ------------------------------------------------------
    final_dir = Path(cfg.output_dir) / "final"
    try:
        trainer.save_model(str(final_dir))
    except Exception as exc:  # noqa: BLE001 -- classified into RED vs REFUSE below
        environment = _environment_failure_reason(exc)
        if environment is not None:
            # Say plainly that the GPU hours were spent: the weights trained and
            # the filesystem refused them. An operator who reads only the verdict
            # must not conclude that nothing ran.
            _mark(
                Step.REFUSE,
                f"the final save could not be written: {environment}. The weights "
                "trained; the filesystem refused them. That is a property of the "
                "machine, not of the training plane, so the run is unscored rather "
                f"than failed. Underlying: {exc!r}",
            )
            return EXIT_REFUSE
        _mark(Step.RED, f"final save failed: {exc!r}")
        return EXIT_RED
    # The format the save gate reads is asserted on the ARTIFACT, not inferred
    # from the TrainingArguments knob bound in step 6. Two different releases
    # reach this line by two different routes -- 4.x because the knob was
    # accepted, 5.x because the behaviour is unconditional -- and neither route
    # is evidence that the bytes on disk are safetensors. A legacy .bin here
    # would sail past a gate that globs *.safetensors and find nothing, which is
    # the vacuous pass this codebase exists to refuse (doctrine 1): zero shards
    # examined is UNMEASURED, and it would be reported as clean.
    shards = sorted(final_dir.glob("*.safetensors"))
    legacy = sorted(final_dir.glob("*.bin"))
    if legacy and not shards:
        _mark(
            Step.RED,
            f"final save wrote {len(legacy)} legacy .bin shard(s) and 0 "
            f"safetensors shard(s) under transformers {_tf_version()}; the save "
            "gate reads safetensors and would examine nothing",
        )
        return EXIT_RED
    if not shards:
        _mark(
            Step.UNMEASURED,
            f"final save produced 0 safetensors shards in {final_dir} "
            f"(contents: {sorted(p.name for p in final_dir.iterdir())}); the "
            "format the gate depends on is absent, so its verdict would be vacuous",
        )
        _emit_manifest(
            cfg,
            stage="done",
            extra={"exit": EXIT_UNMEASURED},
            declared=declared_ckpt,
            notes=manifest_notes,
            telemetry=telemetry,
        )
        return EXIT_UNMEASURED
    _mark(Step.SAVED, f"final checkpoint -> {final_dir} ({len(shards)} safetensors shard(s))")

    # --- 9. Adjudicate. UNMEASURED is not PASS. -----------------------------
    report, err = _run_save_gates(REGISTRY, final_dir)
    if report is None:
        _mark(
            Step.ADJUDICATE,
            f"0/0 gates could run ({err}); collapsing UNMEASURED into PASS is forbidden",
        )
        _emit_manifest(
            cfg,
            stage="done",
            extra={"exit": EXIT_UNMEASURED},
            declared=declared_ckpt,
            notes=manifest_notes,
            telemetry=telemetry,
        )
        return EXIT_UNMEASURED
    _mark(Step.ADJUDICATE, report.render())
    if report.is_vacuous:
        _mark(Step.ADJUDICATE, "0 gates executed: UNMEASURED")
        _emit_manifest(
            cfg,
            stage="done",
            extra={"exit": EXIT_UNMEASURED},
            declared=declared_ckpt,
            notes=manifest_notes,
            telemetry=telemetry,
        )
        return EXIT_UNMEASURED
    rc = EXIT_RED if report.blocking else EXIT_PASS
    done = "PASS" if rc == EXIT_PASS else "RED: blocking save-gate verdict on the final checkpoint"
    # The objective gate's backstop arm is folded in HERE rather than returned
    # at train end, because the two verdicts are about different artifacts and
    # returning early would have thrown away a trained model to report on a
    # missing log line. It only moves a PASS: a blocking save-gate verdict on
    # the final checkpoint is a measurement, and a measurement outranks an
    # abstention (doctrine 5 -- UNMEASURED is not PASS, and it is not RED
    # either).
    if rc == EXIT_PASS and objective_callback.unmeasured:
        rc = EXIT_UNMEASURED
        done = (
            "UNMEASURED: the run trained and saved, but no training log ever "
            "carried a loss, so the objective gates had nothing to read"
        )
    # Declared precision and the OBSERVED dtype histogram are SEPARATE keys
    # (1d): the whole point is that a reader can see when they disagreed,
    # which requires neither to be collapsed into the other. The histogram is
    # measured at the FIRST save by the save-gate callback and may be absent
    # (no save ever happened) -- recorded as None then, never as {}.
    agreement = gate_callback.precision_agreement
    done_extra: dict[str, Any] = {
        "exit": rc,
        "precision_declared": cfg.precision,
        "precision_observed_dtype_histogram": (
            json.dumps(agreement.observed, sort_keys=True)
            if agreement is not None and agreement.observed is not None
            else None
        ),
        "precision_verdict": agreement.status if agreement is not None else None,
    }
    _emit_manifest(
        cfg,
        stage="done",
        extra=done_extra,
        declared=declared_ckpt,
        notes=manifest_notes,
        telemetry=telemetry,
    )
    _mark(Step.DONE, done)
    return rc
