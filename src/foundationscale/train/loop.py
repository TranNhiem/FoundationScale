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

import inspect
import json
import os
import sys
from collections.abc import Callable
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
    RUN = "fs:train:run"
    SAVED = "fs:train:saved"
    SAVE_GATE = "fs:train:save_gate"
    MANIFEST = "fs:train:manifest"
    ADJUDICATE = "fs:train:adjudicate"
    DONE = "fs:train:done"
    RED = "fs:train:red"
    # UNMEASURED is a marker, not the absence of one (doctrine 5). Before this
    # existed, the two UNMEASURED returns in train() borrowed ADJUDICATE, so a
    # log reader could see the exit code 95 but not which step abstained --
    # and a step that abstains silently is indistinguishable from one that ran.
    UNMEASURED = "fs:train:unmeasured"


MARKERS: tuple[str, ...] = (
    Step.START,
    Step.TOPOLOGY,
    Step.PROFILE,
    Step.CONSISTENCY,
    Step.VALIDATED,
    Step.BLOCKED,
    Step.REFUSE,
    Step.DEPS,
    Step.DATA,
    Step.TRAINER,
    Step.RUN,
    Step.SAVED,
    Step.SAVE_GATE,
    Step.MANIFEST,
    Step.ADJUDICATE,
    Step.DONE,
    Step.RED,
    Step.UNMEASURED,
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


def _effective_topology(cfg: TrainConfig) -> Topology | Finding | None:
    """The topology the runtime actually built, derived from torchrun env.

    Returns ``None`` on the driver process (no WORLD_SIZE), a :class:`Finding`
    when the runtime evidence cannot form a topology, else the effective one.
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
    mpw = cfg.tp * cfg.pp * cfg.ep * cfg.cp
    gpn = cfg.gpus_per_node
    try:
        return Topology(
            dp=max(world // mpw, 1),
            tp=cfg.tp,
            pp=cfg.pp,
            ep=cfg.ep,
            cp=cfg.cp,
            nodes=-(-world // gpn),
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


class FoundationScaleSaveGate(_CallbackBase):
    """``TrainerCallback`` wiring the registered checkpoint gates into ``on_save``.

    On every save it runs the gate sweep for the lifecycle event over the
    just-written checkpoint directory. A blocking verdict BOTH sets
    ``control.should_training_stop = True`` AND is recorded on the instance
    (``.blocked`` / ``.reports`` / ``.records``). A gate that fires and lets
    the run continue is a check that cannot fail; this callback fails closed.

    Importable without transformers/torch: with the extra absent the base
    degrades to ``object`` and the class is driven directly in tests.
    """

    def __init__(
        self,
        registry: GateRegistry | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.registry = registry if registry is not None else REGISTRY
        self.context_builder = context_builder or _default_context_builder
        self.reports: list[GateReport] = []
        self.records: list[dict[str, Any]] = []
        self.blocked = False
        self._saves = 0

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        step = getattr(state, "global_step", self._saves)
        event = Lifecycle.FIRST_SAVE if self._saves == 0 else Lifecycle.SAVE
        self._saves += 1
        ckpt_dir = Path(getattr(args, "output_dir", ".")) / f"checkpoint-{step}"
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
        },
        "extra": extra or {},
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

    Returns the declaration and a dict of audit notes for the manifest's config
    block, so the basis of every number here survives into the artifact.
    """
    from foundationscale.gates.checkpoint_gates import matches_expert_family, mentions_expert
    from foundationscale.provenance.manifest import DeclaredCheckpoint

    state = model.state_dict()
    names = set(state)
    tied = _tied_aliases(model, names)
    declared = names - tied
    notes = {
        "declaration.source": "model.state_dict() in memory, before the first save",
        "declaration.state_dict_keys": str(len(names)),
        "declaration.tied_excluded": ",".join(sorted(tied)) or "(none)",
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


def _build_run_manifest(
    cfg: TrainConfig,
    *,
    stage: str,
    extra: dict[str, Any] | None,
    declared: Any = None,
    notes: dict[str, str] | None = None,
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

    for key, value in detail["config"].items():
        if isinstance(value, dict):
            # Flatten rather than str() a dict. `topology.tp = 2` is greppable
            # and diffable against another run's manifest; "{'dp': 1, 'tp': 2}"
            # is one opaque string whose equality depends on key insertion order.
            for sub, subvalue in value.items():
                _put(f"{key}.{sub}", subvalue, "cli")
        else:
            _put(key, value, "cli")
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
    )


def _emit_manifest(
    cfg: TrainConfig,
    *,
    stage: str,
    extra: dict[str, Any] | None = None,
    declared: Any = None,
    notes: dict[str, str] | None = None,
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
    manifest = _build_run_manifest(cfg, stage=stage, extra=extra, declared=declared, notes=notes)
    if manifest is None:
        # Degrade loudly. The previous implementation probed provenance for four
        # writer names it does not export and fell through here silently on
        # every single run, which read like integration and was none.
        path.write_text(
            json.dumps(
                _manifest_payload(cfg, stage=stage, extra=extra),
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
    """Run the thin path and adjudicate it.

    Returns 0/5/95/96 per the exit-code contract. Never raises for an
    expected condition; unexpected trainer exceptions adjudicate as RED.
    """
    _mark(
        Step.START,
        f"model={cfg.model} dataset={cfg.dataset} "
        f"output_dir={cfg.output_dir} dry_run={cfg.dry_run}",
    )

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
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        model = AutoModelForCausalLM.from_pretrained(cfg.model)
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
    except Exception as exc:  # noqa: BLE001 -- download/construction failure is RED
        _mark(Step.RED, f"model/dataset construction failed: {exc!r}")
        return EXIT_RED
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _mark(
        Step.DATA,
        f"{len(tokenized)} examples tokenized (split={split}, max_length={TOKENIZE_MAX_LENGTH})",
    )

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
    kwargs: dict[str, object] = {
        "output_dir": str(cfg.output_dir),
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.per_device_batch_size,
        "learning_rate": cfg.learning_rate,
        "save_strategy": "steps",
        "save_steps": cfg.save_interval,
        "seed": cfg.seed,
        "logging_steps": 10,
        "report_to": [],
        "ddp_find_unused_parameters": False,
    }
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
    gate_callback = FoundationScaleSaveGate()
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
    callbacks: list[Any] = [gate_callback]
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
    _emit_manifest(cfg, stage="train", declared=declared_ckpt, notes=decl_notes)
    _mark(Step.RUN, "training starts")
    try:
        trainer.train()
    except Exception as exc:  # noqa: BLE001
        _mark(Step.RED, f"Trainer.train() raised: {exc!r}")
        return EXIT_RED
    if gate_callback.blocked:
        _mark(
            Step.RED,
            "a save gate fired during training and the run was stopped "
            "(should_training_stop=True) -- adjudicating as RED",
        )
        return EXIT_RED

    # --- 8. Final save ------------------------------------------------------
    final_dir = Path(cfg.output_dir) / "final"
    try:
        trainer.save_model(str(final_dir))
    except Exception as exc:  # noqa: BLE001
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
            notes=decl_notes,
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
            notes=decl_notes,
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
            notes=decl_notes,
        )
        return EXIT_UNMEASURED
    rc = EXIT_RED if report.blocking else EXIT_PASS
    _emit_manifest(cfg, stage="done", extra={"exit": rc}, declared=declared_ckpt, notes=decl_notes)
    _mark(
        Step.DONE,
        "PASS" if rc == EXIT_PASS else "RED: blocking save-gate verdict on the final checkpoint",
    )
    return rc
