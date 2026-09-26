
"""Measured probe of the installed FoundationScale capabilities."""
from __future__ import annotations

import ast
import importlib.util
import os
import pkgutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FSCapabilities:
    """Capabilities measured from the installed FS, never assumed."""

    available: bool
    fs_version: str | None
    train_entry: str = "foundationscale-train"
    train_flags: frozenset[str] = frozenset()
    train_objectives: tuple[str, ...] = ()
    sharding_strategies: tuple[str, ...] = ()
    executed_axes: tuple[str, ...] = ()
    refused_axes: tuple[str, ...] = ()
    axes_measured: bool = False
    rl_algorithms: tuple[str, ...] = ()
    # Registered is not runnable: RLTrainer refuses most of the registry
    # (measured on 77bfa65: only dr_grpo, gspo, dapo run). name -> None when it
    # runs, else FS's own refusal text. Empty when unmeasured.
    rl_runnable: dict[str, str | None] = field(default_factory=dict)
    families: dict[str, tuple[str, ...]] = field(default_factory=dict)
    backends: tuple[str, ...] = ()
    # flag -> allowed values (argparse `choices`); a value outside them is refused by FS
    train_flag_choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Reward kinds FS RL can verify, measured behaviourally (77bfa65: mcq_letter only)
    rl_reward_kinds: tuple[str, ...] = ()
    # Whether RLTrainer persists the trained policy (77bfa65: False); None = unmeasured
    rl_saves_checkpoint: bool | None = None
    notes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["train_flags"] = sorted(self.train_flags)
        data["families"] = {k: list(v) for k, v in self.families.items()}
        data["train_flag_choices"] = {k: list(v) for k, v in self.train_flag_choices.items()}
        for key in ("train_objectives", "sharding_strategies", "executed_axes", "refused_axes", "rl_algorithms",
                    "rl_reward_kinds", "backends", "notes", "errors"):
            data[key] = list(getattr(self, key))
        return data

    def check(
        self,
        stage: str,
        *,
        algorithm: str | None = None,
        backend: str = "fsdp",
        tp: int = 1,
        pp: int = 1,
        ep: int = 1,
        cp: int = 1,
        multi_gpu_rl: bool = False,
        require_checkpoint: bool = True,
        answer_kind: str | None = None,
    ) -> str | None:
        """Return None if executable, else a 'missing: ...' reason naming the gap.

        For rl/preference, ``require_checkpoint`` asks whether the stage can hand
        weights to the next one (the default: a stage that trains and keeps
        nothing has not completed). Pass False only for measurement-only runs.
        ``answer_kind`` names the gold the data carries (e.g. "free_form")."""
        if not self.available:
            return "missing: importable foundationscale"
        if stage not in _KNOWN_STAGES:
            return f"missing: unknown stage {stage!r} (known: {', '.join(_KNOWN_STAGES)})"
        objectives = {o.lower() for o in self.train_objectives}
        if stage in {"pretrain", "cpt", "sft"} and "sft" not in objectives:
            opts = ", ".join(self.train_objectives) or "<none>"
            return f"missing: sft objective support for stage {stage} (installed FS objectives: {opts})"
        if stage in {"preference", "rl"}:
            if not algorithm:
                return f"missing: algorithm for stage {stage}"
            if algorithm not in self.rl_algorithms:
                algs = ", ".join(self.rl_algorithms) or "<none unmeasured>"
                return f"missing: RL algorithm {algorithm} (installed FS algorithms: {algs})"
            if algorithm not in self.rl_runnable:
                return f"missing: runnability of {algorithm} unmeasured by probe (RLTrainer objective check)"
            reason = self.rl_runnable[algorithm]
            if reason is not None:
                runs = ", ".join(sorted(a for a, r in self.rl_runnable.items() if r is None)) or "<none>"
                return f"missing: {algorithm} is registered but FS RLTrainer refuses it ({reason[:160]}); runnable today: {runs}"
            # Runnability first: "multi-GPU" is the wrong refusal for an
            # algorithm that cannot run on any number of GPUs.
            if answer_kind is not None and answer_kind not in self.rl_reward_kinds:
                kinds = ", ".join(self.rl_reward_kinds) or "<unmeasured>"
                return f"missing: FS RL rewards only {kinds}; data answers are {answer_kind}"
            if multi_gpu_rl:
                return "missing: multi-GPU RL (FS RLTrainer is single-device)"
            if require_checkpoint and self.rl_saves_checkpoint is not True:
                why = "unmeasured" if self.rl_saves_checkpoint is None else "does not persist the trained policy (no checkpoint)"
                return f"missing: FS RLTrainer {why}"
            return None  # the RL driver is single-device: backend/axes do not apply
        if backend not in self.backends:
            backs = ", ".join(self.backends) or "<none>"
            return f"missing: {backend} backend (installed FS has: {backs})"
        axes = {"tp": tp, "pp": pp, "ep": ep, "cp": cp}
        refused = set(self.refused_axes)
        for axis, value in axes.items():
            if int(value) > 1 and axis in refused:
                return f"missing: {axis}>1 is REFUSED by installed FS"
            if int(value) > 1 and not self.axes_measured:
                return f"missing: {axis} unmeasured; run probe(deep=True)"
        return None


def _add(items: list[str], seen: set[str], value: str) -> None:
    if value not in seen:
        seen.add(value)
        items.append(value)


def _fs_version(module: Any) -> str | None:
    try:
        import importlib.metadata

        return str(importlib.metadata.version("foundationscale"))
    except Exception:
        try:
            return str(getattr(module, "__version__")) if hasattr(module, "__version__") else None
        except Exception:
            return None


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(str(x) for x in value if isinstance(x, str))


def _probe_sharding(errors: list[str]) -> tuple[str, ...]:
    try:
        spec = importlib.util.find_spec("foundationscale.train.loop")
        if spec is None or spec.origin is None:
            errors.append("cannot locate foundationscale.train.loop for AST probe")
            return ()
        source = Path(spec.origin).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=spec.origin)
        for node in tree.body:
            target_name: str | None = None
            value: ast.AST | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_name = node.target.id
                value = node.value
            if target_name != "SHARDING_STRATEGIES" or not isinstance(value, ast.Tuple):
                continue
            out: list[str] = []
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.append(elt.value)
            return tuple(out)
        errors.append("SHARDING_STRATEGIES tuple not found in foundationscale.train.loop")
        return ()
    except Exception as exc:
        errors.append(f"sharding probe failed: {type(exc).__name__}: {exc}")
        return ()


def _probe_backends(sharding: tuple[str, ...], fs_module: Any) -> tuple[str, ...]:
    mapping = {"ddp": "ddp", "fsdp": "fsdp"}
    backends: list[str] = []
    seen: set[str] = set()
    for strategy in sharding:
        backend = mapping.get(strategy.lower())
        if backend:
            _add(backends, seen, backend)
    try:
        fs_path = getattr(fs_module, "__path__", None)
        if fs_path is not None:
            for info in pkgutil.walk_packages(fs_path, prefix="foundationscale.", onerror=lambda name: None):
                if "megatron" in info.name:
                    _add(backends, seen, "megatron")
                    break
    except Exception:
        pass
    return tuple(backends)


_KNOWN_STAGES = ("pretrain", "cpt", "sft", "preference", "rl")


def _probe_flag_choices(errors: list[str]) -> dict[str, tuple[str, ...]]:
    try:
        from foundationscale.train import cli  # type: ignore

        choices = {
            opt: tuple(str(c) for c in action.choices)
            for action in getattr(cli.build_parser(), "_actions", [])
            if getattr(action, "choices", None)
            for opt in action.option_strings if str(opt).startswith("--")
        }
        # FS forwards these to transformers unvalidated; the installed transformers
        # defines what is legal (measured: "adamw" was refused by 5.13 at
        # TrainingArguments construction, after GPUs were allocated).
        try:
            from transformers.trainer_utils import SchedulerType  # type: ignore
            from transformers.training_args import OptimizerNames  # type: ignore

            choices.setdefault("--optimizer", tuple(o.value for o in OptimizerNames))
            choices.setdefault("--lr-scheduler-type", tuple(t.value for t in SchedulerType))
        except Exception as exc:  # noqa: BLE001 - recorded; values stay unvalidated
            errors.append(f"transformers optimizer/scheduler vocabulary unmeasured: {type(exc).__name__}: {exc}")
        return choices
    except Exception as exc:  # noqa: BLE001
        errors.append(f"flag choices probe failed: {type(exc).__name__}: {exc}")
        return {}


def _probe_rl_rewards(errors: list[str]) -> tuple[tuple[str, ...], bool | None]:
    """Behavioural probe: hand FS's own gold reader a letter and free-form golds;
    read RLTrainer's source for any checkpoint write. No model, no GPU."""
    kinds: list[str] = []
    try:
        from foundationscale.rl import corpus  # type: ignore

        if corpus._declared_gold({"answer": "B"}, "probe", 0, "answer") == "B":
            kinds.append("mcq_letter")
        if any(corpus._declared_gold({"answer": v}, "probe", 0, "answer") is not None for v in ("42", "x = 3/4")):
            kinds.append("free_form")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rl reward-kind probe failed: {type(exc).__name__}: {exc}")
    saves: bool | None = None
    try:
        import inspect

        from foundationscale.rl.trainer import RLTrainer  # type: ignore

        src = inspect.getsource(RLTrainer)
        saves = "save_pretrained" in src or ".save(" in src
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rl checkpoint probe failed: {type(exc).__name__}: {exc}")
    return tuple(kinds), saves


def _probe_cli(notes: list[str], errors: list[str]) -> tuple[frozenset[str], tuple[str, ...]]:
    try:
        from foundationscale.train import cli  # type: ignore

        parser = cli.build_parser()
        actions = getattr(parser, "_actions", [])
        flags = frozenset(opt for action in actions for opt in getattr(action, "option_strings", []) if str(opt).startswith("--"))
        objective_action = next((a for a in actions if "--objective" in getattr(a, "option_strings", [])), None)
        objectives: tuple[str, ...] = ()
        if objective_action is not None:
            default = getattr(objective_action, "default", None)
            help_text = str(getattr(objective_action, "help", "") or "").lower()
            if default is not None:
                objectives = (str(default),)
            if "the only objective" not in help_text:
                notes.append("objective vocabulary unmeasured")
        else:
            errors.append("--objective action not found in train CLI parser")
        return flags, objectives
    except Exception as exc:
        errors.append(f"train CLI probe failed: {type(exc).__name__}: {exc}")
        return frozenset(), ()


# The probe is only a measurement if the same command line WITHOUT the axis
# passes; otherwise every axis "refuses" for an unrelated reason (a missing
# required flag did exactly that once). A refusal also has to name the axis.
_PROBE_BASE = (
    "-m", "foundationscale.train.cli", "--model", "probe/none", "--dataset", "probe.jsonl",
    "--dry-run", "--nodes", "1", "--gpus-per-node", "2",
    "--sharding-strategy", "fsdp", "--profile-name", "local-single-node",
)


def _dry_run(executable: str, extra: tuple[str, ...]) -> tuple[int | None, str]:
    """Run one FS dry-run; return (rc, last [fs:train:refuse] line or error)."""
    try:
        with tempfile.TemporaryDirectory(prefix="fskills-axis-") as tmpdir:
            completed = subprocess.run(
                [executable, *_PROBE_BASE, "--output-dir", tmpdir, *extra],
                env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=180, check=False,
            )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return None, f"{type(exc).__name__}: {exc}"
    refusals = [ln for ln in completed.stdout.splitlines() if ln.startswith("[fs:train:refuse]")]
    return completed.returncode, (refusals[-1] if refusals else "")


def _probe_axes(deep: bool, python: str | None, notes: list[str]) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    if not deep:
        return (), (), False
    from concurrent.futures import ThreadPoolExecutor

    executable = python or sys.executable
    runs: dict[str, tuple[str, ...]] = {"control": ("--dp", "2")}
    runs.update({axis: ("--dp", "1", f"--{axis}", "2") for axis in ("tp", "pp", "ep", "cp")})
    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        results = dict(zip(runs, pool.map(lambda extra: _dry_run(executable, extra), runs.values())))
    control_rc, control_msg = results.pop("control")
    if control_rc != 0:
        notes.append(f"axis probe control failed (rc={control_rc}: {control_msg}); axes unmeasured")
        return (), (), False
    executed: list[str] = []
    refused: list[str] = []
    for axis, (rc, msg) in results.items():
        if rc == 96 and f"{axis}=" in msg:
            refused.append(axis)
            notes.append(f"axis {axis} refused by FS: {msg.split(']', 1)[-1].strip()[:200]}")
        elif rc == 0:
            executed.append(axis)
        else:
            notes.append(f"axis {axis} unmeasured (rc={rc}: {msg[:200]})")
    measured = len(executed) + len(refused) == len(results)
    return tuple(executed), tuple(refused), measured


def _probe_rl_runnable(names: tuple[str, ...], errors: list[str]) -> dict[str, str | None]:
    """Ask FS's own objective resolver (the check RLTrainer runs before loading
    a model) whether each registered algorithm is runnable. No model, no GPU."""
    import types

    try:
        from foundationscale.rl.trainer import RLTrainer, TrainerRefusal
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rl runnability unmeasured: {type(exc).__name__}: {exc}")
        return {}
    resolve = getattr(RLTrainer, "_resolve_objective", None)
    if resolve is None:
        errors.append("rl runnability unmeasured: RLTrainer._resolve_objective not found")
        return {}
    out: dict[str, str | None] = {}
    for name in names:
        try:
            resolve(types.SimpleNamespace(config=types.SimpleNamespace(algorithm=name)))
            out[name] = None
        except TrainerRefusal as exc:
            out[name] = str(exc)
        except Exception as exc:  # noqa: BLE001 - not a refusal: leave unmeasured
            errors.append(f"rl runnability of {name} unmeasured: {type(exc).__name__}: {exc}")
    return out


def probe(deep: bool = False, python: str | None = None) -> FSCapabilities:
    """Probe installed FoundationScale. Failures are recorded, never fatal."""
    notes: list[str] = []
    errors: list[str] = []
    try:
        import foundationscale as fs_module  # type: ignore
    except Exception as exc:
        return FSCapabilities(available=False, fs_version=None, errors=(f"import foundationscale failed: {type(exc).__name__}: {exc}",),)

    flags, objectives = _probe_cli(notes, errors)
    sharding = _probe_sharding(errors)
    backends = _probe_backends(sharding, fs_module)
    executed_axes, refused_axes, axes_measured = _probe_axes(deep, python, notes)

    rl_algorithms: tuple[str, ...] = ()
    rl_runnable: dict[str, str | None] = {}
    rl_reward_kinds: tuple[str, ...] = ()
    rl_saves_checkpoint: bool | None = None
    try:
        from foundationscale.rl.registry import available_algorithm_names  # type: ignore

        rl_algorithms = _str_tuple(tuple(available_algorithm_names()))
        rl_runnable = _probe_rl_runnable(rl_algorithms, errors)
        rl_reward_kinds, rl_saves_checkpoint = _probe_rl_rewards(errors)
    except Exception as exc:
        errors.append(f"RL registry probe failed: {type(exc).__name__}: {exc}")

    families: dict[str, tuple[str, ...]] = {}
    try:
        from foundationscale.families.registry import REGISTRY as FAMILY_REGISTRY  # type: ignore

        values = FAMILY_REGISTRY.values() if hasattr(FAMILY_REGISTRY, "values") else FAMILY_REGISTRY
        for spec in values:
            name = getattr(spec, "name", None)
            model_types = getattr(spec, "model_types", None)
            if name is not None and model_types is not None:
                families[str(name)] = tuple(str(x) for x in model_types)
    except Exception as exc:
        errors.append(f"families registry probe failed: {type(exc).__name__}: {exc}")

    return FSCapabilities(
        available=True,
        fs_version=_fs_version(fs_module),
        train_flags=frozenset(sorted(flags)),
        train_objectives=objectives,
        sharding_strategies=tuple(sharding),
        executed_axes=executed_axes,
        refused_axes=refused_axes,
        axes_measured=axes_measured,
        rl_algorithms=tuple(rl_algorithms),
        rl_runnable=rl_runnable,
        families=families,
        backends=tuple(backends),
        train_flag_choices=_probe_flag_choices(errors),
        rl_reward_kinds=rl_reward_kinds,
        rl_saves_checkpoint=rl_saves_checkpoint,
        notes=tuple(notes),
        errors=tuple(errors),
    )


KNOWN_BASELINE: dict[str, Any] = {
    "fs_commit": "77bfa65",
    "objectives": ("sft",),
    "sharding": ("ddp", "fsdp"),
    "refused_axes": ("pp", "ep"),
    "executed_axes": ("tp", "cp"),
    "rl_algorithm_count": 18,
    "rl_reward_kinds": ("mcq_letter",),
    "rl_saves_checkpoint": False,
    "note": "drift-test baseline only; never use for decisions",
}
