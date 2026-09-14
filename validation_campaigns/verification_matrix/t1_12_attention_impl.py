#!/usr/bin/env python3
"""T1-12: eager / sdpa / flash_attention_2 agree on logits and differ in step time.

Three arms, ONE axis: ``--attn-implementation`` is the only flag that changes
between arms, and that is asserted against the composed argv of all three arms
(``_assert_single_axis``), not merely intended. The flag itself was checked
against the shipped ``foundationscale-train`` CLI -- it exists, and
``_trainer_supports_axis`` re-checks it at run time, because a campaign that
invented the flag would compose argv the trainer rejects. Everything else is
fixed by SPEC: model ``Qwen/Qwen2.5-1.5B``, dataset ``fancyzhx/ag_news`` (both
already in the tray's HF cache -- a cache miss is UNMEASURED, never a
download), one seed for all three arms (an arm that differs by seed measures
nothing) and ``--precision bf16`` everywhere.

The matrix names this row's refuting observation and it is the verdict,
encoded verbatim as ``CONTROL_RULE``: identical step times across impls = the
impl never changed -> RED. That is the silent-fallback shape: a trainer that
accepts ``--attn-implementation`` and then builds the same attention module
anyway produces three runs, three agreeing loss curves and ONE step time, and
"the arm ran" would still look like success. Agreement gets a tolerance (bf16
kernels reduce in different orders, so equal computations wiggle); identity
gets a band (timing is never bitwise, so "identical" means inside 5%).

``flash_attention_2`` does not exist on aarch64 in this container. That arm is
probed with a real ``import flash_attn`` on the tray interpreter BEFORE any
GPU work; when the import fails the arm is reported 95 UNMEASURED with the
ImportError text as the reason -- present in its record and in the
denominator of 3, never RED and never silently dropped.

Measurements come from the RunManifest ``telemetry`` section (API.md), never
from stdout: ``train_runtime_s``, ``steps_per_second``,
``peak_memory_allocated_bytes``, ``total_flos`` and the per-step loss series.
The series is spelled ``train_loss_curve``, a list of ``[step, loss]`` pairs;
a real run's labels start at 1, because the trainer logs AFTER the first
optimizer update, so step 0 never appears -- the step-0 guard in the
adjudicator must hold when step 0 is absent, never require it. An entry whose
source is ``unmeasured`` is a statement -- its value is the reason -- and a
row whose deciding metric is unmeasured adjudicates 95 with that reason
carried verbatim into ``<row>_<arm>.json``.

  --self-test   exercises the adjudicator against synthetic arm records,
                including sets the control rule REFUTES (asserts 5). No GPU,
                no network, no model download, under one second.

Exit codes follow the four-state contract: 0 claim upheld, 5 refuted, 95 ran
but could not measure the deciding quantity, 96 refused to run at all --
including argparse usage errors, whose stdlib exit 2 sits outside the
contract (#387), so ``error()`` is overridden to refuse instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

from t1_interpreter_floor import (
    DEFAULT_ROUNDS,
    capability_floor_reasons,
    classify_boundary_exception,
    floor_controls,
    gpu_reachability_controls,
    gpu_reachability_reason,
    interleaved_round_plan,
    python_floor_reason,
    reachable_gpu_count,
)

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

ROW = "t1_12"
ROW_ID = "T1-12"
MODEL = "Qwen/Qwen2.5-1.5B"
DATASET = "fancyzhx/ag_news"
PRECISION = "bf16"
IMPLS = ("eager", "sdpa", "flash_attention_2")
# Not an arm, and deliberately not a member of IMPLS: the #413 warmup exists
# to be DISCARDED, and a name that cannot be mistaken for an arm is how the
# evidence tree says so to a reader who only has the directory.
_WARMUP_ARM = "warmup_discarded"
SCALAR_KEYS = ("train_runtime_s", "steps_per_second", "peak_memory_allocated_bytes", "total_flos")

CLAIM = "eager / sdpa / flash_attention_2 agree on logits, differ in step time"

# The matrix's refuting observation for this row, verbatim. It is the verdict,
# not a comment: the adjudicator below applies it.
CONTROL_RULE = "identical step times across impls = the impl never changed -> RED"

# bf16 attention kernels reduce in different orders, so two impls computing the
# SAME function still wiggle around each other; agreement is judged within
# atol + rtol * magnitude, not bitwise.
LOSS_ATOL = 1e-3
LOSS_RTOL = 2e-2
# Timing is never bitwise identical, so "identical step times" is
# operationalised as a band: a spread under 5% across three attention impls on
# a 1.5B model means the knob never reached the model.
STEP_TIME_IDENTICAL_RTOL = 0.05

# #421, stated where the row can be read rather than only in a ledger. The shared
# helper's paired reducer, ``reduce_paired_rounds``, is DELIBERATELY not imported
# here: it reduces a SIGNED two-arm claim (baseline vs treatment, "lower" or
# "higher"), and this row's claim is three-arm and unsigned -- the impls must
# AGREE on logits and DIFFER in step time, with no arm declared the faster one.
# Handing it a direction would fabricate the very assumption the row exists to
# avoid, so the paired statistic is an honest UNMEASURED for T1-12 and the
# cross-round rule below is unanimity instead.
PAIRED_STATISTIC_ABSTENTION = (
    "PAIRED STATISTIC: UNMEASURED for T1-12. reduce_paired_rounds reduces a "
    "SIGNED two-arm claim (baseline vs treatment, 'lower' or 'higher'); this "
    "row's claim is three-arm and unsigned -- the impls must agree on logits and "
    "differ in step time, with no arm declared faster (#421). Inventing a "
    "direction would fabricate the assumption the row exists to avoid, so the "
    "cross-round rule here is unanimity over whole-round verdicts, not a paired "
    "delta. This is an abstention on ONE axis, not on the row."
)

ARM_TIMEOUT_S = 3600
FLASH_PROBE_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Import bootstrap
# ---------------------------------------------------------------------------


def _repo_src_root(start: Path) -> Path | None:
    """Walk upward from ``start`` for the ``src/`` of a src-layout checkout, or None.

    The marker is ``src/foundationscale/__init__.py`` -- the package's own module file,
    not a directory that merely happens to be called ``src`` -- so an unrelated tree with
    the same directory name does not answer for this repository.
    """
    for parent in [start, *start.parents]:
        candidate = parent / "src"
        if (candidate / "foundationscale" / "__init__.py").is_file():
            return candidate
    return None


def _ensure_package_importable() -> str | None:
    """Make ``foundationscale`` importable; return the reason it is not, or None.

    The trainer is invoked as ``python -m foundationscale.train.cli`` in a child
    process, so the package must resolve under the interpreter that is running
    THIS file. An installed distribution cannot be assumed, so this falls back
    to the checkout's own ``src/`` (the child gets it via PYTHONPATH, see
    ``_child_env``). An environment where NEITHER works is UNMEASURED (95),
    never RED (5): a measurement that could not be taken is a different fact
    from one that disagreed.
    """
    try:
        import foundationscale  # noqa: F401  -- probing importability, not using it
    except ImportError:
        pass
    else:
        return None
    here = Path(__file__).resolve().parent
    src = _repo_src_root(here)
    if src is None:
        return f"no src/foundationscale/__init__.py in any ancestor of {here}"
    sys.path.insert(0, os.fspath(src))
    try:
        import foundationscale  # noqa: F401  -- probing importability, not using it
    except ImportError as exc:
        return f"{src} is on sys.path and `import foundationscale` still failed: {exc}"
    return None


# ---------------------------------------------------------------------------
# Environment probes -- every precondition names its refusal reason
# ---------------------------------------------------------------------------


def _hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cache_misses() -> list[str]:
    """SPEC pins the cache: a miss is UNMEASURED, never a download, never an error."""
    hub = _hf_hub_cache()
    wanted = {
        MODEL: f"models--{MODEL.replace('/', '--')}",
        DATASET: f"datasets--{DATASET.replace('/', '--')}",
    }
    misses: list[str] = []
    for label, dirname in wanted.items():
        present = hub.is_dir() and any(hub.glob(f"{dirname}*"))
        if not present:
            misses.append(
                f"{label} is not in the HF cache under {hub}; offline mode forbids "
                "downloading, so this is UNMEASURED, not an error"
            )
    return misses


def _child_env(src: Path | None) -> dict[str, str]:
    """The trainer child's environment: offline, and importable from the checkout.

    The offline flags make a cache miss fail fast and loud inside the trainer
    instead of silently downloading -- SPEC forbids the download. PYTHONPATH
    carries the discovered ``src/`` so the child resolves the same package the
    bootstrap just proved importable here.
    """
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    if src is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.fspath(src) + (os.pathsep + existing if existing else "")
    return env


def _trainer_supports_axis() -> str | None:
    """None when the shipped CLI really has ``--attn-implementation``; else the reason.

    SPEC's instruction for this row is "check, do not assume": the flag exists
    in the attached cli.py, and this re-checks it against the trainer that will
    actually run. A parser that cannot even be BUILT (its construction needs
    installed distribution metadata) is not a refusal -- the subprocess remains
    the backstop and will refuse on its own.
    """
    try:
        from foundationscale.train.cli import build_parser

        parser = build_parser()
    except Exception:  # noqa: BLE001 -- the subprocess is the backstop; see docstring
        return None
    options = {opt for action in parser._actions for opt in action.option_strings}
    if "--attn-implementation" not in options:
        return (
            "--attn-implementation is absent from the shipped foundationscale-train "
            "CLI: the attention axis this row varies has no flag, and inventing one "
            "is worse than refusing"
        )
    return None


def _flash_import_error(env: dict[str, str]) -> str | None:
    """None if ``flash_attn`` imports on the tray interpreter, else the ImportError text.

    The probe is a real import in a child process, run BEFORE any GPU work: on
    aarch64 in this container the package does not exist, and SPEC requires the
    arm to be reported 95 UNMEASURED with the ImportError text as the reason --
    not RED, and not a skipped arm that vanishes from the denominator.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import flash_attn"],
            env=env,
            capture_output=True,
            text=True,
            timeout=FLASH_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"`import flash_attn` did not finish within {FLASH_PROBE_TIMEOUT_S}s"
    if proc.returncode == 0:
        return None
    lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return f"`import flash_attn` exited {proc.returncode} with no stderr"


# ---------------------------------------------------------------------------
# Arm plan -- three arms, exactly one axis, asserted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArmSpec:
    name: str
    impl: str
    flags: dict[str, str]
    work_dir: Path


def _arm_specs(args: argparse.Namespace) -> list[_ArmSpec]:
    specs: list[_ArmSpec] = []
    for impl in IMPLS:
        work_dir = args.out_dir / "arms" / impl
        flags = {
            "--model": MODEL,
            "--dataset": DATASET,
            "--output-dir": str(work_dir),
            "--nodes": str(args.nodes),
            "--gpus-per-node": str(args.gpus_per_node),
            # #439: the trainer's Topology requires dp x tp x pp x ep x cp ==
            # nodes x gpus_per_node. This row emits no other degree, so all four take
            # the default of 1 (and are REFUSED above 1 -- #375) and dp IS the world
            # size. Omitting it left dp=1 against a width of 2, and every arm of pass
            # p414 refused "topology is not constructible (nothing touched)" before a
            # model was loaded -- which torchrun then reported as launcher rc 1 (#171).
            "--dp": str(args.nodes * args.gpus_per_node),
            "--precision": PRECISION,
            "--seed": str(args.seed),
            "--max-steps": str(args.max_steps),
            # The derived logging cadence samples ~2 points in a 20-step run;
            # --logging-steps 1 makes the equivalence claim rest on every step.
            "--logging-steps": "1",
            "--attn-implementation": impl,
        }
        if args.profile_name is not None:
            flags["--profile-name"] = args.profile_name
        else:
            flags["--profile-path"] = str(args.profile_path)
        specs.append(_ArmSpec(name=impl, impl=impl, flags=flags, work_dir=work_dir))
    return specs


def _round_spec(spec: _ArmSpec, round_index: int) -> _ArmSpec:
    """A per-round copy of ``spec`` with its OWN trainer directory.

    ``_find_manifest`` searches a directory, so two rounds sharing one work dir
    would hand round 1 the manifest round 0 left behind -- and every round would
    then agree, which is a worse failure than a crash because it looks like
    evidence. ``--output-dir`` is the axis assertion's declared mechanical
    exception (it already differs per arm), so suffixing it cannot widen the
    row's single axis; the assertion runs on the base specs regardless.
    """
    work_dir = spec.work_dir.parent / f"{spec.work_dir.name}_r{round_index}"
    flags = dict(spec.flags)
    flags["--output-dir"] = str(work_dir)
    return replace(spec, flags=flags, work_dir=work_dir)


def _assert_single_axis(specs: list[_ArmSpec]) -> None:
    """SPEC: arms of one row differ in EXACTLY ONE axis -- asserted, not intended.

    ``--output-dir`` is excluded from the comparison: each arm needs its own
    scratch directory for the trainer, a mechanical difference that cannot
    reach the measurement. Everything else -- model, dataset, seed, precision,
    steps, profile -- must be identical, so the only key whose value may vary
    across arms is ``--attn-implementation``. The shared seed falls out of this
    same check: it is one flag with one value, so a per-arm seed would raise
    here rather than silently measure nothing.
    """
    mechanical = {"--output-dir"}
    base = specs[0].flags
    varied: set[str] = set()
    for spec in specs[1:]:
        assert set(spec.flags) == set(base), (
            f"arm {spec.name} has different flag KEYS: {sorted(set(spec.flags) ^ set(base))}"
        )
        varied |= {k for k in base if spec.flags[k] != base[k]}
    varied -= mechanical
    assert varied == {"--attn-implementation"}, (
        f"arms vary in {sorted(varied)}; the row allows exactly one axis (--attn-implementation)"
    )


# ---------------------------------------------------------------------------
# Telemetry readers -- the manifest is the measurement, stdout is never scraped
# ---------------------------------------------------------------------------


def _find_manifest(work_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    candidates = sorted(
        work_dir.rglob("*.json"),
        key=lambda p: ("manifest" not in p.name, str(p)),
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "schema_version" in data and "findings" in data:
            return data, None
    return None, f"no RunManifest-shaped JSON (schema_version + findings) under {work_dir}"


def _metric(telemetry: Mapping[str, Any], key: str) -> dict[str, Any]:
    """One scalar telemetry entry, normalised. An ``unmeasured`` source is a
    statement: its value IS the reason string (API.md) and is carried verbatim."""
    raw = telemetry.get(key)
    if raw is None:
        return {
            "key": key,
            "value": None,
            "source": "unmeasured",
            "unit": None,
            "reason": f"key {key!r} is absent from the manifest telemetry section",
        }
    entry = raw if isinstance(raw, dict) else {"value": raw, "source": "measured", "unit": None}
    if entry.get("source") == "unmeasured":
        return {
            "key": key,
            "value": None,
            "source": "unmeasured",
            "unit": entry.get("unit"),
            "reason": str(entry.get("value")),
        }
    value = entry.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {
            "key": key,
            "value": float(value),
            "source": str(entry.get("source") or "measured"),
            "unit": entry.get("unit"),
        }
    return {
        "key": key,
        "value": None,
        "source": "unmeasured",
        "unit": entry.get("unit"),
        "reason": f"value for {key!r} is not numeric: {value!r}",
    }


def _loss_series(
    telemetry: Mapping[str, Any],
) -> tuple[list[list[float]] | None, str | None]:
    """The per-step loss series as ``[step, loss]`` pairs, or the reason it cannot be read.

    ``train_loss_curve`` is the genuine per-step series and is read FIRST. A
    malformed curve is reported with the shape it actually had -- never
    skipped in favour of the bare ``train_loss`` aggregate, which is the run
    mean and must not pour into a curve comparison (a one-point GB200 "curve"
    was exactly that aggregate, and it returned GREEN). Without the curve, two
    legacy spellings remain: one list-valued entry (labelled 0..n-1, the only
    labels available), or one scalar per step under a ``train_loss_step_<digits>``
    key -- that exact form, not any key sharing the ``train_loss`` prefix,
    because a prefix match would also claim the run mean as a series point. An
    ``unmeasured`` loss entry short-circuits all spellings: its value IS the
    reason.
    """

    def too_short(points: list[list[float]]) -> str:
        return (
            f"per-step loss series has {len(points)} point(s); comparing curves "
            "needs at least 2 -- a single point is the run aggregate, not a "
            "curve. logging_steps_effective and max_steps are the knobs that "
            "produce more points"
        )

    def enough(points: list[list[float]]) -> tuple[list[list[float]] | None, str | None]:
        # Enforced here, at the single return gate, so every spelling obeys it:
        # one scalar per arm can show neither agreement nor divergence, and an
        # empty list is absence of evidence -- UNMEASURED, never a refutation.
        if len(points) < 2:
            return None, too_short(points)
        return points, None

    if "train_loss_curve" in telemetry:
        raw = telemetry["train_loss_curve"]
        entry = raw if isinstance(raw, dict) else {"value": raw, "source": "measured"}
        if entry.get("source") == "unmeasured":
            return None, str(entry.get("value"))
        value = entry.get("value")

        def malformed() -> str:
            return (
                "train_loss_curve is malformed: expected a list of [step, loss] "
                "pairs (int-like step, real-number loss, bool excluded -- bool is "
                f"an int subclass), got {value!r}"
            )

        if not isinstance(value, list):
            return None, malformed()
        points: list[list[float]] = []
        for element in value:
            if (
                not (isinstance(element, (list, tuple)) and len(element) == 2)
                or isinstance(element[0], bool)
                or not isinstance(element[0], int)
                or isinstance(element[1], bool)
                or not isinstance(element[1], (int, float))
            ):
                return None, malformed()
            points.append([int(element[0]), float(element[1])])
        return enough(points)

    candidates = {k: v for k, v in telemetry.items() if k == "loss" or k.startswith("train_loss")}
    if not candidates:
        return None, "no per-step loss series in telemetry (no 'loss'/'train_loss*' key)"
    for key in sorted(candidates):
        raw = candidates[key]
        entry = raw if isinstance(raw, dict) else {"value": raw, "source": "measured"}
        if entry.get("source") == "unmeasured":
            return None, str(entry.get("value"))
        value = entry.get("value")
        if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            return enough([[i, float(x)] for i, x in enumerate(value)])
    stepped: list[tuple[int, float]] = []
    for key in sorted(candidates):
        if not key.startswith("train_loss_step_"):
            continue
        raw = candidates[key]
        entry = raw if isinstance(raw, dict) else {"value": raw, "source": "measured"}
        value = entry.get("value")
        suffix = key[len("train_loss_step_") :]
        if suffix.isdigit() and isinstance(value, (int, float)) and not isinstance(value, bool):
            stepped.append((int(suffix), float(value)))
    if stepped:
        return enough([[step, loss] for step, loss in sorted(stepped)])
    return None, f"loss-like keys present but none is a numeric series: {sorted(candidates)}"


# ---------------------------------------------------------------------------
# Records -- one JSON file per arm, <row>_<arm>.json, reasons verbatim
# ---------------------------------------------------------------------------


def _base_record(spec: _ArmSpec) -> dict[str, Any]:
    return {
        "row": ROW_ID,
        "file": Path(__file__).name,
        "claim": CLAIM,
        "control_rule": CONTROL_RULE,
        "arm": spec.name,
        "axis": {"attn_implementation": spec.impl},
        "fixed": {
            "model": MODEL,
            "dataset": DATASET,
            "precision": PRECISION,
            "seed": int(spec.flags["--seed"]),
            "max_steps": int(spec.flags["--max-steps"]),
        },
        "status": "measured",
        "reason": None,
        "metrics": {},
    }


def _unmeasured_arm_record(spec: _ArmSpec, reason: str) -> dict[str, Any]:
    record = _base_record(spec)
    record["status"] = "unmeasured"
    record["reason"] = reason
    return record


def _write_record(out_dir: Path, arm: str, record: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ROW}_{arm}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def _fmt(value: Any, spec: str = ".4f") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unmeasured"
    return format(value, spec)


def _arm_line(record: dict[str, Any]) -> str:
    label = f"ARM {record['arm']}:"
    if record["status"] != "measured":
        return f"{label:<26} UNMEASURED -- {record['reason']}"
    m = record["metrics"]
    series = m["loss_series"]["value"] or []
    # Each entry is a [step, loss] pair; the human-readable line shows the loss.
    first_loss = series[0][1] if series else None
    return (
        f"{label:<26} steps/s={_fmt(m['steps_per_second']['value'])} "
        f"runtime={_fmt(m['train_runtime_s']['value'], '.2f')}s "
        f"peak={_fmt(m['peak_memory_allocated_bytes']['value'], '.0f')}B "
        f"flos={_fmt(m['total_flos']['value'], '.3e')} "
        f"steps={len(series)} loss[0]={_fmt(first_loss, '.5f')}"
    )


# ---------------------------------------------------------------------------
# Arm runner
# ---------------------------------------------------------------------------


def _run_arm(
    spec: _ArmSpec,
    arm_index: int,
    master_port_base: int,
    env: dict[str, str],
    round_index: int = 0,
) -> dict[str, Any]:
    """Invoke the shipped trainer once via torchrun, then read its RunManifest telemetry.

    The launcher -- ``python -m torch.distributed.run``, never a bare
    ``torchrun`` executable, which on this estate resolves to a host script
    whose shebang swaps the interpreter (#128) -- is what produces the runtime
    topology evidence (LOCAL_WORLD_SIZE et al.) the trainer refuses to default
    from cfg; a bare interpreter dies on ``train.local_world_size_unset``.
    Those variables are never set by hand in this file: the launcher must be
    their genuine producer (#375).

    stdout is captured for failure diagnosis only -- the measurement is read
    from the manifest, never scraped from the stream.
    """
    spec.work_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic distinct rendezvous port per arm AND per round: a fixed port
    # collides when arms run concurrently on one node, and a port reused by the
    # next round can still be in TIME_WAIT and refuse the rendezvous. A random
    # port would make a failure unreproducible, so each round gets its own
    # contiguous block of len(IMPLS).
    master_port = master_port_base + round_index * len(IMPLS) + arm_index
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        spec.flags["--nodes"],
        "--nproc-per-node",
        spec.flags["--gpus-per-node"],
        "--master-port",
        str(master_port),
        "-m",
        "foundationscale.train.cli",
    ]
    for flag, value in spec.flags.items():
        argv.extend([flag, value])
    try:
        proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=ARM_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _unmeasured_arm_record(spec, f"trainer did not finish within {ARM_TIMEOUT_S}s")
    if proc.returncode != 0:
        tail = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()][-3:]
        # torchrun flattens the child's exit code (#171): proc.returncode is the
        # LAUNCHER's code -- a 95 refusal and a crash both arrive as 1 -- so a
        # non-zero code is UNMEASURED, never RED; they cannot be told apart.
        reason = (
            f"launcher (torch.distributed.run) exited {proc.returncode}; the "
            "trainer's declared exit code is not observable through torchrun "
            "(#171), so a refusal and a crash are indistinguishable"
        )
        if tail:
            reason += f": {' | '.join(tail)}"
        return _unmeasured_arm_record(spec, reason)
    manifest, why = _find_manifest(spec.work_dir)
    if manifest is None:
        return _unmeasured_arm_record(spec, why or "no manifest")
    telemetry = manifest.get("telemetry")
    if not isinstance(telemetry, dict):
        return _unmeasured_arm_record(
            spec, "RunManifest has no telemetry mapping (a pre-telemetry writer?)"
        )
    record = _base_record(spec)
    # proc.returncode here is the launcher's: torchrun flattens the child's
    # exit code (#171), so the trainer's declared verdict is not observable.
    record["trainer"] = {
        "argv": argv,
        "launcher_returncode": proc.returncode,
        "returncode_note": (
            "torch.distributed.run's exit code, not the trainer's declared "
            "verdict -- torchrun flattens the child code (#171)"
        ),
    }
    metrics = {key: _metric(telemetry, key) for key in SCALAR_KEYS}
    series, series_reason = _loss_series(telemetry)
    loss_entry: dict[str, Any] = {
        "key": "loss_series",
        "value": series,
        "source": "measured" if series is not None else "unmeasured",
        "unit": None,
    }
    if series_reason is not None:
        loss_entry["reason"] = series_reason
    metrics["loss_series"] = loss_entry
    record["metrics"] = metrics
    holes = [m for m in (metrics["steps_per_second"], loss_entry) if m["source"] == "unmeasured"]
    if holes:
        record["status"] = "unmeasured"
        record["reason"] = "; ".join(f"{m['key']}: {m['reason']}" for m in holes)
    return record


# ---------------------------------------------------------------------------
# Adjudicator -- pure: no I/O, no clock, no GPU
# ---------------------------------------------------------------------------


def _max_loss_divergence(series: dict[str, dict[int, float]]) -> tuple[float, str]:
    """Worst per-step |delta| over all arm PAIRS, and where it happened.

    Series are compared on step LABELS, never positionally: two arms logged at
    different cadences hold losses for different steps at the same list index,
    and a zip would compare step-against-wrong-step and call the result a
    verdict. The comparison therefore runs over the INTERSECTION of the arms'
    step labels; the caller has already refused an intersection smaller than 2.
    """
    common = set.intersection(*(set(points) for points in series.values()))
    worst = 0.0
    where = "no pairs compared"
    compared = 0
    names = sorted(series)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for step in sorted(common):
                compared += 1
                delta = abs(series[left][step] - series[right][step])
                if delta > worst:
                    worst = delta
                    where = f"{left} vs {right} at step {step}"
    if compared and where == "no pairs compared":
        # Every delta was exactly 0, so the `worst` sentinel never moved. Saying
        # "no pairs compared" here would report perfect agreement in the same
        # words as a comparison that never ran -- the one distinction a reader
        # of a GREEN needs. Name the count instead; the sentinel stays reachable
        # for the genuinely-zero case (a single arm has no pair).
        where = f"all {compared} compared point(s) agree exactly"
    return worst, where


def adjudicate(arms: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """Apply the row's control rule to arm records.

    Order is fixed by the contract: an arm that could not be measured makes the
    ROW unmeasured (95) -- the flash arm's ImportError is a statement, not a
    refutation, and a row that cannot evaluate its control rule is never GREEN.
    Only three measured arms give the rule itself a verdict.
    """
    details: list[str] = []
    failures: list[str] = []
    unmeasured = [a for a in arms if a.get("status") != "measured"]
    if unmeasured:
        for arm in unmeasured:
            details.append(f"ARM {arm['arm']}: UNMEASURED -- {arm['reason']}")
        details.append(
            f"deciding metrics exist for {len(arms) - len(unmeasured)}/{len(arms)} arms; "
            "the control rule cannot be evaluated"
        )
        return UNMEASURED, details

    series = {
        a["arm"]: {int(step): float(loss) for step, loss in a["metrics"]["loss_series"]["value"]}
        for a in arms
    }
    common = set.intersection(*(set(points) for points in series.values()))
    step_sets = "; ".join(
        f"{name} logged steps {sorted(points)}" for name, points in sorted(series.items())
    )
    if not common:
        # Cadences that share no step cannot be compared at all: absence of a
        # shared measurement is UNMEASURED, never a refutation.
        details.append(
            "no loss step is common to every arm, and positional comparison "
            f"would compare step-against-wrong-step ({step_sets}); the curves "
            "cannot be said to agree or to diverge"
        )
        return UNMEASURED, details
    if all(step == 0 for step in common):
        # Step 0 precedes the first optimizer update: both impls start from the
        # same weights and see the same first batch, so they agree there
        # whatever the backward pass does. This is a guard, not a requirement
        # -- real trainers log after the update and never contain step 0.
        details.append(
            "every compared loss step is 0, and step 0 is before the first "
            "optimizer update: identical initial weights and an identical first "
            "batch make the impls agree there by construction, so a verdict "
            "resting on it says nothing about the attention implementations"
        )
        return UNMEASURED, details
    if len(common) < 2:
        details.append(
            f"only {len(common)} loss step is common to every arm ({step_sets}); "
            "one shared point is a comparison of two numbers, not of two curves"
        )
        return UNMEASURED, details
    restricted = {name: {s: pts[s] for s in common} for name, pts in series.items()}
    worst, where = _max_loss_divergence(restricted)
    magnitude = max(abs(x) for pts in restricted.values() for x in pts.values())
    allowed = LOSS_ATOL + LOSS_RTOL * magnitude
    if worst > allowed:
        failures.append(
            f"FAIL: loss curves DIVERGE over the {len(common)}-step intersection -- "
            f"max |delta| {worst:.6g} ({where}) exceeds {allowed:.6g} "
            f"(atol={LOSS_ATOL}, rtol={LOSS_RTOL}); the impls do not agree on logits"
        )
    else:
        details.append(
            f"loss curves agree over the {len(common)}-step intersection: max |delta| "
            f"{worst:.6g} ({where}) within {allowed:.6g} (atol={LOSS_ATOL}, "
            f"rtol={LOSS_RTOL})"
        )

    sps = {a["arm"]: float(a["metrics"]["steps_per_second"]["value"]) for a in arms}
    detail_sps = ", ".join(f"{name}={value:.4f} steps/s" for name, value in sps.items())
    slowest = min(sps.values())
    spread = (max(sps.values()) - slowest) / slowest if slowest > 0 else math.inf
    if slowest <= 0:
        failures.append(f"FAIL: steps/s is not positive ({detail_sps}); timing is broken")
    elif spread < STEP_TIME_IDENTICAL_RTOL:
        failures.append(
            f"FAIL: CONTROL RULE REFUTED -- {CONTROL_RULE}. Observed {detail_sps}: "
            f"spread {spread:.2%} sits inside the {STEP_TIME_IDENTICAL_RTOL:.0%} band, "
            "the silent-fallback shape"
        )
    else:
        details.append(
            f"step times differ: {detail_sps}; spread {spread:.2%} clears the "
            f"{STEP_TIME_IDENTICAL_RTOL:.0%} identity band"
        )

    if failures:
        return RED, failures + details
    return GREEN, details


def _reduce_rounds(per_round: list[list[dict[str, Any]]]) -> tuple[int, list[str]]:
    """Combine whole-round verdicts across INTERLEAVED rounds (#418).

    The reduction rule here is UNANIMITY, not a paired delta: see
    ``PAIRED_STATISTIC_ABSTENTION`` above for why ``reduce_paired_rounds`` is
    not applicable to a three-arm unsigned claim (#421). Unanimity is the
    weaker instrument, and saying so is the point -- an abstention that is
    named can be closed later, a fabricated direction cannot be un-published.

    Rounds that disagree are 95. A verdict that depends on WHICH round ran is a
    property of the pass -- cache state, co-tenancy, clock drift -- not of the
    attention implementation, and the rotation between rounds is what makes the
    disagreement visible instead of being absorbed into whichever round ran
    first.
    """
    lines: list[str] = []
    if len(per_round) < 2:
        lines.append(
            f"CANNOT MEASURE: {len(per_round)} round(s) reached the cross-round "
            "reducer; with a single round the arm order cannot rotate, so order "
            "stays confounded with arm identity (#418)"
        )
        return REFUSE, lines

    codes: list[int] = []
    for round_index, arms in enumerate(per_round):
        code, details = adjudicate(arms)
        codes.append(code)
        lines.append(f"--- round {round_index}: rc={code} ---")
        lines.extend(details)
    lines.append(PAIRED_STATISTIC_ABSTENTION)

    unresolved = [index for index, code in enumerate(codes) if code not in (GREEN, RED)]
    if unresolved:
        lines.append(
            f"round(s) {unresolved} of {len(per_round)} could not evaluate the "
            "control rule, so the row is UNMEASURED: a round that measured "
            "nothing is not evidence either way, and the remaining rounds are "
            "not a smaller experiment -- they are a different one"
        )
        return UNMEASURED, lines
    if all(code == RED for code in codes):
        lines.append(
            f"every one of {len(per_round)} interleaved rounds refuted the claim, "
            "with the arm order rotated between them -- the refutation is not an "
            "order artefact"
        )
        return RED, lines
    if all(code == GREEN for code in codes):
        lines.append(
            f"all {len(per_round)} interleaved rounds upheld the claim under "
            "rotated arm order, so the agreement does not rest on one ordering"
        )
        return GREEN, lines
    greens = [index for index, code in enumerate(codes) if code == GREEN]
    reds = [index for index, code in enumerate(codes) if code == RED]
    lines.append(
        f"rounds DISAGREE: {greens} upheld the claim and {reds} refuted it. "
        "Which round ran is not a property of the attention implementation "
        "(#418), so the row is UNMEASURED rather than a coin-flip verdict"
    )
    return UNMEASURED, lines


def _paired_statistic_record(rounds: int) -> dict[str, Any]:
    """The #421 abstention as a RECORD, not only as a line of stdout.

    An abstention that lives in a print statement dies with the scrollback. The
    ``arm`` name is deliberately not one of ``IMPLS`` and ``kind`` says
    ``control``, so a reader collecting arm payloads cannot mistake it for a
    fourth attention implementation -- the row's denominator is still three.
    """
    return {
        "row": ROW_ID,
        "file": Path(__file__).name,
        "claim": CLAIM,
        "control_rule": CONTROL_RULE,
        "arm": "paired_statistic",
        "kind": "control",
        "axis": {"cross_round_statistic": "reduce_paired_rounds"},
        "status": "unmeasured",
        "verdict": "UNMEASURED",
        "reason": PAIRED_STATISTIC_ABSTENTION,
        "rounds": rounds,
        "metrics": {},
    }


# ---------------------------------------------------------------------------
# Self-test -- synthetic arm records, including sets the control rule REFUTES
# ---------------------------------------------------------------------------


def _synthetic_arm(
    arm: str,
    steps_per_second: float,
    losses: list[float],
    steps: list[int] | None = None,
) -> dict[str, Any]:
    """The record shape ``_run_arm`` emits, without the trainer.

    ``steps`` defaults to 1..n: a real trainer logs AFTER the first optimizer
    update, so step 0 never appears in a measured series -- the synthetic arms
    must exercise the same shape the hardware produces. Loss series entries
    are ``[step, loss]`` pairs, the shape ``_loss_series`` now returns.
    """
    if steps is None:
        steps = list(range(1, len(losses) + 1))
    return {
        "row": ROW_ID,
        "arm": arm,
        "status": "measured",
        "reason": None,
        "metrics": {
            "steps_per_second": {
                "key": "steps_per_second",
                "value": steps_per_second,
                "source": "derived",
                "unit": "steps/s",
            },
            "loss_series": {
                "key": "loss_series",
                "value": [[s, v] for s, v in zip(steps, losses, strict=True)],
                "source": "measured",
            },
        },
    }


def _synthetic_unmeasured_arm(arm: str, reason: str) -> dict[str, Any]:
    record = _synthetic_arm(arm, 0.0, [])
    record["status"] = "unmeasured"
    record["reason"] = reason
    record["metrics"] = {}
    return record


def _self_test() -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    floor_controls("T1-12", record)
    gpu_reachability_controls("T1-12", record)

    record(
        "C1 CONTROL_RULE is the matrix's verbatim text",
        CONTROL_RULE == "identical step times across impls = the impl never changed -> RED",
        CONTROL_RULE,
    )

    losses = [2.302, 2.101, 1.974, 1.881]
    wiggle = [x + 2e-4 for x in losses]  # bf16 reduction-order noise, inside tolerance

    passing = [
        _synthetic_arm("eager", 1.000, losses),
        _synthetic_arm("sdpa", 1.220, wiggle),
        _synthetic_arm("flash_attention_2", 1.470, losses),
    ]
    code, details = adjudicate(passing)
    record(
        "C2 agreeing curves + distinct step times is GREEN",
        code == GREEN,
        f"rc={code} :: {details[-1]}",
    )

    identical = [
        _synthetic_arm("eager", 1.000, losses),
        _synthetic_arm("sdpa", 1.004, wiggle),
        _synthetic_arm("flash_attention_2", 0.998, losses),
    ]
    code, details = adjudicate(identical)
    record(
        "C3 the control rule REFUTES identical step times",
        code == RED,
        f"rc={code} :: {details[0]}",
    )

    edge = [
        _synthetic_arm("eager", 1.000, losses),
        _synthetic_arm("sdpa", 1.030, wiggle),
        _synthetic_arm("flash_attention_2", 1.060, losses),
    ]
    code, details = adjudicate(edge)
    record(
        "C4 a 6% spread is NOT 'identical' (band fires both ways)",
        code == GREEN,
        f"rc={code} :: {details[-1]}",
    )

    diverged = [
        _synthetic_arm("eager", 1.000, losses),
        _synthetic_arm("sdpa", 1.220, wiggle),
        _synthetic_arm("flash_attention_2", 1.470, [x + 0.4 for x in losses]),
    ]
    code, details = adjudicate(diverged)
    record(
        "C5 diverging loss curves are RED",
        code == RED,
        f"rc={code} :: {details[0]}",
    )

    flash_reason = "ModuleNotFoundError: No module named 'flash_attn'"
    with_hole = [
        _synthetic_arm("eager", 1.000, losses),
        _synthetic_arm("sdpa", 1.220, wiggle),
        _synthetic_unmeasured_arm("flash_attention_2", flash_reason),
    ]
    code, details = adjudicate(with_hole)
    joined = "\n".join(details)
    record(
        "C6 an unmeasured arm is 95, reason verbatim, denominator stays 3",
        code == UNMEASURED and flash_reason in joined and len(with_hole) == 3,
        f"rc={code} :: {details[0]}",
    )

    def _spec(impl: str, seed: str = "42") -> _ArmSpec:
        return _ArmSpec(
            name=impl,
            impl=impl,
            flags={
                "--model": MODEL,
                "--seed": seed,
                "--output-dir": f"/tmp/{impl}",
                "--attn-implementation": impl,
            },
            work_dir=Path("/tmp"),
        )

    try:
        _assert_single_axis([_spec(impl) for impl in IMPLS])
    except AssertionError as exc:
        record("C7 single-axis assertion accepts the real plan", False, str(exc)[:80])
    else:
        record(
            "C7 single-axis assertion accepts the real plan",
            True,
            "only --attn-implementation varies",
        )
    try:
        _assert_single_axis([_spec("eager"), _spec("sdpa", seed="7")])
    except AssertionError as exc:
        record("C8 the assertion refuses a second axis (seed)", True, str(exc)[:80])
    else:
        record("C8 the assertion refuses a second axis (seed)", False, "no raise")

    telemetry = {
        "steps_per_second": {
            "key": "steps_per_second",
            "value": 1.25,
            "source": "derived",
            "unit": "steps/s",
        },
        "peak_memory_allocated_bytes": {
            "key": "peak_memory_allocated_bytes",
            "value": "CUDA not initialised on this host",
            "source": "unmeasured",
            "unit": "bytes",
        },
    }
    measured = _metric(telemetry, "steps_per_second")
    hole = _metric(telemetry, "peak_memory_allocated_bytes")
    absent = _metric(telemetry, "total_flos")
    record(
        "C9 _metric reads measured / unmeasured / absent entries",
        measured["value"] == 1.25
        and hole["reason"] == "CUDA not initialised on this host"
        and absent["source"] == "unmeasured",
        f"{measured['value']} :: {hole['reason']} :: {absent['reason'][:40]}",
    )

    as_list, _ = _loss_series({"train_loss": {"value": [2.3, 2.1], "source": "measured"}})
    as_steps, _ = _loss_series(
        {
            "train_loss_step_1": {"value": 2.1, "source": "measured"},
            "train_loss_step_0": {"value": 2.3, "source": "measured"},
        }
    )
    none, series_reason = _loss_series(
        {"train_loss": {"value": "loss logging disabled", "source": "unmeasured"}}
    )
    record(
        "C10 _loss_series reads list, per-step and unmeasured spellings",
        as_list == [[0, 2.3], [1, 2.1]]
        and as_steps == [[0, 2.3], [1, 2.1]]
        and none is None
        and series_reason == "loss logging disabled",
        f"{as_list} :: {as_steps} :: {series_reason}",
    )

    curve, _ = _loss_series(
        {
            "train_loss_curve": {
                "value": [[0, 2.9], [1, 2.7], [2, 2.4]],
                "source": "measured",
            }
        }
    )
    curve_steps = [step for step, _ in curve or []]
    curve_losses = [loss for _, loss in curve or []]
    record(
        "C11 train_loss_curve reads as losses at labelled steps, in order",
        curve_steps == [0, 1, 2] and curve_losses == [2.9, 2.7, 2.4],
        f"steps={curve_steps} losses={curve_losses}",
    )

    agg_none, agg_reason = _loss_series({"train_loss": {"value": [2.31], "source": "measured"}})
    code, details = adjudicate(
        [
            _synthetic_unmeasured_arm("sdpa", agg_reason or "no series"),
            _synthetic_unmeasured_arm("eager", agg_reason or "no series"),
        ]
    )
    record(
        "C12 the one-point aggregate is 95, never GREEN (the GB200 T1-9 shape)",
        agg_none is None
        and code == UNMEASURED
        and code != GREEN
        and "run aggregate" in (agg_reason or "")
        and "logging_steps_effective" in (agg_reason or "")
        and "max_steps" in (agg_reason or ""),
        f"rc={code} :: {agg_reason}",
    )

    code, details = adjudicate(
        [
            _synthetic_arm("sdpa", 1.200, [2.9, 2.7], steps=[0, 1]),
            _synthetic_arm("eager", 1.440, [2.9, 2.5], steps=[0, 9]),
        ]
    )
    joined = "\n".join(details)
    record(
        "C13 agreement at step 0 alone is equal by construction -- 95",
        code == UNMEASURED
        and code != GREEN
        and "0" in joined
        and "before the first optimizer update" in joined,
        f"rc={code} :: {details[0]}",
    )

    with_agg, _ = _loss_series(
        {
            "train_loss_curve": {"value": [[0, 2.9], [1, 2.7]], "source": "measured"},
            "train_loss": {"value": 99.0, "source": "measured"},
        }
    )
    record(
        "C14 the bare train_loss aggregate never enters the series",
        with_agg == [[0, 2.9], [1, 2.7]]
        and 99.0 not in [x for pair in with_agg or [] for x in pair],
        f"{with_agg}",
    )

    flat, flat_reason = _loss_series(
        {
            "train_loss_curve": {"value": [2.9, 2.7], "source": "measured"},
            "train_loss": {"value": 99.0, "source": "measured"},
        }
    )
    record(
        "C15 a flat train_loss_curve is malformed -- 95, no fall-through",
        flat is None and flat_reason is not None and "[step, loss]" in flat_reason,
        f"{flat_reason}",
    )

    code_disj, details_disj = adjudicate(
        [
            _synthetic_arm("sdpa", 1.200, [2.9, 2.7], steps=[1, 2]),
            _synthetic_arm("eager", 1.440, [2.9, 2.7], steps=[3, 4]),
        ]
    )
    joined_disj = "\n".join(details_disj)
    # The overlap fixture is built so the two comparison rules DISAGREE: the
    # arms share steps 2 and 3, where both log 2.9 then 2.7, so a step-keyed
    # comparison is GREEN. Compared positionally the same two series are
    # 3.1-vs-2.9, 2.9-vs-2.7, 2.7-vs-2.4 and go RED. A fixture whose losses
    # line up under BOTH rules would pass against an implementation that
    # never learned to key on the step label.
    code_olap, details_olap = adjudicate(
        [
            _synthetic_arm("sdpa", 1.200, [3.1, 2.9, 2.7], steps=[1, 2, 3]),
            _synthetic_arm("eager", 1.440, [2.9, 2.7, 2.4], steps=[2, 3, 4]),
        ]
    )
    joined_olap = "\n".join(details_olap)
    record(
        "C16 step labels: disjoint is 95, partial overlap uses the intersection",
        code_disj == UNMEASURED
        and "[1, 2]" in joined_disj
        and "[3, 4]" in joined_disj
        and code_olap == GREEN
        and "2-step intersection" in joined_olap
        # A GREEN must say the comparison HAPPENED. "no pairs compared" is the
        # zero-comparison sentinel, and exact agreement used to borrow it.
        and "no pairs compared" not in joined_olap,
        f"rc={code_disj}/{code_olap} :: {details_disj[0]} :: {details_olap[0]}",
    )

    # ---- #419: the capability floor, reachable through its injectable seams --
    # Without the seams the refusing arms would be unreachable decoration, and a
    # precondition that can never fail is indistinguishable from one that always
    # passes. These drive the real helper, not a copy of its logic.

    class _StubCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    class _StubTorch:
        cuda = _StubCuda

    def _healthy_import(name: str) -> object:
        if name == "torch":
            return _StubTorch
        raise AssertionError(f"unexpected module probe: {name}")

    def _dead_import(name: str) -> object:
        raise OSError(f"libcudart.so.12: cannot open shared object file (probing {name})")

    def _dead_smoke(_torch_module: object) -> None:
        raise RuntimeError("CUDA error: no kernel image is available for execution")

    fires = capability_floor_reasons(import_module=_dead_import)
    record(
        "C17 the capability floor FIRES on a dead native library",
        bool(fires) and any("native libraries are dead" in reason for reason in fires),
        f"{len(fires)} reason(s)",
    )
    clear = capability_floor_reasons(
        import_module=_healthy_import, smoke_op=lambda _torch_module: None
    )
    record(
        "C18 a healthy tray clears the floor (it does not refuse everything)",
        clear == [],
        f"{len(clear)} reason(s)",
    )
    smoke_fails = capability_floor_reasons(import_module=_healthy_import, smoke_op=_dead_smoke)
    record(
        "C19 a device that counts but cannot run a kernel is a floor miss",
        bool(smoke_fails),
        f"{len(smoke_fails)} reason(s)",
    )

    # C20: three arms rotate, so no impl keeps the cache-cold first slot. With a
    # fixed order the LAST arm always inherits two warm arms' filesystem cache
    # and looks fastest -- on a row whose verdict is a step-time spread, that
    # bias points straight at the claim (#418).
    plan = interleaved_round_plan(list(IMPLS), rounds=3)
    order = [[name for round_index, name in plan if round_index == r] for r in range(3)]
    first_slots = {round_order[0] for round_order in order}
    record(
        "C20 the interleaved plan rotates arm order between rounds (#418)",
        order[0] == list(IMPLS)
        and len(first_slots) == 3
        and all(sorted(round_order) == sorted(IMPLS) for round_order in order),
        f"{order}",
    )

    def _round(eager_sps: float, sdpa_sps: float, flash_sps: float) -> list[dict[str, Any]]:
        return [
            _synthetic_arm("eager", eager_sps, losses),
            _synthetic_arm("sdpa", sdpa_sps, wiggle),
            _synthetic_arm("flash_attention_2", flash_sps, losses),
        ]

    code, _lines = _reduce_rounds([_round(1.000, 1.220, 1.470) for _ in range(3)])
    record("C21 three agreeing rounds are GREEN", code == GREEN, f"rc={code}")

    # C22: THE MUST-FIRE control for #418 on this row. Round 0 ran cold and its
    # three impls came out inside the identity band -- the single-round rule
    # calls that RED -- while rounds 1 and 2 spread cleanly. Whichever round ran
    # alone would have become the verdict. Disagreement is 95.
    code, lines = _reduce_rounds(
        [_round(1.000, 1.004, 0.998), _round(1.000, 1.220, 1.470), _round(1.010, 1.230, 1.460)]
    )
    record(
        "C22 rounds that disagree are 95, never a coin-flip verdict",
        code == UNMEASURED and any("DISAGREE" in line for line in lines),
        f"rc={code}",
    )

    # C23: the silent-fallback shape survives the round loop. Adding rounds must
    # not be able to launder the row's own control rule into an abstention.
    code, lines = _reduce_rounds([_round(1.000, 1.004, 0.998) for _ in range(3)])
    record(
        "C23 the identical-step-time shape is RED in every round, and stays RED",
        code == RED and any(CONTROL_RULE in line for line in lines),
        f"rc={code}",
    )

    # C24: one unmeasured arm in ONE round sinks the row to 95. The other rounds
    # are not a smaller experiment, they are a different one.
    code, lines = _reduce_rounds(
        [
            _round(1.000, 1.220, 1.470),
            [
                _synthetic_arm("eager", 1.000, losses),
                _synthetic_arm("sdpa", 1.220, wiggle),
                _synthetic_unmeasured_arm("flash_attention_2", flash_reason),
            ],
        ]
    )
    record(
        "C24 an unmeasured round makes the row 95, not a majority verdict",
        code == UNMEASURED and any("could not evaluate" in line for line in lines),
        f"rc={code}",
    )

    # C25: the #421 abstention is a RECORD and a LINE, not a comment. It must
    # name the issue, and its arm name must not be mistakable for a fourth impl.
    abstention = _paired_statistic_record(3)
    _, abstain_lines = _reduce_rounds([_round(1.000, 1.220, 1.470) for _ in range(2)])
    record(
        "C25 the paired statistic abstains in the open, and is not a fourth arm",
        "#421" in abstention["reason"]
        and abstention["status"] == "unmeasured"
        and abstention["arm"] not in IMPLS
        and any(PAIRED_STATISTIC_ABSTENTION in line for line in abstain_lines),
        f"arm={abstention['arm']} status={abstention['status']}",
    )

    # C26: each round gets its OWN trainer directory. _find_manifest searches a
    # directory, so a shared one would hand round 1 the manifest round 0 left
    # behind -- and every round would then agree, which looks like evidence.
    base = _spec("eager")
    rounded = [_round_spec(base, r) for r in range(3)]
    dirs = {str(spec.work_dir) for spec in rounded}
    # The port formula is the one _run_arm uses; a port reused by the next round
    # can still be in TIME_WAIT and refuse the rendezvous.
    ports = {29612 + r * len(IMPLS) + i for r in range(3) for i in range(len(IMPLS))}
    record(
        "C26 rounds get disjoint work dirs and disjoint rendezvous ports",
        len(dirs) == 3
        and str(base.work_dir) not in dirs
        and all(spec.flags["--output-dir"] == str(spec.work_dir) for spec in rounded)
        and len(ports) == 3 * len(IMPLS),
        f"{len(dirs)} dirs, {len(ports)} ports",
    )

    # C27: one round is not a measurement. --rounds 1 refuses 96 BEFORE any GPU,
    # trainer or import probe is touched, so the refusal costs nothing.
    with tempfile.TemporaryDirectory(prefix="t1_12_rounds_") as tmp:
        one_round = _build_parser().parse_args(
            ["--out-dir", tmp, "--profile-name", "self-test", "--rounds", "1"]
        )
        code = _run(one_round)
    record("C27 --rounds 1 refuses 96 before touching a GPU", code == REFUSE, f"rc={code}")

    default_rounds = (
        _build_parser().parse_args(["--out-dir", "/tmp", "--profile-name", "self-test"]).rounds
    )
    record(
        "C28 --rounds defaults to the shared DEFAULT_ROUNDS",
        default_rounds == DEFAULT_ROUNDS and default_rounds >= 2,
        f"default={default_rounds}",
    )

    # C29: the #413 warmup is DISCARDED, and disjointness is what makes that
    # true rather than merely claimed. Its work dir and its rendezvous port must
    # miss every (round, arm) pair the plan can emit -- a warmup at
    # round_index=0 would overwrite round 0's eager directory and hand
    # _find_manifest the warmup's manifest, which is the C26 failure wearing a
    # different hat. The port formula and the 29612 base are the ones _run_arm
    # and the parser use.
    warm = _round_spec(_spec("eager"), 3)
    plan_dirs = {str(_round_spec(_spec(impl), r).work_dir) for impl in IMPLS for r in range(3)}
    plan_ports = {29612 + r * len(IMPLS) + i for r in range(3) for i in range(len(IMPLS))}
    warm_port = 29612 + 3 * len(IMPLS)
    record(
        "C29 the #413 warmup is disjoint from every round and feeds no verdict",
        str(warm.work_dir) not in plan_dirs
        and warm_port not in plan_ports
        and _WARMUP_ARM not in IMPLS,
        f"dir={warm.work_dir.name} port={warm_port} vs {len(plan_ports)} plan ports",
    )

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(f"T1-12 self-test: {len(checks) - len(failed)}/{len(checks)} controls PASS")
    if failed:
        for name in failed:
            print(f"FAIL: {name}")
        return RED
    return GREEN


# ---------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error -- the one code a launcher cannot
    interpret (#387). The override refuses at the source instead."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(REFUSE)


def _build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="t1_12_attention_impl.py",
        description=(
            "T1-12: eager / sdpa / flash_attention_2 agree on logits, differ in step "
            "time. Three arms, one axis (--attn-implementation), fixed seed, model, "
            "dataset and bf16 precision. Exit codes: 0 GREEN, 5 RED, 95 UNMEASURED, "
            "96 REFUSE."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for <row>_<arm>.json records and per-arm trainer output",
    )
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=(
            # Percent signs doubled -- argparse interpolates help through `%`.
            "how many INTERLEAVED rounds the three arms run (default "
            "%(default)s). The arm order rotates every round: a fixed order "
            "confounds order with arm identity, and it biases TOWARD the claim, "
            "because a later arm inherits the earlier arms' warm filesystem "
            "cache and so looks faster -- which is exactly the step-time spread "
            "this row adjudicates (#418). Two is the floor. Each extra round "
            "costs one more trainer subprocess PER ARM."
        ),
    )
    p.add_argument(
        "--master-port-base",
        type=int,
        default=29612,
        help=(
            # Percent signs doubled -- argparse interpolates help through `%`.
            "torchrun rendezvous port for round 0 arm 0 (default %(default)s); "
            "round r arm i uses base + r*3 + i, so neither concurrent arms nor a "
            "port still in TIME_WAIT from the previous round can collide "
            "(deterministic, never random)"
        ),
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile-name", help="name of a built-in ClusterProfile")
    group.add_argument("--profile-path", type=Path, help="path to a ClusterProfile JSON")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the SAME seed is passed to every arm; changing it changes all three",
    )
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument(
        "--self-test",
        action="store_true",
        help="exercise the adjudicator on synthetic records and exit; no GPU needed",
    )
    return p


def _run(args: argparse.Namespace) -> int:
    if args.rounds < 2:
        print(
            f"REFUSE: --rounds {args.rounds} cannot rotate the arm order, so order "
            "stays confounded with arm identity and the later arms keep the warm "
            "cache the earlier ones left them (#418). Pass 2 or more."
        )
        return REFUSE
    reason = _ensure_package_importable()
    if reason is not None:
        print(f"UNMEASURED: foundationscale is not importable -- {reason}")
        return UNMEASURED
    # The three gates below ask whether this row's INPUTS are present. The
    # capability floor asks the deeper question -- is the device the driver
    # counts one a kernel can actually run on, is the wheel ABI-consistent, are
    # the native libraries alive -- and its misses are 95 by the helper's own
    # contract. None of them say anything about attention implementations, so
    # none of them may read as RED.
    floor_misses = capability_floor_reasons()
    if floor_misses:
        for miss in floor_misses:
            print(f"UNMEASURED: {miss}")
        return UNMEASURED
    # The gates above ask whether this process can run a kernel at all. #432 asks
    # the next question down: the tray was allocated N GPUs and the process can
    # open fewer. A tray allocated gres/gpu=4 that exposes 2 measures the claim on
    # 2 and publishes it as 4 -- a 2x denominator error no arm would ever notice,
    # because every arm inherits the same wrong denominator.
    reachable = reachable_gpu_count()
    gpu_reason = gpu_reachability_reason(args.gpus_per_node, reachable=reachable)
    if gpu_reason is not None:
        # Unknowable and short are different states: one was not measured, the
        # other was measured and came up short. They get different exit codes.
        if reachable is None:
            print(f"UNMEASURED: {gpu_reason}")
            return UNMEASURED
        print(f"REFUSE: {gpu_reason}")
        return REFUSE
    axis_reason = _trainer_supports_axis()
    if axis_reason is not None:
        print(f"REFUSE: {axis_reason}")
        return REFUSE
    missing = _cache_misses()
    if missing:
        for m in missing:
            print(f"UNMEASURED: {m}")
        return UNMEASURED

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = _repo_src_root(Path(__file__).resolve().parent)
    env = _child_env(src)
    specs = _arm_specs(args)
    _assert_single_axis(specs)

    print(f"T1-12: {CLAIM}")
    print(f"control rule: {CONTROL_RULE}")
    print(
        f"fixed across arms: model={MODEL} dataset={DATASET} precision={PRECISION} "
        f"seed={args.seed} max_steps={args.max_steps}"
    )
    print(f"rounds: {args.rounds}, arm order rotated between them (#418)")

    flash_reason = _flash_import_error(env)
    spec_by_name = {spec.name: spec for spec in specs}
    arm_position = {spec.name: index for index, spec in enumerate(specs)}

    # #413 -- the ONE global warmup, run BEFORE the plan is consumed. This is
    # interleaved_round_plan's own stated contract ("the ONE global warmup the
    # budget allows is the row's to run before consuming this plan"), and t1_11
    # honours it with its sensitivity probe, whose step time already carries no
    # meaning. This row had no such free arm, so round 0 paid the cold-cache
    # cost alone -- and that cost is not noise: t1_11 measured a cache-cold pass
    # at -20.1% against +21.7..+30.2% warm, a SIGN FLIP on the same metric.
    # Under _reduce_rounds a single disagreeing round is 95, so a cold round 0
    # does not merely widen the spread, it decides the row.
    #
    # It runs the first arm at round_index=args.rounds, arm_index=0 -- one past
    # every index the plan can emit -- so its work dir and its rendezvous port
    # are disjoint from all of them (C29). The expensive cold state is the
    # shared filesystem cache, not anything per-arm, so WHICH arm warms it does
    # not matter; specs[0] is chosen because eager is the one arm no import can
    # gate. The record IS written, under a name that is not an arm, and it never
    # enters per_round: "discarded" has to be something the evidence tree can
    # show, not something that happens silently.
    warmup_spec = _round_spec(specs[0], args.rounds)
    warmup_record = _run_arm(warmup_spec, 0, args.master_port_base, env, round_index=args.rounds)
    warmup_record["arm"] = _WARMUP_ARM
    warmup_record["discarded"] = (
        "#413 global warmup: run to pull the model and dataset shards through the "
        "shared filesystem cache every later round reads. Its measurement feeds NO "
        "verdict and is in NO round's sample."
    )
    warmup_path = _write_record(args.out_dir, _WARMUP_ARM, warmup_record)
    print(f"WARMUP (discarded, #413)   {_arm_line(warmup_record)}", flush=True)
    print(f"{'':<26} [record: {warmup_path}]", flush=True)

    per_round: list[list[dict[str, Any]]] = [[] for _ in range(args.rounds)]
    for round_index, name in interleaved_round_plan(list(spec_by_name), rounds=args.rounds):
        spec = _round_spec(spec_by_name[name], round_index)
        if spec.impl == "flash_attention_2" and flash_reason is not None:
            record = _unmeasured_arm_record(spec, flash_reason)
            record["detail"] = (
                "probed `import flash_attn` on the tray interpreter before any GPU "
                "work; the arm stays in the denominator (3) as UNMEASURED, never RED"
            )
        else:
            record = _run_arm(
                spec,
                arm_position[name],
                args.master_port_base,
                env,
                round_index=round_index,
            )
        record["round"] = round_index
        per_round[round_index].append(record)
        # Two paths per arm: the round-stamped one is the evidence, and the
        # unsuffixed one keeps the name every existing reader already looks for
        # (last round wins, and it is named in the record's own "round" field).
        _write_record(args.out_dir, f"{spec.name}_r{round_index}", record)
        path = _write_record(args.out_dir, spec.name, record)
        print(f"r{round_index} {_arm_line(record)}  [record: {path}]", flush=True)

    abstention = _paired_statistic_record(args.rounds)
    control_path = _write_record(args.out_dir, "paired_statistic", abstention)
    print(f"{'CONTROL paired_statistic:':<26} UNMEASURED (#421)  [record: {control_path}]")

    code, details = _reduce_rounds(per_round)
    print("=" * 72)
    for d in details:
        print(d)
    if code == GREEN:
        print("T1-12 VERDICT GREEN: loss curves agree within tolerance and step times differ")
    elif code == RED:
        print("T1-12 VERDICT RED: the agreement half or the control rule refuted the claim")
    elif code == REFUSE:
        print("T1-12 VERDICT CANNOT_MEASURE: the cross-round reducer had nothing to reduce")
    else:
        print("T1-12 VERDICT UNMEASURED: a deciding metric is a statement, not a number")
    return code


def main(argv: list[str]) -> int:
    floor_reason = python_floor_reason()
    if floor_reason is not None:
        print(f"T1-12 VERDICT REFUSE: {floor_reason}")
        return REFUSE

    if not argv:
        print(__doc__)
        return REFUSE
    try:
        # Checked before ANY environment probe: the self-test is synthetic by
        # contract and must run on a bare CI interpreter with no GPU and no
        # foundationscale install.
        if argv[0] == "--self-test":
            return _self_test()
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            # _Parser.error raises SystemExit(REFUSE); --help raises 0 and passes
            # through untouched -- the same split cli.py makes for #387.
            if exc.code is None or exc.code == 0:
                raise
            return REFUSE
        if args.self_test:
            return _self_test()
        return _run(args)
    except Exception as exc:  # noqa: BLE001 -- an escaped exception is classified, never adjudicated
        traceback.print_exc(file=sys.stderr)
        code, reason = classify_boundary_exception(exc)
        verdict = "UNMEASURED" if code == UNMEASURED else "CANNOT_MEASURE"
        detail = f"{reason}; unexpected {type(exc).__name__} escaped the run body: {exc}"
        record = {
            "row": ROW_ID,
            "file": Path(__file__).name,
            "claim": CLAIM,
            "control_rule": CONTROL_RULE,
            "status": verdict,
            "verdict": verdict,
            "boundary_code": code,
            "reason": detail,
            "boundary_reason": reason,
            "traceback": traceback.format_exc(),
        }
        try:
            path = _write_record(args.out_dir, verdict.lower(), record)
            print(f"{verdict.lower()} record: {path}")
        except Exception:  # noqa: BLE001 -- a failed write must not mask the verdict
            pass
        print("=" * 72)
        print(f"T1-12 VERDICT {verdict}: {detail}")
        return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
