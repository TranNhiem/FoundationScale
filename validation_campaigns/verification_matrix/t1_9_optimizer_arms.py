#!/usr/bin/env python3
"""T1-9: the optimizer knob must MOVE the loss curve, or it is decoration.

The claim under test is not "four trainings run". It is the matrix row's
narrower one: adamw / adamw_torch_fused / adafactor / sgd differ in step time
and loss curve. Four arms therefore differ in exactly one axis --
``--optimizer`` -- under one seed, bf16, Qwen/Qwen2.5-1.5B on
fancyzhx/ag_news, and the row's control rule is the verdict, not a comment:

    identical curves across optimizers = the knob did nothing -> RED

The rule is ALL-arms-identical on purpose. adamw_torch and adamw_torch_fused
are the same mathematics behind two kernels, so that pair agreeing on a curve
is expected and proves nothing either way; what refutes the claim is every
arm landing on the SAME curve, the signature of the flag being dropped before
it ever reached TrainingArguments -- the silent-fallback shape this row
exists to catch. Step times are recorded as evidence (the fused kernel earns
its keep there or nowhere), but the verdict reads the curves, because the
matrix names the curves.

Two ways to run it:

  --self-test   exercises the ADJUDICATOR against synthetic arm records,
                including a set the control rule REFUTES (asserting a 5),
                the tolerance boundary on both sides, the single-axis
                assertion on both sides, the one-point "curve" that must
                read UNMEASURED (the false GREEN real GB200 hardware
                produced), and the argparse refusal. No GPU,
                no network, no model, no foundationscale import -- the
                adjudicator is pure, so its test must be too.

  --out-dir D   runs the four arms. Each arm invokes the shipped
                ``foundationscale-train`` CLI as a subprocess -- the thing
                under test is the thing that ships, not an in-process
                reimplementation -- then reads the RunManifest's ``telemetry``
                section: ``train_runtime_s``, ``steps_per_second``,
                ``peak_memory_allocated_bytes``, ``total_flos`` and the
                per-step loss series (``train_loss_curve`` first, else
                ``train_loss_step_<n>`` entries). stdout is never scraped.
                One JSON record per arm, ``t1_9_<arm>.json``, under D.

The model and dataset are expected in the tray's HF cache and are NOT
downloaded (the child runs with HF_HUB_OFFLINE=1); a cache miss is 95
UNMEASURED, not an error. A telemetry entry with ``source == "unmeasured"``
is a statement whose value is the reason string: if the deciding metric --
the loss curve -- is unmeasured on any arm, the row adjudicates 95 and the
emitted JSON carries that reason verbatim, so the operator learns WHY
without re-running. The flags every arm needs are checked against the
trainer's live parser before the first arm starts; were ``--optimizer``
ever to vanish from cli.py, the run refuses 96 and names it rather than
inventing a knob.

Exit codes follow the four-state contract: 0 claim upheld, 5 the control
rule refuted it, 95 ran but the deciding quantity could not be measured,
96 refused to run at all. argparse's own exit 2 is out of contract (#387),
so the parser's error() is overridden to exit 96.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from t1_interpreter_floor import floor_controls, python_floor_reason

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

ROW = "t1_9"
MODEL = "Qwen/Qwen2.5-1.5B"
DATASET = "fancyzhx/ag_news"
PRECISION = "bf16"

CLAIM = "adamw / adamw_torch_fused / adafactor / sgd differ in step time and loss curve"

# The matrix entry's refuting observation, verbatim. It is the verdict.
CONTROL_RULE = "identical curves across optimizers = the knob did nothing -> RED"

# arm name -> value handed to the trainer's --optimizer flag. The matrix says
# "adamw"; adamw_torch is what TrainingArguments' `optim` calls that kernel.
ARMS: tuple[tuple[str, str], ...] = (
    ("adamw", "adamw_torch"),
    ("adamw_torch_fused", "adamw_torch_fused"),
    ("adafactor", "adafactor"),
    ("sgd", "sgd"),
)

# Curves agreeing everywhere within this read as IDENTICAL: a dropped knob
# re-runs the same computation, so only kernel nondeterminism separates the
# arms, while a real optimizer change moves the loss by orders more.
IDENTICAL_TOL = 1e-6

# Fewer points than this can show neither divergence nor agreement: a lone
# point is the run's aggregate train_loss, not a curve -- the exact input
# that produced the false GREEN on real GB200 hardware. Such a series reads
# as UNMEASURED, never GREEN and never RED.
MIN_CURVE_POINTS = 2

# Every trainer flag the arms invoke. Checked against the shipped parser
# before the first arm starts -- an axis with no flag behind it refuses (96)
# and names the missing flag; it never invents one.
REQUIRED_TRAINER_FLAGS: tuple[str, ...] = (
    "--model",
    "--dataset",
    "--output-dir",
    "--max-steps",
    "--per-device-batch-size",
    "--learning-rate",
    "--seed",
    "--precision",
    "--optimizer",
    "--nodes",
    "--gpus-per-node",
    # Dense loss sampling: at the trainer's derived cadence a 20-step run
    # yields ~2 curve points and the equivalence claim rests on nothing.
    "--logging-steps",
)

# The four scalars read out of telemetry alongside the loss series.
SCALAR_KEYS: tuple[str, ...] = (
    "train_runtime_s",
    "steps_per_second",
    "peak_memory_allocated_bytes",
    "total_flos",
)

VERDICT_NAMES = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSED"}


@dataclass
class ArmRecord:
    """One arm's outcome. ``reason`` is the verbatim unmeasured-reason string."""

    arm: str
    optimizer: str
    status: str  # "measured" | "unmeasured"
    reason: str | None = None
    # The torchrun LAUNCHER's exit code, not the trainer's: torchrun flattens
    # the child's code (#171), so the trainer's declared verdict code is not
    # observable through the launcher.
    launcher_exit_code: int | None = None
    loss_curve: list[float] = field(default_factory=list)
    telemetry: dict[str, dict[str, Any]] = field(default_factory=dict)


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

    Same bootstrap the sibling rows carry: an installed distribution cannot be
    assumed, so this falls back to the checkout's own ``src/``. The found root
    is ALSO exported through PYTHONPATH, because the trainer runs as a
    subprocess and ``sys.path`` does not cross a process boundary. An
    environment where neither works is UNMEASURED (95), never RED (5): a
    measurement that could not be taken is a different fact from one that
    disagreed.
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
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        os.fspath(src) if not existing else os.fspath(src) + os.pathsep + existing
    )
    return None


# ---------------------------------------------------------------------------
# Preconditions -- checked, not assumed
# ---------------------------------------------------------------------------


def _cache_misses() -> list[str]:
    """HF-cache entries the row needs that are absent. A miss is 95, never a download."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    misses = []
    for kind, repo in (("models", MODEL), ("datasets", DATASET)):
        expected = hub / f"{kind}--{repo.replace('/', '--')}"
        if not expected.is_dir():
            misses.append(f"{repo} (expected {expected})")
    return misses


def _missing_trainer_flags() -> list[str]:
    """Flags this row invokes that the shipped trainer parser does not have.

    cli.py is attached to this row's spec and carries ``--optimizer``, but a
    campaign that ASSUMES a flag exists is how an invented knob gets run, so
    the live parser is asked. argparse exposes no public "every option string"
    API; ``_actions`` is the stable internals every introspection recipe uses.
    """
    from foundationscale.train.cli import build_parser

    option_strings: set[str] = set()
    for action in build_parser()._actions:
        option_strings.update(action.option_strings)
    return [flag for flag in REQUIRED_TRAINER_FLAGS if flag not in option_strings]


# ---------------------------------------------------------------------------
# Arms -- one axis, asserted
# ---------------------------------------------------------------------------


def _arm_flags(args: argparse.Namespace, arm_dir: Path, optimizer: str) -> dict[str, str]:
    """One arm's trainer invocation as flag -> value, so the single-axis
    assertion can diff dicts instead of guessing at argv positions."""
    flags = {
        "--model": MODEL,
        "--dataset": DATASET,
        "--output-dir": os.fspath(arm_dir),
        "--max-steps": str(args.max_steps),
        "--per-device-batch-size": str(args.per_device_batch_size),
        "--learning-rate": str(args.learning_rate),
        "--seed": str(args.seed),
        "--precision": PRECISION,
        "--optimizer": optimizer,
        "--nodes": str(args.nodes),
        "--gpus-per-node": str(args.gpus_per_node),
        "--logging-steps": "1",
    }
    if args.profile_name is not None:
        flags["--profile-name"] = args.profile_name
    else:
        flags["--profile-path"] = os.fspath(args.profile_path)
    return flags


def _assert_single_axis(per_arm: dict[str, dict[str, str]]) -> None:
    """Prove the arms differ in EXACTLY ONE axis -- asserted, not intended.

    ``--output-dir`` is bookkeeping, not an axis: each arm needs its own
    trainer output directory or the second arm would overwrite the first's
    manifest. Everything else that varies across arms must be exactly
    ``--optimizer``; a second varying key (a seed that drifted, a precision
    typo) means the row measures nothing, and that must be loud.
    """
    arms = list(per_arm)
    reference = per_arm[arms[0]]
    varying: set[str] = set()
    for arm in arms[1:]:
        flags = per_arm[arm]
        if set(flags) != set(reference):
            raise AssertionError(f"arm {arm} flags {sorted(flags)} != {sorted(reference)}")
        varying.update(key for key in reference if flags[key] != reference[key])
    varying.discard("--output-dir")
    if varying != {"--optimizer"}:
        raise AssertionError(
            "T1-9 arms must differ in exactly one axis, --optimizer; "
            f"observed varying axes: {sorted(varying) if varying else 'none'}"
        )


# ---------------------------------------------------------------------------
# Reading the measurement -- telemetry, not stdout
# ---------------------------------------------------------------------------


def _find_manifest(arm_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The newest JSON under ``arm_dir`` that carries a telemetry mapping.

    The manifest's filename is the trainer's business, not this row's; what
    the row depends on is the telemetry SECTION, whose shape API.md pins.
    """
    candidates = sorted(
        (p for p in arm_dir.rglob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    for path in reversed(candidates):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("telemetry"), dict):
            return data, None
    return None, f"no JSON manifest with a telemetry section under {arm_dir}"


def _loss_step_order(key: str) -> tuple[int, int, str]:
    """Sort ``train_loss_step_2`` before ``train_loss_step_10`` -- numerically,
    not lexicographically, or a ten-step run reads its curve out of order."""
    tail = key.replace("/", "_").rsplit("_", 1)[-1]
    if tail.isdigit():
        return (0, int(tail), key)
    return (1, 0, key)


def _is_loss_step_key(key: str) -> bool:
    """Whether ``key`` is a series point: exactly ``train_loss_step_<digits>``.

    A prefix match is too wide a detector for this axis: it also admits the
    bare ``train_loss`` run aggregate and ``train_loss_curve`` itself, both
    different quantities that merely share a prefix.
    """
    return key.startswith("train_loss_step_") and key[len("train_loss_step_") :].isdigit()


def _loss_curve_series(value: Any) -> tuple[list[float] | None, str | None]:
    """Losses in ascending step order from a ``train_loss_curve`` value, or why
    it is not one.

    The value must be a list of ``[step, loss]`` pairs: an int-like step and
    a real-number loss, ``bool`` excluded -- it is an ``int`` subclass, not
    data. A malformed value is described by the shape it actually had, so the
    reason names what was seen; silently falling through to the bare
    aggregate would re-admit the one-point curve this gate exists to refuse.
    """
    if not isinstance(value, list):
        return None, f"value is a {type(value).__name__}, not a list of [step, loss] pairs"
    pairs: list[tuple[int, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None, f"element {item!r:.60} is not a [step, loss] pair"
        step, loss = item
        if not isinstance(step, int) or isinstance(step, bool):
            return None, f"step {step!r:.60} is not an int"
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            return None, f"loss {loss!r:.60} is not a real number"
        pairs.append((int(step), float(loss)))
    pairs.sort(key=lambda pair: pair[0])
    return [loss for _, loss in pairs], None


def _extract_telemetry(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[float], str | None]:
    """Pull the four scalars and the per-step loss series out of a manifest.

    Returns (scalars, loss_curve, unmeasured_reason). ``train_loss_curve`` is
    the authoritative series when present -- a list of ``[step, loss]``
    pairs; a malformed one is UNMEASURED naming the shape, never silently
    skipped in favour of the bare ``train_loss`` aggregate, which is the run
    MEAN: a different quantity that merely shares a prefix. Absent that key,
    only exact ``train_loss_step_<digits>`` keys make up the series. The
    loss curve is the deciding metric; when no measured series exists the
    reason says why -- an unmeasured entry's own value, verbatim, when there
    is one, because that string IS the statement the trainer made, and the
    emitted record must carry it so the operator learns WHY without
    re-running.
    """
    section = manifest.get("telemetry")
    if not isinstance(section, dict):
        return {}, [], "manifest carries no telemetry mapping"
    scalars: dict[str, dict[str, Any]] = {}
    for key in SCALAR_KEYS:
        entry = section.get(key)
        if isinstance(entry, dict):
            scalars[key] = {
                "value": entry.get("value"),
                "source": entry.get("source"),
                "unit": entry.get("unit"),
            }
        else:
            scalars[key] = {"value": None, "source": "absent", "unit": None}

    losses: list[float] | None = None
    curve_entry = section.get("train_loss_curve")
    if isinstance(curve_entry, dict):
        source = curve_entry.get("source")
        value = curve_entry.get("value")
        if source == "unmeasured" and isinstance(value, str) and value:
            return scalars, [], value
        if source in ("measured", "derived"):
            series, bad = _loss_curve_series(value)
            if bad is not None:
                return scalars, [], f"malformed train_loss_curve: {bad}"
            losses = series
        else:
            return (
                scalars,
                [],
                (f"train_loss_curve has source {source!r} but no usable series or reason string"),
            )
    if losses is None:
        loss_entries = [
            (k, v)
            for k, v in section.items()
            if isinstance(v, dict) and (_is_loss_step_key(k) or k == "train_loss")
        ]
        measured = [
            (k, v["value"])
            for k, v in loss_entries
            if _is_loss_step_key(k)
            and v.get("source") in ("measured", "derived")
            and isinstance(v.get("value"), (int, float))
            and not isinstance(v.get("value"), bool)
        ]
        measured.sort(key=lambda kv: _loss_step_order(kv[0]))
        if measured:
            losses = [float(value) for _, value in measured]
        else:
            for _, v in sorted(loss_entries, key=lambda kv: _loss_step_order(kv[0])):
                if (
                    v.get("source") == "unmeasured"
                    and isinstance(v.get("value"), str)
                    and v["value"]
                ):
                    return scalars, [], v["value"]
            # A lone bare train_loss IS a one-point series. Route it through
            # the gate below rather than claiming nothing was measured, so
            # the reason names the actual defect -- the run aggregate
            # standing where a curve should be (the GB200 false GREEN).
            aggregate = section.get("train_loss")
            if (
                isinstance(aggregate, dict)
                and aggregate.get("source") in ("measured", "derived")
                and isinstance(aggregate.get("value"), (int, float))
                and not isinstance(aggregate.get("value"), bool)
            ):
                losses = [float(aggregate["value"])]
            else:
                return (
                    scalars,
                    [],
                    ("no measured train_loss_step_<n> telemetry entries in the RunManifest"),
                )
    # ONE minimum-length gate governing both the train_loss_curve path and
    # the per-step fallback; two copies would drift apart.
    if len(losses) < MIN_CURVE_POINTS:
        return (
            scalars,
            [],
            (
                f"loss series has {len(losses)} point(s), fewer than the minimum "
                f"{MIN_CURVE_POINTS}: it cannot show divergence or agreement. Raise "
                "logging_steps_effective (denser sampling) and max_steps (more steps) "
                "to produce more points -- a single point is the run aggregate (the "
                "mean train_loss), not a curve"
            ),
        )
    return scalars, losses, None


# ---------------------------------------------------------------------------
# The adjudicator -- pure, so the self-test can refute it without a GPU
# ---------------------------------------------------------------------------


def _curves_identical(curves: list[list[float]]) -> tuple[bool, float]:
    """Whether every curve matches the first within IDENTICAL_TOL, and the
    largest per-step gap seen. A length mismatch is NOT identical."""
    reference = curves[0]
    max_delta = 0.0
    for curve in curves[1:]:
        if len(curve) != len(reference):
            return False, float("inf")
        for a, b in zip(reference, curve, strict=True):
            max_delta = max(max_delta, abs(a - b))
    return max_delta <= IDENTICAL_TOL, max_delta


def adjudicate(records: list[ArmRecord]) -> tuple[int, str]:
    """Apply the row's control rule to per-arm records. Pure: no I/O, no GPU.

    The deciding metric is the loss curve. Any arm whose curve is unmeasured
    makes the row 95 -- a row that produces numbers but cannot evaluate its
    control rule is UNMEASURED, never GREEN. All four curves identical is the
    refuting observation the matrix names, and it is the verdict: RED.
    """
    for record in records:
        if record.status != "measured":
            return UNMEASURED, f"arm {record.arm}: {record.reason or 'no reason recorded'}"
        if not record.loss_curve:
            return UNMEASURED, f"arm {record.arm}: measured status but an empty loss curve"
    identical, max_delta = _curves_identical([r.loss_curve for r in records])
    if identical:
        return RED, (
            f"{CONTROL_RULE} (max |delta| across arms {max_delta:.3e} <= tol {IDENTICAL_TOL:.1e})"
        )
    return GREEN, (
        f"loss curves diverge across optimizers (max |delta| {max_delta:.3e} > tol "
        f"{IDENTICAL_TOL:.1e}); the knob moved the training"
    )


# ---------------------------------------------------------------------------
# Record writer -- one JSON per arm, <row>_<arm>.json
# ---------------------------------------------------------------------------


def _write_records(out_dir: Path, records: list[ArmRecord], verdict: int, detail: str) -> None:
    for record in records:
        payload = {
            "row": ROW,
            "arm": record.arm,
            "claim": CLAIM,
            "control_rule": CONTROL_RULE,
            "optimizer": record.optimizer,
            "status": record.status,
            "reason": record.reason,
            "launcher_exit_code": record.launcher_exit_code,
            "telemetry": record.telemetry,
            "loss_curve": record.loss_curve,
            "verdict": VERDICT_NAMES[verdict],
            "verdict_detail": detail,
        }
        path = out_dir / f"{ROW}_{record.arm}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _run_arm(
    args: argparse.Namespace,
    arm: str,
    arm_index: int,
    optimizer: str,
    out_dir: Path,
    env: dict[str, str],
) -> ArmRecord:
    arm_dir = out_dir / "_runs" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    # Module form of torchrun, never a bare `torchrun` executable: on this
    # estate the bare name resolves to a HOST script whose anaconda shebang
    # swaps the interpreter out from under the container (#128). The
    # launcher must be the genuine producer of the topology -- WORLD_SIZE /
    # LOCAL_WORLD_SIZE / RANK are never set by hand here, because writing
    # them would manufacture the very fact the run exists to observe (#375).
    # The port is base + the arm's zero-based index: arms run concurrently
    # on one node (a fixed port collides), and it stays deterministic -- a
    # random port would make a failure unreproducible.
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(args.nodes),
        "--nproc-per-node",
        str(args.gpus_per_node),
        "--master-port",
        str(args.master_port_base + arm_index),
        "-m",
        "foundationscale.train.cli",
    ]
    for flag, value in _arm_flags(args, arm_dir, optimizer).items():
        cmd.extend([flag, value])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()
        if not lines:
            lines = (proc.stdout or "").strip().splitlines()
        # torchrun flattens the child's exit code (#171): this is the
        # LAUNCHER's code, so a trainer refusal and a crash are
        # indistinguishable -- a non-zero code adjudicates UNMEASURED,
        # never RED.
        reason = (
            f"torchrun launcher exited {proc.returncode} (the launcher's code; "
            "the trainer's declared code is not observable through torchrun, #171)"
        )
        if lines:
            reason = f"{reason}; last line: {lines[-1][:160]}"
        return ArmRecord(
            arm=arm,
            optimizer=optimizer,
            status="unmeasured",
            reason=reason,
            launcher_exit_code=proc.returncode,
        )
    manifest, miss = _find_manifest(arm_dir)
    if manifest is None:
        return ArmRecord(
            arm=arm,
            optimizer=optimizer,
            status="unmeasured",
            reason=miss,
            launcher_exit_code=proc.returncode,
        )
    scalars, curve, reason = _extract_telemetry(manifest)
    if reason is not None:
        return ArmRecord(
            arm=arm,
            optimizer=optimizer,
            status="unmeasured",
            reason=reason,
            launcher_exit_code=proc.returncode,
            telemetry=scalars,
        )
    return ArmRecord(
        arm=arm,
        optimizer=optimizer,
        status="measured",
        launcher_exit_code=proc.returncode,
        loss_curve=curve,
        telemetry=scalars,
    )


def _run(args: argparse.Namespace) -> int:
    if args.out_dir is None:
        print("REFUSED: --out-dir is required for a run; there is nowhere to write records")
        return REFUSE
    if (args.profile_name is None) == (args.profile_path is None):
        print("REFUSED: exactly one of --profile-name / --profile-path is required")
        return REFUSE
    reason = _ensure_package_importable()
    if reason is not None:
        print(f"UNMEASURED: foundationscale is not importable -- {reason}")
        return UNMEASURED
    missing = _missing_trainer_flags()
    if missing:
        print(
            f"REFUSED: the shipped trainer parser has no {missing} flag(s); "
            "the row's axis cannot be declared and no substitute will be invented"
        )
        return REFUSE
    misses = _cache_misses()
    if misses:
        print(
            "UNMEASURED: HF cache miss on the tray -- "
            + "; ".join(misses)
            + " (this row never downloads)"
        )
        return UNMEASURED

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_arm = {arm: _arm_flags(args, out_dir / "_runs" / arm, optimizer) for arm, optimizer in ARMS}
    _assert_single_axis(per_arm)

    out: list[str] = []
    out.append(f"claim:        {CLAIM}")
    out.append(f"control rule: {CONTROL_RULE}")
    out.append(
        f"setup: seed={args.seed} precision={PRECISION} max_steps={args.max_steps} "
        f"on every arm; arms differ in exactly one axis (--optimizer)"
    )

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"

    records: list[ArmRecord] = []
    for arm_index, (arm, optimizer) in enumerate(ARMS):
        record = _run_arm(args, arm, arm_index, optimizer, out_dir, env)
        records.append(record)
        if record.status == "measured":
            sps = record.telemetry.get("steps_per_second", {}).get("value")
            out.append(
                f"ARM {arm:<18} measured   -- {len(record.loss_curve)} loss points, "
                f"final={record.loss_curve[-1]:.4f}, steps/s={sps}"
            )
        else:
            out.append(f"ARM {arm:<18} UNMEASURED -- {record.reason}")

    verdict, detail = adjudicate(records)
    _write_records(out_dir, records, verdict, detail)
    for line in out:
        print(line)
    print("=" * 72)
    print(f"T1-9 VERDICT {VERDICT_NAMES[verdict]}: {detail}")
    print(f"records: {out_dir}/{ROW}_<arm>.json")
    return verdict


# ---------------------------------------------------------------------------
# Argument parsing -- argparse's exit 2 is out of contract (#387)
# ---------------------------------------------------------------------------


class _ContractParser(argparse.ArgumentParser):
    """A usage error is a refusal, not a crash: argparse would exit 2, which
    sits outside 0/5/95/96 and is the one code a launcher's case statement
    does not handle (#387). Overridden here to exit 96."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(REFUSE, f"{self.prog}: error: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = _ContractParser(
        prog="t1_9_optimizer_arms",
        description=(
            "T1-9 verification row: four optimizer arms, one axis, one seed. "
            "Exit codes: 0 claim upheld, 5 control rule refuted, "
            "95 UNMEASURED, 96 REFUSE."
        ),
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="exercise the adjudicator on synthetic records; no GPU, no network",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for t1_9_<arm>.json records and per-arm trainer output",
    )
    p.add_argument("--seed", type=int, default=42, help="the SAME seed on every arm")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument(
        "--master-port-base",
        type=int,
        default=29609,
        help="rendezvous port base for torchrun; arm i uses base + i -- "
        "deterministic, never random, so concurrent arms cannot collide "
        "and a failure reproduces",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--profile-name", default=None, help="passed through to the trainer")
    group.add_argument(
        "--profile-path", type=Path, default=None, help="passed through to the trainer"
    )
    return p


# ---------------------------------------------------------------------------
# Self-test -- the adjudicator must fire BOTH ways, or it proves nothing
# ---------------------------------------------------------------------------


def _synthetic_record(arm: str, curve: list[float]) -> ArmRecord:
    return ArmRecord(arm=arm, optimizer=arm, status="measured", loss_curve=list(curve))


def _self_test() -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    floor_controls("T1-9", record)

    base = [2.302, 2.101, 1.987, 1.912]

    # C1: diverging curves uphold the claim. The fused arm sits within
    # tolerance of plain adamw -- same mathematics, different kernel -- and
    # the row must still be GREEN, because the rule is ALL-arms-identical.
    diverging = [
        _synthetic_record("adamw", base),
        _synthetic_record("adamw_torch_fused", [v + 1e-7 for v in base]),
        _synthetic_record("adafactor", [2.301, 2.205, 2.118, 2.066]),
        _synthetic_record("sgd", [2.302, 2.290, 2.281, 2.274]),
    ]
    rc, detail = adjudicate(diverging)
    record("C1 diverging curves uphold the claim", rc == GREEN, f"rc={rc} {detail[:60]}")

    # C2: the control rule REFUTES four identical curves. This is the control
    # the matrix exists for -- a self-test that only feeds passing data
    # proves nothing.
    identical = [_synthetic_record(arm, base) for arm, _ in ARMS]
    rc, detail = adjudicate(identical)
    record(
        "C2 the control rule REFUTES identical curves",
        rc == RED and CONTROL_RULE in detail,
        f"rc={rc}",
    )

    # C3: an unmeasured deciding metric is 95, and the reason string must
    # survive into the emitted JSON verbatim -- the operator learns WHY
    # without re-running.
    why = "cuda error: no device found on tray"
    degraded = [
        _synthetic_record("adamw", base),
        _synthetic_record("adamw_torch_fused", base),
        _synthetic_record("adafactor", base),
        ArmRecord(arm="sgd", optimizer="sgd", status="unmeasured", reason=why),
    ]
    rc, detail = adjudicate(degraded)
    ok = rc == UNMEASURED and why in detail
    with tempfile.TemporaryDirectory(prefix="t1_9_selftest_") as tmp:
        _write_records(Path(tmp), degraded, rc, detail)
        payload = json.loads((Path(tmp) / f"{ROW}_sgd.json").read_text())
        ok = ok and payload["reason"] == why and payload["verdict"] == "UNMEASURED"
    record("C3 unmeasured deciding metric is 95, reason verbatim in JSON", ok, f"rc={rc}")

    # C4: the tolerance boundary fires both ways -- below it curves read as
    # identical (kernel noise is not an optimizer effect), above it they
    # read as diverging.
    within = [
        _synthetic_record(arm, [v + i * IDENTICAL_TOL / 10 for v in base])
        for i, (arm, _) in enumerate(ARMS)
    ]
    rc, _ = adjudicate(within)
    record("C4a noise below tolerance still reads as identical", rc == RED, f"rc={rc}")
    beyond = [
        _synthetic_record(arm, [v + i * IDENTICAL_TOL * 10 for v in base])
        for i, (arm, _) in enumerate(ARMS)
    ]
    rc, _ = adjudicate(beyond)
    record("C4b movement beyond tolerance reads as diverging", rc == GREEN, f"rc={rc}")

    # C5: the single-axis assertion fires both ways -- optimizer-only
    # variation passes; a second varying axis (a drifting seed) is loud.
    ok_flags = {
        arm: {
            "--seed": "42",
            "--precision": "bf16",
            "--optimizer": optimizer,
            "--output-dir": f"/runs/{arm}",
        }
        for arm, optimizer in ARMS
    }
    try:
        _assert_single_axis(ok_flags)
    except AssertionError as exc:
        record("C5a optimizer-only variation passes the assertion", False, str(exc)[:80])
    else:
        record("C5a optimizer-only variation passes the assertion", True, "no raise")
    bad_flags = {
        arm: {**flags, "--seed": str(i)} for i, (arm, flags) in enumerate(ok_flags.items())
    }
    try:
        _assert_single_axis(bad_flags)
    except AssertionError as exc:
        record("C5b a second varying axis is refused", "--seed" in str(exc), str(exc)[:80])
    else:
        record("C5b a second varying axis is refused", False, "no raise")

    # C6: telemetry extraction keeps numbers as numbers, carries an
    # unmeasured entry's reason verbatim, and orders steps numerically.
    manifest = {
        "telemetry": {
            "train_runtime_s": {
                "key": "train_runtime_s",
                "value": 12.5,
                "source": "measured",
                "unit": "s",
            },
            "steps_per_second": {
                "key": "steps_per_second",
                "value": 1.6,
                "source": "derived",
                "unit": "steps/s",
            },
            "peak_memory_allocated_bytes": {
                "key": "peak_memory_allocated_bytes",
                "value": "torch.cuda is not available on this box",
                "source": "unmeasured",
                "unit": "bytes",
            },
            "train_loss_step_10": {
                "key": "train_loss_step_10",
                "value": 1.9,
                "source": "measured",
                "unit": None,
            },
            "train_loss_step_2": {
                "key": "train_loss_step_2",
                "value": 2.2,
                "source": "measured",
                "unit": None,
            },
            "train_loss_step_1": {
                "key": "train_loss_step_1",
                "value": 2.3,
                "source": "measured",
                "unit": None,
            },
        }
    }
    scalars, curve, reason = _extract_telemetry(manifest)
    ok = (
        curve == [2.3, 2.2, 1.9]
        and scalars["train_runtime_s"]["value"] == 12.5
        and scalars["peak_memory_allocated_bytes"]["value"]
        == "torch.cuda is not available on this box"
        and scalars["total_flos"]["source"] == "absent"
        and reason is None
    )
    record("C6 numbers stay numbers, reasons verbatim, steps in order", ok, f"curve={curve}")

    # C7: an unmeasured loss entry's value IS the reason string.
    dark = {
        "telemetry": {
            "train_loss_step_1": {
                "key": "train_loss_step_1",
                "value": "loss logging gate never fired",
                "source": "unmeasured",
                "unit": None,
            }
        }
    }
    _, curve, reason = _extract_telemetry(dark)
    record(
        "C7 an unmeasured loss entry's value IS the reason",
        curve == [] and reason == "loss logging gate never fired",
        f"reason={reason!r}",
    )

    # C8: empty curves are UNMEASURED, never GREEN -- a row that cannot
    # evaluate its control rule does not get to pass.
    empty = [_synthetic_record(arm, []) for arm, _ in ARMS]
    rc, detail = adjudicate(empty)
    record("C8 empty curves are UNMEASURED, never GREEN", rc == UNMEASURED, f"rc={rc}")

    # C9: argparse's usage error exits 96, not the out-of-contract 2 (#387).
    try:
        _build_parser().parse_args(["--not-a-flag"])
    except SystemExit as exc:
        record("C9 argparse usage error exits 96, not 2 (#387)", exc.code == REFUSE, f"{exc.code}")
    else:
        record("C9 argparse usage error exits 96, not 2 (#387)", False, "no exit")

    # C10: the manifest's train_loss_curve IS the series -- losses in
    # ascending step order.
    curated = {
        "telemetry": {
            "train_loss_curve": {
                "key": "train_loss_curve",
                "value": [[0, 2.9], [1, 2.7], [2, 2.4]],
                "source": "measured",
                "unit": None,
            }
        }
    }
    _, curve, reason = _extract_telemetry(curated)
    record(
        "C10 train_loss_curve is read as the series, in step order",
        curve == [2.9, 2.7, 2.4] and reason is None,
        f"curve={curve}",
    )

    # C11 (MUST-FIRE): the real GB200 input -- every arm's only loss entry is
    # the bare train_loss aggregate. That is a one-point "curve", and
    # comparing one scalar per arm is not comparing curves: the row returned
    # GREEN off exactly this on hardware. It must now be 95, and the reason
    # must name the one-point problem.
    aggregate_only = {
        "telemetry": {
            "train_loss": {
                "key": "train_loss",
                "value": 2.838187426328659,
                "source": "measured",
                "unit": None,
            }
        }
    }
    _, curve, reason = _extract_telemetry(aggregate_only)
    degraded = [
        ArmRecord(
            arm=arm,
            optimizer=optimizer,
            status="unmeasured" if reason is not None else "measured",
            reason=reason,
            loss_curve=curve,
        )
        for arm, optimizer in ARMS
    ]
    rc, _ = adjudicate(degraded)
    reason_str = reason or ""
    ok = (
        rc == UNMEASURED
        and "1 point" in reason_str
        and "aggregate" in reason_str
        and "not a curve" in reason_str
    )
    record(
        "C11 MUST-FIRE: aggregate-only manifest is 95 naming the one-point problem",
        ok,
        f"rc={rc} reason={reason_str[:60]}",
    )

    # C12: the bare aggregate is never appended to a real series.
    mixed = {
        "telemetry": {
            "train_loss_curve": {
                "key": "train_loss_curve",
                "value": [[0, 2.9], [1, 2.7]],
                "source": "measured",
                "unit": None,
            },
            "train_loss": {
                "key": "train_loss",
                "value": 99.0,
                "source": "measured",
                "unit": None,
            },
        }
    }
    _, curve, reason = _extract_telemetry(mixed)
    record(
        "C12 the bare train_loss aggregate never enters a real series",
        curve == [2.9, 2.7] and 99.0 not in curve and reason is None,
        f"curve={curve}",
    )

    # C13: a flat list of numbers is not a list of pairs -- UNMEASURED naming
    # the shape, not a crash and not a silent fall-through to the aggregate.
    flat = {
        "telemetry": {
            "train_loss_curve": {
                "key": "train_loss_curve",
                "value": [2.9, 2.7, 2.4],
                "source": "measured",
                "unit": None,
            }
        }
    }
    _, curve, reason = _extract_telemetry(flat)
    reason_str = reason or ""
    record(
        "C13 a flat-list train_loss_curve is UNMEASURED naming the shape",
        curve == [] and "malformed" in reason_str and "not a [step, loss] pair" in reason_str,
        f"reason={reason_str[:70]}",
    )

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(f"T1-9 self-test: {len(checks) - len(failed)}/{len(checks)} controls PASS")
    if failed:
        for name in failed:
            print(f"FAIL: {name}")
        return RED
    return GREEN


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    floor_reason = python_floor_reason()
    if floor_reason is not None:
        print(f"T1-9 VERDICT REFUSE: {floor_reason}")
        return REFUSE

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # error() raises SystemExit(96), --help SystemExit(0); both become
        # returned codes so nothing escapes main().
        return exc.code if isinstance(exc.code, int) else REFUSE
    try:
        if args.self_test:
            return _self_test()
        return _run(args)
    except Exception as exc:  # noqa: BLE001 -- the contract has no code for "crashed"
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, end="")
        detail = f"unexpected {type(exc).__name__} escaped the run body: {exc}"
        if args.out_dir is not None:
            try:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "row": ROW,
                    "verdict": "RED",
                    "control_rule": CONTROL_RULE,
                    "detail": detail,
                    "traceback": tb,
                }
                path = args.out_dir / f"{ROW}_error.json"
                path.write_text(json.dumps(payload, indent=2) + "\n")
            except OSError:
                pass
        print("=" * 72)
        print(f"T1-9 VERDICT RED: {detail} -- full traceback on stderr and in {ROW}_error.json")
        return RED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
