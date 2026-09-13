#!/usr/bin/env python3
"""T1-10: accum=4,bs=1 and accum=1,bs=4 must trace the SAME loss curve.

The claim under test is equivalence, not difference -- this row is the INVERSE
of its three siblings. ``--gradient-accumulation-steps 4
--per-device-batch-size 1`` and ``--gradient-accumulation-steps 1
--per-device-batch-size 4`` are one effective batch of 4 split two ways, and a
correct trainer produces the same per-step loss curve for both. The refuting
observation is the verdict, encoded verbatim as CONTROL_RULE and applied by
``adjudicate``:

    curves differing beyond tolerance = accumulation is wrong -> RED

The tolerance is a noise floor, not slack. Splitting a batch reorders bf16
reductions, so two correct runs are not bit-identical; LOSS_ATOL and LOSS_RTOL
price that reordering. A gap beyond them means the accumulated gradients were
not the summed gradients -- accumulation that silently trains a different
curve, a defect no other row in the matrix would ever observe.

Both arms invoke the shipped trainer as a SUBPROCESS, one process per arm:
``peak_memory_allocated_bytes`` is a process-lifetime counter, so two arms in
one process would leak arm A's peak into arm B's telemetry. Both get the same
seed, the same ``--precision bf16``, and the same model and dataset from the
tray's HF cache. The cache is the only legal source: HF_HUB_OFFLINE is set so
a cache miss fails fast and adjudicates 95, never a download. The arms differ
in exactly one axis -- the accum x batch split -- and ``_check_single_axis``
proves that from the flag dicts rather than intending it.

The measurement is read from each run's RunManifest ``telemetry`` section,
never scraped from stdout. The deciding series is ``train_loss_curve`` -- a
list of ``[step, loss]`` pairs -- and it is read FIRST: the bare
``train_loss`` aggregate is the run MEAN, a different quantity that merely
shares the prefix, and a prefix match would read it as the step-0 point. An
entry with ``source == "unmeasured"`` is a
statement: its value is the reason, and that reason is carried into this row's
JSON verbatim so the operator learns WHY without re-running. A row that
produces numbers but cannot evaluate its control rule is 95, never GREEN --
and neither is one whose only agreeing step is 0: step 0 precedes the first
optimizer update, so identical models agree there by construction whatever
the accumulation knob did, and a GREEN from it measures initialization, not
equivalence.

Exit codes: 0 curves agree within tolerance, 5 they differ beyond it (or the
harness itself crashed), 95 the deciding curve could not be measured, 96 the
run was refused (bad command line, no cluster profile, a trainer flag this
row needs does not exist). argparse usage errors exit 96 through an error()
override -- argparse's native 2 sits outside the contract (#387).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from t1_interpreter_floor import floor_controls, python_floor_reason

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

ROW = "t1_10"
MODEL = "Qwen/Qwen2.5-1.5B"
DATASET = "fancyzhx/ag_news"
PRECISION = "bf16"

CLAIM = "accum=4,bs=1 and accum=1,bs=4 give the SAME loss curve"

# The refuting observation from the verification matrix, verbatim. This row is
# the inverse of its siblings: AGREEMENT is the pass, and a curve pair that
# differs beyond tolerance is the refutation.
CONTROL_RULE = "curves differing beyond tolerance = accumulation is wrong -> RED"

# bf16 reductions reorder when a batch is split, so equivalent runs are not
# bit-identical. These price that reordering; a floor tight enough to fail
# honest noise would make the row cry wolf.
LOSS_ATOL = 1e-3
LOSS_RTOL = 2e-2

# The one axis this row may move: how the fixed effective batch is split
# between accumulation and per-device batch. Both flags move together, in
# inverse proportion, or the arm pair is not this row.
SPLIT_AXIS = ("gradient_accumulation_steps", "per_device_batch_size")

# Telemetry keys copied into every arm record. The DECIDING metric is the
# per-step loss series; these four are the context the SPEC requires with it.
CONTEXT_TELEMETRY = (
    "train_runtime_s",
    "steps_per_second",
    "peak_memory_allocated_bytes",
    "total_flos",
)


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

    The tray may run this file under an interpreter that has never had
    ``pip install -e .`` applied to it, so an installed distribution cannot be
    assumed; this falls back to the checkout's own ``src/``. An environment
    where NEITHER works is reported by the caller as REFUSE (96): with no
    trainer to invoke, the row refused to run at all, and 96 is the contract's
    "precondition absent" -- never RED, which would read an environment gap as
    a defect in the tree.
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
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """One arm: a name and its two split-axis values. Everything else is shared."""

    name: str
    gradient_accumulation_steps: int
    per_device_batch_size: int

    @property
    def effective_batch(self) -> int:
        return self.gradient_accumulation_steps * self.per_device_batch_size


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(name="accum4_bs1", gradient_accumulation_steps=4, per_device_batch_size=1),
    ArmSpec(name="accum1_bs4", gradient_accumulation_steps=1, per_device_batch_size=4),
)


def _arm_flags(spec: ArmSpec, args: argparse.Namespace) -> dict[str, Any]:
    """Every trainer declaration this arm carries, as a plain dict.

    The single-axis proof diffs THESE dicts, so anything an arm varies must be
    in here: a flag set outside this dict is invisible to the check. Seed and
    precision are included so the proof also catches an arm that drifted on
    what must stay fixed -- an arm that differs by seed measures nothing.
    """
    return {
        "model": MODEL,
        "dataset": DATASET,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "precision": PRECISION,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "per_device_batch_size": spec.per_device_batch_size,
    }


def _check_single_axis(flag_sets: list[dict[str, Any]], effective: list[int]) -> None:
    """Prove the arms differ in the split axis and ONLY the split axis.

    SPEC: arms of one row differ in EXACTLY ONE axis. For T1-10 that axis is
    the accum x batch split of a fixed effective batch, so the differing keys
    must be a subset of SPLIT_AXIS, the split must actually move (a no-op arm
    pair measures nothing), and the effective batch must be identical -- a
    pair that also changed the batch would measure batch size, not
    accumulation.
    """
    reference = flag_sets[0]
    for flags in flag_sets[1:]:
        differing = {k for k in reference if reference[k] != flags[k]}
        extra = differing - set(SPLIT_AXIS)
        if extra:
            raise ValueError(f"arms differ outside the split axis: {sorted(extra)}")
        if not differing:
            raise ValueError("arms do not differ at all: a no-op pair measures nothing")
    if len(set(effective)) != 1:
        raise ValueError(f"effective batch differs across arms: {effective}")


def _missing_trainer_flags() -> list[str]:
    """Flags this row invokes that the shipped trainer CLI does not have.

    Checked against the parser the CLI actually builds, not assumed: SPEC's
    rule is that a row needing an axis with no flag REFUSES 96 and names it,
    and the only honest way to know is to look.
    """
    from foundationscale.train.cli import build_parser

    available: set[str] = set()
    for action in build_parser()._actions:  # noqa: SLF001 -- no public option-string API
        available.update(action.option_strings)
    needed = {
        "--model",
        "--dataset",
        "--output-dir",
        "--max-steps",
        "--learning-rate",
        "--seed",
        "--precision",
        "--nodes",
        "--gpus-per-node",
        "--profile-name",
        "--profile-path",
        "--gradient-accumulation-steps",
        "--per-device-batch-size",
        "--logging-steps",
    }
    return sorted(needed - available)


def _trainer_argv(spec: ArmSpec, args: argparse.Namespace, arm_out: Path) -> list[str]:
    """The shipped CLI's argv for one arm.

    Every flag here is verified against the real parser by
    ``_missing_trainer_flags`` before any arm runs -- checked, not assumed.
    """
    argv = [
        "--model",
        MODEL,
        "--dataset",
        DATASET,
        "--output-dir",
        os.fspath(arm_out),
        "--max-steps",
        str(args.max_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--precision",
        PRECISION,
        "--per-device-batch-size",
        str(spec.per_device_batch_size),
        "--gradient-accumulation-steps",
        str(spec.gradient_accumulation_steps),
        # 1: the deciding curve is loss-per-step; the trainer's DERIVED cadence
        # yields ~2 points in a short run and equivalence would rest on nothing.
        "--logging-steps",
        "1",
        "--nodes",
        str(args.nodes),
        "--gpus-per-node",
        str(args.gpus_per_node),
    ]
    if args.profile_name is not None:
        argv += ["--profile-name", args.profile_name]
    else:
        argv += ["--profile-path", os.fspath(args.profile_path)]
    return argv


# ---------------------------------------------------------------------------
# Reading the measurement -- telemetry, never stdout
# ---------------------------------------------------------------------------


def _find_manifest(arm_out: Path) -> tuple[Path, dict[str, Any]] | None:
    """Locate the run's RunManifest JSON under the arm's output dir, or None.

    By shape, not by guessed filename: a manifest is a JSON object carrying a
    ``telemetry`` mapping (API.md). Reading by shape keeps this row honest
    about what it consumed -- if nothing under the output dir is a manifest,
    that is UNMEASURED with the search described, not the wrong file read
    with confidence.
    """
    if not arm_out.is_dir():
        return None
    for candidate in sorted(arm_out.rglob("*.json")):
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("telemetry"), dict):
            return candidate, payload
    return None


def _is_number(value: Any) -> bool:
    """True for a real JSON number -- bool is excluded: it is a flag, not a reading."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar_entry(telemetry: dict[str, Any], key: str) -> dict[str, Any]:
    """One telemetry entry, record-ready; an absent key reads as unmeasured.

    A present ``unmeasured`` entry keeps its reason string verbatim (SPEC: the
    operator learns WHY without re-running); an absent key gets a reason that
    says it was absent, so the two holes read differently in the JSON.
    """
    entry = telemetry.get(key)
    if not isinstance(entry, dict):
        return {
            "source": "unmeasured",
            "value": f"manifest telemetry carries no {key!r} entry",
            "unit": None,
        }
    return {
        "source": entry.get("source", "unmeasured"),
        "value": entry.get("value"),
        "unit": entry.get("unit"),
    }


def _step_label(key: str) -> int | None:
    """The step index of a ``train_loss_step_<digits>`` key, or None.

    A series point is a key of that EXACT form. The bare ``train_loss`` is the
    run aggregate -- a different quantity that merely shares the prefix -- and
    a key with no trailing digits carries no step at all: an unparsable key is
    a refusal, never step zero. The digits are read as a number because an
    unpadded suffix does not sort as text ("step_10" < "step_2").
    """
    prefix = "train_loss_step_"
    if not key.startswith(prefix):
        return None
    suffix = key[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def _short_series_reason(found: int, lead: str) -> str:
    """Why a loss series of ``found`` points cannot decide this row.

    A one-point "curve" is the run aggregate wearing a curve's name, and
    equivalence claimed over one point is not evidence -- this is the false
    GREEN the prefix-match reader produced on real hardware. The knobs that
    add per-step points are ``logging_steps_effective`` (emit more often) and
    ``max_steps`` (run longer); no tolerance or adjudication can rescue a
    series that was never a curve.
    """
    return (
        f"{lead}: found {found} loss point(s); fewer than 2 is not a curve -- "
        "a single point is the run aggregate, not a curve, and agreement over "
        "one point is no evidence of equivalence. Emit more per-step points "
        "with a smaller logging_steps_effective or a larger max_steps."
    )


def _loss_curve(
    telemetry: dict[str, Any],
) -> tuple[list[int] | None, list[float] | None, str | None]:
    """The per-step loss series as ``(steps, losses, None)``, or ``(None, None, reason)``.

    The deciding metric of this row. ``train_loss_curve`` -- the manifest's
    own list of ``[step, loss]`` pairs -- is read FIRST when present, and a
    malformed one is a refusal naming the shape it actually had, never a
    silent skip back to the aggregate: an instrument that cannot tell the
    series from the run mean must say so. When the key is absent the older
    spellings remain, but the per-step branch accepts ONLY keys of the exact
    form ``train_loss_step_<digits>`` -- a prefix match is too wide a detector
    for this axis, because the bare ``train_loss`` aggregate merely shares it.
    That aggregate is still read in its one legitimate spelling, a single
    ``train_loss`` entry whose value is a LIST of numbers (positions are the
    steps); a bare SCALAR is the run mean and is never promoted to a
    one-point curve. An entry whose source is ``unmeasured`` is a statement:
    its value IS the reason and is returned verbatim.
    """
    curve_entry = telemetry.get("train_loss_curve")
    if curve_entry is not None:
        if isinstance(curve_entry, dict) and curve_entry.get("source") == "unmeasured":
            return None, None, str(curve_entry.get("value"))
        source = curve_entry.get("source") if isinstance(curve_entry, dict) else None
        if source not in ("measured", "derived"):
            return (
                None,
                None,
                (
                    f"train_loss_curve has source {source!r}, not 'measured' or "
                    "'derived' -- the deciding series was not measured"
                ),
            )
        value = curve_entry.get("value") if isinstance(curve_entry, dict) else None
        if not isinstance(value, list):
            return (
                None,
                None,
                (f"train_loss_curve value is not a list of [step, loss] pairs: {value!r}"),
            )
        steps: list[int] = []
        losses: list[float] = []
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None, None, (f"train_loss_curve point is not a [step, loss] pair: {point!r}")
            step, loss = point
            if not isinstance(step, int) or isinstance(step, bool):
                return None, None, f"train_loss_curve step is not an integer: {step!r}"
            if not _is_number(loss):
                return (
                    None,
                    None,
                    (f"train_loss_curve loss at step {step} is not a number: {loss!r}"),
                )
            steps.append(step)
            losses.append(float(loss))
        pairs = sorted(zip(steps, losses, strict=True))
        if len(pairs) < 2:
            return None, None, _short_series_reason(len(pairs), "train_loss_curve")
        return [s for s, _ in pairs], [v for _, v in pairs], None

    # Fallback spellings, only when train_loss_curve is absent. The wide
    # prefix filter is gone: per-step points are train_loss_step_<digits>
    # EXACTLY, and the bare train_loss is handled on its own terms below.
    step_keys = sorted(k for k in telemetry if _step_label(k) is not None)
    bare = telemetry.get("train_loss")
    if not step_keys and "train_loss" not in telemetry:
        return None, None, "manifest telemetry carries no train_loss* entries"
    if isinstance(bare, dict) and bare.get("source") == "unmeasured":
        return None, None, str(bare.get("value"))
    series: list[tuple[int, float]] = []
    for key in step_keys:
        entry = telemetry[key]
        if isinstance(entry, dict) and entry.get("source") == "unmeasured":
            return None, None, str(entry.get("value"))
        value = entry.get("value") if isinstance(entry, dict) else None
        label = _step_label(key)
        if label is None or not _is_number(value):
            return None, None, f"train_loss entry {key!r} is not a number: {value!r}"
        series.append((label, float(value)))
    if series:
        series.sort()
        if len(series) < 2:
            return None, None, _short_series_reason(len(series), "per-step loss entries")
        return [s for s, _ in series], [v for _, v in series], None

    value = bare.get("value") if isinstance(bare, dict) else None
    if isinstance(value, list) and value and all(_is_number(x) for x in value):
        losses = [float(x) for x in value]
        if len(losses) < 2:
            return None, None, _short_series_reason(len(losses), "the train_loss list")
        return list(range(len(losses))), losses, None
    if _is_number(value):
        # The defect this rewrite exists to kill: the bare scalar is the
        # trainer's AGGREGATE mean over the whole run, and reading it as the
        # step-0 point made a one-point "curve" that any two arms agreed on.
        return (
            None,
            None,
            _short_series_reason(1, "the bare train_loss scalar is the run aggregate mean"),
        )
    return None, None, f"the single train_loss entry is not numeric: {value!r}"


def _unmeasured_curve(reason: str) -> dict[str, Any]:
    """A loss_curve record entry that states WHY there is no curve."""
    return {"source": "unmeasured", "value": reason, "unit": None}


def _write_record(out_dir: Path, arm: str, record: dict[str, Any]) -> Path:
    """Write one arm's JSON record, ``<row>_<arm>.json``, into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ROW}_{arm}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def _run_arm(
    spec: ArmSpec, args: argparse.Namespace, arm_index: int
) -> tuple[dict[str, Any], ArmCurve]:
    """Run one arm through the shipped trainer and read its RunManifest.

    The record is built whether or not the arm measured anything: the JSON is
    where an UNMEASURED arm says WHY, so it is written in every case.
    ``arm_index`` selects this arm's rendezvous port among the concurrently
    launched arms -- derived, never random, so a failure stays reproducible.
    """
    import foundationscale

    arm_out = args.out_dir / f"{ROW}_{spec.name}_run"
    # The module form of torchrun, never a bare `torchrun` executable -- #128:
    # on this estate that resolves to a HOST script whose host-anaconda shebang
    # would swap the interpreter out from under this container.
    # No LOCAL_WORLD_SIZE / WORLD_SIZE / RANK / LOCAL_RANK is set anywhere: the
    # launcher is the genuine producer of that runtime evidence, and setting it
    # by hand would MANUFACTURE the very fact the run exists to observe (#375,
    # "equal by construction"). Arms run concurrently on one node, so the
    # rendezvous port is base + arm_index -- deterministic, never random.
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
        *_trainer_argv(spec, args, arm_out),
    ]
    record: dict[str, Any] = {
        "row": ROW,
        "arm": spec.name,
        "claim": CLAIM,
        "control_rule": CONTROL_RULE,
        "model": MODEL,
        "dataset": DATASET,
        "precision": PRECISION,
        "seed": args.seed,
        "flags": _arm_flags(spec, args),
        "trainer_cmd": cmd,
    }
    env = dict(os.environ)
    # The subprocess must import the SAME foundationscale this process resolved
    # -- an installed distribution or the checkout's src/ -- never whatever a
    # bare interpreter would happen to find first.
    package_parent = Path(foundationscale.__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join(
        [os.fspath(package_parent), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(cmd, env=env, check=False)
    # #171: torchrun FLATTENS the child's exit code -- this is the LAUNCHER's
    # code, and the trainer's declared code (95 unmeasured, 96 refused) is not
    # observable through it. A non-zero here can be a refusal OR a crash, so
    # it must read UNMEASURED, never RED.
    record["launcher_exit"] = completed.returncode
    if completed.returncode != GREEN:
        why = (
            f"launcher exited {completed.returncode}; the trainer's declared "
            "code is not observable through torchrun (#171)"
        )
        record["loss_curve"] = _unmeasured_curve(why)
        return record, ArmCurve(spec.name, None, None, why)
    found = _find_manifest(arm_out)
    if found is None:
        why = (
            f"trainer output dir {arm_out} was never created"
            if not arm_out.is_dir()
            else f"no JSON file with a telemetry mapping under {arm_out}"
        )
        record["loss_curve"] = _unmeasured_curve(why)
        return record, ArmCurve(spec.name, None, None, why)
    manifest_path, manifest = found
    record["manifest"] = os.fspath(manifest_path)
    telemetry = manifest.get("telemetry")
    if not isinstance(telemetry, dict):  # the finder matched on this; paranoid re-check
        why = "manifest telemetry section is not a mapping"
        record["loss_curve"] = _unmeasured_curve(why)
        return record, ArmCurve(spec.name, None, None, why)
    record["telemetry"] = {key: _scalar_entry(telemetry, key) for key in CONTEXT_TELEMETRY}
    steps, losses, reason = _loss_curve(telemetry)
    if steps is None or losses is None:
        why = reason or "loss series unreadable"
        record["loss_curve"] = _unmeasured_curve(why)
        arm_curve = ArmCurve(spec.name, None, None, why)
    else:
        record["loss_curve"] = {"source": "measured", "value": losses, "unit": None}
        arm_curve = ArmCurve(spec.name, tuple(steps), tuple(losses), None)
    return record, arm_curve


# ---------------------------------------------------------------------------
# Adjudicator -- pure, so --self-test exercises it with no GPU
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmCurve:
    """One arm's contribution: a labelled curve (step labels + losses), or the reason.

    The step labels ride beside the losses because the adjudicator aligns
    points by label, never by position: arms that never logged the same steps
    have nothing to compare, and arms that partially overlap are compared on
    the intersection only.
    """

    arm: str
    steps: tuple[int, ...] | None
    losses: tuple[float, ...] | None
    unmeasured: str | None


def adjudicate(curves: list[ArmCurve]) -> tuple[int, list[str]]:
    """Apply CONTROL_RULE to the arm curves; return (exit code, detail lines).

    The inverse row: AGREEMENT is the pass. Points are aligned by step label
    and compared on the intersection of the arms' step sets, whose size is
    stated in the verdict detail. A shared step whose losses differ by more
    than LOSS_ATOL + LOSS_RTOL * |reference| refutes the claim (RED). Three
    shapes can never pass: an arm with no measured curve (or with fewer than
    2 points -- a single point is the run aggregate, not a curve), an
    intersection of fewer than 2 shared steps, and a comparison whose every
    shared step is 0 -- step 0 is before the first optimizer update, so the
    arms agree there by construction whatever the accumulation knob did, and
    a GREEN from it would measure initialization, not equivalence.
    """
    if len(curves) < 2:
        return UNMEASURED, [f"need two arms to compare, got {len(curves)}"]
    lines: list[str] = []
    missing = [c for c in curves if c.losses is None or c.steps is None]
    for c in missing:
        lines.append(f"ARM {c.arm}: UNMEASURED -- {c.unmeasured}")
    if missing:
        return UNMEASURED, lines

    measured: list[tuple[str, tuple[int, ...], tuple[float, ...]]] = []
    for c in curves:
        if c.steps is None or c.losses is None:  # narrowed above; keeps the checker honest
            continue
        if len(c.steps) != len(c.losses):
            lines.append(
                f"ARM {c.arm}: {len(c.steps)} step labels for {len(c.losses)} "
                "losses -- a mis-aligned series cannot be compared"
            )
            return UNMEASURED, lines
        if len(c.losses) < 2:
            reason = _short_series_reason(len(c.losses), "loss series")
            lines.append(f"ARM {c.arm}: UNMEASURED -- {reason}")
            return UNMEASURED, lines
        measured.append((c.arm, c.steps, c.losses))

    ref_arm, ref_steps, ref_losses = measured[0]
    ref_by_step = dict(zip(ref_steps, ref_losses, strict=True))

    per_arm: list[tuple[str, dict[int, float], list[int]]] = []
    compared: set[int] = set()
    for arm, steps, losses in measured[1:]:
        by_step = dict(zip(steps, losses, strict=True))
        common = sorted(set(ref_by_step) & set(by_step))
        per_arm.append((arm, by_step, common))
        compared.update(common)

    # Non-trivial support: every compared step being 0 means the row never got
    # past initialization. Step 0 is before the first optimizer update, so the
    # same first batch gives the same forward pass in every arm no matter what
    # the accumulation knob did -- agreement there is equal BY CONSTRUCTION
    # and carries no evidence about accumulation. The guard reads the compared
    # set, never one arm's series, and never requires step 0: real trainers
    # log after the first update, so step 0 is usually absent and that absence
    # is fine.
    if compared and max(compared) == 0:
        return UNMEASURED, [
            "every compared step is 0; step 0 is before the first optimizer "
            "update, so the arms agree there by construction regardless of the "
            "accumulation setting -- the comparison measured initialization, "
            "not equivalence"
        ]

    deviations: list[tuple[float, int, str, float, float]] = []
    for arm, by_step, common in per_arm:
        if len(common) < 2:
            lines.append(
                f"ARM {arm}: step labels {sorted(by_step)} share {len(common)} "
                f"step(s) with reference ARM {ref_arm} step labels "
                f"{sorted(ref_by_step)} -- fewer than 2 shared steps cannot be "
                "compared point-by-point"
            )
            return UNMEASURED, lines
        for step in common:
            ref = ref_by_step[step]
            got = by_step[step]
            gap = abs(ref - got)
            deviations.append((gap, step, arm, ref, got))
            if gap > LOSS_ATOL + LOSS_RTOL * abs(ref):
                lines.append(
                    f"ARM {arm} step {step}: loss {got} vs reference {ref} -- "
                    f"gap {gap:.6g} exceeds {LOSS_ATOL} + {LOSS_RTOL}*|{ref}|; {CONTROL_RULE}"
                )
                return RED, lines
    worst_gap, worst_step, worst_arm, worst_ref, worst_got = max(deviations)
    lines.append(
        f"curves agree on {len(compared)} shared step(s) {sorted(compared)} "
        f"(the intersection of the arms' step labels); worst gap "
        f"{worst_gap:.6g} at ARM {worst_arm} step {worst_step} "
        f"({worst_got} vs {worst_ref})"
    )
    return GREEN, lines


# ---------------------------------------------------------------------------
# Self-test -- eighteen controls, each planted so it fires BOTH ways
# ---------------------------------------------------------------------------


def _self_test() -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    floor_controls("T1-10", record)

    base = (2.31, 1.74, 1.42, 1.21, 1.05)
    base_steps = (0, 1, 2, 3, 4)

    # C1: identical curves are the PASS here -- the inverse of T1-9/11/12.
    rc, _ = adjudicate(
        [ArmCurve("a", base_steps, base, None), ArmCurve("b", base_steps, base, None)]
    )
    record("C1 identical curves adjudicate GREEN", rc == GREEN, f"rc={rc}")

    # C2: the control rule's REFUTING observation -- a step beyond tolerance
    # must return 5. A self-test that only feeds passing data proves nothing.
    shifted = base[:4] + (base[4] * 1.5,)
    rc, lines = adjudicate(
        [ArmCurve("a", base_steps, base, None), ArmCurve("b", base_steps, shifted, None)]
    )
    record(
        "C2 a beyond-tolerance step adjudicates RED",
        rc == RED and any(CONTROL_RULE in line for line in lines),
        f"rc={rc}",
    )

    # C3: bf16-shaped wiggle INSIDE the noise floor still passes -- a floor
    # tight enough to fail honest noise would make the row cry wolf.
    wiggled = tuple(x * 1.005 for x in base)
    rc, _ = adjudicate(
        [ArmCurve("a", base_steps, base, None), ArmCurve("b", base_steps, wiggled, None)]
    )
    record("C3 within-tolerance wiggle adjudicates GREEN", rc == GREEN, f"rc={rc}")

    # C4: an unmeasured deciding metric is 95, reason string carried verbatim.
    reason = "loss logging gate abstained at step 0"
    rc, lines = adjudicate(
        [ArmCurve("a", base_steps, base, None), ArmCurve("b", None, None, reason)]
    )
    record(
        "C4 an unmeasured arm adjudicates UNMEASURED with its reason",
        rc == UNMEASURED and any(reason in line for line in lines),
        f"rc={rc}",
    )

    # C5: step labels, not positions, align points. An arm missing the tail is
    # compared on the steps it did log -- agreement on the 4 shared steps is
    # agreement on what was measured, and the detail must state how many steps
    # that was. (The old positional reader treated truncation itself as the
    # refutation; labelled steps make that reading meaningless.)
    rc, lines = adjudicate(
        [
            ArmCurve("a", base_steps, base, None),
            ArmCurve("b", base_steps[:4], base[:4], None),
        ]
    )
    record(
        "C5 a tail-truncated arm compares on the shared steps only",
        rc == GREEN and any("4 shared step" in line for line in lines),
        f"rc={rc}",
    )

    args = _build_parser().parse_args(["--out-dir", os.devnull])
    good = [_arm_flags(spec, args) for spec in ARMS]

    # C6: the shipped arm pair passes the single-axis proof.
    try:
        _check_single_axis(good, [spec.effective_batch for spec in ARMS])
    except ValueError as exc:
        record("C6 the shipped arm pair passes the single-axis proof", False, str(exc)[:80])
    else:
        record("C6 the shipped arm pair passes the single-axis proof", True, "split only")

    # C7: an extra differing axis is caught.
    bad = [good[0], {**good[1], "learning_rate": 9e-4}]
    try:
        _check_single_axis(bad, [4, 4])
    except ValueError as exc:
        record(
            "C7 an extra differing axis is caught",
            "outside the split axis" in str(exc),
            str(exc)[:80],
        )
    else:
        record("C7 an extra differing axis is caught", False, "no raise")

    # C8: a split that changes the effective batch is caught -- that pair would
    # measure batch size, not accumulation.
    off_batch = [good[0], {**good[1], "gradient_accumulation_steps": 2}]
    try:
        _check_single_axis(off_batch, [4, 8])
    except ValueError as exc:
        record(
            "C8 an unequal effective batch is caught",
            "effective batch" in str(exc),
            str(exc)[:80],
        )
    else:
        record("C8 an unequal effective batch is caught", False, "no raise")

    # C9: per-step loss entries read in STEP order -- "step_10" sorts before
    # "step_2" as text, so string order would silently scramble the curve.
    telemetry = {
        "train_loss_step_10": {"value": 1.1, "source": "measured", "unit": None},
        "train_loss_step_2": {"value": 1.5, "source": "measured", "unit": None},
        "train_loss_step_1": {"value": 2.0, "source": "measured", "unit": None},
    }
    steps9, losses9, reason9 = _loss_curve(telemetry)
    record(
        "C9 per-step losses read in step order",
        steps9 == [1, 2, 10] and losses9 == [2.0, 1.5, 1.1] and reason9 is None,
        f"steps={steps9} losses={losses9}",
    )

    # C10: an unmeasured loss entry hands its reason through untouched.
    telemetry_u = {
        "train_loss_step_1": {
            "value": "objective gate abstained",
            "source": "unmeasured",
            "unit": None,
        }
    }
    steps10, losses10, reason_u = _loss_curve(telemetry_u)
    record(
        "C10 an unmeasured loss entry yields its reason verbatim",
        steps10 is None and losses10 is None and reason_u == "objective gate abstained",
        f"{reason_u!r}",
    )

    # C-a: the manifest's own [step, loss] series, labels and all, is the
    # FIRST thing read -- before any older spelling.
    telemetry_curve = {
        "train_loss_curve": {
            "value": [[0, 2.9], [1, 2.7], [2, 2.4]],
            "source": "measured",
            "unit": None,
        }
    }
    steps_ca, losses_ca, reason_ca = _loss_curve(telemetry_curve)
    record(
        "C13 train_loss_curve reads as labelled points",
        steps_ca == [0, 1, 2] and losses_ca == [2.9, 2.7, 2.4] and reason_ca is None,
        f"steps={steps_ca} losses={losses_ca}",
    )

    # C-b MUST-FIRE: two arms holding nothing but the bare train_loss
    # aggregate hold the run MEAN, not a curve. This is the false GREEN the
    # real GB200 run returned off one point per arm; the fix must turn that
    # exact input into rc 95 with a reason naming the one-point problem.
    telemetry_agg = {"train_loss": {"value": 2.5, "source": "measured", "unit": None}}
    steps_b1, losses_b1, reason_b1 = _loss_curve(telemetry_agg)
    steps_b2, losses_b2, reason_b2 = _loss_curve(dict(telemetry_agg))
    rc, lines = adjudicate(
        [ArmCurve("a", None, None, reason_b1), ArmCurve("b", None, None, reason_b2)]
    )
    record(
        "C14 bare aggregates adjudicate UNMEASURED, never GREEN",
        rc == UNMEASURED
        and losses_b1 is None
        and steps_b1 is None
        and losses_b2 is None
        and steps_b2 is None
        and any("run aggregate" in line and "not a curve" in line for line in lines),
        f"rc={rc}",
    )

    # C-c MUST-FIRE: agreement at step 0 alone is equal BY CONSTRUCTION --
    # step 0 is before the first optimizer update, so identical models agree
    # there whatever the accumulation knob did. Assert NOT GREEN, and that the
    # reason says why.
    rc, lines = adjudicate(
        [
            ArmCurve("a", (0, 1, 2), (2.9, 2.7, 2.4), None),
            ArmCurve("b", (0, 5, 6), (2.9, 9.9, 9.8), None),
        ]
    )
    record(
        "C15 agreement only at step 0 adjudicates UNMEASURED",
        rc == UNMEASURED
        and rc != GREEN
        and any("before the first optimizer update" in line for line in lines),
        f"rc={rc}",
    )

    # C-d: when train_loss_curve is present the bare aggregate never enters
    # the series -- 99.0 must appear nowhere in the read.
    telemetry_both = {
        "train_loss_curve": {
            "value": [[0, 2.9], [1, 2.7]],
            "source": "measured",
            "unit": None,
        },
        "train_loss": {"value": 99.0, "source": "measured", "unit": None},
    }
    steps_cd, losses_cd, reason_cd = _loss_curve(telemetry_both)
    record(
        "C16 the bare aggregate never enters a train_loss_curve read",
        steps_cd == [0, 1]
        and losses_cd == [2.9, 2.7]
        and 99.0 not in losses_cd
        and reason_cd is None,
        f"steps={steps_cd} losses={losses_cd} reason={reason_cd}",
    )

    # C-e: disjoint step labels share nothing to compare on -- UNMEASURED,
    # with both arms' step sets named in the reason.
    rc, lines = adjudicate(
        [
            ArmCurve("a", (1, 2, 3), (2.9, 2.7, 2.4), None),
            ArmCurve("b", (4, 5, 6), (2.9, 2.7, 2.4), None),
        ]
    )
    record(
        "C17 disjoint step labels adjudicate UNMEASURED",
        rc == UNMEASURED and any("[1, 2, 3]" in line and "[4, 5, 6]" in line for line in lines),
        f"rc={rc}",
    )

    # C-f: partially overlapping labels are compared on their intersection,
    # and the verdict detail states the intersection's size.
    rc, lines = adjudicate(
        [
            ArmCurve("a", (0, 1, 2, 3), (2.9, 2.7, 2.4, 2.2), None),
            ArmCurve("b", (2, 3, 4, 5), (2.401, 2.199, 1.9, 1.8), None),
        ]
    )
    record(
        "C18 partial overlap compares on the stated intersection",
        rc == GREEN and any("2 shared step" in line and "intersection" in line for line in lines),
        f"rc={rc}",
    )

    # C11: argparse's native exit 2 is outside the contract (#387); the error()
    # override must make a mistyped flag REFUSE.
    rc = main(["--out-dir", os.devnull, "--no-such-flag"])
    record("C11 an unknown flag adjudicates REFUSE", rc == REFUSE, f"rc={rc}")

    # C12: run mode with no cluster profile refuses before touching anything.
    with tempfile.TemporaryDirectory(prefix="t1_10_selftest_") as tmp:
        rc = main(["--out-dir", tmp])
    record("C12 no cluster profile adjudicates REFUSE", rc == REFUSE, f"rc={rc}")

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(f"T1-10 self-test: {len(checks) - len(failed)}/{len(checks)} controls PASS")
    if failed:
        for name in failed:
            print(f"FAIL: {name}")
        return RED
    return GREEN


# ---------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit 96, not 2.

    #387: argparse's native exit 2 sits outside the 0/5/95/96 contract, so a
    launcher meeting a mistyped flag cannot classify it. A usage error is a
    refused run -- nothing was measured -- so it is REFUSE.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: REFUSE: {message}", file=sys.stderr)
        raise SystemExit(REFUSE)


def _build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="t1_10_accumulation_equivalence.py",
        description=(
            "T1-10: accum=4,bs=1 vs accum=1,bs=4 must give the SAME loss curve. "
            "Exit codes: 0 agree, 5 differ beyond tolerance, 95 unmeasured, 96 refused."
        ),
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run the adjudicator controls and exit; no GPU, no network, no model",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where the <row>_<arm>.json records and per-arm trainer outputs go",
    )
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42, help="the SAME seed is passed to every arm")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument(
        "--master-port-base",
        type=int,
        default=29610,
        help=(
            "first rendezvous port for the per-arm torchrun launchers; arm i "
            "uses base + i. Arms run concurrently on one node, and a derived "
            "port keeps a failure reproducible -- a random one would not."
        ),
    )
    p.add_argument("--profile-name", default=None, help="built-in ClusterProfile name for the tray")
    p.add_argument(
        "--profile-path", type=Path, default=None, help="ClusterProfile JSON for the tray"
    )
    return p


def _print_verdict(rc: int, lines: list[str]) -> None:
    for line in lines:
        print(line)
    print("=" * 72)
    if rc == GREEN:
        print(f"T1-10 VERDICT GREEN: {CLAIM} -- observed within tolerance")
    elif rc == RED:
        print(f"T1-10 VERDICT RED: {CONTROL_RULE}")
    elif rc == UNMEASURED:
        print("T1-10 VERDICT UNMEASURED: the deciding loss curve could not be measured")
    else:
        print("T1-10 VERDICT REFUSE: the row declined to run")


def _run(args: argparse.Namespace) -> int:
    if args.out_dir is None:
        print("REFUSE: --out-dir is required outside --self-test")
        return REFUSE
    if args.profile_name is None and args.profile_path is None:
        print("REFUSE: no cluster profile -- pass --profile-name or --profile-path")
        return REFUSE
    reason = _ensure_package_importable()
    if reason is not None:
        print(f"REFUSE: foundationscale is not importable -- {reason}")
        return REFUSE
    missing = _missing_trainer_flags()
    if missing:
        print(f"REFUSE: the shipped trainer CLI lacks flag(s) this row needs: {missing}")
        return REFUSE

    flag_sets = [_arm_flags(spec, args) for spec in ARMS]
    _check_single_axis(flag_sets, [spec.effective_batch for spec in ARMS])

    # The tray's HF cache is the only legal source (SPEC): offline mode turns a
    # cache miss into a fast failure the trainer reports, which this row then
    # adjudicates 95 -- never a slow download of the wrong artifact.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    lines: list[str] = []
    curves: list[ArmCurve] = []
    for spec_index, spec in enumerate(ARMS):
        arm_record, arm_curve = _run_arm(spec, args, spec_index)
        path = _write_record(args.out_dir, spec.name, arm_record)
        curves.append(arm_curve)
        if arm_curve.losses is None:
            lines.append(f"ARM {spec.name}: UNMEASURED -- {arm_curve.unmeasured} ({path})")
        else:
            lines.append(
                f"ARM {spec.name}: measured -- {len(arm_curve.losses)} loss points "
                f"at steps {list(arm_curve.steps or ())} ({path})"
            )
    # #171: no REFUSE short-circuit on the launcher code -- torchrun flattens
    # the child's code, so a refusal and a crash are indistinguishable, and a
    # non-zero arm reads UNMEASURED above (never RED). The row's own REFUSE
    # paths (bad CLI, no profile, importable package) still exit 96 in _run.
    rc, verdict_lines = adjudicate(curves)
    _print_verdict(rc, lines + verdict_lines)
    return rc


def main(argv: list[str]) -> int:
    floor_reason = python_floor_reason()
    if floor_reason is not None:
        print(f"T1-10 VERDICT REFUSE: {floor_reason}")
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
    except Exception:  # noqa: BLE001 -- the contract has no code for "crashed"
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, end="")
        if getattr(args, "out_dir", None) is not None:
            with contextlib.suppress(OSError):
                _write_record(args.out_dir, "error", {"row": ROW, "traceback": tb})
        print("=" * 72)
        print("T1-10 VERDICT RED: unexpected exception in the run body; traceback on stderr")
        return RED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
