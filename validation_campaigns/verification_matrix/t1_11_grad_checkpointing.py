#!/usr/bin/env python3
"""T1-11: gradient checkpointing must MOVE two numbers, not merely be accepted.

The claim under test is the pair, not either half: same model, same dataset, same
seed, ``--precision bf16`` on both arms, exactly one axis flipped -- with
checkpointing ON, peak memory DROPS and step time RISES. Recompute is a trade and
a trade has two sides: a row that checked only memory would pass a build that
saves activations and never recomputes them, and a row that checked only time
would pass one that recomputes and saves anyway. The failure this row exists to
catch is quieter than either -- the flag parses, is recorded in the manifest, and
reaches a plane that never wires it. That is the silent-fallback shape: every
other gate GREEN, the knob doing nothing. Peak memory and step time are the two
places a no-op cannot hide, so the control rule IS the verdict, encoded verbatim
as CONTROL_RULE and applied by the adjudicator: neither metric moving is RED,
never a pass with a caveat.

  ARM off   ``--gradient-checkpointing false`` -- activations held: peak memory
            high, step time low. The baseline.
  ARM on    ``--gradient-checkpointing true`` -- activations recomputed: peak
            memory lower, step time higher. The treatment.
  ARM probe ``--gradient-checkpointing false`` at HALF the row's
            --per-device-batch-size. Not a claim arm -- an instrument probe:
            checkpointing can only shrink the ACTIVATION share of the peak,
            so if halving the batch with the flag held off does not move the
            peak materially, activations never dominated it on this hardware
            and model, and a flat memory delta between the claim arms says
            nothing about the flag.

Both arms invoke the shipped trainer through ``python -m torch.distributed.run``
-- the module form, since the bare ``torchrun`` script on this estate carries a
host-anaconda shebang that swaps the interpreter out from under the container
(#128) -- running ``-m foundationscale.train.cli``, with the HF hub forced
OFFLINE -- model and dataset are already in the tray's
cache, a cache miss is UNMEASURED, and a download would measure the network
instead of the flag. Every flag passed exists in cli.py (the axis flag is its
``--gradient-checkpointing`` true/false tri-state -- checked, not assumed). The
measurement is read from each run's RunManifest ``telemetry`` section (API.md is
the authoritative shape), never scraped from stdout; an entry whose source is
``unmeasured`` is a statement, its reason string travels into this row's JSON
verbatim, and an unmeasured deciding metric adjudicates 95, never GREEN. Peak
memory further passes a measured sensitivity precondition: the probe arm's
flat peak means the peak is not activation-bound on this configuration, so a
memory delta of zero is no evidence at all and adjudicates 95, never RED --
an instrument cannot convict what it cannot see. Each arm writes ONE record,
``t1_11_<arm>.json``, into ``--out-dir``.

Two ways to run it:

  --self-test       exercises the adjudicator against synthetic arm records:
                    the claimed shape (asserts GREEN), the no-op shape the
                    control rule exists to refute (asserts RED -- a self-test
                    that only feeds passing data proves nothing), an unmeasured
                    deciding metric (asserts 95 with the reason carried
                    through), a refused arm, half-way and wrong-way movement,
                    the instrument-floor shape that read as a false RED on real
                    hardware (a real step-time rise beside an exactly -0.0%
                    memory delta, with the probe reporting a flat peak --
                    asserts 95), the claimed shape seen through a SENSITIVE
                    instrument (asserts GREEN -- a precondition that refuses
                    everything is not a fix), the single-axis assertion firing
                    both ways, and the measured default batch size. Pure
                    stdlib, no GPU, no network, under a second.

  --out-dir DIR     runs both arms and the half-batch probe on the tray.
  --profile-name P  The trainer additionally requires a cluster profile and
                    node/GPU counts; pass them through. No CUDA device is a
                    REFUSE (96), not a verdict.

Exit codes follow the repository's four-state contract: 0 claim upheld, 5 the
control rule refuted it, 95 the deciding quantity could not be measured, 96
refused to run at all. argparse's own exit 2 is the known out-of-contract hole
(#387), so the parser's ``error()`` is overridden to refuse with 96.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import NoReturn

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
    reduce_paired_rounds,
)

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

MODEL = "Qwen/Qwen2.5-1.5B"
DATASET = "fancyzhx/ag_news"
PRECISION = "bf16"

CLAIM = "checkpointing on: peak memory DROPS and step time RISES"

# The verdict rule, verbatim from SPEC.md. The adjudicator applies it and the
# self-test asserts this string never drifts from the contract.
CONTROL_RULE = (
    "neither peak memory nor step time moving = the flag is a no-op, "
    "the silent-fallback shape -> **RED**."
)

# (arm name, value for --gradient-checkpointing). Both arms state the flag
# EXPLICITLY: the CLI's omitted value is None (engine default, nothing claimed),
# and comparing "false" against an omission would be the #342 laundering shape
# in miniature.
_ARMS: tuple[tuple[str, str], ...] = (("off", "false"), ("on", "true"))

_PEAK_KEY = "peak_memory_allocated_bytes"
_SPS_KEY = "steps_per_second"
_DECIDING = (_PEAK_KEY, _SPS_KEY)
_GIB = float(1 << 30)

# The batch-size sensitivity probe's arm name. It is deliberately NOT in _ARMS:
# _ARMS is the pairwise claim the single-axis assertion guards, and the probe
# differs from arm "off" in exactly the axis the claim holds fixed (batch
# size) -- which is what makes it a probe of the memory INSTRUMENT, not of the
# flag. Only its peak memory is ever read; its step time carries no meaning.
_PROBE_ARM = "probe"

# Below 5% a delta reads as tray noise, not movement. The shapes this row
# separates are far apart: a no-op flag moves peak memory ~0 (allocator peaks
# are deterministic for a fixed configuration) while recompute costs roughly a
# third of step time, so 5% is never the deciding call between them.
_MOVE_TOLERANCE = 0.05


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

    ``checks/campaign_self_tests.py`` executes this file with the interpreter that is
    running the suite, and the launcher-contracts CI job runs that suite under a bare
    ``python3`` which has never had ``pip install -e .`` applied to it. An installed
    distribution therefore cannot be assumed, so this falls back to the checkout's own
    ``src/``. An environment where NEITHER works is UNMEASURED (95), never RED (5): a
    measurement that could not be taken is a different fact from one that disagreed, and
    collapsing the two is how an environment gap gets read as a defect in the tree.
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
# Command line -- argparse, with the #387 hole closed
# ---------------------------------------------------------------------------


class _RefusingParser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error -- the known out-of-contract hole (#387).

    A launcher reading this row has cases for 0/5/95/96 and nothing else, so a
    mistyped flag must REFUSE, not escape the contract as a 2.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(REFUSE, f"{self.prog}: error: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = _RefusingParser(
        prog="t1_11_grad_checkpointing",
        description=(
            "T1-11: run the shipped trainer twice, identical but for "
            "--gradient-checkpointing, plus a half-batch checkpointing-OFF "
            "sensitivity probe for the memory instrument, and read peak memory "
            "and step time out of each run's manifest telemetry. Exit codes: "
            "0 GREEN, 5 RED, "
            "95 UNMEASURED, 96 REFUSE."
        ),
    )
    p.add_argument(
        "--out-dir", type=Path, default=None, help="directory the two arm records land in"
    )
    p.add_argument("--seed", type=int, default=42, help="the SAME seed on both arms")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument(
        "--per-device-batch-size",
        type=int,
        default=16,
        help=(
            # The percent signs are DOUBLED because argparse interpolates every
            # help string through `%` before printing it. A bare "-0.0%" is a
            # malformed conversion and argparse raises ValueError('badly formed
            # help string') at add_argument time -- that is, at import of the
            # parser, so the module cannot even reach --self-test. It is printed
            # as a single % to the operator.
            "default 16 because it was MEASURED on GB200 with Qwen2.5-1.5B bf16, "
            "not guessed: at bs=1 checkpointing moved peak memory by exactly "
            "-0.0%% -- the activation term is a rounding error next to "
            "parameters, gradients and optimizer state, so the instrument could "
            "not see the flag it exists to measure -- while bs=16 peaked at "
            "18.01 GiB and dropped 30.9%%, and 18 GiB against a ~180 GiB device "
            "is the measured headroom, so 16 is sensitive AND safe. Halving "
            "must still leave a second distinct size for the sensitivity probe, "
            "so bs=1 is refused outright."
        ),
    )
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=(
            # Percent signs doubled -- argparse interpolates help through `%`.
            "how many INTERLEAVED rounds each arm runs (default %(default)s). The "
            "arm order rotates every round: a fixed order confounds order with "
            "arm identity, and it biases TOWARD the claim, because the second arm "
            "inherits the first's warm filesystem cache and so looks faster "
            "(#418). Two is the floor -- with one round there is no sign "
            "agreement to check, and a one-round GREEN is one sample. Each extra "
            "round costs one more trainer subprocess PER ARM."
        ),
    )
    p.add_argument(
        "--master-port-base",
        type=int,
        default=29611,
        help=(
            "base for each arm's torchrun rendezvous port: arm i uses base+i, "
            "deterministic, never random -- arms run concurrently on one node "
            "and a fixed port collides"
        ),
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--profile-name", help="passed through to the trainer (it requires one)")
    group.add_argument("--profile-path", type=Path, help="passed through to the trainer")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="exercise the adjudicator against synthetic records; no GPU, no trainer",
    )
    return p


# ---------------------------------------------------------------------------
# Arms -- two, differing in exactly one axis, asserted not intended
# ---------------------------------------------------------------------------


def _declaration(
    flag_value: str, *, seed: int, max_steps: int, per_device_batch_size: int
) -> dict[str, object]:
    """One arm's full declaration. Every fixed axis is named here so the single-axis
    assertion compares EVERYTHING the arms state -- not the one knob the author
    remembered to compare."""
    return {
        "model": MODEL,
        "dataset": DATASET,
        "precision": PRECISION,
        "seed": seed,
        "max_steps": max_steps,
        "per_device_batch_size": per_device_batch_size,
        "gradient_checkpointing": flag_value,
    }


def _assert_single_axis(declarations: list[dict[str, object]]) -> None:
    """Arms of this row differ in EXACTLY ONE axis -- asserted, not intended.

    A second accidental difference (a seed, a batch size) and the row measures
    nothing while still printing two numbers; SPEC makes the assert mandatory.
    """
    base = declarations[0]
    for other in declarations[1:]:
        assert set(base) == set(other), f"arm keys diverge: {set(base) ^ set(other)}"
        differing = sorted(k for k in base if base[k] != other[k])
        assert differing == ["gradient_checkpointing"], (
            "arms of T1-11 must differ in exactly ['gradient_checkpointing']; "
            f"they differ in {differing}"
        )


# ---------------------------------------------------------------------------
# Tray preconditions -- refuse (96) what cannot run, 95 what cannot be measured
# ---------------------------------------------------------------------------


def _gpu_refusal() -> str | None:
    """None when a CUDA device is present, else the reason this row refuses (96).

    The deciding metrics are GPU quantities; on a CPU-only box the run either
    crashes one layer down or, worse, produces numbers that say nothing about
    the flag. Refusing is the precondition-absent answer, not a verdict.
    """
    try:
        import torch
    except ImportError as exc:
        return f"torch is not importable ({exc}); the trainer cannot start"
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False; peak GPU memory is a deciding metric"
    return None


def _hf_hub_cache() -> Path:
    override = os.environ.get("HF_HUB_CACHE")
    if override:
        return Path(override)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cache_absences() -> list[str]:
    """Which fixed inputs are missing from the local HF cache (empty when warm).

    SPEC fixes model and dataset and forbids downloading: a miss here is 95
    UNMEASURED, not an error -- the tray was supposed to be warm, and warming it
    is not this row's job.
    """
    root = _hf_hub_cache()
    wanted = (
        ("models--Qwen--Qwen2.5-1.5B", MODEL),
        ("datasets--fancyzhx--ag_news", DATASET),
    )
    return [
        f"{repo} is not in the HF cache (looked for {root / dirname}); "
        "this row never downloads, so the run cannot be measured"
        for dirname, repo in wanted
        if not (root / dirname).is_dir()
    ]


# ---------------------------------------------------------------------------
# The trainer subprocess -- telemetry is the measurement, stdout is never read
# ---------------------------------------------------------------------------


def _trainer_env() -> dict[str, str]:
    """The trainer's environment: the hub forced OFFLINE (the tray's cache is the
    only legal source -- a download would make the row measure the network, not
    the flag), plus this checkout's ``src/`` on PYTHONPATH so the subprocess
    imports the same package the bootstrap found."""
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    src = _repo_src_root(Path(__file__).resolve().parent)
    if src is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            os.fspath(src) if not existing else os.fspath(src) + os.pathsep + existing
        )
    return env


def _trainer_argv(
    flag_value: str,
    args: argparse.Namespace,
    trainer_dir: Path,
    master_port: int,
    per_device_batch_size: int,
) -> list[str]:
    """Compose the launcher argv wrapping the shipped CLI. Every trainer flag below
    exists in cli.py -- checked against the shipped parser, not assumed.

    The trainer runs under ``torch.distributed.run`` AS A MODULE of this same
    interpreter: the bare ``torchrun`` executable on this estate resolves to a
    host script with a host-anaconda shebang, which would swap the interpreter
    out from under the container (#128). The launcher is the genuine producer of
    WORLD_SIZE/LOCAL_WORLD_SIZE/RANK/LOCAL_RANK -- they are never set by hand,
    because manufacturing them would silence the trainer's topology block with
    the very fact the run exists to observe (#375, "equal by construction").

    The batch size arrives as a parameter, not off ``args``: the sensitivity
    probe arm runs this same CLI at HALF the row's batch size, and a
    parameter is how one function serves both.
    """
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(args.nodes),
        "--nproc-per-node",
        str(args.gpus_per_node),
        "--master-port",
        str(master_port),
        "-m",
        "foundationscale.train.cli",
        "--model",
        MODEL,
        "--dataset",
        DATASET,
        "--output-dir",
        os.fspath(trainer_dir),
        "--max-steps",
        str(args.max_steps),
        "--per-device-batch-size",
        str(per_device_batch_size),
        "--seed",
        str(args.seed),
        "--precision",
        PRECISION,
        "--gradient-checkpointing",
        flag_value,
        # The loss curve is compared at the trainer's DERIVED logging cadence,
        # where 20 steps yield ~2 points; log every step so it is dense.
        "--logging-steps",
        "1",
        "--nodes",
        str(args.nodes),
        "--gpus-per-node",
        str(args.gpus_per_node),
        # #439: the trainer's Topology requires dp x tp x pp x ep x cp ==
        # nodes x gpus_per_node. This row emits no other degree, so all four take
        # the default of 1 (and are REFUSED above 1 -- #375) and dp IS the world
        # size. Omitting it left dp=1 against a width of 2, and every arm of pass
        # p414 refused "topology is not constructible (nothing touched)" before a
        # model was loaded -- which torchrun then reported as launcher rc 1 (#171).
        "--dp",
        str(args.nodes * args.gpus_per_node),
    ]
    if args.profile_name is not None:
        argv += ["--profile-name", args.profile_name]
    else:
        argv += ["--profile-path", os.fspath(args.profile_path)]
    return argv


def _find_manifest(trainer_dir: Path) -> Path | str:
    """Locate the RunManifest the trainer wrote, or return the reason it cannot be."""
    direct = trainer_dir / "run_manifest.json"
    if direct.is_file():
        return direct
    hits = sorted(trainer_dir.rglob("*manifest*.json")) if trainer_dir.is_dir() else []
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return f"no RunManifest found under {trainer_dir}"
    names = [os.fspath(h.relative_to(trainer_dir)) for h in hits]
    return f"ambiguous RunManifests under {trainer_dir}: {names}"


def _stderr_tail(stderr: str, lines: int = 8) -> str:
    return "\n".join(stderr.strip().splitlines()[-lines:])


def _run_arm(
    arm_index: int,
    name: str,
    flag_value: str,
    declaration: dict[str, object],
    per_device_batch_size: int,
    args: argparse.Namespace,
    env: dict[str, str],
    out_dir: Path,
    round_index: int = 0,
) -> dict[str, object]:
    """Run one arm and build its record. The measurement comes from the manifest's
    telemetry section; the subprocess's stdout is captured and deliberately
    unread, and stderr is kept only as failure diagnosis, never as a number.

    ``round_index`` scopes BOTH the trainer directory and the rendezvous port.
    The directory must be round-unique or ``_find_manifest`` would hand round 1
    the manifest round 0 left behind -- a rerun that silently re-reads the first
    round's numbers is worse than a crash, because every round would then agree.
    """
    suffix = f"{name}_r{round_index}"
    trainer_dir = out_dir / f"t1_11_{suffix}_trainer"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    # Per-arm rendezvous port: arms run concurrently on one node, so a fixed
    # port collides. base + zero-based index -- deterministic, never random,
    # so a failure stays reproducible. Rounds get disjoint blocks of
    # len(_ARMS) + 1 (both arms plus the probe) so a socket left in TIME_WAIT by
    # one round cannot refuse the next round's rendezvous.
    master_port = args.master_port_base + round_index * (len(_ARMS) + 1) + arm_index
    argv = _trainer_argv(flag_value, args, trainer_dir, master_port, per_device_batch_size)
    record: dict[str, object] = {
        "row": "T1-11",
        "arm": name,
        "round": round_index,
        "claim": CLAIM,
        "control_rule": CONTROL_RULE,
        "declaration": declaration,
        "trainer_argv": argv,
        "trainer_output_dir": os.fspath(trainer_dir),
        "launcher_exit_code": None,
        "telemetry": {},
        "loss_series": {},
        "run_failure": None,
        "refused": False,
    }
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
    # torchrun FLATTENS the child's exit code: any non-zero arrives here as 1
    # (#171), so this is the LAUNCHER's code and the trainer's declared verdict
    # code is not observable through it -- a refusal reads exactly like a crash.
    record["launcher_exit_code"] = proc.returncode
    found = _find_manifest(trainer_dir)
    if isinstance(found, str):
        record["run_failure"] = (
            f"launcher (torchrun) exited {proc.returncode} -- the trainer's own "
            "declared code is not observable through torchrun (#171) -- "
            f"and {found}; stderr tail:\n{_stderr_tail(proc.stderr)}"
        )
        # A genuine trainer REFUSE (96) would be flattened to 1 like any other
        # failure (#171); it can no longer be told apart, so this row never
        # marks a launcher code as refused: every non-zero adjudicates 95,
        # never RED.
        record["refused"] = False
        return record
    record["manifest_path"] = os.fspath(found)
    try:
        manifest = json.loads(found.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        record["run_failure"] = f"manifest at {found} is unreadable: {exc}"
        return record
    telemetry = manifest.get("telemetry") if isinstance(manifest, dict) else None
    if not isinstance(telemetry, dict):
        record["run_failure"] = (
            f"manifest at {found} carries no telemetry mapping; the deciding "
            "metrics live there and nowhere else"
        )
        return record
    record["telemetry"] = telemetry
    record["loss_series"] = {
        k: v for k, v in sorted(telemetry.items()) if k.startswith("train_loss")
    }
    if proc.returncode != 0:
        # The manifest is read above for the record, but this code is the
        # LAUNCHER's, not the trainer's declared verdict (#171). A refusal is
        # indistinguishable from a crash through torchrun, so this adjudicates
        # UNMEASURED (95) via run_failure -- never RED on a code that cannot
        # say what failed.
        record["run_failure"] = (
            f"launcher (torchrun) exited {proc.returncode} despite a readable "
            "manifest; the trainer's declared exit code is not observable "
            "through torchrun (#171). stderr tail:\n"
            f"{_stderr_tail(proc.stderr)}"
        )
    return record


def _write_record(path: Path, record: dict[str, object]) -> None:
    """ONE JSON file per arm -- the operator's evidence, written before the verdict
    so a RED or UNMEASURED row still leaves both arms' numbers behind."""
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _entry_brief(entry: object, scale: float, unit: str) -> str:
    if not isinstance(entry, dict):
        return "absent"
    if entry.get("source") == "unmeasured":
        return f"unmeasured({str(entry.get('value'))[:60]})"
    value = entry.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value * scale:.3f} {unit}"
    return f"non-numeric({str(value)[:40]})"


def _arm_line(name: str, flag_value: str, record: dict[str, object]) -> str:
    prefix = f"ARM {name:<3} (checkpointing={flag_value:<5})"
    failure = record.get("run_failure")
    if failure:
        return f"{prefix}: UNMEASURED -- {str(failure)[:110]}"
    telemetry = record.get("telemetry")
    peak = telemetry.get(_PEAK_KEY) if isinstance(telemetry, dict) else None
    sps = telemetry.get(_SPS_KEY) if isinstance(telemetry, dict) else None
    brief = f"peak={_entry_brief(peak, 1 / _GIB, 'GiB')} rate={_entry_brief(sps, 1.0, 'steps/s')}"
    launcher_rc = record.get("launcher_exit_code")
    return (
        f"{prefix}: {brief}  launcher_rc={launcher_rc} "
        f"(torchrun-flattened, not the trainer's declared code, #171)"
    )


# ---------------------------------------------------------------------------
# Adjudicator -- the control rule is the verdict
# ---------------------------------------------------------------------------


def _adjudicate(records: list[dict[str, object]]) -> tuple[int, list[str]]:
    """Apply the control rule to the two arm records, consulting a third probe
    record -- when one is present -- for the memory instrument's sensitivity
    floor before any verdict may rest on a flat memory delta. Pure: no I/O, no
    torch, no GPU -- the self-test drives it with synthetic records and a
    laptop sees the same verdicts the tray does."""
    by_arm = {str(r.get("arm")): r for r in records}
    values: dict[str, dict[str, float]] = {}
    for name, _flag in _ARMS:
        rec = by_arm.get(name)
        if rec is None:
            return UNMEASURED, [f"T1-11 VERDICT UNMEASURED: no record for arm '{name}'"]
        if rec.get("refused"):
            return REFUSE, [f"T1-11 VERDICT REFUSE: arm '{name}': {rec.get('run_failure')}"]
        if rec.get("run_failure"):
            return UNMEASURED, [f"T1-11 VERDICT UNMEASURED: arm '{name}': {rec['run_failure']}"]
        telemetry = rec.get("telemetry")
        if not isinstance(telemetry, dict):
            return UNMEASURED, [
                f"T1-11 VERDICT UNMEASURED: arm '{name}' carries no telemetry mapping"
            ]
        vals: dict[str, float] = {}
        for key in _DECIDING:
            entry = telemetry.get(key)
            if not isinstance(entry, dict):
                return UNMEASURED, [
                    f"T1-11 VERDICT UNMEASURED: arm '{name}' has no telemetry entry "
                    f"'{key}'; the deciding metric lives there and nowhere else"
                ]
            source = entry.get("source")
            if source == "unmeasured":
                # The reason string IS the value, by contract (API.md) -- carry it
                # verbatim so the operator learns WHY without re-running.
                return UNMEASURED, [
                    f"T1-11 VERDICT UNMEASURED: arm '{name}' reports {key} "
                    f"unmeasured: {entry.get('value')}"
                ]
            if source not in ("measured", "derived"):
                return UNMEASURED, [
                    f"T1-11 VERDICT UNMEASURED: arm '{name}' entry '{key}' has "
                    f"source {source!r}, not one of measured/derived/unmeasured"
                ]
            value = entry.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return UNMEASURED, [
                    f"T1-11 VERDICT UNMEASURED: arm '{name}' entry '{key}' is "
                    f"non-numeric: {value!r}"
                ]
            vals[key] = float(value)
        values[name] = vals

    mem_off = values["off"][_PEAK_KEY]
    mem_on = values["on"][_PEAK_KEY]
    sps_off = values["off"][_SPS_KEY]
    sps_on = values["on"][_SPS_KEY]
    if mem_off <= 0 or sps_off <= 0:
        return UNMEASURED, [
            "T1-11 VERDICT UNMEASURED: degenerate baseline "
            f"(peak={mem_off} bytes, {sps_off} steps/s); deltas would divide by zero"
        ]

    # The sensitivity probe is READ here, before any verdict may rest on a flat
    # memory delta -- but a broken probe is only REMEMBERED, not returned: it
    # matters exactly where the memory half of the claim is silent, and a
    # dropped peak already proves the instrument responded without it.
    probe = by_arm.get(_PROBE_ARM)
    probe_peak: float | None = None
    probe_failure: str | None = None
    if probe is not None:
        if probe.get("run_failure"):
            probe_failure = f"the probe arm failed: {probe['run_failure']}"
        else:
            probe_telemetry = probe.get("telemetry")
            probe_entry = (
                probe_telemetry.get(_PEAK_KEY) if isinstance(probe_telemetry, dict) else None
            )
            if not isinstance(probe_entry, dict):
                probe_failure = (
                    f"the probe arm has no telemetry entry '{_PEAK_KEY}'; the "
                    "instrument floor is read from it and nowhere else"
                )
            elif probe_entry.get("source") == "unmeasured":
                probe_failure = (
                    f"the probe arm reports peak memory unmeasured: {probe_entry.get('value')}"
                )
            else:
                probe_value = probe_entry.get("value")
                if isinstance(probe_value, bool) or not isinstance(probe_value, (int, float)):
                    probe_failure = f"the probe arm's peak memory is non-numeric: {probe_value!r}"
                else:
                    probe_peak = float(probe_value)

    mem_delta = (mem_on - mem_off) / mem_off
    sps_delta = (sps_on - sps_off) / sps_off
    lines = [
        f"DELTA peak memory: {mem_delta:+.1%} "
        f"({mem_off / _GIB:.2f} GiB -> {mem_on / _GIB:.2f} GiB)",
        f"DELTA step time:   {-sps_delta:+.1%} "
        f"(throughput {sps_off:.3f} -> {sps_on:.3f} steps/s; step time is the inverse)",
    ]
    mem_moved = abs(mem_delta) > _MOVE_TOLERANCE
    time_moved = abs(sps_delta) > _MOVE_TOLERANCE
    mem_dropped = mem_delta < -_MOVE_TOLERANCE
    time_rose = sps_delta < -_MOVE_TOLERANCE

    # Instrument-sensitivity precondition, measured -- never a hardcoded batch
    # threshold. Checkpointing shrinks only the ACTIVATION share of the peak;
    # parameters, gradients and optimizer state are untouched. When the probe
    # says halving the batch did not move the peak with the flag held OFF, the
    # peak is not activation-bound on this configuration, the benefit is below
    # the instrument's floor, and an absent memory drop carries no
    # information: that shape is 95, never 5 -- a false RED accuses a working
    # framework and invites somebody to "fix" correct code. Two paths are
    # exempt: a flat STEP TIME still refutes on its own (recompute costs ~a
    # third of step time at any batch size, so the time metric has no such
    # floor), and a memory delta beyond the noise floor means the instrument
    # demonstrably responded.
    if not mem_moved and time_rose and probe is not None:
        if probe_failure is not None or probe_peak is None:
            lines.append(
                "T1-11 VERDICT UNMEASURED: step time rose while peak memory "
                "stayed flat -- the shape of a working flag beneath the memory "
                "instrument's floor -- and the sensitivity probe that could "
                "rule that out is itself unmeasured "
                f"({probe_failure or 'no peak read'}). Raise "
                "--per-device-batch-size until the probe's two batch sizes "
                "move the peak, or the memory half of this claim cannot be "
                "adjudicated."
            )
            return UNMEASURED, lines
        probe_delta = (probe_peak - mem_off) / mem_off
        if abs(probe_delta) <= _MOVE_TOLERANCE:
            lines.append(
                "T1-11 VERDICT UNMEASURED: the memory instrument cannot carry "
                "this claim on the current configuration. With checkpointing "
                f"OFF the peak was {probe_peak / _GIB:.2f} GiB at half the "
                f"row's batch size and {mem_off / _GIB:.2f} GiB at the full "
                f"size ({probe_delta:+.1%}, inside the {_MOVE_TOLERANCE:.0%} "
                "noise floor), so the peak is not activation-bound and its "
                "failure to drop proves nothing. --per-device-batch-size is "
                "the knob that makes activations dominate the peak; a "
                "step-time rise WITHOUT a memory drop is consistent with a "
                "correctly wired flag whose benefit sits below this "
                "instrument's floor. Raise the batch size and re-run rather "
                "than 'fixing' code that may be correct."
            )
            return UNMEASURED, lines

    if mem_dropped and time_rose:
        lines.append(
            "T1-11 VERDICT GREEN: peak memory DROPPED and step time ROSE -- "
            "recompute is trading time for memory, so the flag is wired"
        )
        return GREEN, lines
    if not mem_moved and not time_moved:
        lines.append(f"CONTROL RULE: {CONTROL_RULE}")
        lines.append(
            "T1-11 VERDICT RED: neither deciding metric moved -- the flag parsed, "
            "was recorded, and changed nothing"
        )
        return RED, lines
    mem_word = "dropped as claimed" if mem_dropped else f"did not drop ({mem_delta:+.1%})"
    time_word = "rose as claimed" if time_rose else f"did not rise ({-sps_delta:+.1%})"
    lines.append(
        f"T1-11 VERDICT RED: the claim is the pair -- peak memory {mem_word}; step time {time_word}"
    )
    return RED, lines


def _deciding_values(record: object) -> dict[str, float] | None:
    """The two deciding numbers, or None when either is not a clean measurement.

    By the time the cross-round reducer runs, ``_adjudicate`` has already refused
    every shape this can reject. The guard stays anyway: a silent None here would
    reach ``reduce_paired_rounds`` as a fabricated sample, and the reducer cannot
    tell an invented number from a measured one.
    """
    if not isinstance(record, dict):
        return None
    telemetry = record.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    values: dict[str, float] = {}
    for key in _DECIDING:
        entry = telemetry.get(key)
        if not isinstance(entry, dict) or entry.get("source") not in ("measured", "derived"):
            return None
        value = entry.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values[key] = float(value)
    return values


# The two deciding metrics and the DIRECTION the claim asserts for each. Both are
# "lower": checkpointing is claimed to lower peak memory, and to lower throughput
# because step time is its inverse -- the row prints step time, but steps/s is
# what the manifest measures, so the reducer is fed the measured quantity and the
# claim is restated in its terms rather than the number being flipped.
_ROUND_METRICS: tuple[tuple[str, str, str], ...] = (
    (_PEAK_KEY, "lower", "peak memory DROPS"),
    (_SPS_KEY, "lower", "throughput FALLS, i.e. step time RISES"),
)


def _reduce_rounds(
    per_round: list[list[dict[str, object]]],
    tolerance: float = _MOVE_TOLERANCE,
) -> tuple[int, list[str]]:
    """Adjudicate every round, then require the ROUNDS to agree before a verdict.

    Each round goes through the untouched single-round rule, so every refusal
    that rule owns -- a refused arm, an unmeasured deciding metric, the memory
    instrument's sensitivity floor, the no-op CONTROL_RULE -- still fires on the
    first round that hits it. What is added is the cross-round gate: a GREEN or a
    RED that only one round agrees with is ONE SAMPLE (#418), and
    ``reduce_paired_rounds`` is the instrument that says so. The claim is a PAIR,
    so both metrics must survive the gate; a 95 on either sinks the pair, because
    half a pair is not a verdict on it.
    """
    if len(per_round) < 2:
        return REFUSE, [
            f"T1-11 VERDICT REFUSE: {len(per_round)} round(s) of records reached the "
            "reducer; fewer than two rounds cannot observe sign agreement, which is "
            "the whole instrument -- rerun with --rounds 2 or more"
        ]

    lines: list[str] = []
    round_codes: list[int] = []
    for round_index, records in enumerate(per_round):
        code, round_lines = _adjudicate(records)
        round_codes.append(code)
        lines.append(f"--- round {round_index} ---")
        lines.extend(round_lines)
        if code in (REFUSE, UNMEASURED):
            lines.append(
                f"T1-11 VERDICT {'REFUSE' if code == REFUSE else 'UNMEASURED'}: "
                f"round {round_index} could not be adjudicated, so the round set "
                "carries no agreement to check"
            )
            return code, lines

    if all(code == RED for code in round_codes):
        # Unanimity across rounds IS the cross-round agreement, so the paired
        # reducer is deliberately not consulted here. It measures MOVEMENT, and
        # the shape the control rule refutes hardest -- neither metric moving --
        # reaches the reducer as a straddle of +/-tolerance, which it reports as
        # 95. Letting that outrank a unanimous refutation would retire the row's
        # own SPEC sentence by accident.
        lines.append(f"CONTROL RULE: {CONTROL_RULE}")
        lines.append(
            f"T1-11 VERDICT RED: every one of {len(per_round)} interleaved rounds "
            "refuted the claim under the single-round rule, with the arm order "
            "rotated between them -- the refutation is not an order artefact"
        )
        return RED, lines

    samples: dict[str, dict[str, list[float]]] = {key: {} for key, _d, _l in _ROUND_METRICS}
    for round_index, records in enumerate(per_round):
        by_arm = {str(r.get("arm")): r for r in records}
        for name, _flag in _ARMS:
            values = _deciding_values(by_arm.get(name))
            if values is None:
                lines.append(
                    f"T1-11 VERDICT UNMEASURED: round {round_index} adjudicated "
                    f"cleanly but carries no readable deciding value for arm "
                    f"'{name}'; the cross-round statistic would have to invent one"
                )
                return UNMEASURED, lines
            for key, _direction, _label in _ROUND_METRICS:
                samples[key].setdefault(name, []).append(values[key])

    lines.append("=" * 72)
    verdicts: dict[str, int] = {}
    for key, direction, label in _ROUND_METRICS:
        reduced = reduce_paired_rounds("off", "on", samples[key], tolerance, direction)
        reduced_code = reduced.get("exit_code")
        verdicts[key] = reduced_code if isinstance(reduced_code, int) else UNMEASURED
        movements = reduced.get("round_movements_claim_direction")
        shown = ", ".join(f"{m:+.1%}" for m in movements) if isinstance(movements, list) else "n/a"
        lines.append(
            f"ROUNDS {key} ({label}): {reduced['state']} -- per-round movement "
            f"{shown}; signs agree={reduced.get('movement_signs_agree')} "
            f"sd={reduced.get('sd')} sem={reduced.get('sem')}"
        )
        lines.append(f"    {reduced['reason']}")

    if all(code == GREEN for code in verdicts.values()):
        lines.append(
            "T1-11 VERDICT GREEN: both halves of the pair moved as claimed in "
            f"EVERY one of {len(per_round)} interleaved rounds -- the flag is "
            "wired, and the agreement is what rules out a one-pass artefact"
        )
        return GREEN, lines
    if UNMEASURED in verdicts.values():
        unresolved = [key for key, code in verdicts.items() if code == UNMEASURED]
        lines.append(
            f"T1-11 VERDICT UNMEASURED: {unresolved} did not agree across rounds. "
            "The claim is the pair, and a half-pair is not a verdict on it; "
            "rounds that disagree in direction are the signature of shared "
            "per-pass drift, not of the flag (#418)."
        )
        return UNMEASURED, lines
    refuted = [key for key, code in verdicts.items() if code == RED]
    lines.append(
        f"T1-11 VERDICT RED: {refuted} moved the WRONG way in every round -- "
        "the claim is the pair, and every round agrees this half of it did not "
        "happen"
    )
    return RED, lines


# ---------------------------------------------------------------------------
# Self-test -- synthetic records, including shapes the control rule REFUTES
# ---------------------------------------------------------------------------


def _synth_entry(value: object, source: str = "measured", unit: str | None = None) -> dict:
    return {"key": "synthetic", "value": value, "source": source, "unit": unit}


def _synth_record(name: str, peak_bytes: float, steps_per_second: float) -> dict[str, object]:
    return {
        "row": "T1-11",
        "arm": name,
        "run_failure": None,
        "refused": False,
        "launcher_exit_code": 0,
        "telemetry": {
            _PEAK_KEY: _synth_entry(peak_bytes, unit="bytes"),
            _SPS_KEY: _synth_entry(steps_per_second, unit="steps/s"),
            "train_runtime_s": _synth_entry(100.0, unit="s"),
            "total_flos": _synth_entry(1.0e15, unit="flops"),
        },
    }


def _self_test() -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    floor_controls("T1-11", record)
    gpu_reachability_controls("T1-11", record)

    # C1: the claimed shape -- memory down 42%, step time up 35% -- is GREEN.
    code, _lines = _adjudicate(
        [_synth_record("off", 12.0e9, 10.0), _synth_record("on", 7.0e9, 6.5)]
    )
    record("C1 the claimed shape is GREEN", code == GREEN, f"rc={code}")

    # C2: the control rule's shape -- a no-op flag -- is REFUTED, and the verdict
    # carries the rule verbatim. A self-test that only feeds passing data proves
    # nothing; this is the control that proves the adjudicator can say RED.
    code, lines = _adjudicate(
        [_synth_record("off", 12.0e9, 10.0), _synth_record("on", 12.02e9, 10.03)]
    )
    record(
        "C2 the no-op shape is RED under the control rule",
        code == RED and any(CONTROL_RULE in line for line in lines),
        f"rc={code}",
    )

    # C3: an unmeasured deciding metric is 95, and the reason string survives verbatim.
    reason = "torch.cuda.max_memory_allocated unavailable: CUDA context was reset mid-run"
    broken = _synth_record("on", 7.0e9, 6.5)
    broken["telemetry"] = {
        _PEAK_KEY: _synth_entry(reason, source="unmeasured", unit="bytes"),
        _SPS_KEY: _synth_entry(6.5, unit="steps/s"),
    }
    code, lines = _adjudicate([_synth_record("off", 12.0e9, 10.0), broken])
    record(
        "C3 an unmeasured deciding metric is 95, reason verbatim",
        code == UNMEASURED and any(reason in line for line in lines),
        f"rc={code}",
    )

    # C4: memory alone moving is still RED -- the claim is the pair.
    code, lines = _adjudicate(
        [_synth_record("off", 12.0e9, 10.0), _synth_record("on", 7.0e9, 9.95)]
    )
    record(
        "C4 memory moving alone is still RED (the pair)",
        code == RED and any("pair" in line for line in lines),
        f"rc={code}",
    )

    # C5: memory moving the WRONG way is RED, not a pass on the time half.
    code, _lines = _adjudicate(
        [_synth_record("off", 12.0e9, 10.0), _synth_record("on", 14.0e9, 6.5)]
    )
    record("C5 memory moving the wrong way is RED", code == RED, f"rc={code}")

    # C6: a refused arm refuses the whole row (96), it does not measure it.
    refused = _synth_record("on", 7.0e9, 6.5)
    refused["refused"] = True
    refused["run_failure"] = "trainer exited 96: declaration rejected"
    code, _lines = _adjudicate([_synth_record("off", 12.0e9, 10.0), refused])
    record("C6 a refused arm refuses the whole row", code == REFUSE, f"rc={code}")

    # C7/C8: the single-axis assertion fires BOTH ways -- it accepts the real
    # arms and catches a second accidental difference (a seed), which is the
    # defect that would make the row measure nothing while printing two numbers.
    declarations = [
        _declaration(flag, seed=42, max_steps=20, per_device_batch_size=1) for _name, flag in _ARMS
    ]
    try:
        _assert_single_axis(declarations)
    except AssertionError as exc:
        record("C7 the real arms pass the single-axis assertion", False, str(exc)[:80])
    else:
        record(
            "C7 the real arms pass the single-axis assertion",
            True,
            "differ in ['gradient_checkpointing'] only",
        )
    tampered = [dict(declarations[0]), dict(declarations[1], seed=43)]
    try:
        _assert_single_axis(tampered)
    except AssertionError:
        record("C8 a second differing axis is caught", True, "AssertionError raised")
    else:
        record("C8 a second differing axis is caught", False, "no assertion")

    # C9: the rule string is the SPEC sentence verbatim -- drift here silently
    # changes what the row refutes.
    expected_rule = (
        "neither peak memory nor step time moving = the flag is a no-op, "
        "the silent-fallback shape -> **RED**."
    )
    record(
        "C9 CONTROL_RULE matches the SPEC sentence verbatim",
        expected_rule == CONTROL_RULE,
        CONTROL_RULE,
    )

    # C10/C11: the manifest finder answers both ways -- a reason string when
    # there is nothing to read, the path when the canonical name exists.
    with tempfile.TemporaryDirectory(prefix="t1_11_selftest_") as tmp:
        root = Path(tmp)
        empty = root / "empty"
        empty.mkdir()
        found_empty = _find_manifest(empty)
        record(
            "C10 a missing manifest is a reason, not a crash",
            isinstance(found_empty, str),
            str(found_empty)[:60],
        )
        real_dir = root / "real"
        real_dir.mkdir()
        (real_dir / "run_manifest.json").write_text("{}")
        found_real = _find_manifest(real_dir)
        record(
            "C11 the canonical manifest name is found",
            found_real == real_dir / "run_manifest.json",
            str(found_real)[-60:],
        )

    # C12: a usage error exits 96, not argparse's 2 -- the #387 hole stays closed.
    try:
        _build_parser().parse_args(["--definitely-not-a-flag"])
    except SystemExit as exc:
        record("C12 a usage error exits 96, not 2 (#387)", exc.code == REFUSE, f"code={exc.code}")
    else:
        record("C12 a usage error exits 96, not 2 (#387)", False, "no SystemExit")

    # C13: the MUST-FIRE shape measured on GB200 -- bs=1, peak memory delta
    # exactly -0.0% while step time rose +32.0% (recompute plainly ran), and
    # the sensitivity probe's peak FLAT across two batch sizes, i.e. the peak
    # was never activation-bound and the memory instrument was blind. This is
    # 95, never 5 and never 0 -- the control that would have caught today's
    # false RED.
    code, lines = _adjudicate(
        [
            _synth_record("off", 12.44 * _GIB, 3.031),
            _synth_record("on", 12.44 * _GIB, 2.060),
            _synth_record(_PROBE_ARM, 12.44 * _GIB, 3.031),
        ]
    )
    record(
        "C13 a real step-time rise + blind memory instrument is 95",
        code == UNMEASURED
        and code not in (RED, GREEN)
        and any("activation-bound" in line for line in lines),
        f"rc={code}",
    )

    # C14: the bs=16 shape from the same tray -- memory -30.9%, step time
    # +18.9%, and a probe whose peak MOVED with batch size -- stays GREEN: a
    # precondition that refuses everything is not a fix.
    code, _lines = _adjudicate(
        [
            _synth_record("off", 18.01 * _GIB, 1.538),
            _synth_record("on", 12.45 * _GIB, 1.247),
            _synth_record(_PROBE_ARM, 15.20 * _GIB, 1.538),
        ]
    )
    record(
        "C14 the claimed shape through a seeing instrument is GREEN",
        code == GREEN,
        f"rc={code}",
    )

    # C15: a memory drop WITHOUT a step-time rise is still RED even with the
    # probe proving the instrument sees -- the paired rule is untouched.
    code, lines = _adjudicate(
        [
            _synth_record("off", 18.01 * _GIB, 1.538),
            _synth_record("on", 12.45 * _GIB, 1.545),
            _synth_record(_PROBE_ARM, 15.20 * _GIB, 1.538),
        ]
    )
    record(
        "C15 memory dropping alone is still RED (the pair) with a live probe",
        code == RED and any("pair" in line for line in lines),
        f"rc={code}",
    )

    # C16: the shipped default batch size is the MEASURED one: 16 peaked at
    # 18.01 GiB on a ~180 GiB device (the measured headroom) and dropped
    # 30.9% under checkpointing, while 1 moved -0.0% -- sensitive AND safe.
    default_batch = _build_parser().parse_args([]).per_device_batch_size
    record(
        "C16 the default --per-device-batch-size is the measured 16",
        default_batch == 16,
        f"default={default_batch}",
    )

    # C17/C18: the capability floor FIRES, and does not fire spuriously. Both
    # arms are driven through the injectable import/smoke seams, because on this
    # machine the real tray is either healthy or absent -- without the seams the
    # refusing branches would be unreachable decoration, and an unreachable
    # precondition is indistinguishable from one that always passes.
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

    # C20: the plan ROTATES. Arm order in round 1 is the reverse of round 0, and
    # every arm runs exactly once per round -- that is what decouples order from
    # arm identity (#418).
    plan = interleaved_round_plan([name for name, _flag in _ARMS], rounds=3)
    order = [[name for round_index, name in plan if round_index == r] for r in range(3)]
    record(
        "C20 the interleaved plan rotates arm order between rounds (#418)",
        order[0] == ["off", "on"] and order[1] == ["on", "off"] and order[2] == ["off", "on"],
        f"{order}",
    )

    def _round(peak_off: float, sps_off: float, peak_on: float, sps_on: float) -> list[dict]:
        return [
            _synth_record("off", peak_off, sps_off),
            _synth_record("on", peak_on, sps_on),
            _synth_record(_PROBE_ARM, 15.20 * _GIB, sps_off),
        ]

    # C21: three rounds that all show the claimed shape are GREEN. A cross-round
    # gate that can only refuse is not a gate.
    code, _lines = _reduce_rounds(
        [
            _round(18.01 * _GIB, 1.538, 12.45 * _GIB, 1.247),
            _round(18.02 * _GIB, 1.529, 12.44 * _GIB, 1.251),
            _round(18.00 * _GIB, 1.544, 12.46 * _GIB, 1.243),
        ]
    )
    record("C21 three agreeing rounds are GREEN", code == GREEN, f"rc={code}")

    # C22: THE MUST-FIRE control for #418. Round 0 is cache-cold, so its step
    # time moves the WRONG way while rounds 1 and 2 show the claimed rise. The
    # single-round rule would have called round 0 RED and rounds 1-2 GREEN --
    # whichever one ran alone becomes the verdict. Disagreement is 95: it is the
    # signature of shared per-pass drift, not of the flag either way.
    code, lines = _reduce_rounds(
        [
            _round(18.01 * _GIB, 1.150, 12.45 * _GIB, 1.247),
            _round(18.02 * _GIB, 1.529, 12.44 * _GIB, 1.251),
            _round(18.00 * _GIB, 1.544, 12.46 * _GIB, 1.243),
        ]
    )
    record(
        "C22 rounds that disagree in direction are 95, never a coin-flip verdict",
        code == UNMEASURED and any("disagree" in line for line in lines),
        f"rc={code}",
    )

    # C23: the no-op shape stays RED across rounds. The reducer would call a
    # sub-tolerance movement a straddle (95); the row's own control rule outranks
    # it when EVERY round refutes, so the silent-fallback shape cannot be
    # laundered into an abstention by adding rounds.
    noop = [_round(12.00e9, 10.00, 12.02e9, 10.03) for _ in range(3)]
    code, lines = _reduce_rounds(noop)
    record(
        "C23 the no-op shape is RED in every round, and stays RED",
        code == RED and any(CONTROL_RULE in line for line in lines),
        f"rc={code}",
    )

    # C24: one round is not a measurement. --rounds 1 refuses 96 BEFORE any GPU
    # or trainer is touched, so the refusal costs nothing and cannot be mistaken
    # for a verdict.
    with tempfile.TemporaryDirectory(prefix="t1_11_rounds_") as tmp:
        one_round = _build_parser().parse_args(
            ["--out-dir", tmp, "--profile-name", "self-test", "--rounds", "1"]
        )
        code = _run(one_round)
    record("C24 --rounds 1 refuses 96 before touching a GPU", code == REFUSE, f"rc={code}")

    # C25: the default round count is the module's declared floor-plus, not a
    # local invention -- it comes from the shared helper, so it cannot drift
    # away from the reducer that consumes it.
    default_rounds = _build_parser().parse_args([]).rounds
    record(
        "C25 --rounds defaults to the shared DEFAULT_ROUNDS",
        default_rounds == DEFAULT_ROUNDS and default_rounds >= 2,
        f"default={default_rounds}",
    )

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print(f"T1-11 self-test: {len(checks) - len(failed)}/{len(checks)} controls PASS")
    if failed:
        for name in failed:
            print(f"FAIL: {name}")
        return RED
    return GREEN


# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    if args.out_dir is None:
        print("REFUSE: --out-dir is required; each arm writes t1_11_<arm>.json there")
        return REFUSE
    if args.profile_name is None and args.profile_path is None:
        print(
            "REFUSE: the shipped trainer requires --profile-name or --profile-path; "
            "this row passes one through and declares nothing itself"
        )
        return REFUSE
    if args.rounds < 2:
        print(
            f"REFUSE: --rounds {args.rounds} cannot observe sign agreement between "
            "the paired per-round deltas -- a one-round verdict is one sample, and "
            "with a single round the arm order cannot rotate, so order stays "
            "confounded with arm identity (#418). Pass 2 or more."
        )
        return REFUSE
    refusal = _gpu_refusal()
    if refusal is not None:
        print(f"REFUSE: {refusal}")
        return REFUSE
    # _gpu_refusal above answers the flat-out-absent question (no torch, no CUDA)
    # and that is this row's 96. The capability floor asks the deeper one and its
    # misses are 95 by contract: a device the driver counts but cannot run a
    # kernel on, an ABI-skewed wheel, a dead native library. None of those say
    # anything about gradient checkpointing, so none of them may read as RED.
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
    absences = _cache_absences()
    if absences:
        for absence in absences:
            print(f"UNMEASURED: {absence}")
        return UNMEASURED

    probe_batch = max(args.per_device_batch_size // 2, 1)
    if probe_batch == args.per_device_batch_size:
        # Half the size, floored to 1 -- but when that half IS the size, the
        # probe's two "batch sizes" coincide and it compares a point against
        # itself: a sensitivity check that cannot move. Refuse rather than
        # run a precondition whose answer is vacuous.
        print(
            "REFUSE: --per-device-batch-size "
            f"{args.per_device_batch_size} leaves the sensitivity probe no "
            "second batch size (half floors to the same value); raise it -- "
            "16 was measured on this hardware to make the peak "
            "activation-bound, so the memory instrument can see the flag"
        )
        return REFUSE

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    declarations = [
        _declaration(
            flag,
            seed=args.seed,
            max_steps=args.max_steps,
            per_device_batch_size=args.per_device_batch_size,
        )
        for _name, flag in _ARMS
    ]
    _assert_single_axis(declarations)
    env = _trainer_env()

    flags = dict(_ARMS)
    arm_names = [name for name, _flag in _ARMS]
    declaration_by_arm = dict(zip(arm_names, declarations, strict=True))
    arm_position = {name: index for index, name in enumerate(arm_names)}

    # The sensitivity probe: checkpointing OFF like arm "off", but at half the
    # batch size -- the ONE axis the claim holds fixed is the axis the probe
    # varies, on purpose. It is not in _ARMS, so the single-axis assertion
    # still covers the claim pair alone, and its rendezvous port is
    # len(_ARMS), distinct from both arms'. Only its peak memory feeds a
    # verdict; its step time is meaningless.
    #
    # It runs FIRST, and that ordering is this row's ONE global warmup. It pulls
    # the same model and dataset shards through the shared filesystem cache every
    # later arm reads, so no round pays the cold-cache cost alone -- and that cost
    # is exactly the shared per-pass drift the paired reducer exists to catch (a
    # cache-cold pass measured -20.1% against +21.7..+30.2% on this row's metric).
    # Warming on the probe is free: its step time already carries no meaning, so
    # spending its timing on the cache costs the row nothing it was going to read.
    probe_declaration = _declaration(
        "false",
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_batch_size=probe_batch,
    )
    probe_record = _run_arm(
        len(_ARMS), _PROBE_ARM, "false", probe_declaration, probe_batch, args, env, out_dir
    )
    _write_record(out_dir / f"t1_11_{_PROBE_ARM}.json", probe_record)
    arm_lines: list[str] = [_arm_line(_PROBE_ARM, "false", probe_record)]

    per_round: list[list[dict[str, object]]] = [[] for _ in range(args.rounds)]
    for round_index, name in interleaved_round_plan(arm_names, rounds=args.rounds):
        flag = flags[name]
        record = _run_arm(
            arm_position[name],
            name,
            flag,
            declaration_by_arm[name],
            args.per_device_batch_size,
            args,
            env,
            out_dir,
            round_index=round_index,
        )
        _write_record(out_dir / f"t1_11_{name}_r{round_index}.json", record)
        # The canonical per-arm evidence path stays where every downstream reader
        # already looks; the LAST round wins it. The per-round files beside it are
        # what make the cross-round statistic auditable rather than asserted.
        _write_record(out_dir / f"t1_11_{name}.json", record)
        per_round[round_index].append(record)
        arm_lines.append(f"r{round_index} {_arm_line(name, flag, record)}")

    # Every round is adjudicated against the SAME probe: the memory instrument's
    # sensitivity is a property of the configuration, not of the round, and
    # re-running the probe per round would buy a number nothing reads.
    for round_records in per_round:
        round_records.append(probe_record)

    code, verdict_lines = _reduce_rounds(per_round)
    for line in arm_lines:
        print(line)
    print("=" * 72)
    for line in verdict_lines:
        print(line)
    return code


def main(argv: list[str]) -> int:
    floor_reason = python_floor_reason()
    if floor_reason is not None:
        print(f"T1-11 VERDICT REFUSE: {floor_reason}")
        return REFUSE

    if not argv:
        print(__doc__)
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
            # Before the import probe, deliberately: the adjudicator is pure stdlib,
            # so CI measures it even on a box where the package cannot be imported.
            return _self_test()
        reason = _ensure_package_importable()
        if reason is not None:
            print(f"UNMEASURED: foundationscale is not importable -- {reason}")
            return UNMEASURED
        return _run(args)
    except Exception as exc:  # noqa: BLE001 -- an escaped exception is classified, never adjudicated
        code, reason = classify_boundary_exception(exc)
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, end="")
        record: dict[str, object] = {
            "row": "T1-11",
            "arm": "error",
            "claim": CLAIM,
            "control_rule": CONTROL_RULE,
            "status": "error",
            "reason": reason,
            "traceback": tb,
        }
        if args.out_dir is not None:
            try:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                _write_record(args.out_dir / f"t1_11_{record['arm']}.json", record)
            except OSError:
                pass
        print("=" * 72)
        print(
            f"T1-11 VERDICT {'UNMEASURED' if code == UNMEASURED else 'CANNOT_MEASURE'}: "
            f"unexpected {type(exc).__name__} escaped the run body: {exc}. "
            f"Classified ({code}) at the main() boundary -- {reason} -- with the "
            f"traceback on stderr; nothing was measured, so no verdict on the "
            "scientific claim is emitted."
        )
        return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
