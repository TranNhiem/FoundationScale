#!/usr/bin/env python3
"""T1-20: a text corpus trains end-to-end and the bytes move.

Two arms run the SAME script, model, corpus, step count, seed and adapter geometry. They
differ in exactly one axis: the run arm gets a positive learning rate, the null arm gets
``--learning-rate 0``. The null arm is the harder half of the claim. A run that only shows
weights moving under a nonzero rate says nothing, because weights that moved for any reason
at all -- or no reason -- would look identical without a control.

WHY THE CONTROL IS EXACT AND NOT APPROXIMATE. LoRA's ``lora_B`` is ZERO-INITIALISED, so the
effective weight delta B@A is bit-exactly zero before the first gradient arrives, and every
AdamW update term is scaled by the learning rate. At lr=0 there is no update term at all:
the expected norm is not "small", it is 0.0, exactly. This file therefore compares norms
against zero with no tolerance anywhere. An epsilon is precisely the slack that would hide
the failure this control exists to find -- a broken optimizer, a wash of sign-flips, an
all-ones update -- so there is no epsilon parameter to pass, and none appears below.

THE MEASUREMENT THIS ROW RESTS ON (already taken on a GB200 tray; recorded here so the row
is not an anecdote): model ``Qwen/Qwen2.5-1.5B``, dataset ``fancyzhx/ag_news``, 20 steps,
seed 42, LoRA rank 8 targeting ``q_proj`` and ``v_proj``. The run arm at lr=5e-5 moved 56 of
56 ``lora_B`` tensors, with Frobenius norms from 0.0080 to 0.0204. The null arm at lr=0 left
every one of the same 56 tensors at precisely 0.0.

THE EPISTEMIC LIMIT, STATED PLAINLY. A null arm that is all zeros cannot by itself tell
"lr=0 froze the weights" apart from "the save path always writes zeros" -- both leave the
same silent adapter on disk, and the second reading would make every differential row a
forgery. The run arm is what excludes that second reading: it is the identical code path
with one axis changed, and its save contains nonzero norms, so saving demonstrably records
real values. Conversely the run arm alone cannot attribute its movement to training, because
an always-moves pipeline looks the same without the null arm. NEITHER arm carries the claim
alone; the PAIR carries it. That is why this file adjudicates a pair of norm dictionaries
and never a single one.

Exit contract: 0 GREEN, 5 RED, 95 UNMEASURED, 96 CANNOT-MEASURE or REFUSE. Never 1 or 2. An
arm that did not complete is 95, never 5: an incomplete run refutes nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

# The row directory is a sibling import root, exactly as `python3 t1_20_...py`
# gives it. This row had NO boundary handler at all, so an escaping exception
# left main() and CPython exited 1 -- outside the four-state contract.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_interpreter_floor import classify_boundary_exception  # noqa: E402

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96


class _ContractParser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error and 2 is outside the contract (#387).

    A malformed command line is an unmet precondition, not a refuted claim, so it is 96.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        sys.stderr.write(f"[t1-20:refuse] the command line did not parse: {message}\n")
        raise SystemExit(REFUSE)


def adjudicate(
    run: Mapping[str, float],
    null: Mapping[str, float],
) -> tuple[int, dict[str, object]]:
    """Decide T1-20 from the two arms' ``lora_B`` norms. Pure: no I/O, no model, no GPU.

    Keeping this a pure function of two dictionaries is what lets ``--self-test`` exercise the
    REAL decision rather than a copy of it. A self-test that re-implements the rule it is
    checking can only ever agree with itself.
    """
    payload: dict[str, object] = {}

    # With no lora_B tensors there is no probe at all: nothing was adapted, so "moved" and
    # "still" are both undefined. That is an unmeasured row, not a refuted one.
    if not run:
        payload["reason"] = (
            "the run arm declared 0 lora_B tensors, so no module was adapted and there is "
            "nothing to measure"
        )
        return UNMEASURED, payload

    # The arms must have adapted the same tensors or there is nothing to compare: a
    # difference would then be attributable to the adapter, not to the learning rate, and
    # the differential collapses. This is an unmet precondition, never a refutation.
    if sorted(run) != sorted(null):
        payload["reason"] = (
            f"the arms declared different tensor sets: {len(run)} vs {len(null)} lora_B "
            "tensors. A difference between them would not be attributable to the learning "
            "rate alone."
        )
        return UNMEASURED, payload

    # Bit-exact, deliberately. lora_B starts at zero and every AdamW term is scaled by the
    # learning rate, so the null arm's expected norm is exactly 0.0 -- and "moved" for the
    # run arm is "> 0.0", with no epsilon to hide a small-but-real leak behind.
    moved = sorted(k for k, v in run.items() if v > 0.0)
    leaked = sorted(k for k, v in null.items() if v != 0.0)

    payload.update(
        {
            "tensors": len(run),
            "run_moved": len(moved),
            "null_nonzero": len(leaked),
            "run_norm_min": min(run.values()),
            "run_norm_max": max(run.values()),
            "null_norm_max": max(null.values()),
            "null_leak_examples": {k: null[k] for k in leaked[:3]},
        }
    )

    # The control moving is the worse failure, so it is reported first: at lr=0 there is no
    # update term at all, and a nonzero norm there is movement with no training behind it --
    # the save path wrote something training never produced.
    if leaked:
        payload["reason"] = (
            f"the lr=0 CONTROL moved {len(leaked)} of {len(null)} lora_B tensor(s). lora_B "
            "is zero-initialised and every AdamW update term is scaled by the learning "
            "rate, so a nonzero norm here is movement with no training behind it: the "
            "detector cannot attribute any of the run arm's movement to the objective."
        )
        return RED, payload
    if not moved:
        payload["reason"] = (
            f"the positive-lr arm moved 0 of {len(run)} lora_B tensor(s), so the probe "
            "cannot fire at all and the control's silence establishes nothing. The pair "
            "carries the claim and this half of the pair is missing."
        )
        return RED, payload

    payload["reason"] = (
        f"the run arm moved {len(moved)} of {len(run)} lora_B tensor(s) (norms "
        f"{min(run.values()):.6g} to {max(run.values()):.6g}) while the lr=0 control left "
        f"every one of the same {len(null)} tensors at exactly 0.0. Movement is "
        "attributable to the learning rate."
    )
    return GREEN, payload


def _read_b_norms(adapter: Path) -> dict[str, float]:
    """Frobenius norm of every ``lora_B`` tensor in a saved adapter."""
    import torch
    from safetensors import safe_open

    out: dict[str, float] = {}
    with safe_open(str(adapter), "pt") as handle:
        for key in handle.keys():  # noqa: SIM118 -- safe_open exposes keys(), not __iter__
            if ".lora_B." in key:
                out[key] = float(torch.linalg.norm(handle.get_tensor(key).float()))
    return out


def _run_arm(label: str, args: argparse.Namespace, learning_rate: float, out_dir: Path) -> int:
    """Train one arm as a subprocess. Returns the child's return code."""
    cmd = [
        sys.executable,
        "-m",
        "foundationscale.train.cli",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--output-dir",
        str(out_dir),
        "--nodes",
        "1",
        "--gpus-per-node",
        "1",
        "--dp",
        "1",
        "--profile-name",
        "local-single-node",
        "--max-steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--logging-steps",
        "5",
        "--per-device-batch-size",
        "1",
        "--adapter",
        "lora",
        "--adapter-rank",
        str(args.adapter_rank),
        # THE ENTIRE DIFFERENTIAL IS THIS ONE VALUE. Model, dataset, steps, seed, adapter,
        # rank and targets come from the same Namespace for either arm and are byte-for-byte
        # identical between the two command lines; only --learning-rate differs. If anything
        # else varied, a movement difference could be attributed to that instead of to the
        # learning rate, and the row would be unmeasurable by construction.
        "--learning-rate",
        str(learning_rate),
    ]
    # One flag per pattern: --adapter-target is action="append", and passing several values
    # space-separated makes argparse read the extras as positionals (a 96 refusal).
    for pattern in args.adapter_target:
        cmd += ["--adapter-target", pattern]

    print(f"[t1-20:arm] {label}: learning_rate={learning_rate!r}")
    return subprocess.run(cmd, check=False).returncode


def _measure(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir)
    # The run arm's rate is configurable; the null arm's rate is hardcoded to 0 on purpose.
    # A configurable null would let someone "tune the control" until both arms move, which
    # is the failure mode this row exists to catch.
    arms = {"run": args.learning_rate, "null": 0.0}
    norms: dict[str, dict[str, float]] = {}
    for label, learning_rate in arms.items():
        arm_dir = out_root / f"t1_20_{label}"
        rc = _run_arm(label, args, learning_rate, arm_dir)
        if rc != 0:
            print(
                f"[t1-20:unmeasured] the {label} arm exited {rc}, so it did not complete. "
                "An incomplete arm refutes nothing."
            )
            return UNMEASURED
        adapter = arm_dir / "final" / "adapter_model.safetensors"
        try:
            norms[label] = _read_b_norms(adapter)
        except Exception as exc:  # noqa: BLE001 -- any read failure is an absent measurement
            print(f"[t1-20:unmeasured] could not read the {label} arm's adapter: {exc}")
            return UNMEASURED

    rc, payload = adjudicate(norms["run"], norms["null"])
    print(json.dumps(payload, indent=1, sort_keys=True))
    print(f"[t1-20:{ {0: 'green', 5: 'red', 95: 'unmeasured', 96: 'refuse'}[rc] }] rc={rc}")
    return rc


# --- self-test ---------------------------------------------------------------------------
# Runs with no GPU, no model and no network: it drives the real adjudicate() over synthetic
# norm dictionaries. Every verdict branch has a control that MUST fire, so a clean run is the
# rule working rather than a rule that cannot fail.

_Q1 = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
_Q2 = "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight"
_V1 = "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"
_EXTRA = "base_model.model.model.layers.2.self_attn.k_proj.lora_B.weight"


def _self_test() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, got: int, want: int) -> None:
        checks.append((name, got == want, f"rc={got} want={want}"))

    green_run = {_Q1: 0.0204, _Q2: 0.008, _V1: 0.0121}
    green_null = {_Q1: 0.0, _Q2: 0.0, _V1: 0.0}

    check(
        "C1 GREEN: the run arm moved, the null arm is bit-exactly zero",
        adjudicate(green_run, green_null)[0],
        GREEN,
    )
    check(
        "C2 RED MUST-FIRE: the lr=0 control moved one tensor",
        adjudicate(green_run, {_Q1: 0.0, _Q2: 0.01, _V1: 0.0})[0],
        RED,
    )
    check(
        "C3 RED MUST-FIRE: the control moved by 1e-12 -- tiny but not zero. This is the "
        "control that proves no epsilon crept in anywhere in this file",
        adjudicate(green_run, {_Q1: 0.0, _Q2: 0.0, _V1: 1e-12})[0],
        RED,
    )
    check(
        "C4 RED MUST-FIRE: the run arm moved nothing, so the control's silence is the "
        "absence of a probe and establishes nothing",
        adjudicate(green_null, green_null)[0],
        RED,
    )
    check(
        "C5 95 MUST-FIRE: the arms declared different tensor-key sets, so a difference "
        "between them is not attributable to the learning rate",
        adjudicate({**green_run, _EXTRA: 0.003}, green_null)[0],
        UNMEASURED,
    )
    check(
        "C6 95 MUST-FIRE: the run arm declared zero lora_B tensors, so there is no probe at all",
        adjudicate({}, green_null)[0],
        UNMEASURED,
    )
    check(
        "C7 the claim is at least ONE tensor moved, not that every tensor did",
        adjudicate({_Q1: 0.013, _Q2: 0.0, _V1: 0.0}, green_null)[0],
        GREEN,
    )

    # OUTCOME: the payload reports the counts it actually decided on, as numbers, so a reader
    # cannot be shown a verdict without the quantities behind it.
    rc, payload = adjudicate(green_run, green_null)
    numbers_ok = (
        rc == GREEN
        and payload["tensors"] == 3
        and payload["run_moved"] == 3
        and payload["null_nonzero"] == 0
        and payload["null_norm_max"] == 0.0
        and payload["run_norm_min"] == 0.008
    )
    checks.append(("C8 the payload carries the counts the verdict rests on", numbers_ok, ""))

    failed = 0
    for name, ok, detail in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())
        failed += 0 if ok else 1
    print(f"  SELF-TEST DENOMINATOR: {len(checks) - failed} of {len(checks)} controls behaved")
    return GREEN if failed == 0 else RED


def build_parser() -> argparse.ArgumentParser:
    p = _ContractParser(prog="t1_20_text_movement_arms", description=__doc__)
    p.add_argument("--model", help="local path or id of the text model")
    p.add_argument("--dataset", help="the text corpus both arms train on")
    p.add_argument("--out-dir", help="directory to write the two arm output dirs into")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter-rank", type=int, default=8)
    p.add_argument(
        "--adapter-target",
        action="append",
        default=None,
        help=(
            "module-name suffix selecting the adapted modules; repeatable (e.g. "
            "--adapter-target q_proj --adapter-target v_proj). One flag per pattern: "
            "action=append means space-separating several values makes argparse read the "
            "extras as positionals, which is a 96 refusal"
        ),
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help=(
            "the RUN arm's learning rate. The null arm's rate is hardcoded to 0 and is "
            "deliberately not configurable: a tunable control can always be tuned into "
            "agreement"
        ),
    )
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return _self_test()
    missing = [
        name
        for name in ("model", "dataset", "out_dir", "adapter_target")
        if not getattr(args, name)
    ]
    if missing:
        print(
            "[t1-20:refuse] these are required to measure the row and none of them has a "
            f"defensible default: {sorted(missing)}. Paths and module patterns are estate "
            "facts, so this file takes them as arguments rather than assuming them."
        )
        return REFUSE
    try:
        return _measure(args)
    except Exception as exc:  # noqa: BLE001 - classified, never adjudicated (#417)
        import traceback

        traceback.print_exc()
        code, why = classify_boundary_exception(exc)
        name = "UNMEASURED" if code == UNMEASURED else "CANNOT_MEASURE"
        print(
            f"[t1-20:{name.lower()}] unexpected {type(exc).__name__} escaped the "
            f"measurement: {exc}; classified {name} ({code}): {why}"
        )
        return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - never 1, never 2, never 5
        import traceback

        traceback.print_exc()
        _code, _why = classify_boundary_exception(exc)
        raise SystemExit(_code) from None
