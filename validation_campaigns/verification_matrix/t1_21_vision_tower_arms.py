#!/usr/bin/env python3
"""T1-21: an image-bearing corpus trains and the VISION TOWER's weights move.

The declared control is the other half of the claim and the harder half: a text-only arm
through the same path, whose tower weights must NOT move. A run that only shows the tower
moving says nothing, because a tower that moved on every batch regardless of modality would
look identical.

WHY LORA IS THE INSTRUMENT AND NOT A SHORTCUT. Watching real weights move needs a full
fine-tune, and a full fine-tune of a VLM of this size does not fit: the bf16 weights plus the
fp32 Adam state are several times the free device memory on the trays this row can reach. The
row therefore sat unmeasured, which is the worst outcome -- a claim nobody can price. LoRA
turns it into an exact question instead. ``lora_B`` is ZERO-INITIALISED, so the effective
weight delta B@A is bit-exactly zero until a gradient arrives, and AdamW skips a parameter
whose ``grad`` is None before any decay is applied. So ``norm(B) > 0`` means "this module was
trained", with no tolerance to argue about -- which is strictly better evidence than comparing
two full weight tensors, where the answer depends on an epsilon somebody has to defend.

Both arms attach the SAME adapter to the SAME modules. A difference between them therefore
cannot be an adapter-placement artifact, only a difference in what reached the modules.

WHY THE ADAPTER ALSO TOUCHES ONE LANGUAGE MODULE. The adapter spans two groups: the vision
modules under test, and exactly one language module that is NOT under test. That module is
the control's control. A text-only arm whose vision LoRA is still zero proves nothing if the
arm never trained at all -- a crashed arm, a zero-step arm and a genuinely unmoved tower all
produce the same zeros. The language anchor moving is what makes the null result a
measurement. It is also a hard requirement rather than a nicety: measured on hardware, a
VISION-ONLY adapter leaves a text-only batch with no gradient path whatsoever and BOTH arms
died with ``RuntimeError: element 0 of tensors does not require grad and does not have a
grad_fn``.

GRADIENT CHECKPOINTING IS OFF ON PURPOSE. Pixel values never require grad, so a checkpointed
vision segment can silently produce no gradient at all. That would leave the tower unmoved in
the image arm and this file would report RED -- a fabricated refutation of a true claim.

Exit contract: 0 GREEN, 5 RED, 95 UNMEASURED, 96 CANNOT-MEASURE or REFUSE. Never 1 or 2. An
arm that did not complete is 95, never 5: an incomplete run refutes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96


class _ContractParser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error and 2 is outside the contract (#387).

    A malformed command line is an unmet precondition, not a refuted claim, so it is 96.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        sys.stderr.write(f"[t1-21:refuse] the command line did not parse: {message}\n")
        raise SystemExit(REFUSE)


def _module_of(tensor_name: str) -> str:
    """The module path a peft ``lora_B`` tensor belongs to.

    peft names a tensor ``<module path>.lora_B.<adapter>.weight``, so the module path is
    everything left of ``.lora_B.``. Splitting rather than stripping a fixed suffix keeps
    this correct when the adapter is not named ``default``.
    """
    return tensor_name.split(".lora_B.")[0]


def _matches(module: str, suffixes: Sequence[str]) -> bool:
    """peft selects target modules by name SUFFIX, so classification must use the same rule.

    Anything else would classify a module differently from the way peft chose it, and the
    groups would stop describing what was actually adapted.
    """
    return any(module.endswith(s) for s in suffixes)


def adjudicate(
    image: Mapping[str, float],
    text: Mapping[str, float],
    vision_targets: Sequence[str],
    language_targets: Sequence[str],
) -> tuple[int, dict[str, object]]:
    """Decide T1-21 from the two arms' ``lora_B`` norms. Pure: no I/O, no model, no GPU.

    Keeping this a pure function of two dictionaries is what lets ``--self-test`` exercise the
    REAL decision rather than a copy of it. A self-test that re-implements the rule it is
    checking can only ever agree with itself.
    """
    payload: dict[str, object] = {}

    # The arms must have adapted the same modules or there is nothing to compare: a
    # difference would then be attributable to the adapter, not to the images.
    if sorted(image) != sorted(text):
        payload["reason"] = (
            f"the arms adapted different module sets: {len(image)} vs {len(text)} tensors. "
            "A difference between them would not be attributable to the modality."
        )
        return REFUSE, payload

    vision: dict[str, float] = {}
    language: dict[str, float] = {}
    unclassified: list[str] = []
    ambiguous: list[str] = []
    for name in image:
        module = _module_of(name)
        in_v = _matches(module, vision_targets)
        in_l = _matches(module, language_targets)
        if in_v and in_l:
            ambiguous.append(module)
        elif in_v:
            vision[name] = image[name]
        elif in_l:
            language[name] = image[name]
        else:
            unclassified.append(module)

    if ambiguous:
        payload["reason"] = (
            f"{len(ambiguous)} module(s) match BOTH the vision and language targets, so the "
            f"group under test overlaps its own control: {sorted(ambiguous)[:3]}"
        )
        return REFUSE, payload
    if unclassified:
        payload["reason"] = (
            f"{len(unclassified)} adapted module(s) match neither target group, so the "
            f"adapter reached somewhere this row cannot account for: {sorted(unclassified)[:3]}"
        )
        return REFUSE, payload
    if not vision or not language:
        payload["reason"] = (
            f"both groups are required: {len(vision)} vision module(s) and {len(language)} "
            "language anchor(s). With no vision group there is no claim; with no language "
            "anchor a still tower in the control arm cannot be told from an arm that never "
            "trained."
        )
        return REFUSE, payload

    # Bit-exact, deliberately. lora_B starts at zero and an untouched parameter is skipped by
    # the optimizer, so ">= some epsilon" would only be a way to ignore real movement.
    moved = lambda d: sorted(k for k, v in d.items() if v > 0.0)  # noqa: E731
    img_v, img_l = moved(vision), moved(language)
    txt_v = moved({k: text[k] for k in vision})
    txt_l = moved({k: text[k] for k in language})

    payload.update(
        {
            "vision_modules": len(vision),
            "language_modules": len(language),
            "image_arm": {
                "vision_moved": len(img_v),
                "language_moved": len(img_l),
                "vision_worst_abs_b_norm": max(vision.values()),
            },
            "text_arm": {
                "vision_moved": len(txt_v),
                "language_moved": len(txt_l),
                "vision_worst_abs_b_norm": max(text[k] for k in vision),
            },
        }
    )

    # Checked BEFORE the claim, because it decides whether the control arm is evidence at
    # all. An arm that moved nothing did not train, and its still tower is an absence.
    if not txt_l:
        payload["reason"] = (
            "the text-only arm moved NOTHING, not even its language anchor, so it did not "
            "train. Its unmoved vision tower is the absence of a run, not evidence about "
            "the tower."
        )
        return UNMEASURED, payload
    if not img_v:
        payload["reason"] = (
            f"the image arm did not move any of the {len(vision)} vision-tower module(s). "
            "The claim that an image-bearing corpus moves the tower is refuted."
        )
        return RED, payload
    if txt_v:
        payload["reason"] = (
            f"the text-only CONTROL moved {len(txt_v)} vision-tower module(s). The image "
            "arm's movement is therefore not attributable to the images."
        )
        return RED, payload

    payload["reason"] = (
        f"the image arm moved {len(img_v)} of {len(vision)} vision-tower module(s); the "
        f"text-only control moved 0 of {len(vision)} while still moving its language anchor "
        f"{len(txt_l)} of {len(language)}, so that arm ran and the tower still did not move."
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


def _run_arm(
    label: str,
    args: argparse.Namespace,
    dataset: str,
    out_dir: Path,
    image_column: str | None,
) -> int:
    """Train one arm as a subprocess. Returns the child's return code."""
    cmd = [
        sys.executable,
        "-m",
        "foundationscale.train.cli",
        "--model",
        args.model,
        "--dataset",
        dataset,
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
        "--logging-steps",
        "5",
        "--per-device-batch-size",
        "1",
        "--adapter",
        "lora",
        "--adapter-rank",
        str(args.adapter_rank),
    ]
    # One flag per pattern: --adapter-target is action="append", and passing several values
    # space-separated makes argparse read the extras as positionals (a 96 refusal).
    for pattern in (*args.vision_target, *args.language_target):
        cmd += ["--adapter-target", pattern]

    env = dict(os.environ)
    if image_column is None:
        # Declaring an image column over a text-only corpus is refused 96 by design, so the
        # control arm must not merely point at a different file -- it must not declare one.
        env.pop("FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN", None)
    else:
        env["FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN"] = image_column

    print(f"[t1-21:arm] {label}: dataset={dataset} image_column={image_column!r}")
    return subprocess.run(cmd, env=env, check=False).returncode


def _measure(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir)
    arms = {
        "image": (args.image_dataset, args.image_column),
        "text": (args.text_dataset, None),
    }
    norms: dict[str, dict[str, float]] = {}
    for label, (dataset, column) in arms.items():
        arm_dir = out_root / f"t1_21_{label}"
        rc = _run_arm(label, args, dataset, arm_dir, column)
        if rc != 0:
            print(
                f"[t1-21:unmeasured] the {label} arm exited {rc}, so it did not complete. "
                "An incomplete arm refutes nothing."
            )
            return UNMEASURED
        adapter = arm_dir / "final" / "adapter_model.safetensors"
        try:
            norms[label] = _read_b_norms(adapter)
        except Exception as exc:  # noqa: BLE001 -- any read failure is an absent measurement
            print(f"[t1-21:unmeasured] could not read the {label} arm's adapter: {exc}")
            return UNMEASURED

    rc, payload = adjudicate(
        norms["image"], norms["text"], args.vision_target, args.language_target
    )
    print(json.dumps(payload, indent=1, sort_keys=True))
    print(f"[t1-21:{ {0: 'green', 5: 'red', 95: 'unmeasured', 96: 'refuse'}[rc] }] rc={rc}")
    return rc


# --- self-test ---------------------------------------------------------------------------
# Runs with no GPU, no model and no network: it drives the real adjudicate() over synthetic
# norm dictionaries. Every verdict branch has a control that MUST fire, so a clean run is the
# rule working rather than a rule that cannot fail.

_V = ("mlp.down_proj.linear",)
_L = ("language_model.layers.0.self_attn.q_proj",)
_VT = "base_model.model.model.vision_tower.encoder.layers.0.mlp.down_proj.linear.lora_B.weight"
_VT2 = "base_model.model.model.vision_tower.encoder.layers.1.mlp.down_proj.linear.lora_B.weight"
_LT = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"


def _self_test() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, got: int, want: int) -> None:
        checks.append((name, got == want, f"rc={got} want={want}"))

    green_i = {_VT: 0.017, _VT2: 0.014, _LT: 0.9}
    green_t = {_VT: 0.0, _VT2: 0.0, _LT: 0.8}

    check(
        "C1 GREEN: tower moves on images, not on text, control arm trained",
        adjudicate(green_i, green_t, _V, _L)[0],
        GREEN,
    )
    check(
        "C2 RED MUST-FIRE: the image arm left the tower at zero",
        adjudicate({_VT: 0.0, _VT2: 0.0, _LT: 0.9}, green_t, _V, _L)[0],
        RED,
    )
    check(
        "C3 RED MUST-FIRE: the control moved the tower too",
        adjudicate(green_i, {_VT: 0.01, _VT2: 0.0, _LT: 0.8}, _V, _L)[0],
        RED,
    )
    check(
        "C4 RED MUST-FIRE: the control moved only SOME of the tower -- still unattributable",
        adjudicate(green_i, {_VT: 0.0, _VT2: 1e-9, _LT: 0.8}, _V, _L)[0],
        RED,
    )
    check(
        "C5 95 MUST-FIRE: the control arm moved nothing at all, so it never trained",
        adjudicate(green_i, {_VT: 0.0, _VT2: 0.0, _LT: 0.0}, _V, _L)[0],
        UNMEASURED,
    )
    check(
        "C6 95 beats RED: an untrained control is unmeasured even when the tower is still",
        adjudicate({_VT: 0.0, _VT2: 0.0, _LT: 0.0}, {_VT: 0.0, _VT2: 0.0, _LT: 0.0}, _V, _L)[0],
        UNMEASURED,
    )
    check(
        "C7 96 MUST-FIRE: the arms adapted different module sets",
        adjudicate(green_i, {_VT: 0.0, _LT: 0.8}, _V, _L)[0],
        REFUSE,
    )
    check(
        "C8 96 MUST-FIRE: no vision group, so there is no claim to decide",
        adjudicate({_LT: 0.9}, {_LT: 0.8}, _V, _L)[0],
        REFUSE,
    )
    check(
        "C9 96 MUST-FIRE: no language anchor, so a still tower is indistinguishable from "
        "an arm that never ran",
        adjudicate({_VT: 0.017}, {_VT: 0.0}, _V, _L)[0],
        REFUSE,
    )
    check(
        "C10 96 MUST-FIRE: an adapted module matching neither group is unaccounted for",
        adjudicate(
            {**green_i, "base_model.model.model.audio_tower.x.lora_B.weight": 0.1},
            {**green_t, "base_model.model.model.audio_tower.x.lora_B.weight": 0.1},
            _V,
            _L,
        )[0],
        REFUSE,
    )
    check(
        "C11 96 MUST-FIRE: a module matching BOTH groups overlaps the test with its control",
        adjudicate(green_i, green_t, _V, (*_L, "mlp.down_proj.linear"))[0],
        REFUSE,
    )
    check(
        "C12 no epsilon is smuggled in: a tiny nonzero norm counts as MOVED",
        adjudicate({_VT: 1e-12, _VT2: 1e-12, _LT: 0.9}, green_t, _V, _L)[0],
        GREEN,
    )
    check(
        "C13 exactly 0.0 is NOT moved -- the boundary is zero, not near-zero",
        adjudicate({_VT: 0.0, _VT2: 1e-12, _LT: 0.9}, green_t, _V, _L)[0],
        GREEN,
    )
    check(
        "C14 the claim is that the tower MOVES, not that every module does",
        adjudicate({_VT: 0.017, _VT2: 0.0, _LT: 0.9}, green_t, _V, _L)[0],
        GREEN,
    )

    # OUTCOME: the payload reports the counts it actually decided on, as numbers, so a reader
    # cannot be shown a verdict without the quantities behind it.
    rc, payload = adjudicate(green_i, green_t, _V, _L)
    numbers_ok = (
        rc == GREEN
        and payload["vision_modules"] == 2
        and payload["language_modules"] == 1
        and payload["image_arm"]["vision_moved"] == 2  # type: ignore[index]
        and payload["text_arm"]["vision_moved"] == 0  # type: ignore[index]
        and payload["text_arm"]["vision_worst_abs_b_norm"] == 0.0  # type: ignore[index]
    )
    checks.append(("C15 the payload carries the counts the verdict rests on", numbers_ok, ""))

    failed = 0
    for name, ok, detail in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())
        failed += 0 if ok else 1
    print(f"  T1-21 self-test: {len(checks) - failed}/{len(checks)} controls PASS")
    return GREEN if failed == 0 else RED


def build_parser() -> argparse.ArgumentParser:
    p = _ContractParser(prog="t1_21_vision_tower_arms", description=__doc__)
    p.add_argument("--model", help="local path or id of the vision-language model")
    p.add_argument("--image-dataset", help="jsonl carrying an image column")
    p.add_argument("--text-dataset", help="the same rows with the image column removed")
    p.add_argument("--image-column", default="image")
    p.add_argument("--out-dir", help="directory to write the two arm output dirs into")
    p.add_argument(
        "--vision-target",
        action="append",
        default=None,
        help=(
            "module-name SUFFIX that selects vision-tower modules and nothing else; "
            "repeatable. Verify uniqueness at the MODULE level, not the tensor level -- a "
            "suffix can be unique among checkpoint keys while a wrapper module elsewhere "
            "still carries the same name"
        ),
    )
    p.add_argument(
        "--language-target",
        action="append",
        default=None,
        help=(
            "module-name suffix for the ONE language module that anchors the control arm. "
            "Not under test; it exists so that a still tower in the text arm can be told "
            "apart from an arm that never trained"
        ),
    )
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--adapter-rank", type=int, default=8)
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return _self_test()
    missing = [
        name
        for name in (
            "model",
            "image_dataset",
            "text_dataset",
            "out_dir",
            "vision_target",
            "language_target",
        )
        if not getattr(args, name)
    ]
    if missing:
        print(
            "[t1-21:refuse] these are required to measure the row and none of them has a "
            f"defensible default: {sorted(missing)}. Paths and module patterns are estate "
            "facts, so this file takes them as arguments rather than assuming them."
        )
        return REFUSE
    return _measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
