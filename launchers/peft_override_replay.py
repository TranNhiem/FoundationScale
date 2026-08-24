#!/usr/bin/env python3
"""fix42 — peft override REPLAY probe (#73).

Positive evidence, in the code, that the peft.* overrides the launcher ships
are the values composition actually resolves. The #73 incident: the recipe
DID set cfg.peft, but `_is_omegaconf_problematic` tested callable() before
dataclasses.is_dataclass(), so the whole peft subtree was dropped from the
OmegaConf view and every peft.* override died in Hydra composition. The repair
is a one-branch reorder of that shared predicate; this probe is the run-time
control that the repaired channel keeps carrying what is shipped.

Doctrine 3 is why this exists as a program and not a comment. On the shipped
configuration the recipe defaults equal the launcher's defaults at 4 of 4
knobs, so an undrilled CLEAR can never distinguish "override landed" from
"default sat there" — a MUST_FIRE exists (env knob FS_PEFT_DRILL_RANK, armed
in the launcher) which perturbs dim off both defaults, and this probe prints
the resolved values so the launcher can demand the DRILL value by name. A
probe that cannot also CLEAR on a healthy tree is half a proof; the undrilled
PROBE run is this probe's MUST_PASS twin.

Method, and it matters: the launcher passes the byte-identical $CLI_OVERRIDES
token stream this same shell passes to the training command, and this probe
drives the REAL `process_config_with_overrides` with it — the oracle is the
composition that training will perform, not a re-implementation (the fix39
lesson: oracles that merely resemble the decision procedure certify defects).
Evidence ends at composition, stated honestly in the verdict line: whether the
model builder reads the resolved dim is G2/G3's question, not this probe's.

Verdict vocabulary, shipped and unrenumbered: 0 CLEAR / 1 BLOCKED /
3 UNMEASURED. Exactly one REPLAY_VERDICT= line is printed on every path,
because torchrun is measured to launder any nonzero child exit to 1 (fix40
receipt) — the launcher keys on the line and treats rc as one corroborating
bit. Denominators are printed, not implied: knobs_checked=4, and the replayed
override count is named in the verdict.
"""

import argparse
import dataclasses
import sys

KNOBS = ("dim", "alpha", "dropout", "target_modules")


def _parse_args(argv: "list[str]") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the real config composition and report the resolved peft knobs."
    )
    parser.add_argument(
        "--recipe", required=True, help="Recipe function name in megatron.bridge.recipes"
    )
    parser.add_argument(
        "--peft_scheme", required=True, help="Value run_recipe forwards as the `peft` kwarg"
    )
    parser.add_argument("--hf_path", required=True, help="Local HF source path (recipe kwarg)")
    parser.add_argument(
        "--seq_length", required=True, type=int, help="Sequence length (recipe kwarg)"
    )
    parser.add_argument("--expect-dim", dest="expect_dim", required=True, type=int)
    parser.add_argument("--expect-alpha", dest="expect_alpha", required=True, type=int)
    parser.add_argument("--expect-dropout", dest="expect_dropout", required=True, type=float)
    parser.add_argument(
        "--expect-targets-csv",
        dest="expect_targets_csv",
        required=True,
        help="Comma-joined shipped LORA_TARGETS (single-quoted by the launcher against globbing)",
    )
    # MUST be the last option: it swallows the byte-identical CLI_OVERRIDES tail.
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args(sys.argv[1:])

    expected_targets = [t for t in args.expect_targets_csv.split(",") if t]
    if not expected_targets:
        # doctrine 1: a probe over zero targets must abstain and name the 0.
        print("REPLAY_VERDICT=UNMEASURED (empty --expect-targets-csv: 0 targets offered; "
              "a sweep over 0 units never CLEARs)")
        return 3
    if not args.overrides:
        print("REPLAY_VERDICT=UNMEASURED (0 override tokens offered to the replay; "
              "evidence over 0 overrides never reads as a pass)")
        return 3

    try:
        import megatron.bridge.recipes as recipes
        from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
    except Exception as e:  # the failure shape IS the report
        print(f"REPLAY_VERDICT=UNMEASURED (imports failed: {type(e).__name__}: {e})")
        return 3

    build = getattr(recipes, args.recipe, None)
    if build is None or not callable(build):
        print(f"REPLAY_VERDICT=UNMEASURED (recipe '{args.recipe}' not registered in "
              "megatron.bridge.recipes — composition never ran)")
        return 3

    try:
        # Mirrors run_recipe.load_recipe's fixed forward set for this recipe:
        # peft_scheme -> peft, hf_path, seq_length. Nothing else is forwarded.
        cfg = build(peft=args.peft_scheme, hf_path=args.hf_path, seq_length=args.seq_length)
    except Exception as e:
        print(f"REPLAY_VERDICT=UNMEASURED (recipe construction failed: {type(e).__name__}: {e})")
        return 3

    pre_peft = getattr(cfg, "peft", None)
    if pre_peft is None:
        print(f"REPLAY_VERDICT=BLOCKED (recipe '{args.recipe}' built cfg.peft=None under "
              f"--peft_scheme {args.peft_scheme}: 4 shipped peft.* overrides have nothing to "
              "address — the recipe/forwarding arm of #73 (diagnosis A) is live again)")
        return 1
    if not dataclasses.is_dataclass(pre_peft):
        print(f"REPLAY_VERDICT=UNMEASURED (cfg.peft is {type(pre_peft).__name__}, expected a "
              "dataclass transform; the mechanism this probe measures has drifted)")
        return 3

    pre = {k: getattr(pre_peft, k, None) for k in KNOBS}

    try:
        process_config_with_overrides(cfg, cli_overrides=list(args.overrides))
    except Exception as e:
        print(f"REPLAY_VERDICT=BLOCKED (composition raised on the shipped override set of "
              f"n={len(args.overrides)}: {type(e).__name__}: {e})")
        return 1

    post_obj = getattr(cfg, "peft", None)
    shape_intact = (
        post_obj is not None and dataclasses.is_dataclass(post_obj) and callable(post_obj)
    )
    post = {k: getattr(post_obj, k, None) for k in KNOBS}

    expected = {
        "dim": args.expect_dim,
        "alpha": args.expect_alpha,
        "dropout": args.expect_dropout,
        "target_modules": expected_targets,
    }
    mismatches = []
    for knob, want in expected.items():
        got = post.get(knob)
        if got != want:
            mismatches.append(f"{knob}: shipped={want!r} resolved={got!r}")

    # Discrimination evidence: did composition CHANGE anything, or re-derive the
    # defaults? The launcher only demands 1 when the FS_PEFT_DRILL_RANK drill
    # has perturbed a default; undrilled runs print this as a stated limit.
    discriminating = 1 if any(post[k] != pre[k] for k in KNOBS) else 0
    print(f"REPLAY_KNOB_DISCRIMINATING={discriminating} (pre-composition dim={pre['dim']} "
          f"alpha={pre['alpha']} dropout={pre['dropout']} targets={pre['target_modules']})")
    print(f"REPLAY_PEFT dim={post['dim']} alpha={post['alpha']} dropout={post['dropout']} "
          f"targets={post['target_modules']} type={type(post_obj).__name__} "
          f"transform_intact={shape_intact} knobs_checked={len(KNOBS)}")

    if not shape_intact:
        print(f"REPLAY_VERDICT=BLOCKED (composition returned but peft lost its transform "
              f"shape: type={type(post_obj).__name__} callable={callable(post_obj)} — a "
              "dict-shaped peft would attach nothing downstream)")
        return 1
    if mismatches:
        print(f"REPLAY_VERDICT=BLOCKED ({len(mismatches)} of {len(KNOBS)} shipped knobs did "
              f"not resolve: {'; '.join(mismatches)} — the silent-revert class is firing, "
              "or the override never landed)")
        return 1

    print(f"REPLAY_VERDICT=CLEAR ({len(KNOBS)} of {len(KNOBS)} shipped peft knobs resolved "
          f"through the real process_config_with_overrides over n={len(args.overrides)} "
          "override tokens; evidence ends at composition — G2/G3 prove attach and trainability)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
