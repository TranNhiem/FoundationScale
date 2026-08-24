#!/usr/bin/env python3
"""fix39 — LoRA target census whose oracle is the matcher that will do the attaching.

Founding shape of the defect this tool exists to kill: the launcher's
preflight scored each LoRA target with `grep -cF "$t"` over a module dump —
a SUBSTRING oracle — while the system under test decides attachment with
ModuleMatcher.match (megatron/bridge/peft/module_matcher.py):

    full_name = f"{prefix}.{name}" if prefix else name
    ...
    if name == pattern or wildcard_match(pattern, full_name): ...

where `name` is the LEAF attribute name, `full_name` the dotted FQN, and
wildcard_match (peft/utils.py:208) compiles "^" + pattern.replace("*", "(.*)")
+ "$" — fully anchored, '*' the only wildcard. Measured 2026-08-24 on the
real E4B tree (1556 modules): grep scored the shipped 'mlp.linear_fc1' 42
(substring of the doubled '.mlp.mlp.' spelling) while the real matcher scored
it 0 — and grep scored the CORRECT wildcard repairs 0, so the substring
oracle both certified the silent no-op and blocked its own repair. This probe
therefore calls the SHIPPED matcher, never a paraphrase of it, over the same
population PEFT's walk offers, in the training interpreter stack, and refuses
to render any verdict unless its own controls pass.

POPULATION RULE (stated, with its evidence): every module
model.named_modules() yields EXCEPT the root. Rationale: PEFT.__call__ ->
_walk_model -> walk(model, func) with leaf_only defaulting to False
(peft/base.py:120-124), and _map_module applies the function to the module
itself before recursing into named_children (peft/walk_utils.py:224,232) —
the system under test is offered EVERY module, not only leaves. Skipping the
root is verdict-neutral for any non-empty pattern: leaf-name equality would
need pattern == "", and the anchored regex cannot match an empty FQN. A
leaf-only census is NARROWER than the system under test and invents false
alarms exactly as readily as a wider one invents false greens (doctrine 5 is
symmetric): this family's 'linear_proj' is a TERowParallelLinear that OWNS a
child (post_layernorm), and filtering leaves deletes all 42 of them — that
exact false alarm is what control 3 below exists to catch forever.

CONTROLS (doctrine 3 — a census that cannot distinguish match from no-match
must abstain, not report zeros; N=3, ALL required before any verdict):
  C1 MUST_FIRE      'linear_qkv' must match > 0 modules.
  C2 MUST_NOT_FIRE  'zzz_no_such_module_xyz' must match exactly 0.
  C3 ANTI-NARROWING at least one non-leaf module exists in the population AND
                    at least one 'linear_proj' match is non-leaf. Encodes the
                    measured fact that THIS family's linear_proj owns a child;
                    if a future provider flattens that, C3 fails and this tool
                    ABSTAINS loudly rather than certify a narrowed census.

EXIT VOCABULARY (shipped estate vocabulary, not renumbered):
  0 CLEAR      — every --targets pattern attaches >= 1 module under the real
                 matcher; per-target CENSUS_TARGET rows were printed.
  1 BLOCKED    — one or more NAMED patterns attach ZERO modules; the
                 launcher's strings must be re-spelled. Per-target rows were
                 still printed, so the operator sees every count, not only
                 the first failure.
  3 UNMEASURED — this tool ABSTAINS: controls failed, the population is
                 degenerate, --targets parsed to 0 strings, the model failed
                 to build, or the ModuleMatcher API drifted. An abstention is
                 a first-class outcome (doctrine 5) and it BLOCKS the launch
                 (doctrine 4). Argparse failure exits 2 — deliberately off
                 vocabulary; the launcher maps it to the infrastructure arm.
  Every refusal path prints the number of targets actually certified (0).

Environment: importable ONLY in the training container (megatron.bridge must
import); elsewhere, ast.parse/py_compile is the offline-verifiable minimum
the contract suite enforces. Mirrors dump_gemma4_modules.py's build recipe
verbatim (AutoBridge -> provider, TP=PP=1, --ep, bf16, finalize,
initialize_model_parallel, provide_distributed_model) so the model under
census is the model the run sees. Stated build caveat: at --ep > 1 under
this single-process invocation (world_size 1), parallel-group init may
refuse; that surfaces as MODEL BUILD FAILED -> exit 3, an honest abstention
the launcher blocks on — measured tonight only at --ep 1 on the dense base.
"""

import argparse
import sys

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import is_expert_linear

# Controls, named at module scope so a future edit must look them in the eye.
CONTROL_MUST_FIRE = "linear_qkv"
CONTROL_MUST_NOT_FIRE = "zzz_no_such_module_xyz"
# C3's subject: measured 2026-08-24, this family's linear_proj OWNS a child
# (post_layernorm), so exactly the modules a leaf-only census silently drops.
CONTROL_NONLEAF_SUBJECT = "linear_proj"

EXIT_CLEAR = 0
EXIT_BLOCKED = 1
EXIT_UNMEASURED = 3


def build_model(hf_model_path: str, ep: int):
    """Instantiate the module graph the training run will see.

    Weights are not loaded — the census needs names and parentage, not
    tensors. Recipe mirrors the estate dump tool line for line, because that
    recipe is the measured-good build on this estate; diverging from it would
    make the census about a different model than the run's.
    """
    bridge = AutoBridge.from_hf_pretrained(hf_model_path)
    mp = bridge.to_megatron_provider(load_weights=False)
    mp.tensor_model_parallel_size = 1
    mp.pipeline_model_parallel_size = 1
    mp.expert_model_parallel_size = ep
    mp.expert_tensor_parallel_size = 1
    mp.pipeline_dtype = torch.bfloat16
    mp.finalize()
    mp.initialize_model_parallel(seed=0)
    return mp.provide_distributed_model(wrap_with_ddp=False)[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="fix39 LoRA target census — oracle: the shipped ModuleMatcher itself."
    )
    ap.add_argument("--hf_model_path", required=True)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument(
        "--targets",
        required=True,
        help="Comma-separated; the launcher passes its shipped LORA_TARGETS "
        "verbatim so what is censused is byte-identical to what is launched.",
    )
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        # Doctrine 1, verbatim: a sweep over zero units must ABSTAIN and name
        # 0; it must never print an empty success table.
        print(
            "REFUSAL: --targets parsed to 0 target strings. A census over 0 "
            "targets must abstain and name 0 (doctrine 1); it never renders a verdict.",
            flush=True,
        )
        print("CENSUS_VERDICT=UNMEASURED (0 of 0 targets certified: empty target list)", flush=True)
        return EXIT_UNMEASURED
    if args.ep < 1:
        print(
            f"REFUSAL: --ep={args.ep} is not a parallel geometry; refusing to "
            "guess the build (doctrine 4: an unwired context BLOCKS).",
            flush=True,
        )
        print(
            f"CENSUS_VERDICT=UNMEASURED (0 of {len(targets)} targets certified: bad --ep)",
            flush=True,
        )
        return EXIT_UNMEASURED

    try:
        model = build_model(args.hf_model_path, args.ep)
    except Exception as exc:  # noqa: BLE001 — any build failure is UNMEASURED by design.
        print(
            "MODEL BUILD FAILED — the census cannot examine a model that did "
            "not build, and no per-target count below would be evidence. One "
            "stated possibility: --ep > 1 under this single-process invocation "
            "(world_size 1) is measured only as 'may refuse at group init' — "
            "the correct response is to fix the census invocation for that "
            "base, never to bypass the census. "
            f"Original error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        print(
            f"CENSUS_VERDICT=UNMEASURED (0 of {len(targets)} targets certified: "
            "model build failed)",
            flush=True,
        )
        return EXIT_UNMEASURED

    # ---- population, per the stated rule --------------------------------
    population = []  # (module, leaf_name, prefix, full_fqn)
    has_children = []
    for fqn, module in model.named_modules():
        if not fqn:
            # The root: yielded with an empty FQN. Verdict-neutral for every
            # non-empty pattern (see module docstring); excluded so the
            # denominator counts exactly the offerable modules.
            continue
        prefix, _, leaf = fqn.rpartition(".")
        children = any(True for _ in module.named_children())
        population.append((module, leaf, prefix, fqn))
        has_children.append(children)

    total = len(population)
    if total == 0:
        print(
            "REFUSAL: named_modules() yielded only the root — a census over 0 "
            "modules must ABSTAIN and name 0 (doctrine 1); whatever built is "
            "not the model we think it is.",
            flush=True,
        )
        print(
            f"CENSUS_VERDICT=UNMEASURED (0 of {len(targets)} targets certified: population 0)",
            flush=True,
        )
        return EXIT_UNMEASURED

    n_nonleaf = sum(has_children)
    n_leaf = total - n_nonleaf
    n_expert = sum(1 for (_, _, _, f) in population if is_expert_linear(f))
    nonleaf_fqns = {f for (_, _, _, f), c in zip(population, has_children, strict=True) if c}

    print(
        f"CENSUS_POPULATION total={total} leaf={n_leaf} non_leaf={n_nonleaf} "
        f"is_expert_linear={n_expert}",
        flush=True,
    )
    print(
        "CENSUS_POPULATION_RULE all modules torch named_modules() yields "
        "except the root — the population PEFT's own walk offers "
        "(leaf_only=False, peft/base.py:120-124; apply-then-recurse, "
        "walk_utils.py:224,232). The printed totals are the denominator for "
        "every count below.",
        flush=True,
    )

    def hits(pattern):
        # Ask the SHIPPED matcher — one fresh ModuleMatcher per pattern is the
        # construction measured green in the fix39 sketch run on this tree.
        matcher = ModuleMatcher(target_modules=[pattern])
        found = []
        for module, leaf, prefix, fqn in population:
            if matcher.match(module, leaf, prefix) is not None:
                found.append(fqn)
        return found

    # ---- controls AND census inside one guard: a matcher that raises is an
    # unwired or API-drifted oracle, and doctrine 4 says that BLOCKS — it must
    # never render as a table of zeros.
    try:
        fire_hits = hits(CONTROL_MUST_FIRE)
        null_hits = hits(CONTROL_MUST_NOT_FIRE)
        proj_hits = hits(CONTROL_NONLEAF_SUBJECT)
        per_target = [(t, hits(t)) for t in targets]
    except Exception as exc:  # noqa: BLE001 — any matcher failure is UNMEASURED.
        print(
            "REFUSAL: the shipped ModuleMatcher could not be exercised "
            f"({type(exc).__name__}: {exc}). An unwired or API-drifted oracle "
            "BLOCKS (doctrine 4) — 0 targets were certified, and no CENSUS_TARGET "
            "rows are emitted, so the launcher fails closed on their absence too.",
            flush=True,
        )
        print(
            f"CENSUS_VERDICT=UNMEASURED (0 of {len(targets)} targets certified: matcher raised)",
            flush=True,
        )
        return EXIT_UNMEASURED

    proj_nonleaf = [f for f in proj_hits if f in nonleaf_fqns]
    c1_ok = len(fire_hits) > 0
    c2_ok = len(null_hits) == 0
    c3_ok = n_nonleaf > 0 and len(proj_nonleaf) > 0
    print(
        f"CENSUS_CONTROL MUST_FIRE '{CONTROL_MUST_FIRE}' -> {len(fire_hits)} "
        f"modules (require > 0): {'OK' if c1_ok else 'FAILED'}",
        flush=True,
    )
    print(
        f"CENSUS_CONTROL MUST_NOT_FIRE '{CONTROL_MUST_NOT_FIRE}' -> "
        f"{len(null_hits)} modules (require 0): {'OK' if c2_ok else 'FAILED'}",
        flush=True,
    )
    print(
        f"CENSUS_CONTROL ANTI_NARROWING non_leaf_population={n_nonleaf} "
        f"(require > 0), '{CONTROL_NONLEAF_SUBJECT}' matches that are non-leaf: "
        f"{len(proj_nonleaf)} of {len(proj_hits)} (require >= 1 — a leaf-only "
        f"census reads 0 here and is exactly the fix39 false-alarm class): "
        f"{'OK' if c3_ok else 'FAILED'}",
        flush=True,
    )
    if not (c1_ok and c2_ok and c3_ok):
        print(
            "CONTROLS FAILED — this probe cannot now distinguish match from "
            "no-match on this tree, so every count it could print would be "
            "noise wearing a denominator. Refusing to render a verdict; the "
            "per-target rows below are therefore withheld entirely.",
            flush=True,
        )
        print(
            f"CENSUS_VERDICT=UNMEASURED (0 of {len(targets)} targets certified: controls failed)",
            flush=True,
        )
        return EXIT_UNMEASURED

    # ---- census proper, real matcher vs the secondary substring oracle -----
    # The dump text stands in for the pre-fix preflight oracle: one line per
    # module FQN. The substring column is DECLARED-SECONDARY forensics — the
    # launcher reads only the real-matcher column; divergence between the two
    # columns on the live tree is the fix39 evidence signature and is counted
    # at the end.
    dump_text = "\n".join(fqn for (_, _, _, fqn) in population)
    rows = []  # (target, real_count, grep_oracle_count)
    divergent = 0
    for t, found in per_target:
        real_n = len(found)
        grep_n = dump_text.count(t)
        rows.append((t, real_n, grep_n, found))
        if (real_n > 0) != (grep_n > 0):
            divergent += 1
        # Machine rows: exactly one per shipped target, fixed field layout —
        # the launcher's awk keys on $1==CENSUS_TARGET and string-compares $2,
        # which is literal-safe for the '.' and '*' these patterns now carry.
        print(f"CENSUS_TARGET {t} {real_n} {grep_n}", flush=True)

    print(
        f"CENSUS_NOTE oracle_divergence: {divergent} of {len(rows)} targets "
        "disagree in sign between the real matcher and the secondary "
        "substring oracle (informational only; the substring column decided "
        "nothing — it is the pre-fix oracle, kept so a future operator can "
        "SEE the disagreement rather than trust a claim about it).",
        flush=True,
    )

    print("CENSUS_SAMPLES (first 2 attachment FQNs per target; '-' = attaches nothing)", flush=True)
    for t, real_n, _, found in rows:
        sample = ", ".join(found[:2]) if found else "-"
        print(f"CENSUS_SAMPLE {t} n={real_n} {sample}", flush=True)

    zero = [(t, g) for (t, r, g, _) in rows if r == 0]
    if zero:
        for t, g in zero:
            print(
                f"CENSUS_BLOCK target '{t}' attaches 0 of {total} modules "
                f"under the REAL matcher (secondary substring oracle scored "
                f"{g}). Under the pre-fix launcher this row is precisely what "
                f"grep laundered into a pass.",
                flush=True,
            )
        print(
            f"CENSUS_VERDICT=BLOCKED ({len(zero)} of {len(rows)} shipped "
            f"targets attach nothing; population {total})",
            flush=True,
        )
        return EXIT_BLOCKED

    print(
        f"CENSUS_VERDICT=CLEAR ({len(rows)} of {len(rows)} shipped targets "
        f"attach; population {total} = {n_leaf} leaf + {n_nonleaf} non-leaf; "
        f"controls 3/3 OK)",
        flush=True,
    )
    return EXIT_CLEAR


if __name__ == "__main__":
    sys.exit(main())
