#!/usr/bin/env python3
"""Gate: every tracked Python file sits in exactly one DECLARED mutation-scope set.

WHAT IS MEASURED
    `git ls-files src/foundationscale tools`, filtered to `*.py` without `__init__.py`, is the
    population.  Every file in it must appear in EXACTLY ONE of three sets: MODULE_PATHS in
    tools/mutate.py -- read as TEXT and parsed with `ast`, never imported -- which is COVERED;
    PENDING_ENROLMENT, declared below (load-bearing code the battery does not read yet); and
    OUT_OF_SCOPE, declared below (excluded, each with a stated reason).  Five defects turn the
    gate RED, all accumulated and reported together, never truncated at the first: R1 UNDECLARED
    (tracked, in no set), R2 DOUBLE (in two or more sets), R3 STALE (a declared path -- including
    a MODULE_PATHS value, a fourth path source -- that the git index does not contain; a
    declaration naming a file that does not exist is a claim about nothing), R4 EMPTY REASON
    (whitespace where the reason should be), and R5 GENERIC (one reason string defending two or
    more different files -- a reason that can be pasted onto another file is a reason for
    neither).

WHY IT IS A DEFECT (#316 is the proof, not the motivation)
    tests/tooling/test_mutation_anchor_freshness.py already pins MODULE_PATHS at 9 entries.
    That assertion pins a NUMBER chosen once; it says nothing about the TREE.  The map cannot
    shrink by accident, but it can fall arbitrarily far behind while the pin stays green -- and
    it did: #316 added a fifth objective gate, 444 new lines of adjudication logic, to
    src/foundationscale/gates/objective_gates.py, at 1,691 LOC the largest unmapped library
    file, and every gate in the repository stayed green.  No check was wrong; the denominator
    itself had silently stopped reaching part of the adjudication surface.  PENDING_ENROLMENT
    is therefore an UNMEASURED population, NOT a green one: its members' rules are not checked,
    only counted, and the count is printed on every CLEAR line so the gap cannot grow quietly.
    And coverage-floor touching a file means its LINES EXECUTE under tests; the battery
    covering a file means its RULES are checked against mutants.  Those are different claims,
    and the first is not evidence for the second.

WHAT IS NOT MEASURED (deliberately)
    Which set an undeclared file belongs in is the author's decision; the gate names R1 and
    refuses to guess.  Whether a stated reason is TRUE is beyond a static gate; R4 and R5 only
    require that a reason exist and be unpasteable.  The battery is not run here, and no tool
    module is imported -- tools/mutate.py carries heavy module-level state and the repository
    has been bitten by import side effects, so its scope map is parsed as text or not at all.

DENOMINATOR
    The git index, via `git ls-files src/foundationscale tools` -- never a filesystem walk.
    The working tree can carry stale artefacts (a build/ directory once double-counted the
    package); the index is what the repository claims to be.  Zero tracked files is UNMEASURED,
    never a pass: `all([])` is True, and a vacuous green is the exact truth this repository's
    doctrine exists to refuse.

EXIT CODES
    0   CLEAR      -- the partition holds; the banner states the full split, pending included:
                      CLEAR mutation_scope: N tracked file(s) = C covered + P pending +
                      X out-of-scope; 0 undeclared
    5   RED        -- one or more of R1-R5 fired; all are reported together
    95  UNMEASURED -- tools/mutate.py unreadable, MODULE_PATHS absent or not a literal
                      dict[str, str], `git ls-files` failing, or zero tracked files; an
                      absent input is UNMEASURED, never RED
    96  REFUSE     -- a bad invocation, or an unexpected internal error; a crash is not a verdict
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

# Paths are resolved against THIS FILE, not the working directory, so the verdict cannot be
# changed by where the gate is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[1]
MUTATE_PY = REPO_ROOT / "tools" / "mutate.py"

EXIT_CLEAR = 0
EXIT_RED = 5
EXIT_UNMEASURED = 95
EXIT_REFUSE = 96

# The declaration data of the gate.  PENDING_ENROLMENT is an UNMEASURED population being
# counted, not a green one being hidden.  OUT_OF_SCOPE is excluded only under a reason no
# other file could wear; R4 and R5 police the reasons themselves.
PENDING_ENROLMENT: dict[str, str] = {
    "src/foundationscale/rl/prompt_surface.py": (
        "#371 routing layer: 100% line-covered by 15 legs, but covered is not "
        "MUTATED -- no row yet proves that silently flipping the surface choice, "
        "or dropping add_special_tokens (measured: that duplicates BOS), would be "
        "caught. PENDING rather than MODULE_PATHS because enrolling it with zero "
        "rows would make the battery report a pass over nothing."
    ),
    "src/foundationscale/checkpoint/dcp_meta.py": (
        "parses dcp headers into storage_id identity and run-manifest facts that decide "
        "aliasing before any tensor comparison"
    ),
    "src/foundationscale/gates/objective_gates.py": (
        "the five objective gates -- declared, loss_components, metrics, reward_scale, "
        "hparam_drift -- and the largest unmapped library file at 1,691 LOC; #316 added "
        "444 lines of new adjudication logic to it with every gate in the repo green"
    ),
    "src/foundationscale/gates/probe.py": (
        "derives declared config facts, censuses expert families, and adjudicates the "
        "real-checkpoint alias control feeding gate contexts"
    ),
    "src/foundationscale/rl/online.py": (
        "binds the four online objectives through one parameterised algorithm, and is the "
        "only binding in the plane that also declares what one report row COUNTS; the "
        "RowsUnit check is measured against pairedness derived from the objective's own "
        "schema, so a mutant that weakens it lets a preference-pair count be reported as a "
        "completion count -- a denominator that is wrong by 2x while every shape still fits"
    ),
    "src/foundationscale/rl/online_objectives.py": (
        "the online and iterative family -- Online DPO, Iterative DPO, RAFT and "
        "Best-of-N -- whose selection rules decide WHICH rows train; a mutated "
        "selection silently trains on the wrong subset and still reports a finite loss"
    ),
    "src/foundationscale/rl/reward_model.py": (
        "the reward-model training path, whose scores become the reward every "
        "downstream algorithm optimises; a mutated pairwise comparison inverts the "
        "preference direction with no shape change and no exception"
    ),
    "src/foundationscale/rl/ppo.py": (
        "composes the stage-3f objectives into one algorithm and is the first binding that "
        "measures a learned value estimate once and hands the single reading to two consumers"
    ),
    "src/foundationscale/rl/preference.py": (
        "derives two facts about an objective from its own declarations rather than its "
        "class -- whether it is reference-anchored, and whether it is paired -- and the "
        "pairedness derivation decides the SHAPE of the forward closure, so a wrong "
        "reading mis-scores every row instead of refusing the batch"
    ),
    "src/foundationscale/rl/structural.py": (
        "the #397 introspection seam: it reads a protocol's member set from the "
        "runtime's OWN reader and compares each member's call SHAPE, because "
        "@runtime_checkable answers presence and nothing else. Its two verdicts must "
        "stay distinct -- a shape that CANNOT BE MEASURED is an abstention, a shape "
        "that DOES NOT MATCH is a refusal -- and a mutant that collapses either into "
        "the other, or that empties the compared set, certifies a mismatched call "
        "shape as conforming while raising nothing and changing no signature"
    ),
    "src/foundationscale/rl/preference_objectives.py": (
        "five preference objectives whose arithmetic decides a preference direction from "
        "hand-computable scalars, and whose refusals -- ORPO's exact-zero odds wall, SimPO's "
        "zero-supervision row, KTO's paired-batch rejection -- are the only thing between a "
        "silently halved denominator and a reported loss"
    ),
    "src/foundationscale/rl/value_head.py": (
        "grades an untrusted per-token value estimate against a batch mask and decides which "
        "positions are supervised at all, so a wrong count is an unmeasured advantage"
    ),
    "src/foundationscale/train/loop.py": (
        "sequences pre-GPU blocking, save-gate callback adjudication, missing-extra "
        "refusal, and the 0/5/95/96 exit-code contract"
    ),
    "tools/count_census_modules.py": (
        "is the production LoRA census denominator and accepts wrapped versus bare "
        "payload shapes and record forms"
    ),
    "tools/countables_census.py": (
        "rederives review-draft countable totals from the tree and treats "
        "untouched-subtree mismatches as proof the census itself is wrong"
    ),
    "tools/live_save_gate.py": (
        "returns the launch-time checkpoint decision for full-FT versus intentionally "
        "tiny LoRA populations seconds after save"
    ),
    "tools/preflight/_config.py": (
        "validates launch-config schema and freeze hashes that decide whether a launch "
        "is even admissible to preflight"
    ),
    "tools/preflight/_core.py": (
        "combines check results, coverage, and lane registry order into final clearance "
        "through the clearance algebra"
    ),
    "tools/preflight/items/conversion_coverage.py": (
        "contract-bound item compares declared conversion coverage against safetensors "
        "header contents"
    ),
    "tools/preflight/items/corpus_wiring.py": (
        "contract-bound item verifies corpus sample identity through canonical sha256 wiring"
    ),
    "tools/preflight/items/evidence.py": (
        "decides whether launch evidence is complete enough to support clearance rather "
        "than merely present"
    ),
    "tools/preflight/items/frozen_manifest.py": (
        "matches frozen manifest hashes and line identities to safetensors header facts"
    ),
    "tools/preflight/items/launch_provenance.py": (
        "uses parsed timestamps and launch records to decide whether provenance is "
        "coherent enough to clear"
    ),
    "tools/preflight/items/lora_probe.py": (
        "contract-bound item adjudicates LoRA artifact expectations with coverage and "
        "verdict outcomes"
    ),
    "tools/preflight/items/schedule.py": (
        "small item still finalizes a schedule compliance CheckResult rather than merely "
        "declaring constants"
    ),
    "tools/preflight/items/template_audit.py": (
        "contract-bound audit hashes and compares template state before allowing clearance"
    ),
    "tools/preflight/items/training_dynamics.py": (
        "turns training-dynamics logs into pass or block style CheckResults for preflight"
    ),
    "src/foundationscale/gates/example.py": (
        "carries @register at :147 and is a live REGISTRY gate in the controls harness "
        "(measured), not the teaching scaffold it reads as"
    ),
    "src/foundationscale/models/adapters.py": (
        "_verdict/_collect/_refuse_unless_bool and AdapterRefusal over 26 branch or "
        "raise sites (measured)"
    ),
    "tools/real_checkpoint_probe.py": (
        "decides PASS vs VACUOUS when the shipped gates first meet a real artifact; "
        "consumed by the Makefile, live_save_gate.py and emit_run_manifest.py (measured)"
    ),
    "tools/census_denominator_control.py": (
        "the control harness proving count_census_modules can fail; a control that "
        "cannot fire is this campaign's own recurring defect (#57, #239)"
    ),
    "tools/preflight/_selftest.py": (
        "proves every registered check can both fail and pass; a wrong assertion here "
        "certifies a registry structurally incapable of saying RED"
    ),
    "tools/preflight/items/verdict_schema.py": (
        "drives every registered item with doctored input and requires a schema-valid "
        "verdict; a lenient leg clears a registry that cannot say RED"
    ),
    "tools/preflight/_artifacts.py": (
        "parses safetensors headers, json and sha256 identity that every preflight item "
        "trusts; the claim that downstream items re-adjudicate every parsed value is "
        "UNVERIFIED, so it is declared pending rather than excluded"
    ),
    "src/foundationscale/rl/interfaces.py": (
        "ExperienceBatch, the LossFn/LossDeclaration contracts and "
        "build_objective_gate_context -- the bridge that decides which components and "
        "metrics reach the objective gates at all; a mutation here narrows a "
        "denominator rather than changing a number, which is the failure mode this "
        "campaign keeps finding and the one a value assertion does not catch"
    ),
    "src/foundationscale/rl/losses.py": (
        "the RL objective arithmetic -- SFTLoss's masked mean and DPOLoss's stable "
        "softplus margin, plus the sft_weight field that drives BOTH the computed term "
        "and its declaration; a mutation that desynchronises those two reproduces "
        "design condition (a) exactly, and the sign and comparison sites are where a "
        "silently-wrong loss lives"
    ),
    "src/foundationscale/rl/policy.py": (
        "PolicyPair's role namespace: the refusals that keep 'train'/'generate' "
        "disjoint from reference roles, and the frozen dict copy that stops a caller's "
        "later mutation from changing what a run scores against -- every one of them a "
        "guard whose inversion is silent at runtime"
    ),
    "src/foundationscale/rl/rollout.py": (
        "the seam between what a rollout source CLAIMS it produces and what it actually "
        "produced: check_capabilities reads the claim, verify_generated measures the "
        "artifact, and the refusal on an EMPTY required set is what stops a source from "
        "clearing by asking for nothing -- inverting any of those is a vacuous pass, "
        "not a wrong number"
    ),
    "src/foundationscale/rl/advantage.py": (
        "the advantage arithmetic AND the exclusion bookkeeping around it: the "
        "population-vs-sample divisor, the leave-one-out baseline's n-1, the backward "
        "GAE recursion, and the used/offered/rows triple that says WHICH samples "
        "survived a degenerate group -- a mutation to the rows bookkeeping misattributes "
        "gradients to the wrong sequences while every reported number stays plausible"
    ),
    "src/foundationscale/rl/weightsync.py": (
        "the accounting that makes a partial weight sync unrepresentable: transferred and "
        "skipped must PARTITION offered exactly, a failed rank must not be summed away into "
        "a pass, and verify_sync must refuse a report that moved nothing rather than return "
        "0 as a denominator. Every one of those is an inversion that reads as a healthy "
        "sync -- a stale generate-view produces plausible completions, so nothing "
        "downstream notices the weights never arrived"
    ),
    "src/foundationscale/rl/algorithm.py": (
        "the two gates that read ONE declaration: check_algorithm_wiring compares "
        "AlgorithmRequirements against what setup was handed AND against the wired loss's "
        "own declaration(), and verify_step compares it against what each step observed. "
        "Inverting a set comparison or dropping one direction of it reads as a run whose "
        "objective is fully accounted for, and the reward_stats.count == rows check is what "
        "stops one step carrying two denominators. NOT claimed: substituting the DECLARED "
        "count for the OBSERVED count at verify_step's return is an EQUIVALENT mutant, not "
        "an uncovered one -- every path that reaches the return has already refused a "
        "repeat on both sides and refused set inequality, so the two lengths are equal by "
        "construction. No test can kill it, and one written to try would be asserting a "
        "distinction the guards above it have removed"
    ),
    "src/foundationscale/rl/grpo.py": (
        "the first concrete Algorithm binding, and the only place where a declaration is "
        "produced rather than merely checked: it emits the AlgorithmRequirements that "
        "algorithm.py's two gates then adjudicate, so a mutation here moves BOTH sides of "
        "that comparison at once and the gates above it stay green. Its own arithmetic is "
        "the group-mean baseline, the ratio, the two clip bounds and the k3 KL estimator's "
        "expm1 form -- a flipped clip comparison or a dropped group boundary changes which "
        "sequences are trusted without changing any reported count"
    ),
    "src/foundationscale/rl/policy_gradient.py": (
        "three bindings that differ ONLY in how a reward becomes an advantage, so each "
        "one's estimator is the whole of what distinguishes it: RLOO's leave-one-out "
        "group baseline, REINFORCE's mean-return-seeded EMA and its momentum update, and "
        "Reinforce++'s population mean/std whitening with the two PPO clip bounds derived "
        "from one epsilon. A mutation to any of those swaps one algorithm's estimator for "
        "another's while every declaration, count and metric this file emits stays "
        "internally consistent -- the run reports the name the operator asked for and "
        "optimises something else. The second surface is the degenerate-batch bookkeeping "
        "the estimators sit on: the n-1 divisor that makes a one-row group unrepresentable, "
        "the std == 0.0 branch that must abstain rather than divide, and the "
        "strictly-above frac_above diagnostic whose 0.0 and 1.0 endpoints are DECLARED "
        "degenerate -- inverting a comparison there moves a reading inside its declared "
        "bounds, which is exactly the shape no bounds check can catch"
    ),
    "src/foundationscale/rl/ppo_objectives.py": (
        "the three objectives that make PPO PPO, and each one hides its mutation behind a "
        "number that still looks like a loss. The clipped surrogate is a MIN of two terms "
        "whose relative order flips with the SIGN of the advantage, so swapping min for "
        "max, or swapping the two clip bounds, still trains -- it just stops bounding the "
        "trust region in one direction, and the clip_fraction diagnostic keeps reporting a "
        "plausible number because tokens are still being clipped, only the wrong ones. The "
        "value loss is a MAX of a clipped and an unclipped squared error, the opposite "
        "extremum from the surrogate above it, which is exactly the pair a copy-paste "
        "inversion collapses into one. The KL trio's surface is the dead-band comparison "
        "and the two clamps: inverting `>` and `<` around the target turns the adaptive "
        "controller into a divergence amplifier, and it reports its coefficient truthfully "
        "the whole way down. Its metric_ceiling is the one FINITE bound the objective-gate "
        "plane needs from this module -- mutate it to something unreachable and the bound "
        "examines nothing while still reading as coverage"
    ),
    "src/foundationscale/rl/registry.py": (
        "the name->Algorithm lookup: whether `register_algorithm` refuses a duplicate name "
        "rather than silently rebinding it, and whether `lookup_algorithm` refuses an "
        "unknown name rather than returning a default. Both inversions produce a run that "
        "trains under an algorithm the operator did not name, and neither is visible in any "
        "metric the run emits -- the wrong objective converges to something. It also holds "
        "`_install_default_algorithms`, the ONE place the built-in bindings are named: "
        "dropping an assignment there makes a family unreachable by name after every "
        "reset, and the absence reads as a name that was never built"
    ),
    "src/foundationscale/rl/group_policy_objectives.py": (
        "GSPO, Dr.GRPO and DAPO stated as four declared axes -- ratio_scope, the advantage "
        "estimator, clip_bounds and reduction. The axes are READ by the tensor kernel, so a "
        "mutated axis does not fail: it silently prices a DIFFERENT algorithm under the "
        "requested name. DAPO's asymmetric upper clip is the sharpest case, because a "
        "symmetric bound leaves it arithmetically identical to GRPO"
    ),
    "src/foundationscale/rl/group_policy.py": (
        "the ONE binding all three sequence-level objectives share, and the three "
        "default-configured factories the registry installs. A swapped factory registers a "
        "real, working algorithm under another algorithm's name"
    ),
    "src/foundationscale/rl/torch_backend.py": (
        "the ONE generic tensor kernel: it turns any axes-speaking objective into the 0-dim "
        "differentiable scalar the optimizer sees. Every arithmetic surface here is silent "
        "when wrong -- a flipped loss sign, a wrong reduction denominator, a reversed k3 "
        "direction and an ignored clip bound all still produce a finite loss that descends "
        "to the wrong place. It is held to the pure-python oracle by a numerical-equivalence "
        "suite, and 13 of a 14-row local battery die against it"
    ),
    "src/foundationscale/rl/corpus.py": (
        "turns the on-disk corpus into typed Samples and extracts the verifiable MCQ gold. "
        "Both halves fail silently: a suffix the directory scan does not admit reads as an "
        "ABSENT corpus, and a loosened gold rule invents supervision by guessing a letter "
        "rather than abstaining -- which trains the policy against a wrong target while "
        "reporting a clean accuracy"
    ),
    "src/foundationscale/rl/rewards.py": (
        "scores a generated answer against the extracted gold. A reward that is wrong in a "
        "CONSTANT direction still produces a converging run, aimed at the wrong objective"
    ),
    "src/foundationscale/rl/trainer.py": (
        "the one module that owns torch and transformers end to end: generation, the "
        "no-grad old-logprob recompute, the supervision mask, and the optimizer step. The "
        "attention/response mask width split is the sharpest surface -- conflating the two "
        "either raises or shifts every position by one, and the shifted variant trains"
    ),
}

OUT_OF_SCOPE: dict[str, str] = {
    "src/foundationscale/gates/fixtures.py": (
        "builds deterministic synthetic expert populations consumed by gate controls, so "
        "mutating it only perturbs the controls themselves"
    ),
    "src/foundationscale/integrate.py": (
        "thin re-export shim whose dispatch machinery explicitly lives in gates.core; "
        "only 54 lines of names hand-off"
    ),
    "src/foundationscale/train/__main__.py": (
        "exists only to make the documented python -m foundationscale.train form resolve "
        "into the real cli"
    ),
    "src/foundationscale/train/cli.py": (
        "console-script argparse front end that names the installed command and hands "
        "control to loop.train"
    ),
    "tools/preflight/__main__.py": (
        "three-line -m entry that imports _cli.main once and raises SystemExit around "
        "its return code"
    ),
    "tools/preflight/_base.py": (
        "holds exit statuses, enums, version, chunking, and src-path bootstrap constants "
        "rather than clearance decisions"
    ),
    "tools/preflight/_cli.py": (
        "argparse entry that catches aborts and dispatches to registered checks and the reporter"
    ),
    "tools/preflight/_errors.py": (
        "only defines the ToolError abort taxonomy distinguishing tool failure from "
        "clearance or block outcomes"
    ),
    "tools/preflight/_fixtures.py": (
        "constructs the self-test and red-team worlds, including doctored registries, "
        "rather than production launch facts"
    ),
    "tools/preflight/_report.py": (
        "renders already-final CheckResult and Verdict values into banner, text, or json "
        "without computing clearance"
    ),
    "tools/preflight/_registrations.py": (
        "declarative lane wiring with 0 branch or raise sites (measured); every verdict "
        "it feeds is combined in _core.py, which is itself enrolled"
    ),
    "tools/mutate.py": (
        "the battery cannot sit in its own scope: a mutant applied to the harness "
        "corrupts the instrument judging every other mutant, the same self-application "
        "refusal tools/mutations.json already carries; mutate.py's own rules are "
        "instead measured by tests/tooling/test_mutation_anchor_freshness.py"
    ),
}


@dataclass(frozen=True)
class Finding:
    """One partition defect, anchored to the path (or paths) the defect is about."""

    rule: str
    label: str
    path: str
    detail: str


def parse_module_paths(text: str) -> tuple[dict[str, str] | None, str]:
    """Extract the `MODULE_PATHS = {...}` literal from mutate.py source, or say why not.

    Returns (None, reason) on ANY failure to grasp the map; the caller turns that into
    UNMEASURED.  Importing the module is not an option: its module-level state has side
    effects this gate will not inherit.
    """
    try:
        tree = ast.parse(text, filename="tools/mutate.py")
    except SyntaxError as exc:
        return None, f"tools/mutate.py does not parse ({exc}); the covered set is unreadable"
    for node in tree.body:
        value: ast.expr | None = None
        # Both spellings are accepted because MODULE_PATHS being annotated or bare is a
        # style choice in tools/mutate.py, not a semantic one -- a gate that read only
        # the bare form would silently measure nothing the day someone added a type.
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "MODULE_PATHS" for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "MODULE_PATHS"
        ):
            value = node.value
        if value is None:
            continue
        if not isinstance(value, ast.Dict):
            return None, (
                "MODULE_PATHS is not a dict literal; the gate grasps "
                "`MODULE_PATHS = {name: path, ...}` and nothing computed"
            )
        parsed: dict[str, str] = {}
        for key, val in zip(value.keys, value.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return None, "MODULE_PATHS has a non-literal key; a computed key is unauditable"
            if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
                return None, f"MODULE_PATHS[{key.value!r}] is not a str literal path"
            parsed[key.value] = val.value
        return parsed, ""
    return None, "no module-level MODULE_PATHS assignment found; the covered set is nowhere"


def gather_module_paths(mutate_py: Path) -> tuple[dict[str, str] | None, str]:
    """Read the covered set from disk. Fail CLOSED: unreadable is UNMEASURED, not RED."""
    try:
        text = mutate_py.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, (
            f"cannot read {mutate_py}: {exc} -- the covered set lives in that file, and an "
            "absent input is UNMEASURED, never RED (doctrine 4)"
        )
    return parse_module_paths(text)


def gather_tracked(repo_root: Path) -> tuple[frozenset[str] | None, str]:
    """Return the denominator from the git INDEX; a failing git is UNMEASURED."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "src/foundationscale", "tools"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return None, f"cannot run `git ls-files`: {exc}"
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        return None, f"`git ls-files` exited {proc.returncode}: {detail}"
    tracked = frozenset(
        path
        for path in proc.stdout.splitlines()
        if path.endswith(".py") and not path.endswith("__init__.py")
    )
    return tracked, ""


def check_partition(
    tracked: frozenset[str],
    module_paths: dict[str, str],
    pending: dict[str, str],
    out: dict[str, str],
) -> list[Finding]:
    """Apply R1-R5 to one (index, three sets) pair. Pure; no I/O; all findings, never first."""
    findings: list[Finding] = []
    owners: dict[str, set[str]] = {}
    for path in module_paths.values():
        owners.setdefault(path, set()).add("MODULE_PATHS")
    for path in pending:
        owners.setdefault(path, set()).add("PENDING_ENROLMENT")
    for path in out:
        owners.setdefault(path, set()).add("OUT_OF_SCOPE")

    for path in sorted(tracked - set(owners)):
        findings.append(
            Finding(
                "R1",
                "UNDECLARED",
                path,
                "tracked by git but in none of MODULE_PATHS, PENDING_ENROLMENT or "
                "OUT_OF_SCOPE; which set owns it is a decision this gate refuses to make",
            )
        )
    for path in sorted(owners):
        if len(owners[path]) >= 2:
            findings.append(
                Finding(
                    "R2",
                    "DOUBLE",
                    path,
                    f"claimed by {' + '.join(sorted(owners[path]))}; a file is measured, "
                    "counted-as-gap, or excluded -- never two at once",
                )
            )
    for path in sorted(set(owners) - tracked):
        findings.append(
            Finding(
                "R3",
                "STALE",
                path,
                f"declared in {' + '.join(sorted(owners[path]))} but absent from `git "
                "ls-files`; a declaration naming a file that does not exist is a claim "
                "about nothing",
            )
        )
    reasons: dict[str, list[str]] = {}
    for declared, label in ((pending, "PENDING_ENROLMENT"), (out, "OUT_OF_SCOPE")):
        for path, reason in declared.items():
            if not reason.strip():
                findings.append(
                    Finding(
                        "R4",
                        "EMPTY REASON",
                        path,
                        f"{label} entry carries no reason; the declaration is the claim "
                        "and the reason is its evidence -- none was given",
                    )
                )
            else:
                reasons.setdefault(reason, []).append(path)
    for _reason, paths in reasons.items():
        if len(paths) >= 2:
            findings.append(
                Finding(
                    "R5",
                    "GENERIC",
                    ", ".join(sorted(paths)),
                    f"one reason string defends {len(paths)} different files; a reason that "
                    "can be pasted onto another file is a reason for none of them",
                )
            )
    findings.sort(key=lambda f: (f.rule, f.path))
    return findings


def evaluate(
    tracked: frozenset[str],
    module_paths: dict[str, str],
    pending: dict[str, str],
    out: dict[str, str],
) -> tuple[int, list[str]]:
    """Turn one denominator and three declarations into an exit code and lines. Pure."""
    total = len(tracked)
    if total == 0:
        return EXIT_UNMEASURED, [
            "UNMEASURED mutation_scope: 0 tracked .py files under src/foundationscale and "
            "tools -- zero units is not a pass (doctrine 1). Either the git index was not "
            "read or the repository is not what the gate was pointed at."
        ]
    findings = check_partition(tracked, module_paths, pending, out)
    if findings:
        lines = [
            f"RED mutation_scope: {len(findings)} partition finding(s) over {total} "
            "tracked file(s); every defect is reported, none stops the count early"
        ]
        for f in findings:
            lines.append(f"  {f.rule} {f.label}: {f.path}")
            lines.append(f"      {f.detail}")
        lines.append(
            "  Declare every file in exactly ONE set: MODULE_PATHS in tools/mutate.py (the "
            "battery measures it), PENDING_ENROLMENT (load-bearing, not yet covered -- an "
            "UNMEASURED population, never a pass), or OUT_OF_SCOPE (excluded, under a "
            "reason no other file could wear)."
        )
        return EXIT_RED, lines
    covered = {*module_paths.values()}
    undeclared = tracked - covered - set(pending) - set(out)
    return EXIT_CLEAR, [
        f"CLEAR mutation_scope: {total} tracked file(s) = {len(covered)} covered + "
        f"{len(pending)} pending + {len(out)} out-of-scope; {len(undeclared)} undeclared"
    ]


# --------------------------------------------------------------------------------------------
# Controls.  Each MUST_FIRE runs the REAL detector over a SYNTHETIC declaration -- the shipped
# constants are never mutated -- and asserts the SPECIFIC rule fired, not merely that
# something was refused.  MUST_PASS controls cover a clean synthetic partition, every
# UNMEASURED path (the abstention must be reachable, not merely declared), and the live
# shipped partition over the live git index: this gate is born green, on purpose.
# --------------------------------------------------------------------------------------------

_SYNTH_TRACKED: frozenset[str] = frozenset(
    {"src/foundationscale/alpha.py", "src/foundationscale/beta.py", "tools/gamma.py"}
)
_SYNTH_COVERED: dict[str, str] = {"alpha": "src/foundationscale/alpha.py"}
_SYNTH_PENDING: dict[str, str] = {
    "src/foundationscale/beta.py": "decides admission to the launch lane"
}
_SYNTH_OUT: dict[str, str] = {"tools/gamma.py": "pure re-export shim with no branch sites"}


def _rules_fired(
    tracked: frozenset[str],
    covered: dict[str, str],
    pending: dict[str, str],
    out: dict[str, str],
) -> set[str]:
    return {f.rule for f in check_partition(tracked, covered, pending, out)}


def _fires_only(
    rule: str,
    tracked: frozenset[str],
    covered: dict[str, str],
    pending: dict[str, str],
    out: dict[str, str],
) -> bool:
    rc, _ = evaluate(tracked, covered, pending, out)
    return rc == EXIT_RED and _rules_fired(tracked, covered, pending, out) == {rule}


def c_r1_undeclared_tracked_file() -> bool:
    tracked = _SYNTH_TRACKED | {"src/foundationscale/delta.py"}
    return _fires_only("R1", tracked, _SYNTH_COVERED, _SYNTH_PENDING, _SYNTH_OUT)


def c_r2_file_in_two_sets() -> bool:
    out = {**_SYNTH_OUT, "src/foundationscale/beta.py": "hand-off shim with no decisions"}
    return _fires_only("R2", _SYNTH_TRACKED, _SYNTH_COVERED, _SYNTH_PENDING, out)


def c_r3_stale_pending_entry() -> bool:
    pending = {**_SYNTH_PENDING, "src/foundationscale/ghost.py": "used to gate the lane"}
    return _fires_only("R3", _SYNTH_TRACKED, _SYNTH_COVERED, pending, _SYNTH_OUT)


def c_r3_stale_module_path() -> bool:
    covered = {**_SYNTH_COVERED, "ghost": "src/foundationscale/ghost.py"}
    return _fires_only("R3", _SYNTH_TRACKED, covered, _SYNTH_PENDING, _SYNTH_OUT)


def c_r4_whitespace_only_reason() -> bool:
    pending = {"src/foundationscale/beta.py": "   \n\t  "}
    return _fires_only("R4", _SYNTH_TRACKED, _SYNTH_COVERED, pending, _SYNTH_OUT)


def c_r5_one_reason_two_files() -> bool:
    pending = {"src/foundationscale/beta.py": "nothing here is worth mutating"}
    out = {"tools/gamma.py": "nothing here is worth mutating"}
    return _fires_only("R5", _SYNTH_TRACKED, _SYNTH_COVERED, pending, out)


def c_clean_synthetic_partition() -> bool:
    rc, lines = evaluate(_SYNTH_TRACKED, _SYNTH_COVERED, _SYNTH_PENDING, _SYNTH_OUT)
    return rc == EXIT_CLEAR and "1 pending" in lines[0] and "0 undeclared" in lines[0]


def c_zero_tracked_files_is_unmeasured() -> bool:
    rc, _ = evaluate(frozenset(), {}, {}, {})
    return rc == EXIT_UNMEASURED


def c_unparseable_module_paths_is_unmeasured() -> bool:
    # The abstention must be REACHABLE: a MODULE_PATHS that is not a dict literal is 95,
    # asserted end-to-end through run(), not merely a parse return code.
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "mutate.py"
        bad.write_text("MODULE_PATHS = ['gate_a', 'gate_b']\n", encoding="utf-8")
        paths, err = gather_module_paths(bad)
        if paths is not None or "not a dict literal" not in err:
            return False
        return run(repo_root=Path(td), mutate_py=bad, quiet=True) == EXIT_UNMEASURED


def c_missing_module_paths_literal_abstains() -> bool:
    paths, err = parse_module_paths("COVERED = {'alpha': 'src/a.py'}\n")
    return paths is None and "MODULE_PATHS" in err


def c_unreadable_mutate_py_is_unmeasured() -> bool:
    # Fail CLOSED: a mutate.py that cannot be read is UNMEASURED, not CLEAR and not RED.
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "mutate.py"
        return run(repo_root=Path(td), mutate_py=missing, quiet=True) == EXIT_UNMEASURED


def c_live_tree_partition_is_clean() -> bool:
    # The shipped declaration over the shipped index: zero problems, pending counted.
    if not MUTATE_PY.is_file():
        return False
    return run(quiet=True) == EXIT_CLEAR


def c_mutate_py_flag_is_wired() -> bool:
    # --mutate-py is what makes the gate discriminable from outside, so it needs a control of
    # its own: a flag that PARSES but is never read would leave every suite leg measuring the
    # shipped tree while believing it measured a doctored one. The discrimination is the point.
    # Pointed at a path that does not exist the gate must reach UNMEASURED; if the flag were
    # ignored it would fall through to the live tree and return CLEAR instead.
    with tempfile.TemporaryDirectory() as td:
        missing = str(Path(td) / "absent_mutate.py")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            redirected = main(["--mutate-py", missing])
            live = main([])
    return redirected == EXIT_UNMEASURED and live == EXIT_CLEAR


CONTROLS: list[tuple[str, str, Callable[[], bool]]] = [
    ("R1 fires on an undeclared tracked file", "MUST_FIRE", c_r1_undeclared_tracked_file),
    ("R2 fires on double membership", "MUST_FIRE", c_r2_file_in_two_sets),
    ("R3 fires on a stale PENDING_ENROLMENT path", "MUST_FIRE", c_r3_stale_pending_entry),
    ("R3 fires on a stale MODULE_PATHS path", "MUST_FIRE", c_r3_stale_module_path),
    ("R4 fires on a whitespace-only reason", "MUST_FIRE", c_r4_whitespace_only_reason),
    ("R5 fires on one reason defending two files", "MUST_FIRE", c_r5_one_reason_two_files),
    ("a clean synthetic partition is CLEAR", "MUST_PASS", c_clean_synthetic_partition),
    ("zero tracked files -> 95", "MUST_PASS", c_zero_tracked_files_is_unmeasured),
    ("unparseable MODULE_PATHS -> 95", "MUST_PASS", c_unparseable_module_paths_is_unmeasured),
    ("missing MODULE_PATHS literal abstains", "MUST_PASS", c_missing_module_paths_literal_abstains),
    ("unreadable mutate.py -> 95", "MUST_PASS", c_unreadable_mutate_py_is_unmeasured),
    ("the live shipped partition is clean", "MUST_PASS", c_live_tree_partition_is_clean),
    ("--mutate-py redirects the covered set", "MUST_PASS", c_mutate_py_flag_is_wired),
]


def self_test() -> int:
    """Run every control. Exit 0 only if each did what it claims; print one line per control."""
    failures: list[str] = []
    for name, kind, fn in CONTROLS:
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001 - a crashing control is a failing control
            ok = False
            name = f"{name} (raised {type(exc).__name__}: {exc})"
        print(f"  {'ok' if ok else 'FAILED'} {kind}: {name}")
        if not ok:
            failures.append(name)
    fires = sum(1 for _, k, _ in CONTROLS if k == "MUST_FIRE")
    passes = len(CONTROLS) - fires
    # The tally is printed in the declared "SELF-TEST DENOMINATOR: N of N controls behaved"
    # wording because launchers/test_checks_gates.sh parses it and holds N to a FLOOR. rc=0
    # is not the measurement: a self-test whose control list silently shrinks to one still
    # exits 0, so the denominator is what the suite actually checks.
    if failures:
        print(
            f"SELF-TEST DENOMINATOR: {len(CONTROLS) - len(failures)} of {len(CONTROLS)} "
            f"controls behaved; {fires}x MUST_FIRE, {passes}x MUST_PASS -- FAILED"
        )
        return EXIT_RED
    print(
        f"SELF-TEST DENOMINATOR: {len(CONTROLS)} of {len(CONTROLS)} controls behaved; "
        f"{fires}x MUST_FIRE drove the real detector over a synthetic declaration to its "
        f"declared rule exit, {passes}x MUST_PASS pinned the clean partition, all three "
        f"UNMEASURED arms, the abstain path, the live shipped tree, and the --mutate-py "
        f"redirect the suite's discrimination leg depends on"
    )
    return EXIT_CLEAR


def run(repo_root: Path = REPO_ROOT, mutate_py: Path = MUTATE_PY, quiet: bool = False) -> int:
    """Gather the covered set and the index, evaluate, and return the exit code."""
    module_paths, err = gather_module_paths(mutate_py)
    if module_paths is None:
        if not quiet:
            print(f"UNMEASURED mutation_scope: {err}")
        return EXIT_UNMEASURED
    tracked, track_err = gather_tracked(repo_root)
    if tracked is None:
        if not quiet:
            print(f"UNMEASURED mutation_scope: {track_err}")
        return EXIT_UNMEASURED
    rc, lines = evaluate(tracked, module_paths, PENDING_ENROLMENT, OUT_OF_SCOPE)
    if not quiet:
        print("\n".join(lines))
    return rc


class _RefuseOnError(argparse.ArgumentParser):
    """A bad invocation is REFUSE (96): the gate said nothing about the tree."""

    def error(self, message: str) -> NoReturn:
        print(f"REFUSE mutation_scope: bad invocation: {message}")
        raise SystemExit(EXIT_REFUSE)


def main(argv: list[str]) -> int:
    parser = _RefuseOnError(
        prog="mutation_scope",
        description="Gate: every tracked Python file sits in exactly one declared scope set.",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="run the control suite instead of the gate"
    )
    # run() has always taken the covered-set source as a parameter; only main() pinned it.
    # Exposing it is what makes the gate DISCRIMINABLE from outside: a suite leg can hand it
    # a copy of tools/mutate.py with one path removed and watch R1 fire over the real tree
    # and the real declarations, with the covered set as the only variable. Nothing in the
    # working tree is touched. The shipped declaration constants stay pinned deliberately --
    # they are the thing under test, not an input.
    parser.add_argument(
        "--mutate-py",
        default=None,
        metavar="PATH",
        help="read MODULE_PATHS from PATH instead of the shipped tools/mutate.py",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    return run(mutate_py=Path(args.mutate_py) if args.mutate_py else MUTATE_PY)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - a crash is not a verdict
        print(f"REFUSE mutation_scope: unexpected internal error: {type(exc).__name__}: {exc}")
        sys.exit(EXIT_REFUSE)
