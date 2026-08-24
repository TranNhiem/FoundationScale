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

--out PATH (the finding-#78 PRODUCER half): OPTIONAL. When given and the
verdict is CLEAR, the FULL attachment-parent set -- the union of every live
FQN the shipped targets match under the real matcher, never the 2-per-target
CENSUS_SAMPLE -- is persisted for live_save_gate.py --adapter-modules as a
JSON object {'adapter_modules': [...], 'source': ...}. Entries are
{'fqn', 'out_features', 'in_features'} when every unique parent exposes a
positive-int dims pair, else plain stems: dims are all-or-nothing because the
consumer's own dims-coverage check (live_save_gate.py:824-830) refuses a
partially dimmed file as an unstated mixture, so this producer never emits
one -- bare stems leave the gate's shape check abstaining BY NAME. Stems are
written in the ARTIFACT namespace: the single leading 'module.' segment every
live FQN on this estate is measured to carry is stripped exactly once
(ARTIFACT_STRIP_SEGMENT), and any FQN lacking it -- or left empty by it --
refuses the whole write rather than guess a strip. Fail-closed AT THE
PRODUCER LAYER, in code: an empty attachment set is UNMEASURED with NO file
written -- a zero must never travel as a census (doctrine 1); until #78 that
refusal lived ONLY downstream (live_save_gate.py:811), and that downstream
check is now a BACKSTOP for broken producers, not this producer's license to
emit []. The write is same-dir temp + flush/fsync + rename atomic, so a
crash mid-write cannot leave a truncated census that parses. BLOCKED with a
non-empty set persists nothing either: the surviving subset must not stand
as the launch-intended denominator, and a census of a mis-spelled target
list would outlive its own re-spelling. PATH must resolve OUTSIDE any tree
the gate will judge -- _load_adapter_modules refuses a census resolving
inside the judged tree. Every --out refusal prints the certified count (0)
and exactly one CENSUS_VERDICT= line, the same discipline as every
pre-existing path.

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
import json
import os
import sys
import tempfile

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

# Namespace hand-off for --out (#78), measured 2026-08-24 on this estate's
# live tree: every FQN named_modules() yields carries exactly ONE leading
# "module." segment (the model wrapper) which the checkpoint/artifact
# namespace does NOT carry. Strip exactly that one segment -- a stem renamed
# by more or by less than the measured segment is a denominator entry no
# artifact FQN can match, and a hand-rolled substring trim is a paraphrase
# oracle, the exact fix39 defect class this tool exists to kill.
ARTIFACT_STRIP_SEGMENT = "module."

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


class _CensusRefusal(Exception):
    """Internal control flow for the --out writer (#78): every raise site is a
    reason the census MUST NOT exist at the requested path. main() converts
    each into exactly one REFUSAL line plus exactly one
    CENSUS_VERDICT=UNMEASURED line -- never two verdict lines, never a file."""


def _artifact_stem(fqn: str):
    """Live-tree FQN -> artifact-namespace stem by removing the ONE measured
    leading 'module.' segment. Returns None when the segment is absent or is
    the entire FQN: the caller then REFUSES rather than guesses, because a
    guessed stem is a denominator entry no artifact can match (doctrine 4)
    -- and '' would be malformed-on-read at the consumer anyway."""
    if fqn.startswith(ARTIFACT_STRIP_SEGMENT):
        stem = fqn[len(ARTIFACT_STRIP_SEGMENT):]
        if stem:
            return stem
    return None


def _parent_dims(module):
    """(out_features, in_features) of an attachment parent, or None.

    Acceptance mirrored from the consumer (_load_adapter_modules,
    live_save_gate.py:786-801): positive, non-bool ints only -- JSON booleans
    ARE Python ints, so an unchecked isinstance would let a True/True pair
    read as a plausible (out, in) and mint wrong shapes with an authoritative
    face. A module that lacks the attrs, or whose attrs raise when read,
    yields None; the caller's all-or-nothing rule then degrades the WHOLE
    file to bare stems (the gate's shape check abstains by name) -- never a
    partially-dimmed census, which is refuse-on-read at
    live_save_gate.py:824-830."""
    try:
        out_d = getattr(module, "out_features", None)
        in_d = getattr(module, "in_features", None)
    except Exception:  # noqa: BLE001 -- a raising property reads as absent.
        return None
    if (
        isinstance(out_d, int) and not isinstance(out_d, bool)
        and isinstance(in_d, int) and not isinstance(in_d, bool)
        and out_d > 0 and in_d > 0
    ):
        return (out_d, in_d)
    return None


def _atomic_write_json(out_path, payload) -> None:
    """Persist payload at out_path such that a crash mid-write can never leave
    a TRUNCATED CENSUS THAT PARSES (doctrine 4): mkstemp in the SAME
    directory (a rename is atomic only within one filesystem -- a temp on
    another mount would silently degrade the 'atomic' rename into a copy),
    flush + fsync (a crash must not persist the new name without its
    contents), then os.replace onto out_path. Before the replace, out_path is
    whatever it was; after, it is complete JSON. On any failure the temp file
    is unlinked, best-effort, so a failed probe never leaves a half-file a
    human could later mistake for a census; the cleanup error must never mask
    the real one, and after a successful replace tmp_path no longer exists to
    unlink (OSError swallowed by design)."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=".adapter-census-", suffix=".tmp", dir=out_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _persist_adapter_census(out_path, rows, population, hf_model_path, targets, total) -> None:
    """Write the #78 launch-time attachment-parent census in exactly the shape
    tools/live_save_gate.py:_load_adapter_modules accepts, or raise
    _CensusRefusal with NO file at out_path.

    Denominator discipline is enforced HERE at the producer, not delegated
    downstream:
      * rows carries the FULL per-target match lists (the same `found` lists
        CENSUS_SAMPLE previews only 2 of) -- the file always carries ALL of
        them, de-duplicated into a SORTED union: one module matched by TWO
        shipped patterns would otherwise appear twice and the gate loader
        refuses a census carrying duplicates outright
        (live_save_gate.py:816-823), and sorted order makes the emitted bytes
        a pure function of the attachment SET, so re-ordering or re-spelling
        LORA_TARGETS can never silently diff the denominator file;
      * the EMPTY-SET guard sits directly in front of the only call that
        creates the file. Broken to see red: hand this writer rows whose
        every `found` is [] and it must raise BEFORE any temp file is
        created -- if it ever writes [], the producer-side control against a
        manufactured zero denominator (doctrine 1) is dead. On current
        verdict logic CLEAR already implies non-empty, so this guard firing
        means the verdict logic changed, which is exactly when a guard earns
        its keep;
      * dims are all-or-nothing, the consumer contract at
        live_save_gate.py:824-830: emitting dims "per module where exposed"
        naively would mint exactly the partially-dimmed mixture the gate
        refuses on read -- a file that LOOKS producer-complete and
        adjudicates nothing. So a single parent without clean dims degrades
        EVERY entry to a bare stem, and the gate's shape check abstains by
        name instead of driving against an unstated mixture.
    On return (no raise) the file EXISTS, parses, and carries >= 1 unique
    stem.
    """
    mod_by_fqn = {fqn: module for (module, _leaf, _prefix, fqn) in population}
    raw_matches = 0
    attachment_live = set()
    for (_t, _real_n, _grep_n, found) in rows:
        for fqn in found:
            raw_matches += 1
            attachment_live.add(fqn)

    pairs = []  # [(live_fqn, artifact_stem)], sorted by live_fqn
    strip_failures = []
    for fqn in sorted(attachment_live):
        stem = _artifact_stem(fqn)
        if stem is None:
            strip_failures.append(fqn)
        else:
            pairs.append((fqn, stem))

    if strip_failures:
        # Broken to see red: feed a matched FQN spelled without the measured
        # leading 'module.' (e.g. 'decoder.layers.0.self_attention.linear_qkv')
        # and this raise must fire; passing it through unstripped would ship a
        # census disjoint from every artifact FQN the save can produce.
        raise _CensusRefusal(
            f"--out path {out_path!r}: {len(strip_failures)} of "
            f"{len(attachment_live)} unique matched FQN(s) lack the single "
            f"leading '{ARTIFACT_STRIP_SEGMENT}' segment this census is "
            f"measured to strip (or nothing would remain after it); first "
            f"offender: {strip_failures[0]!r}. The namespace hand-off is "
            "ambiguous here, and a stem emitted by guessing is a denominator "
            "entry no artifact FQN can match -- NO census file was written "
            "(doctrine 4)."
        )
    if not pairs:
        raise _CensusRefusal(
            f"--out path {out_path!r}: the attachment set is EMPTY (0 "
            f"unique parents assembled from {raw_matches} raw matches over "
            f"{len(rows)} targets). CENSUS_VERDICT=UNMEASURED and NO file "
            "written: a zero can never travel as a census (doctrine 1). The "
            "downstream empty-declarations refusal (_load_adapter_modules, "
            "live_save_gate.py:810-815) is a BACKSTOP for broken producers, "
            "not this producer's license to emit []."
        )

    dims = {}
    for fqn, stem in pairs:
        d = _parent_dims(mod_by_fqn[fqn])
        if d is not None:
            dims[stem] = d
    if len(dims) == len(pairs):
        entries = [
            {"fqn": s, "out_features": dims[s][0], "in_features": dims[s][1]}
            for _f, s in pairs
        ]
        dims_note = (
            f"dims=all ({len(dims)} of {len(pairs)} parents carry "
            "positive-int out_features/in_features; gate shape check armed)"
        )
    else:
        entries = [s for _f, s in pairs]
        dims_note = (
            f"dims=none ({len(dims)} of {len(pairs)} parents expose clean "
            "positive-int dims) -- all entries written as bare stems so the "
            "gate's shape check abstains BY NAME; a partially-dimmed census "
            "is refuse-on-read at live_save_gate.py:824-830 and would only "
            "LOOK shipped"
        )

    payload = {
        "adapter_modules": entries,
        # 'source' is the provenance the gate loader folds into its basis
        # text (live_save_gate.py:764-766); a census that cannot say who
        # wrote it earns the louder NO-provenance basis instead.
        "source": (
            "launchers/lora_target_census.py launch-time live-module census "
            f"(#78): shipped ModuleMatcher over the base tree built from "
            f"{hf_model_path!r}; targets [{', '.join(targets)}]; population "
            f"{total} offerable modules; {len(pairs)} unique attachment "
            f"parents from {raw_matches} raw target-module matches"
        ),
    }

    try:
        _atomic_write_json(out_path, payload)
    except Exception as exc:  # noqa: BLE001 -- any write failure refuses.
        raise _CensusRefusal(
            f"--out path {out_path!r} could not be written "
            f"({type(exc).__name__}: {exc}). --out names the destination: "
            "fix the path, its parent directory, or its permissions and "
            "re-run. NO complete census now exists at that path, and none "
            "may be assumed; any previous file there is untouched (the "
            "atomic writer replaces only after a fully serialised temp file "
            "survives fsync)."
        ) from exc

    print(
        f"CENSUS_OUT {os.path.abspath(out_path)} attachment_parents="
        f"{len(pairs)} raw_matches={raw_matches} collapsed_duplicates="
        f"{raw_matches - len(pairs)} {dims_note}",
        flush=True,
    )


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
    ap.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Optional (#78 producer): on a CLEAR verdict, persist the FULL "
        "attachment-parent census -- every FQN the shipped matcher attaches, "
        "not the 2-per-target CENSUS_SAMPLE -- as the JSON "
        "live_save_gate.py --adapter-modules parses, in the artifact "
        "namespace (one measured leading 'module.' segment stripped; parent "
        "dims all-or-nothing). Written atomically and only on CLEAR; an "
        "empty or unpersistable set is CENSUS_VERDICT=UNMEASURED with NO "
        "file at PATH. Write it OUTSIDE any tree the gate will judge -- "
        "_load_adapter_modules refuses a census resolving inside the judged "
        "tree. Absent: byte-identical pre-#78 preflight behaviour, so "
        "existing preflight invocations are unaffected.",
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
        if args.out is not None:
            # A BLOCKED target list must NOT receive a census file: the shipped
            # strings must be re-spelled, and a persisted census of the
            # surviving attachments would outlive that re-spelling -- a file a
            # LATER, differently-spelled launch could be pointed at, laundering
            # tonight's blocked config into a gate input it never earned
            # (doctrine 4). CENSUS_NOTE is not a verdict line; exactly one
            # CENSUS_VERDICT= still follows below.
            print(
                f"CENSUS_NOTE --out path {args.out!r} NOT written: verdict is "
                "BLOCKED; persisting a census now would manufacture a "
                "denominator for strings PEFT was never asked to attach in "
                "this exact spelling.",
                flush=True,
            )
        print(
            f"CENSUS_VERDICT=BLOCKED ({len(zero)} of {len(rows)} shipped "
            f"targets attach nothing; population {total})",
            flush=True,
        )
        return EXIT_BLOCKED

    if args.out is not None:
        # Persist BEFORE the verdict line below: that line must never claim,
        # even by implication, a file this run failed to produce. Every
        # refusal here is its own exit carrying exactly one
        # CENSUS_VERDICT=UNMEASURED line (doctrines 3/4 -- an absent census
        # must never let a launch proceed as-if gate-verified). Reached only
        # on the CLEAR path: the BLOCKED branch above returns first, so a
        # mis-spelled target list never receives a census file, and a
        # measured all-zero population stays a BLOCKED verdict rather than
        # being re-faced as UNMEASURED.
        try:
            _persist_adapter_census(
                args.out, rows, population, args.hf_model_path, targets, total
            )
        except _CensusRefusal as exc:
            print(f"REFUSAL: --out census NOT written: {exc}", flush=True)
            print(
                f"CENSUS_VERDICT=UNMEASURED (0 of {len(rows)} targets "
                "carried to a persisted census: --out refused or failed; the "
                "attachment counts above stay measured, but a census the "
                "operator asked to persist and was NOT persisted must never "
                "let a launch proceed as-if gate-verified)",
                flush=True,
            )
            return EXIT_UNMEASURED

    print(
        f"CENSUS_VERDICT=CLEAR ({len(rows)} of {len(rows)} shipped targets "
        f"attach; population {total} = {n_leaf} leaf + {n_nonleaf} non-leaf; "
        f"controls 3/3 OK)",
        flush=True,
    )
    return EXIT_CLEAR


if __name__ == "__main__":
    sys.exit(main())
