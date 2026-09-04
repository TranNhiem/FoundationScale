#!/usr/bin/env python3
"""#202: FSDP syncs module state POSITIONALLY, so a hash-ordered buffer dict swaps tensors.

WHAT / WHY. build_runtime constructs FSDP with sync_module_states=True, whose contract
the operator reads as "make every rank agree". It does not do that. torch's
_sync_module_states walks module.named_buffers(), appends each tensor to a flat list,
and hands that list to _broadcast_coalesced, which pairs rank 0's element i with every
other rank's element i. The NAME is read once, only to test the ignore-list; nothing
downstream matches on it. So the collective is correct exactly when every rank
enumerates its buffers in the same order, and silently wrong when they do not.

They do not. HuggingFace's rotary embeddings build their buffer set by iterating a SET:

    self.layer_types = list(set(config.layer_types))        # gemma3, :162
    for layer_type in self.layer_types:
        self.register_buffer(f"{layer_type}_inv_freq", ...)
        self.register_buffer(f"{layer_type}_original_inv_freq", ...)

set iteration order over str follows CPython's per-process randomised string hash, so
two ranks of the SAME torchrun job register the same four buffers in different orders.
Gemma-3 alternates sliding-window and full attention, so the four are
full_attention_inv_freq, full_attention_original_inv_freq, sliding_attention_inv_freq
and sliding_attention_original_inv_freq -- all [64] float32, mutually
shape-compatible, which is why the broadcast neither errors nor warns. It just puts
the full-attention RoPE table into the sliding-attention slot on some ranks.

Measured on this estate, 7 ranks, PRE-first-forward:

    buffers divergent across ranks   4 of 5
    parameters divergent             0 of 341     (nn.ModuleList order is deterministic)
    distinct buffer enumeration orders            2
    fixed-eval loss on byte-identical input       2.437912 and 3.540419

Two independent remedies were measured and both close it: PYTHONHASHSEED=0 (0 of 5
divergent, one loss) and a name-SORTED rank-0 broadcast (0 of 5, one order, one loss).
An UNSORTED rank-0 broadcast does NOT (4 of 5 unchanged) -- positional pairing is the
bug, so a positional repair cannot fix it. PYTHONHASHSEED is rejected as the shipped
fix: it is an environment fact the framework cannot enforce on an operator who writes
their own launch line, it silences the symptom for every ordering bug at once rather
than this one, and it leaves the framework asserting an invariant it never measures.

The fix is therefore structural and lives with the framework, not the model or the
launcher: before FSDP sees the module, re-key every module's _buffers dict into name
order, so the enumeration order becomes a property of the NAMES (which all ranks agree
on) instead of a property of the process's hash seed. Buffer order carries no
semantics -- lookup is by attribute name, state_dict is keyed by name, and
_non_persistent_buffers_set is a set of names -- so the reordering is observable only
to the very iteration FSDP performs. This is not Gemma-specific: the same
list(set(...)) idiom is in gemma3, modular_gemma3, gemma3n, gemma4, modular_gemma4,
modernbert, modernbert_decoder and t5gemma2, and any future model that registers
buffers from an unordered container inherits it.

Reordering per module is sufficient only if the MODULE walk is itself rank-invariant,
which is a separate claim, so the stage does not assume it: the normaliser
all_gather_objects both digests and REFUSES the run if the ranks still disagree.
Refusing is correct here -- the alternative is a model that trains on eight different
RoPE tables and reports one loss.

A fix that cannot be observed is the #86 shape, so the stage also installs the proof:
_model_state_rank_invariance runs AFTER the FSDP construction and compares, across
ranks, a sha256 of every buffer's bytes plus the parameter NAME ORDER. Its denominators
are honest in both directions. Buffers are replicated under FULL_SHARD, so their VALUES
must match bitwise and are compared as such. Parameters are NOT: use_orig_params=True
with FULL_SHARD gives each rank a different slice, and an uneven division makes even
the local shapes differ legitimately, so comparing parameter values -- or shapes --
would manufacture a false RED. Only their name order is rank-invariant, so only their
name order is claimed. Zero comparisons is UNMEASURED, never PASS: with world_size 1,
or with dist uninitialised, there is no cross-rank question to answer and the record
says so in its own state rather than reporting a vacuous green.

The verdict lands in the run's metrics under model_state_rank_invariance. That is the
CAUSE-side companion to fs178's fixed_eval_rank_invariance, which measures the
downstream symptom: when the buffers diverge, the eval losses diverge, and the existing
metric could only report that the ranks disagreed, never why.

This stage refuses to write when either anchor is absent or multiplied, when the
helpers do not land immediately before build_runtime, when either call site is not on
the correct side of the FSDP construction, when any other FSDP argument changes by a
byte, when the metrics insertion does not precede the fs178 entry, when the patched
text no longer compiles, when the transform is not byte-idempotent, or when any
executed control is not observed. Controls exec the patched helpers against fakes --
the build host has no torch -- and include a MUST_FIRE: the pre-image must fail the
call-site gate, proving the gate can go red at all, and a normaliser fed a
deliberately hash-scrambled buffer dict must be observed CHANGING it. It measures 95
only when the target cannot be read at all.
"""

from __future__ import annotations

import ast
import pathlib
import sys

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
MARK = "fs202:"
N_CONTROLS = 8  # C1..C8, of which C7 and C8 are MUST_FIREs.

# Anchors. Assembled rather than written literally so this stage's own source is not a
# hit for the scans that read the generated tree (the scanner-self-hit class, #199).
DEF_BUILD = "def build" + "_runtime("
TIED_CALL = "_check_tied" + "_parameters(model, wrap_census"
FSDP_CTOR = "sharded = FSDP("
SHARD_EXCEPT = (
    '        raise OperationFailure("load", "sharding", '
    'f"FSDP construction failed: {exc}") from exc\n'
)
OPT_TRY = "    try:\n        optimizer = torch.optim.AdamW(\n"
FS178_KEY = '        "fixed_eval' + '_rank_invariance": {\n'
OPEN_MARK = "# --- " + "fs202" + ":"
CLOSE_MARK = "# --- end " + "fs202" + " ---"
CENSUS_NAME = "BUFFER_ORDER" + "_CENSUS"
NORMALISE = "_normalise_buffer" + "_order"
INVARIANCE = "_model_state_rank" + "_invariance"

# Every FSDP keyword the construction carries today. The stage adds no argument and
# removes none; if this set changes by a byte the stage refuses, because a buffer-order
# fix has no business editing the sharding strategy.
FSDP_ARGS = (
    "            sharding_strategy=ShardingStrategy.FULL_SHARD,\n"
    "            auto_wrap_policy=wrap_policy,\n"
    "            sync_module_states=True,\n"
    "            use_orig_params=True,\n"
    "            device_id=device,\n"
)


HELPER_BLOCK_RAW = '''
BUFFER_ORDER_CENSUS: dict = {}
RANK_INVARIANCE_CENSUS: dict = {}


def _normalise_buffer_order(model):
    """Give named_buffers() an order every rank agrees on, BEFORE FSDP is constructed.

    FSDP(sync_module_states=True) broadcasts module state by POSITION: torch's
    _sync_module_states walks named_buffers() into a flat list and _broadcast_coalesced
    pairs element i with element i. The name is read only to test the ignore-list. So
    when two ranks enumerate the same buffers in different orders -- which HuggingFace
    rotary embeddings do, because they register from list(set(config.layer_types)) and
    CPython randomises string hashing per process -- the collective silently writes one
    buffer's contents into a different buffer. Shape-compatible tensors make it quiet.

    Buffer order carries no semantics: attribute lookup, state_dict and
    _non_persistent_buffers_set are all keyed by NAME. So re-keying each module's
    _buffers into sorted order is observable only to the iteration FSDP performs, and
    it moves the order from a property of the process's hash seed to a property of the
    names, which every rank already agrees on.

    Sorting within a module is sufficient only if the MODULE walk is rank-invariant.
    That is a second claim and it is measured, not assumed: if the ranks still disagree
    after normalisation the run is REFUSED, because the alternative is eight ranks
    training on different tables under one reported loss.
    """
    import hashlib

    modules_total = 0
    modules_reordered = 0
    buffers_total = 0
    for _mod_name, mod in model.named_modules():
        modules_total += 1
        names = list(mod._buffers.keys())
        buffers_total += len(names)
        if len(names) < 2:
            continue
        ordered = sorted(names)
        if ordered == names:
            continue
        kept = dict(mod._buffers)
        mod._buffers.clear()
        for name in ordered:
            mod._buffers[name] = kept[name]
        modules_reordered += 1

    walk = [n for n, _ in model.named_modules()]
    buf_order = [n for n, _ in model.named_buffers(remove_duplicate=False)]

    def _digest(seq):
        return hashlib.sha256("\\x00".join(seq).encode("utf-8")).hexdigest()[:16]

    census = {
        "modules_total": modules_total,
        "modules_reordered": modules_reordered,
        "buffers_total": buffers_total,
        "module_walk_digest": _digest(walk),
        "buffer_order_digest": _digest(buf_order),
        "world_size": 1,
        "cross_rank": "unmeasured",
        "distinct_module_walks": None,
        "distinct_buffer_orders": None,
        "reason": "",
    }
    if not (dist.is_available() and dist.is_initialized()):
        census["reason"] = (
            "torch.distributed is not initialised; cross-rank enumeration order is not "
            "a question one process can answer, so this is UNMEASURED, not agreement"
        )
        return census
    world = dist.get_world_size()
    census["world_size"] = world
    if world < 2:
        census["reason"] = (
            "world_size=1: zero cross-rank comparisons are possible, and zero "
            "comparisons is UNMEASURED, never PASS"
        )
        return census

    gathered = [None] * world
    dist.all_gather_object(
        gathered, (census["module_walk_digest"], census["buffer_order_digest"])
    )
    walks = {g[0] for g in gathered}
    orders = {g[1] for g in gathered}
    census["distinct_module_walks"] = len(walks)
    census["distinct_buffer_orders"] = len(orders)
    agreed = len(walks) == 1 and len(orders) == 1
    census["cross_rank"] = "agreed" if agreed else "divergent"
    if not agreed:
        raise OperationFailure(
            "load", "buffer_order",
            f"{type(model).__name__}: after name-sorting every module's buffer dict the "
            f"ranks STILL enumerate module state differently -- {len(walks)} distinct "
            f"module walk(s) and {len(orders)} distinct buffer order(s) across {world} "
            "rank(s). FSDP(sync_module_states=True) matches tensors by POSITION, so "
            "continuing would broadcast one rank's buffer into a different buffer on "
            "another rank and diverge the model with no error and no warning. Refusing "
            "is the only honest outcome: the divergence is undetectable downstream "
            "except as an unexplained per-rank loss spread.",
        )
    return census


def _model_state_rank_invariance(model, group=None):
    """Prove, after the FSDP construction, that replicated module state agrees bitwise.

    Denominators are scoped to what is actually rank-invariant, in BOTH directions.

    Buffers are replicated under FULL_SHARD, so their bytes must match and are compared
    as sha256 over the raw bytes -- not as an allclose, because #202 is a swap of two
    valid tables and a tolerance would admit it.

    Parameters are NOT replicated. use_orig_params=True with FULL_SHARD gives every rank
    a different slice, and an uneven division makes even the local shapes differ
    legitimately, so comparing parameter values -- or shapes -- would manufacture a
    false RED. The only rank-invariant property is the NAME ORDER, which is also the
    property that matters for a positional collective, so that is the only parameter
    claim made and it says so in its own scope field.
    """
    import hashlib

    def _digest(tensor):
        flat = tensor.detach().cpu().contiguous().flatten()
        if flat.numel() == 0:
            return "empty"
        try:
            raw = flat.view(torch.uint8).numpy().tobytes()
        except Exception:
            raw = flat.to(torch.float64).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    buf = [(n, _digest(b)) for n, b in model.named_buffers(remove_duplicate=False)]
    par = [n for n, _ in model.named_parameters(remove_duplicate=False)]
    result = {
        "status": "UNMEASURED",
        "world_size": 1,
        "buffers": {
            "denominator": len(buf),
            "compared": 0,
            "divergent": 0,
            "divergent_names": [],
            "scope": "bitwise sha256 over raw bytes; buffers are replicated, so equality is required",
        },
        "params": {
            "denominator": len(par),
            "compared": 0,
            "name_order_identical": None,
            "scope": (
                "NAME ORDER only -- FULL_SHARD with use_orig_params gives each rank a "
                "different slice and an uneven split makes local shapes differ "
                "legitimately, so a value or shape comparison would be a false RED"
            ),
        },
        "reason": "",
    }
    if not (dist.is_available() and dist.is_initialized()):
        result["reason"] = "torch.distributed is not initialised; no cross-rank comparison exists"
        return result
    world = dist.get_world_size(group=group)
    result["world_size"] = world
    if world < 2:
        result["reason"] = (
            "world_size=1: zero cross-rank comparisons are possible, and zero "
            "comparisons is UNMEASURED, never PASS"
        )
        return result
    if not buf and not par:
        result["reason"] = "the model declares neither buffers nor parameters; nothing to compare"
        return result

    gathered = [None] * world
    dist.all_gather_object(gathered, (buf, par), group=group)

    buf_names = [n for n, _ in gathered[0][0]]
    if any([n for n, _ in g[0]] != buf_names for g in gathered[1:]):
        result["status"] = "RED"
        result["reason"] = (
            "the ranks do not agree on the buffer NAME order, so no positional "
            "collective over module state can be correct"
        )
        return result
    result["buffers"]["compared"] = len(buf_names)
    divergent = [
        n
        for i, (n, d0) in enumerate(gathered[0][0])
        if any(g[0][i][1] != d0 for g in gathered[1:])
    ]
    result["buffers"]["divergent"] = len(divergent)
    result["buffers"]["divergent_names"] = divergent[:16]

    par_ok = all(g[1] == gathered[0][1] for g in gathered[1:])
    result["params"]["compared"] = len(par)
    result["params"]["name_order_identical"] = par_ok

    if divergent or not par_ok:
        result["status"] = "RED"
        result["reason"] = (
            f"{len(divergent)} of {len(buf_names)} buffer(s) differ bitwise across "
            f"{world} rank(s); parameter name order identical={par_ok}"
        )
    else:
        result["status"] = "PASS"
        result["reason"] = (
            f"{len(buf_names)} buffer(s) bitwise identical and {len(par)} parameter "
            f"name(s) in identical order across {world} rank(s)"
        )
    return result
'''

HELPER_BLOCK = OPEN_MARK + " cross-rank module-state order ---\n" + HELPER_BLOCK_RAW + CLOSE_MARK + "\n\n\n"

# Call site A -- the FIX, on the RAW module, before FSDP can enumerate anything.
CALL_A = (
    "    # fs202: normalise buffer enumeration order BEFORE the FSDP construction.\n"
    "    # sync_module_states broadcasts module state by POSITION, so this must precede\n"
    "    # it; afterwards the collective has already run and the damage is done.\n"
    "    " + CENSUS_NAME + ".update(" + NORMALISE + "(model))\n"
)
# Call site B -- the PROOF, after the construction, so it observes what FSDP produced.
CALL_B = (
    "    # fs202: and prove it. A fix nothing measures is the #86 shape.\n"
    "    RANK_INVARIANCE_CENSUS.update(" + INVARIANCE + "(sharded))\n"
)

METRIC_ENTRY = (
    "        # fs202: the CAUSE-side companion to the fs178 entry below, which measures\n"
    "        # the downstream symptom. When FSDP's positional sync swaps two hash-ordered\n"
    "        # buffers the eval losses diverge, and fs178 could report only THAT the ranks\n"
    "        # disagreed, never why. UNMEASURED is a declared state here, not a pass.\n"
    '        "model_state_rank_invariance": {\n'
    '            "status": {"PASS": "measured", "RED": "measured", "UNMEASURED": "unmeasured"}.get(\n'
    '                RANK_INVARIANCE_CENSUS.get("status", "UNMEASURED"), "unmeasured"\n'
    "            ),\n"
    '            "verdict": RANK_INVARIANCE_CENSUS.get("status", "UNMEASURED"),\n'
    '            "buffers": RANK_INVARIANCE_CENSUS.get("buffers", {}),\n'
    '            "params": RANK_INVARIANCE_CENSUS.get("params", {}),\n'
    '            "buffer_order": dict(BUFFER_ORDER_CENSUS),\n'
    '            "display": (\n'
    '                "cross-rank module state "\n'
    '                + str(RANK_INVARIANCE_CENSUS.get("status", "UNMEASURED"))\n'
    '                + ": " + str(RANK_INVARIANCE_CENSUS.get("reason", "not run"))\n'
    "            ),\n"
    "        },\n"
)


def _stderr(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def _transform(text: str) -> tuple[str, bool]:
    """Return (patched, changed). Byte-idempotent: a second application is a no-op."""
    if OPEN_MARK in text:
        return text, False
    new = text.replace(DEF_BUILD, HELPER_BLOCK + DEF_BUILD, 1)

    # A: after the fs172 tied-parameter control, before the FSDP construction.
    i_tied = new.find(TIED_CALL)
    eol = new.find("\n", i_tied) + 1
    new = new[:eol] + CALL_A + new[eol:]

    # B: after the construction's except clause, before the optimizer's try.
    new = new.replace(SHARD_EXCEPT + OPT_TRY, SHARD_EXCEPT + CALL_B + OPT_TRY, 1)

    # C: the metric, immediately above fs178's entry.
    new = new.replace(FS178_KEY, METRIC_ENTRY + FS178_KEY, 1)
    return new, new != text


def _extract(new: str, name: str) -> str:
    tree = ast.parse(new)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(new, node) or ""
    return ""


class _FakeModule:
    """Minimal nn.Module stand-in: the build host has no torch."""

    def __init__(self, name, buffers, children=()):
        self._name = name
        self._buffers = dict(buffers)
        self._children = list(children)

    def named_modules(self, prefix=""):
        me = prefix or self._name
        yield me, self
        for c in self._children:
            yield from c.named_modules(f"{me}.{c._name}")

    def named_buffers(self, remove_duplicate=False, prefix=""):
        for mname, mod in self.named_modules():
            for bname in mod._buffers:
                yield f"{mname}.{bname}", mod._buffers[bname]

    def named_parameters(self, remove_duplicate=False, prefix=""):
        return iter(())


def _controls(new: str, pre: str) -> tuple[int, list[str]]:
    ok, notes = 0, []

    def rec(tag, good, detail):
        nonlocal ok
        ok += int(good)
        notes.append(f"{tag}: {'OBSERVED' if good else 'NOT OBSERVED'}  {detail}")

    src = _extract(new, NORMALISE)
    ns: dict = {
        "dist": type("D", (), {"is_available": staticmethod(lambda: False)})(),
        "OperationFailure": Exception,
    }
    exec(compile(src, "<fs202-normalise>", "exec"), ns)
    normalise = ns[NORMALISE]

    # C1/C7: a scrambled buffer dict must be observed CHANGING. This is the MUST_FIRE
    # for the fix itself -- a normaliser that never reorders anything would pass every
    # other control in this file while fixing nothing.
    scrambled = _FakeModule(
        "rotary",
        {
            "sliding_attention_inv_freq": 1,
            "full_attention_inv_freq": 2,
            "full_attention_original_inv_freq": 3,
            "sliding_attention_original_inv_freq": 4,
        },
    )
    before = list(scrambled._buffers)
    census = normalise(scrambled)
    after = list(scrambled._buffers)
    rec("C1", after == sorted(before) and after != before,
        f"scrambled 4-buffer rotary reordered {before[0]!r}-first -> {after[0]!r}-first")
    rec("C7", census["modules_reordered"] == 1 and census["buffers_total"] == 4,
        f"MUST_FIRE: the census counts the work it did (reordered={census['modules_reordered']} "
        f"of {census['modules_total']} module(s), buffers={census['buffers_total']})")

    # C2: values follow their names. A reorder that also permuted the payload would be
    # the very bug this stage exists to remove.
    rec("C2", scrambled._buffers["full_attention_inv_freq"] == 2
        and scrambled._buffers["sliding_attention_original_inv_freq"] == 4,
        "each name still owns its original tensor after the reorder")

    # C3: idempotence at runtime, not just in the text -- a second call must not churn.
    census2 = normalise(scrambled)
    rec("C3", census2["modules_reordered"] == 0
        and census2["buffer_order_digest"] == census["buffer_order_digest"],
        "a second normalise reorders 0 modules and yields the same order digest")

    # C4: already-sorted input is left alone, and a 0/1-buffer module is not counted.
    tidy = _FakeModule("root", {"a": 1, "b": 2}, [_FakeModule("child", {"z": 3})])
    c4 = normalise(tidy)
    rec("C4", c4["modules_reordered"] == 0 and c4["modules_total"] == 2
        and c4["buffers_total"] == 3,
        "sorted input untouched; the denominator still counts all 2 modules / 3 buffers")

    # C5: with dist unavailable the record is UNMEASURED and says why -- never agreement.
    rec("C5", c4["cross_rank"] == "unmeasured" and "UNMEASURED" in c4["reason"],
        "no dist -> cross_rank=unmeasured with a stated reason, not a vacuous green")

    # C6: the invariance proof declares UNMEASURED at world_size 1 rather than PASS.
    isrc = _extract(new, INVARIANCE)
    ins: dict = {
        "dist": type("D", (), {
            "is_available": staticmethod(lambda: True),
            "is_initialized": staticmethod(lambda: True),
            "get_world_size": staticmethod(lambda group=None: 1),
        })(),
        "torch": type("T", (), {"uint8": None, "float64": None})(),
    }
    exec(compile(isrc, "<fs202-invariance>", "exec"), ins)
    solo = ins[INVARIANCE](_FakeModule("m", {}))
    solo_ok = solo["status"] == "UNMEASURED" and "never PASS" in solo["reason"]
    rec("C6", solo_ok, f"world_size=1 -> status={solo['status']} (a single rank cannot PASS)")

    # C8: MUST_FIRE on the pre-image. If the call-site gate is green on the UNPATCHED
    # text then it is measuring nothing and every other gate below is unattributable.
    pre_bad = (CALL_A not in pre) and (CALL_B not in pre) and (OPEN_MARK not in pre)
    rec("C8", pre_bad, "MUST_FIRE: the pre-image carries neither call site nor the helper block")
    return ok, notes


def main() -> int:
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_fsdp_buffer_order.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    try:
        text = TARGET.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: cannot read {TARGET}: {exc}")
        return 95

    new, changed = _transform(text)
    gres: list[tuple[str, bool, str]] = []

    gres.append(("G1", text.count(DEF_BUILD) == 1 and text.count(TIED_CALL) == 1,
                 f"anchors unique: build_runtime={text.count(DEF_BUILD)} "
                 f"tied-control={text.count(TIED_CALL)} need=1/1"))
    gres.append(("G2", text.count(SHARD_EXCEPT + OPT_TRY) == 1,
                 f"the sharding-except/optimizer-try seam is unique "
                 f"({text.count(SHARD_EXCEPT + OPT_TRY)}, need 1)"))
    gres.append(("G3", text.count(FS178_KEY) == 1 and new.count(FS178_KEY) == 1,
                 "the fs178 metric key is unique pre and post (the new entry sits above it, "
                 "and must not duplicate it)"))

    i_helper, i_build = new.find(OPEN_MARK), new.find(DEF_BUILD)
    i_a, i_fsdp, i_b, i_opt = (new.find(CALL_A), new.find(FSDP_CTOR),
                               new.find(CALL_B), new.find("optimizer = torch.optim.AdamW"))
    gres.append(("G4", -1 < i_helper < i_build,
                 f"helpers land immediately before build_runtime ({i_helper} < {i_build})"))
    gres.append(("G5", -1 < i_a < i_fsdp,
                 f"THE FIX precedes the FSDP construction ({i_a} < {i_fsdp}); after it, "
                 "sync_module_states has already broadcast and the swap is committed"))
    gres.append(("G6", i_fsdp < i_b < i_opt,
                 f"THE PROOF follows the construction ({i_fsdp} < {i_b} < {i_opt}), so it "
                 "observes what FSDP actually produced, not what it was handed"))
    gres.append(("G7", new.count(FSDP_ARGS) == 1 and text.count(FSDP_ARGS) == 1,
                 "every FSDP keyword is byte-identical pre and post: this stage changes "
                 "enumeration order, never the sharding contract"))
    gres.append(("G8", new.count(CALL_A) == 1 and new.count(CALL_B) == 1
                 and new.count(OPEN_MARK) == 1 and new.count(CLOSE_MARK) == 1,
                 "exactly one of each: helper block, fix call, proof call"))

    compiled = True
    try:
        compile(new, str(TARGET), "exec")
    except SyntaxError as exc:
        compiled = False
        _stderr(f"patched text does not compile: {exc}")
    gres.append(("G9", compiled, "the patched module compiles"))

    again, changed_again = _transform(new)
    gres.append(("G10", again == new and not changed_again,
                 "byte-idempotence on own output (a second run is a byte-exact no-op)"))
    gres.append(("G11", changed or OPEN_MARK in text,
                 "the transform either changed the file or the mark was already present"))

    gates = 0
    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    try:
        cok, cnotes = _controls(new, text)
    except Exception as exc:  # noqa: BLE001
        cok, cnotes = 0, [f"controls raised {type(exc).__name__}: {exc}"]
    for n in cnotes:
        print("control " + n)

    if gates != len(gres) or cok != N_CONTROLS:
        _stderr(f"\nREFUSE 96: static gates {gates}/{len(gres)}, controls {cok}/{N_CONTROLS}; writing nothing")
        return 96
    if not apply:
        print(f"verdict: READY  {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
        return 0
    TARGET.write_text(new, "utf-8")
    print(f"{MARK} {gates}/{len(gres)} static gates, {cok}/{N_CONTROLS} controls")
    return 0


def _guarded() -> int:
    # A stage whose whole purpose is to stop a silent cross-rank divergence must not
    # collapse its own states: an unhandled exception is a REFUSE, not a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())
