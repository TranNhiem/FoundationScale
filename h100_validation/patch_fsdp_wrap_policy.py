#!/usr/bin/env python3
"""#172: wrap only the layer classes the model itself declares; never shard a tied weight.

WHAT / WHY. build_runtime passed the size-based auto-wrap policy to FSDP. That policy
is structure-blind: with its 1e8 default it wrapped model.model.embed_tokens
(Qwen3-4B: 151936 x 2560 = 388,956,160 params) into its own FULL_SHARD unit, and the
unit re-sharded as soon as the embedding forward returned. Qwen3 sets
tie_word_embeddings: true, so lm_head.weight IS embed_tokens.weight; when lm_head ran
at the end of the same forward it saw the flat 1-D shard. Measured on 8xH100 job
37300, all 8 ranks:

    size mismatch, got input (1024), mat (1024x2560), vec (48619520)

and 48,619,520 = 388,956,160 / 8. This is not model-specific: Gemma and tied Llama
variants fail identically.

The fix resolves the transformer auto-wrap policy's transformer_layer_cls from the
model's OWN declaration, _no_split_modules -- measured in the run container
(transformers 5.5.0, torch 2.11.0a0):

    Qwen3ForCausalLM   ['Qwen3DecoderLayer']
    Gemma3ForCausalLM  ['Gemma3DecoderLayer','SiglipVisionEmbeddings',
                        'SiglipEncoderLayer','SiglipMultiheadAttentionPoolingHead']
    LlamaForCausalLM   ['LlamaDecoderLayer']

Wrapping only those classes leaves embeddings and lm_head in the ROOT FSDP unit,
which stays all-gathered for the entire forward -- that is what fixes the tied
weight. A partial resolution (Gemma3 declares four names, a text-only load
instantiates one) is NORMAL and is recorded in the census, not refused. There is NO
fallback: a silent size-based fallback is precisely the wrong-but-green policy that
shards a tied parameter and dies 400 lines later in a matmul, so an absent, empty,
or zero-resolving declaration raises OperationFailure instead of guessing. After the
FSDP construction succeeds, a tied-parameter control groups the PRE-wrap module's
named_parameters on data_ptr()/id() and refuses if any tied group spans two
different FSDP units; an untied model records a measured tied_groups=0.

This stage refuses to write when the broken import or call site is absent or
multiplied, when the build_runtime anchors are not unique, when RuntimeBundle
already carries a census-like field (the stage would have to choose a different
stash), when the old policy name survives anywhere in the patched text, when any
other FSDP argument changes by a byte, when the helper lands anywhere but
immediately before build_runtime, when functools cannot be imported exactly once,
when the patched text no longer compiles, when the transform is not
byte-idempotent, or when any executed control is not observed. Controls exec the
patched helpers against lightweight fakes -- this host has neither torch nor
transformers -- and include a MUST_FIRE: the pre-image text must FAIL the
transformer-policy gate, proving the gate can go red at all. It measures 95 only
when the target cannot be read at all.
"""

from __future__ import annotations

import ast
import functools
import pathlib
import re
import sys

TARGET = pathlib.Path(__file__).resolve().parent / "h100" / "gen" / "fs_train.fixed.py"
MARK = "fs172:"
N_CONTROLS = 10  # C0 extraction provenance + C1..C9.

# Needles are ASSEMBLED, never written as one literal: this stage counts them, and a
# source that contains its own needle is inside its own denominator.
SIZE_POLICY = "size_based" + "_auto_wrap_policy"
XFORM_POLICY = "transformer" + "_auto_wrap_policy"
NOSPLIT = "_no_split" + "_modules"
IMPORT_OLD = "from torch.distributed.fsdp.wrap import " + SIZE_POLICY
IMPORT_NEW = "from torch.distributed.fsdp.wrap import " + XFORM_POLICY
AUTO_OLD = "auto_wrap_policy=" + SIZE_POLICY + ","
AUTO_NEW = "auto_wrap_policy=wrap_policy,"
DEF_BUILD = "def build" + "_runtime("
MODEL_LINE = "    model = _build_model(artifacts)"
SHARD_EXCEPT = ('        raise OperationFailure("load", "sharding", '
                'f"FSDP construction failed: {exc}") from exc')
OPEN_MARK = "# --- " + "fs172" + ":"
CLOSE_MARK = "# --- end " + "fs172" + " ---"
# Needles for G11. The pre-image calls named_parameters ZERO times (measured on
# fs_train.fixed.py), so a post-image count of 1 is exact and unambiguous.
NP_ANY = "named_" + "parameters("
NP_SAFE = "named_" + "parameters(remove_duplicate=False)"
FSDP_CTOR = "sharded = FSDP("
TIED_CALL = "_check_tied_parameters(model,"

FSDP_ARGS = (
    "            sharding_strategy=ShardingStrategy.FULL_SHARD,",
    "            sync_module_states=True,",
    "            use_orig_params=True,",
    "            device_id=device,",
)

HELPER_BLOCK_RAW = '''
# --- fs172: wrap policy resolved from the model's OWN declaration ----------------
# The size-based policy is structure-blind: with its 1e8 default it wrapped
# model.model.embed_tokens (Qwen3-4B: 151936 x 2560 = 388,956,160 params) into its
# own FULL_SHARD unit, and that unit re-sharded the moment the embedding forward
# returned. Qwen3 sets tie_word_embeddings: true, so lm_head.weight IS
# embed_tokens.weight; when lm_head ran at the end of the same forward it saw the
# flat 1-D shard. Measured on 8xH100 job 37300, all 8 ranks:
#     size mismatch, got input (1024), mat (1024x2560), vec (48619520)
# and 48,619,520 = 388,956,160 / 8. This is not model-specific: Gemma and tied
# Llama variants fail identically. The fix wraps only the classes the model's OWN
# @NOSPLIT@ declaration names; measured in the run container (transformers 5.5.0,
# torch 2.11.0a0):
#     Qwen3ForCausalLM   ['Qwen3DecoderLayer']
#     Gemma3ForCausalLM  ['Gemma3DecoderLayer','SiglipVisionEmbeddings',
#                         'SiglipEncoderLayer','SiglipMultiheadAttentionPoolingHead']
#     LlamaForCausalLM   ['LlamaDecoderLayer']
# Embeddings and lm_head then stay in the ROOT FSDP unit, which stays all-gathered
# for the entire forward -- that is what fixes the tied weight.
WRAP_POLICY_CENSUS: dict = {}


def _resolve_wrap_policy(model):
    """Resolve @XFORM@ from the model's own @NOSPLIT@ declaration.

    Returns (policy, census); census is a plain dict of measured counts. Refuses
    -- and NEVER falls back to a size-based or any other policy -- when the
    declaration is absent/empty or resolves to zero live modules. That silent
    fallback is precisely the failure just measured on job 37300: a
    wrong-but-green policy that shards a tied parameter and dies 400 lines later
    in a matmul. A declared refusal is correct; a guess is not.
    """
    model_class = type(model).__name__
    declared = getattr(type(model), "@NOSPLIT@", None)
    if declared is None:
        declared = getattr(model, "@NOSPLIT@", None)
    modules = list(model.modules())
    n_modules = len(modules)
    if (not isinstance(declared, (list, tuple)) or not declared
            or not all(isinstance(name, str) for name in declared)):
        raise OperationFailure(
            "load", "wrap_policy",
            f"{model_class}: no usable @NOSPLIT@ declaration (need a non-empty "
            f"sequence of class-name strings, got {declared!r}); 0 of 0 declared "
            f"class name(s) usable among {n_modules} live modules; refusing to "
            "guess a wrap policy",
        )
    declared_names = list(declared)
    n_declared = len(declared_names)
    wanted = set(declared_names)
    matched = {}
    n_instances = 0
    for m in modules:
        cls_name = type(m).__name__
        if cls_name in wanted:
            matched.setdefault(cls_name, type(m))
            n_instances += 1
    n_resolved = len(matched)
    if n_resolved == 0:
        raise OperationFailure(
            "load", "wrap_policy",
            f"{model_class}: resolved 0 of {n_declared} declared @NOSPLIT@ class "
            f"name(s) among {n_modules} live modules; refusing to guess a wrap "
            "policy",
        )
    if n_instances == 0:
        raise OperationFailure(
            "load", "wrap_policy",
            f"{model_class}: resolved {n_resolved} of {n_declared} declared "
            f"@NOSPLIT@ class name(s) but 0 live instances among {n_modules} "
            "live modules; refusing to guess a wrap policy",
        )
    # 0 < n_resolved < n_declared is NORMAL and must NOT refuse:
    # Gemma3ForCausalLM declares four names and a text-only load instantiates
    # only Gemma3DecoderLayer. Record it in the census; a partial resolution is
    # not an error.
    policy = functools.partial(
        @XFORM@,
        transformer_layer_cls=set(matched.values()),
    )
    census = {
        "declared_names": declared_names,
        "declared": n_declared,
        "resolved": n_resolved,
        "instances": n_instances,
        "model_class": model_class,
    }
    return policy, census


def _check_tied_parameters(model, declared_names, census):
    """Assert every tied parameter group is co-located in ONE FSDP unit.

    Runs BEFORE the FSDP construction, on the pre-wrap module. Both halves of
    that are measured facts, not preferences:

    * torch's parameter iterator defaults to remove_duplicate=True, so a tied
      pair is yielded ONCE under one name and no group can ever reach size 2.
      Measured in the run container: a module tying head.weight to emb.weight
      yields 1 name by default and 2 with remove_duplicate=False. Taking the
      default here would make this detector VACUOUS -- it would report
      tied_groups=0 for the very Qwen3 tie that killed job 37300. (The call is
      spelled once, below; naming it in prose would put this docstring inside
      G11's own denominator.)
    * data_ptr() is 0 for an empty tensor (measured), and after FULL_SHARD with
      use_orig_params=True the parameters a rank does not own are exactly that.
      Grouping on data_ptr post-wrap would collapse every unowned parameter
      into one enormous false tied group and refuse a healthy run.

    Unit membership needs no wrapped model to read: the policy wraps exactly the
    declared classes that matched, so the assignment is already determined here.

    Zero tied groups is a legitimate measured zero (an untied model) and is
    recorded as tied_groups=0 in the census, not silently passed over.
    """
    groups = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        try:
            ptr = p.data_ptr()
        except Exception:
            ptr = 0
        # A materialized tensor groups by STORAGE, which catches object-identity
        # ties and storage-sharing views alike; ptr == 0 (meta, or an unowned
        # shard) falls back to object IDENTITY, which is what HF tie_weights()
        # produces -- and never to one shared bucket holding everything.
        key = ("ptr", ptr) if ptr else ("obj", id(p))
        groups.setdefault(key, []).append(name)
    tied = [names for names in groups.values() if len(names) > 1]
    census["tied_groups"] = len(tied)
    if not tied:
        return
    wrapped = set(declared_names)
    unit_of = {}
    # Read on the PRE-wrap tree, so a module's own class name IS the unit
    # boundary; there are no FSDP wrappers to unwrap through yet.
    for mod_name, m in model.named_modules():
        cls_name = type(m).__name__
        unit_of[mod_name] = cls_name if cls_name in wrapped else None

    def _unit_for(param_name):
        prefix = param_name.rsplit(".", 1)[0] if "." in param_name else ""
        parts = prefix.split(".") if prefix else []
        for i in range(len(parts), -1, -1):
            cand = ".".join(parts[:i])
            if unit_of.get(cand) is not None:
                return unit_of[cand]
        return "<root>"

    spanning = sum(1 for names in tied if len({_unit_for(n) for n in names}) > 1)
    if spanning:
        raise OperationFailure(
            "load", "tied_parameters",
            f"{type(model).__name__}: {spanning} of {len(tied)} tied parameter "
            "group(s) span different FSDP units; a tied weight sharded in one "
            "unit and read flat in another is exactly the job-37300 matmul "
            "mismatch",
        )
# --- end fs172 ---


'''

HELPER_BLOCK = (HELPER_BLOCK_RAW
                .replace("@XFORM@", XFORM_POLICY)
                .replace("@NOSPLIT@", NOSPLIT))

CALL_INSERT = (
    "    # fs172: resolve the wrap policy from the model's OWN declaration before the\n"
    "    # FSDP construction. The census lives in the module-level WRAP_POLICY_CENSUS\n"
    "    # because RuntimeBundle has no suitable field and the dataclass is NOT edited\n"
    "    # here -- the blast radius stays small.\n"
    "    wrap_policy, wrap_census = _resolve_wrap_policy(model)\n"
    "    WRAP_POLICY_CENSUS.update(wrap_census)\n"
)

TIED_INSERT = (
    "    # fs172: tied-parameter control, measured not assumed, and deliberately BEFORE\n"
    "    # the FSDP construction. Post-wrap, FULL_SHARD leaves every parameter this rank\n"
    "    # does not own as an empty tensor whose data_ptr() is 0 (measured), which would\n"
    "    # group unrelated parameters into one false tied group and refuse a healthy run.\n"
    "    # Pre-wrap the pointers are real, and unit membership is already fixed by the\n"
    "    # policy resolved above. An untied model records a measured tied_groups=0.\n"
    "    _check_tied_parameters(model, wrap_census[\"declared_names\"], WRAP_POLICY_CENSUS)\n"
)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _transform(text: str) -> tuple[str, bool]:
    if OPEN_MARK in text:
        return text, True
    new = text.replace(IMPORT_OLD, IMPORT_NEW)
    if not re.search(r"^import functools\b", new, re.M):
        lines = new.splitlines(keepends=True)
        for i, ln in enumerate(lines):
            if ln.rstrip("\r\n") == "import torch":
                lines.insert(i, "import functools\n")
                break
        new = "".join(lines)
    new = new.replace(DEF_BUILD, HELPER_BLOCK + DEF_BUILD, 1)
    new = new.replace(MODEL_LINE + "\n", MODEL_LINE + "\n" + CALL_INSERT + TIED_INSERT, 1)
    new = new.replace(AUTO_OLD, AUTO_NEW)
    return new, False


# --- control harness: fakes, no torch, no transformers ---------------------------


class _OpFail(Exception):
    """Stand-in for the target's OperationFailure(phase, kind, message)."""

    def __init__(self, phase: str, kind: str, message: str) -> None:
        super().__init__(f"{phase}/{kind}: {message}")
        self.phase = phase
        self.kind = kind
        self.message = message


class _FakeParam:
    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def data_ptr(self) -> int:
        return self._ptr


class _FakeModule:
    """Minimal nn.Module stand-in: modules(), named_modules(), named_parameters()."""

    def __init__(self) -> None:
        self._children: dict[str, _FakeModule] = {}
        self._params: dict[str, _FakeParam] = {}

    def add(self, name: str, child: _FakeModule) -> _FakeModule:
        self._children[name] = child
        return child

    def add_param(self, name: str, param: _FakeParam) -> None:
        self._params[name] = param

    def modules(self):
        yield self
        for child in self._children.values():
            yield from child.modules()

    def named_modules(self, prefix: str = ""):
        yield prefix, self
        for name, child in self._children.items():
            yield from child.named_modules(prefix + ("." if prefix else "") + name)

    def _iter_params(self, prefix: str = ""):
        for name, param in self._params.items():
            yield (prefix + "." if prefix else "") + name, param
        for name, child in self._children.items():
            yield from child._iter_params(prefix + ("." if prefix else "") + name)

    def named_parameters(self, prefix: str = "", remove_duplicate: bool = True):
        # Faithful to torch, and that fidelity is the whole point: the real
        # nn.Module.named_parameters DEDUPLICATES by default (measured), so a
        # tied pair arrives as one name. A fake that always yielded both would
        # let C6 certify a detector that can never fire on a real model.
        seen = set()
        for name, param in self._iter_params(prefix):
            if remove_duplicate:
                if id(param) in seen:
                    continue
                seen.add(id(param))
            yield name, param


def _cls(name: str, no_split: list[str] | None = None):
    attrs: dict = {}
    if no_split is not None:
        attrs[NOSPLIT] = no_split
    return type(name, (_FakeModule,), attrs)


def _extract_funcs(new: str) -> dict[str, str]:
    out: dict[str, str] = {}
    tree = ast.parse(new)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "_resolve_wrap_policy", "_check_tied_parameters"):
            seg = ast.get_source_segment(new, node)
            if seg:
                out[node.name] = seg
    return out


def _controls(new: str, pre: str) -> tuple[int, list[str]]:
    notes: list[str] = []
    ok = 0
    srcs = _extract_funcs(new)
    missing = {"_resolve_wrap_policy", "_check_tied_parameters"} - set(srcs)

    def _policy_stub(module, recurse, nonwrapped_numel, transformer_layer_cls=None):
        return True

    ns: dict = {"OperationFailure": _OpFail, "functools": functools,
                XFORM_POLICY: _policy_stub}
    if missing:
        notes.append("C0 extraction: FAIL could not ast-extract "
                     + ", ".join(sorted(missing))
                     + " from the patched text; controls not interpretable")
        return ok, notes
    try:
        for fname in ("_resolve_wrap_policy", "_check_tied_parameters"):
            exec(compile(srcs[fname], "<" + MARK + fname + ">", "exec"), ns)
    except Exception as exc:
        notes.append(f"C0 extraction: FAIL exec of the extracted helpers raised "
                     f"{type(exc).__name__}: {exc}; controls not interpretable")
        return ok, notes
    ok += 1
    notes.append("C0 extraction: PASS both helpers ast-extracted from the patched text and "
                 "exec'd against fakes (this host has no torch and no transformers)")
    resolve = ns["_resolve_wrap_policy"]
    tied_check = ns["_check_tied_parameters"]

    def run(name: str, fn) -> None:
        nonlocal ok
        try:
            fn()
        except Exception as exc:
            notes.append(f"{name}: FAIL {type(exc).__name__}: {exc}")
        else:
            ok += 1
            notes.append(f"{name}: PASS")

    def c1() -> None:
        dec = _cls("Fs172StubDecoder")
        model_cls = _cls("Fs172StubForCausalLM", ["Fs172StubDecoder"])
        m = model_cls()
        m.add("block0", dec())
        m.add("block1", dec())
        policy, census = resolve(m)
        assert census["declared"] == 1, census
        assert census["resolved"] == 1, census
        assert census["instances"] == 2, census
        assert census["declared_names"] == ["Fs172StubDecoder"], census
        assert census["model_class"] == "Fs172StubForCausalLM", census
        assert isinstance(policy, functools.partial), type(policy)
        assert policy.func is _policy_stub, policy.func
        assert policy.keywords.get("transformer_layer_cls") == {dec}, policy.keywords

    def c2() -> None:
        names = ["Fs172GemDecoder", "Fs172GemVision", "Fs172GemEncoder", "Fs172GemPool"]
        model_cls = _cls("Fs172GemmaStub", names)
        dec = _cls("Fs172GemDecoder")
        m = model_cls()
        m.add("layer", dec())
        policy, census = resolve(m)  # partial resolution must NOT refuse
        assert isinstance(policy, functools.partial), type(policy)
        assert census["declared"] == 4, census
        assert census["resolved"] == 1, census
        assert census["instances"] == 1, census

    def c3() -> None:
        m = _cls("Fs172BareStub")()
        m.add("x", _cls("Fs172SomeLayer")())
        try:
            resolve(m)
        except _OpFail as exc:
            assert exc.phase == "load" and exc.kind == "wrap_policy", (exc.phase, exc.kind)
            assert "Fs172BareStub" in exc.message, exc.message
        else:
            raise AssertionError("absent declaration did not refuse")

    def c4() -> None:
        m = _cls("Fs172EmptyStub", [])()
        m.add("x", _cls("Fs172SomeLayer")())
        try:
            resolve(m)
        except _OpFail as exc:
            assert exc.kind == "wrap_policy", exc.kind
            assert "Fs172EmptyStub" in exc.message, exc.message
        else:
            raise AssertionError("empty declaration did not refuse")

    def c5() -> None:
        m = _cls("Fs172ZeroStub", ["Fs172GhostDecoder"])()
        m.add("x", _cls("Fs172RealLayer")())
        try:
            resolve(m)
        except _OpFail as exc:
            assert exc.kind == "wrap_policy", exc.kind
            assert "resolved 0 of 1 declared" in exc.message, exc.message
        else:
            raise AssertionError("zero-resolving declaration did not refuse")

    def c6() -> None:
        dec = _cls("Fs172TiedDecoder")
        model_cls = _cls("Fs172TiedStub", ["Fs172TiedDecoder"])
        # 6a: a tied pair co-located in the ROOT unit -> exactly ONE tied group, no refusal.
        m = model_cls()
        shared = _FakeParam(0xF5172)
        m.add_param("embed_tokens.weight", shared)
        m.add("lm_head", _FakeModule()).add_param("weight", shared)
        census: dict = {}
        tied_check(m, ["Fs172TiedDecoder"], census)
        assert census.get("tied_groups") == 1, census
        # 6b: the same tied pair split across a wrapped class and the root -> refusal.
        m2 = model_cls()
        shared2 = _FakeParam(0xF5173)
        m2.add("embed_tokens", dec()).add_param("weight", shared2)
        m2.add("lm_head", _FakeModule()).add_param("weight", shared2)
        census2: dict = {}
        try:
            tied_check(m2, ["Fs172TiedDecoder"], census2)
        except _OpFail as exc:
            assert exc.kind == "tied_parameters", exc.kind
            assert "1 of 1 tied parameter group(s) span different FSDP units" in exc.message, exc.message
        else:
            raise AssertionError("a spanning tied group did not refuse")
        assert census2.get("tied_groups") == 1, census2

    def c7() -> None:
        dec = _cls("Fs172UntiedDecoder")
        m = _cls("Fs172UntiedStub", ["Fs172UntiedDecoder"])()
        m.add("layer", dec()).add_param("weight", _FakeParam(1))
        m.add_param("head.weight", _FakeParam(2))
        census: dict = {}
        tied_check(m, ["Fs172UntiedDecoder"], census)
        assert "tied_groups" in census, census  # a MEASURED zero, not a silent skip
        assert census["tied_groups"] == 0, census

    def c8() -> None:
        # MUST_FIRE: the pre-image (the size-based policy still in place) must FAIL
        # the gate that requires the transformer policy -- proving G4 can go red.
        gate_holds_on_pre = pre.count(IMPORT_NEW) == 1 and SIZE_POLICY not in pre
        assert not gate_holds_on_pre, "the transformer-policy gate passed on the broken pre-image; it is vacuous"
        assert pre.count(IMPORT_OLD) == 1, "pre-image premise stale: the broken import was not found"

    def c9() -> None:
        # MUST_FIRE against the VACUITY trap, which is the failure mode this
        # detector is most likely to regress into. torch's named_parameters()
        # deduplicates by default (measured), so a tied pair reads as ONE name
        # and a detector calling the default reports tied_groups=0 on the very
        # model whose tie killed job 37300 -- green, and blind. The fake below
        # reproduces torch's dedup faithfully, so if the injected code ever
        # drops remove_duplicate=False this control goes red instead of quiet.
        model_cls = _cls("Fs172DedupStub", ["Fs172DedupDecoder"])
        m = model_cls()
        shared = _FakeParam(0xF5174)
        m.add_param("embed_tokens.weight", shared)
        m.add("lm_head", _FakeModule()).add_param("weight", shared)
        assert len(list(m.named_parameters())) == 1, (
            "fake is not faithful to torch: the default must deduplicate a tied pair")
        assert len(list(m.named_parameters(remove_duplicate=False))) == 2, (
            "fake is not faithful to torch: remove_duplicate=False must yield both names")
        census: dict = {}
        tied_check(m, ["Fs172DedupDecoder"], census)
        assert census.get("tied_groups") == 1, (
            "the tied detector saw %r group(s) on a tied model: it is reading the "
            "DEDUPLICATED default and cannot fire" % (census.get("tied_groups"),))

    run("C1 declared-and-present stub resolves", c1)
    run("C2 partial resolution (declares 4, instantiates 1) passes and records resolved=1 declared=4", c2)
    run("C3 absent declaration refuses", c3)
    run("C4 empty declaration refuses", c4)
    run("C5 declaration resolving to zero live instances refuses", c5)
    run("C6 tied stub detected as ONE tied group (and a spanning group refuses)", c6)
    run("C7 untied stub records a measured tied_groups=0", c7)
    run("C8 MUST_FIRE pre-image fails the transformer-policy gate", c8)
    run("C9 MUST_FIRE tied detector reads remove_duplicate=False, not the vacuous default", c9)
    return ok, notes


def main() -> int:
    # The build driver invokes every stage as `python3 <stage>` with NO arguments, so
    # bare invocation must APPLY. Requiring a flag here would have made the stage a
    # no-op inside the build while passing by hand -- the #86 orphan shape, one
    # layer up.
    argv = sys.argv[1:]
    if argv not in ([], ["--apply"], ["--check"]):
        _stderr("usage: patch_fsdp_wrap_policy.py [--apply|--check]   (no argument == --apply)")
        return 96
    apply = argv != ["--check"]
    if not TARGET.exists():
        _stderr(f"UNMEASURED 95: target missing: {TARGET}")
        return 95
    try:
        text = TARGET.read_text("utf-8")
    except OSError as exc:
        _stderr(f"UNMEASURED 95: target unreadable: {exc}")
        return 95

    new, already = _transform(text)
    if already:
        print("verdict: already applied; byte-idempotent no-op")
        return 0

    gates = 0
    gres: list[tuple[str, bool, str]] = []

    gres.append(("G1", text.count(IMPORT_OLD) == 1 and text.count(AUTO_OLD) == 1,
                 f"pre-image broken policy: import={text.count(IMPORT_OLD)} "
                 f"call-site={text.count(AUTO_OLD)} need=1/1"))
    pre_functools = bool(re.search(r"^import functools\b", text, re.M))
    torch_anchor = sum(1 for ln in text.splitlines() if ln.rstrip("\r\n") == "import torch")
    g2 = (text.count(DEF_BUILD) == 1 and text.count(MODEL_LINE) == 1
          and text.count(SHARD_EXCEPT) == 1 and (pre_functools or torch_anchor == 1))
    gres.append(("G2", g2,
                 f"anchors: build_runtime={text.count(DEF_BUILD)} "
                 f"model-build={text.count(MODEL_LINE)} "
                 f"sharding-except={text.count(SHARD_EXCEPT)} need=1/1/1; "
                 f"functools pre-imported={pre_functools} or unique 'import torch' "
                 f"anchor lines={torch_anchor} need=True-or-1"))
    rb = re.search(r"class RuntimeBundle\b(.*?)(?:\nclass |\ndef )", text, re.S)
    rb_body = rb.group(1) if rb else ""
    gres.append(("G3", rb is not None and "census" not in rb_body and "wrap_policy" not in rb_body,
                 f"RuntimeBundle dataclass found={rb is not None}, census-like field present="
                 f"{'census' in rb_body or 'wrap_policy' in rb_body} need=False (no suitable field, so the "
                 "census goes to the module-level WRAP_POLICY_CENSUS and the dataclass is NOT edited)"))
    gres.append(("G4", IMPORT_OLD not in new and SIZE_POLICY not in new and new.count(IMPORT_NEW) == 1,
                 f"post-image: old import={new.count(IMPORT_OLD)} old policy name anywhere="
                 f"{new.count(SIZE_POLICY)} transformer import={new.count(IMPORT_NEW)} need=0/0/1 "
                 "(an unused import of the broken policy invites a future re-introduction)"))
    g5 = new.count(AUTO_NEW) == 1 and all(text.count(a) == 1 and new.count(a) == 1 for a in FSDP_ARGS)
    gres.append(("G5", g5,
                 f"auto_wrap_policy=wrap_policy count={new.count(AUTO_NEW)} need=1; every other FSDP "
                 "argument byte-identical pre/post: "
                 + ", ".join(f"{a.strip().rstrip(',')}={text.count(a)}/{new.count(a)}" for a in FSDP_ARGS)
                 + " need=1/1 each"))
    i_open = new.find(OPEN_MARK)
    i_close = new.find(CLOSE_MARK)
    i_build = new.find(DEF_BUILD)
    i_resolve = new.find("def _resolve_wrap_policy(")
    g6 = (new.count(OPEN_MARK) == 1 and new.count(CLOSE_MARK) == 1
          and new.count("def _resolve_wrap_policy(") == 1
          and new.count("def _check_tied_parameters(") == 1
          and new.count("WRAP_POLICY_CENSUS") >= 3
          and -1 < i_open < i_resolve < i_close < i_build)
    gres.append(("G6", g6,
                 f"markers once={new.count(OPEN_MARK)}/{new.count(CLOSE_MARK)} open={i_open} "
                 f"resolve={i_resolve} close={i_close} build_runtime={i_build} "
                 "(the helper must sit immediately before build_runtime)"))
    n_functools = len(re.findall(r"^import functools\b", new, re.M))
    gres.append(("G7", n_functools == 1,
                 f"import functools count={n_functools} need=1 "
                 f"({'pre-existing, not re-added' if pre_functools else 'added because the pre-image lacked it; checked, not assumed'})"))
    census_keys = ('"declared_names"', '"declared"', '"resolved"', '"instances"',
                   '"model_class"', '"tied_groups"')
    missing_keys = [k for k in census_keys if k not in new]
    gres.append(("G8", not missing_keys,
                 f"census keys missing={missing_keys} need=[] "
                 "(declared_names/declared/resolved/instances/model_class, plus tied_groups measured later)"))
    try:
        compile(new, str(TARGET), "exec")
        g9, g9d = True, "compile() clean (the python analog of bash -n)"
    except SyntaxError as exc:
        g9, g9d = False, f"compile() SyntaxError: {exc}"
    gres.append(("G9", g9, g9d))
    again, already_again = _transform(new)
    gres.append(("G10", again == new and already_again,
                 "byte-idempotence on own output (a second run is a byte-exact no-op and exits 0)"))
    i_tied = new.find(TIED_CALL)
    i_fsdp = new.find(FSDP_CTOR)
    g11 = (new.count(NP_ANY) == 1 and new.count(NP_SAFE) == 1
           and text.count(NP_ANY) == 0 and -1 < i_tied < i_fsdp)
    gres.append(("G11", g11,
                 f"tied read: pre-image named_parameters={text.count(NP_ANY)} need=0; post-image "
                 f"total={new.count(NP_ANY)} dedup-safe={new.count(NP_SAFE)} need=1/1 (the torch "
                 f"default deduplicates a tied pair, which would make the check vacuous); call "
                 f"site={i_tied} FSDP construction={i_fsdp} need=tied-before-FSDP (post-wrap an "
                 "unowned shard has data_ptr()==0 and would form one false tied group)"))

    for name, good, detail in gres:
        print(f"{name}: {'PASS' if good else 'FAIL'}  {detail}")
        gates += int(good)
    cok, cnotes = _controls(new, text)
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
    # This stage exists to stop a wrong-but-green policy from reaching a matmul 400
    # lines away, so it must not collapse its own states either: an unhandled
    # exception is a REFUSE, not a bare rc=1.
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberate: no traceback may escape as rc=1
        _stderr(f"REFUSE 96: {MARK} stage raised {type(exc).__name__}: {exc}")
        return 96


if __name__ == "__main__":
    raise SystemExit(_guarded())