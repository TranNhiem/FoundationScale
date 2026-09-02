"""foundationscale.gates.adjudication -- the checkpoint-adjudication decision API.

This module is the production decision layer for adjudicating a checkpoint written
by a LIVE training run: it derives what the run DECLARED it would write, measures
what is actually on disk, runs the registered gates over the two, exercises the
MUST_FIRE controls that prove those gates can still fire, and returns a decision.

It was extracted verbatim from ``tools/live_save_gate.py`` (review finding
T2_lib_script_boundary#0). Before the extraction the entire decision API -- 71
module-level symbols, 2,430 of that script's 2,841 lines -- lived in a script, so
the only supported way to reach it was to shell out to a CLI. Nothing importable
adjudicated a checkpoint. ``tools/live_save_gate.py`` is now an argparse wrapper
over this module and re-exports every name below, so every existing caller,
test and mutation anchor resolves unchanged.

Import-time cost is deliberately low: the heavy readers (torch, safetensors) are
imported lazily by ``foundationscale.checkpoint`` inside the functions that need
them, so importing this module on a login node with nothing installed is safe.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from foundationscale.checkpoint.dcp import CheckpointFormatError
from foundationscale.checkpoint.dcp_meta import CheckpointMetadata, read_metadata

# Private names imported deliberately, per the controls contract established by
# real_checkpoint_probe.py: controls must exercise the gate's OWN selection
# logic, not a paraphrase; byte pricing must use the gate's OWN dtype table. If
# these drift, this import failing loudly is the intended signal.
#
# _SHARD_SUFFIX_RE and _expert_weights are not called anywhere below BY DESIGN:
# importing them IS the tripwire (a rename or deletion in checkpoint_gates.py
# raises here, at import time, before any gate runs). The two ruff F401
# findings are therefore discharged with scoped per-member suppressions on the
# member lines below (spelled out there, not here -- writing the directive in
# prose makes ruff read THIS line as a malformed directive), carrying this
# reasoning, never by deleting the names -- removal
# would silently strip a tripwire that exists on purpose, a regression
# disguised as hygiene. The alternative of making the names artificially
# "live" was weighed and rejected on doctrinal grounds: an isinstance/callable
# assertion would assert facts about checkpoint_gates.py's internals that this
# file cannot verify from its own evidence (a claim broader than its evidence,
# and a false shape-assertion would itself become the next generation's
# mystery tripwire), while a bare re-reference tuple would add zero signal.
#
# fix25 adds one more deliberate private member: _expert_named. The probe's
# dense-declaration census classifies with `_expert_named(fqn) or
# _matches_expert_family(fqn)` so the census names EXACTLY the population the
# expert gates would examine; the base-header census wired into
# derive_declared_block adopts the same pair for the same reason -- a census
# over a paraphrased classifier could read 0 where the gates would have
# counted, corroborating "dense" over a population the gates' own empty-set
# door would have indicted.
from foundationscale.gates.checkpoint_gates import (
    _DTYPE_BYTES,
    _SHARD_SUFFIX_RE,  # noqa: F401 -- drift tripwire by design; see the block above
    CheckpointGateContext,
    ExpertByteVolumeGate,
    ExpertDistinctnessGate,
    FirstSaveGate,
    SaveCompletenessGate,
    TensorMeta,
    _expert_named,
    _expert_weight_candidates,
    _expert_weights,  # noqa: F401 -- drift tripwire by design; see the block above
    _layer_normalized_stem,
    _matches_expert_family,
    _split_expert_layouts,
)
from foundationscale.gates.core import Gate, GateResult, Verdict

# The probe's independent-declaration machinery and its REAL-artifact alias
# control now live IN the package: foundationscale.gates.probe. Defect #219:
# they used to be imported from the sibling script tools/real_checkpoint_probe.py
# behind a try/except ImportError ladder with a _PROBE_IMPORT_ERROR sentinel,
# but tools/ is not distributed ([tool.setuptools.packages.find]
# where = ["src"]), so on a clean pip install the ladder always fell through
# and every call into the decision path refused -- the headline API was
# unusable exactly where it was installed. The dependency is inverted (same
# rule as T2_lib_script_boundary#0: the library owns the logic, the script is
# a thin CLI that imports it back), so this import either succeeds or the
# package does not load at all, which is the correct fail-closed behaviour.
# The ladder, the sentinel, the Optional-typed slots and the Protocol
# declarations that served them are all dead and removed. The alias names
# keep the pre-inversion slot names so every downstream call site in this
# file resolves byte-identically.
from foundationscale.gates.probe import derive_declared as _probe_derive_declared
from foundationscale.gates.probe import run_alias_control as _probe_alias_control

EXIT_CLEAR = 0
EXIT_BLOCKED = 1
EXIT_UNMEASURED = 3  # 2 is argparse's; kept distinct from the probe's contract

_ALIAS_STORAGE_ID = "live-gate://injected-alias"

# Gates run on every save. The FirstSaveGate composite is added ONLY for
# --event first_save: its contract is scoped to Lifecycle.FIRST_SAVE, and
# running it on save #347 would be a claim outside its own declared events.
_ALWAYS_GATES: tuple[type[Gate], ...] = (
    ExpertDistinctnessGate,
    ExpertByteVolumeGate,
    SaveCompletenessGate,
)

_ST_DTYPE = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "BOOL": "bool",
}

# Candidate keys in a launcher-resolved training config. # UNVERIFIED: the
# actual key names in this estate's recipe config (README shows run_recipe.py
# --peft_scheme; the full dump of the resolved config was not supplied).
_KIND_KEYS = ("peft_scheme", "peft", "run_kind", "training_mode", "scheme")
_RANK_KEYS = ("lora_rank", "lora_r", "r", "peft_rank", "lora.dim")
_TARGET_KEYS = ("lora_targets", "target_modules", "peft_targets", "lora.target_modules")
_FREEZE_KEYS = ("freeze_modules", "frozen_regex", "trainable_modules", "freeze")


# ---------------------------------------------------------------------------
# Adapter naming: two halves, two TYPES of knob, one enforced agreement
# ---------------------------------------------------------------------------
#
# Recognition and generation are different jobs with honestly different inputs:
# a regex can RECOGNIZE ".lora_A.weight" but cannot GENERATE it -- you cannot
# synthesise one character of an FQN out of `/\.(lora_[AB](?:\.weight)?)$/`.
# The pre-patch tree wired the single --adapter-suffix regex knob to both
# consumers anyway, and only the recognizer read it: the generator hardcoded
# the PEFT literals while the adjacent comment told the operator to calibrate
# the knob. A CORRECT calibration then produced a declared set disjoint from
# the recognized set -- a ~100%-missing false alarm fired BY the act of
# calibrating correctly, and every scrap of its evidence pointed at the
# checkpoint, away from the knobs. The literals now live in their own
# parameters; the regex keeps only the recognition job; and the agreement
# between the two is CHECKED before any verdict exists, because under this
# project's doctrine an agreement that was never checked is a vacuous
# agreement -- independently configurable halves with no consistency check is
# the same defect wearing a second knob.

# The two adapter-naming conventions this tool can be calibrated to, each
# established by reading the code that WRITES it -- a naming default in this: a
# tool is a claim about what its silent consumers will save, and a claim
# broader than its evidence is the defect this file exists to catch.
#
# MEGATRON-BRIDGE (this estate's ONLY LoRA stack; fix30 measurement against
# $REPO/src/megatron/bridge/peft/): peft/lora.py LoRA.transform wraps each
# matched parallel linear as LoRALinear(base, ParallelLinearAdapter(...));
# TEFusedLoRALinear subclasses LoRALinear and LoRATopKRouter shares the same
# AdapterWrapper base, so all three save identically. AdapterWrapper
# (peft/adapter_wrapper.py) writes the adapter under f"{prefix}adapter." in
# BOTH state_dict and sharded_state_dict, and ParallelLinearAdapter
# (peft/utils.py) writes linear_in./linear_out. immediately inside that, each
# built bias=False so .weight is the only tensor it owns. linear_in is
# constructed in_features -> dim, weight (dim, in_features) = (rank, in);
# linear_out is dim -> out_features, weight (out, rank). A real save of this
# estate therefore carries, per adapted linear:
#     <base_linear_fqn>.adapter.linear_in.weight   -- the A matrix, (rank, in)
#     <base_linear_fqn>.adapter.linear_out.weight  -- the B matrix, (out, rank)
# HF PEFT (kept as a named, explicit preset; no current consumer in this
# estate): <parent>.lora_A.weight is (rank, in), <parent>.lora_B.weight is
# (out, rank).
_HF_PEFT_ADAPTER_SUFFIX_RE = r"\.(lora_[AB](?:\.weight)?)$"
_HF_PEFT_ADAPTER_SUFFIX_A = ".lora_A.weight"
_HF_PEFT_ADAPTER_SUFFIX_B = ".lora_B.weight"
_HF_PEFT_ADAPTER_SUFFIXES = (_HF_PEFT_ADAPTER_SUFFIX_A, _HF_PEFT_ADAPTER_SUFFIX_B)

_MEGATRON_BRIDGE_ADAPTER_SUFFIX_RE = r"\.adapter\.linear_(?:in|out)\.weight$"
_MEGATRON_BRIDGE_ADAPTER_SUFFIX_A = ".adapter.linear_in.weight"
_MEGATRON_BRIDGE_ADAPTER_SUFFIX_B = ".adapter.linear_out.weight"
_MEGATRON_BRIDGE_ADAPTER_SUFFIXES = (
    _MEGATRON_BRIDGE_ADAPTER_SUFFIX_A,
    _MEGATRON_BRIDGE_ADAPTER_SUFFIX_B,
)

# THE SHIPPED DEFAULTS ARE THE MEGATRON-BRIDGE SET (fix30). They were born the
# HF set, and the default WAS the calibration: no caller in this estate passes
# --adapter-* flags, so the HF defaults matched ZERO tensors of every adapter
# save this estate can produce, and the structural sweep answered every
# healthy LoRA artifact with the 0-of-N vacuity refusal -- fail-closed
# machinery, wrong constant. Shape convention stays POSITIONAL across both
# conventions and applies to any calibrated replacement pair: the first
# template generates the (rank, in_features) matrix's FQN, the second the
# (out_features, rank) matrix's -- the (A, B) of an (A @ B) low-rank update.
# Recognizer shape, argued from the same measurement: the suffix is CONTIGUOUS
# in the save FQN (the wrapper glues "adapter." straight onto the module's own
# prefix chain and the adapter glues "linear_in./linear_out." straight inside
# it), it names the ONLY two tensors an adapter owns, and it is END-ANCHORED
# so a future "...linear_in.weight_extra" cannot silently extend a match. It
# deliberately does NOT match the other two layouts this peft tree can emit
# (CanonicalLoRA's ...adapter.adapter_q/k/v/up/gate.linear_* one ModuleDict
# segment deeper, and LinearAdapter/TELinearAdapter's self-mounted
# <module>.linear_in/out.weight with no ".adapter." segment): those are
# refused BY NAME in the sweep below, because a regex wide enough to catch
# three layouts adjudicates shapes it was never calibrated against.
_DEFAULT_ADAPTER_SUFFIX_RE = _MEGATRON_BRIDGE_ADAPTER_SUFFIX_RE
_DEFAULT_ADAPTER_SUFFIX_A = _MEGATRON_BRIDGE_ADAPTER_SUFFIX_A
_DEFAULT_ADAPTER_SUFFIX_B = _MEGATRON_BRIDGE_ADAPTER_SUFFIX_B
_DEFAULT_ADAPTER_SUFFIXES = (_DEFAULT_ADAPTER_SUFFIX_A, _DEFAULT_ADAPTER_SUFFIX_B)

# fix30(c): the two OTHER adapter layouts the estate's peft tree can emit,
# spelled so the structural sweep can refuse them BY NAME instead of filing
# them under the generic vacuity text. The self-mounted pattern's lookbehind
# excludes only the calibrated plain-LoRA shape; CanonicalLoRA FQNs also
# satisfy it one segment deeper, so the sweep classifies canonical FIRST and
# every tensor is counted under exactly one layout name (doctrine 2: the
# counts reported below must partition the population they name, and fixing
# the classification order is what makes the partition hold).
_CANONICAL_LORA_LAYOUT_RE = re.compile(
    r"\.adapter\.adapter_(?:q|k|v|up|gate)\.linear_(?:in|out)\.weight$"
)
_SELF_MOUNTED_ADAPTER_LAYOUT_RE = re.compile(r"(?<!\.adapter)\.linear_(?:in|out)\.weight$")


def _verify_adapter_naming_agreement(
    adapter_suffix_re: str,
    adapter_prefix: str,
    adapter_suffixes: tuple[str, str],
) -> None:
    """Refuse (GateUnmeasured, -> UNMEASURED) if generator and recognizer disagree.

    Called at the top of adjudication, before any artifact is measured and
    before any gate or control executes: a knob disagreement is a TOOL-
    configuration defect, not a property of the checkpoint, so it must never
    surface as a checkpoint BLOCK verdict (exit 1). It refuses (exit 3) and
    names exactly which elements disagree.

    The check cannot compare the knobs textually -- one side is a regex, the
    other literals, and "do they agree" has no string answer across types. So
    it ROUND-TRIPS: build a synthetic adapter FQN exactly the way
    derive_declared_block builds real ones (prefix + parent stem + literal
    template), then cut it exactly the way lora_structural_findings cuts real
    artifact FQNs (regex search, slice to match start, conditional prefix
    strip), and demand that the parent stem put in comes back out. This
    catches both failure modes, not just the obvious one: a recognizer that
    does not match the templates at all makes the declared and recognized sets
    disjoint (the defect's headline), and a recognizer that matches but
    consumes too little -- a bare ``lora_[AB]`` without the literal dots and
    the end anchor -- recovers a stem with characters glued on, and would
    phantom-block every healthy adapter the estate saves.
    """
    if len(adapter_suffixes) != 2:
        raise GateUnmeasured(
            f"adapter_suffixes must be a (suffix_a, suffix_b) pair, got "
            f"{adapter_suffixes!r} -- the (rank, in)/(out, rank) shape "
            f"convention is positional and has no meaning at any other arity"
        )
    suffix_a, suffix_b = adapter_suffixes
    if suffix_a == suffix_b:
        raise GateUnmeasured(
            f"adapter naming disagreement (refusing to run before any verdict "
            f"is issued): --adapter-suffix-a and --adapter-suffix-b are "
            f"identical ({suffix_a!r}) -- both templates would declare the "
            f"SAME fqn with two different shapes, and one would silently "
            f"overwrite the other in the derived map"
        )
    try:
        recognizer = re.compile(adapter_suffix_re)
    except re.error as exc:
        raise GateUnmeasured(
            f"--adapter-suffix is not a compilable regex: {adapter_suffix_re!r} "
            f"({exc}) -- the recognizer is unexercisable, not merely miscalibrated"
        ) from exc
    parent_stem = "calibration.parent"
    problems: list[str] = []
    for label, suffix in (("--adapter-suffix-a", suffix_a), ("--adapter-suffix-b", suffix_b)):
        if not isinstance(suffix, str) or not suffix:
            problems.append(f"{label} must be a non-empty literal string, got {suffix!r}")
            continue
        candidate = f"{adapter_prefix}{parent_stem}{suffix}"
        match = recognizer.search(candidate)
        if match is None:
            problems.append(
                f"{label} {suffix!r} generates adapter FQNs (e.g. {candidate!r}) "
                f"that the recognizer --adapter-suffix /{adapter_suffix_re}/ does "
                f"not match at all: declared and recognized sets would be "
                f"disjoint -- the exact false alarm this check exists to refuse"
            )
            continue
        recovered = candidate[: match.start()]
        if adapter_prefix and recovered.startswith(adapter_prefix):
            # Mirrored from lora_structural_findings: the prefix is stripped
            # only where present, never silently corrected where absent.
            recovered = recovered[len(adapter_prefix) :]
        if recovered != parent_stem:
            problems.append(
                f"{label} {suffix!r} is matched by --adapter-suffix "
                f"/{adapter_suffix_re}/, but cutting at the match recovers "
                f"parent {recovered!r} instead of {parent_stem!r} -- leftover "
                f"characters glue into the parent lookup and phantom-block "
                f"healthy adapters; the recognizer must consume the generated "
                f"suffix exactly (anchor it, as the default "
                f"{_DEFAULT_ADAPTER_SUFFIX_RE!r} does)"
            )
    if problems:
        raise GateUnmeasured(
            "adapter naming disagreement (refusing to run before any verdict "
            "is issued): " + " | ".join(problems) + ". Calibrate "
            "--adapter-suffix, --adapter-suffix-a and --adapter-suffix-b "
            "TOGETHER against one saved adapter, then pin them in the wrapper."
        )


class GateUnmeasured(RuntimeError):
    """The tool could not measure. Distinct from 'measured, and it blocks'."""


_REFUSAL_ADAPTER_PREFIX_UNPINNED = "adapter_prefix_unpinned"
_REFUSAL_ADAPTER_CENSUS_UNAVAILABLE = "adapter_census_unavailable"
_REFUSAL_CHECKPOINT_UNREADABLE = "checkpoint_unreadable"
_REFUSAL_OTHER = "other_unmeasured"


def _refusal_class(message: str) -> str:
    """Classify a GateUnmeasured refusal for the ON-DISK refusal record.

    Two consumers must never drift apart (fix44 / #77-B2/B3): the launcher
    maps exactly ONE member of the exit-3 class -- the deliberately unpinned
    --adapter-prefix, a CHOSEN abstention -- to a calibrated rc 0, and every
    other member (unreadable artifact, missing base files, unresolvable mode,
    tool bug, missing --adapter-modules census) to the rc-92 wiring class. It
    reads this classification out of the JSON record this CLI writes on every
    unmeasured exit. The string constants are the whole contract:
    reclassifying a cause is a deliberate act here, never a side effect of
    rewording an error message downstream. ORDER IS LOAD-BEARING: since #78
    the adapter-prefix refusal text names --adapter-modules in its guidance
    (the census file is the fix path for the namespace split), so the prefix
    arm is tested FIRST -- with the order inverted, every prefix refusal
    would misclassify as a census refusal and silently change which launcher
    arm it lands on. The launcher today calibrates only the prefix member;
    the census member lands rc-92 there until a coordinate edit teaches the
    launcher about the wired census, which is the intended direction.
    """
    if message.startswith("--adapter-prefix was not pinned"):
        return _REFUSAL_ADAPTER_PREFIX_UNPINNED
    if "--adapter-modules" in message:
        return _REFUSAL_ADAPTER_CENSUS_UNAVAILABLE
    if "checkpoint unreadable:" in message:
        return _REFUSAL_CHECKPOINT_UNREADABLE
    return _REFUSAL_OTHER


# ---------------------------------------------------------------------------
# Independent source A: the base model (config.json + safetensors header)
# ---------------------------------------------------------------------------


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise GateUnmeasured(f"{what} not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateUnmeasured(f"{what} unreadable: {path} ({exc!r})") from exc
    if not isinstance(raw, dict):
        raise GateUnmeasured(f"{what} is not a JSON object: {path}")
    return raw


def _read_safetensors_header(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Parse ONLY the header of one safetensors shard (8-byte len + JSON)."""
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                raise GateUnmeasured(f"safetensors shard too short: {path}")
            (header_len,) = struct.unpack("<Q", raw_len)
            header = json.loads(fh.read(header_len).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as exc:
        raise GateUnmeasured(f"cannot parse safetensors header {path}: {exc!r}") from exc
    header.pop("__metadata__", None)
    out: dict[str, tuple[tuple[int, ...], str]] = {}
    for fqn, entry in header.items():
        dtype = _ST_DTYPE.get(str(entry.get("dtype", "")).upper())
        if dtype is None:
            # Not coerced: an unknown dtype prices bytes wrong; surface it.
            raise GateUnmeasured(
                f"base tensor {fqn!r} in {path} has unrecognized safetensors "
                f"dtype {entry.get('dtype')!r} — extend _ST_DTYPE deliberately, "
                f"do not default"
            )
        out[fqn] = (tuple(int(d) for d in entry["shape"]), dtype)
    return out


@dataclass(frozen=True)
class BaseModel:
    """The independent base-model artifact, read fresh from disk."""

    model_dir: Path
    config: dict[str, Any]
    tensors: dict[str, tuple[tuple[int, ...], str]]  # fqn -> (shape, dtype)
    tensors_source: str

    @classmethod
    def load(cls, model_dir: Path) -> BaseModel:
        if not model_dir.is_dir():
            raise GateUnmeasured(f"base model dir not found: {model_dir}")
        config = _read_json(model_dir / "config.json", "base model config.json")
        idx = model_dir / "model.safetensors.index.json"
        if idx.is_file():  # sharded base
            weight_map = _read_json(idx, "safetensors index").get("weight_map", {})
            if not weight_map:
                raise GateUnmeasured(f"index has empty weight_map: {idx}")
            tensors: dict[str, tuple[tuple[int, ...], str]] = {}
            for shard in sorted(set(weight_map.values())):
                tensors.update(_read_safetensors_header(model_dir / shard))
            source = f"{idx} ({len(set(weight_map.values()))} shards)"
        else:
            single = model_dir / "model.safetensors"
            tensors = _read_safetensors_header(single)
            source = str(single)
        if not tensors:
            raise GateUnmeasured(f"base model exposes zero tensors under {model_dir}")
        return cls(model_dir=model_dir, config=config, tensors=tensors, tensors_source=source)


# ---------------------------------------------------------------------------
# Independent source B: the training config (launcher-resolved)
# ---------------------------------------------------------------------------


def _load_train_config(path: Path | None) -> tuple[dict[str, Any], str]:
    """Accept a JSON object or a KEY=VALUE dump (e.g. `env` snapshot from the launcher).

    The estate's launchers are env-driven; instruct the operator to snapshot the
    RESOLVED config at submit time:
        declare -p TRAIN_ITERS PEFT_SCHEME ... > $OUT_DIR/resolved-train-config.env
    and pass that file here. JSON preferred."""
    if path is None:
        return {}, "no --train-config supplied"
    text = path.read_text(encoding="utf-8") if path.is_file() else None
    if text is None:
        raise GateUnmeasured(f"--train-config not found: {path}")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, f"{path} (JSON)"
    except json.JSONDecodeError:
        pass
    kv: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip().removeprefix("declare -x ").removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        kv[k.strip()] = v.strip().strip("'\"")  # values stay strings; ints coerced below
    return kv, f"{path} (KEY=VALUE)"


def _load_fqn_map(path: Path) -> tuple[tuple[str, ...], str]:
    """Parse the operator-supplied --fqn-map: the denominator the low-overlap
    full-FT basis text has pointed at all along.

    Accepts a bare JSON list of artifact-namespace FQNs, or an object carrying
    a 'declared_fqns' (or 'fqns') list. Everything malformed is GateUnmeasured,
    i.e. UNMEASURED -- an operator handed a blocking basis that names a flag is
    owed a flag whose failure modes also fail closed. Above all: an EMPTY map
    is refused at the source, because an empty declared set is the vacuity this
    file exists to refuse, wearing a provenance costume."""
    if not path.is_file():
        raise GateUnmeasured(f"--fqn-map not found: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateUnmeasured(f"--fqn-map unreadable: {path} ({exc!r})") from exc
    if isinstance(obj, dict):
        seq = obj.get("declared_fqns", obj.get("fqns"))
        if seq is None:
            raise GateUnmeasured(
                f"--fqn-map object must carry a 'declared_fqns' (or 'fqns') list: {path}"
            )
    elif isinstance(obj, list):
        seq = obj
    else:
        raise GateUnmeasured(
            f"--fqn-map must be a JSON list or object, got {type(obj).__name__}: {path}"
        )
    bad = [s for s in seq if not isinstance(s, str) or not s]
    if bad:
        raise GateUnmeasured(
            f"--fqn-map contains {len(bad)} non-string/empty entries (first: {bad[0]!r}): {path}"
        )
    fqns = tuple(seq)
    if not fqns:
        raise GateUnmeasured(
            f"--fqn-map declares ZERO fqns: {path} -- delete the flag or populate "
            f"the map; an empty denominator cannot adjudicate anything"
        )
    return fqns, (
        f"--fqn-map file {path} ({len(fqns)} artifact-namespace FQNs; provenance: "
        f"exported by the planner/operator at submit time, never read from the run)"
    )


@dataclass(frozen=True)
class _AdapterModuleCensus:
    """The parsed --adapter-modules census (#78): adapter-target module stems
    in the ARTIFACT's namespace, optional per-stem parent dims
    (out_features, in_features), and the provenance text every report repeats.
    Frozen: a denominator that can be mutated after parsing is two
    denominators."""

    stems: tuple[str, ...]
    dims: dict[str, tuple[int, int]] | None
    basis: str


def _load_adapter_modules(path: Path, *, judged_dir: Path) -> _AdapterModuleCensus:
    """Parse --adapter-modules: the lora declared denominator (#78).

    Contract (producer side, stated once, here): a JSON list of module-FQN
    strings, or a JSON object carrying 'adapter_modules' (or 'modules') whose
    entries are strings or {'fqn', 'out_features', 'in_features'}; 'source' /
    'producer' are carried into the basis text. The intended producer is the
    launcher at SUBMIT time, persisting the step-(5) live-module census (the
    names the census computes with the shipped matcher over the BASE tree)
    to fs_gate/adapter-modules.json. Everything malformed is GateUnmeasured
    (exit 3) -- fail closed, the same discipline as _load_fqn_map:
      * empty list            -- a zero denominator adjudicates nothing
                                 (doctrine 1);
      * duplicates            -- a census with duplicates is a broken census;
                                 silent dedup would hide the break;
      * partial dims          -- shape knowledge is all-or-nothing, or the
                                 (rank, in)/(out, rank) check would compare
                                 against an unstated mixture;
      * non-positive dims     -- a bogus (out, in) pair would mint wrong
                                 shapes with an authoritative face;
      * inside the judged tree -- the tautology guard, enforced IN CODE:
                                 a denominator read from the tree under
                                 judgment is the all([]) shape with better
                                 bookkeeping, whichever flag it rode in on.
    What this loader deliberately CANNOT verify and therefore says in the
    basis rather than hides: that an unnamed hand actually ran the launch-time
    census rather than hand-typing stems. A planner-frozen, versioned
    expectation (the --fqn-map producer's shape, written by the conversion
    pipeline before any run exists) is the fully independent form; wiring
    that is named in the flag's help text, not absorbed here by guessing.
    """
    try:
        census_resolved = path.resolve()
        judged_resolved = judged_dir.resolve()
    except OSError as exc:
        raise GateUnmeasured(
            f"--adapter-modules: cannot resolve the census or the judged path "
            f"({exc!r}) -- the denominator's independence from the judged "
            f"tree is itself unverifiable here, and that refuses; it never "
            f"defaults to trusted"
        ) from exc
    if census_resolved == judged_resolved or census_resolved.is_relative_to(judged_resolved):
        raise GateUnmeasured(
            f"--adapter-modules resolves INSIDE the tree under judgment "
            f"(census {census_resolved} vs judged {judged_resolved}) -- a "
            f"denominator read from the judged tree is the all([]) tautology "
            f"with better bookkeeping; the census must come from the "
            f"INDEPENDENT base tree at launch time"
        )
    if not census_resolved.is_file():
        raise GateUnmeasured(f"--adapter-modules not found: {path}")
    try:
        obj = json.loads(census_resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateUnmeasured(f"--adapter-modules unreadable: {path} ({exc!r})") from exc
    provenance = ""
    if isinstance(obj, dict):
        seq = obj.get("adapter_modules", obj.get("modules"))
        if seq is None:
            raise GateUnmeasured(
                f"--adapter-modules object must carry an 'adapter_modules' "
                f"(or 'modules') list: {path}"
            )
        raw_src = obj.get("source", obj.get("producer"))
        if raw_src:
            provenance = f"; producer recorded in-file: {raw_src}"
    elif isinstance(obj, list):
        seq = obj
    else:
        raise GateUnmeasured(
            f"--adapter-modules must be a JSON list or object, got {type(obj).__name__}: {path}"
        )
    if not isinstance(seq, list):
        raise GateUnmeasured(
            f"--adapter-modules carries a non-list 'adapter_modules' entry "
            f"({type(seq).__name__}): {path}"
        )
    stems: list[str] = []
    dims: dict[str, tuple[int, int]] = {}
    bad = 0
    for entry in seq:
        if isinstance(entry, str) and entry:
            stems.append(entry)
            continue
        if isinstance(entry, dict):
            # JSON booleans are Python ints; exclude them explicitly or a
            # true/true pair would rank as a plausible (out, in) anywhere
            # downstream -- a wrong denominator wearing a valid face.
            fqn = entry.get("fqn")
            out_d = entry.get("out_features")
            in_d = entry.get("in_features")
            if (
                isinstance(fqn, str)
                and fqn
                and isinstance(out_d, int)
                and not isinstance(out_d, bool)
                and isinstance(in_d, int)
                and not isinstance(in_d, bool)
                and out_d > 0
                and in_d > 0
            ):
                stems.append(fqn)
                dims[fqn] = (out_d, in_d)
                continue
        bad += 1
    if bad:
        raise GateUnmeasured(
            f"--adapter-modules contains {bad} of {len(seq)} malformed "
            f"entries (a non-empty string module FQN, or "
            f"{{'fqn', 'out_features', 'in_features'}} with positive ints): "
            f"{path}"
        )
    if not stems:
        raise GateUnmeasured(
            f"--adapter-modules declares ZERO modules: {path} -- a zero "
            f"denominator adjudicates nothing (doctrine 1); delete the flag "
            f"or produce a real census"
        )
    dupes = sorted(s for s in set(stems) if stems.count(s) > 1)
    if dupes:
        raise GateUnmeasured(
            f"--adapter-modules contains {len(dupes)} duplicate module(s) of "
            f"{len(stems)} (first: {dupes[0]!r}): {path} -- a census with "
            f"duplicates is a broken census, and silent deduplication would "
            f"hide the break"
        )
    if dims and len(dims) != len(stems):
        raise GateUnmeasured(
            f"--adapter-modules dims cover {len(dims)} of {len(stems)} "
            f"modules: {path} -- parent dimensions must be all-or-nothing; "
            f"a partially-dimmed census would drive the (rank, in)/(out, "
            f"rank) shape check against an unstated mixture"
        )
    basis = (
        f"--adapter-modules file {path} ({len(stems)} artifact-namespace "
        "module stems"
        + (
            f", parent dims for all {len(stems)}"
            if dims
            else ", no parent dims -- shape check abstains by name"
        )
        + provenance
        + (
            ""
            if provenance
            else "; NO producer provenance recorded in-file -- the loader verified "
            "shape/uniqueness/dims/independence-from-the-judged-tree only, "
            "not who wrote it"
        )
        + ")"
    )
    return _AdapterModuleCensus(stems=tuple(sorted(stems)), dims=dims or None, basis=basis)


def _first_key(cfg: dict[str, Any], keys: tuple[str, ...]) -> tuple[Any, str] | tuple[None, None]:
    for k in keys:
        if k in cfg and cfg[k] not in (None, ""):
            return cfg[k], k
    return None, None


@dataclass(frozen=True)
class TrainSpec:
    run_kind: str  # "full" | "lora" | "auto" (auto resolved later, recorded)
    kind_basis: str
    lora_rank: int | None
    rank_basis: str
    lora_targets: tuple[str, ...]
    targets_basis: str
    frozen_regex: re.Pattern[str] | None
    frozen_basis: str
    cfg_source: str


def resolve_train_spec(
    cfg: dict[str, Any], cfg_source: str, run_kind_arg: str, frozen_arg: str | None
) -> TrainSpec:
    kind, kbasis = run_kind_arg, ""
    if run_kind_arg == "auto":
        raw, key = _first_key(cfg, _KIND_KEYS)
        if raw is None:
            kind, kbasis = (
                "auto",
                (
                    f"no peft/kind key ({'/'.join(_KIND_KEYS)}) in {cfg_source}; "
                    f"kind will be inferred from the artifact AFTER measurement and "
                    f"populations cross-checked (denominators stay independent)"
                ),
            )
        elif "lora" in str(raw).lower() or str(raw).lower() in {"peft", "adapter"}:
            kind, kbasis = "lora", f"{cfg_source}: {key}={raw!r}"
        else:
            kind, kbasis = "full", f"{cfg_source}: {key}={raw!r}"
    else:
        kbasis = f"operator override --run-kind {run_kind_arg}"

    rank_raw, rkey = _first_key(cfg, _RANK_KEYS)
    rank: int | None = None
    try:
        rank = int(rank_raw) if rank_raw is not None else None
    except (TypeError, ValueError) as exc:
        # Chain, don't amputate: an operator staring at UNMEASURED deserves to
        # see what int() actually choked on (the raw value is already in our
        # message; the interpreter's own complaint is the corroboration).
        raise GateUnmeasured(f"lora rank value {rank_raw!r} at key {rkey!r} is not an int") from exc
    rbasis = (
        f"{cfg_source}: {rkey}={rank}"
        if rank is not None
        else f"no rank key ({'/'.join(_RANK_KEYS)}) in {cfg_source}"
    )

    tgt_raw, tkey = _first_key(cfg, _TARGET_KEYS)
    if isinstance(tgt_raw, str):
        targets = tuple(s for s in re.split(r"[,\s]+", tgt_raw) if s)
    elif isinstance(tgt_raw, (list, tuple)):
        targets = tuple(str(s) for s in tgt_raw)
    else:
        targets = ()
    tbasis = (
        f"{cfg_source}: {tkey}={list(targets)}"
        if targets
        else f"no target key ({'/'.join(_TARGET_KEYS)}) — recorded for "
        f"provenance only: since #78 the adapter declared set derives "
        f"from the --adapter-modules live-module census, not from "
        f"targets x base header, a cross-namespace product measured "
        f"disjoint from every save this estate can produce"
    )

    fz_raw, fkey = _first_key(cfg, _FREEZE_KEYS)
    fre: re.Pattern[str] | None = None
    fbasis = "no freeze key found; full base population declared trainable+saved"
    if frozen_arg:
        fre = re.compile(frozen_arg)
        fbasis = f"operator --frozen-regex {frozen_arg!r}"
    elif isinstance(fz_raw, str) and fz_raw:
        fre = re.compile(fz_raw)
        # UNVERIFIED: the semantics of this estate's freeze knob -- regex, or a
        # module list? -- are not established from any source this tool reads.
        fbasis = f"{cfg_source}: {fkey}={fz_raw!r}"

    return TrainSpec(kind, kbasis, rank, rbasis, targets, tbasis, fre, fbasis, cfg_source)


# ---------------------------------------------------------------------------
# The declared block: per-run-kind scope over independent sources
# ---------------------------------------------------------------------------


@dataclass
class Declared:
    fqns: tuple[str, ...] | None
    fqns_basis: str
    num_experts: int | None
    experts_basis: str
    num_moe_layers: int | None
    moe_layers_basis: str
    expected_expert_bytes: int | None
    bytes_basis: str
    derived_adapter: dict[str, tuple[int, ...]]  # lora: fqn -> expected shape
    notes: list[str] = field(default_factory=list)
    # lora: the --adapter-modules census stems in artifact namespace (#78).
    # The structural sweep binds every adapter's parent against THIS set when
    # it exists -- the only namespace this estate's saves use. Empty for full
    # runs and for any caller that never supplied a census.
    adapter_modules: tuple[str, ...] = ()


# _gemma4_moe_override was DELETED by fix25, not narrowed. Verified against
# the attached probe source: derive_declared already reads enable_moe_block
# in text_config-first scope (via _enable_moe_block_flag, through the shared
# _ENABLE_MOE_BLOCK_KEY), so this function never had a scope-selection gap to
# fill. Its remaining behavior was `d = {**d, "num_experts": 0}` -- a mint
# from ONE source (the config asserts; nothing corroborates) that overwrote
# whatever the probe decided, including the probe's own refusals: when the
# un-bridged probe abstained (num_experts=None), the note branch did not run
# and the mint still did, which is precisely how "the mint survived while its
# stated basis did not" measured out. The affirmative-statement job belongs
# to the config the probe reads; the verdict belongs to the probe alone
# (MINT_ZERO_ONLY_IN_PROBE, enforced at the call site below). The measured
# estate facts the deleted UNVERIFIED comment asked for are recorded at the
# census site in derive_declared_block.


def _lora_target_attaches(stem: str, fqn: str, target: str) -> bool:
    """One rule for "does this train-config target name THIS base module".

    HISTORY, corrected rather than inherited (a comment naming wrong call
    sites is a claim broader than its evidence): until #78 this predicate
    had exactly two call sites inside derive_declared_block -- the
    adapter-scope expert census and the declared-set derivation -- and the
    old text pinned their anti-drift contract. Those sites derived in the HF
    namespace against a Megatron-namespace save, which is the measured #78
    defect; both are RETIRED (derivation and expert scoping now read the
    launch-time live-module census, --adapter-modules), so no call site
    remains IN THIS FILE. The function is kept, unmodified in behaviour:
    its stem convention (FQN minus one trailing .weight/.bias segment,
    because adapter suffixes attach to the module, not to a tensor) remains
    the documented name-matching lineage and external/test callers may pin
    it. Deleting it would be churn without signal; keeping it with a false
    "two call sites" comment would be the doctrine this file enforces,
    broken in a comment.
    """
    return stem.endswith(target) or f".{target}." in fqn


def derive_declared_block(
    base: BaseModel,
    spec: TrainSpec,
    artifact_real_fqns: set[str],
    adapter_prefix: str,
    adapter_suffixes: tuple[str, str] = _DEFAULT_ADAPTER_SUFFIXES,
    fqn_map: tuple[tuple[str, ...], str] | None = None,
    # --adapter-modules, parsed and judged-tree-checked by adjudicate_checkpoint.
    # None refuses the lora branch (#78: there is no other honest denominator);
    # ignored by the full branch, which keeps its own two sources.
    adapter_modules: _AdapterModuleCensus | None = None,
) -> Declared:
    notes: list[str] = []
    # --- Bridge to the probe's two-source dense contract (fix25) ------------
    # INVARIANT (grep anchor MINT_ZERO_ONLY_IN_PROBE): no code path in this
    # file writes num_experts = 0 for the BASE declaration. A zero expert
    # denominator for the base arrives ONLY as derive_declared's mint: an
    # affirmative dense statement in the config CORROBORATED by the artifact
    # census handed over here. The only other assignment of 0 in this file is
    # the lora ADAPTER-SCOPE statement far below, which declares what THIS
    # artifact was told to contain (train-config targets x base header), not
    # what the base model is; it is commented as the declared exception at
    # its own site, so a grep for the founding defect finds both sites
    # documented.
    #
    # Census source, decided explicitly (fix25-s4): the BASE HEADER, key set
    # UNFILTERED by spec.frozen_regex, classified with the gates' own atoms
    # (_expert_named or _matches_expert_family -- the pair the probe's
    # _census_expert_family uses, per the controls contract against
    # paraphrase).
    #   * NOT in_scope / expert_base: both pass through spec.frozen_regex. A
    #     --frozen-regex matching the expert stem would empty the counted
    #     set and let a real MoE base corroborate a dense declaration -- the
    #     founding incident with a user-supplied mask on. Frozen scope says
    #     which tensors a RUN must save; it is not evidence about what the
    #     BASE model IS.
    #   * NOT artifact_real_fqns: that set answers "what did this run
    #     write", not "does the base have MoE blocks". A LoRA adapter of an
    #     MoE base holds zero expert tensors BY DESIGN, so an artifact
    #     census would corroborate a false dense base for exactly the
    #     adapter population this file exists to adjudicate gently;
    #     DCP-layout full artifacts rename FQNs (the --fqn-map case), so
    #     family classifiers could read 0 artifactually; and this function
    #     is called with artifact_real_fqns=set() by TestMoeOverride, where
    #     an artifact census reads 0 for reasons having nothing to do with
    #     density. One source again.
    #   Denominator (doctrine 2): the tally is over len(base.tensors)
    #   base-header names from base.tensors_source, surfaced twice -- in the
    #   framing note below, and in the probe's own census note relayed
    #   verbatim into decl.notes further down.
    #
    # Measured estate facts (fix25-s6; these REPLACE the deleted UNVERIFIED
    # markers): the real gemma-4-E4B-it config.json carries
    # text_config.enable_moe_block = False AND text_config.num_experts
    # present-but-null (the key exists with a null value -- the probe records
    # present-null as absent: neither an MoE statement nor a contradiction),
    # and its safetensors headers expose zero expert-family tensor names. So
    # on the real base this census reads 0 of N, the two independent sources
    # agree, the mint fires, and the first real launch is CLEAR-able instead
    # of being blocked by the tool's own honesty rule.
    census_fqns = sorted(
        fqn
        for fqn in base.tensors
        if "_extra_state" not in fqn and (_expert_named(fqn) or _matches_expert_family(fqn))
    )
    notes.append(
        f"expert-family census: {len(census_fqns)} of {len(base.tensors)} "
        f"base-header tensor names match the expert classifiers "
        f"({base.tensors_source}); computed over the UNFILTERED base key set "
        f"-- frozen scope governs what a run must save, never what the base "
        f"model is. This census is the artifact half of the probe's "
        f"two-source dense corroboration; where the relayed probe notes "
        f"below say 'the measured artifact', in this tool that artifact IS "
        f"the base header"
    )
    d = _probe_derive_declared(
        base.config,
        expert_family_census=len(census_fqns),
        expert_family_sample=tuple(census_fqns[:8]),
    )
    # Relay the probe's notes verbatim: they are the corroboration record --
    # the probe's own census-count line, present-null-treated-as-absent
    # records, coercion notices -- and the decision report must carry the
    # same evidence the probe would have printed, not a digest of it.
    notes.extend(d["notes"])
    num_experts = d["num_experts"]
    num_moe_layers = d["num_moe_layers"]
    experts_basis = f"base config.json via probe derive_declared: {d['basis']['num_experts']}"
    layers_basis = f"base config.json via probe derive_declared: {d['basis']['num_moe_layers']}"

    base_keys = set(base.tensors)
    in_scope = {f for f in base_keys if not (spec.frozen_regex and spec.frozen_regex.search(f))}

    # Expert byte volume: layout-invariant total from the BASE HEADER, so it
    # stays a valid denominator even when the artifact uses DCP/Megatron FQNs.
    expert_base = [f for f in in_scope if _matches_expert_family(f)]
    base_expert_bytes = sum(
        math.prod(base.tensors[f][0]) * _DTYPE_BYTES[base.tensors[f][1]] for f in expert_base
    )

    derived_adapter: dict[str, tuple[int, ...]] = {}
    fqns: tuple[str, ...] | None = None
    expected_expert_bytes: int | None = None

    if spec.run_kind == "lora":
        if fqn_map is not None:
            notes.append(
                "--fqn-map supplied for a lora run; IGNORED: the adapter declared "
                "set derives from the launch-time live-module census "
                "(--adapter-modules), never from a full-model FQN list"
            )
        # ================================================================
        # #78 -- the declared-FQN oracle is the LAUNCH-TIME LIVE-MODULE
        # CENSUS, in the artifact's own namespace. What this replaces, and
        # why the old oracle could only ever produce the founding incident's
        # shape here: the pre-#78 code derived adapter declared FQNs from
        # the HF base header (HF namespace) x training-config targets x
        # rank. Measured on <compute-node> with this tool's own documented
        # autopsy recipe (--adapter-prefix '') against a real, healthy
        # PROBE save: the save carries Megatron FQNs -- 168 stems, 42 per
        # target (linear_qkv/linear_proj/linear_fc1/linear_fc2), sample
        # stem language_model.decoder.layers.0.mlp.mlp.linear_fc1 -- the
        # header intersects them NOWHERE, declared fell to 0, save_complete
        # went VACUOUS, the drop MUST_FIRE control was unconstructable, and
        # the gate reported BLOCKED over a denominator of zero: all([]) at
        # the oracle layer. No value of --adapter-prefix can repair a
        # WHOLE-NAMESPACE split (a prefix is a leading segment WITHIN one
        # namespace); the repair is a denominator censused from the
        # population the trainer itself attaches.
        # INDEPENDENCE, and its stated limit. The census is produced at
        # launch time from the BASE tree's live module population with the
        # shipped matcher: temporally prior to and physically separate from
        # the judged save, so "what is there matches what is there" cannot
        # occur -- a truncated, renamed, or mis-targeted save still fails
        # against it. The loader additionally refuses a census resolving
        # inside the judged tree (the tautology guard lives in
        # _load_adapter_modules). HONEST RESIDUAL, named not hidden: census
        # and run share the trainer's matcher and module-construction code,
        # so a defect IN that code is invisible to both. That residual is
        # discharged outside this file -- by the census probe's MUST_FIRE /
        # MUST_NOT_FIRE / ANTI_NARROWING controls and the launcher's
        # post-run attach-line cross-check -- and a FULLY independent
        # denominator would be a planner- or conversion-pipeline-produced,
        # versioned expectation frozen before any run exists (the full-FT
        # --fqn-map producer's exact shape; producer and discharge named in
        # the flag's help text). Fqns_basis below repeats this -- every
        # claim its evidence, not one claim stronger.
        # ================================================================
        if adapter_modules is None:
            raise GateUnmeasured(
                "--adapter-modules was not supplied for a lora adjudication, "
                "so the declared adapter set has NO honest source (exit 3 "
                "-- a refused measurement, not a checkpoint verdict). The "
                "pre-#78 oracle derived it from the HF base header x "
                "targets x rank; measured on <compute-node> against a real, "
                "healthy PROBE save (the gate's own documented "
                "'--adapter-prefix ''' autopsy): the save's 168 adapter "
                "stems are Megatron-namespaced, the header intersects them "
                "nowhere, declared fell to 0, save_complete went VACUOUS, "
                "the drop MUST_FIRE control was unconstructable, and the "
                "gate reported BLOCKED over a denominator of zero. Supply "
                "--adapter-modules PATH: the launch-time live-module census "
                "(artifact namespace), written from the INDEPENDENT base "
                "tree before the run starts -- never from the checkpoint "
                "under judgment."
            )
        census_stems = adapter_modules.stems
        census_dims = adapter_modules.dims
        census_basis_text = adapter_modules.basis

        # Adapter-scope expert census, re-based on the census (#78): does
        # any module THIS artifact was told to adapt live inside an expert
        # block? Same doctrine as the fix38 mint it re-anchors -- ARTIFACT
        # scope, never base identity; frozen-regex guard kept;
        # MINT_ZERO_ONLY_IN_PROBE's one named exception preserved -- but the
        # population is the one the save can actually carry, so the
        # classifiers read census stems, not an HF header that names none of
        # them. Each stem is classified in its TENSOR spelling
        # ("<stem>.weight"): the gates' name atoms were written and measured
        # against full tensor FQNs, and this tool does not paraphrase their
        # anchoring conventions. STATED UNCERTAINTY (doctrine 5): how those
        # atoms classify Megatron expert stems has never been measured --
        # tonight's base is dense (0 expert stems in every honest census of
        # it), so both the mint arm and the retention arm are exercised only
        # at 0; the first measured-MoE launch owes this line its
        # verification, which the retention note's denominator makes legible.
        # Denominator (doctrine 2): len(expert_stems) of len(census_stems).
        expert_stems = sorted(
            s
            for s in census_stems
            if _expert_named(f"{s}.weight") or _matches_expert_family(f"{s}.weight")
        )
        if not spec.frozen_regex and not expert_stems:
            # Adapter scope excludes experts entirely (measured over the
            # census): the run DECLARED zero expert tensors.
            # MINT_ZERO_ONLY_IN_PROBE -- the one declared exception,
            # re-anchored on the census so the grep that audits the founding
            # defect finds it argued, not hidden. This 0 is NOT the base
            # model's declaration: the probe's two-source verdict above
            # stands untouched (contradictions included) and is quoted into
            # experts_basis below. It cannot launder an MoE BASE into dense
            # (full runs never reach this branch), and a lora-labeled
            # artifact actually carrying expert tensors is caught by the
            # MODE/lora cross-checks and the structural sweep's phantom
            # binding against these same census stems -- not by this
            # denominator.
            num_experts, num_moe_layers = 0, 0
            experts_basis = (
                "ADAPTER SCOPE: 0 of "
                f"{len(census_stems)} census modules (artifact namespace; "
                f"{census_basis_text}) classify as expert-family by the "
                f"gates' own name classifiers; base model's own declaration "
                f"was {d['num_experts']}; this artifact was declared to "
                f"contain zero expert tensors"
            )
            layers_basis = "adapter scope: no MoE-layer adapter tensors declared"
        elif expert_stems:
            # At least one census module is expert-family (measured over
            # the save's own population): the artifact's expert scope is NOT
            # zero and the mint must not touch it. Recorded with its
            # denominator because this arm is load-bearing -- it is the
            # over-application fence -- and a silent no-op here once read
            # as a vacuous green.
            notes.append(
                f"ADAPTER SCOPE RETAINS EXPERTS: {len(expert_stems)} of "
                f"{len(census_stems)} census modules classify as "
                f"expert-family by the gates' own name classifiers (first: "
                f"{expert_stems[0]!r}; census: {census_basis_text}) -- the "
                f"adapter-scope zero mint does not apply, and the base "
                f"model's expert denominator (num_experts={d['num_experts']}) "
                f"stays attached for the expert gates to adjudicate inside "
                f"this adapter's scope"
            )
        else:
            # Non-expert census under a pinned frozen_regex: the shipped
            # conservative refusal, kept. Minting a zero expert scope under
            # a scope filter this tool cannot interpret would guess at
            # exactly the denominator that decides whether an expert-
            # tensor-carrying adapter green-lights.
            notes.append(
                "adapter-scope expert mint refused: a frozen_regex is "
                "pinned and its semantics are unverified, so the expert "
                "denominator stays the base model's own rather than a zero "
                "this configuration cannot prove"
            )
        # Expected adapter population = census modules x the two naming
        # templates (prefix + stem + literal). The templates remain the
        # adapter_suffixes LITERALS -- never the recognizer regex, which
        # cannot generate -- and adjudicate_checkpoint has already run
        # _verify_adapter_naming_agreement before this function is reached,
        # so the set generated here and the set the structural sweep
        # recognizes cannot have been calibrated apart. Shape convention
        # stays positional: suffix A -> (rank, in_features), suffix B ->
        # (out_features, rank). SHAPES are declared only from independent
        # evidence -- parent (out_features, in_features) dims carried in the
        # census itself AND the rank resolved from the training config;
        # missing either, the FQN is declared with an empty shape and the
        # sweep's shape check abstains BY NAME rather than comparing against
        # a guess. Rank identity at the artifact level remains the
        # launcher's G3 integer check; this tool still refuses to re-derive
        # rank from a percentage window.
        suffix_a, suffix_b = adapter_suffixes
        for stem in census_stems:
            fqn_a = f"{adapter_prefix}{stem}{suffix_a}"
            fqn_b = f"{adapter_prefix}{stem}{suffix_b}"
            if census_dims is not None and spec.lora_rank is not None:
                out_d, in_d = census_dims[stem]  # loader guarantees full coverage
                derived_adapter[fqn_a] = (spec.lora_rank, in_d)
                derived_adapter[fqn_b] = (out_d, spec.lora_rank)
            else:
                derived_adapter[fqn_a] = ()
                derived_adapter[fqn_b] = ()
        if census_dims is None or spec.lora_rank is None:
            notes.append(
                "ADAPTER SHAPES DECLARED WITHOUT SHAPE CHECK "
                f"({len(derived_adapter)} FQNs): abstained by name because "
                + (
                    "the census carries no parent dimensions"
                    if census_dims is None
                    else f"no lora rank resolved ({spec.rank_basis})"
                )
                + " -- the structural sweep binds each adapter to its census "
                "parent but does not check (rank, in)/(out, rank) shape; "
                "wire shape-bearing census entries and the rank key to "
                "close this"
            )
        fqns = tuple(sorted(derived_adapter))
        adapter_overlap = len(artifact_real_fqns & set(fqns))
        fqns_basis = (
            f"DERIVED: {len(fqns)} adapter tensors = {len(census_stems)} "
            f"census modules x 2 naming templates {suffix_a!r}/{suffix_b!r} "
            f"(literals; agreement with the recognizer regex verified at "
            f"startup); census: {census_basis_text} -- produced from the "
            f"INDEPENDENT base tree's live modules at launch time, "
            f"temporally prior to and physically separate from the judged "
            f"save (residual shared-code fate with the trainer's matcher "
            f"disclosed in the derive comment; a planner-frozen expectation "
            f"is the fully independent form); artifact overlap "
            f"{adapter_overlap}/{len(artifact_real_fqns)}"
        )
        if not adapter_overlap and artifact_real_fqns:
            notes.append(
                "--adapter-modules shares ZERO names with the artifact on "
                "disk: the census is stale, was produced for another "
                "namespace/run, or the save itself drifted namespace. The "
                "completeness verdict and the unconstructable drop control "
                "below are that failure, BLOCKING, not a vacuity."
            )
        expected_expert_bytes = None
        bytes_basis = (
            "adapter checkpoints carry no base expert weights; base expert volume is out of scope"
        )
    else:
        declared_full = tuple(sorted(in_scope))
        if fqn_map is not None:
            map_fqns, map_basis = fqn_map
            fqns = map_fqns
            overlap = len(artifact_real_fqns & set(map_fqns))
            fqns_basis = (
                f"{map_basis}; artifact overlap {overlap}/{len(artifact_real_fqns)}"
                f" -- a LOW overlap with the base header is expected (the map "
                f"exists precisely because header names do not describe this "
                f"layout); a ZERO overlap with the artifact names a stale or "
                f"wrong map, and completeness will then FAIL, not pass over it"
            )
            if not overlap and artifact_real_fqns:
                notes.append(
                    "--fqn-map shares zero names with the artifact on disk: the "
                    "map is stale, or this artifact is not the run it was exported "
                    "for. The gate verdict below is that failure, not a vacuity."
                )
        elif declared_full:
            overlap = len(artifact_real_fqns & in_scope)
            ratio = overlap / max(1, len(artifact_real_fqns))
            if ratio >= 0.90 or not artifact_real_fqns:
                fqns = declared_full
                fqns_basis = (
                    f"base model header key set ({base.tensors_source}; "
                    f"{len(base_keys)} keys, frozen-filtered to {len(in_scope)}; "
                    f"{spec.frozen_basis}); artifact overlap {overlap}/"
                    f"{len(artifact_real_fqns)} = {ratio:.3f}"
                )
            else:
                # HF names do not describe this artifact (e.g. Megatron DCP
                # layout). Refuse to auto-map: a guessed mapping is a fabricated
                # denominator. Byte + distinctness gates still run below.
                fqns = None
                fqns_basis = (
                    f"artifact FQNs do not resemble the base HF key set (overlap "
                    f"{overlap}/{len(artifact_real_fqns)} = {ratio:.3f}) and no "
                    f"--fqn-map was supplied -- declared_fqns stays None and "
                    f"checkpoint.save_complete will VACUOUS-block rather than "
                    f"compare against a manufactured list. Provide --fqn-map to close this."
                )
        if num_experts and expert_base:
            expected_expert_bytes = base_expert_bytes
            bytes_basis = (
                f"sum of implied bytes over {len(expert_base)} expert-family tensors "
                f"in the BASE header ({base.tensors_source}) -- layout-invariant, "
                f"valid across HF/DCP naming; assumes frozen scope excludes no experts"
            )
        else:
            bytes_basis = (
                f"base header contains no expert-family tensors in scope "
                f"(declared experts={num_experts}); no byte denominator"
            )

    return Declared(
        fqns,
        fqns_basis,
        num_experts,
        experts_basis,
        num_moe_layers,
        layers_basis,
        expected_expert_bytes,
        bytes_basis,
        derived_adapter,
        notes,
        adapter_modules=(
            tuple(adapter_modules.stems)
            if (spec.run_kind == "lora" and adapter_modules is not None)
            else ()
        ),
    )


# ---------------------------------------------------------------------------
# Measurement (mirrors the probe; never reads tensor payloads)
# ---------------------------------------------------------------------------


def _measure(path: Path) -> CheckpointMetadata:
    try:
        return read_metadata(path)
    except (CheckpointFormatError, OSError) as exc:
        raise GateUnmeasured(f"checkpoint unreadable: {path}: {exc}") from exc


def _context(meta: CheckpointMetadata, decl: Declared, origin: str) -> CheckpointGateContext:
    tensors = tuple(
        TensorMeta(
            fqn=fqn,
            shape=tuple(tm.shape),
            dtype=str(tm.dtype).removeprefix("torch."),
            storage_id=tm.storage_id,
            kind=("extra_state" if (tm.is_extra_state or "_extra_state" in fqn) else "tensor"),
        )
        for fqn, tm in meta.tensors.items()
    )
    return CheckpointGateContext(
        tensors=tensors,
        declared_fqns=decl.fqns,
        num_experts=decl.num_experts,
        num_moe_layers=decl.num_moe_layers,
        expected_expert_bytes=decl.expected_expert_bytes,
        origin=origin,
    )


def _real(meta: CheckpointMetadata) -> list[tuple[str, Any]]:
    return [
        (f, tm) for f, tm in meta.tensors.items() if not (tm.is_extra_state or "_extra_state" in f)
    ]


# ---------------------------------------------------------------------------
# Mode cross-checks -- the anti-false-alarm layer (both directions)
# ---------------------------------------------------------------------------


# #80: a healthy adapter save is not ONLY adapter tensors -- measured on the
# production Megatron adapter save, the 679 real entries are 672
# language_model.* + 6 optimizer.* + 1 rng_state. Those last 7 are legitimate
# non-adapter checkpoint content (optimizer state and RNG), misjudged by the
# pre-#80 lora branch as "unrecognized adapter content" (exit 1 on a good
# save, reproducible the moment #78's sibling wiring reaches this branch).
# NARROW BY CONTRACT: membership is decided on the FQN's ROOT segment only
# (see _is_non_adapter_namespace), so a module merely named with the letters
# "optimizer" is still adjudicated. The predicate has THREE call sites --
# 1460 (_infer_auto_kind's judged pool), 1540 (the lora set-aside below),
# 1574 (the unmarked sweep) -- and the controls pin them site by site; the
# earlier wording here said TWO call sites and charged both to the one
# MUST_PASS, a doctrine-5 miscount. DELETING the frozenset reddens every
# caller (NameError on first use). Deleting 1574 turns the MUST_PASS
# test_calibrated_nondefault_naming_clears_end_to_end red -- its fixture
# carries the measured 7 entries (6 optimizer.* + 1 rng_state) that must
# CLEAR; that path pins run_kind="lora", so deleting 1460 or 1540 leaves
# it green. Deleting 1540 voids the decoy MUST_FIRE's pinned strings -- the
# judged count slides "3 of 27" -> "3 of 34" and the pinned "7
# non-adapter" quote stops matching. Deleting 1460 turns
# test_auto_kind_denominator_excludes_save_state red -- its sized-to-swing
# fixture slides 4/4 = 1.00 back to 4/16 = 0.25 < 0.6 and kind flips to
# "full". WIDENING the match turns the decoy MUST_FIRE
# test_optimizer_shaped_decoy_still_flagged_as_unmarked red: its three
# decoys -- "optimizer" embedded in a module stem
# (layers.3.self_attn.optimizer_gate.weight), a bare root
# (optimizer_gate.x), an exact mid-path "optimizer" segment
# (layers.9.self_attn.optimizer.exp_avg.weight) -- stay judged, pinning
# substring ("optimizer" in fqn), prefix (startswith), and any-segment
# widenings respectively. Add an entry ONLY with a measured save to cite:
# an evidence-free entry here silently shrinks every lora denominator,
# which is doctrine-5 scope creep wearing a fix's clothes.
_NON_ADAPTER_CHECKPOINT_NAMESPACE_ROOTS = frozenset({"optimizer", "rng_state"})


def _is_non_adapter_namespace(fqn: str) -> bool:
    """TRUE only when fqn's ROOT segment is a measured non-adapter namespace.

    partition rather than split: we want the first segment only, and it makes
    the bare single-segment case explicit -- 'rng_state' (no dot) partitions
    to ('rng_state', '', '') and matches by EXACT segment equality, which is
    precisely how the RNG entry appears on disk. Root-anchored means
    'layers.3.self_attn.optimizer_gate.weight' and even 'optimizer_gate.x'
    are NOT excused: the decoy control for that distinction lives in the
    test module; do not "simplify" to a substring test, that is the
    red-maker it watches for."""
    return fqn.partition(".")[0] in _NON_ADAPTER_CHECKPOINT_NAMESPACE_ROOTS


def _infer_auto_kind(
    real_fqns: set[str],
    markers: re.Pattern[str],
) -> tuple[str, str]:
    """Auto-run-kind inference over the JUDGED population only.

    Extracted from adjudicate_checkpoint so the denominator has its own
    firing control (test_auto_kind_denominator_excludes_save_state) -- the
    second, latent bite of #80 lived inline here and could not be probed
    without driving a whole adjudication."""
    judged = sorted(f for f in real_fqns if not _is_non_adapter_namespace(f))
    excluded = len(real_fqns) - len(judged)
    if not judged:
        # Fail closed (doctrine 4) against vacuous truth (doctrine 1): the
        # caller guarantees real_fqns is non-empty, but EVERY entry can be
        # save state -- e.g. a probe pointed at a trainer scratch artifact.
        # The old inline code computed frac = 0/N = 0 -> "full" here, a guess
        # laundered through arithmetic. Zero judged entries is UNMEASURED and
        # the operator must answer with --run-kind instead.
        raise GateUnmeasured(
            f"auto kind inference over {len(real_fqns)} real tensor(s): all sit "
            f"in non-adapter checkpoint namespaces "
            f"{sorted(_NON_ADAPTER_CHECKPOINT_NAMESPACE_ROOTS)} -- zero "
            f"adapter-namespace entries to measure, so there is no fraction to "
            f"classify; pin --run-kind rather than let the tool guess"
        )
    marked = sum(1 for f in judged if markers.search(f))
    # #80 denominator: pre-patch this was marked / len(real_fqns). On the
    # measured production save the error is invisible (the fraction is ~0.99
    # either way) and both launchers pin --run-kind, so this is LATENT --
    # but on a small artifact under `--run-kind auto` or via a library
    # caller, the 6-7 optimizer/rng entries drag the fraction under 0.6, kind
    # resolves "full", and the LoRA save routes into the MODE/full
    # "population looks partial" blocker: the SAME false alarm re-worded.
    # Fixing only the `unmarked` append in cross_check_population would
    # RELOCATE #80 instead of ending it, which is why one constant feeds
    # both sites.
    frac = marked / len(judged)
    kind = "lora" if frac >= 0.6 else "full"
    basis = (
        f"auto: {marked}/{len(judged)} adapter-namespace tensors carry an "
        f"adapter marker ({frac:.2f}), with {excluded} non-adapter "
        f"checkpoint namespace entries excluded from the denominator per "
        f"#80 -> {kind!r}; corroborate with --run-kind when the train "
        f"config has no peft key"
    )
    return kind, basis


def cross_check_population(
    kind: str,
    real_fqns: set[str],
    base: BaseModel,
    decl: Declared,
    adapter_marker: re.Pattern[str],
    modules_to_save: frozenset[str],
) -> list[str]:
    """Blocking reasons arising from artifact-vs-declared-mode disagreement."""
    reasons: list[str] = []
    if kind == "full":
        if decl.fqns and len(real_fqns) < max(8, len(decl.fqns) // 2):
            reasons.append(
                f"MODE/full: {len(real_fqns)} real tensors on disk vs "
                f"{len(decl.fqns)} declared from the base header -- this population "
                f"looks partial (adapter-scale or truncated save) while the run "
                f"declared a full fine-tune"
            )
        extras = sorted(real_fqns - set(decl.fqns or ()))
        if extras:  # reported, not blocking by default: trainer buffers happen
            decl.notes.append(
                f"{len(extras)} artifact tensor(s) outside the declared set (first: "
                f"{extras[0]}) -- reported; use --strict-extras to make this blocking"
            )
    else:  # lora
        if not real_fqns:
            reasons.append("MODE/lora: adapter checkpoint contains zero real tensors")
        contaminated = sorted(
            f for f in real_fqns if f in base.tensors and f not in modules_to_save
        )
        if contaminated:
            reasons.append(
                f"MODE/lora: {len(contaminated)} tensor(s) are verbatim BASE-WEIGHT "
                f"FQNs (first: {contaminated[0]}) -- an adapter checkpoint carrying "
                f"base weights masks a broken adapter save behind plausible bytes"
            )
        # #80: every healthy Megatron adapter save also carries optimizer
        # state and RNG entries (measured: 6 optimizer.* + 1 rng_state among
        # 679 real entries, see _NON_ADAPTER_CHECKPOINT_NAMESPACE_ROOTS).
        # Pre-#80 the `unmarked` sweep below adjudicated them as
        # "unrecognized adapter content" -- exit 1 on a healthy save, masked
        # only because #78 left this branch unreached in production. They are
        # set aside BEFORE judging, by anchored ROOT-SEGMENT match only: a
        # module merely named with the letters "optimizer"
        # (layers.3.self_attn.optimizer_gate.weight) is NOT save state and
        # must stay judged -- test_optimizer_shaped_decoy_still_flagged_as_
        # unmarked is the firing control for that distinction.
        non_adapter = sorted(f for f in real_fqns if _is_non_adapter_namespace(f))
        judged = len(real_fqns) - len(non_adapter)
        if non_adapter:
            # Doctrine 2, on the GREEN path too: an exclusion that silently
            # shrinks a population is indistinguishable from a detector that
            # stopped working, so the shrink is recorded on EVERY lora
            # adjudication, not just red ones -- as a non-blocking
            # declared-basis note. It excuses measured content and must never
            # allege a defect on its own; the red paths quote the same count
            # inline so the denominator is visible in the blocking reason
            # itself.
            decl.notes.append(
                f"set {len(non_adapter)} non-adapter checkpoint namespace "
                f"entry(ies) aside from lora adjudication (roots "
                f"{sorted(_NON_ADAPTER_CHECKPOINT_NAMESPACE_ROOTS)}; first: "
                f"{non_adapter[0]}) -- measured save state excused per #80's "
                f"anchored root-segment match; they remain counted in the "
                f"artifact inventory, only outside the judged population"
            )
        if real_fqns and not judged:
            # Doctrine 1: sweeping zero judged tensors proves nothing
            # (all([]) is True), so an artifact reduced to pure save state is
            # UNMEASURED, never CLEAR. Without this leg the exclusion itself
            # could hollow the detector out from inside -- a one-line way to
            # "fix #80" that no test would distinguish from fixing it.
            reasons.append(
                f"MODE/lora: all {len(real_fqns)} real tensor(s) sit in "
                f"non-adapter checkpoint namespaces (first: {non_adapter[0]}) "
                f"-- zero adapter-namespace entries remain to adjudicate, and "
                f"zero units judged is UNMEASURED, never PASS"
            )
        unmarked = sorted(
            f
            for f in real_fqns
            if not adapter_marker.search(f)
            and f not in modules_to_save
            and not _is_non_adapter_namespace(f)
        )
        if unmarked:
            reasons.append(
                f"MODE/lora: {len(unmarked)} of {judged} adapter-namespace "
                f"tensor(s) carry no adapter marker /{adapter_marker.pattern}/ "
                f"and are not declared modules_to_save (first: {unmarked[0]}; "
                f"{len(non_adapter)} non-adapter checkpoint namespace entr(ies) "
                f"set aside per #80, see the declared-basis note) -- "
                f"unrecognized adapter content is not assumed healthy"
            )
    return reasons


def lora_structural_findings(
    real: list[tuple[str, Any]],
    base: BaseModel,
    decl: Declared,
    spec: TrainSpec,
    *,
    adapter_prefix: str = "",
    adapter_suffix: str = _DEFAULT_ADAPTER_SUFFIX_RE,
    # #78: the artifact-namespace parent pool (census stems). None keeps the
    # legacy base-header binding for direct callers; the gate passes the
    # census whenever the lora path reached it (the gate's own lora flow
    # refuses before this sweep when no census exists, so None here means a
    # direct/library caller, stated not hidden).
    census_parents: frozenset[str] | None = None,
) -> list[str]:
    """Base-referenced binding of the adapter: every adapter's PARENT module must
    be REAL -- checked against the independent census module set (artifact
    namespace) when one is supplied, which since #78 is the only namespace
    this estate's saves use, and against the HF base header's tensor key set
    otherwise (retained for estates whose saves and base header share one
    namespace); rank-shaped pairs must match (rank, in) / (out, rank)
    from independent evidence. Recognition here uses the --adapter-suffix REGEX;
    the declared-set derivation GENERATES from the --adapter-suffix-a/-b
    literals, and _verify_adapter_naming_agreement has already proven at
    startup -- before any verdict -- that every generated literal round-trips
    through this recognizer back to the parent stem it was built on. (The
    first sentence of the pre-repair docstring claimed this sweep was
    "independent of FQN templates"; it never was once the recognizer was
    wired, and the claim is corrected here rather than inherited.) The prefix
    is the same parameter the derivation consumes, and for lora adjudication
    it is now pinned-or-refused: the old empty default was a guess, and
    GateUnmeasured has replaced the guess.

    History, kept because it names the failure shape this sweep still stands
    against: a pre-patch version hardcoded both recognizer and prefix out: a
    pinned custom suffix bound ZERO adapters here while the basis text told
    the operator this check "carries the lora verdict", and a correctly pinned
    prefix made every parent lookup miss (100% phantom) no matter what the
    operator did -- the two knobs contradicted each other. A sweep that binds
    zero adapters under the pinned suffix is named as exactly what it is:
    a vacuous detector, blocking, with a calibration instruction. ("0 of N"
    -- the denominator travels with the verdict, as everywhere in this file.)"""
    reasons: list[str] = []
    stripped = re.compile(adapter_suffix)
    phantom, shape_bad = [], []
    matched = 0
    # fix30(c): partition out the two layouts this gate refuses by name BEFORE
    # the sweep runs, so their tensors cannot launder into either the bound
    # population (they cannot match the calibrated recognizer) or an
    # unexplained vacuity count. Canonical first: its FQNs also satisfy the
    # self-mounted pattern one segment deeper, and each tensor may appear
    # under exactly one layout name -- see the constants' comment above.
    canonical_foreign = sorted(f for f, _tm in real if _CANONICAL_LORA_LAYOUT_RE.search(f))
    canonical_foreign_set = set(canonical_foreign)
    self_mounted_foreign = sorted(
        f
        for f, _tm in real
        if f not in canonical_foreign_set and _SELF_MOUNTED_ADAPTER_LAYOUT_RE.search(f)
    )
    for fqn, tm in real:
        m = stripped.search(fqn)
        if not m:
            continue
        matched += 1
        stem = fqn[: m.start()]
        if adapter_prefix and stem.startswith(adapter_prefix):
            # Parent names live in the BASE namespace; the prefix is export
            # clothing. Strip it only where present -- a prefix that matches
            # nothing is not silently corrected, it just falls through to the
            # phantom check, which is where naming disagreements belong.
            stem = stem[len(adapter_prefix) :]
        if census_parents is not None:
            # #78: for a Megatron-namespace save the HF header names NONE of
            # these parents (measured 336/336 phantom on a healthy artifact
            # -- the false red, not a real defect population), so the binding
            # pool is the independent launch-time census in the artifact's
            # own namespace.
            in_base = stem in census_parents
        else:
            in_base = (stem + ".weight") in base.tensors
        if not in_base:
            phantom.append(fqn)
            continue
        if spec.lora_rank and decl.derived_adapter:
            want = decl.derived_adapter.get(fqn)
            if want and tuple(tm.shape) != want:
                shape_bad.append(f"{fqn} shape {tuple(tm.shape)} != declared {want}")
    if matched == 0 and real:
        reasons.append(
            f"lora: 0 of {len(real)} real tensors could be bound to a base parent "
            f"under the pinned adapter suffix /{adapter_suffix}/ -- the structural "
            f"check that carries the lora verdict while completeness abstains "
            f"examined ZERO adapters. That is a vacuous detector, and it blocks. "
            f"The recognizer regex already AGREES with the generator templates "
            f"(the startup check guarantees it), so this coherently-pinned set "
            f"of knobs is pointing at a naming this artifact does not use. "
            f"Since #78, check the DENOMINATOR first: a 0-of-N bind is exactly "
            f"what a stale or wrong-namespace --adapter-modules census "
            f"produces against a healthy save (measured pre-fix: the "
            f"HF-header oracle read every healthy adapter as phantom, "
            f"336/336). Verify the census's namespace against one saved "
            f"adapter's stem; only then recalibrate --adapter-suffix, "
            f"--adapter-suffix-a, --adapter-suffix-b and --adapter-prefix "
            f"TOGETHER, and pin them in the wrapper"
        )
    if phantom:
        reasons.append(
            f"lora: {len(phantom)} adapter tensor(s) attach to parents absent from "
            f"the base model (first: {phantom[0]}) -- adapters on phantom modules "
            f"are a save/rename defect"
        )
    if shape_bad:
        reasons.append(
            f"lora: {len(shape_bad)} adapter tensor(s) violate the declared "
            f"(rank, in)/(out, rank) shapes (first: {shape_bad[0]})"
        )
    if canonical_foreign:
        reasons.append(
            f"lora: {len(canonical_foreign)} of {len(real)} real tensor(s) "
            f"match the CanonicalLoRA split-adapter layout "
            f"('.adapter.adapter_q/k/v/up/gate.linear_in|out.weight'; first: "
            f"{canonical_foreign[0]}) -- a layout this gate deliberately does "
            f"NOT adjudicate: only the plain-LoRA "
            f"'<module>.adapter.linear_in|linear_out.weight' shape carries a "
            f"calibration here, and recognizing half a layout would be a "
            f"claim broader than the calibrated evidence. This estate's "
            f"launcher selects LoRA (PEFT_SCHEME=lora), not CanonicalLoRA, so "
            f"this content means the run is not what its manifest describes; "
            f"CanonicalLoRA support, if ever wanted, arrives with its own "
            f"calibration and its own controls, never by loosening this "
            f"refusal"
        )
    if self_mounted_foreign:
        reasons.append(
            f"lora: {len(self_mounted_foreign)} of {len(real)} real "
            f"tensor(s) match the LinearAdapter/TELinearAdapter self-mounted "
            f"layout ('<module>.linear_in|linear_out.weight', no '.adapter.' "
            f"segment; first: {self_mounted_foreign[0]}) -- emitted only when "
            f"LoRA.transform takes the plain-nn.Linear / exact-te.Linear / "
            f"FSDP-DTensor-quantized monkey-patch branch, none of which the "
            f"estate's measured parallel-linear targets can reach, and such "
            f"saves also interleave verbatim base weights at '<module>.weight'"
            f" (which the MODE/lora contamination check separately refuses). "
            f"This gate does not calibrate that layout; treat its presence "
            f"as a MODE anomaly and investigate the run -- do not recalibrate "
            f"around it"
        )
    return reasons


# ---------------------------------------------------------------------------
# MUST_FIRE controls on copies of THIS artifact's metadata
# ---------------------------------------------------------------------------


def _attributed_status(res: GateResult, base_res: GateResult | None) -> tuple[str, bool, str]:
    """Resolve (status, confounded, inconclusive_reason) for one injected-defect run.

    Verbatim discipline from the probe's Finding-2 repair (pinned by
    tests/test_hunt_finding_repairs.py), applied to the controls THIS file builds
    itself: detection must be ATTRIBUTABLE to the injection.

      * baseline absent or already blocking -> the experiment is confounded; a
        blocking verdict on the injected copy proves nothing about the injected
        defect -> "inconclusive" (a stated abstention that BLOCKS downstream --
        never laundered into "fired", never silently filed as "inapplicable").
      * verdict FAIL with a clean baseline -> "fired". FAIL is the only verdict
        that asserts the gate examined its units and found the injected defect.
      * verdict blocking-but-not-FAIL -> ERROR means the detector CRASHED on real
        content and VACUOUS means it examined nothing; crediting either as a fire
        is the verifier-exception-counted-as-pass fallacy -> "inconclusive".
      * non-blocking -> "not_fired" (a true negative, and the loop blocks on it).
    """
    if base_res is None:
        return (
            "inconclusive",
            True,
            "no baseline verdict for this detector was supplied, so a block "
            "on the injected copy cannot be attributed to the injection",
        )
    if base_res.blocking:
        return (
            "inconclusive",
            True,
            f"baseline {base_res.verdict.value}: the unmodified artifact "
            f"already blocks this detector, so blocking on the injected copy "
            f"is not evidence the injection was seen",
        )
    if res.verdict is Verdict.FAIL:
        return ("fired", False, "")
    if res.blocking:
        return (
            "inconclusive",
            False,
            f"detector answered the injected defect with {res.verdict.value}, "
            f"a blocking verdict that is not FAIL -- a malfunction (ERROR) or "
            f"a coverage failure, not a detection; crediting it would be the "
            f"verifier-exception-counted-as-pass fallacy",
        )
    return ("not_fired", False, "")


def control_drop(
    ctx: CheckpointGateContext,
    # Unused BY DESIGN, and suppressed inline rather than in pyproject.toml:
    # that file is outside the visible scope of this patch (preflight.py holds
    # the analogous exemption there; whoever next has both files open may
    # unify them). Every _CONTROL_BUILDERS entry must accept the uniform
    # (ctx, baselines) arity -- the consume loop invokes builders through one
    # signature -- and THIS control attributes by name-evidence instead of
    # baseline deltas (its docstring carries the full argument). Narrowing the
    # signature to satisfy the linter would break the builder protocol.
    baselines: dict[str, GateResult] | None = None,  # noqa: ARG001
    n: int = 4,
) -> dict[str, Any]:
    """Universal control, constructable on EVERY artifact with a declared set:
    delete N real tensors from a metadata copy; SaveCompletenessGate MUST fire
    and MUST name a dropped FQN. Unconstructable => caller treats as BLOCKING.

    No baseline consultation, BY DESIGN: crediting requires the rerun gate to
    NAME an injected-dropped FQN in its own missing-list evidence. A verdict
    caused by a pre-existing defect cannot conjure those names (the dropped
    tensors were present on the unmodified artifact), and an ERROR/VACUOUS
    answer carries no 'missing' evidence at all -- so `res.blocking and
    bool(named)` is already FAIL-only in effect, immune to the confounding the
    other controls route through _attributed_status for. The uniform
    (ctx, baselines) builder arity is what _CONTROL_BUILDERS invokes; this
    control simply has nothing to attribute against."""
    if not ctx.declared_fqns:
        return {
            "control": "drop",
            "status": "unconstructable",
            "reason": "no declared_fqns (independent denominator absent) -- there is "
            "nothing for a completeness verdict to be measured against; the "
            "drop control cannot be built, and an unexercised detector "
            "proves nothing (treated as BLOCKING by this tool)",
        }
    declared = set(ctx.declared_fqns)
    eligible = [
        t
        for t in ctx.tensors
        if t.kind == "tensor" and "_extra_state" not in t.fqn and t.fqn in declared
    ]
    if not eligible:
        return {
            "control": "drop",
            "status": "unconstructable",
            "reason": "zero overlap between present tensors and the independent declared "
            "set -- that mismatch is itself already a blocking gate verdict",
        }
    step = max(1, len(eligible) // n)
    targets = [t.fqn for t in eligible[::step][:n]]
    modified = dataclasses.replace(
        ctx, tensors=tuple(t for t in ctx.tensors if t.fqn not in set(targets))
    )
    res = SaveCompletenessGate().run(modified)
    evidence = res.to_dict().get("evidence") or {}
    missing = set(evidence.get("missing", []))
    named = sorted(set(targets) & missing)
    fired = res.blocking and bool(named)
    return {
        "control": "drop",
        "status": "fired" if fired else "not_fired",
        "dropped": targets,
        "verdict": res.verdict.value,
        "detail": res.detail,
        "named_dropped": named,
        "confounded": None,
    }


def control_alias(
    ctx: CheckpointGateContext,
    baselines: dict[str, GateResult] | None = None,
    n: int = 4,
) -> dict[str, Any]:
    """Expert aliasing. Sharded layout: the probe's control VERBATIM -- with the
    baseline WIRED THROUGH. The repaired probe returns "inconclusive" whenever
    attribution to the injection cannot be established, and the pre-patch wiring
    passed baseline=None unconditionally: on sharded experts (the estate's
    primary Megatron layout) the aliasing control therefore could NEVER be
    credited, while the consume loop's if/elif chain let "inconclusive" fall
    through silently and the drop control quietly became load-bearing for every
    layout. Pure stacked layout: alias two stacked tensors in different layers
    onto one storage_id -- the one aliasing signature metadata can see under
    stacking -- with the same attribution rule applied to the verdict. Dense
    artifact: 'inapplicable' (the claim itself is absent), NOT 'passed', and
    covered by the any_fired floor via the drop control.

    Router discipline (defect caught when this suite met the global
    per-expert spelling): this router and the probe both group with the
    shared _split_expert_layouts now, so they cannot disagree on any known
    layout. A probe answer of 'skipped' AFTER the router found a shard group
    of >= 2 is therefore not 'inapplicable' -- it is a classifier divergence:
    one of the two classifiers is lying, the live gate cannot adjudicate
    which at runtime, and the honest grade for "unproven detector" is the
    blocking 'unconstructable' (the probe's own status and reason stay on
    the record). Routed-in-then-skipped must never again file itself under
    recorded-only: that quiet fall-through once shipped CLEAR with the
    load-bearing aliasing detector unexercised on the estate's primary
    layout."""
    candidates = _expert_weight_candidates(ctx.tensors)
    if not candidates:
        return {
            "control": "alias",
            "status": "inapplicable",
            "reason": f"artifact declares and contains no expert tensors (num_experts="
            f"{ctx.num_experts}) -- the aliasing claim does not exist here; "
            f"the drop control is the load-bearing MUST_FIRE on this artifact",
        }
    base_res = (baselines or {}).get(ExpertDistinctnessGate.id)
    shard_groups, stacked, _unknown = _split_expert_layouts(candidates)
    if any(len(m) >= 2 for m in shard_groups.values()):
        # No "unimportable" guard here any more. The alias control used to be
        # imported from an unpackaged tools/ script behind a sentinel, so this
        # branch had to carry an 'unconstructable' refusal for the case where
        # the import had failed -- which is exactly what a `pip install` gave
        # you (#219). The helper now lives in `foundationscale.gates.probe`, so
        # it either imports with the package or the package does not load at
        # all. The refusal became unreachable, and an unreachable DECLARED
        # state is itself a defect (#200), so it is gone rather than left to
        # look like coverage it no longer provides.
        out = _probe_alias_control(ctx, n, baseline=base_res)
        out["control"] = "alias(sharded, probe-verbatim)"
        if out.get("status") == "skipped":
            # A "skipped" HERE cannot mean "inapplicable": this branch runs
            # only because the router used the shared classifier to find a
            # shard group of >= 2 -- aliasable work provably EXISTS on this
            # artifact. A control answering "nothing to alias" after being
            # handed known-present work has drifted from the router (exactly
            # how the global-spelling layout once printed CLEAR unexamined),
            # and which side is lying is unknowable from this seat. What IS
            # knowable: the aliasing detector is unproven on this artifact.
            # Unconstructable is this tool's word for that (an unexercised
            # detector proves nothing -> BLOCKING in the consume loop). The
            # probe's own status and reason are preserved, not erased: an
            # auditable divergence, not a laundered one.
            out = {
                **out,
                "status": "unconstructable",
                "probe_status": "skipped",
                "router_verdict": (
                    "_split_expert_layouts found shard group(s) of >= 2: "
                    f"{sorted(k for k, m in shard_groups.items() if len(m) >= 2)}"
                ),
                "reason": (
                    "classifier divergence: the router sent this artifact to "
                    "the sharded alias control because shard group(s) of >= 2 "
                    "exist, but the control declined: "
                    f"{out.get('reason', '<no reason stated>')}. Whether the "
                    "router or the control is lying is for a human to settle; "
                    "until then the detector is unexercised on this artifact, "
                    "and routed-in-then-skipped must never read as "
                    "recorded-only (the all([]) outcome wearing a router)"
                ),
            }
        return out
    by_stem: dict[str, list[TensorMeta]] = {}
    for t in stacked:
        by_stem.setdefault(_layer_normalized_stem(t.fqn), []).append(t)
    pairs = [m for m in by_stem.values() if len(m) >= 2]
    if not pairs:
        return {
            "control": "alias",
            "status": "inapplicable",
            "reason": "expert tensors exist but neither a shard group nor two same-stem "
            "stacked tensors (different layers) exist to alias",
        }
    victims = pairs[0][:2]
    victim_fqns = {t.fqn for t in victims}
    injected = dataclasses.replace(
        ctx,
        tensors=tuple(
            dataclasses.replace(t, storage_id=_ALIAS_STORAGE_ID) if t.fqn in victim_fqns else t
            for t in ctx.tensors
        ),
    )
    res = ExpertDistinctnessGate().run(injected)
    status, confounded, inconclusive_reason = _attributed_status(res, base_res)
    return {
        "control": "alias(stacked-cross-layer)",
        "status": status,
        "aliased": sorted(victim_fqns),
        "verdict": res.verdict.value,
        "detail": res.detail,
        "confounded": confounded,
        "baseline_verdict": (base_res.verdict.value if base_res is not None else "absent"),
        "inconclusive_reason": inconclusive_reason,
        "leg_observed": "storage span" in res.detail,
    }


def control_underfill(
    ctx: CheckpointGateContext,
    baselines: dict[str, GateResult] | None = None,
) -> dict[str, Any]:
    """Stacked-layout truncation: shrink one stacked expert tensor's leading dim
    to 1/8 of the declared count -- the incident ratio in stacked clothing.
    Credit is FAIL-only and baseline-attributed, exactly as in control_alias:
    a crash on the truncated copy is a malfunction, not a detection."""
    candidates = _expert_weight_candidates(ctx.tensors)
    _sg, stacked, _u = _split_expert_layouts(candidates)
    if not ctx.num_experts:
        return {
            "control": "underfill",
            "status": "inapplicable",
            "reason": "no declared expert count for this artifact's scope",
        }
    if not stacked:
        return {
            "control": "underfill",
            "status": "inapplicable",
            "reason": "no stacked expert tensors (sharded layouts are exercised by "
            "the alias and drop controls instead)",
        }
    if ctx.num_experts < 8:
        return {
            "control": "underfill",
            "status": "unconstructable",
            "reason": f"declared experts={ctx.num_experts} < 8; the 1/8 incident ratio "
            f"cannot be reproduced without degenerating to zero",
        }
    # Victim selection is byte-priced, and a missing price is not a low
    # price: TensorMeta.implied_nbytes is int | None (None when the
    # metadata cannot price a tensor), and the old max() key made that both
    # the type error under repair here and a runtime TypeError on
    # None-vs-int comparison -- which the launcher then measured as exit 3,
    # "a tool bug is not a checkpoint verdict", on artifacts whose expert
    # tensors provably exist. The honest semantics: an underfill the tool
    # cannot MEASURE is one it must not claim to have injected. Unpriced
    # candidates are EXCLUDED from victim selection, never priced as zero;
    # the exclusion is priced into the emitted record below with its
    # denominator (doctrine 2); and if exclusion empties the candidate set
    # the control is unconstructable -- blocking, NAMING 0 of N examined
    # (doctrine 1) -- and never "inapplicable", because stacked expert
    # tensors provably EXIST at this point of the control: the truncation
    # claim is present; it is the yardstick that is missing.
    size_known: list[tuple[TensorMeta, int]] = []
    for cand in stacked:
        cand_nbytes = cand.implied_nbytes
        if cand_nbytes is not None:
            size_known.append((cand, cand_nbytes))
    if not size_known:
        return {
            "control": "underfill",
            "status": "unconstructable",
            "reason": f"0 of {len(stacked)} stacked expert tensors expose a "
            f"derivable byte size (implied_nbytes is None on every "
            f"candidate) -- the underfill cannot be measured, and an "
            f"unexercised detector proves nothing (treated as BLOCKING "
            f"by this tool)",
        }
    victim, victim_nbytes = max(size_known, key=lambda priced: priced[1])
    small = (max(1, ctx.num_experts // 8),) + tuple(victim.shape[1:])
    injected = dataclasses.replace(
        ctx,
        tensors=tuple(
            dataclasses.replace(t, shape=small) if t.fqn == victim.fqn else t for t in ctx.tensors
        ),
    )
    res = ExpertDistinctnessGate().run(injected)
    base_res = (baselines or {}).get(ExpertDistinctnessGate.id)
    status, confounded, inconclusive_reason = _attributed_status(res, base_res)
    return {
        "control": "underfill",
        "status": status,
        "tensor": victim.fqn,
        "shape": [victim.shape, small],
        # Denominator disclosure for the exclusion above (doctrine 2):
        # how many stacked candidates could be byte-priced at all, and
        # the chosen victim's honest price, so a partial exclusion can
        # never read as a sweep over the full candidate population.
        "candidates": f"{len(size_known)} of {len(stacked)} stacked "
        f"expert tensors byte-priced; victim prices at "
        f"{victim_nbytes} bytes",
        "verdict": res.verdict.value,
        "detail": res.detail,
        "confounded": confounded,
        "baseline_verdict": (base_res.verdict.value if base_res is not None else "absent"),
        "inconclusive_reason": inconclusive_reason,
    }


_CONTROL_BUILDERS = {"drop": control_drop, "alias": control_alias, "underfill": control_underfill}


# ---------------------------------------------------------------------------
# Decision object
# ---------------------------------------------------------------------------


class DeclaredBasis(TypedDict):
    """The declared-denominator basis block as it actually appears on the
    wire: five string basis statements plus the notes LIST. This field was
    annotated dict[str, str] while every construction site passed decl.notes
    under "notes" -- a type-level claim one list narrower than the evidence
    (doctrine 5), visible only once report's heterogeneity stopped
    collapsing every value to object. main() and the test suite already
    read exactly these keys; this names what they read."""

    run_kind: str
    fqns: str
    num_experts: str
    num_moe_layers: str
    expected_expert_bytes: str
    notes: list[str]


@dataclass
class GateDecision:
    checkpoint: str
    event: str
    run_kind: str
    verdict: str  # CLEAR | BLOCKED | UNMEASURED
    exit_code: int
    gate_results: list[dict[str, Any]]
    controls: list[dict[str, Any]]
    blocking_reasons: list[str]
    declared_basis: DeclaredBasis
    report: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.exit_code == EXIT_CLEAR

    def raise_if_blocking(self) -> None:
        if self.exit_code != EXIT_CLEAR:
            raise RuntimeError(
                f"live_save_gate: {self.verdict} (exit {self.exit_code}) on "
                f"{self.checkpoint}: " + "; ".join(self.blocking_reasons[:8])
            )


# ---------------------------------------------------------------------------
# #83: interpreter provenance -- measured, always recorded, adjudicated
# whenever the caller states an expectation
# ---------------------------------------------------------------------------

# The launcher knows which python it MEANT to run the gate under and says so
# through this variable ("host": no torch; "container": torch + DCP). Unset
# or empty means no expectation was expressed: the run is still attributed
# (provenance is recorded on every exit path) but uncontested, and the
# record says exactly that.
_EXPECTED_INTERPRETER_ENV = "LIVE_SAVE_GATE_EXPECT_INTERPRETER"


def _interpreter_provenance() -> dict[str, Any]:
    """Measure THIS interpreter for the #83 record; a measurement, never an
    authored constant, and total: it must never itself explode, because both
    the CLEAR/BLOCKED report and the UNMEASURED refusal record have to be
    able to carry it.

    Keys mirror the manifest's training-stack vocabulary exactly
    (emit_run_manifest._training_stack_entries: python_executable,
    python_version, torch_record) -- one spelling for one fact, so the
    manifest-side directive resolves to a field this file really writes.
    torch is LOCATED (find_spec + dist metadata), not imported: importing
    just to record it would let the measurement change the measured. Both
    importlib names are imported locally -- the module's import block is
    outside the listed lines.
    """
    import importlib.metadata
    import importlib.util

    torch_detected = False
    torch_origin: str | None = None
    try:
        torch_spec = importlib.util.find_spec("torch")
    except ValueError:
        # "torch" in sys.modules with __spec__ None: PRESENT but beyond
        # find_spec's reach. Record presence with the origin unread, never
        # an authored absence -- unreadable is not empty.
        torch_detected = True
        torch_origin = "<in sys.modules with __spec__ None>"
    else:
        if torch_spec is not None:
            torch_detected = True
            torch_origin = torch_spec.origin
    torch_version: str | None = None
    if torch_detected:
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_record": {
            "detected": torch_detected,
            "origin": torch_origin,
            "dist_version": torch_version,
            "basis": (
                "importlib.util.find_spec + importlib.metadata.version, "
                "run in the very process being recorded; torch itself never "
                "imported for this record"
            ),
        },
    }


def _resolve_expected_interpreter(cli_expected: str | None) -> str | None:
    """The caller's half of #83: the kwarg wins, else the env var. A value
    outside the {host, container} vocabulary is a REFUSAL, never a silent
    normalization -- an expectation this tool cannot parse is an expectation
    it cannot have met, and adjudicating anyway would manufacture
    attribution."""
    raw = cli_expected if cli_expected is not None else os.environ.get(_EXPECTED_INTERPRETER_ENV)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value not in ("host", "container"):
        source = (
            "the expected_interpreter kwarg"
            if cli_expected is not None
            else f"${_EXPECTED_INTERPRETER_ENV}"
        )
        raise GateUnmeasured(
            f"expected interpreter {raw!r} (from {source}) is neither "
            f"'host' nor 'container' -- refusing to adjudicate under an "
            f"expectation vocabulary this tool cannot read"
        )
    return value


def _refuse_on_interpreter_mismatch(
    provenance: dict[str, Any] | None, expected: str | None
) -> None:
    """#83's second half: record vs reality vs EXPECTED. On any failure the
    tool REFUSES (UNMEASURED) instead of BLOCKING: the artifact may be fine,
    but THIS run can prove nothing while it disagrees with the caller about
    which interpreter it even is -- and BLOCKED would accuse the checkpoint
    of what is the run's own misattribution.

    Three tripwires fire before any expectation is even consulted, loudest
    first: the record ABSENT (unattributable whether or not the caller
    stated an expectation -- unreadable is not empty), and the record
    LYING, caught against a fresh probe of this very interpreter rather
    than anyone's say-so. Callers on the verdict path pass the record
    _interpreter_provenance() has just measured; the re-probe costs one
    find_spec and guards the seam against a foreign or authored record,
    which is exactly what the MUST_FIRE legs plant. The expectation
    vocabulary is re-validated here too, so a direct caller cannot slip an
    unreadable expectation past the channel resolver: an expectation this
    tool cannot read is an expectation it cannot have met.
    """
    if provenance is None:
        raise GateUnmeasured(
            "NO interpreter provenance record was produced -- a verdict "
            "without it is unattributable on either python, expectation or "
            "no expectation (#83) -> UNMEASURED"
        )
    fresh = _interpreter_provenance()
    for key in ("python_executable", "python_version"):
        if provenance.get(key) != fresh[key]:
            raise GateUnmeasured(
                f"provenance record says {key}={provenance.get(key)!r} but "
                f"a fresh probe of THIS interpreter reads {fresh[key]!r} "
                f"-- a record that lies about the interpreter that wrote "
                f"it attributes nothing -> UNMEASURED"
            )
    detected_now = fresh["torch_record"]["detected"]
    if provenance.get("torch_record", {}).get("detected") is not detected_now:
        raise GateUnmeasured(
            f"provenance record says torch_record.detected="
            f"{provenance.get('torch_record', {}).get('detected')!r} but a "
            f"fresh probe of THIS interpreter reads {detected_now!r} -- an "
            f"authored record is not a measurement -> UNMEASURED"
        )
    if expected is None:
        return
    if expected not in ("host", "container"):
        raise GateUnmeasured(
            f"expected interpreter {expected!r} is neither 'host' nor "
            f"'container' -- an expectation this tool cannot read is an "
            f"expectation it cannot have met -> UNMEASURED"
        )
    needs_torch = expected == "container"
    if detected_now is not needs_torch:
        raise GateUnmeasured(
            f"interpreter mismatch: caller expected the {expected} python "
            f"(torch {'present' if needs_torch else 'absent'}), but this "
            f"interpreter records {provenance.get('torch_record')!r} at "
            f"{provenance.get('python_executable')!r} -- host and container "
            f"verdicts are not interchangeable, and this must be settled "
            f"BEFORE any gate runs -> UNMEASURED"
        )


def _interpreter_report_entry(cli_expected: str | None) -> dict[str, Any]:
    """The complete #83 fact for the CLEAR/BLOCKED report: measured
    interpreter + caller expectation + the adjudication's outcome. Reaching
    the return means any supplied expectation was MET -- the raise in
    _refuse_on_interpreter_mismatch is the refusal -- so the entry never
    overstates itself, and says so when uncontested."""
    expected = _resolve_expected_interpreter(cli_expected)
    provenance = _interpreter_provenance()
    _refuse_on_interpreter_mismatch(provenance, expected)
    entry: dict[str, Any] = {
        **provenance,
        "expected_interpreter": expected,
        "expectation_met": None if expected is None else True,
    }
    if expected is None:
        entry["expectation_note"] = (
            "no expectation supplied (kwarg unset and "
            f"${_EXPECTED_INTERPRETER_ENV} unset): provenance recorded but "
            "uncontested; attribution rests on the record alone"
        )
    return entry


def _refusal_interpreter_entry() -> dict[str, Any]:
    """The same fact riding the UNMEASURED refusal record. A refusal must
    never refuse: the expectation is transcribed RAW -- if it is malformed,
    that malformed value is very possibly WHY the refusal exists, and
    laundering it here would erase the cause. Enforcement lives in
    _interpreter_report_entry, on the verdict path."""
    raw = os.environ.get(_EXPECTED_INTERPRETER_ENV)
    return {
        **_interpreter_provenance(),
        "expected_interpreter": raw if raw and raw.strip() else None,
        "expectation_met": None,
    }


# ---------------------------------------------------------------------------
# The callable the launcher invokes
# ---------------------------------------------------------------------------


def adjudicate_checkpoint(
    ckpt_dir: str | os.PathLike[str],
    *,
    event: str = "save",  # "save" | "first_save"
    run_kind: str = "auto",  # "auto" | "full" | "lora"
    base_model_dir: str | os.PathLike[str] | None = None,
    train_config_path: str | os.PathLike[str] | None = None,
    overrides: dict[str, Any] | None = None,  # --set key=value, merged over config
    controls: tuple[str, ...] = ("drop", "alias", "underfill"),
    adapter_marker: str = r"(?:lora_[AB]|adapter)",
    adapter_prefix: str | None = None,
    adapter_suffix_re: str = _DEFAULT_ADAPTER_SUFFIX_RE,
    adapter_suffixes: tuple[str, str] = _DEFAULT_ADAPTER_SUFFIXES,
    modules_to_save: tuple[str, ...] = (),
    fqn_map: str | os.PathLike[str] | None = None,  # --fqn-map: see _load_fqn_map
    strict_extras: bool = False,
    json_out: str | None = None,
    # --adapter-modules: the #78 lora declared denominator; see
    # _load_adapter_modules. Trailing and defaulted so positional callers
    # (library use) are untouched; None keeps the lora refusal path.
    adapter_modules: str | os.PathLike[str] | None = None,
    # #83: "host" | "container" | None; None defers to
    # $LIVE_SAVE_GATE_EXPECT_INTERPRETER (see _resolve_expected_interpreter;
    # the env var is the CLI-facing channel while main()'s argparse wiring
    # is outside this change). Trailing and defaulted under the same
    # discipline as adapter_modules, so positional callers are untouched. A
    # value outside the vocabulary is a refusal, never a silent
    # normalization.
    expected_interpreter: str | None = None,
) -> GateDecision:
    ckpt_path = Path(ckpt_dir)

    # The one consistency check that runs for BOTH run kinds, first: if the
    # operator configured the adapter-naming knobs at all, they configured
    # them against SOME artifact, and contradictory knobs deserve a loud stop
    # even on a full-FT adjudication that would never consult them (silently
    # ignoring contradictory operator input is its own doctrine violation).
    # With untouched defaults this is a sub-microsecond no-op proof.
    _verify_adapter_naming_agreement(adapter_suffix_re, adapter_prefix or "", adapter_suffixes)

    # #83: attribute this run to a measured interpreter BEFORE anything is
    # measured. Built once here and shipped verbatim in the CLEAR/BLOCKED
    # report below; a stated expectation this interpreter cannot meet is a
    # refusal (GateUnmeasured -> UNMEASURED on the refusal path), never a
    # wrong-attribution verdict.
    interpreter_entry = _interpreter_report_entry(expected_interpreter)

    base_dir = (
        Path(base_model_dir)
        if base_model_dir
        else Path(
            os.environ.get(
                "HF_MODEL",  # launcher-set in this estate
                "<CLUSTER_HOME>/pretraining_weights/Vision-Language-Models/"
                "Google/Gemma4/gemma-4-E4B-it",
            )
        )
    )
    base = BaseModel.load(base_dir)  # independent source A
    cfg, cfg_source = _load_train_config(
        Path(train_config_path) if train_config_path else None
    )  # independent source B
    if overrides:
        cfg = {**cfg, **overrides}
        cfg_source += " + --set overrides"
    spec = resolve_train_spec(cfg, cfg_source, run_kind, frozen_arg=None)

    meta = _measure(ckpt_path)  # the artifact
    real = _real(meta)
    real_fqns = {f for f, _ in real}
    if not real:
        raise GateUnmeasured(
            f"checkpoint at {ckpt_path} exposes zero real tensor entries -- read_metadata "
            f"already refuses zero-key sources; an empty artifact reaches this branch only "
            f"via a reader change, measured as UNMEASURED not CLEAR"
        )

    # Resolve 'auto' kind now that populations exist; denominators stay
    # independent -- only the KIND classification may consult the artifact,
    # and the population cross-check below polices the answer.
    kind = spec.run_kind
    markers = re.compile(adapter_marker)
    if kind == "auto":
        # #80's second bite, latent today (both launchers pin --run-kind and
        # production-scale fractions sit far clear of the 0.6 cut) but real
        # for --run-kind auto and library callers on small artifacts: the old
        # denominator counted the same optimizer/rng entries the lora branch
        # now excludes, so a healthy LoRA save could resolve kind='full' and
        # route into the MODE/full "population looks partial" blocker -- the
        # same false alarm re-worded. Fixing only the `unmarked` append would
        # RELOCATE #80, not end it; the seamed helper also gives the
        # denominator its own firing control
        # (test_auto_kind_denominator_excludes_save_state).
        kind, basis = _infer_auto_kind(real_fqns, markers)
        spec = dataclasses.replace(spec, run_kind=kind, kind_basis=basis)

    # The --adapter-prefix question, answered as a DEMAND rather than a
    # default. This file's own evidence cannot establish the estate's adapter
    # export layout: "" is correct for an unprefixed raw-namespace export and
    # wrong-but-loud for an HF-PEFT "base_model.model."-style export, and the
    # old signature silently GUESSED the former. A guess the tool sometimes
    # gets away with is still a guess. For lora adjudication the operator must
    # now PIN the prefix -- where an explicit "" is an ASSERTION of the
    # unprefixed layout, not the absence of an answer. The demand is placed
    # after kind resolution so the auto path cannot smuggle the guess past it
    # (kind inference may consult the artifact; the prefix question may not be
    # answered by it), and before any verdict exists. Full-FT adjudication
    # never consults adapter naming and is unaffected.
    if kind == "lora" and adapter_prefix is None:
        raise GateUnmeasured(
            "--adapter-prefix was not pinned for a lora adjudication (exit 3 "
            "-- a refused measurement, not a checkpoint verdict): whether this "
            "estate's adapter saves carry a constant leading segment (an "
            "extra module-root / wrapper prefix) before the base-module stem "
            "cannot be established from any independent source this tool may "
            "read, and the empty prefix is a guess this tool refuses to own. "
            "The suffixes need no such measurement -- since fix30 the "
            "defaults ARE the estate's measured Megatron-Bridge shape -- so "
            "the prefix is the only knob still awaiting evidence. "
            "#78, measured on <compute-node> with that very autopsy against a "
            "real, healthy PROBE save: pinning '' produced BLOCKED with 0 "
            "declared, because this estate's save stems are Megatron-"
            "namespaced and the old oracle censused the HF base header -- a "
            "WHOLE-NAMESPACE split no leading segment can bridge. The "
            "old phantom-stem recipe implied one; it is RETIRED for this "
            "estate and remains valid only where the save and the declared "
            "source share ONE namespace with an extra wrapper root. The fix "
            "path here is --adapter-modules (the launch-time live-module "
            "census in artifact namespace) with --adapter-prefix '' asserted "
            "alongside it; pin a non-empty prefix only on an estate whose "
            "saves genuinely carry a constant leading segment within one "
            "namespace."
        )
    # Bound once, used below: the demand makes this non-None for lora, and ""
    # for full runs where the value is never consulted. Declaring the pinned
    # intent under one name keeps the two consumption sites (generation,
    # structural binding) unable to drift from each other.
    pinned_adapter_prefix = adapter_prefix if adapter_prefix is not None else ""

    # Independent source D (lora): the launch-time live-module census, the
    # #78 denominator. Loaded BEFORE derivation like the other sources, so a
    # missing/malformed/empty/inside-the-judged-tree census is UNMEASURED,
    # never a guessed denominator. ORDER IS LOAD-BEARING, in the other
    # direction from what one might guess: the prefix demand sits ABOVE this
    # block precisely because production launches today pin neither -- the
    # launcher maps the prefix refusal class to its calibrated rc 0, and
    # that calibration must keep matching byte-for-byte until a coordinated
    # edit wires this flag and sunsets the arm (named there, named here).
    # Consumed only by the lora branch of derive_declared_block; a full run
    # that passes the flag is still parsed strictly, because operator input
    # this tool cannot honour must fail closed, not ride along inert.
    adapter_modules_loaded: _AdapterModuleCensus | None = None
    if adapter_modules is not None:
        adapter_modules_loaded = _load_adapter_modules(Path(adapter_modules), judged_dir=ckpt_path)

    # Independent source C (optional): the planner/operator-exported FQN map.
    # Loaded BEFORE derivation so a missing or empty map is UNMEASURED, never a
    # guessed denominator.
    fqn_map_loaded: tuple[tuple[str, ...], str] | None = None
    if fqn_map is not None:
        fqn_map_loaded = _load_fqn_map(Path(fqn_map))
    decl = derive_declared_block(
        base,
        spec,
        real_fqns,
        pinned_adapter_prefix,
        adapter_suffixes,
        fqn_map=fqn_map_loaded,
        adapter_modules=adapter_modules_loaded,
    )
    ctx = _context(meta, decl, f"{meta.origin} [gates=live; base={base_dir}; cfg={cfg_source}]")

    gates: list[Gate] = [g() for g in _ALWAYS_GATES]
    if event == "first_save":
        gates.append(FirstSaveGate())
    results = [g.run(ctx) for g in gates]

    blocking = [r for r in results if r.blocking]
    reasons = [
        f"{r.gate_id}={r.verdict.value}: {r.detail.split(chr(10))[0][:200]}" for r in blocking
    ]
    reasons += cross_check_population(
        kind, real_fqns, base, decl, markers, frozenset(modules_to_save)
    )
    if kind == "lora":
        reasons += lora_structural_findings(
            real,
            base,
            decl,
            spec,
            adapter_prefix=pinned_adapter_prefix,
            adapter_suffix=adapter_suffix_re,
            census_parents=(frozenset(decl.adapter_modules) if decl.adapter_modules else None),
        )
    extras_blocking = (
        [n for n in decl.notes if "outside the declared set" in n] if strict_extras else []
    )
    reasons += extras_blocking

    # Framework self-check, verbatim discipline from the probe: PASS over zero
    # examined units is the framework's own founding bug surfacing.
    for r in results:
        if r.verdict is Verdict.PASS and r.coverage.checked == 0:
            reasons.append(f"{r.gate_id}: PASS over 0 checked units -- framework invariant breach")

    # MUST_FIRE controls, on copies of THIS artifact.
    control_reports: list[dict[str, Any]] = []
    # Built before the builders run and HANDED to them: attribution of an
    # injected verdict to the injection requires the unmodified artifact's own
    # verdicts, computed above from the same gates the sweep just ran. The old
    # post-hoc `confounded` override is gone -- a flag rewritten after the fact,
    # with status left at "fired", was a self-contradictory record (detection
    # and non-attribution asserted at once).
    baseline_by_gate = {r.gate_id: r for r in results}
    for name in controls:
        build = _CONTROL_BUILDERS.get(name)
        if build is None:
            control_reports.append(
                {
                    "control": name,
                    "status": "unconstructable",
                    "reason": f"unknown control {name!r}",
                }
            )
            continue
        control_reports.append(build(ctx, baseline_by_gate))
    any_fired = False
    for c in control_reports:
        st = c["status"]
        if st == "fired":
            any_fired = True
        elif st == "not_fired":
            reasons.append(
                f"MUST_FIRE control {c['control']} stayed QUIET on real "
                f"content -- the detector cannot be trusted on this artifact"
            )
        elif st == "unconstructable":
            reasons.append(
                f"MUST_FIRE control {c['control']} could not be built on "
                f"this artifact: {c.get('reason')} (an unexercised detector "
                f"proves nothing -> BLOCKING)"
            )
        elif st == "inconclusive":
            # The control RAN but the experiment proves nothing: the unmodified
            # artifact already blocks the detector (confounded), no baseline was
            # available, or the detector answered the injection with ERROR/a
            # coverage verdict -- a malfunction, not a detection. This is a
            # STATED ABSTENTION, and it blocks: filing it under the 'inapplicable'
            # fall-through is how the sharded-layout aliasing control spent its
            # whole life exercised and never credited, and the any_fired floor
            # alone only catches the case where NOTHING else fired.
            why = c.get("inconclusive_reason") or c.get("reason") or "no reason stated"
            reasons.append(
                f"MUST_FIRE control {c['control']} is INCONCLUSIVE: {why}"
                f" -- an exercised-but-unattributable control proves "
                f"nothing -> BLOCKING"
            )
        elif st in ("inapplicable", "skipped"):
            # The control's claim is absent on this artifact (dense => no
            # aliasing claim); "skipped" is the probe's word for the same fact.
            # Recorded, non-blocking, and ONLY tolerable because the any_fired
            # floor below still requires a DIFFERENT control to have fired.
            pass
        else:
            reasons.append(
                f"MUST_FIRE control {c.get('control')} returned an "
                f"unrecognized status {st!r} -- a control vocabulary this "
                f"loop cannot read is an unproven detector -> BLOCKING"
            )
    if not any_fired:
        reasons.append(
            "no MUST_FIRE control fired on this artifact -- the run "
            "proved nothing about the detectors"
        )

    exit_code = EXIT_BLOCKED if reasons else EXIT_CLEAR
    verdict = "CLEAR" if exit_code == EXIT_CLEAR else "BLOCKED"

    # Well-typed locals first, THEN the report: a heterogeneous dict
    # literal infers its value type as object, so anything subscripted out
    # of it arrives untyped at GateDecision's declared field types -- that
    # is precisely where the two `object` argument errors came from.
    # Building the two typed members as named locals and passing THOSE to
    # the decision object lets the code state what it already knows,
    # without a cast. Object identity matches the old flow: report["gates"]
    # / report["declared_basis"] and the constructor arguments were the
    # same list/dict objects before, and they are the same objects here.
    gate_dicts: list[dict[str, Any]] = [r.to_dict() for r in results]
    declared_basis: DeclaredBasis = {
        "run_kind": spec.kind_basis,
        "fqns": decl.fqns_basis,
        "num_experts": decl.experts_basis,
        "num_moe_layers": decl.moe_layers_basis,
        "expected_expert_bytes": decl.bytes_basis,
        "notes": decl.notes,
    }
    report = {
        "checkpoint": str(ckpt_path),
        "event": event,
        "run_kind": kind,
        "inventory": {
            "origin": meta.origin,
            "format": meta.format,
            "entries_total": len(meta.tensors),
            "real_tensors": len(real),
            "base_tensors": len(base.tensors),
            "base_source": base.tensors_source,
        },
        "declared_basis": declared_basis,
        "gates": gate_dicts,
        "controls": control_reports,
        "blocking_reasons": reasons,
        "exit_code": exit_code,
        # #83: WHICH python produced this verdict; measured at the head of
        # adjudicate_checkpoint, so a refused mismatch can never reach this
        # dict wearing a false attribution.
        "interpreter": interpreter_entry,
    }
    if json_out:
        try:
            Path(json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            raise GateUnmeasured(f"could not write --json report {json_out}: {exc}") from exc

    return GateDecision(
        str(ckpt_path),
        event,
        kind,
        verdict,
        exit_code,
        gate_dicts,
        control_reports,
        reasons,
        declared_basis,
        report,
    )


# ---------------------------------------------------------------------------
# CLI: render like the probe, exit like the probe
# ---------------------------------------------------------------------------


def _record_refusal(args: argparse.Namespace, message: str) -> None:
    """Write the refusal record for an UNMEASURED exit, best-effort and loud.

    Until fix44 / #77-B3 the unmeasured paths below returned exit 3 having
    written NOTHING, while the launcher's mapping text told the operator the
    abstention "is recorded on disk at $ART_REPORT" -- a claim about a file
    that was never produced (measured on jobs 1787517960364/1787518637847:
    fs_gate/ held only resolved-train-config.json; report-lora.json did not
    exist). The refusal itself is the record worth keeping: the tool knows
    WHY it refused, and that reason is the evidence the launcher
    demultiplexes the multiplexed exit-3 class with. Denominators are stated
    honestly: 0 of 3 gates and 0 of 3 controls ran, on every unmeasured path
    by definition, because the refusal precedes any verdict. A refusal record
    that cannot itself be written is reported on stderr and left for the
    launcher to indict: it verifies the record's presence -- and its
    classification -- before ever claiming it, per the same fix.
    """
    if not args.json_out:
        print(
            "live_gate refusal record: NO --json path was given, so this "
            "refusal exists nowhere on disk -- the caller waived the record",
            file=sys.stderr,
        )
        return
    record = {
        "checkpoint": str(args.ckpt_dir),
        "event": args.event,
        "run_kind": args.run_kind,
        "verdict": "UNMEASURED",
        "exit_code": EXIT_UNMEASURED,
        "refusal": message,
        "refusal_class": _refusal_class(message),
        "gates_exercised": "0 of 3",
        "controls_exercised": "0 of 3",
        "record_basis": (
            "written at the point of refusal, BEFORE any gate or control "
            "ran; the refusal IS the adjudication of record for an "
            "unmeasured exit"
        ),
        # #83: a refusal is as env-dependent as a CLEAR, so the same
        # measured interpreter record rides here too (expectation raw; see
        # _refusal_interpreter_entry above).
        "interpreter": _refusal_interpreter_entry(),
    }
    try:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(
            "live_gate could NOT record its own refusal report at "
            f"{args.json_out}: {exc} -- refusing to fail silently; the "
            "launcher maps a claimed-but-absent record to its rc-92 class",
            file=sys.stderr,
        )
