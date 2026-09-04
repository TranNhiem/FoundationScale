"""Tests for tools/live_save_gate.py -- the gate that adjudicates a LIVE save.

Two doctrines are load-bearing throughout and are stated here once instead of
in every docstring:

  * Every "this is BLOCKED" test has a MUST_PASS twin on the same code path
    (doctrine 5, symmetric): a gate that blocks everything is exactly as
    broken as one that blocks nothing, and it is the one that gets
    uninstalled. The twins: T-full-healthy pairs T-full-truncated /
    T-full-as-lora; T-lora-healthy pairs T-lora-contaminated /
    T-lora-as-full / T-phantom / T-shape-bad / T-prefix; T-quiet-detector
    pairs every CLEAR test.
  * Denominators are asserted NUMERICALLY (real_tensors, base_tensors,
    declared counts), not just via verdict strings (doctrine 2).

Labels: [FAILS-BEFORE] = red with the corresponding ## Patch edit reverted.
[PASSES-BEFORE] = green on the current tree; the docstring names the
one-line weakening of the tool that would turn it red.

Calibration policy (no skips exist in this file): two facts could not be
verified from the three source files handed over with the task -- the exact
on-disk shape read_metadata() accepts, and the probe's config-key schema for
expert counts. Fixtures that depend on them raise a loud AssertionError
naming the calibration duty instead of skipping, per FS_FORBID_SKIPS.
"""

from __future__ import annotations

import json
import struct
import sys
from math import prod
from pathlib import Path

import pytest

import foundationscale
from foundationscale.gates.core import Coverage, GateResult, Verdict


def _tools_on_path() -> None:
    pkg_file = Path(foundationscale.__file__).resolve()
    for candidate in pkg_file.parents:
        tools_dir = candidate / "tools"
        if (tools_dir / "real_checkpoint_probe.py").is_file():
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
            return
    raise AssertionError(
        "tools/real_checkpoint_probe.py not found beside the installed "
        "foundationscale package -- run these tests from a repo checkout; "
        "silently skipping them would be a vacuous green"
    )


_tools_on_path()

import live_save_gate as lsg_cli  # noqa: E402 (path fixup first)

from foundationscale.gates import adjudication as lsg  # noqa: E402 (path fixup first)

_ST_NBYTES = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8}


def _write_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, ...], str]]) -> None:
    """A minimal, valid safetensors file: header + zero payload, offsets honest."""
    header: dict[str, dict] = {}
    offset = 0
    for fqn in sorted(tensors):
        shape, dtype = tensors[fqn]
        nbytes = prod(shape) * _ST_NBYTES[dtype]
        header[fqn] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


def _materialize_artifact(tmp_path: Path, tensors: dict, *, name: str = "ckpt") -> Path:
    """Write the artifact under real candidate layouts and return the first one
    read_metadata actually accepts. Refusal on every shape is a LOUD
    calibration failure -- never a skip."""
    attempts: list[str] = []

    d1 = tmp_path / f"{name}-st"
    d1.mkdir()
    _write_safetensors(d1 / "model.safetensors", tensors)
    try:
        lsg._measure(d1)
        return d1
    except Exception as exc:  # noqa: BLE001 -- candidate probing records, not swallows
        attempts.append(f"single-safetensors-dir: {exc!r}")

    items = sorted(tensors.items())
    mid = max(1, (len(items) + 1) // 2)
    shards = [dict(items[:mid])] + ([dict(items[mid:])] if items[mid:] else [])
    d2 = tmp_path / f"{name}-st-idx"
    d2.mkdir()
    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        _write_safetensors(d2 / fname, shard)
        for fqn in shard:
            weight_map[fqn] = fname
    (d2 / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    try:
        lsg._measure(d2)
        return d2
    except Exception as exc:  # noqa: BLE001
        attempts.append(f"index-sharded-safetensors-dir: {exc!r}")

    try:  # optional third shape: a REAL torch DCP save, if torch is importable
        import torch  # noqa: PLC0415
        import torch.distributed.checkpoint as dcp  # noqa: PLC0415
        from torch.distributed.checkpoint import FileSystemWriter  # noqa: PLC0415

        d3 = tmp_path / f"{name}-dcp"
        d3.mkdir()
        sd = {
            fqn: torch.zeros(
                *shape,
                dtype=getattr(
                    torch,
                    {"F32": "float32", "BF16": "bfloat16", "F16": "float16", "I64": "int64"}[dtype],
                ),
            )
            for fqn, (shape, dtype) in tensors.items()
        }
        dcp.save(sd, storage_writer=FileSystemWriter(str(d3)))
        try:
            lsg._measure(d3)
            return d3
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"torch-dcp-dir: {exc!r}")
    except Exception as exc:  # noqa: BLE001
        attempts.append(f"torch unavailable: {exc!r}")

    raise AssertionError(
        "CALIBRATION FAILURE (not a skip): read_metadata accepted NONE of the "
        "candidate on-disk shapes -- " + "; ".join(attempts) + ". Pin the "
        "estate's real artifact layout into _materialize_artifact ONCE against "
        "the live reader. A suite that cannot materialize an artifact has "
        "verified nothing, and FS_FORBID_SKIPS forbids saying so quietly."
    )


def _make_base(tmp_path: Path, tensors: dict, config: dict, *, name: str = "base") -> Path:
    base = tmp_path / name
    base.mkdir()
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _write_safetensors(base / "model.safetensors", tensors)
    loaded = lsg.BaseModel.load(base)  # the base writer is NOT candidates-tried:
    assert len(loaded.tensors) == len(tensors), (
        f"base writer drifted from BaseModel.load: wrote {len(tensors)}, "
        f"parsed {len(loaded.tensors)} -- fix the writer, never the assertion"
    )
    return base


def _write_cfg(tmp_path: Path, cfg: dict, *, name: str = "resolved-train-config.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _census(stems, *, dims=None, source="fix45 test fixture"):
    """Build an _AdapterModuleCensus in memory, for direct
    derive_declared_block(...) calls. `stems` is a sequence of module-FQN
    strings in the ARTIFACT (Megatron) namespace. `dims`, when given, maps
    stem -> (out_features, in_features) and must cover EVERY stem or none
    (the loader refuses partial dims). Returns _AdapterModuleCensus."""
    stems = tuple(stems)
    if not stems:
        raise AssertionError(
            "census fixture carries ZERO stems -- the loader refuses a zero "
            "denominator (doctrine 1); an honest fixture never mints one"
        )
    if len(set(stems)) != len(stems):
        raise AssertionError(
            "census fixture carries duplicate stems -- the loader refuses "
            "duplicates as a broken census; a fixture the loader would "
            "refuse measures nothing"
        )
    if dims is not None and set(dims) != set(stems):
        raise AssertionError(
            f"census dims cover {len(dims)} of {len(stems)} stems -- the "
            f"loader refuses partial dims; a fixture the loader would refuse "
            f"measures nothing (missing: {sorted(set(stems) - set(dims))!r})"
        )
    basis = (
        f"in-memory test census ({len(stems)} artifact-namespace module stems"
        + (
            f", parent dims for all {len(stems)}"
            if dims
            else ", no parent dims -- shape check abstains by name"
        )
        + f"; producer: {source})"
    )
    return lsg._AdapterModuleCensus(
        stems=tuple(sorted(stems)),
        dims=dict(dims) if dims is not None else None,
        basis=basis,
    )


def _census_file(dirpath, stems, *, dims=None, source="fix45 test fixture"):
    """Write an --adapter-modules JSON census under `dirpath` and return its
    Path, for CLI-level and adjudicate_checkpoint-level tests. `dirpath` MUST
    be outside the judged checkpoint tree -- the loader refuses a census that
    resolves inside it, and that guard is load-bearing (a denominator read
    from the tree under judgment is the all([]) tautology). Returns Path."""
    built = _census(stems, dims=dims, source=source)  # one refusal posture
    if built.dims is None:
        doc: dict = {"adapter_modules": list(built.stems), "source": source}
    else:
        doc = {
            "adapter_modules": [
                {"fqn": s, "out_features": built.dims[s][0], "in_features": built.dims[s][1]}
                for s in built.stems
            ],
            "source": source,
        }
    path = Path(dirpath) / "adapter-modules.json"
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return path


def _lora_census_stems() -> tuple[str, ...]:
    """The honest census for the healthy-lora fixture family (#78): the 12
    module stems `_megatron_named_lora_tensors` actually writes adapters
    under (6 layers x {q_proj, v_proj}), in the ARTIFACT namespace. DERIVED
    from that fixture's keys by stripping the shipped Megatron-Bridge suffix
    pair, so the census fixture cannot drift from the artifact fixture -- a
    hand-copied stem list at every call site would recreate the fixture-and-
    defect-share-one-shape failure this file's own calibration record (above
    `_lora_tensors`) exists to forbid. Module-level names
    (`_megatron_named_lora_tensors` is defined beside the adapter-naming
    tests, below) resolve at call time -- the idiom `_lora_tensors` itself
    already documents above."""
    return tuple(
        sorted(
            {
                key[: -len(suffix)]
                for key in _megatron_named_lora_tensors()
                for suffix in lsg._MEGATRON_BRIDGE_ADAPTER_SUFFIXES
                if key.endswith(suffix)
            }
        )
    )


def _dense_full_tensors() -> dict:
    return {
        f"layers.{i}.self_attn.{w}.weight": ((8, 8), "F32")
        for i in range(6)
        for w in ("q_proj", "v_proj")
    }


def _lora_tensors(prefix: str = "") -> dict:
    # fix34 calibration record (fixture, not assertion -- the rule this file
    # laid down for itself above _dense_full_tensors after the MoE
    # denominator drift: "Calibrate the fixture, never the assertion.").
    # Pre-fix30-T2 this helper emitted HF PEFT-shaped FQNs (.lora_A/.lora_B),
    # and the adapter suite's green rested on the fixture matching THAT
    # retired default recognizer -- the fixture and the defect were the same
    # shape, so the green was the defect certifying itself. T2 made the
    # measured Megatron-Bridge layout the shipped default; this population
    # follows it. Emission is delegated to _megatron_named_lora_tensors
    # (defined beside the adapter-naming tests below; module-level names
    # resolve at call time) so the file keeps exactly ONE adapter-population
    # source -- a second hand-copied emission loop would drift from the first
    # and recreating that drift is how this failure started. A consumer that
    # genuinely exercises the retired HF convention must now say so in its
    # own text and pass lsg._HF_PEFT_ADAPTER_SUFFIX_RE /
    # _HF_PEFT_ADAPTER_SUFFIXES explicitly -- the option-(b) precedent the
    # explicit-suffix-tuple call sites at the bottom of this file already
    # set. The 24-tensor count and the (rank, in)/(out, rank) shapes are
    # byte-identical to the old emission -- only the suffix segments changed
    # -- so every numeric assertion downstream (24 real / 12 base / derived
    # 24) holds unchanged.
    return _megatron_named_lora_tensors(prefix)


def _moe_full_tensors() -> dict:
    t = {f"layers.{ly}.self_attn.q_proj.weight": ((8, 8), "F32") for ly in range(2)}
    for ly in range(2):
        for e in range(8):
            for proj in ("linear_fc1", "linear_fc2"):
                t[f"layers.{ly}.experts.{e}.{proj}.weight"] = ((4, 4), "F32")
    return t


DENSE_CFG = {
    "model_type": "calibration-dense",
    "text_config": {
        "num_hidden_layers": 6,
        "hidden_size": 8,
        "enable_moe_block": False,
        "num_experts": None,
    },
}

MOE_CFG = {
    "model_type": "calibration-moe",
    "num_experts": 8,
    "num_moe_layers": 2,
    "text_config": {"num_experts": 8, "num_moe_layers": 2, "num_hidden_layers": 2},
}

LORA_TRAIN = {"peft_scheme": "lora", "lora_rank": 4, "lora_targets": ["q_proj", "v_proj"]}


def _probe_declared_or_calibrate(cfg: dict, want_experts, want_layers):
    """Guard for the one thing testable only against the real probe: its
    config-key schema. Loud, pinnable, never skipped."""
    assert lsg._probe_derive_declared is not None, (
        "probe import failed -- the sys.path insertion in _load_gate regressed"
    )
    d = lsg._probe_derive_declared(cfg)
    if d.get("num_experts") != want_experts or d.get("num_moe_layers") != want_layers:
        raise AssertionError(
            f"CALIBRATION (not a skip): probe derive_declared read num_experts="
            f"{d.get('num_experts')}, num_moe_layers={d.get('num_moe_layers')} from "
            f"the fixture config; expected {want_experts}/{want_layers}. Align the "
            f"fixture's key spellings with the probe's schema once and pin them."
        )
    return d


def _dense_base_with_ckpt(tmp_path: Path, tensors: dict | None = None):
    base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
    ckpt = _materialize_artifact(tmp_path, tensors or _dense_full_tensors())
    return base, ckpt


def _healthy_lora(tmp_path: Path, prefix: str = ""):
    base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
    ckpt = _materialize_artifact(tmp_path, _lora_tensors(prefix))
    cfg = _write_cfg(tmp_path, LORA_TRAIN)
    return base, ckpt, cfg


def _control_by_prefix(decision, prefix: str) -> dict:
    for c in decision.controls:
        if c["control"].startswith(prefix):
            return c
    raise AssertionError(
        f"no control named {prefix!r}* in {decision.controls!r} -- the report "
        f"dropped a control; that absence must be loud, not defaulted"
    )


def _gr(verdict: Verdict, gate_id: str, detail: str = "") -> GateResult:
    return GateResult(
        gate_id=gate_id, verdict=verdict, coverage=Coverage(2, "units"), detail=detail
    )


def _renamed_full_tensors():
    return {f"megatron.{k}": v for k, v in _dense_full_tensors().items()}


def _stacked_moe_full_tensors() -> dict:
    """HF-stacked MoE artifact: 3 dense attention layers + 2 stacked MoE layers.

    10 real tensors (over the MODE/full partial-population threshold of 8),
    stacked expert count 4 == num_moe_layers 2 x family width 2, so neither the
    byte gate's coverage nor the cross-check fires for extraneous reasons: the
    ONLY thing left to block is the composite's not-established leg."""
    t = {
        f"model.language_model.layers.{ly}.attn.{w}.weight": ((8, 8), "F32")
        for ly in range(3)
        for w in ("q_proj", "v_proj")
    }
    for ly in range(2):
        for proj, inner in (("gate_up_proj", (16, 32)), ("down_proj", (32, 16))):
            t[f"model.language_model.layers.{ly}.experts.{proj}"] = ((8, *inner), "BF16")
    return t


def _megatron_named_lora_tensors(prefix: str = "") -> dict:
    """The estate's own adapter naming -- measured on the estate (fix30) and
    the shape the tool's SHIPPED DEFAULT recognizer matches since
    fix30-T2: the low-rank matrices named adapter.linear_in /
    adapter.linear_out immediately under the adapted module's FQN.
    Docstring corrected by fix34: an earlier revision of this docstring
    called the naming "a plausible NON-PEFT export ... the calibration the
    tool's own comments kept promising", written when the shipped defaults
    were HF PEFT and this layout existed here only to pin the new generator
    templates. Post-T2 that framing is inverted -- this naming is what the
    estate's Megatron-Bridge launcher actually saves and what the defaults
    recognize; HF PEFT is the retained, explicitly-named preset
    (_HF_PEFT_ADAPTER_SUFFIXES). Describing the shipped default as
    hypothetical is a doctrine-5 over-claim (a claim broader than its
    evidence), so the record is corrected rather than inherited. Shapes
    follow the positional convention (in-template -> (rank, in),
    out-template -> (out, rank)) against the (8, 8) dense parents at rank 4.
    The optional prefix prepends a constant wrapper segment ahead of the
    base-module stem -- the export shape the --adapter-prefix demand exists
    to make the operator assert rather than the tool guess."""
    out = {}
    for i in range(6):
        for w in ("q_proj", "v_proj"):
            stem = f"layers.{i}.self_attn.{w}"
            out[f"{prefix}{stem}.adapter.linear_in.weight"] = ((4, 8), "F32")
            out[f"{prefix}{stem}.adapter.linear_out.weight"] = ((8, 4), "F32")
    # #80: a real save is not ONLY adapter tensors. Measured on the
    # production Megatron adapter save: 672 language_model.* + 6 optimizer.*
    # + 1 rng_state = 679 real entries. This fixture historically carried
    # ZERO non-adapter namespaces -- the "fixture and defect share one shape"
    # failure this file's calibration record warns about, which is exactly
    # why #80 was invisible here while reproducible on every real save. The
    # 7 entries below restore the measured shape. They deliberately do NOT
    # take `prefix` -- save-state roots live at checkpoint scope, above the
    # adapter export's module wrapper -- and the optimizer keys are rooted at
    # a real `optimizer.state.` namespace rather than adapter-suffixed, so
    # only an anchored namespace ROOT match can excuse them; a marker, a
    # suffix, or luck cannot. Adding them turns
    # test_calibrated_nondefault_naming_clears_end_to_end RED on the unfixed
    # tree (measured: EXIT 1) and keeps it red under the mutation that
    # deletes the exclusion; the decoy-based MUST_FIRE covers the opposite
    # mutation (the match widened to a substring). Shapes are placeholders --
    # no shape gate reads optimizer or RNG content.
    for i in range(6):
        out[f"optimizer.state.exp_avg.layers.{i}.mlp.linear_fc1.weight"] = ((8, 8), "F32")
    out["rng_state"] = ((4,), "F32")
    return out


# Re-export surface consumed by the seven test_live_save_gate_*.py modules that
# were split out of the original 3,486-line file. Declared explicitly so a name
# used only by a sibling module does not read to the linter as dead.
__all__ = [
    "Coverage",
    "DENSE_CFG",
    "GateResult",
    "LORA_TRAIN",
    "MOE_CFG",
    "Verdict",
    "_census",
    "_census_file",
    "_control_by_prefix",
    "_dense_base_with_ckpt",
    "_dense_full_tensors",
    "_gr",
    "_healthy_lora",
    "_lora_census_stems",
    "_lora_tensors",
    "_make_base",
    "_materialize_artifact",
    "_megatron_named_lora_tensors",
    "_moe_full_tensors",
    "_probe_declared_or_calibrate",
    "_renamed_full_tensors",
    "_stacked_moe_full_tensors",
    "_write_cfg",
    "_write_safetensors",
    "json",
    "lsg",
    "lsg_cli",
    "pytest",
    "struct",
]
