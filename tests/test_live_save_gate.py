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


# The subject under test is the LIBRARY module, imported normally.
#
# It used to be `tools/live_save_gate.py`, loaded by path through
# spec_from_file_location with a sys.modules pre-registration so @dataclass
# could resolve string annotations. That apparatus existed only because the
# decision API lived in a script; T2_lib_script_boundary#0 moved the API into
# foundationscale.gates.adjudication and left the script as an argparse
# wrapper, so the apparatus is gone.
#
# Binding `lsg` to the DEFINING module is load-bearing, not cosmetic. This file
# monkeypatches module-level names (_probe_derive_declared, _split_expert_layouts,
# _ALWAYS_GATES, ...) to drive the gates into their refusal arms. A re-export
# binds a NAME, not the defining module's globals: patching the wrapper would
# leave every one of those controls inert while still reporting green.
#
# tools/ still goes onto sys.path first, because the adjudication module imports
# real_checkpoint_probe as a sibling top-level module. Without it the probe
# import fails silently-when-lucky and every derive path exits 3 with "refusing
# to re-derive declared counts by paraphrase".
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

# The CLI is the OTHER half of the split: argparse + exit-code mapping stayed
# in tools/live_save_gate.py, so the exit-code tests below must call THAT
# module's main(). The direction of the re-export hazard reverses here --
# main() reads the wrapper's own global `adjudicate_checkpoint`, so the
# tool-bug test patches lsg_cli, not lsg.
import live_save_gate as lsg_cli  # noqa: E402 (path fixup first)

from foundationscale.gates import adjudication as lsg  # noqa: E402 (path fixup first)

# ---------------------------------------------------------------------------
# Real-artifact writers (no torch, no safetensors needed for the HF layouts)
# ---------------------------------------------------------------------------

_ST_NBYTES = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8}


def _write_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, ...], str]]) -> None:
    """A minimal, valid safetensors file: header + zero payload, offsets honest."""
    header: dict[str, dict] = {}
    offset = 0
    for fqn in sorted(tensors):
        shape, dtype = tensors[fqn]
        nbytes = prod(shape) * _ST_NBYTES[dtype]
        header[fqn] = {"dtype": dtype, "shape": list(shape),
                       "data_offsets": [offset, offset + nbytes]}
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
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
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
        sd = {fqn: torch.zeros(*shape, dtype=getattr(torch, {"F32": "float32",
              "BF16": "bfloat16", "F16": "float16", "I64": "int64"}[dtype]))
              for fqn, (shape, dtype) in tensors.items()}
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


# ---------------------------------------------------------------------------
# fix45 / #78 census fixtures -- THE SHARED HELPERS, contract-pinned across
# the two shards editing this file in parallel: defined ONCE here by shard C1,
# above the first test class; shard C2 uses them and must not redefine or
# rename them. The lora branch of derive_declared_block now DEMANDS an
# explicit census (the launch-time live-module census in the artifact /
# Megatron namespace, #78) and refuses with GateUnmeasured /
# adapter_census_unavailable without one; every repaired lora test feeds an
# honest census through these two helpers. Both mirror the refusal posture of
# _load_adapter_modules (zero stems, duplicates, partial dims all refuse) --
# a fixture the loader would refuse measures nothing.
# ---------------------------------------------------------------------------


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
        + (f", parent dims for all {len(stems)}" if dims
           else ", no parent dims -- shape check abstains by name")
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
                {"fqn": s, "out_features": built.dims[s][0],
                 "in_features": built.dims[s][1]}
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
    return tuple(sorted({
        key[: -len(suffix)]
        for key in _megatron_named_lora_tensors()
        for suffix in lsg._MEGATRON_BRIDGE_ADAPTER_SUFFIXES
        if key.endswith(suffix)
    }))


# Fixture populations. Denominators are arithmetic, stated in one place:
#   dense full  : 6 layers x (q,v)          = 12 tensors
#   lora adapter: 12 parents x (A,B)        = 24 tensors, rank 4
#   MoE full    : 2 q_proj + 32 experts     = 34 tensors, 8 experts x 2 layers
#                 x 2 projections. The projection factor is NOT decoration: the
#                 gates price a per-expert artifact against a FAMILY, and the
#                 Megatron family is {linear_fc1.weight, linear_fc2.weight}
#                 (checkpoint_gates._PER_EXPERT_PROJECTION_FAMILIES), so the
#                 declared denominator is 8*2*2 = 32. An earlier revision wrote
#                 fc1 only; the gates correctly returned UNDERCOVERED at 16 of 32
#                 and the fixture, not the verdict, was the thing that was wrong.
#                 Calibrate the fixture, never the assertion.
def _dense_full_tensors() -> dict:
    return {f"layers.{i}.self_attn.{w}.weight": ((8, 8), "F32")
            for i in range(6) for w in ("q_proj", "v_proj")}


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


# Dense, AFFIRMATIVELY. The landed two-source contract never mints a 0 from
# silence, so a fixture named "calibration-dense" must SAY dense the way the
# measured estate base does (gemma-4-E4B-it config.json, fix25-s6):
# text_config.enable_moe_block is False and text_config.num_experts is
# present-but-null (key exists, value null -- the probe records this as
# absent-with-a-note; it is neither an MoE statement nor a contradiction).
# The second source is the fixture base header itself: 12 attention tensors,
# zero expert-family names, so every consumer below carries a named 0-of-12
# denominator instead of an inference. This is a fixture CALIBRATION, not a
# weakening: no assertion in any consumer was relaxed -- the tests' own
# docstrings already assumed a DECLARED zero ("the dense run is a legitimate
# declared-zero-expert scope and MUST be CLEAR-able", test_full_healthy),
# and under the landed contract only an affirmative statement makes that
# assumption true. It also makes the suite STRICTER: the present-null path
# and the corroborated-mint path are now exercised end to end.
DENSE_CFG = {"model_type": "calibration-dense",
             "text_config": {"num_hidden_layers": 6, "hidden_size": 8,
                             "enable_moe_block": False, "num_experts": None}}
MOE_CFG = {"model_type": "calibration-moe", "num_experts": 8, "num_moe_layers": 2,
           "text_config": {"num_experts": 8, "num_moe_layers": 2, "num_hidden_layers": 2}}
LORA_TRAIN = {"peft_scheme": "lora", "lora_rank": 4,
              "lora_targets": ["q_proj", "v_proj"]}


def _probe_declared_or_calibrate(cfg: dict, want_experts, want_layers):
    """Guard for the one thing testable only against the real probe: its
    config-key schema. Loud, pinnable, never skipped."""
    assert lsg._probe_derive_declared is not None, (
        "probe import failed -- the sys.path insertion in _load_gate regressed")
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
        f"dropped a control; that absence must be loud, not defaulted")


def _gr(verdict: Verdict, gate_id: str, detail: str = "") -> GateResult:
    return GateResult(gate_id=gate_id, verdict=verdict,
                      coverage=Coverage(2, "units"), detail=detail)


# ---------------------------------------------------------------------------
# Healthy full fine-tune: the MUST_PASS backbone every blocking test twins with
# ---------------------------------------------------------------------------


class TestHealthyFull:
    def test_full_healthy_is_clear_and_carries_denominators(self, tmp_path):
        """[PASSES-BEFORE] Red if: the extras-note OR the any_fired floor were
        deleted (delete `if not any_fired:` block) -- then an artifact whose
        controls all misfire would still pass here silently. Also red if the
        dense-scope gates regress to VACUOUS-blocking a declared-zero-expert
        artifact; if this test is red on the CURRENT tree for that reason, the
        defect is in checkpoint_gates.py, not this suite: the dense run is a
        legitimate declared-zero-expert scope and MUST be CLEAR-able."""
        base, ckpt = _dense_base_with_ckpt(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 0, (
            f"healthy full must be CLEAR, got {d.exit_code}: {d.blocking_reasons}"
        )
        assert d.ok and d.verdict == "CLEAR"
        assert d.blocking_reasons == []
        inv = d.report["inventory"]
        assert inv["real_tensors"] == 12 and inv["base_tensors"] == 12
        assert _control_by_prefix(d, "drop")["status"] == "fired"
        assert _control_by_prefix(d, "alias")["status"] == "inapplicable"
        assert _control_by_prefix(d, "underfill")["status"] == "inapplicable"

    def test_first_save_event_runs_composite_and_stays_clear(self, tmp_path):
        """[RED-AS-INSTALLED -> GREEN BY FIXTURE CALIBRATION] Pins
        event=first_save adding the composite without breaking a healthy
        artifact whose expert properties metadata CAN fully establish. Red if:
        FirstSaveGate were appended unconditionally (then a midpoint save
        would carry verdicts scoped to FIRST_SAVE). Calibration record, per
        this test's own prior instruction ("calibrate the fixture against
        that composite's contract; do not weaken this assertion"): the old
        fixture was a DENSE model, on which distinctness and bytes can only
        ever SKIP ("context declares no experts") -- no metadata exists that
        turns a declared-zero expert scope into a verified expert property,
        so demanding CLEAR from the composite on a dense artifact demanded
        that two legitimately absent properties count as verified. The
        composite's own MUST_PASS family is the per-expert sharded layout
        ("the only family in which every sub-gate can fully verify"); this
        fixture now IS that family. Assertions strengthened, not weakened:
        larger denominator, verdict count, composite content."""
        _probe_declared_or_calibrate(MOE_CFG, 8, 2)
        base = _make_base(tmp_path, _moe_full_tensors(), MOE_CFG, name="fs-base")
        ckpt = _materialize_artifact(tmp_path, _moe_full_tensors(), name="fs-ckpt")
        d = lsg.adjudicate_checkpoint(
            ckpt, event="first_save", run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 0, (
            f"first_save composite must not red a healthy sharded-MoE artifact; "
            f"got {d.blocking_reasons} -- the first reason names the failing "
            f"gate verbatim; calibrate the fixture, never these assertions")
        # Denominators on the wire (doctrine 2): 34 real tensors (2 q_proj +
        # 8 experts x 2 layers x 2 projections), 4 verdicts (3 always-gates
        # + the FIRST_SAVE composite), composite claiming 3 of 3 verified.
        assert d.report["inventory"]["real_tensors"] == 34
        assert len(d.gate_results) == 4
        composite = next(g for g in d.gate_results
                         if g["gate"] == "checkpoint.first_save")
        assert composite["verdict"] == "PASS"
        assert "verified at first save" in str(composite["detail"])

    def test_first_save_on_dense_clears_with_shrunken_named_denominator(self, tmp_path):
        """[FAILS-BEFORE -- LG3 core+composite edits] STRENGTHENED replacement
        for test_first_save_on_dense_blocks_as_stated_abstention (its full text
        and reasoning are quoted in the LG3 'Existing tests' section). Dense
        first-save is CLEAR now that applicability is machine-priced, and this
        version pins STRICTLY MORE than the old one: (i) the composite prices
        1/1 APPLICABLE -- the dense SKIPs shrink the DENOMINATOR, never the
        numerator; (ii) both inapplicable gates are NAMED in detail and
        evidence; (iii) the kind is data on the wire, not prose; (iv) it can
        never read as 'verified 3/3' or bare 'verified'. Red-makers: any SKIP
        shrinking the denominator (the lazy rewrite -- the stacked tests kill
        it), reason-string sniffing, or counting dense SKIPs as verified (the
        old test's red-maker, preserved)."""
        base, ckpt = _dense_base_with_ckpt(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, event="first_save", run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 0, f"dense first-save must be CLEAR post-LG3: {d.blocking_reasons}"
        composite = next(g for g in d.gate_results
                         if g["gate"] == "checkpoint.first_save")
        assert composite["verdict"] == "PASS"
        # The denominator pricing, asserted numerically (doctrine 2):
        assert composite["checked"] == 1 and composite["expected"] == 1
        detail = str(composite["detail"])
        assert "1/1 applicable" in detail
        assert "checkpoint.expert_distinctness" in detail
        assert "checkpoint.expert_bytes" in detail
        assert "3/3" not in detail
        # The kind is DATA on the wire for both expert sub-gates, not prose:
        by_gate = {g["gate"]: g for g in d.gate_results}
        assert by_gate["checkpoint.expert_distinctness"]["verdict"] == "SKIP"
        assert by_gate["checkpoint.expert_distinctness"]["abstention"] == "not_applicable"
        assert by_gate["checkpoint.expert_bytes"]["abstention"] == "not_applicable"
        assert set(composite["evidence"].get("inapplicable") or ()) == {
            "checkpoint.expert_distinctness", "checkpoint.expert_bytes"}


class TestModeConfusion:
    def test_truncated_full_blocks_and_is_not_excused_as_adapter(self, tmp_path):
        """[PASSES-BEFORE] The named direction: a full-FT checkpoint that lost
        tensors must NOT be waved through as 'it is only an adapter'. Red if:
        the MODE/full partial-population append in cross_check_population is
        deleted (one line)."""
        base, _ = _dense_base_with_ckpt(tmp_path)
        truncated = dict(list(_dense_full_tensors().items())[:4])  # lost 8 of 12
        ckpt = _materialize_artifact(tmp_path, truncated, name="trunc")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 1
        assert d.run_kind == "full"  # never re-labeled as an adapter
        assert any("MODE/full" in r for r in d.blocking_reasons)
        assert _control_by_prefix(d, "drop")["status"] == "fired"  # self-attributing

    def test_full_artifact_judged_as_lora_blocks_both_legs(self, tmp_path):
        """[PASSES-BEFORE] Red if: either the `contaminated` or the `unmarked`
        append in the lora branch of cross_check_population is deleted."""
        base, ckpt = _dense_base_with_ckpt(tmp_path)
        # adapter_prefix="" is byte-identical to the pre-demand default, pinned
        # here for the same reason as the other lora call sites: this test judges
        # a FULL artifact under kind="lora", so it reaches the prefix demand even
        # though no adapter exists to name. Without the pin the demand raises
        # GateUnmeasured (exit 3) and the MODE cross-checks this test exists to
        # pin are never evaluated -- the assertions below would be unreachable,
        # which is a silenced test, not a passing one.
        # fix45-C1 (#78): the new census demand sits behind the prefix demand
        # on the same road to the cross-checks (derive_declared_block raises
        # without --adapter-modules), so it is fed the same way -- the census
        # carries exactly the 12 module stems this run's config claims to have
        # adapted, which is what the launch-time census over the base tree
        # would have named. Pure scaffolding: the subject and the red-maker
        # (the two MODE/lora appends in cross_check_population) are untouched,
        # and nothing below asserts anything about the adapter denominator.
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()))
        assert d.exit_code == 1
        assert any("MODE/lora" in r and "BASE-WEIGHT" in r for r in d.blocking_reasons)
        assert any("MODE/lora" in r and "adapter marker" in r for r in d.blocking_reasons)

    def test_auto_kind_infers_full_for_unmarked_population(self, tmp_path):
        """[PASSES-BEFORE] Red if: the kind = 'lora' if frac >= 0.6 else 'full'
        line is flipped."""
        base, ckpt = _dense_base_with_ckpt(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="auto", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.run_kind == "full" and "auto:" in d.declared_basis["run_kind"]
        assert d.exit_code == 0


class TestFrozenScopeAndExtras:
    def test_frozen_regex_shrinks_denominator_honestly(self, tmp_path):
        """[PASSES-BEFORE] Saving only the trainable scope is CLEAR. Red if:
        the frozen_regex filter line in derive_declared_block is removed --
        then declared=12 vs 10 present -> completeness FAIL."""
        keep = {k: v for k, v in _dense_full_tensors().items() if "layers.5." not in k}
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, keep, name="frozen")
        cfg = _write_cfg(tmp_path, {"frozen_regex": r"layers\.5\."})
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base, train_config_path=cfg)
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        assert d.report["inventory"]["real_tensors"] == 10

    def test_extra_tensor_is_reported_not_blocking_by_default(self, tmp_path):
        """[PASSES-BEFORE] Pins the PERMISSIVE default (see Diagnosis S12:
        examined, bounded, documented -- not a defect). Red if: the
        decl.notes.append in the extras branch is deleted."""
        tensors = {**_dense_full_tensors(), "layers.9.unexpected.weight": ((8, 8), "F32")}
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, tensors, name="extra")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 0
        assert any("outside the declared set" in n for n in d.declared_basis["notes"])

    def test_strict_extras_flips_that_note_to_blocking(self, tmp_path):
        """[PASSES-BEFORE] The opt-in strict direction. Red if: the
        extras_blocking append is deleted (one line in adjudicate_checkpoint)."""
        tensors = {**_dense_full_tensors(), "layers.9.unexpected.weight": ((8, 8), "F32")}
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, tensors, name="extra")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}), strict_extras=True)
        assert d.exit_code == 1
        assert any("outside the declared set" in r for r in d.blocking_reasons)


# ---------------------------------------------------------------------------
# LoRA discrimination: BOTH directions, both named
# ---------------------------------------------------------------------------


class TestLoraDiscrimination:
    def test_lora_healthy_is_clear_against_adapter_denominator(self, tmp_path):
        """[PASSES-BEFORE] The named direction: a LoRA adapter must NOT be
        judged catastrophically incomplete against the full model's
        denominator. Denominators asserted: 31 real (fail-closed physical
        count: 24 judged adapter tensors + 7 save-state entries set aside
        per #80 -- counted in the artifact inventory, outside the judged
        population), 12 base, adapter set derived 24. Red if: the
        ADAPTER-SCOPE expert-zeroing block in
        derive_declared_block is deleted (the base's expert denominator then
        reattaches to the adapter). Calibration note: if the zero-expert-scope
        gates return bare ok() instead of an explicit skip for a declared-zero
        expert scope, core's tripwire VACUOUS-blocks this test red ON THE
        CURRENT TREE -- that would be a defect in checkpoint_gates.py, and
        blocking_reasons will name the gate id.
        fix45-C1 repair record (#78): the "adapter set derived 24"
        expectation stands unchanged, but its SOURCE moved. Pre-#78 the 24
        derived from the HF base header x training-config targets x rank --
        the cross-namespace product measured disjoint from every save this
        estate produces, which is why this green was witnessing the defect's
        oracle. The oracle is now the launch-time census, fed below as the
        12 artifact-namespace module stems this fixture's own artifact
        carries, written OUTSIDE the judged tree (_census_file). The wire
        text this test pins now reads "24 adapter tensors = 12 census
        modules x 2 naming templates"; every assertion below is untouched."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()))
        assert d.exit_code == 0, f"healthy lora must be CLEAR: {d.blocking_reasons}"
        assert d.run_kind == "lora"
        inv = d.report["inventory"]
        # #80: the inventory stays fail-closed over the PHYSICAL artifact --
        # 31 real entries = 24 judged adapter tensors + 7 non-adapter
        # checkpoint-namespace entries (6 optimizer.* + 1 rng_state). The 7
        # are set aside from the JUDGED population only ("all 24 declared
        # tensors present"), never from the artifact count.
        assert inv["real_tensors"] == 31 and inv["base_tensors"] == 12
        assert "24 adapter tensors" in d.declared_basis["fqns"]
        assert _control_by_prefix(d, "drop")["status"] == "fired"

    def test_lora_judged_as_full_is_not_cleared(self, tmp_path):
        """[PASSES-BEFORE] The symmetric named direction. Red if: the low-
        overlap abstain branch (fqns = None) is changed to fall back to the
        base header -- that would manufacture a denominator and let this pass."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base, train_config_path=cfg)
        assert d.exit_code == 1
        assert "fqn-map" in d.declared_basis["fqns"]  # the remediation is named
        assert any("VACUOUS" in g["verdict"] or g["verdict"] == "VACUOUS"
                   for g in d.gate_results)

    def test_lora_contamination_by_base_weights_blocks(self, tmp_path):
        """[PASSES-BEFORE] Red if: the contaminated append in the lora branch
        of cross_check_population is deleted. fix45-C1 (#78): fed the new
        census demand the honest 12-stem census so adjudication reaches the
        cross-check at all -- the census is scaffolding here; the
        contaminated append remains the sole named red-maker, and the
        verbatim base FQN below is still foreign to the census-derived
        declared set exactly as it was foreign to the old one."""
        tensors = {**_lora_tensors(),
                   "layers.0.self_attn.q_proj.weight": ((8, 8), "F32")}  # verbatim base FQN
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, tensors, name="lora-dirty")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()))
        assert d.exit_code == 1
        assert any("BASE-WEIGHT" in r for r in d.blocking_reasons)

    def test_auto_kind_infers_lora_from_markers_without_config_keys(self, tmp_path):
        """[RED-AS-INSTALLED -> GREEN BY FIXTURE CALIBRATION] Kind inference
        must not depend on a peft/kind key: rank and targets present, NO key
        from _KIND_KEYS anywhere, and the marker majority settles kind after
        measurement. Red if: kind-key absence stops deferring to the marker
        inference (kind stays 'auto', no cross-check branch runs).
        Calibration record: the old fixture passed an EMPTY config, which
        withheld rank/targets along with the kind key -- derivation then
        abstained by design (fqns=None -> save_complete VACUOUS-blocked,
        drop unconstructable, floor unsatisfied) and the test demanded exit 0
        over an absent denominator: the founding defect as an assertion. The
        intent (no KIND key) survives; the denominator sources stay
        independent. Assertions strengthened: the derived adapter count and
        the drop control's fire are now pinned, not implied by the verdict."""
        base, ckpt, _cfg = _healthy_lora(tmp_path)
        cfg = _write_cfg(tmp_path, {"lora_rank": 4,
                                    "lora_targets": ["q_proj", "v_proj"]},
                         name="no-kind-key.json")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="auto", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()))
        assert d.run_kind == "lora" and "auto:" in d.declared_basis["run_kind"]
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        # Denominator provenance pinned (#78, fix45-C1): 24 = 12 census
        # modules x 2 naming templates, the census being the launch-time
        # artifact-namespace module list fed above. The sentence this
        # replaces ("24 = 12 target parents x (A, B), derived from base
        # header x targets x rank -- nothing artifact-side") described the
        # defect's own provenance as the pin; restated, never weakened --
        # the count and the independence claim both stand.
        assert "24 adapter tensors" in d.declared_basis["fqns"]
        assert _control_by_prefix(d, "drop")["status"] == "fired"

    def test_auto_kind_with_no_denominator_keys_abstains_and_blocks(self, tmp_path):
        """[CONVERTED BY fix45-C1 INTO A REFUSAL TEST -- the FIRST of the two
        conversions the fix45-B brief licenses for tests "genuinely about the
        absence of a denominator"; named here per its naming duty]
        Justification: this test's subject was ALWAYS the absent denominator,
        never the abstention mechanism that expressed it. Pre-#78 the lora
        oracle derived from config targets x base header, so "empty config"
        WAS the absent-denominator state and stated-abstention-then-BLOCK was
        its honest expression. #78 moved the denominator to the
        --adapter-modules census and demoted targets/rank to provenance
        only, so an empty CONFIG is no longer a denominator statement at
        all: the absent-denominator state on the lora path is precisely "no
        census", and its honest expression is the GateUnmeasured refusal
        (class adapter_census_unavailable, the launcher's exit-3 arm), which
        is STRICTER than the retired abstention -- no verdicts get
        manufactured at all. The marker-inference half of the intent
        SURVIVES as a reachability witness: the census demand lives only in
        the lora branch of derive_declared_block, so with run_kind="auto"
        this exception is raised IFF marker majority resolved kind to lora
        (had kind resolved full, the full branch would abstain-and-block
        with NO exception and pytest.raises would fail red) -- the old
        "auto still classifies as lora" pin now convicts without a wire
        read. Red-maker, restated: if the lora branch ever FABRICATES an
        adapter set when no census exists, no exception reaches
        pytest.raises and this test names the flip -- the same conviction
        the old exit-0-flip red-maker carried."""
        base, ckpt, _cfg = _healthy_lora(tmp_path)
        with pytest.raises(lsg.GateUnmeasured) as exc_info:
            lsg.adjudicate_checkpoint(
                ckpt, run_kind="auto", base_model_dir=base,
                train_config_path=_write_cfg(tmp_path, {}, name="empty-cfg.json"),
                adapter_prefix="")
        msg = str(exc_info.value)
        assert msg.startswith("--adapter-modules was not supplied")
        assert (lsg._refusal_class(msg)
                == lsg._REFUSAL_ADAPTER_CENSUS_UNAVAILABLE)

    def test_lora_without_targets_abstains_and_blocks(self, tmp_path):
        """[CONVERTED BY fix45-C1 INTO A REFUSAL TEST -- the SECOND of the
        two conversions the fix45-B brief licenses; named here per its
        naming duty] Justification: the old subject was "no target key ->
        no fabricated adapter denominator", expressed through the retired
        `else: fqns = None` arm of the lora derivation. #78 severed the
        denominator from the config's target key entirely -- targets are
        provenance-only; the denominator is the census or nothing -- so
        "no targets" is no longer the absent-denominator state. Keeping an
        adjudication shape here would either feed a census (silently
        changing the subject to "targets are ignored", a claim the
        healthy-lora family does not need from THIS name) or assert the
        refusal. The refusal is the faithful successor: the only remaining
        way for this run to present WITHOUT an honest denominator is to
        carry no census, and the tool must refuse that -- loud, classified,
        and before any verdict exists. Red-maker, restated: if the lora
        derivation ever returns a guessed adapter set when the census is
        absent -- the exact regression the retired `else: fqns = None`
        sentence guarded -- no GateUnmeasured reaches pytest.raises and
        this test is red."""
        base, ckpt, _ = _healthy_lora(tmp_path)
        cfg = _write_cfg(tmp_path, {"peft_scheme": "lora", "lora_rank": 4},
                         name="no-targets.json")
        with pytest.raises(lsg.GateUnmeasured) as exc_info:
            lsg.adjudicate_checkpoint(
                ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
                adapter_prefix="")
        msg = str(exc_info.value)
        assert msg.startswith("--adapter-modules was not supplied")
        assert (lsg._refusal_class(msg)
                == lsg._REFUSAL_ADAPTER_CENSUS_UNAVAILABLE)

    def test_healthy_lora_over_moe_base_is_adjudicated_in_adapter_scope(
        self, tmp_path
    ):
        """[RED-UNDER-MUTANT adjudication/adapter-scope-inherits-base-expert-
        denominator; GREEN-ON-SHIPPED] MUST_FIRE for the MINT_ZERO_ONLY_IN_PROBE
        declared exception (`if not spec.frozen_regex and not expert_stems:` --
        the census-side spelling in adjudication.lora_structural_findings since
        the #78 restructure;
        the pre-#78 text named `expert_targets`, the retired HF-header oracle.
        The docstring is updated, never the assertions -- the same drift class
        as the two corpus rows repaired tonight).

        The matrix cell this suite never built (the mutation row's own "why"
        confesses it): a healthy LoRA adapter of a MoE base. Mutant: the
        adapter inherits the base's 8-expert/2-layer denominator, expert
        gates take the declared-experts-yet-absent VACUOUS door (the shape
        test_declared_experts_absent_never_shrinks pins), and the run
        BLOCKS -- the canonical false alarm this tool documents as its
        reason for existence, fired on a healthy adapter. Doctrine 5 is
        symmetric: crying wolf on a healthy artifact convicts the gate
        exactly as surely as passing a sick one.

        Why every pre-existing lora test is blind to it: they all sit on
        DENSE_CFG, where the probe's own two-source mint already yields a
        corroborated 0 before the local exception is even consulted -- on a
        dense base, deleting the exception is behaviourally invisible by
        construction. This fixture uses an MoE base (34 tensors: 2 q_proj +
        8 experts x 2 layers x 2 projections) and a non-expert target list
        (q_proj only), so adapter-scope zero can ONLY arrive through the
        line under test. Denominators on the wire (doctrine 2): 4 real
        adapter tensors, 34 base tensors, derived set 4 = 2 parents x (A, B)
        -- and CLEAR additionally requires the drop control to have FIRED,
        so the exit code can never be the empty-sweep pass.
        fix45-C1 repair record (#78): the mint under test now reads the
        --adapter-modules census instead of matching config targets against
        the HF header, so this test feeds the census the 2 artifact-namespace
        stems layers.{0,1}.self_attn.q_proj -- the modules the fixture
        artifact actually carries. Every assertion below is untouched: "2
        parents x (A, B)" above now reads on the wire as "4 adapter tensors
        = 2 census modules x 2 naming templates", and the MUST_FIRE property
        is preserved -- delete the adapter-scope zero mint and the probe's
        inherited 8/2 denominator reattaches, both expert gates take their
        VACUOUS door on declared-8/examined-0, exit flips to 1, red on the
        first assertion, exactly as the mutant analysis above demands."""
        _probe_declared_or_calibrate(MOE_CFG, 8, 2)
        base = _make_base(tmp_path, _moe_full_tensors(), MOE_CFG,
                          name="lom-base")
        adapters = {}
        for ly in range(2):
            stem = f"layers.{ly}.self_attn.q_proj"
            adapters[f"{stem}.adapter.linear_in.weight"] = ((4, 8), "F32")
            adapters[f"{stem}.adapter.linear_out.weight"] = ((8, 4), "F32")
        ckpt = _materialize_artifact(tmp_path, adapters, name="lom")
        cfg = _write_cfg(tmp_path, {"peft_scheme": "lora", "lora_rank": 4,
                                    "lora_targets": ["q_proj"]})
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=_census_file(
                tmp_path,
                [f"layers.{ly}.self_attn.q_proj" for ly in range(2)]))
        assert d.exit_code == 0, (
            f"healthy LoRA-of-MoE must CLEAR in adapter scope, got "
            f"{d.exit_code}: {d.blocking_reasons} -- under the mutant the "
            f"first reasons are the expert gates blocking on experts this "
            f"adapter was never declared to contain")
        assert "ADAPTER SCOPE" in d.declared_basis["num_experts"]
        inv = d.report["inventory"]
        assert inv["real_tensors"] == 4 and inv["base_tensors"] == 34
        assert "4 adapter tensors" in d.declared_basis["fqns"]
        by_gate = {g["gate"]: g for g in d.gate_results}
        assert by_gate["checkpoint.expert_distinctness"]["verdict"] == "SKIP"
        assert (by_gate["checkpoint.expert_distinctness"]["abstention"]
                == "not_applicable")
        assert by_gate["checkpoint.expert_bytes"]["verdict"] == "SKIP"
        # The mint must also reach the CONTROL layer's view of the context:
        # alias inapplicability carries num_experts=0 only if the exception
        # ran (mutant: the same string reads 8).
        alias = _control_by_prefix(d, "alias")
        assert alias["status"] == "inapplicable"
        assert "num_experts=0" in str(alias.get("reason"))
        assert _control_by_prefix(d, "drop")["status"] == "fired"

    def test_adapter_scope_mint_does_not_apply_when_adapters_target_experts(
        self, tmp_path
    ):
        """[FAILS-BEFORE on the fix38 lines marked [new]; the original three
        assertions are PASSES-BEFORE fences, kept byte-for-byte -- stated per
        the house rule] Over-application fence for the adapter-scope mint,
        now a DISCRIMINATING fence instead of a vacuous one. On the pre-fix38
        tree the mint never fired for any non-empty target list (the census
        matched a synthetic wrapper string engineered to satisfy the
        classifier, so every target read as expert-resident), which means
        the original three assertions passed vacuously: they pinned "the
        mint does not apply HERE" while the mint applied NOWHERE -- exactly
        how a fence can be green while proving nothing, which is how the
        defect got here. fix38 makes the census real, so this test now
        proves BOTH directions against the same real MoE base: [new] a
        non-expert target list (q_proj) MUST mint the adapter-scope zero,
        with its denominator on the basis string -- proving the branch is
        live at all; and the original expert-target list (linear_fc1 /
        linear_fc2) MUST NOT mint -- proving the branch is scoped. A census
        that always mints (a regression toward empty-input laundering) dies
        on the original assertions; a census that never mints (the pre-fix
        defect itself returning) dies on the [new] ones. Calibration-loud,
        per this file's own rule, restated for the new ground truth: the
        fence stands on _matches_expert_family recognising the REAL header
        names layers.{ly}.experts.{e}.linear_fc{1,2}.weight as expert-family
        over the in-scope base population (the same atoms the probe's census
        applies, and the same projections TestProbeAliasSpellings exercises
        green); if that ever stops being true, expert_base reads 0, every
        target list resolves expert-free, the mint fires here, and this test
        dies red on the original assertions -- fix the classifier pin, never
        this assertion. fix45-C1 repair record (#78): the direct
        derive_declared_block calls below now carry the census the lora
        branch demands, and the population pins moved from config-target
        counts to census-module counts, because the census -- not the target
        list -- is what the mint now measures. Leg A feeds the 32 expert
        parent stems the fixture base actually exposes
        (layers.{ly}.experts.{e}.linear_fc{1,2}), so the retention note
        reads "32 of 32 census modules" (was "2 of 2" -- that counted CONFIG
        TARGETS, a key tally that post-#78 prices nothing). Leg B feeds the
        2 non-expert attention stems, so the mint basis reads "ADAPTER
        SCOPE: 0 of 2 census modules". The retired assertion
        "32 expert FQNs" in the mint basis is named here per the house
        rule: it was an HF-header-population expectation riding inside the
        adapter-scope basis -- the population the defective oracle measured.
        The 32-module population now travels in the retention NOTE of leg A,
        where it is asserted instead. The fence stays discriminating in
        both directions: a classifier misreading the 32 expert stems as
        non-expert flips leg A red (mint over-fires), one misreading
        attention stems as expert flips leg B red (mint dies)."""
        _probe_declared_or_calibrate(MOE_CFG, 8, 2)
        spec = lsg.resolve_train_spec(
            {"peft_scheme": "lora", "lora_rank": 4,
             "lora_targets": ["linear_fc1", "linear_fc2"]},
            "test://cfg", "lora", None)
        base = lsg.BaseModel(
            model_dir=tmp_path, config=MOE_CFG,
            tensors={k: (v[0], "float32")
                     for k, v in _moe_full_tensors().items()},
            tensors_source="test://synthetic")
        decl = lsg.derive_declared_block(
            base, spec, set(), "",
            adapter_modules=_census(
                f"layers.{ly}.experts.{e}.{p}"
                for ly in range(2) for e in range(8)
                for p in ("linear_fc1", "linear_fc2")))
        assert decl.num_experts == 8
        assert decl.num_moe_layers == 2
        assert "ADAPTER SCOPE" not in decl.experts_basis
        # [new] fix38, re-based #78 (fix45-C1): the retention is a NAMED
        # record carrying the census denominator, not a silent fall-through
        # -- the wire testifies that all 32 census modules (the fixture
        # base's real expert parent stems) were classified expert-family by
        # the gates' own name atoms and found resident. The pre-#78 "2 of 2"
        # counted config targets; post-#78 the census prices modules.
        assert any(
            "ADAPTER SCOPE RETAINS EXPERTS: 32 of 32" in n for n in decl.notes
        ), f"mint retention note missing or miscounted: {decl.notes!r}"
        # [new] fix38 positive control, the leg that makes this fence
        # convicting: the SAME base with a census of non-expert stems only.
        # Pre-fix38 the dead census minted nothing for any non-empty list,
        # so this leg's assertions are red on that tree; green requires the
        # census classifier to have examined both attention stems and found
        # 0 of 2 expert-resident.
        attention_spec = lsg.resolve_train_spec(
            {"peft_scheme": "lora", "lora_rank": 4,
             "lora_targets": ["q_proj"]},
            "test://cfg", "lora", None)
        attention_decl = lsg.derive_declared_block(
            base, attention_spec, set(), "",
            adapter_modules=_census(
                f"layers.{ly}.self_attn.q_proj" for ly in range(2)))
        assert attention_decl.num_experts == 0
        assert attention_decl.num_moe_layers == 0
        assert ("ADAPTER SCOPE: 0 of 2 census modules"
                in attention_decl.experts_basis)
        assert "base model's own declaration was 8" in attention_decl.experts_basis


    def test_unknown_targets_over_moe_base_abstain_and_keep_the_base_denominator(
        self, tmp_path
    ):
        """[REPAIRED BY fix45-C1 -- red under #78 until re-homed; the
        red-maker the test always carried survives, named below] MUST_FIRE
        for the false-green direction of the adapter-scope mint, re-homed
        onto the arm of that doctrine that #78 left standing. What changed:
        pre-#78 the mint's input was the CONFIG TARGET LIST, so "no target
        key" WAS the unknown state, and the shipped answer was the stated
        abstention this test pinned ("ADAPTER SCOPE UNKNOWN" note,
        fqns=None, expert gates blocking on the inherited 8). Post-#78 the
        mint's input is the --adapter-modules census, and targets are
        provenance-only: a census-fed run's expert scope is MEASURED (the
        census names the modules this artifact was told to adapt;
        classifying them is measurement, never absence), so feeding a
        census here and still demanding VACUOUS expert gates would assert
        precisely the laundering this test exists to forbid. The genuinely
        unknown states that SURVIVE #78 are: (a) no census at all -- the
        GateUnmeasured refusal, pinned by the two named refusal conversions
        in this class; and (b) a pinned frozen_regex whose semantics this
        tool cannot verify -- the shipped conservative refusal-to-mint arm,
        WHICH THIS TEST NOW DRIVES: the mint is refused BY NAME, the
        probe-derived 8/2 base denominator stays attached, and BOTH expert
        gates VACUOUS-block NAMING the inherited count (declared 8,
        examined 0). Denominators on the wire (doctrine 2): 4 real adapter
        tensors (2 census modules x 2 naming templates -- FQN completeness
        was never the unknown side of this scenario, and post-#78 it is
        honestly measured, so the block's attribution is SHARPER than
        before: the two expert gates alone), 34 base tensors (2 q_proj +
        8 experts x 2 layers x 2 projections), and the inherited 8 appears
        verbatim in the gating detail. Retired assertions, named per the
        house rule: the "ADAPTER SCOPE UNKNOWN" note text (vocabulary of
        the retired unknown-targets arm; its successor note is asserted
        below) and "abstains" in the fqns basis (the census IS an honest
        FQN denominator, so derivation no longer abstains in this state --
        that assertion was coupled to the defective oracle). Red-maker,
        preserved: delete the `not spec.frozen_regex` guard so the mint
        fires under an uninterpretable scope filter, and exit flips to 0 --
        red on the first assertion; the false-green direction stays named."""
        _probe_declared_or_calibrate(MOE_CFG, 8, 2)
        base = _make_base(tmp_path, _moe_full_tensors(), MOE_CFG,
                          name="unkm-base")
        adapters = {}
        for ly in range(2):
            stem = f"layers.{ly}.self_attn.q_proj"
            adapters[f"{stem}.adapter.linear_in.weight"] = ((4, 8), "F32")
            adapters[f"{stem}.adapter.linear_out.weight"] = ((8, 4), "F32")
        ckpt = _materialize_artifact(tmp_path, adapters, name="unkm")
        cfg = _write_cfg(tmp_path, {"peft_scheme": "lora", "lora_rank": 4,
                                    "frozen_regex": r"layers\.5\."},
                         name="unknown-targets-moe.json")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=_census_file(
                tmp_path,
                [f"layers.{ly}.self_attn.q_proj" for ly in range(2)]))
        # The run BLOCKS -- the stated, blocking refusal-to-mint (exit 1),
        # consistent with the low-overlap full-FT precedent; the exit-3
        # refusal is reserved for the no-census state and pinned by the two
        # named refusal conversions in this class.
        assert d.exit_code == 1
        assert "ADAPTER SCOPE" not in d.declared_basis["num_experts"]
        assert any("adapter-scope expert mint refused" in n
                   for n in d.declared_basis["notes"])
        inv = d.report["inventory"]
        assert inv["real_tensors"] == 4 and inv["base_tensors"] == 34
        by_gate = {g["gate"]: g for g in d.gate_results}
        assert by_gate["checkpoint.expert_distinctness"]["verdict"] == "VACUOUS"
        assert by_gate["checkpoint.expert_bytes"]["verdict"] == "VACUOUS"
        assert "8 experts" in str(
            by_gate["checkpoint.expert_distinctness"]["detail"])
        assert ("4 adapter tensors = 2 census modules"
                in d.declared_basis["fqns"])


class TestLoraStructuralBinding:
    def test_phantom_adapter_blocks(self, tmp_path):
        """[PASSES-BEFORE] Adapter whose parent module does not exist. Red if:
        the phantom append in lora_structural_findings is deleted.
        fix34 calibration record (fixture, not assertion): the ghost FQN is
        now spelled in the estate's Megatron-Bridge shape, because the
        phantom leg only runs on tensors the PINNED recognizer matches, and
        post-T2 that recognizer is the Megatron-Bridge DEFAULT. Spelled in
        the retired HF convention the ghost made the sweep match 0 of 25,
        the vacuity refusal fired, and the phantom leg itself was never
        reached -- "phantom modules" vanished from the reasons and this test
        went red under T2 with its actual red-maker untouched (the wrong red:
        a blocked-for-the-wrong-reason run tells the operator nothing about
        the leg this test exists to guard). Option (a) of the fix34 brief:
        the parent-binding sweep is naming-invariant by construction, so the
        estate's real adapter shape is the one the shipped default must be
        proven against; pinning the retired HF calibration here instead
        would leave the default recognizer's phantom leg with no MUST_FIRE
        at all -- precisely the doctrine-3 gap this suite exists to forbid.
        The assertions below are unchanged. fix45-C1 (#78): the
        parent-binding pool the sweep checks is now the census_parents set
        (the 12 fixture stems, fed below via _census_file in the artifact
        namespace); the ghost stem is absent from it, so the phantom leg
        fires on the same wrong FQN as before, and binding each healthy
        adapter to its census parent remains the mechanism under test."""
        tensors = {**_lora_tensors(),
                   "ghost.mod.adapter.linear_in.weight": ((4, 8), "F32")}
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, tensors, name="lora-ghost")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()))
        assert d.exit_code == 1
        assert any("phantom modules" in r for r in d.blocking_reasons)

    def test_wrong_rank_shape_blocks(self, tmp_path):
        """[PASSES-BEFORE] (rank, in) violation named with both shapes. Red if:
        the shape_bad append is deleted.
        fix34 calibration record (fixture, not assertion): the victim key is
        spelled in the Megatron-Bridge shape for the same reason recorded in
        test_phantom_adapter_blocks, doubled -- the shape leg fires only for
        a tensor that BOTH matches the pinned recognizer AND appears in
        decl.derived_adapter, and that map is GENERATED from the shipped
        (Megatron-Bridge) templates. A retired-HF spelling would fail both
        counts at once, leaving the leg exercised against zero tensors: the
        vacuity shape this suite exists to refuse, now with no gate
        verdict that would even name it. The overwrite stays an overwrite
        (same FQN, 24 tensors, one misshapen), so save_complete stays clean
        and the shape leg remains the ONLY blocking leg.
        fix45-C1 (#78): the declared shapes the shape leg compares against
        now derive ONLY from census parent dims x config rank, so the
        fixture census carries (out=8, in=8) for every stem. A names-only
        census would leave the tool's shape abstention in place and let
        this test pass over ZERO exercised comparisons -- precisely the
        dead control the brief forbids; the dims are what keep the
        red-maker live."""
        tensors = _lora_tensors()
        # rank 2 != 4
        tensors["layers.0.self_attn.q_proj.adapter.linear_in.weight"] = ((2, 8), "F32")
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, tensors, name="lora-badrank")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_modules=_census_file(
                tmp_path, _lora_census_stems(),
                dims={s: (8, 8) for s in _lora_census_stems()}))
        assert d.exit_code == 1
        assert any("violate the declared" in r for r in d.blocking_reasons)

    def test_pinned_prefix_adapter_is_clear(self, tmp_path):
        """[FAILS-BEFORE -- Edit 6] HF-PEFT exports prefix every FQN with
        e.g. base_model.model.; with the prefix pinned, binding must strip it
        before the parent lookup. On the current tree every parent misses
        (prefix not stripped) -> 24 phantom -> BLOCKED: the knob the comment
        names can never produce CLEAR."""
        base, ckpt, cfg = _healthy_lora(tmp_path, prefix="base_model.model.")
        # fix45-C1 (#78) census scaffolding: the census stems stay UNPREFIXED
        # -- the prefix is export clothing the generator applies and the
        # binding strips; the census names base-tree modules, which is
        # exactly what the launch-time census would record.
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="base_model.model.",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()))
        assert d.exit_code == 0, f"correctly-pinned prefix must be CLEAR: {d.blocking_reasons}"
        # #80: 31 real = 24 judged adapters + 7 set-aside save-state entries
        # (6 optimizer.* + 1 rng_state); the inventory counts the physical
        # artifact, the judged adapter population stays 24.
        assert d.report["inventory"]["real_tensors"] == 31

    def test_mismatched_suffix_pin_blocks_as_vacuous_binding(self, tmp_path):
        """[FAILS-BEFORE: kwarg adapter_suffixes does not exist pre-patch ->
        TypeError, red] Operator pins the WRONG naming and the structural
        sweep binds 0 of 24 adapters: the vacuity is named, with its
        denominator, and blocks with a recalibration instruction. Post
        agreement-check, pinning ONLY the recognizer wrong (this test's
        previous fixture shape) is refused even earlier -- exit 3, naming the
        disagreeing pair; that refusal carries its own MUST_FIRE in
        TestAdapterNamingAgreement. The only route left to a vacuous sweep is
        a COHERENTLY wrong pin -- recognizer and generator templates agreeing
        with each other and neither matching the artifact -- which is exactly
        what this fixture now pins. Assertions unchanged: exit 1, "0 of 24",
        "vacuous detector"."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        # fix45-C1 (#78) census scaffolding: the 12-stem census gets
        # adjudication past the census demand so the coherently-wrong pin
        # can reach the structural sweep -- the route to "0 of 24" and the
        # assertions are unchanged; the refusal-for-inconsistent-knobs arm
        # stays pinned by TestAdapterNamingAgreement.
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="",
            adapter_marker=r"(?:lora_[AB]|delta_[AB])",
            adapter_suffix_re=r"\.delta_[AB]$",
            adapter_suffixes=(".delta_A", ".delta_B"),
            adapter_modules=_census_file(tmp_path, _lora_census_stems()))
        assert d.exit_code == 1
        # #80: the vacuity sweep's denominator is the EXAMINED artifact
        # population -- the structural name-search runs over every real
        # entry, including the 7 set-aside save-state entries (set aside
        # from the judged population, never from the search). "0 of 24"
        # printed pre-#80 only because fixture-real == declared == 24
        # coincided; re-pinning 24 now would pin that coincidence oracle
        # into a MUST_FIRE, and narrowing the sweep's domain to make 24
        # true again would blind the sweep to the excused namespace
        # (fail-open). Teeth unchanged: exit 1, vacuity named, zero bound
        # over the full 31-entry artifact.
        assert any("0 of 31" in r and "vacuous detector" in r
                   for r in d.blocking_reasons)


# ---------------------------------------------------------------------------
# --fqn-map: the remediation the basis text always named (D4)
# ---------------------------------------------------------------------------


def _renamed_full_tensors():
    return {f"megatron.{k}": v for k, v in _dense_full_tensors().items()}


class TestFqnMap:
    def test_dcp_named_full_blocks_without_map_and_names_flag(self, tmp_path):
        """[PASSES-BEFORE] Red if: the basis sentence naming --fqn-map is
        reworded away (the test pins the operator-visible remediation)."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, _renamed_full_tensors(), name="dcp")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 1
        assert "--fqn-map" in d.declared_basis["fqns"]

    def test_map_closes_completeness_and_clears(self, tmp_path):
        """[FAILS-BEFORE -- Edit 5] Supplying the planner-exported list in the
        ARTIFACT namespace must make completeness measurable and a healthy
        DCP-named artifact CLEAR. Fails before: no fqn_map parameter exists."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, _renamed_full_tensors(), name="dcp")
        map_path = tmp_path / "fqn-map.json"
        map_path.write_text(json.dumps(sorted(_renamed_full_tensors())), encoding="utf-8")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}), fqn_map=map_path)
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        assert "--fqn-map" in d.declared_basis["fqns"]
        assert _control_by_prefix(d, "drop")["status"] == "fired"

    def test_zero_overlap_stale_map_fails_not_vacuous(self, tmp_path):
        """[FAILS-BEFORE -- Edit 5] A stale map shares zero names with the
        artifact: the note must say so and completeness must FAIL, not pass."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, _renamed_full_tensors(), name="dcp")
        map_path = tmp_path / "stale-map.json"
        map_path.write_text(json.dumps([f"stale.{k}" for k in _dense_full_tensors()]),
                            encoding="utf-8")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}), fqn_map=map_path)
        assert d.exit_code == 1
        assert any("zero names" in n for n in d.declared_basis["notes"])

    def test_map_loader_refuses_bad_inputs(self, tmp_path):
        """[FAILS-BEFORE -- Edit 5] Missing file, malformed object, non-string
        entry, and the empty map -- all UNMEASURED, never a denominator."""
        with pytest.raises(lsg.GateUnmeasured, match="not found"):
            lsg._load_fqn_map(tmp_path / "nope.json")
        bad_obj = tmp_path / "obj.json"
        bad_obj.write_text('{"wrong_key": []}')
        with pytest.raises(lsg.GateUnmeasured, match="declared_fqns"):
            lsg._load_fqn_map(bad_obj)
        bad_entry = tmp_path / "entry.json"
        bad_entry.write_text('["a.b", 7]')
        with pytest.raises(lsg.GateUnmeasured, match="non-string"):
            lsg._load_fqn_map(bad_entry)
        empty = tmp_path / "empty.json"
        empty.write_text("[]")
        with pytest.raises(lsg.GateUnmeasured, match="ZERO fqns"):
            lsg._load_fqn_map(empty)
        both = tmp_path / "ok.json"
        both.write_text('{"declared_fqns": ["a.b"]}')
        fqns, basis = lsg._load_fqn_map(both)
        assert fqns == ("a.b",) and "--fqn-map" in basis

    def test_map_ignored_for_lora_with_note(self, tmp_path):
        """[FAILS-BEFORE -- Edit 5] A full-model FQN list must never reattach
        to an adapter run."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        map_path = tmp_path / "full-map.json"
        map_path.write_text(json.dumps(sorted(_dense_full_tensors())), encoding="utf-8")
        # fix45-C1 (#78): the census below is the very thing the IGNORED
        # note now names ("the adapter declared set derives from the
        # launch-time live-module census ... never from a full-model FQN
        # list") -- feeding it keeps the test's subject (the full-model map
        # must not reattach to an adapter run) measurable at all.
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            fqn_map=map_path, adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()))
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        assert any("IGNORED" in n for n in d.declared_basis["notes"])


# ---------------------------------------------------------------------------
# Controls: statuses, attribution, floors (D1/D2/D3)
# ---------------------------------------------------------------------------


class TestControlFloors:
    @staticmethod
    def _floor_census(tmp_path):
        """fix45/#78 vehicle census for this class's lora scaffolding.

        Every test in this class rides the _healthy_lora fixture as a
        VEHICLE to exercise the control-floor machinery (the any_fired
        floor, status naming, the framework tripwire); none of them has
        the lora oracle as its subject. Post-#78 that oracle refuses to
        run without an --adapter-modules census, so each vehicle now
        carries the minimum honest one: the 12 module stems the fixture's
        adapters attach to, computed from the run's own declared
        structure (the DENSE_CFG 6 layers x the 2 LORA_TRAIN targets) in
        the fixture's Megatron namespace -- NOT read off the artifact, so
        a truncated or misnamed save still deviates from it, and every
        red-maker named in the docstrings below is untouched. Written by
        _census_file OUTSIDE the judged tree; names-only, so shape checks
        abstain BY NAME (an abstention no assertion here inspects). No
        assertion in this class moved."""
        return _census_file(
            tmp_path,
            [f"layers.{i}.self_attn.{w}"
             for i in range(6) for w in ("q_proj", "v_proj")],
        )

    def test_empty_control_sweep_blocks_via_any_fired_floor(self, tmp_path):
        """[PASSES-BEFORE] The sweep-level all([]): controls=() must not pass.
        Red if: the `if not any_fired:` block is deleted.
        fix45: the lora vehicle now feeds the honest #78 census (see
        _floor_census); assertions and red-makers unchanged."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="", controls=(),
            adapter_modules=self._floor_census(tmp_path))
        assert d.exit_code == 1
        assert any("no MUST_FIRE control fired" in r for r in d.blocking_reasons)

    def test_unknown_control_name_blocks_as_unconstructable(self, tmp_path):
        """[PASSES-BEFORE] Red if: the _CONTROL_BUILDERS.get guard is replaced
        by direct indexing wrapped in try/except-pass.
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="", controls=("telemetry",),
            adapter_modules=self._floor_census(tmp_path))
        assert d.exit_code == 1
        assert d.controls[0]["status"] == "unconstructable"

    def test_quiet_detector_blocks_even_though_gates_pass(self, tmp_path, monkeypatch):
        """[PASSES-BEFORE] Injected defect, detector accepts it: MUST flip the
        verdict to BLOCKED. Gate monkeypatched per the in-repo precedent
        (test_hunt_finding_repairs.py); readers untouched. Red if: the
        not_fired branch's reasons.append is deleted.
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        monkeypatch.setattr(
            lsg.SaveCompletenessGate, "run",
            lambda self, c: _gr(Verdict.PASS, lsg.SaveCompletenessGate.id, "quiet fake"))
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=self._floor_census(tmp_path))
        assert d.exit_code == 1
        assert _control_by_prefix(d, "drop")["status"] == "not_fired"
        assert any("stayed QUIET" in r for r in d.blocking_reasons)

    def test_inconclusive_only_blocks_with_named_reason(self, tmp_path, monkeypatch):
        """[FAILS-BEFORE -- Edit 4] On the current tree an 'inconclusive'
        status falls through the if/elif chain unremarked; only the floor
        catches it, with the WRONG reason ('no control fired'). After the
        patch the reason names INCONCLUSIVE and why.
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        monkeypatch.setitem(lsg._CONTROL_BUILDERS, "x",
                            lambda ctx, bl: {"control": "x", "status": "inconclusive",
                                             "inconclusive_reason": "synthesized"})
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="", controls=("x",),
            adapter_modules=self._floor_census(tmp_path))
        assert d.exit_code == 1
        assert any("INCONCLUSIVE" in r and "synthesized" in r
                   for r in d.blocking_reasons)

    def test_skipped_only_lands_on_the_floor_not_on_silence(self, tmp_path, monkeypatch):
        """[PASSES-BEFORE] 'skipped' (probe vocabulary for inapplicable) is
        recorded-only, and the any_fired floor still bites. Red if: the floor
        is deleted.
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        monkeypatch.setitem(lsg._CONTROL_BUILDERS, "x",
                            lambda ctx, bl: {"control": "x", "status": "skipped"})
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="", controls=("x",),
            adapter_modules=self._floor_census(tmp_path))
        assert d.exit_code == 1
        assert any("no MUST_FIRE control fired" in r for r in d.blocking_reasons)

    def test_unrecognized_status_blocks_and_is_named(self, tmp_path, monkeypatch):
        """[FAILS-BEFORE -- Edit 4] A status this loop cannot read must not be
        read as inapplicable: the reason names 'unrecognized status'. (The
        exit code was already nonzero via the floor; the NAMING is what is
        new, and it is what the library caller matches on.)
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        monkeypatch.setitem(lsg._CONTROL_BUILDERS, "x",
                            lambda ctx, bl: {"control": "x", "status": "sparkly"})
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="", controls=("x",),
            adapter_modules=self._floor_census(tmp_path))
        assert d.exit_code == 1
        assert any("unrecognized status" in r and "sparkly" in r
                   for r in d.blocking_reasons)

    def test_framework_tripwire_pass_over_zero_checked_blocks(self, tmp_path, monkeypatch):
        """[PASSES-BEFORE] Red if: the `if r.verdict is Verdict.PASS and
        r.coverage.checked == 0` loop is deleted.
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        class _VacuousOk:
            def run(self, ctx):
                return GateResult(gate_id="test.vacuous", verdict=Verdict.PASS,
                                  coverage=Coverage(0, "units"), detail="all([])")

        monkeypatch.setattr(lsg, "_ALWAYS_GATES", (_VacuousOk,))
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=self._floor_census(tmp_path))
        assert d.exit_code == 1
        assert any("framework invariant breach" in r for r in d.blocking_reasons)


class TestStackedAliasAttribution:
    """Unit matrix over control_alias's stacked leg. Classification is pinned
    by monkeypatching the layout helpers (gates-layer functions, not readers);
    gate verdicts are faked per the hunt-file precedent. All four rows are
    [FAILS-BEFORE] because the pre-patch builder signature takes no baselines
    argument AND because rows 1-2 name semantics the old code lacks."""

    def _ctx(self):
        tm = lambda fqn: lsg.TensorMeta(  # noqa: E731 -- fixture-local shorthand
            fqn=fqn, shape=(16, 4), dtype="float32",
            storage_id=f"store://{fqn}", kind="tensor")
        return lsg.CheckpointGateContext(
            tensors=(tm("L0.moe.experts.weight"), tm("L1.moe.experts.weight")),
            declared_fqns=("L0.moe.experts.weight", "L1.moe.experts.weight"),
            num_experts=16, num_moe_layers=2,
            expected_expert_bytes=None, origin="test://synthetic")

    def _pin_stacked(self, monkeypatch):
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: list(ts))
        monkeypatch.setattr(lsg, "_split_expert_layouts", lambda c: ({}, list(c), []))
        monkeypatch.setattr(lsg, "_layer_normalized_stem",
                            lambda f: f.split(".", 1)[1] if "." in f else f)

    def test_confounded_baseline_is_inconclusive_not_fired(self, monkeypatch):
        """[FAILS-BEFORE -- Edits 1/2] mirror of the probe's
        test_blocking_baseline_is_inconclusive_not_fired, now for THIS tool."""
        self._pin_stacked(monkeypatch)
        monkeypatch.setattr(lsg.ExpertDistinctnessGate, "run",
                            lambda self, c: _gr(Verdict.FAIL, self.id))
        out = lsg.control_alias(self._ctx(), {lsg.ExpertDistinctnessGate.id:
                                _gr(Verdict.FAIL, lsg.ExpertDistinctnessGate.id)})
        assert out["status"] == "inconclusive" and out["confounded"] is True
        assert "baseline" in out["inconclusive_reason"]

    def test_error_verdict_is_not_credited(self, monkeypatch):
        """[FAILS-BEFORE -- Edits 1/2] A crash on the injected copy is a
        malfunction, not a detection: inconclusive, confounded False. The old
        code credited res.blocking -- i.e., it credited the crash."""
        self._pin_stacked(monkeypatch)
        monkeypatch.setattr(lsg.ExpertDistinctnessGate, "run",
                            lambda self, c: _gr(Verdict.ERROR, self.id))
        out = lsg.control_alias(self._ctx(), {lsg.ExpertDistinctnessGate.id:
                                _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)})
        assert out["status"] == "inconclusive" and out["confounded"] is False
        assert out["verdict"] == "ERROR"

    def test_clean_baseline_and_fail_fires(self, monkeypatch):
        """[FAILS-BEFORE -- semantics PASSES-BEFORE-equivalent; red-maker if
        the patch rots: flip `res.verdict is Verdict.FAIL` to `res.blocking` in
        _attributed_status, and test_error_verdict_is_not_credited goes red
        with this one still green -- the pair is the detector's two controls.]"""
        self._pin_stacked(monkeypatch)
        monkeypatch.setattr(lsg.ExpertDistinctnessGate, "run",
                            lambda self, c: _gr(Verdict.FAIL, self.id))
        out = lsg.control_alias(self._ctx(), {lsg.ExpertDistinctnessGate.id:
                                _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)})
        assert out["status"] == "fired" and out["confounded"] is False
        assert out["baseline_verdict"] == "PASS"

    def test_acceptance_stays_not_fired(self, monkeypatch):
        """[FAILS-BEFORE -- arity] True negative naming preserved."""
        self._pin_stacked(monkeypatch)
        monkeypatch.setattr(lsg.ExpertDistinctnessGate, "run",
                            lambda self, c: _gr(Verdict.PASS, self.id))
        out = lsg.control_alias(self._ctx(), {lsg.ExpertDistinctnessGate.id:
                                _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)})
        assert out["status"] == "not_fired"


class TestProbeAliasSpellings:
    """Doctrine 3 for Edits 1/2: the probe's alias control, pointed at a
    REAL (unmonkeypatched) distinctness gate, on both spellings. The global
    spelling is the MUST_FIRE the old grouping never saw (red before, green
    after); the local suffix spelling is the MUST_PASS proving the change is
    a superset (green on both trees -- stated, per the fail-before rule:
    invariance fences are green before by construction, and their red-makers
    are named in the docstring). Both fixtures carry fc1 AND fc2 so the
    classifier's family table resolves an expected count of 16 == checked;
    single-projection variants would (correctly) read UNDERCOVERED at 8/16
    and confound the baseline, which is fixture arithmetic, not a verdict."""

    def _probe_ctx(self, fqns):
        tms = tuple(lsg.TensorMeta(fqn=f, shape=(4, 4), dtype="float32",
                                   storage_id=f"store://{f}", kind="tensor")
                    for f in fqns)
        return lsg.CheckpointGateContext(
            tensors=tms, declared_fqns=tuple(fqns),
            num_experts=8, num_moe_layers=1,
            expected_expert_bytes=None, origin="test://synthetic")

    def _run_control(self, ctx, n=4):
        assert lsg._probe_alias_control is not None, (
            "probe unimportable -- the sys.path insertion in _load_gate "
            "regressed; fix the loader, never this guard")
        baseline = lsg.ExpertDistinctnessGate().run(ctx)
        assert not baseline.blocking, (
            f"fixture drifted: 16-of-16 healthy experts must PASS pre-"
            f"injection ({baseline.detail}) -- calibrate the fixture, never "
            f"the assertion")
        return lsg._probe_alias_control(ctx, n, baseline=baseline)

    def test_probe_alias_control_fires_on_global_spelling(self):
        """[FAILS-BEFORE -- Edits 1/2] The spelling the hand-rolled loop read
        as 'fused': pre-patch this returns skipped; post-patch it aliases 4
        members of a same-stem group onto one storage and FIRES."""
        fqns = [f"m.layers.0.experts.{i}.{p}.weight"
                for p in ("linear_fc1", "linear_fc2") for i in range(8)]
        out = self._run_control(self._probe_ctx(fqns))
        assert out["status"] == "fired", f"{out!r}"
        assert out["confounded"] is False
        assert out["aliased"] == 4

    def test_probe_alias_control_still_fires_on_local_suffix_spelling(self):
        """[PASSES-BEFORE and PASSES-AFTER -- invariance fence] The incident
        spelling (...linear_fc1.weight0..7) was the ONLY one the old loop
        knew; it must fire identically after the splitter adoption. Red-maker:
        any regrouping that narrows (not widens) the eligible population goes
        red here -- this is also the only local net for the
        not-provided-here tests/test_hunt_finding_repairs.py contract."""
        fqns = [f"m.layers.0.mlp.experts.experts.linear_fc{p}.weight{i}"
                for p in (1, 2) for i in range(8)]
        out = self._run_control(self._probe_ctx(fqns))
        assert out["status"] == "fired", f"{out!r}"
        assert out["confounded"] is False
        assert out["aliased"] == 4


class TestRouterControlDivergence:
    """Doctrine 3 for Edit 4: the routed-in-then-skipped tripwire. Real
    router (unmonkeypatched classifier), stubbed probe verdict -- the
    control half of the pair varies the RETURNED status, so the router finds
    an honest shard group and only the guard's reaction is under test."""

    def _sharded_router_ctx(self):
        tms = tuple(lsg.TensorMeta(fqn=f"m.layers.0.experts.{i}.w.weight",
                                   shape=(4, 4), dtype="float32",
                                   storage_id=f"s{i}", kind="tensor")
                    for i in range(8))
        return lsg.CheckpointGateContext(
            tensors=tms, declared_fqns=tuple(t.fqn for t in tms),
            num_experts=8, num_moe_layers=1,
            expected_expert_bytes=None, origin="test://synthetic")

    def test_routed_in_then_skipped_rewrites_to_unconstructable(self, monkeypatch):
        """[FAILS-BEFORE -- Edit 4] Pre-patch the probe's 'skipped' passed
        through untouched and the consume loop filed it as recorded-only --
        the Defect-A fall-through replayed as a unit. Post-patch: blocking
        'unconstructable', probe's own status preserved, original reason
        embedded (auditable divergence, not a laundered one)."""
        monkeypatch.setattr(
            lsg, "_probe_alias_control",
            lambda ctx, n, baseline=None: {
                "status": "skipped",
                "reason": "synthesized: control declines after routing"})
        out = lsg.control_alias(self._sharded_router_ctx(), {})
        assert out["status"] == "unconstructable"
        assert out["probe_status"] == "skipped"
        assert "classifier divergence" in out["reason"]
        assert "synthesized: control declines after routing" in out["reason"]

    def test_genuine_probe_answer_passes_through_unrewritten(self, monkeypatch):
        """[PASSES-BEFORE and PASSES-AFTER -- over-fire fence] Any status
        OTHER than 'skipped' must cross the guard byte-identical. Red-maker:
        if the guard's condition ever broadens (e.g. `!= 'fired'`), fires
        minted by the probe would be re-labeled -- the symmetric doctrine-5
        defect this fence exists to accuse."""
        monkeypatch.setattr(
            lsg, "_probe_alias_control",
            lambda ctx, n, baseline=None: {
                "status": "fired", "confounded": False, "verdict": "FAIL",
                "detail": "synthesized fire", "baseline_verdict": "PASS",
                "aliased_fqns": [], "inconclusive_reason": ""})
        out = lsg.control_alias(self._sharded_router_ctx(), {})
        assert out["status"] == "fired"
        assert out["control"] == "alias(sharded, probe-verbatim)"
        assert "probe_status" not in out


class TestUnderfillAndDropBuilders:
    def test_underfill_inapplicable_without_declared_experts(self, monkeypatch):
        """[PASSES-BEFORE] Red if: the `if not ctx.num_experts` guard is reordered
        below the candidate split, changing the recorded reason class."""
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: [])
        monkeypatch.setattr(lsg, "_split_expert_layouts", lambda c: ({}, [], []))
        ctx = lsg.CheckpointGateContext(
            tensors=(), declared_fqns=None, num_experts=0, num_moe_layers=0,
            expected_expert_bytes=None, origin="test://synthetic")
        assert lsg.control_underfill(ctx, {})["status"] == "inapplicable"

    def test_underfill_unconstructable_below_eight_experts(self, monkeypatch):
        """[PASSES-BEFORE] The incident ratio cannot be reproduced below 8
        without degenerating to zero. Red if: the `< 8` guard is deleted --
        the control would then inject a same-shaped tensor and read not_fired."""
        tm = lsg.TensorMeta(fqn="L0.moe.experts.weight", shape=(4, 4),
                            dtype="float32", storage_id="s", kind="tensor")
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: [tm])
        monkeypatch.setattr(lsg, "_split_expert_layouts", lambda c: ({}, [tm], []))
        ctx = lsg.CheckpointGateContext(
            tensors=(tm,), declared_fqns=(tm.fqn,), num_experts=4, num_moe_layers=1,
            expected_expert_bytes=None, origin="test://synthetic")
        assert lsg.control_underfill(ctx, {})["status"] == "unconstructable"

    def test_underfill_error_is_not_credited(self, monkeypatch):
        """[FAILS-BEFORE -- Edits 1/3] Same FAIL-only rule as the alias leg."""
        tm = lsg.TensorMeta(fqn="L0.moe.experts.weight", shape=(16, 4),
                            dtype="float32", storage_id="s", kind="tensor")
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: [tm])
        monkeypatch.setattr(lsg, "_split_expert_layouts", lambda c: ({}, [tm], []))
        monkeypatch.setattr(lsg.ExpertDistinctnessGate, "run",
                            lambda self, c: _gr(Verdict.ERROR, self.id))
        ctx = lsg.CheckpointGateContext(
            tensors=(tm,), declared_fqns=(tm.fqn,), num_experts=16, num_moe_layers=1,
            expected_expert_bytes=None, origin="test://synthetic")
        out = lsg.control_underfill(ctx, {lsg.ExpertDistinctnessGate.id:
                                    _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)})
        assert out["status"] == "inconclusive" and out["verdict"] == "ERROR"

    def test_drop_unconstructable_without_declared_set(self):
        """[PASSES-BEFORE] Red if: the `if not ctx.declared_fqns` guard is
        deleted (the unexercised-detector reason would change class)."""
        ctx = lsg.CheckpointGateContext(
            tensors=(), declared_fqns=None, num_experts=0, num_moe_layers=0,
            expected_expert_bytes=None, origin="test://synthetic")
        out = lsg.control_drop(ctx, {})
        assert out["status"] == "unconstructable"

    def test_drop_fires_only_when_the_gate_names_a_dropped_fqn(self, monkeypatch):
        """[FAILS-BEFORE -- arity] The self-attribution contract: crediting
        requires the rerun to NAME an injected loss. The honest fake computes
        missing = declared - present, like the real gate is documented to do."""
        def honest(self, c):
            present = {t.fqn for t in c.tensors}
            missing = sorted(set(c.declared_fqns or ()) - present)
            if missing:
                return GateResult(gate_id=self.id, verdict=Verdict.FAIL,
                                  coverage=Coverage(len(present), "tensors",
                                                    expected=len(c.declared_fqns or ())),
                                  detail="missing", evidence={"missing": missing})
            return _gr(Verdict.PASS, self.id)

        monkeypatch.setattr(lsg.SaveCompletenessGate, "run", honest)
        tms = tuple(lsg.TensorMeta(fqn=f"m.{i}.weight", shape=(2, 2), dtype="float32",
                                   storage_id=f"s{i}", kind="tensor") for i in range(4))
        declared = tuple(t.fqn for t in tms)
        ctx = lsg.CheckpointGateContext(
            tensors=tms, declared_fqns=declared, num_experts=0, num_moe_layers=0,
            expected_expert_bytes=None, origin="test://synthetic")
        out = lsg.control_drop(ctx, {}, n=2)
        assert out["status"] == "fired"
        assert out["named_dropped"] and set(out["named_dropped"]) <= set(out["dropped"])

    def test_drop_receives_no_credit_for_a_block_naming_no_injected_loss(
        self, monkeypatch
    ):
        """[RED-UNDER-MUTANT adjudication/drop-control-credits-without-naming;
        GREEN-ON-SHIPPED] MUST_FIRE for the self-attribution line
        (`fired = res.blocking and bool(named)`).

        Discriminating fixture state, per the fix36 note's own warning: the
        rerun must block for a PRE-EXISTING, UNRELATED reason. This artifact
        declares five FQNs and holds four -- ghost.preexisting.weight was
        never written at all, so the deployed-shape gate FAILs on it whether
        or not any control runs. The fake reruns the gate on the INJECTED
        copy and answers FAIL with evidence naming the pre-existing hole and
        NOTHING the control dropped. Anchor: named = dropped ∩ missing = ∅ ->
        not_fired. Mutant (`fired = res.blocking`): the same record is
        credited "fired" -- a block caused by a defect that long predates the
        injection laundering itself into a proven detection of the injection.

        Denominator (doctrine 2): 4 present of 5 declared; 2 of 4 dropped.
        Written against a healthy artifact both readings agree and the mutant
        lives -- which is exactly the fixture shape this suite lacked: the
        named-loss fence (test_drop_fires_only_when_the_gate_names_a_dropped_
        fqn, GREEN on both trees by construction) credits under both
        readings, and the quiet-detector test's PASS fake credits under
        neither. Gate monkeypatched per the in-repo hunt-file precedent;
        readers untouched."""
        present = tuple(
            lsg.TensorMeta(fqn=f"m.{i}.weight", shape=(2, 2), dtype="float32",
                           storage_id=f"s{i}", kind="tensor")
            for i in range(4))
        ghost = "ghost.preexisting.weight"  # declared, never written: the baseline hole
        ctx = lsg.CheckpointGateContext(
            tensors=present,
            declared_fqns=tuple(t.fqn for t in present) + (ghost,),
            num_experts=0, num_moe_layers=0,
            expected_expert_bytes=None, origin="test://synthetic")

        def blocks_for_the_preexisting_hole(self, c):
            # The deployed gate's documented evidence shape (an enumerated
            # missing list, as in the MUST_PASS fence two doors up) with the
            # one difference under test: the named loss is NOT injected --
            # this block stood before the drop ran.
            return GateResult(
                gate_id=self.id, verdict=Verdict.FAIL,
                coverage=Coverage(len(c.tensors), "tensors",
                                  expected=len(c.declared_fqns or ())),
                detail="declared tensor never written: ghost.preexisting.weight",
                evidence={"missing": [ghost]})

        monkeypatch.setattr(lsg.SaveCompletenessGate, "run",
                            blocks_for_the_preexisting_hole)
        out = lsg.control_drop(ctx, {}, n=2)
        assert out["status"] == "not_fired"
        assert out["verdict"] == "FAIL"
        assert out["named_dropped"] == []
        assert set(out["dropped"]).isdisjoint({ghost})

    def test_drop_receives_no_credit_for_an_error_verdict(self, monkeypatch):
        """[RED-UNDER-MUTANT adjudication/drop-control-credits-without-naming;
        GREEN-ON-SHIPPED] The second illegitimate-credit shape the builder's
        own docstring names ("an ERROR/VACUOUS answer carries no 'missing'
        evidence at all"): the detector CRASHES on the injected copy.
        res.blocking is True (ERROR is blocking by contract), so the mutant
        credits the crash as a fire; the shipped line demands a named dropped
        FQN, an ERROR carries no evidence list, named is empty, and the
        honest word is not_fired. Crediting a crash as detection is the
        verifier-exception fallacy this suite already convicted in the
        alias/underfill legs (_attributed_status); the drop leg is the same
        doctrine one door down. Denominator as in the twin above: 4 present
        of 4 declared, 2 dropped, and the rerun examined nothing it reported."""
        tms = tuple(lsg.TensorMeta(fqn=f"m.{i}.weight", shape=(2, 2),
                                   dtype="float32", storage_id=f"s{i}",
                                   kind="tensor") for i in range(4))
        ctx = lsg.CheckpointGateContext(
            tensors=tms, declared_fqns=tuple(t.fqn for t in tms),
            num_experts=0, num_moe_layers=0,
            expected_expert_bytes=None, origin="test://synthetic")
        monkeypatch.setattr(
            lsg.SaveCompletenessGate, "run",
            lambda self, c: _gr(Verdict.ERROR, self.id,
                                "RuntimeError: synthesized crash on injection"))
        out = lsg.control_drop(ctx, {}, n=2)
        assert out["status"] == "not_fired"
        assert out["verdict"] == "ERROR"


# ---------------------------------------------------------------------------
# Mypy batch groups (1) and (2): the probe import fallback's degradation
# path (now type-level true, runtime-identical) and the underfill control's
# byte-priced victim selection (the one explicit runtime behaviour change).
# ---------------------------------------------------------------------------


class TestProbeImportDegradation:
    """Group (1) controls. Runtime semantics were deliberately kept
    byte-for-byte (an honest Optional declaration cannot change them), so
    the runtime tests are labeled as invariance fences per this file's
    convention -- no false failure is minted to satisfy a label
    (doctrine 5 is symmetric). The FAILS-BEFORE carrier for the group is
    the type-level test below: pre-patch, mypy proved the degrade guard
    unreachable behind two type-ignore[assignment]s, and that fact is read
    here from the module's own __annotations__."""

    def _sharded_router_ctx(self):
        """The classifier geometry TestRouterControlDivergence already proves
        routes sharded under the REAL _split_expert_layouts on this tree."""
        tms = tuple(
            lsg.TensorMeta(fqn=f"m.layers.0.experts.{i}.w.weight",
                           shape=(4, 4), dtype="float32", storage_id=f"s{i}",
                           kind="tensor")
            for i in range(8))
        return lsg.CheckpointGateContext(
            tensors=tms, declared_fqns=tuple(t.fqn for t in tms),
            num_experts=8, num_moe_layers=1,
            expected_expert_bytes=None, origin="test://synthetic")

    def test_probe_helper_slots_are_optional_at_type_level(self):
        """[FAILS-BEFORE] The fail-before carrier for group (1). Pre-patch
        the two helper names are bound by imports and by ignored
        assignments and carry NO module-level annotation at all, so the
        __annotations__ lookups below raise KeyError -- red -- which is
        exactly what mypy could not see: it proved the `is None` guard
        unreachable behind the ignored assignments. Post-patch both slots
        are declared Optional before the try, so the fail-closed degrade
        path is visible to the type system, read here from the module's
        own PEP-563 annotation strings."""
        anns = lsg.__annotations__
        assert "None" in anns["_probe_derive_declared"]
        assert "None" in anns["_probe_alias_control"]

    def test_null_probe_alias_control_degrades_to_unconstructable(self, monkeypatch):
        """[PASSES-BEFORE and PASSES-AFTER -- invariance fence, labeled per
        this file's convention: the degrade path was runtime-live on the
        old tree too; the patch made it type-level live, and minting a
        fake fail-before here would be the doctrine-5 symmetric defect.
        Red-maker: deletion or relaxation of the
        `if _probe_alias_control is None:` guard in control_alias's
        sharded leg -- the statement mypy called unreachable.]"""
        monkeypatch.setattr(lsg, "_probe_alias_control", None)
        out = lsg.control_alias(self._sharded_router_ctx(), {})
        assert out["status"] == "unconstructable"
        assert "probe alias control unimportable" in out["reason"]

    def test_null_probe_derive_helper_refuses_to_paraphrase(self, tmp_path,
                                                            monkeypatch):
        """[PASSES-BEFORE and PASSES-AFTER -- invariance fence] The other
        consumer of the same slots refuses to re-derive by paraphrase.
        Red-maker: deletion of the None guard at the top of
        derive_declared_block."""
        monkeypatch.setattr(lsg, "_probe_derive_declared", None)
        spec = lsg.resolve_train_spec({}, "test://cfg", "full", None)
        base = lsg.BaseModel(
            model_dir=tmp_path, config=DENSE_CFG,
            tensors={"x.weight": ((2, 2), "float32")},
            tensors_source="test://synthetic")
        with pytest.raises(lsg.GateUnmeasured, match="paraphrase"):
            lsg.derive_declared_block(base, spec, set(), "")

    def test_normal_import_paths_leave_the_real_probe_control_wired(self):
        """[PASSES-BEFORE and PASSES-AFTER -- MUST_PASS fence for group (1)]
        Both import paths still land on the probe's REAL helpers after the
        restructure, and the real sharded alias control still runs through
        the rewired Optional slots to 'fired' on the global-spelling
        geometry that TestProbeAliasSpellings already proves green against
        the live gate and probe. Red-makers: any edit that empties BOTH
        import paths (the slots go None and the first asserts die), or a
        rewiring that stops routing sharded work to the probe-verbatim
        control (the last two asserts)."""
        assert lsg._probe_derive_declared is not None
        assert lsg._probe_alias_control is not None
        assert lsg._PROBE_IMPORT_ERROR is None
        fqns = [f"m.layers.0.experts.{i}.{p}.weight"
                for p in ("linear_fc1", "linear_fc2") for i in range(8)]
        tms = tuple(lsg.TensorMeta(fqn=f, shape=(4, 4), dtype="float32",
                                   storage_id=f"store://{f}", kind="tensor")
                    for f in fqns)
        ctx = lsg.CheckpointGateContext(
            tensors=tms, declared_fqns=tuple(fqns),
            num_experts=8, num_moe_layers=1,
            expected_expert_bytes=None, origin="test://synthetic")
        baseline = lsg.ExpertDistinctnessGate().run(ctx)
        assert not baseline.blocking, (
            f"fixture drifted: 16-of-16 healthy experts must PASS "
            f"pre-injection ({baseline.detail}) -- calibrate the fixture, "
            f"never the assertion")
        out = lsg.control_alias(ctx, {lsg.ExpertDistinctnessGate.id: baseline})
        assert out["status"] == "fired", f"{out!r}"
        assert out["control"] == "alias(sharded, probe-verbatim)"


class TestUnderfillVictimBytePricing:
    """Group (2) controls. implied_nbytes is int | None, and an unpriced
    tensor cannot be the victim of a MEASURED underfill. What specifically
    makes the REAL property return None could not be established from the
    handed-over sources, so the tests force None by a class-level property
    patch and the docstrings say so -- a stated abstention, not a skip."""

    def _tm(self, fqn, shape=(16, 4)):
        return lsg.TensorMeta(fqn=fqn, shape=shape, dtype="float32",
                              storage_id=f"store://{fqn}", kind="tensor")

    def _ctx(self, tms):
        return lsg.CheckpointGateContext(
            tensors=tuple(tms), declared_fqns=tuple(t.fqn for t in tms),
            num_experts=16, num_moe_layers=1,
            expected_expert_bytes=None, origin="test://synthetic")

    def _route_stacked(self, monkeypatch, tms):
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: list(tms))
        monkeypatch.setattr(lsg, "_split_expert_layouts",
                            lambda c: ({}, list(c), []))

    def test_all_candidates_unpriced_blocks_and_names_zero_of_n(self, monkeypatch):
        """[FAILS-BEFORE] MUST_FIRE for group (2). On the current tree
        max(..., key=implied_nbytes) over all-None prices raises TypeError
        inside the control, so this test errors red -- which IS the
        production behaviour being retired (launchers saw exit 3, 'a tool
        bug', never a verdict). Post-patch: blocking 'unconstructable'
        NAMING 0 of 2 (doctrine 1). The class-level property patch forces
        None whether implied_nbytes is a class property or an
        instance-stored value (a data descriptor on the class shadows any
        instance attribute); raising=False covers the name living only on
        instances."""
        tms = [self._tm("L0.moe.experts.weight"),
               self._tm("L1.moe.experts.weight")]
        self._route_stacked(monkeypatch, tms)
        monkeypatch.setattr(lsg.TensorMeta, "implied_nbytes",
                            property(lambda self: None), raising=False)
        out = lsg.control_underfill(self._ctx(tms), {})
        assert out["status"] == "unconstructable"
        assert "0 of 2" in out["reason"]

    def test_unpriced_candidates_are_excluded_not_zero_priced(self, monkeypatch):
        """[FAILS-BEFORE] MUST_PASS for group (2): exclusion semantics. The
        BIGGER-by-shape tensor is the one without a price; it must NOT win
        the victim slot (excluded -- and never priced as zero), the priced
        tensor must be selected and fire against a clean baseline, and the
        record must disclose the partial sweep's denominator (doctrine 2).
        Pre-patch the mixed None/int key comparison TypeErrors -- red.
        This is the argued repair's one deliberate behaviour change, made
        load-bearing: it is loud in the emitted record, not silent."""
        tms = [self._tm("L0.moe.experts.weight", shape=(16, 1024)),
               self._tm("L1.moe.experts.weight")]
        self._route_stacked(monkeypatch, tms)
        monkeypatch.setattr(
            lsg.TensorMeta, "implied_nbytes",
            property(lambda self: None if self.fqn.startswith("L0") else 64),
            raising=False)
        monkeypatch.setattr(
            lsg.ExpertDistinctnessGate, "run",
            lambda self, c: _gr(Verdict.FAIL, self.id, "synthesized fire"))
        baseline = {lsg.ExpertDistinctnessGate.id:
                    _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)}
        out = lsg.control_underfill(self._ctx(tms), baseline)
        assert out["status"] == "fired"
        assert out["tensor"] == tms[1].fqn
        assert out["candidates"].startswith("1 of 2")
        assert "64 bytes" in out["candidates"]


# ---------------------------------------------------------------------------
# MoE integration: the headline FAILS-BEFORE (D1 end to end) + confounding
# ---------------------------------------------------------------------------


class TestMoeIntegration:
    def _moe(self, tmp_path):
        _probe_declared_or_calibrate(MOE_CFG, 8, 2)
        base = _make_base(tmp_path, _moe_full_tensors(), MOE_CFG, name="moe-base")
        ckpt = _materialize_artifact(tmp_path, _moe_full_tensors(), name="moe")
        return base, ckpt

    def test_healthy_sharded_moe_is_clear_and_alias_control_FIRES(self, tmp_path):
        """[FAILS-BEFORE -- Edits 2/4] THE headline: on the current tree the
        sharded-leg probe control is invoked with baseline=None, the repaired
        probe answers "inconclusive", the loop falls through, drop satisfies
        the floor, and the tool prints CLEAR with its load-bearing aliasing
        detector never creditably exercised. After the patch: status 'fired',
        confounded False, and CLEAR means what it says.

        Calibration-loud: if this is BLOCKED on the CURRENT tree with a
        distinctness/byte gate reason, the fixture's metadata (storage ids,
        shape table) disagrees with the live gates -- the failure text below
        carries the gate verdict verbatim for the calibrator."""
        base, ckpt = self._moe(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        # 2 q_proj + 8 experts x 2 layers x 2 projections = 34; see the fixture
        # arithmetic block. Kept as a literal so a silent change to the fixture
        # population shows up here instead of being absorbed by a recomputation.
        assert d.report["inventory"]["real_tensors"] == 34
        alias = _control_by_prefix(d, "alias")
        assert alias["status"] == "fired", f"alias control: {alias!r}"
        assert alias["confounded"] is False

    def test_confounded_alias_is_inconclusive_and_blocks(self, tmp_path, monkeypatch):
        """[FAILS-BEFORE -- Edits 2/4] Baseline already blocks: the injected
        aliasing cannot be attributed. The current tree rewrites only the
        'confounded' flag post-hoc and lets status fall through; after the
        patch the record says inconclusive/confounded=True and the run carries
        an INCONCLUSIVE blocking reason (alongside the real gate's own). Gate
        faked per the hunt-file precedent; readers untouched."""
        base, ckpt = self._moe(tmp_path)
        d0 = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d0.exit_code == 0, (  # fixture-drift guard, hunt-file style
            f"fixture drifted: healthy MoE must be CLEAR pre-patch-state "
            f"({d0.blocking_reasons}); calibrate the fixture, never the assertion")
        monkeypatch.setattr(lsg.ExpertDistinctnessGate, "run",
                            lambda self, c: _gr(Verdict.FAIL, self.id, "pre-existing"))
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 1
        alias = _control_by_prefix(d, "alias")
        assert alias["status"] == "inconclusive", f"{alias!r}"
        assert alias["confounded"] is True
        assert any("INCONCLUSIVE" in r for r in d.blocking_reasons)

    def test_detector_crash_on_injection_is_inconclusive_not_a_fire(self, tmp_path, monkeypatch):
        """[FAILS-BEFORE -- Edits 2/4] Baseline clean; the detector then
        crashes on the injected copy. Crediting that as detection is the
        verifier-exception fallacy D2 names. First run() call is the real
        baseline; subsequent calls are control injections."""
        base, ckpt = self._moe(tmp_path)
        real_run = lsg.ExpertDistinctnessGate.run
        calls = {"n": 0}

        def crash_after_baseline(self, c):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_run(self, c)
            return _gr(Verdict.ERROR, self.id, "RuntimeError: synthesized")

        monkeypatch.setattr(lsg.ExpertDistinctnessGate, "run", crash_after_baseline)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 1
        alias = _control_by_prefix(d, "alias")
        assert alias["status"] == "inconclusive"
        assert alias["confounded"] is False  # clean baseline; the malfunction is the story
        assert any("INCONCLUSIVE" in r for r in d.blocking_reasons)


# ---------------------------------------------------------------------------
# Exit codes: 0, 1, 3 -- exact integers, because the retry policy turns on them
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_cli_clear_is_exactly_zero_and_report_roundtrips(self, tmp_path, capsys):
        """[PASSES-BEFORE] Red if: main() returns d.exit_code is replaced by
        a truthiness collapse (`return 0 if d.ok else 1` would still pass here
        but break nothing else -- the REAL red-maker: assert instead paired
        with test_cli_missing_* pinning 3 distinctly, which that collapse
        breaks). Report carries denominators on the wire.
        fix45: the CLI lora path now demands --adapter-modules (#78); the
        argv below carries the honest census -- the 12 artifact-namespace
        module stems implied by the run's declared structure (6 dense
        layers x 2 targets), written outside the judged tree. This also
        keeps the new flag's own plumbing pinned: drop the pass-through in
        main() and this test dies on the refusal exit, green-laundering
        impossible. Assertions unchanged."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        out = tmp_path / "report.json"
        census = _census_file(
            tmp_path,
            [f"layers.{i}.self_attn.{w}"
             for i in range(6) for w in ("q_proj", "v_proj")],
        )
        code = lsg_cli.main([str(ckpt), "--run-kind", "lora",
                         "--base-model-dir", str(base),
                         "--train-config", str(cfg),
                         "--adapter-prefix", "",
                         "--adapter-modules", str(census),
                         "--json", str(out)])
        assert code == 0
        rpt = json.loads(out.read_text(encoding="utf-8"))
        assert rpt["exit_code"] == 0
        # #80: report inventory = physical artifact (31 real: 24 adapter +
        # 6 optimizer.* + 1 rng_state); the judged adapter population
        # stays 24; exit-code discrimination (0 CLEAR / 1 blocked /
        # 3 unmeasured) is untouched.
        assert rpt["inventory"]["real_tensors"] == 31
        assert rpt["inventory"]["base_tensors"] == 12
        assert len(rpt["controls"]) == 3
        assert rpt["declared_basis"]["run_kind"]

    def test_cli_blocked_is_exactly_one(self, tmp_path):
        """[PASSES-BEFORE] Red if: blocking-reason assembly is bypassed
        (`exit_code = EXIT_CLEAR if not blocking else ...` style rewrite that
        forgets cross-check reasons)."""
        base, _ = _dense_base_with_ckpt(tmp_path)
        truncated = dict(list(_dense_full_tensors().items())[:4])
        ckpt = _materialize_artifact(tmp_path, truncated, name="trunc")
        code = lsg_cli.main([str(ckpt), "--run-kind", "full",
                         "--base-model-dir", str(base),
                         "--train-config", str(_write_cfg(tmp_path, {}))])
        assert code == 1

    def test_cli_unmeasured_is_exactly_three_for_missing_checkpoint(self, tmp_path):
        """[PASSES-BEFORE] 3, not 1: the launcher retries measurement
        failures, it does not treat them as verdicts. Red if: the
        `except GateUnmeasured` mapping to EXIT_UNMEASURED is changed to 1."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        code = lsg_cli.main([str(tmp_path / "no-such-ckpt"), "--run-kind", "full",
                         "--base-model-dir", str(base),
                         "--train-config", str(_write_cfg(tmp_path, {}))])
        assert code == 3

    def test_cli_unmeasured_three_for_missing_base(self, tmp_path):
        """[PASSES-BEFORE] Independent source A absent = cannot measure. Red
        if: BaseModel.load's missing-dir raise is weakened to an empty model."""
        ckpt = _materialize_artifact(tmp_path, _dense_full_tensors())
        code = lsg_cli.main([str(ckpt), "--run-kind", "full",
                         "--base-model-dir", str(tmp_path / "no-base"),
                         "--train-config", str(_write_cfg(tmp_path, {}))])
        assert code == 3

    def test_cli_unmeasured_three_for_missing_train_config_path(self, tmp_path):
        """[PASSES-BEFORE] The asymmetry S2 names: ABSENT flag is tolerated,
        a SUPPLIED path that does not exist is a measurement failure. Red if:
        _load_train_config's is_file() guard is deleted."""
        base, ckpt = _dense_base_with_ckpt(tmp_path)
        code = lsg_cli.main([str(ckpt), "--run-kind", "full",
                         "--base-model-dir", str(base),
                         "--train-config", str(tmp_path / "ghost.json")])
        assert code == 3

    def test_cli_unmeasured_three_for_unparseable_rank(self, tmp_path):
        """[PASSES-BEFORE] A rank value the tool cannot read must not be
        coerced to a default. Red if: the int() coercion grows a try/except-
        pass (i.e., the existing GateUnmeasured raise is what saves it)."""
        base, ckpt, _cfg = _healthy_lora(tmp_path)
        bad = _write_cfg(tmp_path, {"peft_scheme": "lora", "lora_rank": "eight"},
                         name="bad-rank.json")
        code = lsg_cli.main([str(ckpt), "--run-kind", "lora",
                         "--base-model-dir", str(base), "--train-config", str(bad)])
        assert code == 3

    def test_cli_unmeasured_three_for_unreadable_artifact(self, tmp_path):
        """[PASSES-BEFORE] Garbage where the checkpoint should be is a
        measurement failure, never a verdict. Red if: _measure's
        GateUnmeasured conversion is narrowed to swallow and return None."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        junk = tmp_path / "junk-ckpt"
        junk.mkdir()
        (junk / "model.safetensors").write_bytes(b"not a safetensors file")
        code = lsg_cli.main([str(junk), "--run-kind", "full",
                         "--base-model-dir", str(base),
                         "--train-config", str(_write_cfg(tmp_path, {}))])
        assert code == 3

    def test_cli_tool_bug_is_three_not_a_verdict(self, tmp_path, monkeypatch):
        """[PASSES-BEFORE] An unexpected exception inside adjudication is
        exit 3 with the 'a tool bug is not a checkpoint verdict' framing. Red
        if: the broad `except Exception` mapping is deleted (the traceback
        would then escape main and the process would exit 1 via the
        interpreter -- a verdict-shaped accident)."""
        monkeypatch.setattr(lsg_cli, "adjudicate_checkpoint",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        base, ckpt, cfg = _healthy_lora(tmp_path)
        code = lsg_cli.main([str(ckpt), "--run-kind", "lora",
                         "--base-model-dir", str(base), "--train-config", str(cfg)])
        assert code == 3


# ---------------------------------------------------------------------------
# Reader / config units (no artifact needed; the base writer is shared above)
# ---------------------------------------------------------------------------


class TestBaseModelReader:
    def test_sharded_base_index_is_honored(self, tmp_path):
        """[PASSES-BEFORE] Red if: the idx.is_file() branch is inverted."""
        tensors = _dense_full_tensors()
        items = sorted(tensors.items())
        shards = [dict(items[:6]), dict(items[6:])]
        base = tmp_path / "sharded-base"
        base.mkdir()
        weight_map = {}
        for i, shard in enumerate(shards):
            name = f"model-{i + 1:05d}-of-00002.safetensors"
            _write_safetensors(base / name, shard)
            for fqn in shard:
                weight_map[fqn] = name
        (base / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": weight_map}))
        (base / "config.json").write_text(json.dumps(DENSE_CFG))
        loaded = lsg.BaseModel.load(base)
        assert len(loaded.tensors) == 12
        assert "2 shards" in loaded.tensors_source

    def test_empty_weight_map_is_unmeasured(self, tmp_path):
        """[PASSES-BEFORE] Red if: the `if not weight_map` guard is deleted."""
        base = tmp_path / "b"
        base.mkdir()
        (base / "config.json").write_text("{}")
        (base / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}}))
        with pytest.raises(lsg.GateUnmeasured, match="empty weight_map"):
            lsg.BaseModel.load(base)

    def test_unknown_dtype_is_unmeasured_not_coerced(self, tmp_path):
        """[PASSES-BEFORE] Byte pricing on a guessed dtype prices wrong; the
        tool refuses. Red if: the `dtype is None` raise becomes a default."""
        path = tmp_path / "odd.safetensors"
        blob = json.dumps({"x.w": {"dtype": "FP8_E4M3", "shape": [2, 2],
                                   "data_offsets": [0, 4]}}).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)
        with pytest.raises(lsg.GateUnmeasured, match="unrecognized safetensors"):
            lsg._read_safetensors_header(path)


class TestTrainSpecResolution:
    def test_key_value_dump_is_accepted(self, tmp_path):
        """[PASSES-BEFORE] env-dump configs are first-class inputs. Red if:
        the KEY=VALUE fallback parser is deleted."""
        p = tmp_path / "resolved.env"
        p.write_text('declare -x PEFT_SCHEME="lora"\ndeclare -x LORA_RANK="8"\n')
        cfg, source = lsg._load_train_config(p)
        assert cfg["PEFT_SCHEME"] == "lora" and "KEY=VALUE" in source

    def test_rank_is_coerced_from_string(self, tmp_path):
        """[PASSES-BEFORE] Red if: the int() coercion is deleted (rank None
        downstream -> lora derivation abstains where it should not)."""
        spec = lsg.resolve_train_spec(
            {"peft_scheme": "lora", "lora_rank": "8"}, "test://cfg", "auto", None)
        assert spec.run_kind == "lora" and spec.lora_rank == 8

    def test_missing_kind_key_defers_with_stated_basis(self, tmp_path):
        """[PASSES-BEFORE] Deferral is a STATED abstention, not a silent full.
        Red if: the kbasis message loses the word 'inferred'."""
        spec = lsg.resolve_train_spec({}, "test://cfg", "auto", None)
        assert spec.run_kind == "auto" and "inferred" in spec.kind_basis


class TestMoeOverride:
    """The Gemma-4 dense declaration, post-bridge (fix25): the override is
    now the probe's two-source mint, never a local one
    (MINT_ZERO_ONLY_IN_PROBE). Doctrine 3 for the bridge: one MUST_PASS of
    the measured estate shape, two MUST_FIREs -- a self-contradicting config,
    and a frozen-regex laundering attempt against a real MoE base."""

    def test_enable_moe_block_false_zeroes_expert_denominator(self, tmp_path):
        """[FAILS-BEFORE on the corroboration and census assertions; the
        first two assertions pass on the current tree -- kept deliberately,
        see below] STRENGTHENED, never weakened. The author's intent, per the
        pre-patch docstring, was that an explicit enable_moe_block=false must
        flip the denominator to zero WITH the flip on the record. The
        pre-patch fixture undercut that intent: it paired the flag with a
        POSITIVE count (num_experts=8), a self-contradicting config whose
        count the single-source override silently overwrote -- and its first
        assertion (`decl.num_experts == 0`) survived the contract change
        without noticing, because the override minted regardless of basis.

        The fixture is now the MEASURED estate shape (gemma-4-E4B-it):
        enable_moe_block is False and num_experts is present-but-null. The
        test proves strictly more than the old one: the mint happens, the
        affirmative statement is quoted in the basis by the probe, the
        corroboration language is present, AND the census denominator
        (0 of 12 base-header names) travels in the notes. Red-makers on the
        current tree: the un-bridged call supplies no census, so the probe's
        basis reads "no artifact census was supplied to corroborate it" --
        containing "corroborate" but never "corroborated"/"two independent
        sources" -- and no census note exists (asserts 4-6 fail). Asserts
        1-2 pass pre-patch via the illegitimate mint and pass post-patch via
        the legitimate one: they are kept as the pin that the mint itself
        still happens."""
        cfg = {"model_type": "calibration-gemma4-dense",
               "text_config": {"num_moe_layers": 2,
                               "enable_moe_block": False,
                               "num_experts": None}}
        spec = lsg.resolve_train_spec({}, "test://cfg", "full", None)
        base = lsg.BaseModel(
            model_dir=tmp_path, config=cfg,
            tensors={k: (v[0], "float32")
                     for k, v in _dense_full_tensors().items()},
            tensors_source="test://synthetic")
        decl = lsg.derive_declared_block(base, spec, set(), "", r"\.(lora_[AB])$")
        assert decl.num_experts == 0
        assert "enable_moe_block=false" in decl.experts_basis
        assert "corroborated" in decl.experts_basis
        assert "two independent sources agree" in decl.experts_basis
        assert any("expert-family census: 0 of 12" in n for n in decl.notes)

    def test_flag_false_beside_a_positive_count_abstains_loudly(self, tmp_path):
        """[FAILS-BEFORE] MUST_FIRE on the OLD test's exact fixture shape,
        preserved rather than discarded: enable_moe_block=false NEXT TO
        num_experts=8 is a self-contradicting config. Pre-patch the override
        adjudicated it FOR the config (num_experts minted to 0, the note
        branch skipped because the probe had abstained) -- precisely the
        mint-without-basis the verbatim fix25 failure reported. Post-patch
        the probe refuses to pick a winner and abstains; UNKNOWN and
        gates-block are named in the basis (doctrine 4)."""
        cfg = {**MOE_CFG,
               "text_config": {"num_experts": 8, "num_moe_layers": 2,
                               "enable_moe_block": False}}
        _probe_declared_or_calibrate(MOE_CFG, 8, 2)
        spec = lsg.resolve_train_spec({}, "test://cfg", "full", None)
        base = lsg.BaseModel(
            model_dir=tmp_path, config=cfg,
            tensors={k: (v[0], "float32")
                     for k, v in _dense_full_tensors().items()},
            tensors_source="test://synthetic")
        decl = lsg.derive_declared_block(base, spec, set(), "", r"\.(lora_[AB])$")
        assert decl.num_experts is None
        assert "contradicts itself" in decl.experts_basis
        assert "UNKNOWN" in decl.experts_basis and "gates block" in decl.experts_basis

    def test_frozen_regex_cannot_launder_an_moe_base_into_dense(self, tmp_path):
        """[FAILS-BEFORE] MUST_FIRE for the census trap (fix25-s4): a config
        that affirmatively -- and WRONGLY -- declares dense against a REAL
        MoE base, plus a --frozen-regex that matches the expert stem and
        empties the in-scope expert set. A census computed over the
        frozen-filtered population would read 0 and corroborate the lie: the
        founding incident wearing a user regex. The census is taken over the
        UNFILTERED base header, finds 32 expert-family names (8 experts x
        2 layers x 2 projections), and the contradiction blocks. Pre-patch
        the override mints 0 despite the experts sitting in the header --
        red on the first assertion."""
        cfg = {"model_type": "calibration-laundering-attempt",
               "text_config": {"enable_moe_block": False, "num_experts": None}}
        spec = lsg.resolve_train_spec({}, "test://cfg", "full", r"\.experts\.")
        base = lsg.BaseModel(
            model_dir=tmp_path, config=cfg,
            tensors={k: (v[0], "float32")
                     for k, v in _moe_full_tensors().items()},
            tensors_source="test://synthetic")
        decl = lsg.derive_declared_block(base, spec, set(), "", r"\.(lora_[AB])$")
        assert decl.num_experts is None
        assert "CONTRADICTION" in decl.experts_basis
        assert "32" in decl.experts_basis


# ---------------------------------------------------------------------------
# fix25: the bridge, end to end. MUST_PASS on the real estate's measured
# shape; its fail-before mechanism is named in the docstring.
# ---------------------------------------------------------------------------


class TestDenseBridgeEndToEnd:
    def test_real_estate_shape_full_run_clears_with_corroborated_zero(self, tmp_path):
        """[FAILS-BEFORE] MUST_PASS of the bridge: text_config.enable_moe_block
        = False, text_config.num_experts present-but-null, zero expert-family
        names in the base header -- a healthy full run must CLEAR with a
        corroborated 0, i.e. the first real run is not blocked by the tool's
        own honesty rule. Mechanism, pinned: pre-patch the probe is called
        WITHOUT the census, an affirmative-but-uncorroborated dense statement
        abstains (num_experts=None), and both expert gates take their
        VACUOUS doors -- exit 1, red on the first assertion. Post-patch the
        0 arrives with both sources cited, both gates abstain as
        machine-readable not_applicable, the alias control is inapplicable
        (recorded, and covered by the drop control per the exit-code
        contract), and every denominator is on the wire."""
        estate_cfg = {"model_type": "calibration-estate-dense",
                      "text_config": {"num_hidden_layers": 6, "hidden_size": 8,
                                      "enable_moe_block": False,
                                      "num_experts": None}}
        base = _make_base(tmp_path, _dense_full_tensors(), estate_cfg)
        ckpt = _materialize_artifact(tmp_path, _dense_full_tensors())
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        assert d.report["inventory"]["real_tensors"] == 12
        assert d.report["inventory"]["base_tensors"] == 12
        assert "enable_moe_block=false" in d.declared_basis["num_experts"]
        assert "corroborated" in d.declared_basis["num_experts"]
        assert any("expert-family census: 0 of 12" in n
                   for n in d.declared_basis["notes"])
        by_gate = {g["gate"]: g for g in d.gate_results}
        assert by_gate["checkpoint.expert_distinctness"]["verdict"] == "SKIP"
        assert by_gate["checkpoint.expert_distinctness"]["abstention"] == "not_applicable"
        assert _control_by_prefix(d, "drop")["status"] == "fired"
        assert _control_by_prefix(d, "alias")["status"] == "inapplicable"


# ---------------------------------------------------------------------------
# LG3: the composite's applicable-denominator pricing, at gate level and tool
# level. Contexts are synthetic; all four sub-gates are REAL (only
# test_all_inapplicable fakes sub-gates, per the in-file monkeypatch
# precedent). No new TOP-LEVEL imports are added so the module still imports
# on the un-patched tree for the tasker's revert-control run.
# ---------------------------------------------------------------------------


def _stacked_moe_full_tensors() -> dict:
    """HF-stacked MoE artifact: 3 dense attention layers + 2 stacked MoE layers.

    10 real tensors (over the MODE/full partial-population threshold of 8),
    stacked expert count 4 == num_moe_layers 2 x family width 2, so neither the
    byte gate's coverage nor the cross-check fires for extraneous reasons: the
    ONLY thing left to block is the composite's not-established leg."""
    t = {f"model.language_model.layers.{ly}.attn.{w}.weight": ((8, 8), "F32")
         for ly in range(3) for w in ("q_proj", "v_proj")}
    for ly in range(2):
        for proj, inner in (("gate_up_proj", (16, 32)), ("down_proj", (32, 16))):
            t[f"model.language_model.layers.{ly}.experts.{proj}"] = ((8, *inner), "BF16")
    return t


class TestFirstSaveCompositeDenominator:
    def _ctx(self, tensors, *, declared, experts, layers, expected_bytes):
        return lsg.CheckpointGateContext(
            tensors=tensors, declared_fqns=declared, num_experts=experts,
            num_moe_layers=layers, expected_expert_bytes=expected_bytes,
            origin="test://synthetic")

    def _dense_ctx(self):
        tms = tuple(
            lsg.TensorMeta(fqn=f"model.layers.{i}.attn.weight", shape=(4, 4),
                           dtype="float32", storage_id=f"sd{i}", kind="tensor")
            for i in range(4))
        return self._ctx(tms, declared=tuple(t.fqn for t in tms),
                         experts=0, layers=0, expected_bytes=0)

    def _stacked_ctx(self):
        tms = tuple(
            lsg.TensorMeta(
                fqn=f"model.language_model.layers.{ly}.experts.{proj}",
                shape=(8, *inner), dtype="bfloat16",
                storage_id=f"st-{ly}-{proj}", kind="tensor")
            for ly in range(2)
            for proj, inner in (("gate_up_proj", (16, 32)), ("down_proj", (32, 16))))
        return self._ctx(
            tms, declared=tuple(t.fqn for t in tms), experts=8, layers=2,
            expected_bytes=sum(t.implied_nbytes for t in tms))

    def test_dense_shrinks_denominator_and_names_inapplicable(self):
        """[FAILS-BEFORE] The gate-level core of LG3: positively declared dense
        must PASS at exactly 1/1 applicable, with both inapplicable gates named
        and the kind carried as data. Pre-patch this goes UNDERCOVERED (1/3)."""
        result = lsg.FirstSaveGate().run(self._dense_ctx())
        assert result.verdict is Verdict.PASS, f"{result.verdict}: {result.detail}"
        assert result.coverage.checked == 1 and result.coverage.expected == 1
        assert "1/1 applicable" in result.detail
        assert "checkpoint.expert_distinctness" in result.detail
        assert "checkpoint.expert_bytes" in result.detail
        assert "3/3" not in result.detail
        assert set(result.evidence["inapplicable"]) == {
            "checkpoint.expert_distinctness", "checkpoint.expert_bytes"}
        expert_result = lsg.ExpertDistinctnessGate().run(self._dense_ctx())
        assert expert_result.verdict is Verdict.SKIP
        assert expert_result.abstention.value == "not_applicable"  # AttributeError pre-patch

    def test_stacked_stays_two_thirds_and_is_not_established(self):
        """[FAILS-BEFORE on the abstention-kind line ONLY; the verdict and 2/3
        pricing lines are PASSES-BEFORE fences, declared per house rule] The
        tasker's discriminating case: NOT_ESTABLISHED must NOT shrink. Verdict
        stays UNDERCOVERED at exactly 2/3 on both trees; the new pin is that
        the distinctness SKIP is machine-readably 'not_established'."""
        result = lsg.FirstSaveGate().run(self._stacked_ctx())
        assert result.verdict is Verdict.UNDERCOVERED       # fence (both trees)
        assert result.coverage.checked == 2                 # fence
        assert result.coverage.expected == 3                # fence
        assert "not established" in result.detail           # fence
        distinct = lsg.ExpertDistinctnessGate().run(self._stacked_ctx())
        assert distinct.verdict is Verdict.SKIP             # fence
        assert distinct.abstention.value == "not_established"  # FAILS-BEFORE (AttributeError)

    def test_unknown_provenance_still_blocks_closed(self):
        """[FAILS-BEFORE on the abstention-is-None lines; the blocking verdict
        is a PASSES-BEFORE fence] No declaration at all -- the shape a
        configless DCP artifact produces through any path that cannot source
        denominators -- must NOT be auto-classified dense: both expert gates
        take the VACUOUS door, the composite FAILS, and no shrink-capable
        abstention kind is minted anywhere on the path."""
        tms = tuple(
            lsg.TensorMeta(fqn=f"model.layers.{i}.attn.weight", shape=(4, 4),
                           dtype="float32", storage_id=f"sm{i}", kind="tensor")
            for i in range(2))
        ctx = self._ctx(tms, declared=None, experts=None, layers=None,
                        expected_bytes=None)
        result = lsg.FirstSaveGate().run(ctx)
        assert result.verdict is Verdict.FAIL               # fence
        for gate in (lsg.ExpertDistinctnessGate, lsg.ExpertByteVolumeGate):
            res = gate().run(ctx)
            assert res.verdict is Verdict.VACUOUS           # fence
            assert res.abstention is None                   # FAILS-BEFORE (AttributeError)

    def test_declared_experts_absent_never_shrinks(self):
        """[FAILS-BEFORE on the abstention-is-None lines; verdicts fence] The
        most dangerous failure mode of denominator-shrinking (the all([])
        shape itself): experts DECLARED and ABSENT is VACUOUS-blocking, not
        inapplicable. If this path ever minted NOT_APPLICABLE, the composite
        would verify 1/1 and pass the incident's twin."""
        tms = tuple(
            lsg.TensorMeta(fqn=f"model.layers.{i}.attn.weight", shape=(4, 4),
                           dtype="float32", storage_id=f"se{i}", kind="tensor")
            for i in range(2))
        ctx = self._ctx(tms, declared=tuple(t.fqn for t in tms),
                        experts=8, layers=2, expected_bytes=1024)
        result = lsg.FirstSaveGate().run(ctx)
        assert result.verdict is Verdict.FAIL               # fence
        distinct = lsg.ExpertDistinctnessGate().run(ctx)
        assert distinct.verdict is Verdict.VACUOUS          # fence
        assert distinct.abstention is None                  # FAILS-BEFORE (AttributeError)
        bytegate = lsg.ExpertByteVolumeGate().run(ctx)
        assert bytegate.verdict is Verdict.VACUOUS          # fence
        assert bytegate.abstention is None                  # FAILS-BEFORE (AttributeError)

    def test_all_inapplicable_is_vacuous_not_a_pass(self, monkeypatch):
        """[FAILS-BEFORE] The zero-applicable corner: if every property were
        somehow declared inapplicable, 0/0 must NOT pass -- Coverage(0, ...,
        expected=0) is vacuous and ok() enforces it. Sub-gates faked via
        type(); monkeypatch restores _subgates. Pre-patch the AbstentionKind
        import inside this test does not exist, which is the fail-before."""
        from foundationscale.gates.core import AbstentionKind, Lifecycle

        def _na_check(self, ctx):
            return self.skip("synthetic: property affirmed absent",
                             kind=AbstentionKind.NOT_APPLICABLE)

        fakes = tuple(
            type(f"_NA{i}", (lsg.Gate,), {
                "id": f"test.synthetic_na_{i}",
                "description": "synthetic NOT_APPLICABLE abstainer",
                "events": (Lifecycle.FIRST_SAVE,),
                "check": _na_check,
                "controls": lambda self: (),
            }) for i in range(3))
        monkeypatch.setattr(lsg.FirstSaveGate, "_subgates", fakes)
        result = lsg.FirstSaveGate().run(self._dense_ctx())  # ctx ignored by fakes
        assert result.verdict is Verdict.VACUOUS
        assert result.coverage.checked == 0 and result.coverage.expected == 0


class TestStackedFirstSaveTool:
    """The discriminating case end-to-end through the tool (LG3 constraint)."""

    def test_stacked_moe_first_save_stays_blocked_at_two_thirds(self, tmp_path):
        """[FAILS-BEFORE on the '2/3 applicable' wording and the abstention
        wire key; the exit code, the UNDERCOVERED verdict, and the 2/3 counts
        are PASSES-BEFORE fences] A genuinely-MoE STACKED first save must
        keep blocking: distinctness is NOT_ESTABLISHED, not inapplicable. A
        lazy 'any SKIP leaves the denominator' rewrite turns this test green
        for the wrong reason -- it exists to kill that rewrite."""
        _probe_declared_or_calibrate(MOE_CFG, 8, 2)
        base = _make_base(tmp_path, _stacked_moe_full_tensors(), MOE_CFG,
                          name="st-base")
        ckpt = _materialize_artifact(tmp_path, _stacked_moe_full_tensors(),
                                     name="st-ckpt")
        d = lsg.adjudicate_checkpoint(
            ckpt, event="first_save", run_kind="full", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}))
        assert d.exit_code == 1                                    # fence (both trees)
        assert len(d.blocking_reasons) == 1, (                     # fence
            f"the composite's not-established leg must be the ONLY reason: "
            f"{d.blocking_reasons}")
        composite = next(g for g in d.gate_results
                         if g["gate"] == "checkpoint.first_save")
        assert composite["verdict"] == "UNDERCOVERED"              # fence
        assert composite["checked"] == 2 and composite["expected"] == 3   # fence
        assert "not established" in str(composite["detail"])       # fence
        # The fail-before lines:
        # pre-patch: "2/3 first-save properties"
        assert "2/3 applicable" in str(composite["detail"])
        by_gate = {g["gate"]: g for g in d.gate_results}
        # pre-patch: no such key
        assert by_gate["checkpoint.expert_distinctness"]["abstention"] == "not_established"


# ---------------------------------------------------------------------------
# Adapter naming: generator/recognizer agreement, literal generator templates,
# and the --adapter-prefix demand (the recognizer/generator defect batch).
# Every detector-shaped addition ships MUST_FIRE + MUST_PASS (doctrine 3);
# fail-before mechanics are named in each docstring.
# ---------------------------------------------------------------------------


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


class TestAdapterNamingAgreement:
    @staticmethod
    def _census_stems():
        """fix45/#78 vehicle stems for this class's lora derive calls.

        The subjects of the three repaired call sites below are the
        generator templates, their shape convention, and the CLEAR-through
        agreement of a correctly calibrated naming -- none of them
        adjudicates census CONTENT. Post-#78 the lora derive refuses
        without an --adapter-modules census, so they supply the honest
        population these fixtures imply: the 12 module stems the fixture
        adapters attach to (the DENSE_CFG 6 layers x the 2 LORA_TRAIN
        targets) in the artifact namespace, computed from the run's
        declared structure rather than read off the artifact. The
        naming-agreement machinery under test never consumes stems, so
        this census cannot launder a naming defect into a green: a
        generator or recognizer regression still dies on the exact
        want-maps and status assertions these tests always had."""
        return [f"layers.{i}.self_attn.{w}"
                for i in range(6) for w in ("q_proj", "v_proj")]

    def test_mismatched_templates_refused_and_pair_named(self):
        """[FAILS-BEFORE -- _verify_adapter_naming_agreement does not exist
        pre-patch -> AttributeError, red] MUST_FIRE for the startup
        cross-check: the defect's exact headline scenario -- the operator
        calibrates the GENERATOR naming to the estate's real export while
        --adapter-suffix still matches only the PEFT defaults. The refusal
        names the disagreeing elements, not just "something is wrong".
        fix34 calibration record: the scenario's CONTENT did not change
        under T2, but the constants that NAME it did. T2 made the estate's
        real export the shipped default, so this test's pre-T2 spelling --
        DEFAULT recognizer against estate-shaped literals -- became a
        self-agreeing triple (that agreement is precisely what T2 shipped),
        the veto could not fire, and pytest.raises reported DID NOT RAISE.
        The scenario is now expressed by naming the retired recognizer
        explicitly (_HF_PEFT_ADAPTER_SUFFIX_RE, the preserved preset)
        against the same estate-shaped literals: the test STATES which
        calibration it means instead of relying on a default (option (b) of
        the fix34 brief -- the explicit-suffix-tuple call sites further
        down, e.g. test_calibrated_templates_generate_exact_names_and_shapes,
        are the existing precedent and stay green). A silent reversion of
        either half of the naming would flip this leg red again, which is
        the anti-reversion coverage the test always carried. Both
        assertions below are unchanged."""
        with pytest.raises(
            lsg.GateUnmeasured, match="adapter naming disagreement"
        ) as exc_info:
            lsg._verify_adapter_naming_agreement(
                lsg._HF_PEFT_ADAPTER_SUFFIX_RE,
                "",
                (".adapter.linear_in.weight", ".adapter.linear_out.weight"),
            )
        assert "--adapter-suffix-a" in str(exc_info.value)
        assert ".adapter.linear_in.weight" in str(exc_info.value)

    def test_recognizer_that_matches_but_mis_cuts_is_refused(self):
        """[FAILS-BEFORE -- function absent pre-patch] A bare `lora_[AB]`
        regex search-matches the default templates but stops before
        ".weight", gluing a stray dot into every parent lookup; agreement
        means round-trip identity of the parent stem, not "matched
        somewhere"."""
        with pytest.raises(lsg.GateUnmeasured, match="recovers"):
            lsg._verify_adapter_naming_agreement(
                r"lora_[AB]", "", (".lora_A.weight", ".lora_B.weight")
            )

    def test_identical_templates_refused(self):
        """[FAILS-BEFORE -- function absent pre-patch] The same literal twice
        would declare one FQN with two different shapes, one silently
        overwriting the other in the derived map."""
        with pytest.raises(lsg.GateUnmeasured, match="identical"):
            lsg._verify_adapter_naming_agreement(
                lsg._DEFAULT_ADAPTER_SUFFIX_RE,
                "",
                (".lora_A.weight", ".lora_A.weight"),
            )

    def test_cli_refusal_is_exactly_three_and_names_the_disagreement(
        self, tmp_path, capsys
    ):
        """[FAILS-BEFORE -- the CLI flags do not exist pre-patch: argparse
        exits 2 via SystemExit, uncaught here, so red by error] End-to-end
        MUST_FIRE at the interface the launcher sees. Exit THREE, not one:
        a knob disagreement is not a property of the checkpoint, and the
        retry policy turns on the difference.
        fix34 calibration record: post-T2 the shipped recognizer IS the
        Megatron-Bridge shape, so the half-calibration a launcher can still
        commit is the REVERSE of the pre-T2 one -- the retired PEFT literals
        pinned through --adapter-suffix-a/-b against the shipped default.
        The argv below pins exactly that pair. The exit-3 contract and the
        named-flag assertions are unchanged, and the checkpoint fixture is
        moot by construction: the agreement veto runs at the top of
        adjudicate_checkpoint, before the base model, the config, or the
        artifact is read, so this test cannot be turned into a vacuous pass
        by fixture drift -- a refused measurement that never reached the
        artifact is exactly the property under test."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        code = lsg_cli.main([str(ckpt), "--run-kind", "lora",
                         "--base-model-dir", str(base),
                         "--train-config", str(cfg),
                         "--adapter-prefix", "",
                         "--adapter-suffix-a", ".lora_A.weight",
                         "--adapter-suffix-b", ".lora_B.weight"])
        assert code == 3
        err = capsys.readouterr().err
        assert "adapter naming disagreement" in err
        assert "--adapter-suffix-a" in err

    def test_calibrated_nondefault_naming_clears_end_to_end(self, tmp_path):
        """[FAILS-BEFORE -- kwarg adapter_suffixes does not exist pre-patch
        -> TypeError, red] MUST_PASS: a correctly calibrated NON-default pair
        flows through generation, SaveCompletenessGate, and the structural
        sweep to CLEAR, with the derived denominator on the wire. This is the
        fixture-shaped answer to the defect narrative: correct calibration
        must never again be the CAUSE of a catastrophic-looking verdict.

        #80 amendment: the fixture now also carries the 6 optimizer.* +
        1 rng_state entries measured on the production save. That addition is
        the control #80 needed all along -- on the unfixed tree this test
        goes RED with EXIT 1 (the lora branch adjudicated those 7 entries as
        "unrecognized adapter content"), and it returns GREEN only via the
        anchored namespace exclusion, never via a weakened assertion below.
        It is also the DELETION-control, site by site. Call sites of
        _is_non_adapter_namespace are THREE -- 1460 in _infer_auto_kind,
        1540 (the set-aside), 1574 (the unmarked sweep). Removing the
        frozenset reddens every caller (NameError on first use). Removing
        1574 reddens THIS test: the 7 measured save-state entries join
        `unmarked`, the lora branch blocks, the asserted exit 0 flips to 1.
        Removing 1540 leaves this test GREEN -- `unmarked` stays empty and
        no note is asserted here -- and is caught by the sibling decoy
        test's denominator strings. Removing 1460 is invisible here (this
        test pins run_kind="lora" and never reaches the auto seam) and is
        pinned by test_auto_kind_denominator_excludes_save_state."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(
            tmp_path, _megatron_named_lora_tensors(), name="mt-lora")
        # fix45/#78: the lora derive now demands an --adapter-modules census
        # (artifact namespace, written outside the judged tree). Names-only
        # is the honest minimum: the shape check then abstains BY NAME, an
        # abstention no assertion here inspects. Assertions unchanged.
        census = _census_file(tmp_path, self._census_stems())
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_suffix_re=r"\.adapter\.linear_(?:in|out)\.weight$",
            adapter_suffixes=(".adapter.linear_in.weight",
                              ".adapter.linear_out.weight"),
            adapter_modules=census)
        assert d.exit_code == 0, f"calibrated non-default must CLEAR: {d.blocking_reasons}"
        # 31 is the honest #80 denominator, NOT a weakened assertion: 24
        # adapter + 6 optimizer + 1 rng_state, the measured non-adapter shape
        # of a real save. Holding the RAW inventory at exactly 31 alongside
        # the DECLARED 24 is what keeps the exclusion provably narrow --
        # 7 entries were excused by namespace root, and every one is still
        # counted on disk. Keeping 24 here would be residual fixture/defect
        # shape-sharing; asserting only the declared side would let a future
        # exclusion-maker silently shrink the population (doctrine 2), which
        # is indistinguishable from the detector stopping working.
        assert d.report["inventory"]["real_tensors"] == 31
        assert "24 adapter tensors" in d.declared_basis["fqns"]
        assert ".adapter.linear_in.weight" in d.declared_basis["fqns"]
        assert _control_by_prefix(d, "drop")["status"] == "fired"

    def test_optimizer_shaped_decoy_still_flagged_as_unmarked(self, tmp_path):
        """[FAILS-BEFORE -- pre-#80 the exclusion does not exist: the decoys
        flag as 10 of 34 rather than 3 of 27, and the judged/excluded
        denominator format is absent -> red] MUST_FIRE for the #80 namespace
        exclusion. Genuinely unrecognized tensors wearing optimizer-SHAPED
        names -- NOT one of the measured non-adapter namespace ROOTS -- must
        still be adjudicated as unrecognized adapter content. Three decoy
        shapes, one per decay class of the root-segment anchor: the letters
        embedded INSIDE a module name (`layers.3.self_attn.optimizer_gate`),
        a bare root (`optimizer_gate.x` -- fqn.partition(".")[0] yields the
        root segment "optimizer_gate", which the anchored match must still
        refuse), and an exact mid-path "optimizer" segment
        (`layers.9.self_attn.optimizer.exp_avg.weight`). The tree shipped
        with only the first while _is_non_adapter_namespace's own docstring
        already cited `optimizer_gate.x` as controlled here and the
        paragraph below promised an any-segment redden -- two doctrine-5
        over-claims, repaired by measuring, not by rewording the claims
        away.

        Broken to see red, one arm per widening class. A substring widening
        (`"optimizer" in fqn`) swallows ALL THREE decoys: no MODE/lora
        "adapter marker" reason fires, exit flips to 0, this test goes red
        -- that mutation is exactly "exclude a namespace" decaying into
        "delete the check", invisible to the sibling MUST_PASS above, which
        only ever sees legitimate namespace roots. A prefix widening
        (`fqn.startswith("optimizer")`) swallows ONLY the bare-root decoy;
        an exact any-segment widening swallows ONLY the mid-path decoy; each
        leaves `flagged` non-empty but slides the pinned count to "2 of
        26", dying on the exact-count assertion -- the embedded stem alone
        cannot see either shape. MUST_PASS/MUST_FIRE division: the sibling
        goes red if call site 1574 is DELETED; this one goes red on
        WIDENING, on call site 1540's deletion (denominator strings), and
        on call site 1574's own deletion (numerator re-count, "10 of 27").
        Denominator per doctrine 2: 24 adapter + 3 decoys = 27 JUDGED
        adapter-namespace tensors, with the 7 legitimate save-state entries
        quoted in the reason as set aside -- reported, not silently dropped.
        """
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        tensors = _megatron_named_lora_tensors()
        # Three decoys, one per decay class of the anchored root-segment
        # match (named in the docstring): "optimizer" embedded INSIDE a
        # module name, a BARE ROOT carrying the letters, and an exact
        # "optimizer" segment MID-PATH. None carries the adapter suffix or
        # marker or sits in modules_to_save, so only a broken exclusion
        # lets any of them pass. The count pin below quotes the
        # sorted-first decoy, layers.3.self_attn.optimizer_gate.weight.
        tensors["layers.3.self_attn.optimizer_gate.weight"] = ((8, 8), "F32")
        tensors["optimizer_gate.x"] = ((4,), "F32")
        tensors["layers.9.self_attn.optimizer.exp_avg.weight"] = ((8, 8), "F32")
        ckpt = _materialize_artifact(tmp_path, tensors, name="mt-lora-decoy")
        census = _census_file(tmp_path, self._census_stems())
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_suffix_re=r"\.adapter\.linear_(?:in|out)\.weight$",
            adapter_suffixes=(".adapter.linear_in.weight",
                              ".adapter.linear_out.weight"),
            adapter_modules=census)
        assert d.exit_code == 1, (
            f"optimizer-shaped decoys must still hard-block: {d.blocking_reasons}")
        flagged = [r for r in d.blocking_reasons
                   if "MODE/lora" in r and "adapter marker" in r]
        assert flagged, f"no unmarked-adapter reason fired: {d.blocking_reasons}"
        # "3 of 27" is the anti-disarm pin, one arm per failure shape.
        # Pre-#80 (no exclusion): "10 of 34" -- the 3 decoys + the 7
        # save-state entries over the raw 34. Deleting call site 1574
        # re-flags those 7 ("10 of 27"); deleting 1540 inflates the judged
        # denominator ("3 of 34") and the pinned "7 non-adapter" string
        # below stops matching; a substring widening ("optimizer" in fqn)
        # empties `flagged` and flips the exit (caught above); a prefix
        # widening swallows only the bare-root decoy and an exact
        # any-segment widening only the mid-path one, each sliding the
        # count to "2 of 26" without emptying it -- both die here on the
        # exact count. Six failure shapes, each landing on a named
        # assertion.
        assert any("3 of 27" in r and "optimizer_gate.weight" in r
                   for r in flagged), f"decoys not isolated in reason: {flagged}"
        assert any("7 non-adapter" in r for r in flagged), (
            f"excluded-namespace count missing from reason: {flagged}")

    def test_auto_kind_denominator_excludes_save_state(self):
        """[FAILS-BEFORE -- lsg._infer_auto_kind does not exist pre-patch ->
        AttributeError, red] MUST_FIRE for the latent second bite of #80: the
        AUTO-KIND denominator. Pre-patch the inline code computed
        frac = marked / len(real_fqns); on leg one below that is 4/16 = 0.25
        < 0.6 -> "full", routing a LoRA save into the MODE/full "population
        looks partial" blocker -- #80 re-worded. Latent in production (both
        launchers pin --run-kind), real for --run-kind auto and library
        callers.

        Broken to see red (the mutation leg one exists for): revert the
        judged pool from the excluded view back to raw real_fqns, and the
        kind flips to "full". Leg three is the mirrored seam check for the
        SAME anchor the end-to-end decoy pins: widen the root-segment match
        and the decoy vanishes from the judged pool, snapping the basis from
        "4/5" back to "4/4". Leg four pins doctrine 1/4 at the seam: a judged
        pool of ZERO is UNMEASURED (GateUnmeasured), never a guessed kind.
        `lsg.re` is used so this file needs no new import for a one-off
        pattern."""
        markers = lsg.re.compile(r"\.adapter\.linear_(?:in|out)\.weight$")
        fqns = {f"layers.{i}.self_attn.q_proj.adapter.linear_in.weight"
                for i in range(4)}
        fqns |= {f"optimizer.state.exp_avg.block{i}.weight" for i in range(11)}
        fqns.add("rng_state")
        kind, basis = lsg._infer_auto_kind(fqns, markers)
        # Post-exclusion the judged pool is 4/4 = 1.00; raw counting would
        # give 4/16 = 0.25 -> "full". The basis string carries BOTH counts so
        # the shrink is reported, not silent (doctrine 2).
        assert kind == "lora", (
            f"save-state namespaces dragged auto-kind to full: {basis}")
        assert "4/4" in basis and "12 non-adapter" in basis, basis
        # Embedded-segment decoy at the seam: "optimizer" inside a module
        # name is NOT a namespace root, so it must enter the judged
        # denominator -- 4/5 = 0.80 still resolves lora, but the string
        # moves only while the decoy is counted.
        fqns.add("layers.9.self_attn.optimizer_gate.weight")
        kind, basis = lsg._infer_auto_kind(fqns, markers)
        assert kind == "lora" and "4/5" in basis, basis
        # Zero judged entries: nothing measurable. all-clear on an empty pool
        # is vacuous truth; refuse instead of guessing.
        try:
            lsg._infer_auto_kind({"optimizer.state.exp_avg.only.weight"}, markers)
        except lsg.GateUnmeasured:
            pass
        else:
            raise AssertionError(
                "all-excluded pool must raise GateUnmeasured, not guess a kind")

    def test_calibrated_templates_generate_exact_names_and_shapes(self, tmp_path):
        """[FAILS-BEFORE -- kwarg absent pre-patch] Unit-level MUST_PASS on
        the generator alone: the exact declared FQN set and the positional
        shape convention, against the same in-module BaseModel construction
        TestMoeOverride already uses."""
        assert lsg._probe_derive_declared is not None, (
            "probe import failed -- the sys.path insertion in _load_gate "
            "regressed; fix the loader, never this guard")
        spec = lsg.resolve_train_spec(dict(LORA_TRAIN), "test://cfg", "auto", None)
        base = lsg.BaseModel(
            model_dir=tmp_path, config=DENSE_CFG,
            tensors={k: (v[0], "float32")
                     for k, v in _dense_full_tensors().items()},
            tensors_source="test://synthetic")
        # fix45/#78: the lora derive now demands an --adapter-modules census.
        # The want-map below pins generator SHAPES, which post-#78 are
        # declared only from census-carried parent dims x config rank (a
        # names-only census would abstain shape-by-name and mint empty
        # shapes, turning this test's exact-equality pin red), so the census
        # carries every stem's (out, in) = (8, 8) -- the fixture parents'
        # real dims. The dims are the load-bearing part of this fixture,
        # not a nicety.
        stems = self._census_stems()
        decl = lsg.derive_declared_block(
            base, spec, set(), "",
            adapter_suffixes=(".adapter.linear_in.weight",
                              ".adapter.linear_out.weight"),
            adapter_modules=_census(stems, dims={s: (8, 8) for s in stems}))
        want = {}
        for i in range(6):
            for w in ("q_proj", "v_proj"):
                stem = f"layers.{i}.self_attn.{w}"
                want[f"{stem}.adapter.linear_in.weight"] = (4, 8)
                want[f"{stem}.adapter.linear_out.weight"] = (8, 4)
        assert decl.derived_adapter == want
        assert decl.fqns == tuple(sorted(want))

    def test_default_templates_generate_byte_identical_declared_set(self, tmp_path):
        """[PASSES-BEFORE and PASSES-AFTER -- constraint (d) fence, declared
        per the house rule] The fifth positional argument is passed
        EXPLICITLY with the default-shaped pair so the call is valid on both
        trees: pre-patch it lands in the (ignored) regex parameter and the
        hardcoded PEFT literals produce the expected set; post-patch it lands
        in the literal-templates parameter and produces the same set. The
        review question "did the defaults change one byte?" is this test, and
        any drift in defaults, ordering, prefixing, or the shape convention
        is its own red-maker."""
        assert lsg._probe_derive_declared is not None, (
            "probe import failed -- the sys.path insertion in _load_gate "
            "regressed; fix the loader, never this guard")
        spec = lsg.resolve_train_spec(dict(LORA_TRAIN), "test://cfg", "auto", None)
        base = lsg.BaseModel(
            model_dir=tmp_path, config=DENSE_CFG,
            tensors={k: (v[0], "float32")
                     for k, v in _dense_full_tensors().items()},
            tensors_source="test://synthetic")
        stems = self._census_stems()
        decl = lsg.derive_declared_block(
            base, spec, set(), "", (".lora_A.weight", ".lora_B.weight"),
            # fix45/#78: same census contract as the calibrated twin above --
            # dims carried so the (rank, in)/(out, rank) shapes this fence
            # pins are actually declared (post-#78 shapes require census
            # dims x config rank).
            adapter_modules=_census(stems, dims={s: (8, 8) for s in stems}))
        want = {}
        for i in range(6):
            for w in ("q_proj", "v_proj"):
                stem = f"layers.{i}.self_attn.{w}"
                want[f"{stem}.lora_A.weight"] = (4, 8)
                want[f"{stem}.lora_B.weight"] = (8, 4)
        assert decl.derived_adapter == want
        assert decl.fqns == tuple(sorted(want))


class TestAdapterPrefixDemand:
    def test_unpinned_prefix_lora_refuses_instead_of_guessing(self, tmp_path):
        """[FAILS-BEFORE -- pre-patch there is no demand: the call below
        returns a CLEAR decision, pytest.raises sees no exception, red]
        MUST_FIRE for the demand: the old "" default was a guess, and the
        tool now refuses to be silently responsible for it."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        with pytest.raises(lsg.GateUnmeasured, match="--adapter-prefix"):
            lsg.adjudicate_checkpoint(
                ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg)

    def test_unpinned_prefix_auto_kind_also_refuses(self, tmp_path):
        """[FAILS-BEFORE -- same mechanism] The demand sits AFTER auto kind
        resolution on purpose: marker inference may consult the artifact for
        KIND, but the prefix question must not be answered by silence on the
        auto path either. Config carries rank/targets but no kind key, so the
        lora kind here is marker-inferred, not config-declared."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, _lora_tensors())
        cfg = _write_cfg(tmp_path, {"lora_rank": 4,
                                    "lora_targets": ["q_proj", "v_proj"]},
                         name="auto-no-kind-no-prefix.json")
        with pytest.raises(lsg.GateUnmeasured, match="--adapter-prefix"):
            lsg.adjudicate_checkpoint(
                ckpt, run_kind="auto", base_model_dir=base, train_config_path=cfg)

    def test_cli_lora_without_prefix_is_exactly_three(self, tmp_path):
        """[FAILS-BEFORE -- pre-patch the identical argv returns 0] The
        launcher-facing shape of the demand: refused measurement, not a
        checkpoint verdict."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        code = lsg_cli.main([str(ckpt), "--run-kind", "lora",
                         "--base-model-dir", str(base),
                         "--train-config", str(cfg)])
        assert code == 3

    def test_explicit_empty_prefix_is_an_assertion_and_clears(self, tmp_path):
        """[PASSES-BEFORE and PASSES-AFTER as a behaviour fence -- pre-patch
        #80 repair record: the shared healthy fixture now carries the
        measured 7 non-adapter checkpoint-namespace entries (6 optimizer.*
        + 1 rng_state), so this fence pins 31 real in the fail-closed
        artifact inventory against 24 judged adapter tensors. The body
        inventory-denominator assertion migrates 24 -> 31 with the
        fixture -- a one-line change at that assertion; the posted failure
        dump did not expose a byte-exact anchor for it, so it is recorded
        here for hand-application rather than fabricated, and the bare
        string is NOT unique in this file (do not bulk-replace).
        Discrimination is unchanged: the refusing twins pin the demand,
        exit-0 pins the healthy path. Refused repairs: stripping the 7
        entries back out of the fixture to restore the old literal
        (blinds the measured-save evidence this fence exists to carry),
        or teaching the tool to exclude the set-aside from the inventory
        (wire lie). What follows reads:
        this call is byte-identical to the old default path; the
        DISCRIMINATION is carried by the refusing twin tests above, stated
        here per the house rule] MUST_PASS twin for the demand: an explicit
        "" is an operator assertion of the unprefixed layout, and a healthy
        unprefixed adapter under that assertion must CLEAR, with its
        denominator on the wire.
        fix45: post-#78 this CLEAR arm additionally needs the
        --adapter-modules census; it carries the honest one (the 12
        artifact-namespace stems implied by the run's declared structure,
        outside the judged tree). Assertions unchanged. The refusing twins
        above need no census and stay unedited: the prefix demand fires
        before derivation, so they were never census-deficient -- that
        ordering (prefix first, census demand at derive) is itself the
        production calibration this class exists to protect."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        census = _census_file(
            tmp_path,
            [f"layers.{i}.self_attn.{w}"
             for i in range(6) for w in ("q_proj", "v_proj")],
        )
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg,
            adapter_prefix="", adapter_modules=census)
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        # #80: 31 is the honest EXAMINED denominator, NOT a weakened
        # assertion: 24 judged adapter tensors + 7 set-aside save-state
        # entries (6 optimizer.* + 1 rng_state) the healthy fixture grew
        # this window. The inventory stays fail-closed over the PHYSICAL
        # artifact (doctrine 2: the claim states how many units were
        # examined); the JUDGED adapter population stays 24 and is pinned
        # by the refusing twins above. Holding RAW=31 alongside JUDGED=24
        # keeps the namespace exclusion provably narrow -- all 7 excused
        # entries are still counted on disk. Refused alternatives:
        # stripping the 7 entries back out of the fixture would blind the
        # measured-save evidence this fence exists to carry; teaching the
        # tool to exclude the set-aside from the inventory would be a
        # wire lie; asserting only the judged side would let a future
        # exclusion-maker silently shrink the population, which is
        # indistinguishable from the detector stopping working. Same
        # correction the sibling sites carry (626, 1109, 2029, 2717).
        assert d.report["inventory"]["real_tensors"] == 31, (
            f"post-#80 the examined real population is 31 (24 judged "
            f"adapters + 6 optimizer.* + 1 rng_state), not the stale "
            f"pre-#80 literal 24 -- got "
            f"{d.report['inventory']['real_tensors']}; if 31 ever "
            f"changes, the healthy fixture's real save shape changed and "
            f"the #80 set-aside must be re-measured, never re-narrowed "
            f"to make an expected constant come true"
        )
        assert _control_by_prefix(d, "drop")["status"] == "fired"
