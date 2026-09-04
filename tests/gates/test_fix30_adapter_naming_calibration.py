"""fix30: the adapter-naming calibration must describe the estate's real LoRA stack.

Measured (fix30, read on the estate): Megatron-Bridge's
``peft/lora.py:LoRA.transform`` wraps matched parallel linears as
``LoRALinear(base, ParallelLinearAdapter(...))``; ``peft/adapter_wrapper.py``
writes the adapter under ``f"{prefix}adapter."`` in both ``state_dict`` and
``sharded_state_dict``; ``peft/utils.py:ParallelLinearAdapter`` writes
``linear_in.``/``linear_out.`` inside that, ``bias=False``. The saved tensors
are therefore, per adapted linear:

    <base_linear_fqn>.adapter.linear_in.weight   # A matrix, (rank, in)
    <base_linear_fqn>.adapter.linear_out.weight  # B matrix, (out, rank)

The pre-fix defaults were the HF PEFT convention (.lora_A/.lora_B), which
match ZERO tensors of any save this estate can produce -- and no caller
passes the overriding flags, so the default WAS the calibration. This file
is the doctrine-3 pair for the recalibration: one MUST_PASS over a synthetic
Megatron-Bridge-shaped adapter set, one MUST_FIRE proving the retired
calibration really was fatal (constructed from frozen literals, not
asserted), plus the named refusals for the two layouts the same peft tree
can emit (fix30c) and the agreement gate's veto of half-calibrations. Every
test states its fail-before/pass-after direction at its head, and legs that
pin machinery predating this patch are labeled as such -- a repair must
never claim to have built what it merely pointed at reality (doctrine 5).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS_DIR = (
    next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "tools"
)
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# The decision API is a library module since T2_lib_script_boundary#0;
# tools/live_save_gate.py is now an argparse wrapper over it. The sys.path
# fixup above stays: adjudication imports real_checkpoint_probe as a sibling.
from foundationscale.gates import adjudication as lsg  # noqa: E402 (path fixup first)

# The estate shape is the INVARIANT here: these two literals are what the
# measured stack writes, whatever vintage the tool under test is. The
# calibration under test is the variable; fixtures pinned to
# lsg._DEFAULT_ADAPTER_SUFFIX_* would tautologically follow the patch and
# could never FAIL on the pre-fix tree.
_MB_SUFFIX_A = ".adapter.linear_in.weight"  # (rank, in_features)
_MB_SUFFIX_B = ".adapter.linear_out.weight"  # (out_features, rank)
_MB_SUFFIX_RE = r"\.adapter\.linear_(?:in|out)\.weight$"

RANK = 32
# Megatron-spelled module stems (the LoRA launcher's own target spellings,
# seen in its census comments) -> (out_features, in_features). Five parents
# x two matrices = a 10-tensor synthetic adapter save; every denominator in
# every assertion below names this 10.
_PARENTS: dict[str, tuple[int, int]] = {
    "module.decoder.layers.0.self_attention.linear_qkv": (6144, 2560),
    "module.decoder.layers.0.self_attention.linear_proj": (2560, 2560),
    "module.decoder.layers.0.mlp.mlp.linear_fc1": (10240, 2560),
    "module.decoder.layers.0.mlp.mlp.linear_fc2": (2560, 10240),
    "module.decoder.layers.1.self_attention.linear_qkv": (6144, 2560),
}
N_TENSORS = 2 * len(_PARENTS)


def _base() -> lsg.BaseModel:
    """Synthetic base header: exactly the five parent .weight rows, nothing else."""
    return lsg.BaseModel(
        model_dir=Path("/synthetic-fix30"),
        config={},
        tensors={f"{p}.weight": (shape, "bfloat16") for p, shape in _PARENTS.items()},
        tensors_source="synthetic 5-parent base header (fix30 control; no file on disk)",
    )


def _real_estate() -> list[tuple[str, SimpleNamespace]]:
    """A healthy Megatron-Bridge adapter save for _PARENTS, in both matrices."""
    out: list[tuple[str, SimpleNamespace]] = []
    for stem, (out_f, in_f) in _PARENTS.items():
        out.append((f"{stem}{_MB_SUFFIX_A}", SimpleNamespace(shape=(RANK, in_f))))
        out.append((f"{stem}{_MB_SUFFIX_B}", SimpleNamespace(shape=(out_f, RANK))))
    return out


def _decl() -> lsg.Declared:
    """The declared adapter map for the fixture, keying shape checks on."""
    derived: dict[str, tuple[int, ...]] = {}
    for stem, (out_f, in_f) in _PARENTS.items():
        derived[f"{stem}{_MB_SUFFIX_A}"] = (RANK, in_f)
        derived[f"{stem}{_MB_SUFFIX_B}"] = (out_f, RANK)
    return lsg.Declared(
        fqns=tuple(sorted(derived)),
        fqns_basis="synthetic fixture: 5 parents x 2 templates x rank 32 (fix30)",
        num_experts=0,
        experts_basis="dense fixture",
        num_moe_layers=0,
        moe_layers_basis="dense fixture",
        expected_expert_bytes=None,
        bytes_basis="no experts in fixture",
        derived_adapter=derived,
        notes=[],
    )


def _spec() -> lsg.TrainSpec:
    return lsg.TrainSpec(
        run_kind="lora",
        kind_basis="synthetic fixture (fix30)",
        lora_rank=RANK,
        rank_basis="synthetic fixture rank 32",
        lora_targets=("linear_qkv", "linear_proj", "mlp.linear_fc1", "mlp.linear_fc2"),
        targets_basis="synthetic fixture: the LoRA launcher's dense-base target set",
        frozen_regex=None,
        frozen_basis="no freeze in fixture",
        cfg_source="synthetic",
    )


def test_default_calibration_binds_every_estate_adapter_to_its_parent() -> None:
    """MUST_PASS: FAILS on the pre-fix tree, PASSES after the patch.

    Fail-before mechanism: the fixture is Megatron-Bridge-shaped (pinned by
    the _MB_* literals above) while the recognizer argument below is the
    tool's SHIPPED default -- the pre-fix default is the HF PEFT regex, which
    matches 0 of 10 fixture tensors, so the sweep returns the vacuous-detector
    refusal and `findings == []` fails. Direction confirmed by construction,
    not execution: this reasoning runs against the handed source only.
    """
    real = _real_estate()
    findings = lsg.lora_structural_findings(
        real,
        _base(),
        _decl(),
        _spec(),
        adapter_prefix="",
        adapter_suffix=lsg._DEFAULT_ADAPTER_SUFFIX_RE,
    )
    matched = [f for f, _tm in real if re.search(lsg._DEFAULT_ADAPTER_SUFFIX_RE, f)]
    # Denominator named (doctrine 2): all 10 adapter tensors recognized AND
    # bound -- not merely "no complaint", which on a 0-of-10 sweep is what
    # the pre-fix defect sounded like.
    assert len(matched) == N_TENSORS
    assert findings == []

    # No-regression pins for the prefix mechanics the PROBE measure-then-pin
    # loop depends on. These legs are GREEN on both trees: the strip-and-miss
    # semantics predate this patch, and that is stated so the patch is not
    # credited with building them (doctrine 5). (i) A correctly pinned
    # constant segment still binds 10/10; the derived-map shape check is
    # keyed on unprefixed names and declines cleanly under a prefix, as in
    # production. (ii) A WRONG (here: absent but needed) pin is loud --
    # 10 phantom parents, never silently corrected.
    base, decl, spec = _base(), _decl(), _spec()
    prefixed = [(f"savewrap.{f}", tm) for f, tm in real]
    assert (
        lsg.lora_structural_findings(
            prefixed,
            base,
            decl,
            spec,
            adapter_prefix="savewrap.",
            adapter_suffix=lsg._DEFAULT_ADAPTER_SUFFIX_RE,
        )
        == []
    )
    wrong = lsg.lora_structural_findings(
        prefixed,
        base,
        decl,
        spec,
        adapter_prefix="",
        adapter_suffix=lsg._DEFAULT_ADAPTER_SUFFIX_RE,
    )
    assert len(wrong) == 1
    assert f"{N_TENSORS} adapter tensor(s) attach to parents absent" in wrong[0]


def test_retired_hf_peft_calibration_is_fatal_on_the_estate_shape() -> None:
    """MUST_FIRE: FAILS on the pre-fix tree (anti-reversion leg), PASSES after.

    The firing input is the pre-fix DEFAULT, frozen here as literals -- the
    three values _DEFAULT_ADAPTER_SUFFIX_{RE,A,B} held before the patch --
    because a control pinned to the constants the patch changes would
    tautologically follow the patch and prove nothing. Two sub-legs with
    honestly different directions, labeled: the refusal itself (first
    asserts) is machinery that PREDATES the patch and is green on both
    trees, pinned so it cannot silently regress; the anti-reversion /
    preset legs are what is red before and green after.
    """
    hf_re = r"\.(lora_[AB](?:\.weight)?)$"
    hf_a, hf_b = ".lora_A.weight", ".lora_B.weight"

    # The detector still sees the defect class, with its denominator: the
    # retired calibration against the estate shape examines 10 tensors and
    # binds 0 -- exactly one blocking refusal, never a crash or a silent pass.
    findings = lsg.lora_structural_findings(
        _real_estate(),
        _base(),
        _decl(),
        _spec(),
        adapter_prefix="",
        adapter_suffix=hf_re,
    )
    assert len(findings) == 1
    assert f"lora: 0 of {N_TENSORS} real tensors could be bound to a base parent" in findings[0]

    # Fail-before leg: on the pre-fix tree the shipped defaults ARE this
    # calibration and this assertion fails; after the patch it guards the
    # repair against silent reversion to the HF convention.
    assert hf_re != lsg._DEFAULT_ADAPTER_SUFFIX_RE
    assert (hf_a, hf_b) != lsg._DEFAULT_ADAPTER_SUFFIXES

    # The retired convention is narrowed, not amputated: it remains as an
    # explicit preset for estates that truly train with HF peft. Red before
    # (the preset names do not exist), green after.
    assert hf_re == lsg._HF_PEFT_ADAPTER_SUFFIX_RE
    assert (hf_a, hf_b) == lsg._HF_PEFT_ADAPTER_SUFFIXES


def test_other_peft_layouts_are_refused_by_name() -> None:
    """fix30(c) MUST_FIRE: FAILS before (no named refusals exist), PASSES after.

    Fail-before mechanism: both foreign FQNs carry the substring "adapter",
    so the pre-fix `unmarked` net missed them; under the pre-fix HF default
    recognizer the whole fixture was the generic 0-of-2 vacuity text with no
    layout name anywhere. Post-fix each layout is refused by its own name,
    with its own denominator, under exactly one name (canonical first).
    """
    real = [
        # CanonicalLoRA split-QKV: one ModuleDict segment deeper than plain LoRA.
        (
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.adapter_q.linear_in.weight",
            SimpleNamespace(shape=(RANK, 2560)),
        ),
        # LinearAdapter/TELinearAdapter self-mounted: no ".adapter." segment at all.
        (
            "module.decoder.layers.0.mlp.mlp.linear_fc1.linear_in.weight",
            SimpleNamespace(shape=(RANK, 2560)),
        ),
    ]
    findings = lsg.lora_structural_findings(
        real,
        _base(),
        _decl(),
        _spec(),
        adapter_prefix="",
        adapter_suffix=lsg._DEFAULT_ADAPTER_SUFFIX_RE,
    )
    text = "\n".join(findings)
    assert "CanonicalLoRA" in text
    assert f"1 of {len(real)} real tensor(s) match the CanonicalLoRA" in text
    assert "LinearAdapter/TELinearAdapter" in text
    assert f"1 of {len(real)} real tensor(s) match the LinearAdapter/TELinearAdapter" in text
    # The calibrated recognizer must NOT secretly swallow either layout:
    # zero bound under the shipped default, so the 0-of-N vacuity refusal
    # accompanies the named ones. MUST_PASS for this detector is exercised
    # above: the healthy estate fixture produces no named refusals at all.
    assert f"lora: 0 of {len(real)} real tensors could be bound" in text


def test_agreement_gate_accepts_defaults_and_vetoes_half_calibrations() -> None:
    """Decision (a)/(b) pinned: FAILS before (defaults-equal legs), PASSES after.

    The no-raise self-agreement and both veto legs are pre-existing machinery,
    green on both trees -- labeled, not claimed by this patch. The
    defaults-equal literals are the red-before legs, and they fix the exact
    measured strings so a silent drift of the calibration is a legible test
    failure, not a surprise in a launch log.
    """
    assert lsg._DEFAULT_ADAPTER_SUFFIX_A == _MB_SUFFIX_A
    assert lsg._DEFAULT_ADAPTER_SUFFIX_B == _MB_SUFFIX_B
    assert lsg._DEFAULT_ADAPTER_SUFFIX_RE == _MB_SUFFIX_RE

    hf_re = r"\.(lora_[AB](?:\.weight)?)$"
    hf_pair = (".lora_A.weight", ".lora_B.weight")
    # The shipped defaults agree with themselves: no exception.
    lsg._verify_adapter_naming_agreement(
        lsg._DEFAULT_ADAPTER_SUFFIX_RE, "", lsg._DEFAULT_ADAPTER_SUFFIXES
    )
    # Half-calibration, direction one: new recognizer, old templates.
    with pytest.raises(lsg.GateUnmeasured):
        lsg._verify_adapter_naming_agreement(lsg._DEFAULT_ADAPTER_SUFFIX_RE, "", hf_pair)
    # Half-calibration, direction two: old recognizer, new templates -- the
    # precise pre-fix state had the defaults been changed on one side only.
    with pytest.raises(lsg.GateUnmeasured):
        lsg._verify_adapter_naming_agreement(hf_re, "", (_MB_SUFFIX_A, _MB_SUFFIX_B))
