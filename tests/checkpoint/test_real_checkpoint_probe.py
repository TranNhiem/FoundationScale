"""Laptop proof for tools/real_checkpoint_probe.py against known ground truth.

Why this suite exists
---------------------
The probe's exit code is about to be quoted on a GPU cluster as evidence that
the framework's checkpoint gates work on real artifacts. Everything here is
built so that the correct answer is known *before* the probe runs: safetensors
files written by this file (including one whose header deliberately puts four
expert FQNs on one byte span — the incident's mechanism, constructed, not
assumed) and config.json files whose denominators are stated exactly. If the
probe cannot reproduce ground truth on a 200-byte artifact it owns nothing to
say about a 16 GB one.

The load-bearing assertions are about honesty, not just verdicts: abstentions
must be *stated*, a fabricated-denominator route must not exist, a requested
control that cannot run must weigh against CLEAR, and a probe bug must surface
as exit 3 — never as Python's exit 1, which a pipeline will misfile as
"gates blocked the checkpoint".
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from foundationscale.gates.core import Verdict

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_PROBE_PATH = REPO_ROOT / "tools" / "real_checkpoint_probe.py"

_SPEC = importlib.util.spec_from_file_location("real_checkpoint_probe", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

_ST_DTYPE = {"bfloat16": "BF16"}


def _write_st_file(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))


def _write_header_shard(
    path: Path,
    entries: list[tuple[str, str, tuple[int, ...], int, int]],
) -> None:
    """Write a safetensors file whose header states *exactly* these offsets.

    Offsets are caller-controlled so a test can put several expert names on one
    byte span — aliasing ground truth this suite constructs rather than hopes
    for. Only header plus a nominal data pad are written: the reader under test
    is metadata-only, and a test that needed real tensor bytes would be testing
    the wrong layer.
    """
    header = {
        name: {
            "dtype": _ST_DTYPE[dtype],
            "shape": list(shape),
            "data_offsets": [start, end],
        }
        for name, dtype, shape, start, end in entries
    }
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    data_len = max((entry[4] for entry in entries), default=0)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * data_len)


def _dense_checkpoint(tmp_path: Path) -> Path:
    """A small dense artifact: the Gemma-family shape of *no routed experts*."""
    ckpt = tmp_path / "dense"
    ckpt.mkdir()
    _write_st_file(
        ckpt / "model.safetensors",
        {
            "model.embed_tokens.weight": torch.zeros((8, 4), dtype=torch.bfloat16),
            "model.layers.0.mlp.down_proj.weight": torch.zeros((4, 8), dtype=torch.bfloat16),
        },
    )
    (ckpt / "config.json").write_text(json.dumps({"model_type": "probe_dense"}), encoding="utf-8")
    return ckpt


def _moe_checkpoint(tmp_path: Path, *, aliased: bool) -> Path:
    """1 MoE layer x 2 expert weights x 4 experts; spans distinct or shared 4:1."""
    ckpt = tmp_path / ("moe_aliased" if aliased else "moe_healthy")
    ckpt.mkdir()
    stem = "model.layers.0.mlp.experts"
    if aliased:
        entries: list[tuple[str, str, tuple[int, ...], int, int]] = []
        for fc in (1, 2):
            base = 0 if fc == 1 else 8
            for i in range(4):
                # weight0..3 of one linear_fc all name the same 8 bytes.
                entries.append(
                    (f"{stem}.linear_fc{fc}.weight{i}", "bfloat16", (2, 2), base, base + 8)
                )
        entries.append(("model.embed_tokens.weight", "bfloat16", (8, 4), 16, 80))
        _write_header_shard(ckpt / "model.safetensors", entries)
    else:
        tensors: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.zeros((8, 4), dtype=torch.bfloat16)
        }
        for fc in (1, 2):
            for i in range(4):
                tensors[f"{stem}.linear_fc{fc}.weight{i}"] = torch.zeros(
                    (2, 2), dtype=torch.bfloat16
                )
        _write_st_file(ckpt / "model.safetensors", tensors)
    config = {
        "model_type": "probe_moe",
        "num_local_experts": 4,
        "num_moe_layers": 1,
    }
    (ckpt / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return ckpt


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = probe.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    ckpt: Path,
    *extra: str,
) -> tuple[int, dict[str, Any], str]:
    out_json = tmp_path / "probe-report.json"
    code, out, _err = _run(capsys, str(ckpt), "--json", str(out_json), *extra)
    report: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
    return code, report, out


def _verdict_of(report: dict[str, Any], gate_id: str) -> str:
    """The verdict a gate actually recorded, read from the structured report.

    Why this exists rather than substring-matching the stdout render: the
    ``"{gate_id}={verdict}"`` spelling is emitted by exactly ONE renderer --
    the ``blocking_reasons`` list -- so it can only ever appear for a gate
    that BLOCKED. Asserting ``f"{gate}={Verdict.SKIP.value}" not in out`` is
    therefore vacuously true for every input this suite can construct: the
    detector cannot fire, so its silence is not evidence (doctrine 3). Two
    such assertions shipped in this file and one more was nearly added; all
    three now read the verdict from ``report["gates"]``, where SKIP, PASS,
    VACUOUS and FAIL are all equally expressible and a wrong verdict is
    therefore actually detectable.

    Raises rather than defaulting when the gate did not run at all: a gate
    absent from the population is a different fact from a gate that skipped,
    and collapsing them would rebuild the vacuity this helper exists to kill.
    """
    for entry in report["gates"]:
        if entry["gate"] == gate_id:
            verdict: str = entry["verdict"]
            return verdict
    raise AssertionError(
        f"{gate_id} is absent from the report's gate population "
        f"({sorted(e['gate'] for e in report['gates'])}) -- 'did not run' is "
        f"not 'skipped', and this assertion refuses to read it as one"
    )


def test_dense_checkpoint_abstains_stated_and_completeness_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The honest answer on a clean dense artifact is NOT "clear": the probe
    # must SAY what it could not declare and why, and completeness must block,
    # because config.json states no tensor list. The word CLEAR must not appear
    # anywhere.
    #
    # What "abstains, stated" pins post-contract: this fixture's config states
    # NOTHING about routed experts (no count key, no enable_moe_block), and
    # under the two-source mint rule an uncommented absence is UNKNOWN, not
    # dense. The abstention therefore lives in the DECLARED BLOCK -- num_experts
    # stays None with its reason written next to it -- and the expert gates must
    # BLOCK on it rather than SKIP NOT_APPLICABLE. This test used to enforce
    # "stated, not silence" by matching the literal string "dense model" in
    # stdout: decorative prose from the retired mint-0 path, and a match that
    # could not distinguish "the reason was stated" from "some code path
    # happened to print those words while the JSON said something else". The
    # replacement asserts the property at four altitudes the pipeline
    # guarantees by construction -- the declared field, the provenance slot the
    # schema reserves for it, the stdout echo of that exact slot, and the
    # consequence the gates act on -- and pins contract vocabulary rather than
    # any one branch's sentence: "UNKNOWN" and "the gates block" are the module
    # docstring's own words ("Absence yields UNKNOWN (None, and the gates
    # block)") and appear in every None branch of derive_declared, so these
    # matches survive rewording of prose but not a change of reason class.
    ckpt = _dense_checkpoint(tmp_path)
    code, report, out = _report(tmp_path, capsys, ckpt)
    assert code == probe.EXIT_BLOCKED
    assert "CLEAR (exit 0)" not in out
    declared = report["declared"]
    # (1) The denominator: absent, not minted. Pre-contract this exact run
    # carried num_experts=0 -- an affirmative dense declaration fabricated
    # from silence -- and the expert gates honored it as NOT_APPLICABLE.
    assert declared["num_experts"] is None
    # (2) Stated: the refusal sits in the slot reserved for this field's
    # provenance, non-empty (an empty basis would be silence wearing a
    # schema), naming its reason class and its consequence.
    basis = declared["basis"]["num_experts"]
    assert basis
    assert "UNKNOWN" in basis
    assert "gates block" in basis
    # (3) Attributable: _print_declared renders that same string to stdout, so
    # the operator-facing output and the machine-readable report cannot tell
    # two different stories about why nothing was declared.
    assert basis in out
    # (4) Consequence: zero routed-expert declarations AND zero measured expert
    # tensors leave the distinctness gate exactly one honest road -- a
    # blocking verdict naming a zero-unit sweep (doctrine 1) -- so the gate
    # must appear in the list that owns the exit code. Membership is asserted,
    # not a verdict letter: VACUOUS vs UNDERCOVERED is checkpoint_gates'
    # choice, and pinning it from this file would assert past what we can see.
    assert any(
        reason.startswith("checkpoint.expert_distinctness=")
        for reason in report["blocking_reasons"]
    )
    # The minted-0 rendering of this run skipped that gate as NOT_APPLICABLE;
    # a SKIP from it is now reachable only through an earned, corroborated
    # dense 0 -- which test_affirmatively_dense_checkpoint_earns_the_expert_skip
    # below proves is still possible.
    assert _verdict_of(report, "checkpoint.expert_distinctness") != Verdict.SKIP.value
    assert f"checkpoint.save_complete={Verdict.VACUOUS.value}" in out
    assert report["control"] is None
    assert report["exit_code"] == code
    # The probe audits the framework's own no-PASS-over-zero-coverage rule on
    # every run; a breach would land here and would have blocked the exit code.
    assert report["framework_invariant_breaches"] == []


def test_healthy_sharded_moe_distinctness_passes_and_run_still_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Ground truth: 4 experts x 2 weights, each its own span. Distinctness must
    # reach a real PASS; byte volume must SKIP for want of a stated denominator;
    # completeness VACUOUS still owns the exit code.
    ckpt = _moe_checkpoint(tmp_path, aliased=False)
    code, report, out = _report(tmp_path, capsys, ckpt)
    assert code == probe.EXIT_BLOCKED
    for bad in (Verdict.FAIL, Verdict.VACUOUS, Verdict.UNDERCOVERED, Verdict.ERROR):
        assert f"checkpoint.expert_distinctness={bad.value}" not in out
    assert "checkpoint.expert_bytes: run manifest does not declare" in out
    assert report["framework_invariant_breaches"] == []


def test_aliased_expert_spans_are_caught_from_metadata_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The incident's mechanism, assembled header-first: 8 right-shaped expert
    # FQNs name 2 physical spans. This is the property the cluster run relies
    # on, proven here against a span map the test itself wrote.
    ckpt = _moe_checkpoint(tmp_path, aliased=True)
    code, _report_dict, out = _report(tmp_path, capsys, ckpt)
    assert code == probe.EXIT_BLOCKED
    assert f"checkpoint.expert_distinctness={Verdict.FAIL.value}" in out
    assert "aliased" in out


def test_inject_alias_control_fires_on_real_distinct_spans(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _moe_checkpoint(tmp_path, aliased=False)
    code, report, _out = _report(tmp_path, capsys, ckpt, "--inject-alias", "3")
    control = report["control"]
    assert control["status"] == "fired"
    assert control["aliased"] == 3
    # The unmodified artifact does NOT block this gate, so the block is
    # attributable to the injection — the flag exists to say when it is not.
    assert control["confounded"] is False
    assert control["aliasing_leg_observed"] is True
    assert code == probe.EXIT_BLOCKED  # save-completeness still owns the exit


def test_inject_alias_control_says_confounded_on_already_blocking_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # On an artifact that already blocks the gate, "fired" proves only that the
    # gate blocks — the control must say it cannot attribute the block.
    ckpt = _moe_checkpoint(tmp_path, aliased=True)
    _code, report, _out = _report(tmp_path, capsys, ckpt, "--inject-alias", "2")

    # This test asserted status == "fired" alongside confounded is True. The
    # comment one line up is the author's own statement of the invariant, and
    # the pair contradicted it: the record made the headline claim ("fired" —
    # the detector caught the injected aliasing) and then walked it back in a
    # second field the reader had to know to check. A consumer keying on
    # status, which is what a status field is for, read a causal attribution
    # the run never earned. The status itself now abstains, which is what
    # "cannot attribute the block" means; the confounded flag and its note are
    # unchanged and still assert-able, so no evidence was dropped in the move.
    assert report["control"]["status"] == "inconclusive"
    assert report["control"]["confounded"] is True
    assert "already blocked this gate" in report["control"]["inconclusive_reason"]
    assert "not that the injected" in report["control"]["inconclusive_reason"]


def test_skipped_control_weighs_against_clear(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A requested MUST_FIRE control that cannot run leaves "the detector fires
    # on real aliasing" unverified; that must appear as a blocking reason, not
    # ride along as a quiet pass — the all([]) result one level up.
    ckpt = _dense_checkpoint(tmp_path)
    code, report, _out = _report(tmp_path, capsys, ckpt, "--inject-alias", "4")
    assert report["control"]["status"] == "skipped"
    assert "inapplicable" in report["control"]["reason"]
    assert any("--inject-alias" in reason for reason in report["blocking_reasons"])
    assert code == probe.EXIT_BLOCKED


def test_missing_config_is_unmeasured_not_clear(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without the independent denominator source the probe has nothing to
    # compare against; that is exit 3, never a verdict about the checkpoint.
    ckpt = _dense_checkpoint(tmp_path)
    (ckpt / "config.json").unlink()
    code, _out, err = _run(capsys, str(ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "config file not found" in err


def test_invalid_config_json_is_unmeasured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _dense_checkpoint(tmp_path)
    (ckpt / "config.json").write_text("{not json", encoding="utf-8")
    code, _out, err = _run(capsys, str(ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "config unreadable or not valid JSON" in err


def test_non_object_config_is_unmeasured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _dense_checkpoint(tmp_path)
    (ckpt / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
    code, _out, err = _run(capsys, str(ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "config is not a JSON object" in err


def test_non_checkpoint_directory_is_unmeasured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_ckpt = tmp_path / "not_a_ckpt"
    not_a_ckpt.mkdir()
    code, _out, err = _run(capsys, str(not_a_ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "no recognizable checkpoint layout" in err


def test_zero_tensor_export_is_refused_not_vacuously_cleared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A header declaring zero keys is all([]) raw material; read_metadata
    # raises, and the probe must surface that as "could not measure".
    ckpt = tmp_path / "empty_export"
    ckpt.mkdir()
    _write_header_shard(ckpt / "model.safetensors", [])
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    code, _out, err = _run(capsys, str(ckpt))
    assert code == probe.EXIT_UNMEASURED
    assert "declares 0 tensors" in err


def test_probe_bug_after_measurement_exits_three_not_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The contract's sharpest edge: an exception while rendering must be
    # "could not measure" (3). Plain exit 1 means "a gate reached a blocking
    # verdict", and a cluster pipeline will file the run accordingly.

    def _boom(_inventory: dict[str, Any]) -> None:
        raise RuntimeError("synthetic probe bug")

    monkeypatch.setattr(probe, "_print_inventory", _boom)
    code, _out, err = _run(capsys, str(_dense_checkpoint(tmp_path)))
    assert code == probe.EXIT_UNMEASURED
    assert "a probe bug is not a checkpoint verdict" in err


def test_inject_alias_of_one_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # N=1 would "alias" one tensor onto itself — a control-shaped no-op. It is
    # refused as a usage error (argparse's 2), not measured as anything.
    ckpt = _moe_checkpoint(tmp_path, aliased=False)
    with pytest.raises(SystemExit) as excinfo:
        probe.main([str(ckpt), "--inject-alias", "1"])
    assert excinfo.value.code == 2
    capsys.readouterr()


def test_config_can_never_supply_the_completeness_denominator() -> None:
    # This locks the probe's own rule: an HF config states hyperparameters, not
    # tensor FQNs. The day this fails, someone has taught derive_declared to
    # guess the declared tensor set — the fabricated-denominator defect the
    # probe exists to refuse, and the only road to a false CLEAR.
    declared = probe.derive_declared({"num_experts": 4, "num_moe_layers": 1})
    assert declared["declared_fqns"] is None
    assert declared["expected_expert_bytes"] is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (8, 8),
        (8.0, 8),  # integral float: used, and the coercion is written in notes
        (7.5, None),  # truncating 7.5 to 7 mints a count the config never stated
        ("8", None),  # a string that converts cleanly is still not a declaration
        (True, None),
        (None, None),
        (-2, None),  # a negative count is not a denominator
        (float("inf"), None),  # int(inf) raises OverflowError; must be absent quietly
    ],
)
def test_scoped_int_accepts_only_stated_json_integers(raw: Any, expected: int | None) -> None:
    notes: list[str] = []
    value, _basis = probe._scoped_int({"num_experts": raw}, ("num_experts",), "num_experts", notes)
    assert value == expected
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        assert notes == []
    else:
        # Every non-plain-integer decision is written down. A denominator with
        # no visible provenance is how the probe would start telling stories.
        assert notes != []


def test_scoped_int_prefers_text_config_scope() -> None:
    # Gemma-style nesting: the LM scope is where the expert count lives, so it
    # outranks a multimodal top-level value.
    config = {"num_experts": 2, "text_config": {"num_experts": 4}}
    value, basis = probe._scoped_int(config, ("num_experts",), "num_experts", [])
    assert value == 4
    assert basis is not None and basis.startswith("text_config.")


def test_malformed_expert_count_never_mints_a_denominator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # End-to-end for the _scoped_int contract and the two-source mint rule
    # together. `num_experts: 7.5` is not a declaration: it must not become a
    # usable 7 (the truncation this test was named for), and -- the half the
    # retired `== 0` assertion could not see -- it must not become ANY OTHER
    # denominator either. 0 IS a denominator: an affirmative dense declaration
    # that shrinks the expert gates' examined population to nothing. On this
    # artifact that shrink is live-fire: the shard genuinely holds 8 expert
    # tensors (4 experts x 2 linears, inventory-proven in (4) below), so a
    # minted 0 would have told the aliasing detector "dense model, nothing to
    # examine" over 8 real expert weights. Under the pre-contract probe this
    # exact run produced num_experts=0 and the expert gates SKIPped it as
    # NOT_APPLICABLE; the old assertion pinned that skip-shaped outcome while
    # its own name said "never mints".
    #
    # On the exit code, stated rather than implied: it blocks before AND after
    # this repair, because declared_fqns is underivable from an HF config by
    # design and save_complete is VACUOUS on every run today. The exit code
    # therefore cannot distinguish "abstained" from "minted dense" -- which is
    # exactly why every load-bearing assertion below sits below exit altitude.
    ckpt = _moe_checkpoint(tmp_path, aliased=False)
    (ckpt / "config.json").write_text(
        json.dumps({"num_experts": 7.5, "num_moe_layers": 1}), encoding="utf-8"
    )
    code, report, out = _report(tmp_path, capsys, ckpt)
    declared = report["declared"]
    # (1) No usable denominator was derived from 7.5, and none was substituted:
    # None is the only value that is not a denominator. `== 0` could never say
    # this sentence; pre-contract this line is where the test fails.
    assert declared["num_experts"] is None
    # (2) The malformed input stayed visible rather than being silently
    # dropped: a denominator with no visible provenance is how a probe starts
    # telling stories. (True pre-contract as well; kept so (1) never reads as
    # "the value was lost".)
    assert any("not an integral number" in note for note in declared["notes"])
    # (3) The artifact half of the corroboration ran and named its count: the
    # census is the denominator this refusal was measured against (doctrine 2).
    assert any("expert-family census: 8" in note for note in declared["notes"])
    # (4) Inventory ground truth for that census: 8 expert tensors plus the
    # embedding = 9 real tensors, so this run had real expert units to examine.
    assert report["inventory"]["real_tensors"] == 9
    # (5) The abstention is stated in contract vocabulary, in the field's own
    # provenance slot, with the consequence named -- not inferred from silence.
    basis = declared["basis"]["num_experts"]
    assert "UNKNOWN" in basis
    assert "gates block" in basis
    # (6) Behavioral tripwire for the substituted-denominator half: a SKIP
    # from the distinctness gate is reachable only via NOT_APPLICABLE, which is
    # reachable only via a dense declaration this config never made. Under the
    # mint-0 contract this run rendered exactly that SKIP over 8 live experts.
    assert _verdict_of(report, "checkpoint.expert_distinctness") != Verdict.SKIP.value
    # (7) Exit altitude, asserted and then disclaimed in the header comment:
    # blocked, owned by completeness -- before and after alike.
    assert code == probe.EXIT_BLOCKED
    assert report["exit_code"] == code
    assert f"checkpoint.save_complete={Verdict.VACUOUS.value}" in out


# ---------------------------------------------------------------------------
# The mint rule's four decision classes, pinned directly.
#
# derive_declared now decides between: (a) absence -> None, (b) an affirmative
# dense statement with no artifact census -> None (one source is an assertion,
# two independent sources are evidence), (c) affirmative statement +
# corroborating census of 0 -> an earned 0, (d) contradiction -> None, stated.
# The two repaired tests above hold (a). The tests below hold (b), (c) and (d)
# -- including, per doctrine 3, the rule's own MUST_PASS: (c) is the only test
# in this file that requires derive_declared to produce a 0 at all, so without
# it a derive_declared returning None unconditionally would leave this entire
# suite green. Configs that need the discriminator key reference it through
# the probe's own import rather than a retyped literal: the key's NAME is part
# of the emitter/probe corroboration contract, and a test that hardcodes it
# would keep passing over a drifted key while claiming to exercise it.
# ---------------------------------------------------------------------------


def test_affirmative_dense_statement_without_census_stays_unknown() -> None:
    # Class (b): both affirmative shapes -- the discriminator at false and the
    # explicit zero count -- must stay None when no census is supplied, with
    # the basis naming WHAT was missing. Fail-before: the pre-contract
    # derive_declared minted 0 from the mere absence of a count key and read
    # no discriminator, so it returned 0 for BOTH of these configs; `is None`
    # fails against it without touching any post-contract machinery.
    for config in ({probe._ENABLE_MOE_BLOCK_KEY: False}, {"num_experts": 0}):
        declared = probe.derive_declared(config)
        assert declared["num_experts"] is None
        basis = declared["basis"]["num_experts"]
        assert "UNKNOWN" in basis
        assert "census" in basis  # the refusal names its missing second source


def test_corroborated_dense_declaration_mints_zero() -> None:
    # Class (c), the MUST_PASS half of the rule: an affirmative statement AND a
    # measured census of 0 yield the earned 0, with the two-source provenance
    # stated in the basis. Fail-before: the pre-contract signature had no
    # expert_family_census parameter at all, so both calls raise TypeError
    # there -- the corroboration machinery this test proves did not exist.
    for config in ({probe._ENABLE_MOE_BLOCK_KEY: False}, {"num_experts": 0}):
        declared = probe.derive_declared(config, expert_family_census=0)
        assert declared["num_experts"] == 0
        assert "corroborated" in declared["basis"]["num_experts"]


def test_affirmatively_dense_checkpoint_earns_the_expert_skip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Class (c) end-to-end: the same artifact and helper as the dense test
    # above, but a config that AFFIRMATIVELY declares dense. The probe measures
    # census 0 over the real shard, both sources agree, and the earned 0 lets
    # the distinctness gate SKIP NOT_APPLICABLE -- the very verdict letter the
    # malformed-count test forbids for a config that never declared dense.
    # Provenance, not the letter, is what changed, so the census note leads:
    # without it this run cannot tell a corroborated 0 from a laundered one.
    # Completeness still owns the exit: an earned dense declaration is not a
    # declared tensor list, so this run must not -- and does not -- clear.
    ckpt = _dense_checkpoint(tmp_path)
    (ckpt / "config.json").write_text(
        json.dumps({probe._ENABLE_MOE_BLOCK_KEY: False}), encoding="utf-8"
    )
    code, report, out = _report(tmp_path, capsys, ckpt)
    assert any("expert-family census: 0" in note for note in report["declared"]["notes"])
    assert report["declared"]["num_experts"] == 0
    assert "corroborated" in report["declared"]["basis"]["num_experts"]
    assert _verdict_of(report, "checkpoint.expert_distinctness") == Verdict.SKIP.value
    assert code == probe.EXIT_BLOCKED
    assert f"checkpoint.save_complete={Verdict.VACUOUS.value}" in out
    assert report["exit_code"] == code
    assert report["framework_invariant_breaches"] == []


def test_dense_moe_contradictions_stay_unknown_and_say_so() -> None:
    # Class (d): the two sides disagree, and in every shape the answer is None
    # -- stated, never silently adjudicated in the config's favor.
    #
    # Shape 1: the config contradicts ITSELF (explicit zero count beside an
    # MoE-affirming discriminator). Fail-before: pre-contract code read only
    # count keys, knew no discriminator, and honored the explicit 0 -- so it
    # returned 0 and `is None` fails, with no census machinery involved.
    declared = probe.derive_declared({"num_experts": 0, probe._ENABLE_MOE_BLOCK_KEY: True})
    assert declared["num_experts"] is None
    assert "contradicts itself" in declared["basis"]["num_experts"]
    # Shapes 2 and 3: the config declares dense (either affirmative shape) but
    # the measured artifact holds expert-family tensors. Fail-before: both
    # calls raise TypeError on the pre-contract signature, which had no
    # expert_family_census parameter to contradict against.
    declared = probe.derive_declared(
        {probe._ENABLE_MOE_BLOCK_KEY: False},
        expert_family_census=3,
        expert_family_sample=("model.layers.0.mlp.experts.0.w",),
    )
    assert declared["num_experts"] is None
    assert "CONTRADICTION" in declared["basis"]["num_experts"]
    declared = probe.derive_declared({"num_experts": 0}, expert_family_census=3)
    assert declared["num_experts"] is None
    assert "CONTRADICTION" in declared["basis"]["num_experts"]
