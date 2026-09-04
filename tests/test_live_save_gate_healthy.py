"""Healthy full fine-tune: the MUST_PASS backbone every blocking test twins against.

Split verbatim from tests/test_live_save_gate.py; shared helpers live in
tests/_live_save_gate_fixtures.py. No line here differs from the original.
"""

from __future__ import annotations

from _live_save_gate_fixtures import (
    DENSE_CFG,
    LORA_TRAIN,
    MOE_CFG,
    _census_file,
    _control_by_prefix,
    _dense_base_with_ckpt,
    _dense_full_tensors,
    _lora_census_stems,
    _make_base,
    _materialize_artifact,
    _moe_full_tensors,
    _probe_declared_or_calibrate,
    _write_cfg,
    lsg,
)


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
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
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
            ckpt,
            event="first_save",
            run_kind="full",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}),
        )
        assert d.exit_code == 0, (
            f"first_save composite must not red a healthy sharded-MoE artifact; "
            f"got {d.blocking_reasons} -- the first reason names the failing "
            f"gate verbatim; calibrate the fixture, never these assertions"
        )
        # Denominators on the wire (doctrine 2): 34 real tensors (2 q_proj +
        # 8 experts x 2 layers x 2 projections), 4 verdicts (3 always-gates
        # + the FIRST_SAVE composite), composite claiming 3 of 3 verified.
        assert d.report["inventory"]["real_tensors"] == 34
        assert len(d.gate_results) == 4
        composite = next(g for g in d.gate_results if g["gate"] == "checkpoint.first_save")
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
            ckpt,
            event="first_save",
            run_kind="full",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}),
        )
        assert d.exit_code == 0, f"dense first-save must be CLEAR post-LG3: {d.blocking_reasons}"
        composite = next(g for g in d.gate_results if g["gate"] == "checkpoint.first_save")
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
            "checkpoint.expert_distinctness",
            "checkpoint.expert_bytes",
        }


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
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
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
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()),
        )
        assert d.exit_code == 1
        assert any("MODE/lora" in r and "BASE-WEIGHT" in r for r in d.blocking_reasons)
        assert any("MODE/lora" in r and "adapter marker" in r for r in d.blocking_reasons)

    def test_auto_kind_infers_full_for_unmarked_population(self, tmp_path):
        """[PASSES-BEFORE] Red if: the kind = 'lora' if frac >= 0.6 else 'full'
        line is flipped."""
        base, ckpt = _dense_base_with_ckpt(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="auto", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
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
            ckpt, run_kind="full", base_model_dir=base, train_config_path=cfg
        )
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
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
        assert d.exit_code == 0
        assert any("outside the declared set" in n for n in d.declared_basis["notes"])

    def test_strict_extras_flips_that_note_to_blocking(self, tmp_path):
        """[PASSES-BEFORE] The opt-in strict direction. Red if: the
        extras_blocking append is deleted (one line in adjudicate_checkpoint)."""
        tensors = {**_dense_full_tensors(), "layers.9.unexpected.weight": ((8, 8), "F32")}
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, tensors, name="extra")
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="full",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}),
            strict_extras=True,
        )
        assert d.exit_code == 1
        assert any("outside the declared set" in r for r in d.blocking_reasons)
