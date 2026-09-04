"""End-to-end: MoE integration, the dense bridge, and the LG3 composite denominator.

Split verbatim from tests/test_live_save_gate.py; shared helpers live in
tests/_live_save_gate_fixtures.py. No line here differs from the original.
"""

from __future__ import annotations

from _live_save_gate_fixtures import (
    MOE_CFG,
    Verdict,
    _control_by_prefix,
    _dense_full_tensors,
    _gr,
    _make_base,
    _materialize_artifact,
    _moe_full_tensors,
    _probe_declared_or_calibrate,
    _stacked_moe_full_tensors,
    _write_cfg,
    lsg,
)


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
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
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
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
        assert d0.exit_code == 0, (  # fixture-drift guard, hunt-file style
            f"fixture drifted: healthy MoE must be CLEAR pre-patch-state "
            f"({d0.blocking_reasons}); calibrate the fixture, never the assertion"
        )
        monkeypatch.setattr(
            lsg.ExpertDistinctnessGate,
            "run",
            lambda self, c: _gr(Verdict.FAIL, self.id, "pre-existing"),
        )
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
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
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
        assert d.exit_code == 1
        alias = _control_by_prefix(d, "alias")
        assert alias["status"] == "inconclusive"
        assert alias["confounded"] is False  # clean baseline; the malfunction is the story
        assert any("INCONCLUSIVE" in r for r in d.blocking_reasons)


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
        estate_cfg = {
            "model_type": "calibration-estate-dense",
            "text_config": {
                "num_hidden_layers": 6,
                "hidden_size": 8,
                "enable_moe_block": False,
                "num_experts": None,
            },
        }
        base = _make_base(tmp_path, _dense_full_tensors(), estate_cfg)
        ckpt = _materialize_artifact(tmp_path, _dense_full_tensors())
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        assert d.report["inventory"]["real_tensors"] == 12
        assert d.report["inventory"]["base_tensors"] == 12
        assert "enable_moe_block=false" in d.declared_basis["num_experts"]
        assert "corroborated" in d.declared_basis["num_experts"]
        assert any("expert-family census: 0 of 12" in n for n in d.declared_basis["notes"])
        by_gate = {g["gate"]: g for g in d.gate_results}
        assert by_gate["checkpoint.expert_distinctness"]["verdict"] == "SKIP"
        assert by_gate["checkpoint.expert_distinctness"]["abstention"] == "not_applicable"
        assert _control_by_prefix(d, "drop")["status"] == "fired"
        assert _control_by_prefix(d, "alias")["status"] == "inapplicable"


class TestFirstSaveCompositeDenominator:
    def _ctx(self, tensors, *, declared, experts, layers, expected_bytes):
        return lsg.CheckpointGateContext(
            tensors=tensors,
            declared_fqns=declared,
            num_experts=experts,
            num_moe_layers=layers,
            expected_expert_bytes=expected_bytes,
            origin="test://synthetic",
        )

    def _dense_ctx(self):
        tms = tuple(
            lsg.TensorMeta(
                fqn=f"model.layers.{i}.attn.weight",
                shape=(4, 4),
                dtype="float32",
                storage_id=f"sd{i}",
                kind="tensor",
            )
            for i in range(4)
        )
        return self._ctx(
            tms, declared=tuple(t.fqn for t in tms), experts=0, layers=0, expected_bytes=0
        )

    def _stacked_ctx(self):
        tms = tuple(
            lsg.TensorMeta(
                fqn=f"model.language_model.layers.{ly}.experts.{proj}",
                shape=(8, *inner),
                dtype="bfloat16",
                storage_id=f"st-{ly}-{proj}",
                kind="tensor",
            )
            for ly in range(2)
            for proj, inner in (("gate_up_proj", (16, 32)), ("down_proj", (32, 16)))
        )
        return self._ctx(
            tms,
            declared=tuple(t.fqn for t in tms),
            experts=8,
            layers=2,
            expected_bytes=sum(t.implied_nbytes for t in tms),
        )

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
            "checkpoint.expert_distinctness",
            "checkpoint.expert_bytes",
        }
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
        assert result.verdict is Verdict.UNDERCOVERED  # fence (both trees)
        assert result.coverage.checked == 2  # fence
        assert result.coverage.expected == 3  # fence
        assert "not established" in result.detail  # fence
        distinct = lsg.ExpertDistinctnessGate().run(self._stacked_ctx())
        assert distinct.verdict is Verdict.SKIP  # fence
        assert distinct.abstention.value == "not_established"  # FAILS-BEFORE (AttributeError)

    def test_unknown_provenance_still_blocks_closed(self):
        """[FAILS-BEFORE on the abstention-is-None lines; the blocking verdict
        is a PASSES-BEFORE fence] No declaration at all -- the shape a
        configless DCP artifact produces through any path that cannot source
        denominators -- must NOT be auto-classified dense: both expert gates
        take the VACUOUS door, the composite FAILS, and no shrink-capable
        abstention kind is minted anywhere on the path."""
        tms = tuple(
            lsg.TensorMeta(
                fqn=f"model.layers.{i}.attn.weight",
                shape=(4, 4),
                dtype="float32",
                storage_id=f"sm{i}",
                kind="tensor",
            )
            for i in range(2)
        )
        ctx = self._ctx(tms, declared=None, experts=None, layers=None, expected_bytes=None)
        result = lsg.FirstSaveGate().run(ctx)
        assert result.verdict is Verdict.FAIL  # fence
        for gate in (lsg.ExpertDistinctnessGate, lsg.ExpertByteVolumeGate):
            res = gate().run(ctx)
            assert res.verdict is Verdict.VACUOUS  # fence
            assert res.abstention is None  # FAILS-BEFORE (AttributeError)

    def test_declared_experts_absent_never_shrinks(self):
        """[FAILS-BEFORE on the abstention-is-None lines; verdicts fence] The
        most dangerous failure mode of denominator-shrinking (the all([])
        shape itself): experts DECLARED and ABSENT is VACUOUS-blocking, not
        inapplicable. If this path ever minted NOT_APPLICABLE, the composite
        would verify 1/1 and pass the incident's twin."""
        tms = tuple(
            lsg.TensorMeta(
                fqn=f"model.layers.{i}.attn.weight",
                shape=(4, 4),
                dtype="float32",
                storage_id=f"se{i}",
                kind="tensor",
            )
            for i in range(2)
        )
        ctx = self._ctx(
            tms, declared=tuple(t.fqn for t in tms), experts=8, layers=2, expected_bytes=1024
        )
        result = lsg.FirstSaveGate().run(ctx)
        assert result.verdict is Verdict.FAIL  # fence
        distinct = lsg.ExpertDistinctnessGate().run(ctx)
        assert distinct.verdict is Verdict.VACUOUS  # fence
        assert distinct.abstention is None  # FAILS-BEFORE (AttributeError)
        bytegate = lsg.ExpertByteVolumeGate().run(ctx)
        assert bytegate.verdict is Verdict.VACUOUS  # fence
        assert bytegate.abstention is None  # FAILS-BEFORE (AttributeError)

    def test_all_inapplicable_is_vacuous_not_a_pass(self, monkeypatch):
        """[FAILS-BEFORE] The zero-applicable corner: if every property were
        somehow declared inapplicable, 0/0 must NOT pass -- Coverage(0, ...,
        expected=0) is vacuous and ok() enforces it. Sub-gates faked via
        type(); monkeypatch restores _subgates. Pre-patch the AbstentionKind
        import inside this test does not exist, which is the fail-before."""
        from foundationscale.gates.core import AbstentionKind, Lifecycle

        def _na_check(self, ctx):
            return self.skip(
                "synthetic: property affirmed absent", kind=AbstentionKind.NOT_APPLICABLE
            )

        fakes = tuple(
            type(
                f"_NA{i}",
                (lsg.Gate,),
                {
                    "id": f"test.synthetic_na_{i}",
                    "description": "synthetic NOT_APPLICABLE abstainer",
                    "events": (Lifecycle.FIRST_SAVE,),
                    "check": _na_check,
                    "controls": lambda self: (),
                },
            )
            for i in range(3)
        )
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
        base = _make_base(tmp_path, _stacked_moe_full_tensors(), MOE_CFG, name="st-base")
        ckpt = _materialize_artifact(tmp_path, _stacked_moe_full_tensors(), name="st-ckpt")
        d = lsg.adjudicate_checkpoint(
            ckpt,
            event="first_save",
            run_kind="full",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}),
        )
        assert d.exit_code == 1  # fence (both trees)
        assert len(d.blocking_reasons) == 1, (  # fence
            f"the composite's not-established leg must be the ONLY reason: {d.blocking_reasons}"
        )
        composite = next(g for g in d.gate_results if g["gate"] == "checkpoint.first_save")
        assert composite["verdict"] == "UNDERCOVERED"  # fence
        assert composite["checked"] == 2 and composite["expected"] == 3  # fence
        assert "not established" in str(composite["detail"])  # fence
        # The fail-before lines:
        # pre-patch: "2/3 first-save properties"
        assert "2/3 applicable" in str(composite["detail"])
        by_gate = {g["gate"]: g for g in d.gate_results}
        # pre-patch: no such key
        assert by_gate["checkpoint.expert_distinctness"]["abstention"] == "not_established"
