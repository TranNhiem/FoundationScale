"""Controls: statuses, attribution, floors, and the probe import fallback's degradation.

Split verbatim from tests/test_live_save_gate.py; shared helpers live in
tests/_live_save_gate_fixtures.py. No line here differs from the original.
"""

from __future__ import annotations

from _live_save_gate_fixtures import (
    Coverage,
    GateResult,
    Verdict,
    _census_file,
    _control_by_prefix,
    _gr,
    _healthy_lora,
    lsg,
)


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
            [f"layers.{i}.self_attn.{w}" for i in range(6) for w in ("q_proj", "v_proj")],
        )

    def test_empty_control_sweep_blocks_via_any_fired_floor(self, tmp_path):
        """[PASSES-BEFORE] The sweep-level all([]): controls=() must not pass.
        Red if: the `if not any_fired:` block is deleted.
        fix45: the lora vehicle now feeds the honest #78 census (see
        _floor_census); assertions and red-makers unchanged."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            controls=(),
            adapter_modules=self._floor_census(tmp_path),
        )
        assert d.exit_code == 1
        assert any("no MUST_FIRE control fired" in r for r in d.blocking_reasons)

    def test_unknown_control_name_blocks_as_unconstructable(self, tmp_path):
        """[PASSES-BEFORE] Red if: the _CONTROL_BUILDERS.get guard is replaced
        by direct indexing wrapped in try/except-pass.
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            controls=("telemetry",),
            adapter_modules=self._floor_census(tmp_path),
        )
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
            lsg.SaveCompletenessGate,
            "run",
            lambda self, c: _gr(Verdict.PASS, lsg.SaveCompletenessGate.id, "quiet fake"),
        )
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=self._floor_census(tmp_path),
        )
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
        monkeypatch.setitem(
            lsg._CONTROL_BUILDERS,
            "x",
            lambda ctx, bl: {
                "control": "x",
                "status": "inconclusive",
                "inconclusive_reason": "synthesized",
            },
        )
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            controls=("x",),
            adapter_modules=self._floor_census(tmp_path),
        )
        assert d.exit_code == 1
        assert any("INCONCLUSIVE" in r and "synthesized" in r for r in d.blocking_reasons)

    def test_skipped_only_lands_on_the_floor_not_on_silence(self, tmp_path, monkeypatch):
        """[PASSES-BEFORE] 'skipped' (probe vocabulary for inapplicable) is
        recorded-only, and the any_fired floor still bites. Red if: the floor
        is deleted.
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        monkeypatch.setitem(
            lsg._CONTROL_BUILDERS, "x", lambda ctx, bl: {"control": "x", "status": "skipped"}
        )
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            controls=("x",),
            adapter_modules=self._floor_census(tmp_path),
        )
        assert d.exit_code == 1
        assert any("no MUST_FIRE control fired" in r for r in d.blocking_reasons)

    def test_unrecognized_status_blocks_and_is_named(self, tmp_path, monkeypatch):
        """[FAILS-BEFORE -- Edit 4] A status this loop cannot read must not be
        read as inapplicable: the reason names 'unrecognized status'. (The
        exit code was already nonzero via the floor; the NAMING is what is
        new, and it is what the library caller matches on.)
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""
        monkeypatch.setitem(
            lsg._CONTROL_BUILDERS, "x", lambda ctx, bl: {"control": "x", "status": "sparkly"}
        )
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            controls=("x",),
            adapter_modules=self._floor_census(tmp_path),
        )
        assert d.exit_code == 1
        assert any("unrecognized status" in r and "sparkly" in r for r in d.blocking_reasons)

    def test_framework_tripwire_pass_over_zero_checked_blocks(self, tmp_path, monkeypatch):
        """[PASSES-BEFORE] Red if: the `if r.verdict is Verdict.PASS and
        r.coverage.checked == 0` loop is deleted.
        fix45: vehicle census added (see _floor_census); assertions and
        red-makers unchanged."""

        class _VacuousOk:
            def run(self, ctx):
                return GateResult(
                    gate_id="test.vacuous",
                    verdict=Verdict.PASS,
                    coverage=Coverage(0, "units"),
                    detail="all([])",
                )

        monkeypatch.setattr(lsg, "_ALWAYS_GATES", (_VacuousOk,))
        base, ckpt, cfg = _healthy_lora(tmp_path)
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=self._floor_census(tmp_path),
        )
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
            fqn=fqn, shape=(16, 4), dtype="float32", storage_id=f"store://{fqn}", kind="tensor"
        )
        return lsg.CheckpointGateContext(
            tensors=(tm("L0.moe.experts.weight"), tm("L1.moe.experts.weight")),
            declared_fqns=("L0.moe.experts.weight", "L1.moe.experts.weight"),
            num_experts=16,
            num_moe_layers=2,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )

    def _pin_stacked(self, monkeypatch):
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: list(ts))
        monkeypatch.setattr(lsg, "_split_expert_layouts", lambda c: ({}, list(c), []))
        monkeypatch.setattr(
            lsg, "_layer_normalized_stem", lambda f: f.split(".", 1)[1] if "." in f else f
        )

    def test_confounded_baseline_is_inconclusive_not_fired(self, monkeypatch):
        """[FAILS-BEFORE -- Edits 1/2] mirror of the probe's
        test_blocking_baseline_is_inconclusive_not_fired, now for THIS tool."""
        self._pin_stacked(monkeypatch)
        monkeypatch.setattr(
            lsg.ExpertDistinctnessGate, "run", lambda self, c: _gr(Verdict.FAIL, self.id)
        )
        out = lsg.control_alias(
            self._ctx(),
            {lsg.ExpertDistinctnessGate.id: _gr(Verdict.FAIL, lsg.ExpertDistinctnessGate.id)},
        )
        assert out["status"] == "inconclusive" and out["confounded"] is True
        assert "baseline" in out["inconclusive_reason"]

    def test_error_verdict_is_not_credited(self, monkeypatch):
        """[FAILS-BEFORE -- Edits 1/2] A crash on the injected copy is a
        malfunction, not a detection: inconclusive, confounded False. The old
        code credited res.blocking -- i.e., it credited the crash."""
        self._pin_stacked(monkeypatch)
        monkeypatch.setattr(
            lsg.ExpertDistinctnessGate, "run", lambda self, c: _gr(Verdict.ERROR, self.id)
        )
        out = lsg.control_alias(
            self._ctx(),
            {lsg.ExpertDistinctnessGate.id: _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)},
        )
        assert out["status"] == "inconclusive" and out["confounded"] is False
        assert out["verdict"] == "ERROR"

    def test_clean_baseline_and_fail_fires(self, monkeypatch):
        """[FAILS-BEFORE -- semantics PASSES-BEFORE-equivalent; red-maker if
        the patch rots: flip `res.verdict is Verdict.FAIL` to `res.blocking` in
        _attributed_status, and test_error_verdict_is_not_credited goes red
        with this one still green -- the pair is the detector's two controls.]"""
        self._pin_stacked(monkeypatch)
        monkeypatch.setattr(
            lsg.ExpertDistinctnessGate, "run", lambda self, c: _gr(Verdict.FAIL, self.id)
        )
        out = lsg.control_alias(
            self._ctx(),
            {lsg.ExpertDistinctnessGate.id: _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)},
        )
        assert out["status"] == "fired" and out["confounded"] is False
        assert out["baseline_verdict"] == "PASS"

    def test_acceptance_stays_not_fired(self, monkeypatch):
        """[FAILS-BEFORE -- arity] True negative naming preserved."""
        self._pin_stacked(monkeypatch)
        monkeypatch.setattr(
            lsg.ExpertDistinctnessGate, "run", lambda self, c: _gr(Verdict.PASS, self.id)
        )
        out = lsg.control_alias(
            self._ctx(),
            {lsg.ExpertDistinctnessGate.id: _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)},
        )
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
        tms = tuple(
            lsg.TensorMeta(
                fqn=f, shape=(4, 4), dtype="float32", storage_id=f"store://{f}", kind="tensor"
            )
            for f in fqns
        )
        return lsg.CheckpointGateContext(
            tensors=tms,
            declared_fqns=tuple(fqns),
            num_experts=8,
            num_moe_layers=1,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )

    def _run_control(self, ctx, n=4):
        assert lsg._probe_alias_control is not None, (
            "probe unimportable -- the sys.path insertion in _load_gate "
            "regressed; fix the loader, never this guard"
        )
        baseline = lsg.ExpertDistinctnessGate().run(ctx)
        assert not baseline.blocking, (
            f"fixture drifted: 16-of-16 healthy experts must PASS pre-"
            f"injection ({baseline.detail}) -- calibrate the fixture, never "
            f"the assertion"
        )
        return lsg._probe_alias_control(ctx, n, baseline=baseline)

    def test_probe_alias_control_fires_on_global_spelling(self):
        """[FAILS-BEFORE -- Edits 1/2] The spelling the hand-rolled loop read
        as 'fused': pre-patch this returns skipped; post-patch it aliases 4
        members of a same-stem group onto one storage and FIRES."""
        fqns = [
            f"m.layers.0.experts.{i}.{p}.weight"
            for p in ("linear_fc1", "linear_fc2")
            for i in range(8)
        ]
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
        fqns = [
            f"m.layers.0.mlp.experts.experts.linear_fc{p}.weight{i}"
            for p in (1, 2)
            for i in range(8)
        ]
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
        tms = tuple(
            lsg.TensorMeta(
                fqn=f"m.layers.0.experts.{i}.w.weight",
                shape=(4, 4),
                dtype="float32",
                storage_id=f"s{i}",
                kind="tensor",
            )
            for i in range(8)
        )
        return lsg.CheckpointGateContext(
            tensors=tms,
            declared_fqns=tuple(t.fqn for t in tms),
            num_experts=8,
            num_moe_layers=1,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )

    def test_routed_in_then_skipped_rewrites_to_unconstructable(self, monkeypatch):
        """[FAILS-BEFORE -- Edit 4] Pre-patch the probe's 'skipped' passed
        through untouched and the consume loop filed it as recorded-only --
        the Defect-A fall-through replayed as a unit. Post-patch: blocking
        'unconstructable', probe's own status preserved, original reason
        embedded (auditable divergence, not a laundered one)."""
        monkeypatch.setattr(
            lsg,
            "_probe_alias_control",
            lambda ctx, n, baseline=None: {
                "status": "skipped",
                "reason": "synthesized: control declines after routing",
            },
        )
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
            lsg,
            "_probe_alias_control",
            lambda ctx, n, baseline=None: {
                "status": "fired",
                "confounded": False,
                "verdict": "FAIL",
                "detail": "synthesized fire",
                "baseline_verdict": "PASS",
                "aliased_fqns": [],
                "inconclusive_reason": "",
            },
        )
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
            tensors=(),
            declared_fqns=None,
            num_experts=0,
            num_moe_layers=0,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )
        assert lsg.control_underfill(ctx, {})["status"] == "inapplicable"

    def test_underfill_unconstructable_below_eight_experts(self, monkeypatch):
        """[PASSES-BEFORE] The incident ratio cannot be reproduced below 8
        without degenerating to zero. Red if: the `< 8` guard is deleted --
        the control would then inject a same-shaped tensor and read not_fired."""
        tm = lsg.TensorMeta(
            fqn="L0.moe.experts.weight",
            shape=(4, 4),
            dtype="float32",
            storage_id="s",
            kind="tensor",
        )
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: [tm])
        monkeypatch.setattr(lsg, "_split_expert_layouts", lambda c: ({}, [tm], []))
        ctx = lsg.CheckpointGateContext(
            tensors=(tm,),
            declared_fqns=(tm.fqn,),
            num_experts=4,
            num_moe_layers=1,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )
        assert lsg.control_underfill(ctx, {})["status"] == "unconstructable"

    def test_underfill_error_is_not_credited(self, monkeypatch):
        """[FAILS-BEFORE -- Edits 1/3] Same FAIL-only rule as the alias leg."""
        tm = lsg.TensorMeta(
            fqn="L0.moe.experts.weight",
            shape=(16, 4),
            dtype="float32",
            storage_id="s",
            kind="tensor",
        )
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: [tm])
        monkeypatch.setattr(lsg, "_split_expert_layouts", lambda c: ({}, [tm], []))
        monkeypatch.setattr(
            lsg.ExpertDistinctnessGate, "run", lambda self, c: _gr(Verdict.ERROR, self.id)
        )
        ctx = lsg.CheckpointGateContext(
            tensors=(tm,),
            declared_fqns=(tm.fqn,),
            num_experts=16,
            num_moe_layers=1,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )
        out = lsg.control_underfill(
            ctx, {lsg.ExpertDistinctnessGate.id: _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)}
        )
        assert out["status"] == "inconclusive" and out["verdict"] == "ERROR"

    def test_drop_unconstructable_without_declared_set(self):
        """[PASSES-BEFORE] Red if: the `if not ctx.declared_fqns` guard is
        deleted (the unexercised-detector reason would change class)."""
        ctx = lsg.CheckpointGateContext(
            tensors=(),
            declared_fqns=None,
            num_experts=0,
            num_moe_layers=0,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )
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
                return GateResult(
                    gate_id=self.id,
                    verdict=Verdict.FAIL,
                    coverage=Coverage(len(present), "tensors", expected=len(c.declared_fqns or ())),
                    detail="missing",
                    evidence={"missing": missing},
                )
            return _gr(Verdict.PASS, self.id)

        monkeypatch.setattr(lsg.SaveCompletenessGate, "run", honest)
        tms = tuple(
            lsg.TensorMeta(
                fqn=f"m.{i}.weight",
                shape=(2, 2),
                dtype="float32",
                storage_id=f"s{i}",
                kind="tensor",
            )
            for i in range(4)
        )
        declared = tuple(t.fqn for t in tms)
        ctx = lsg.CheckpointGateContext(
            tensors=tms,
            declared_fqns=declared,
            num_experts=0,
            num_moe_layers=0,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )
        out = lsg.control_drop(ctx, {}, n=2)
        assert out["status"] == "fired"
        assert out["named_dropped"] and set(out["named_dropped"]) <= set(out["dropped"])

    def test_drop_receives_no_credit_for_a_block_naming_no_injected_loss(self, monkeypatch):
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
            lsg.TensorMeta(
                fqn=f"m.{i}.weight",
                shape=(2, 2),
                dtype="float32",
                storage_id=f"s{i}",
                kind="tensor",
            )
            for i in range(4)
        )
        ghost = "ghost.preexisting.weight"  # declared, never written: the baseline hole
        ctx = lsg.CheckpointGateContext(
            tensors=present,
            declared_fqns=tuple(t.fqn for t in present) + (ghost,),
            num_experts=0,
            num_moe_layers=0,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )

        def blocks_for_the_preexisting_hole(self, c):
            # The deployed gate's documented evidence shape (an enumerated
            # missing list, as in the MUST_PASS fence two doors up) with the
            # one difference under test: the named loss is NOT injected --
            # this block stood before the drop ran.
            return GateResult(
                gate_id=self.id,
                verdict=Verdict.FAIL,
                coverage=Coverage(len(c.tensors), "tensors", expected=len(c.declared_fqns or ())),
                detail="declared tensor never written: ghost.preexisting.weight",
                evidence={"missing": [ghost]},
            )

        monkeypatch.setattr(lsg.SaveCompletenessGate, "run", blocks_for_the_preexisting_hole)
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
        tms = tuple(
            lsg.TensorMeta(
                fqn=f"m.{i}.weight",
                shape=(2, 2),
                dtype="float32",
                storage_id=f"s{i}",
                kind="tensor",
            )
            for i in range(4)
        )
        ctx = lsg.CheckpointGateContext(
            tensors=tms,
            declared_fqns=tuple(t.fqn for t in tms),
            num_experts=0,
            num_moe_layers=0,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )
        monkeypatch.setattr(
            lsg.SaveCompletenessGate,
            "run",
            lambda self, c: _gr(
                Verdict.ERROR, self.id, "RuntimeError: synthesized crash on injection"
            ),
        )
        out = lsg.control_drop(ctx, {}, n=2)
        assert out["status"] == "not_fired"
        assert out["verdict"] == "ERROR"


class TestProbeImportDegradation:
    """Group (1) controls -- now a single leg, and the shrink is the finding.

    This class used to hold four. Three of them pinned a DEGRADED state: the
    probe helpers were reached through a try/except ImportError ladder, so the
    two slots were ``Optional``, and the consumers carried ``is None`` guards
    that answered ``unconstructable`` / refused-to-paraphrase when the import
    had failed. The three legs were, respectively, the Optional declaration
    read off ``__annotations__``, the null-alias-control degrade, and the
    null-derive-helper refusal.

    Finding #219 established that the ladder's fallback arm was reaching into
    ``tools/``, which ``[tool.setuptools.packages.find] where = ["src"]`` does
    not distribute -- so on a clean install the import ALWAYS failed and the
    degraded state was not a rare fallback but the only state a wheel could
    reach. The fix moved the machinery into ``foundationscale.gates.probe``.
    The slots are now plain top-level imports: they cannot be None, the guards
    are unreachable, and an unreachable DECLARED state is itself a defect
    (#200). Guards and tests went together.

    They are deleted rather than kept green by monkeypatching, because a test
    that sets a slot to None to observe a refusal would be measuring a state
    the module no longer has -- it would report coverage of a path that cannot
    occur. What replaces them is narrower and stronger: the surviving leg below
    asserts the slots are the IN-PACKAGE definitions by identity, and
    ``tests/test_gates_probe_packaging.py`` proves the decision path can derive
    a declared block with only ``src/`` on ``sys.path``, with a MUST_FIRE half
    that doctors the package to confirm the check can still go red.
    """

    def test_normal_import_paths_leave_the_real_probe_control_wired(self):
        """[PASSES-BEFORE and PASSES-AFTER -- MUST_PASS fence for group (1)]
        The slots still land on the probe's REAL helpers after the
        restructure, and the real sharded alias control still runs through
        them to 'fired' on the global-spelling geometry that
        TestProbeAliasSpellings already proves green against the live gate
        and probe. Red-makers: a rewiring that leaves a slot unbound or
        bound to a paraphrase (the first asserts), or one that stops routing
        sharded work to the probe-verbatim control (the last two).

        There used to be two import paths and an ``_PROBE_IMPORT_ERROR``
        sentinel, and this leg asserted the sentinel was None -- which only
        ever said "the fallback ladder found the helpers SOMEWHERE". Finding
        #219 showed that somewhere was ``tools/``, which is not distributed,
        so the assertion was green in a checkout and unreachable in a wheel.
        The ladder is gone. The replacement asserts identity against
        ``foundationscale.gates.probe``, which is a strictly narrower claim:
        not merely bound, but bound to the in-package definition."""
        from foundationscale.gates import probe as fs_probe

        assert lsg._probe_derive_declared is fs_probe.derive_declared
        assert lsg._probe_alias_control is fs_probe.run_alias_control
        fqns = [
            f"m.layers.0.experts.{i}.{p}.weight"
            for p in ("linear_fc1", "linear_fc2")
            for i in range(8)
        ]
        tms = tuple(
            lsg.TensorMeta(
                fqn=f, shape=(4, 4), dtype="float32", storage_id=f"store://{f}", kind="tensor"
            )
            for f in fqns
        )
        ctx = lsg.CheckpointGateContext(
            tensors=tms,
            declared_fqns=tuple(fqns),
            num_experts=8,
            num_moe_layers=1,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )
        baseline = lsg.ExpertDistinctnessGate().run(ctx)
        assert not baseline.blocking, (
            f"fixture drifted: 16-of-16 healthy experts must PASS "
            f"pre-injection ({baseline.detail}) -- calibrate the fixture, "
            f"never the assertion"
        )
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
        return lsg.TensorMeta(
            fqn=fqn, shape=shape, dtype="float32", storage_id=f"store://{fqn}", kind="tensor"
        )

    def _ctx(self, tms):
        return lsg.CheckpointGateContext(
            tensors=tuple(tms),
            declared_fqns=tuple(t.fqn for t in tms),
            num_experts=16,
            num_moe_layers=1,
            expected_expert_bytes=None,
            origin="test://synthetic",
        )

    def _route_stacked(self, monkeypatch, tms):
        monkeypatch.setattr(lsg, "_expert_weight_candidates", lambda ts: list(tms))
        monkeypatch.setattr(lsg, "_split_expert_layouts", lambda c: ({}, list(c), []))

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
        tms = [self._tm("L0.moe.experts.weight"), self._tm("L1.moe.experts.weight")]
        self._route_stacked(monkeypatch, tms)
        monkeypatch.setattr(
            lsg.TensorMeta, "implied_nbytes", property(lambda self: None), raising=False
        )
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
        tms = [
            self._tm("L0.moe.experts.weight", shape=(16, 1024)),
            self._tm("L1.moe.experts.weight"),
        ]
        self._route_stacked(monkeypatch, tms)
        monkeypatch.setattr(
            lsg.TensorMeta,
            "implied_nbytes",
            property(lambda self: None if self.fqn.startswith("L0") else 64),
            raising=False,
        )
        monkeypatch.setattr(
            lsg.ExpertDistinctnessGate,
            "run",
            lambda self, c: _gr(Verdict.FAIL, self.id, "synthesized fire"),
        )
        baseline = {lsg.ExpertDistinctnessGate.id: _gr(Verdict.PASS, lsg.ExpertDistinctnessGate.id)}
        out = lsg.control_underfill(self._ctx(tms), baseline)
        assert out["status"] == "fired"
        assert out["tensor"] == tms[1].fqn
        assert out["candidates"].startswith("1 of 2")
        assert "64 bytes" in out["candidates"]
