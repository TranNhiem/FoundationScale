"""LoRA discrimination in both directions, and the structural binding behind it.

Split verbatim from tests/test_live_save_gate.py; shared helpers live in
tests/_live_save_gate_fixtures.py. No line here differs from the original.
"""

from __future__ import annotations

from _live_save_gate_fixtures import (
    DENSE_CFG,
    LORA_TRAIN,
    MOE_CFG,
    _census,
    _census_file,
    _control_by_prefix,
    _dense_full_tensors,
    _healthy_lora,
    _lora_census_stems,
    _lora_tensors,
    _make_base,
    _materialize_artifact,
    _moe_full_tensors,
    _probe_declared_or_calibrate,
    _write_cfg,
    lsg,
    pytest,
)


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
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()),
        )
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
            ckpt, run_kind="full", base_model_dir=base, train_config_path=cfg
        )
        assert d.exit_code == 1
        assert "fqn-map" in d.declared_basis["fqns"]  # the remediation is named
        assert any("VACUOUS" in g["verdict"] or g["verdict"] == "VACUOUS" for g in d.gate_results)

    def test_lora_contamination_by_base_weights_blocks(self, tmp_path):
        """[PASSES-BEFORE] Red if: the contaminated append in the lora branch
        of cross_check_population is deleted. fix45-C1 (#78): fed the new
        census demand the honest 12-stem census so adjudication reaches the
        cross-check at all -- the census is scaffolding here; the
        contaminated append remains the sole named red-maker, and the
        verbatim base FQN below is still foreign to the census-derived
        declared set exactly as it was foreign to the old one."""
        tensors = {
            **_lora_tensors(),
            "layers.0.self_attn.q_proj.weight": ((8, 8), "F32"),
        }  # verbatim base FQN
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, tensors, name="lora-dirty")
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()),
        )
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
        cfg = _write_cfg(
            tmp_path,
            {"lora_rank": 4, "lora_targets": ["q_proj", "v_proj"]},
            name="no-kind-key.json",
        )
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="auto",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()),
        )
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
                ckpt,
                run_kind="auto",
                base_model_dir=base,
                train_config_path=_write_cfg(tmp_path, {}, name="empty-cfg.json"),
                adapter_prefix="",
            )
        msg = str(exc_info.value)
        assert msg.startswith("--adapter-modules was not supplied")
        assert lsg._refusal_class(msg) == lsg._REFUSAL_ADAPTER_CENSUS_UNAVAILABLE

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
        cfg = _write_cfg(tmp_path, {"peft_scheme": "lora", "lora_rank": 4}, name="no-targets.json")
        with pytest.raises(lsg.GateUnmeasured) as exc_info:
            lsg.adjudicate_checkpoint(
                ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg, adapter_prefix=""
            )
        msg = str(exc_info.value)
        assert msg.startswith("--adapter-modules was not supplied")
        assert lsg._refusal_class(msg) == lsg._REFUSAL_ADAPTER_CENSUS_UNAVAILABLE

    def test_healthy_lora_over_moe_base_is_adjudicated_in_adapter_scope(self, tmp_path):
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
        base = _make_base(tmp_path, _moe_full_tensors(), MOE_CFG, name="lom-base")
        adapters = {}
        for ly in range(2):
            stem = f"layers.{ly}.self_attn.q_proj"
            adapters[f"{stem}.adapter.linear_in.weight"] = ((4, 8), "F32")
            adapters[f"{stem}.adapter.linear_out.weight"] = ((8, 4), "F32")
        ckpt = _materialize_artifact(tmp_path, adapters, name="lom")
        cfg = _write_cfg(
            tmp_path, {"peft_scheme": "lora", "lora_rank": 4, "lora_targets": ["q_proj"]}
        )
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=_census_file(
                tmp_path, [f"layers.{ly}.self_attn.q_proj" for ly in range(2)]
            ),
        )
        assert d.exit_code == 0, (
            f"healthy LoRA-of-MoE must CLEAR in adapter scope, got "
            f"{d.exit_code}: {d.blocking_reasons} -- under the mutant the "
            f"first reasons are the expert gates blocking on experts this "
            f"adapter was never declared to contain"
        )
        assert "ADAPTER SCOPE" in d.declared_basis["num_experts"]
        inv = d.report["inventory"]
        assert inv["real_tensors"] == 4 and inv["base_tensors"] == 34
        assert "4 adapter tensors" in d.declared_basis["fqns"]
        by_gate = {g["gate"]: g for g in d.gate_results}
        assert by_gate["checkpoint.expert_distinctness"]["verdict"] == "SKIP"
        assert by_gate["checkpoint.expert_distinctness"]["abstention"] == "not_applicable"
        assert by_gate["checkpoint.expert_bytes"]["verdict"] == "SKIP"
        # The mint must also reach the CONTROL layer's view of the context:
        # alias inapplicability carries num_experts=0 only if the exception
        # ran (mutant: the same string reads 8).
        alias = _control_by_prefix(d, "alias")
        assert alias["status"] == "inapplicable"
        assert "num_experts=0" in str(alias.get("reason"))
        assert _control_by_prefix(d, "drop")["status"] == "fired"

    def test_adapter_scope_mint_does_not_apply_when_adapters_target_experts(self, tmp_path):
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
            {"peft_scheme": "lora", "lora_rank": 4, "lora_targets": ["linear_fc1", "linear_fc2"]},
            "test://cfg",
            "lora",
            None,
        )
        base = lsg.BaseModel(
            model_dir=tmp_path,
            config=MOE_CFG,
            tensors={k: (v[0], "float32") for k, v in _moe_full_tensors().items()},
            tensors_source="test://synthetic",
        )
        decl = lsg.derive_declared_block(
            base,
            spec,
            set(),
            "",
            adapter_modules=_census(
                f"layers.{ly}.experts.{e}.{p}"
                for ly in range(2)
                for e in range(8)
                for p in ("linear_fc1", "linear_fc2")
            ),
        )
        assert decl.num_experts == 8
        assert decl.num_moe_layers == 2
        assert "ADAPTER SCOPE" not in decl.experts_basis
        # [new] fix38, re-based #78 (fix45-C1): the retention is a NAMED
        # record carrying the census denominator, not a silent fall-through
        # -- the wire testifies that all 32 census modules (the fixture
        # base's real expert parent stems) were classified expert-family by
        # the gates' own name atoms and found resident. The pre-#78 "2 of 2"
        # counted config targets; post-#78 the census prices modules.
        assert any("ADAPTER SCOPE RETAINS EXPERTS: 32 of 32" in n for n in decl.notes), (
            f"mint retention note missing or miscounted: {decl.notes!r}"
        )
        # [new] fix38 positive control, the leg that makes this fence
        # convicting: the SAME base with a census of non-expert stems only.
        # Pre-fix38 the dead census minted nothing for any non-empty list,
        # so this leg's assertions are red on that tree; green requires the
        # census classifier to have examined both attention stems and found
        # 0 of 2 expert-resident.
        attention_spec = lsg.resolve_train_spec(
            {"peft_scheme": "lora", "lora_rank": 4, "lora_targets": ["q_proj"]},
            "test://cfg",
            "lora",
            None,
        )
        attention_decl = lsg.derive_declared_block(
            base,
            attention_spec,
            set(),
            "",
            adapter_modules=_census(f"layers.{ly}.self_attn.q_proj" for ly in range(2)),
        )
        assert attention_decl.num_experts == 0
        assert attention_decl.num_moe_layers == 0
        assert "ADAPTER SCOPE: 0 of 2 census modules" in attention_decl.experts_basis
        assert "base model's own declaration was 8" in attention_decl.experts_basis

    def test_unknown_targets_over_moe_base_abstain_and_keep_the_base_denominator(self, tmp_path):
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
        base = _make_base(tmp_path, _moe_full_tensors(), MOE_CFG, name="unkm-base")
        adapters = {}
        for ly in range(2):
            stem = f"layers.{ly}.self_attn.q_proj"
            adapters[f"{stem}.adapter.linear_in.weight"] = ((4, 8), "F32")
            adapters[f"{stem}.adapter.linear_out.weight"] = ((8, 4), "F32")
        ckpt = _materialize_artifact(tmp_path, adapters, name="unkm")
        cfg = _write_cfg(
            tmp_path,
            {"peft_scheme": "lora", "lora_rank": 4, "frozen_regex": r"layers\.5\."},
            name="unknown-targets-moe.json",
        )
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=_census_file(
                tmp_path, [f"layers.{ly}.self_attn.q_proj" for ly in range(2)]
            ),
        )
        # The run BLOCKS -- the stated, blocking refusal-to-mint (exit 1),
        # consistent with the low-overlap full-FT precedent; the exit-3
        # refusal is reserved for the no-census state and pinned by the two
        # named refusal conversions in this class.
        assert d.exit_code == 1
        assert "ADAPTER SCOPE" not in d.declared_basis["num_experts"]
        assert any("adapter-scope expert mint refused" in n for n in d.declared_basis["notes"])
        inv = d.report["inventory"]
        assert inv["real_tensors"] == 4 and inv["base_tensors"] == 34
        by_gate = {g["gate"]: g for g in d.gate_results}
        assert by_gate["checkpoint.expert_distinctness"]["verdict"] == "VACUOUS"
        assert by_gate["checkpoint.expert_bytes"]["verdict"] == "VACUOUS"
        assert "8 experts" in str(by_gate["checkpoint.expert_distinctness"]["detail"])
        assert "4 adapter tensors = 2 census modules" in d.declared_basis["fqns"]


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
        tensors = {**_lora_tensors(), "ghost.mod.adapter.linear_in.weight": ((4, 8), "F32")}
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, tensors, name="lora-ghost")
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()),
        )
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
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_modules=_census_file(
                tmp_path, _lora_census_stems(), dims={s: (8, 8) for s in _lora_census_stems()}
            ),
        )
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
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="base_model.model.",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()),
        )
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
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            adapter_marker=r"(?:lora_[AB]|delta_[AB])",
            adapter_suffix_re=r"\.delta_[AB]$",
            adapter_suffixes=(".delta_A", ".delta_B"),
            adapter_modules=_census_file(tmp_path, _lora_census_stems()),
        )
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
        assert any("0 of 31" in r and "vacuous detector" in r for r in d.blocking_reasons)
