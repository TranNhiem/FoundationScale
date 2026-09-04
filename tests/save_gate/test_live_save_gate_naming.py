"""Adapter naming: generator/recognizer agreement, prefix demand, and the --fqn-map remediation.

Split verbatim from tests/test_live_save_gate.py; shared helpers live in
tests/_live_save_gate_fixtures.py. No line here differs from the original.
"""

from __future__ import annotations

from _live_save_gate_fixtures import (
    DENSE_CFG,
    LORA_TRAIN,
    _census,
    _census_file,
    _control_by_prefix,
    _dense_full_tensors,
    _healthy_lora,
    _lora_census_stems,
    _lora_tensors,
    _make_base,
    _materialize_artifact,
    _megatron_named_lora_tensors,
    _renamed_full_tensors,
    _write_cfg,
    json,
    lsg,
    lsg_cli,
    pytest,
)


class TestFqnMap:
    def test_dcp_named_full_blocks_without_map_and_names_flag(self, tmp_path):
        """[PASSES-BEFORE] Red if: the basis sentence naming --fqn-map is
        reworded away (the test pins the operator-visible remediation)."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, _renamed_full_tensors(), name="dcp")
        d = lsg.adjudicate_checkpoint(
            ckpt, run_kind="full", base_model_dir=base, train_config_path=_write_cfg(tmp_path, {})
        )
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
            ckpt,
            run_kind="full",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}),
            fqn_map=map_path,
        )
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        assert "--fqn-map" in d.declared_basis["fqns"]
        assert _control_by_prefix(d, "drop")["status"] == "fired"

    def test_zero_overlap_stale_map_fails_not_vacuous(self, tmp_path):
        """[FAILS-BEFORE -- Edit 5] A stale map shares zero names with the
        artifact: the note must say so and completeness must FAIL, not pass."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, _renamed_full_tensors(), name="dcp")
        map_path = tmp_path / "stale-map.json"
        map_path.write_text(
            json.dumps([f"stale.{k}" for k in _dense_full_tensors()]), encoding="utf-8"
        )
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="full",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, {}),
            fqn_map=map_path,
        )
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
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            fqn_map=map_path,
            adapter_prefix="",
            adapter_modules=_census_file(tmp_path, _lora_census_stems()),
        )
        assert d.exit_code == 0, f"{d.blocking_reasons}"
        assert any("IGNORED" in n for n in d.declared_basis["notes"])


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
        return [f"layers.{i}.self_attn.{w}" for i in range(6) for w in ("q_proj", "v_proj")]

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
        with pytest.raises(lsg.GateUnmeasured, match="adapter naming disagreement") as exc_info:
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

    def test_cli_refusal_is_exactly_three_and_names_the_disagreement(self, tmp_path, capsys):
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
        code = lsg_cli.main(
            [
                str(ckpt),
                "--run-kind",
                "lora",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(cfg),
                "--adapter-prefix",
                "",
                "--adapter-suffix-a",
                ".lora_A.weight",
                "--adapter-suffix-b",
                ".lora_B.weight",
            ]
        )
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
        ckpt = _materialize_artifact(tmp_path, _megatron_named_lora_tensors(), name="mt-lora")
        # fix45/#78: the lora derive now demands an --adapter-modules census
        # (artifact namespace, written outside the judged tree). Names-only
        # is the honest minimum: the shape check then abstains BY NAME, an
        # abstention no assertion here inspects. Assertions unchanged.
        census = _census_file(tmp_path, self._census_stems())
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_suffix_re=r"\.adapter\.linear_(?:in|out)\.weight$",
            adapter_suffixes=(".adapter.linear_in.weight", ".adapter.linear_out.weight"),
            adapter_modules=census,
        )
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
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=_write_cfg(tmp_path, LORA_TRAIN),
            adapter_prefix="",
            adapter_suffix_re=r"\.adapter\.linear_(?:in|out)\.weight$",
            adapter_suffixes=(".adapter.linear_in.weight", ".adapter.linear_out.weight"),
            adapter_modules=census,
        )
        assert d.exit_code == 1, (
            f"optimizer-shaped decoys must still hard-block: {d.blocking_reasons}"
        )
        flagged = [r for r in d.blocking_reasons if "MODE/lora" in r and "adapter marker" in r]
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
        assert any("3 of 27" in r and "optimizer_gate.weight" in r for r in flagged), (
            f"decoys not isolated in reason: {flagged}"
        )
        assert any("7 non-adapter" in r for r in flagged), (
            f"excluded-namespace count missing from reason: {flagged}"
        )

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
        fqns = {f"layers.{i}.self_attn.q_proj.adapter.linear_in.weight" for i in range(4)}
        fqns |= {f"optimizer.state.exp_avg.block{i}.weight" for i in range(11)}
        fqns.add("rng_state")
        kind, basis = lsg._infer_auto_kind(fqns, markers)
        # Post-exclusion the judged pool is 4/4 = 1.00; raw counting would
        # give 4/16 = 0.25 -> "full". The basis string carries BOTH counts so
        # the shrink is reported, not silent (doctrine 2).
        assert kind == "lora", f"save-state namespaces dragged auto-kind to full: {basis}"
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
            raise AssertionError("all-excluded pool must raise GateUnmeasured, not guess a kind")

    def test_calibrated_templates_generate_exact_names_and_shapes(self, tmp_path):
        """[FAILS-BEFORE -- kwarg absent pre-patch] Unit-level MUST_PASS on
        the generator alone: the exact declared FQN set and the positional
        shape convention, against the same in-module BaseModel construction
        TestMoeOverride already uses."""
        assert lsg._probe_derive_declared is not None, (
            "probe import failed -- the sys.path insertion in _load_gate "
            "regressed; fix the loader, never this guard"
        )
        spec = lsg.resolve_train_spec(dict(LORA_TRAIN), "test://cfg", "auto", None)
        base = lsg.BaseModel(
            model_dir=tmp_path,
            config=DENSE_CFG,
            tensors={k: (v[0], "float32") for k, v in _dense_full_tensors().items()},
            tensors_source="test://synthetic",
        )
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
            base,
            spec,
            set(),
            "",
            adapter_suffixes=(".adapter.linear_in.weight", ".adapter.linear_out.weight"),
            adapter_modules=_census(stems, dims={s: (8, 8) for s in stems}),
        )
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
            "regressed; fix the loader, never this guard"
        )
        spec = lsg.resolve_train_spec(dict(LORA_TRAIN), "test://cfg", "auto", None)
        base = lsg.BaseModel(
            model_dir=tmp_path,
            config=DENSE_CFG,
            tensors={k: (v[0], "float32") for k, v in _dense_full_tensors().items()},
            tensors_source="test://synthetic",
        )
        stems = self._census_stems()
        decl = lsg.derive_declared_block(
            base,
            spec,
            set(),
            "",
            (".lora_A.weight", ".lora_B.weight"),
            # fix45/#78: same census contract as the calibrated twin above --
            # dims carried so the (rank, in)/(out, rank) shapes this fence
            # pins are actually declared (post-#78 shapes require census
            # dims x config rank).
            adapter_modules=_census(stems, dims={s: (8, 8) for s in stems}),
        )
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
                ckpt, run_kind="lora", base_model_dir=base, train_config_path=cfg
            )

    def test_unpinned_prefix_auto_kind_also_refuses(self, tmp_path):
        """[FAILS-BEFORE -- same mechanism] The demand sits AFTER auto kind
        resolution on purpose: marker inference may consult the artifact for
        KIND, but the prefix question must not be answered by silence on the
        auto path either. Config carries rank/targets but no kind key, so the
        lora kind here is marker-inferred, not config-declared."""
        base = _make_base(tmp_path, _dense_full_tensors(), DENSE_CFG)
        ckpt = _materialize_artifact(tmp_path, _lora_tensors())
        cfg = _write_cfg(
            tmp_path,
            {"lora_rank": 4, "lora_targets": ["q_proj", "v_proj"]},
            name="auto-no-kind-no-prefix.json",
        )
        with pytest.raises(lsg.GateUnmeasured, match="--adapter-prefix"):
            lsg.adjudicate_checkpoint(
                ckpt, run_kind="auto", base_model_dir=base, train_config_path=cfg
            )

    def test_cli_lora_without_prefix_is_exactly_three(self, tmp_path):
        """[FAILS-BEFORE -- pre-patch the identical argv returns 0] The
        launcher-facing shape of the demand: refused measurement, not a
        checkpoint verdict."""
        base, ckpt, cfg = _healthy_lora(tmp_path)
        code = lsg_cli.main(
            [
                str(ckpt),
                "--run-kind",
                "lora",
                "--base-model-dir",
                str(base),
                "--train-config",
                str(cfg),
            ]
        )
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
            [f"layers.{i}.self_attn.{w}" for i in range(6) for w in ("q_proj", "v_proj")],
        )
        d = lsg.adjudicate_checkpoint(
            ckpt,
            run_kind="lora",
            base_model_dir=base,
            train_config_path=cfg,
            adapter_prefix="",
            adapter_modules=census,
        )
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
